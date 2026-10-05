"""Direct-offset access to STORED zip members (the "pread fast path").

A STORED member is a contiguous raw byte range inside its archive. After one
pass over the central directory, every blob can be served with a single
positioned read, with no per-read header parsing and no archive-level locking.

Because direct-offset reads bypass zipfile's own checks, each member is
cross-validated before its bytes are served: local signature, local vs
central compression method and filename, no encryption, and in-bounds
payload ranges. The local header sits just before the payload and its extra
field may differ from the central directory's, so this costs one small read
per member. By default it happens on the member's first read, next to the
payload it guards; ``validate_on_open`` runs it for every member at open,
which on a network filesystem or object storage is one round trip each
(issue #11).
"""

from __future__ import annotations

import bisect
import io
import os
import struct
import zipfile
from collections.abc import Iterator

LOCAL_HEADER_SIZE = 30
LOCAL_HEADER_MAGIC = b"PK\x03\x04"
FLAG_ENCRYPTED = 0x1
FLAG_STRONG_ENCRYPTION = 0x40
FLAG_UTF8 = 0x800


class BlobPackError(Exception):
    """Base class for blobpack errors."""


class NotStoredError(BlobPackError):
    """A pack member uses compression; direct-offset reads require STORED."""


class UnsupportedMemberError(BlobPackError):
    """A pack member uses a zip feature blobpack forbids (e.g. encryption)."""


class CorruptPackError(BlobPackError):
    """A pack's structure does not match its central directory."""


def _expected_name_bytes(name: str, flags: int) -> bytes:
    if flags & FLAG_UTF8:
        return name.encode("utf-8")
    try:
        return name.encode("cp437")
    except UnicodeEncodeError:  # zipfile fell back to utf-8 when decoding
        return name.encode("utf-8")


class PackFile:
    """One pack shard: an immutable member index over a byte-range source.

    The index (name -> byte range) is process-independent state and travels
    through pickles; the source owns descriptors or sessions and rebuilds
    them per process. Reads are thread-safe.
    """

    def __init__(self, path_or_source, *, validate_on_open: bool = False):
        from ._sources import LocalSource, RangeSource

        self.source: RangeSource = (
            path_or_source if isinstance(path_or_source, RangeSource) else LocalSource(path_or_source)
        )
        self.index: dict[str, tuple[int, int]] = {}
        # members not yet validated: name -> (header_offset, size, central
        # flags, expected name bytes); index doubles as the cache of members
        # whose local header has been validated
        self._pending: dict[str, tuple[int, int, int, bytes]] = {}
        self._bounds: list[int] = []
        self._min_ends: list[int] = []
        self._payload_end = 0
        try:
            self._build_index(validate_on_open=validate_on_open)
        except BaseException:
            self.close()
            raise

    @property
    def path(self) -> str:
        return self.source.path

    @property
    def is_open(self) -> bool:
        return self.source.is_open

    def release_fd(self) -> bool:
        """Drop idle process-bound state; the index stays usable."""
        return self.source.release()

    def _build_index(self, *, validate_on_open: bool = False) -> None:
        """Parse the central directory once and run every check it alone can
        answer. Each member's local header is checked on its first read, or
        here for every member with ``validate_on_open``."""
        stream = self.source.open_directory_stream()
        try:
            with zipfile.ZipFile(stream) as bundle:
                infos = [info for info in bundle.infolist() if not info.is_dir()]
                # payloads must end before the central directory starts, so a
                # forged member cannot serve directory bytes as blob data
                payload_end = getattr(bundle, "start_dir", None)
        finally:
            stream.close()
        self._payload_end = self.source.size() if payload_end is None else payload_end
        # a forged end-of-directory offset can push zipfile's prefix
        # adjustment negative, and some backends answer negative ranges
        # relative to the object's tail
        if any(info.header_offset < 0 for info in infos):
            raise CorruptPackError(f"{self.path}: negative member header offset")

        if not validate_on_open:
            self._defer_members(infos)
            return

        # One read per member: the central directory already states how long
        # the name should be, so 30 + that covers the header and the name.
        expected_names = [_expected_name_bytes(info.filename, info.flag_bits) for info in infos]
        blocks = self.source.read_batch(
            [(LOCAL_HEADER_SIZE + len(expected), info.header_offset) for info, expected in zip(infos, expected_names)]
        )
        if len(blocks) != len(infos):  # a source must never under-deliver silently
            raise CorruptPackError(f"{self.path}: batched read returned {len(blocks)} of {len(infos)} headers")
        for info, expected, block in zip(infos, expected_names, blocks):
            self._index_member(info, expected, block)
        self._reject_overlaps(infos)

    def _defer_members(self, infos: list[zipfile.ZipInfo]) -> None:
        """Run every check the central directory alone can answer; the rest
        waits for each member's first read."""
        encrypted = FLAG_ENCRYPTED | FLAG_STRONG_ENCRYPTION
        for info in infos:
            name = info.filename
            if name in self._pending:
                raise CorruptPackError(f"{self.path}: duplicate member name {name!r} within one shard")
            if info.flag_bits & encrypted:
                raise UnsupportedMemberError(f"{self.path}: member {name!r} is encrypted")
            if info.compress_type != zipfile.ZIP_STORED:
                raise NotStoredError(
                    f"{self.path}: member {name!r} is not STORED; blobpack requires uncompressed members"
                )
            expected = _expected_name_bytes(name, info.flag_bits)
            self._pending[name] = (info.header_offset, info.file_size, info.flag_bits, expected)
        # (header offset, directory-stated minimum end: header + name +
        # payload, i.e. the true end minus the local extra field)
        spans = sorted(
            (entry[0], entry[0] + LOCAL_HEADER_SIZE + len(entry[3]) + entry[1]) for entry in self._pending.values()
        )
        offsets = [offset for offset, _ in spans]
        if any(a == b for a, b in zip(offsets, offsets[1:])):
            raise CorruptPackError(f"{self.path}: two members share a local header offset")
        if offsets and offsets[-1] >= self._payload_end:
            raise CorruptPackError(f"{self.path}: a member's local header lies inside the central directory")
        # each member's payload must end before the next member's header (or
        # the central directory) and start past the previous member's stated
        # end; checked per member as it resolves
        self._bounds = [*offsets, self._payload_end]
        self._min_ends = [end for _, end in spans]

    #: fused first reads cover the local extra field up to this much; blobpack
    #: writes extras of 0 or 20 bytes, so one request suffices in practice
    _HEADER_SLACK = 64

    def _resolve_member(self, name: str) -> tuple[int, int]:
        """Validate one deferred member's local header and cache its span.

        Deterministic, so concurrent first reads may race freely: both
        compute and install the same entry.
        """
        entry = self._pending.get(name)
        if entry is None:
            raise KeyError(name)
        block = self.source.read_at(LOCAL_HEADER_SIZE + len(entry[3]), entry[0])
        return self._parse_pending_header(name, entry, block)

    def _read_pending(self, name: str, entry: tuple[int, int, int, bytes]) -> bytes:
        """First read of a deferred member: header and payload in one
        request, since remote reads are round-trip bound."""
        header_offset, size, _, expected = entry
        # the directory-claimed size drives the request, so bound it before
        # transferring anything: a forged size must not buy a giant read
        if (
            header_offset + LOCAL_HEADER_SIZE + len(expected) + size
            > self._bounds[bisect.bisect_right(self._bounds, header_offset)]
        ):
            raise CorruptPackError(f"{self.path}: member {name!r} overlaps the next member or the central directory")
        head_len = LOCAL_HEADER_SIZE + len(expected) + self._HEADER_SLACK
        block = self.source.read_at(head_len + size, header_offset)
        data_offset, _ = self._parse_pending_header(name, entry, block[:head_len])
        start = data_offset - header_offset
        payload = block[start : start + size]
        if len(payload) < size:  # a foreign extra field larger than the slack
            payload += self.source.read_at(size - len(payload), header_offset + start + len(payload))
        if len(payload) != size:
            raise CorruptPackError(f"{self.path}: short read for {name!r} ({len(payload)} of {size} bytes)")
        return payload

    def _local_data_offset(self, name, header_offset, flags, expected, block):
        """Validate a local header identically for eager and deferred readers."""
        if len(block) < LOCAL_HEADER_SIZE or block[:4] != LOCAL_HEADER_MAGIC:
            raise CorruptPackError(f"{self.path}: bad local header for {name!r}")
        local_flags, local_method = struct.unpack("<HH", block[6:10])
        name_len, extra_len = struct.unpack("<HH", block[26:30])
        if name_len != len(expected):
            raise CorruptPackError(
                f"{self.path}: local header name length for {name!r} disagrees with the central directory"
            )
        if (flags | local_flags) & (FLAG_ENCRYPTED | FLAG_STRONG_ENCRYPTION):
            raise UnsupportedMemberError(f"{self.path}: member {name!r} is encrypted")
        if local_method != zipfile.ZIP_STORED:
            raise CorruptPackError(
                f"{self.path}: local header of {name!r} disagrees with the central directory on compression method"
            )
        if block[LOCAL_HEADER_SIZE : LOCAL_HEADER_SIZE + name_len] != _expected_name_bytes(name, local_flags):
            raise CorruptPackError(
                f"{self.path}: local header name mismatch at offset {header_offset} (expected {name!r})"
            )
        return header_offset + LOCAL_HEADER_SIZE + name_len + extra_len

    def _parse_pending_header(self, name: str, entry: tuple[int, int, int, bytes], block: bytes) -> tuple[int, int]:
        header_offset, size, flags, expected = entry
        data_offset = self._local_data_offset(name, header_offset, flags, expected, block)
        position = bisect.bisect_right(self._bounds, header_offset)  # bounds[position - 1] is this member
        if data_offset + size > self._bounds[position]:
            raise CorruptPackError(f"{self.path}: member {name!r} overlaps the next member or the central directory")
        # the predecessor's stated end excludes its local extra field, so a
        # header forged into the last extra_len bytes of the predecessor's
        # span is caught when the predecessor itself resolves, not here;
        # nothing is served that the archive's author did not place there
        if position >= 2 and header_offset < self._min_ends[position - 2]:
            raise CorruptPackError(f"{self.path}: member {name!r} overlaps the previous member")
        span = (data_offset, size)
        self.index[name] = span
        return span

    def _span(self, name: str) -> tuple[int, int]:
        span = self.index.get(name)
        return self._resolve_member(name) if span is None else span

    def __contains__(self, name: str) -> bool:
        return name in self.index or name in self._pending

    def member_names(self) -> Iterator[str]:
        """Member names in central-directory order."""
        return iter(self._pending) if self._pending else iter(self.index)

    def member_sizes(self) -> Iterator[tuple[str, int]]:
        """``(name, payload size)`` pairs, without resolving deferred members."""
        if self._pending:
            return ((name, entry[1]) for name, entry in self._pending.items())
        return ((name, span[1]) for name, span in self.index.items())

    @property
    def member_count(self) -> int:
        return len(self._pending) if self._pending else len(self.index)

    def _reject_overlaps(self, infos) -> None:
        """Each member's bytes must be its own: a forged range that reaches
        into another member would let one key alias another's payload."""
        spans = sorted(
            (info.header_offset, self.index[info.filename][0] + self.index[info.filename][1], info.filename)
            for info in infos
        )
        for (_, end, name), (next_start, _, next_name) in zip(spans, spans[1:]):
            if end > next_start:
                raise CorruptPackError(f"{self.path}: member {name!r} overlaps {next_name!r}")

    def _index_member(self, info: zipfile.ZipInfo, expected: bytes, block: bytes) -> None:
        name = info.filename
        if name in self.index:
            raise CorruptPackError(f"{self.path}: duplicate member name {name!r} within one shard")
        if info.compress_type != zipfile.ZIP_STORED:
            raise NotStoredError(f"{self.path}: member {name!r} is not STORED; blobpack requires uncompressed members")
        data_offset = self._local_data_offset(name, info.header_offset, info.flag_bits, expected, block)
        if data_offset + info.file_size > self._payload_end:
            raise CorruptPackError(f"{self.path}: member {name!r} extends into the central directory")
        self.index[name] = (data_offset, info.file_size)

    def _read_at(self, size: int, offset: int) -> bytes:
        return self.source.read_at(size, offset)

    def read(self, name: str) -> bytes:
        span = self.index.get(name)
        if span is None:
            entry = self._pending.get(name)
            if entry is None:
                raise KeyError(name)
            return self._read_pending(name, entry)
        offset, size = span
        data = self.source.read_at(size, offset)
        if len(data) != size:
            raise CorruptPackError(f"{self.path}: short read for {name!r} ({len(data)} of {size} bytes)")
        return data

    def open(self, name: str) -> BlobView:
        """Return a bounded, seekable view of one member."""
        offset, size = self._span(name)
        return BlobView(self, name, offset, size)

    def close(self) -> None:
        self.source.close()


class BlobView(io.RawIOBase):
    """A bounded, seekable, read-only view of one STORED member.

    Nothing is buffered up front, and every request is clamped to the
    member's byte range, so a view can never reach other members or the
    archive's own structures. A view carries its own position and is not
    thread-safe; open one per consumer. The underlying pack file is.
    """

    def __init__(self, shard: PackFile, name: str, start: int, size: int):
        super().__init__()
        self._shard = shard
        self._name = name
        self._start = start
        self._size = size
        self._pos = 0

    @property
    def name(self) -> str:
        return self._name

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def __len__(self) -> int:
        return self._size

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self._pos + offset
        elif whence == os.SEEK_END:
            target = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence!r}")
        if target < 0:
            raise OSError(22, "negative seek position")
        self._pos = target  # seeking past the end is legal; reads return b""
        return self._pos

    def readinto(self, buffer) -> int:
        if self.closed:
            raise ValueError("read from closed blob view")
        remaining = self._size - self._pos
        if remaining <= 0:
            return 0
        want = min(len(buffer), remaining)
        if want == 0:
            return 0
        chunk = self._shard._read_at(want, self._start + self._pos)
        if len(chunk) != want:
            raise CorruptPackError(f"{self._shard.path}: short read for {self._name!r} ({len(chunk)} of {want} bytes)")
        buffer[: len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)

    def readall(self) -> bytes:
        want = max(self._size - self._pos, 0)
        data = self._shard._read_at(want, self._start + self._pos)
        if len(data) != want:
            raise CorruptPackError(f"{self._shard.path}: short read for {self._name!r} ({len(data)} of {want} bytes)")
        self._pos += len(data)
        return data
