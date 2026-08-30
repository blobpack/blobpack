"""Direct-offset access to STORED zip members (the "pread fast path").

A STORED member is a contiguous raw byte range inside its archive. After one
pass over the central directory -- plus one small read per member to size the
local header, whose extra field may differ from the central directory's --
every blob can be served with a single positioned read, with no per-read
header parsing and no archive-level locking.

Because direct-offset reads bypass zipfile's own checks, the index pass
cross-validates each member: local signature, local vs central compression
method and filename, no encryption, and in-bounds payload ranges.
"""

from __future__ import annotations

import io
import os
import struct
import zipfile

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


def _expected_name_bytes(info: zipfile.ZipInfo, flags: int) -> bytes:
    if flags & FLAG_UTF8:
        return info.filename.encode("utf-8")
    try:
        return info.filename.encode("cp437")
    except UnicodeEncodeError:  # zipfile fell back to utf-8 when decoding
        return info.filename.encode("utf-8")


class PackFile:
    """One pack shard: an immutable member index over a byte-range source.

    The index (name -> byte range) is process-independent state and travels
    through pickles; the source owns descriptors or sessions and rebuilds
    them per process. Reads are thread-safe.
    """

    def __init__(self, path_or_source, *, build_index: bool = True):
        from ._sources import LocalSource, RangeSource

        self.source: RangeSource = (
            path_or_source if isinstance(path_or_source, RangeSource) else LocalSource(path_or_source)
        )
        self.index: dict[str, tuple[int, int]] = {}
        if not build_index:
            return  # a catalog supplies byte ranges; skip the central directory
        try:
            self._build_index()
        except BaseException:
            self.close()
            raise

    def drop_index(self) -> None:
        """Release the in-memory index once a catalog holds it."""
        self.index = {}

    @property
    def path(self) -> str:
        return self.source.path

    @property
    def is_open(self) -> bool:
        return self.source.is_open

    def release_fd(self) -> bool:
        """Drop idle process-bound state; the index stays usable."""
        return self.source.release()

    def _build_index(self) -> None:
        """Parse the central directory once, then cross-check every member's
        local header. Header and name reads are issued in two batches so a
        remote shard costs a constant number of requests, not one per
        member."""
        stream = self.source.open_stream()
        try:
            with zipfile.ZipFile(stream) as bundle:
                infos = [info for info in bundle.infolist() if not info.is_dir()]
                # payloads must end before the central directory starts, so a
                # forged member cannot serve directory bytes as blob data
                payload_end = getattr(bundle, "start_dir", None)
        finally:
            stream.close()
        if payload_end is None:
            payload_end = self.source.size()

        # One read per member: the central directory already states how long
        # the name should be, so 30 + that covers the header and the name.
        expected_names = [_expected_name_bytes(info, info.flag_bits) for info in infos]
        blocks = self.source.read_batch(
            [(LOCAL_HEADER_SIZE + len(expected), info.header_offset) for info, expected in zip(infos, expected_names)]
        )
        if len(blocks) != len(infos):  # a source must never under-deliver silently
            raise CorruptPackError(f"{self.path}: batched read returned {len(blocks)} of {len(infos)} headers")
        for info, expected, block in zip(infos, expected_names, blocks):
            if len(block) < LOCAL_HEADER_SIZE or block[:4] != LOCAL_HEADER_MAGIC:
                raise CorruptPackError(f"{self.path}: bad local header for {info.filename!r}")
            local_flags, local_method = struct.unpack("<HH", block[6:10])
            name_len, extra_len = struct.unpack("<HH", block[26:30])
            if name_len != len(expected):
                raise CorruptPackError(
                    f"{self.path}: local header name length for {info.filename!r} disagrees with the central directory"
                )
            local_name = block[LOCAL_HEADER_SIZE : LOCAL_HEADER_SIZE + name_len]
            self._index_member(info, local_flags, local_method, name_len, extra_len, local_name, payload_end)
        self._reject_overlaps(infos)

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

    def _index_member(
        self,
        info: zipfile.ZipInfo,
        local_flags: int,
        local_method: int,
        name_len: int,
        extra_len: int,
        local_name: bytes,
        payload_end: int,
    ) -> None:
        name = info.filename
        if name in self.index:
            raise CorruptPackError(f"{self.path}: duplicate member name {name!r} within one shard")
        encrypted = FLAG_ENCRYPTED | FLAG_STRONG_ENCRYPTION
        if (info.flag_bits | local_flags) & encrypted:
            raise UnsupportedMemberError(f"{self.path}: member {name!r} is encrypted")
        if info.compress_type != zipfile.ZIP_STORED:
            raise NotStoredError(f"{self.path}: member {name!r} is not STORED; blobpack requires uncompressed members")
        if local_method != zipfile.ZIP_STORED:
            raise CorruptPackError(
                f"{self.path}: local header of {name!r} disagrees with the central directory on compression method"
            )
        if local_name != _expected_name_bytes(info, local_flags):
            raise CorruptPackError(
                f"{self.path}: local header name mismatch at offset {info.header_offset} (expected {name!r})"
            )
        data_offset = info.header_offset + LOCAL_HEADER_SIZE + name_len + extra_len
        if data_offset + info.file_size > payload_end:
            raise CorruptPackError(f"{self.path}: member {name!r} extends into the central directory")
        self.index[name] = (data_offset, info.file_size)

    def _read_at(self, size: int, offset: int) -> bytes:
        return self.source.read_at(size, offset)

    def read(self, name: str, *, offset: int | None = None, size: int | None = None) -> bytes:
        if offset is None or size is None:
            offset, size = self.index[name]
        data = self.source.read_at(size, offset)
        if len(data) != size:
            raise CorruptPackError(f"{self.path}: short read for {name!r} ({len(data)} of {size} bytes)")
        return data

    def open(self, name: str, *, offset: int | None = None, size: int | None = None) -> BlobView:
        """Return a bounded, seekable view of one member."""
        if offset is None or size is None:
            offset, size = self.index[name]
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
