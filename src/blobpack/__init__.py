"""blobpack: pack dataset media into plain STORED zip shards.

A Blob Pack is a directory of uncompressed (STORED) zip shards. Every blob is
addressed by a plain string reference such as::

    zip://images/0001.jpg::media/pack-0000.zip

where the part after ``::`` is a path relative to the dataset root, so a
dataset folder stays fully self-contained and relocatable. Any zip tool can
read the shards; this library additionally serves random reads via direct
member offsets (one positioned read per blob).

See SPEC.md for the format specification.
"""

from __future__ import annotations

import io
import os
import posixpath
import random
import threading
import zipfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ._zip import (
    BlobPackError,
    BlobView,
    CorruptPackError,
    NotStoredError,
    PackFile,
    UnsupportedMemberError,
)

try:
    from ._version import __version__
except ImportError:  # raw checkout without the hatch-vcs build hook
    __version__ = "0.0.0+unknown"
__all__ = [
    "BlobPackError",
    "BlobView",
    "CorruptPackError",
    "NotStoredError",
    "PackSet",
    "PackWriter",
    "UnsupportedMemberError",
    "make_ref",
    "parse_ref",
]

DEFAULT_GROUP_FILL = 0.5
DEFAULT_MAX_OPEN_FILES = 64
DEFAULT_MAX_PACK_BYTES = 4 << 30  # classic-zip ceiling, applies to the archive file
DEFAULT_MAX_BLOB_COUNT = 65_535
_LOCAL_OVERHEAD = 30  # local file header, excluding the name
_CENTRAL_OVERHEAD = 46  # central directory entry, excluding the name
_EOCD_SIZE = 22
_ZIP64_EXTRA = 20  # zip64 extended-information extra field (header + two sizes)
_REF_SCHEME = "zip://"
_REF_SEPARATOR = "::"


def make_ref(key: str, pack_path: str) -> str:
    """Build a blob reference: ``zip://<key>::<pack_path>``."""
    return f"{_REF_SCHEME}{key}{_REF_SEPARATOR}{pack_path}"


def parse_ref(ref: str) -> tuple[str, str]:
    """Split a blob reference into ``(key, pack_path)``."""
    if not ref.startswith(_REF_SCHEME) or _REF_SEPARATOR not in ref:
        raise ValueError(f"not a blob reference: {ref!r}")
    key, _, pack_path = ref[len(_REF_SCHEME) :].partition(_REF_SEPARATOR)
    if not key or not pack_path:
        raise ValueError(f"not a blob reference: {ref!r}")
    return key, pack_path


def _validate_key(key: str) -> str:
    if not key or key != key.strip():
        raise ValueError(f"invalid blob key: {key!r}")
    if "\\" in key or key.startswith("/"):
        raise ValueError(f"blob keys are relative POSIX paths: {key!r}")
    if _REF_SEPARATOR in key:
        raise ValueError(f"blob keys must not contain '::': {key!r}")
    parts = key.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"blob keys must not contain empty, '.' or '..' segments: {key!r}")
    return key


class PackWriter:
    """Write blobs into rolling STORED zip shards.

    ``pack_dir`` receives ``<prefix>-NNNN.zip`` shards. ``ref_base`` is the
    pack directory's path as seen from the dataset root and is embedded in
    returned references; it defaults to the pack directory's final path
    component, so a media directory directly under the dataset root needs
    nothing, and a nested one should pass e.g. ``ref_base="assets/media"``.

    Shards roll when adding a blob would push the shard *file* past
    ``max_pack_bytes`` (zip headers included, so the default keeps every
    shard within classic-zip limits) or past ``max_blob_count``; a blob too
    large for one shard forms a singleton shard. Member timestamps are
    fixed, so identical inputs produce byte-identical shards.

    Blobs that belong together (one episode's cameras, one sample's
    modalities) can pass a ``group``. A new group starts a new shard when
    the current one is at least ``group_fill`` full, which keeps a group in
    one shard without producing a shard per group. A group larger than the
    space that policy leaves still spans shards; ``groups_split`` counts
    those, so a caller can report it rather than discover it later.
    """

    def __init__(
        self,
        pack_dir: os.PathLike | str,
        *,
        ref_base: str | None = None,
        prefix: str = "pack",
        max_pack_bytes: int = DEFAULT_MAX_PACK_BYTES,
        max_blob_count: int = DEFAULT_MAX_BLOB_COUNT,
        group_fill: float = DEFAULT_GROUP_FILL,
    ):
        if max_pack_bytes <= 0 or max_blob_count <= 0:
            raise ValueError("max_pack_bytes and max_blob_count must be positive")
        if not 0 <= group_fill <= 1:
            raise ValueError("group_fill must be between 0 and 1")
        if "/" in prefix or "\\" in prefix or prefix in ("", ".", ".."):
            raise ValueError(f"prefix must be a plain filename component: {prefix!r}")
        self.pack_dir = Path(pack_dir)
        self.ref_base = self.pack_dir.name if ref_base is None else posixpath.normpath(ref_base)
        self.prefix = prefix
        self.max_pack_bytes = max_pack_bytes
        self.max_blob_count = max_blob_count
        self.group_fill = group_fill
        self.pack_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.pack_dir.glob("*.zip"))
        if existing:
            raise BlobPackError(
                f"{self.pack_dir} already contains {len(existing)} zip file(s); "
                "packs are immutable, write to a fresh directory"
            )
        self._keys: set[str] = set()
        self._shard_index = -1
        self._writer: zipfile.ZipFile | None = None
        self._shard_bytes = _EOCD_SIZE  # projected shard file size
        self._shard_count = 0
        self.blobs_written = 0
        self.bytes_written = 0
        self.groups_split = 0
        self._group = None
        self._group_shard = None
        self._closed = False

    def _shard_name(self, index: int) -> str:
        return f"{self.prefix}-{index:04d}.zip"

    def _roll(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._shard_index += 1
        self._writer = zipfile.ZipFile(self.pack_dir / self._shard_name(self._shard_index), "w", zipfile.ZIP_STORED)
        self._shard_bytes = _EOCD_SIZE
        self._shard_count = 0

    def add(self, key: str, data: bytes, *, group: str | None = None) -> str:
        """Store ``data`` under ``key`` and return its blob reference."""
        return self._add_stream(key, io.BytesIO(data), len(data), group=group)

    def add_file(self, key: str, source, *, size: int | None = None, group: str | None = None) -> str:
        """Stream a blob from a file path or binary file object in 1 MiB
        chunks. A non-seekable stream must pass ``size`` explicitly, since
        shard rolling needs it up front.
        """
        if isinstance(source, (str, os.PathLike)):
            path = Path(source)
            with open(path, "rb") as handle:
                return self._add_stream(key, handle, path.stat().st_size if size is None else size, group=group)
        if size is None:
            if not source.seekable():
                raise ValueError("size is required for non-seekable streams")
            position = source.tell()
            size = source.seek(0, os.SEEK_END) - position
            source.seek(position)
        return self._add_stream(key, source, size, group=group)

    def _add_stream(self, key: str, reader, size: int, *, group: str | None = None) -> str:
        if self._closed:
            raise BlobPackError("writer is closed")
        _validate_key(key)
        if key in self._keys:
            raise BlobPackError(f"duplicate blob key: {key!r}")
        # projected shard growth: payload + local header + central entry
        # (each carries the name once) + zip64 extras when forced below
        member_bytes = size + _LOCAL_OVERHEAD + _CENTRAL_OVERHEAD + 2 * len(key.encode("utf-8"))
        if size > zipfile.ZIP64_LIMIT:
            member_bytes += 2 * _ZIP64_EXTRA
        starts_group = group is not None and group != self._group
        if (
            starts_group
            and self._writer is not None
            and self._shard_count > 0
            and self._shard_bytes >= self.max_pack_bytes * self.group_fill
        ):
            self._roll()  # give the new group a shard of its own to fill
        if (
            self._writer is None
            or self._shard_count >= self.max_blob_count
            or (self._shard_count > 0 and self._shard_bytes + member_bytes > self.max_pack_bytes)
        ):
            if group is not None and group == self._group and self._group_shard == self._shard_index:
                self.groups_split += 1  # too large for the policy to keep together
            self._roll()
        if group is not None:
            if group != self._group:
                self._group_shard = self._shard_index
            self._group = group
        info = zipfile.ZipInfo(key)  # fixed 1980 timestamp: deterministic output
        info.compress_type = zipfile.ZIP_STORED
        written = 0
        # zipfile refuses a streamed member past ZIP64_LIMIT (~2 GiB) unless
        # zip64 is forced up front, well below the 4 GiB classic-zip ceiling
        with self._writer.open(info, "w", force_zip64=size > zipfile.ZIP64_LIMIT) as member:
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                member.write(chunk)
                written += len(chunk)
        if written != size:
            raise BlobPackError(
                f"short or oversized stream for {key!r}: wrote {written} of {size} declared bytes; "
                "the current shard is now invalid"
            )
        self._keys.add(key)
        self._shard_bytes += member_bytes
        self._shard_count += 1
        self.blobs_written += 1
        self.bytes_written += size
        return make_ref(key, posixpath.join(self.ref_base, self._shard_name(self._shard_index)))

    @property
    def shards_written(self) -> int:
        return self._shard_index + 1

    def close(self) -> None:
        if not self._closed:
            if self._writer is not None:
                self._writer.close()
            self._closed = True
            self._verify_stored()

    def _verify_stored(self) -> None:
        """Re-read each central directory and check every member is STORED:
        the guard SPEC.md asks writers for, since a mixed-path zipfile
        writer can silently produce compressed members."""
        for index in range(self._shard_index + 1):
            path = self.pack_dir / self._shard_name(index)
            with zipfile.ZipFile(path) as bundle:
                for info in bundle.infolist():
                    if info.compress_type != zipfile.ZIP_STORED:
                        raise NotStoredError(f"{path}: member {info.filename!r} was written compressed")

    def __enter__(self) -> PackWriter:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _shard_name(path: str) -> str:
    return posixpath.basename(str(path).replace(os.sep, "/").rstrip("/"))


def _looks_remote(location: str) -> bool:
    """True for fsspec URLs (``s3://``, ``https://``); a Windows drive is not."""
    head, sep, _ = location.partition("://")
    return bool(sep) and len(head) > 1 and head.isalnum()


def _remote_shard_names(fs, root: str, pattern: str) -> list[str]:
    return sorted(_shard_name(entry) for entry in fs.glob(posixpath.join(root, pattern)))


def _open_remote_sources(location: str, pattern: str, storage_options: dict | None) -> list:
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise BlobPackError(
            f"reading {location} needs fsspec; install the remote extra: pip install 'blobpack[remote]'"
        ) from exc
    from ._sources import FsspecSource

    options = dict(storage_options or {})
    fs, root = fsspec.core.url_to_fs(location, **options)
    root = root.rstrip("/")
    return [
        FsspecSource(fs, posixpath.join(root, name), storage_options=options)
        for name in _remote_shard_names(fs, root, pattern)
    ]


class PackSet:
    """Read a directory of pack shards with direct-offset (pread) access.

    Opening a PackSet parses each shard's central directory once and keeps
    a bounded descriptor pool; every ``read`` is then a single
    positioned read. Reads are thread-safe and fastest when issued
    concurrently on network filesystems.

    On object storage, directories load lazily instead: open lists the
    shards, a full reference touches only its own shard, and a member's
    local header is validated on its first read (adding one small ranged
    read) rather than at open. The first bare-key read loads every
    remaining directory, since bare keys resolve through a cross-shard
    unique index. A corrupt or forged member therefore surfaces at first
    read; ``blobpack verify`` remains the at-rest full check.

    ``ref_base`` is the pack directory's path as embedded in references
    (see PackWriter); references whose directory part does not match it are
    rejected rather than resolved to a same-named shard from some other
    dataset. It defaults to the pack directory's final path component; pass
    ``ref_base=""`` to accept any directory part.

    Note: per-read CRC verification is skipped by design; verify packs at
    rest instead (``blobpack verify`` or dataset-level checksums).
    """

    def __init__(
        self,
        pack_dir: os.PathLike | str,
        *,
        ref_base: str | None = None,
        pattern: str = "*.zip",
        max_open_files: int = DEFAULT_MAX_OPEN_FILES,
        storage_options: dict | None = None,
        catalog: bool | str | os.PathLike | None = None,
        _sources: list | None = None,
    ):
        if max_open_files < 1:
            raise ValueError("max_open_files must be at least 1")
        location = os.fspath(pack_dir)
        if _sources is not None:
            sources = _sources
        elif storage_options is not None or _looks_remote(location):
            sources = _open_remote_sources(location, pattern, storage_options)
        else:
            sources = sorted(Path(location).glob(pattern))
        if not sources:
            raise BlobPackError(f"no {pattern} shards under {location}")

        self.pack_dir = Path(location)
        # normalized exactly as the writer normalizes it, so the same value
        # given to both sides always matches
        self.ref_base = self.pack_dir.name if ref_base is None else (ref_base and posixpath.normpath(ref_base))
        from ._sources import DescriptorPool

        self.max_open_files = max_open_files
        self._pool = DescriptorPool(max_open_files)
        self._open_lock = threading.Lock()  # guards deferred shard opening
        self._shards: dict[str, PackFile] = {}
        self._deferred: dict[str, object] = {}  # shard name -> unopened lazy source
        self._by_key: dict[str, PackFile] | None = {}
        self._catalog = None
        try:
            if catalog:
                self._open_with_catalog(sources, catalog)
            else:
                self._open_with_indices(sources, location)
        except BaseException:
            self.close()
            raise

    def _pack_file(self, source, **kwargs):
        from ._sources import LocalSource, RangeSource

        if not isinstance(source, RangeSource):
            source = LocalSource(source)
        if isinstance(source, LocalSource):
            source.pool = self._pool
        return PackFile(source, **kwargs)

    def _open_with_indices(self, sources: list, location: str, *, force_eager: bool = False) -> None:
        """Record every shard; parse directories eagerly for local sources
        and on first touch for lazy ones (object storage), where a full-ref
        read should not pay for shards it never visits (issue #11)."""
        for source in sources:
            if not force_eager and getattr(source, "lazy_validation", False):
                name = _shard_name(source.path)
                if name in self._deferred:
                    raise BlobPackError(f"duplicate shard name {name!r} under {location}")
                self._deferred[name] = source
                continue
            shard = self._pack_file(source, force_eager=force_eager)
            name = _shard_name(shard.path)
            if name in self._shards:
                raise BlobPackError(f"duplicate shard name {name!r} under {location}")
            self._shards[name] = shard
            for key in shard.member_names():
                if key in self._by_key:
                    other = _shard_name(self._by_key[key].path)
                    raise BlobPackError(f"duplicate key {key!r} in {name} and {other}")
                self._by_key[key] = shard
        if self._deferred:
            self._by_key = None  # built when a bare key first needs it

    def _open_deferred(self, name: str) -> PackFile:
        with self._open_lock:
            shard = self._shards.get(name)
            if shard is None:
                shard = self._pack_file(self._deferred[name])
                self._shards[name] = shard
                del self._deferred[name]  # only after success, so a failed open can be retried
            return shard

    def _require_by_key(self) -> dict[str, PackFile]:
        """The cross-shard key map; on a deferred pack set the first bare-key
        use builds it, loading the remaining directories and enforcing key
        uniqueness across shards then rather than at open."""
        by_key = self._by_key
        if by_key is not None:
            return by_key
        with self._open_lock:
            if self._by_key is not None:
                return self._by_key
            for name in sorted(self._deferred):
                self._shards[name] = self._pack_file(self._deferred[name])
                del self._deferred[name]
            by_key = {}
            for name in sorted(self._shards):
                for key in self._shards[name].member_names():
                    if key in by_key:
                        raise BlobPackError(f"duplicate key {key!r} in {name} and {_shard_name(by_key[key].path)}")
                    by_key[key] = self._shards[name]
            self._by_key = by_key
            return by_key

    def _open_with_catalog(self, sources: list, catalog: bool | str | os.PathLike) -> None:
        """Reuse a validated catalog when every shard is byte-identical to the
        one recorded, and rebuild it (with full validation) otherwise."""
        from ._catalog import open_catalog

        self._catalog = open_catalog(self.pack_dir, catalog)
        probe = {}
        for source in sources:
            shard = self._pack_file(source, build_index=False)
            probe[_shard_name(shard.path)] = shard
        if self._catalog.matches(probe):
            self._shards = probe
            return
        for shard in probe.values():
            shard.close()
        # a catalog persists validated offsets, so the rebuild validates
        # every member up front even on a lazy source
        self._open_with_indices(sources, str(self.pack_dir), force_eager=True)
        self._catalog.write(self._shards)
        self._by_key = {}
        for shard in self._shards.values():
            shard.drop_index()  # the catalog holds it now

    @classmethod
    def from_fs(cls, fs, path: str, *, pattern: str = "*.zip", **kwargs) -> PackSet:
        """Open a pack set on an already-configured fsspec filesystem.

        Use this when the caller owns filesystem construction (credentials,
        caching, a test filesystem); ``PackSet(url, storage_options=...)``
        is the shorthand that builds one for you.
        """
        from ._sources import FsspecSource

        root = path.rstrip("/")
        sources = [FsspecSource(fs, posixpath.join(root, name)) for name in _remote_shard_names(fs, root, pattern)]
        return cls(root, _sources=sources, pattern=pattern, **kwargs)

    def __getstate__(self) -> dict:
        """Pickle indices, never descriptors: a spawned dataloader worker
        reuses the parsed member index without re-reading any shard."""
        if self._catalog is not None:
            raise BlobPackError(
                "a catalog-backed PackSet holds an open database and cannot be pickled; "
                "construct it inside each worker instead"
            )
        state = self.__dict__.copy()
        del state["_open_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open_lock = threading.Lock()

    def _locate(self, key: str) -> tuple[PackFile, int | None, int | None]:
        """Find a key's shard, and its byte range when a catalog holds it."""
        if self._catalog is None:
            return self._require_by_key()[key], None, None
        found = self._catalog.locate(key)
        if found is None:
            raise KeyError(key)
        shard_name, offset, size = found
        return self._shards[shard_name], offset, size

    def _resolve(self, key_or_ref: str) -> tuple[PackFile, str, int | None, int | None]:
        """Map a bare key or a ``zip://key::path`` reference to a shard, key
        and (with a catalog) byte range."""
        if not key_or_ref.startswith(_REF_SCHEME):
            shard, offset, size = self._locate(key_or_ref)
            return shard, key_or_ref, offset, size
        key, pack_path = parse_ref(key_or_ref)
        if self.ref_base and posixpath.dirname(pack_path) != self.ref_base:
            raise KeyError(
                f"reference does not belong to this pack set (expected directory {self.ref_base!r}): {key_or_ref!r}"
            )
        name = posixpath.basename(pack_path)
        shard = self._shards.get(name)
        if shard is None and name in self._deferred:
            shard = self._open_deferred(name)
        if shard is None:
            raise KeyError(f"referenced shard not in this pack set: {key_or_ref!r}")
        if self._catalog is None and self._by_key is None:
            # deferred pack set: answer from the referenced shard alone, so a
            # full-ref read never opens shards it does not touch
            if key not in shard:
                raise KeyError(f"blob not in the referenced shard: {key_or_ref!r}")
            return shard, key, None, None
        located, offset, size = self._locate(key)
        if located is not shard:
            raise KeyError(f"blob not in the referenced shard: {key_or_ref!r}")
        return shard, key, offset, size

    def read(self, key_or_ref: str) -> bytes:
        """Read one blob by bare key or by ``zip://key::path`` reference."""
        shard, key, offset, size = self._resolve(key_or_ref)
        return shard.read(key, offset=offset, size=size)

    def __contains__(self, key: str) -> bool:
        if self._catalog is None:
            return key in self._require_by_key()
        return self._catalog.locate(key) is not None

    def __len__(self) -> int:
        return len(self._require_by_key()) if self._catalog is None else self._catalog.count()

    def keys(self) -> Iterator[str]:
        return iter(self._require_by_key()) if self._catalog is None else self._catalog.keys()

    def open(self, key_or_ref: str, *, buffered: bool = True):
        """Open one blob as a bounded, seekable, read-only file object.

        For blobs too large to hold in memory, or for decoders that seek
        (TorchCodec, PyAV, soundfile); every seek stays clamped to the
        blob's byte range. ``buffered=False`` skips the ``BufferedReader``
        wrapper most parsers expect.
        """
        shard, key, offset, size = self._resolve(key_or_ref)
        view = shard.open(key, offset=offset, size=size)
        return io.BufferedReader(view) if buffered else view

    def read_many(self, keys_or_refs: Iterable[str], *, workers: int = 8) -> list[bytes]:
        """Read many blobs concurrently, returning payloads in input order.

        On network filesystems, concurrent reads recover most of the
        throughput that single-threaded random reads forfeit; use this
        instead of hand-rolling a thread pool around ``read``.
        """
        items = list(keys_or_refs)
        if workers <= 1 or len(items) <= 1:
            return [self.read(item) for item in items]
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(self.read, items))

    def iter_blobs(
        self,
        *,
        shuffle_shards: bool = False,
        seed: int = 0,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> Iterator[tuple[str, bytes]]:
        """Yield ``(key, data)`` shard by shard, sequentially within a shard.

        ``shuffle_shards=True`` shuffles shard order while keeping each
        shard's sequential throughput. With ``num_workers > 1`` only
        ``worker_id``'s contiguous member range is yielded -- disjoint and
        complete, so worker count is decoupled from shard count.
        """
        if num_workers < 1 or not 0 <= worker_id < num_workers:
            raise ValueError(f"invalid worker split: worker_id={worker_id}, num_workers={num_workers}")
        if self._catalog is None:
            self._require_by_key()  # full iteration touches every shard anyway
        # name order, not insertion order: on a deferred pack set insertion
        # follows access history, and worker splits must agree across
        # independently constructed instances
        shards = [self._shards[name] for name in sorted(self._shards)]
        if shuffle_shards:
            random.Random(seed).shuffle(shards)
        if self._catalog is None:
            counts = [shard.member_count for shard in shards]
        else:
            counts = [self._catalog.count(_shard_name(shard.path)) for shard in shards]
        total = sum(counts)
        start = worker_id * total // num_workers
        stop = (worker_id + 1) * total // num_workers
        position = 0
        for shard, count in zip(shards, counts):
            if position + count <= start or position >= stop:
                position += count
                continue
            # stream this shard's entries; a catalog holds them in SQLite, so
            # a million-member set never materializes its full listing here
            if self._catalog is None:
                entries = ((key, None, None) for key in shard.member_names())
            else:
                entries = self._catalog.entries(_shard_name(shard.path))
            for index, (key, offset, size) in enumerate(entries):
                if start <= position + index < stop:
                    yield key, shard.read(key, offset=offset, size=size)
            position += count

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()
        for source in self._deferred.values():
            source.close()
        self._pool.close()
        if self._catalog is not None:
            self._catalog.close()
            self._catalog = None

    def __enter__(self) -> PackSet:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
