"""Byte-range sources behind a pack shard.

A shard is read only through positioned range requests, so the same reader
works over a local descriptor and over object storage. Sources own
process-bound state (descriptors, sessions); indices do not, and never
travel inside a pickle.
"""

from __future__ import annotations

import io
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from ._zip import BlobPackError, CorruptPackError

_HAS_PREAD = hasattr(os, "pread")
_PROCESS_LOCK = threading.RLock()


def _reset_process_lock():
    global _PROCESS_LOCK
    _PROCESS_LOCK = threading.RLock()


# An inherited lock may be held by a vanished parent thread. Register once,
# rather than retaining every source/pool through per-instance fork callbacks.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_lock)


class RangeSource:
    """Positioned reads over one shard. Implementations must be thread-safe."""

    #: identifies the shard for error messages and shard-name lookups
    path: str

    #: True when scattered small reads cost a round-trip each (object
    #: storage); readers then defer per-member validation to first read
    #: instead of paying one such read per member at open (issue #11)
    lazy_validation = False

    def size(self) -> int:
        raise NotImplementedError

    def read_at(self, size: int, offset: int) -> bytes:
        raise NotImplementedError

    def read_batch(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        """Read many ``(size, offset)`` ranges; may issue one request."""
        return [self.read_at(size, offset) for size, offset in ranges]

    def open_stream(self):
        """A fresh seekable binary file object over the whole shard.

        Used transiently while parsing the central directory; never shared
        between threads or retained.
        """
        raise NotImplementedError

    def open_directory_stream(self):
        """Like ``open_stream`` but tuned for reading the central directory,
        which lives at the tail of the archive."""
        return self.open_stream()

    @property
    def is_open(self) -> bool:
        return False

    def release(self) -> bool:
        """Drop idle process-bound state; return True if anything was freed."""
        return False

    def close(self) -> None:
        pass


class DescriptorPool:
    """Process-local LRU, enforced at actual reads (including open member views).

    Concurrent borrowed descriptors may temporarily exceed the budget; idle
    descriptors are reclaimed before opening another and after a read ends.
    """

    def __init__(self, limit):
        self.limit = limit
        self._pid = os.getpid()
        self._recent = OrderedDict()
        self._lock = threading.Lock()

    def _process(self):
        if self._pid != os.getpid():
            with _PROCESS_LOCK:
                if self._pid == os.getpid():
                    return
                for source in self._recent:
                    source._process()
                self._recent = OrderedDict()
                self._lock = threading.Lock()
                self._pid = os.getpid()

    def _trim(self, target):
        if len(self._recent) <= target:
            return
        for source in list(self._recent):
            if len(self._recent) <= target:
                break
            if not source.is_open or source.release():
                del self._recent[source]

    def acquire(self, source):
        self._process()
        with self._lock:
            self._recent.pop(source, None)
            self._trim(self.limit - 1)
            fd = source._acquire_local()
            self._recent[source] = None
            return fd

    def trim(self):
        self._process()
        with self._lock:
            self._trim(self.limit)

    def close(self):
        self._process()
        with self._lock:
            for source in self._recent:
                source.close()
            self._recent.clear()

    def __getstate__(self):
        return {"limit": self.limit}

    def __setstate__(self, state):
        self.__init__(state["limit"])


class LocalSource(RangeSource):
    """A local file read with ``pread`` (or a lock-guarded seek+read).

    The descriptor is opened lazily and re-opened whenever the owning
    process changes. After fork, close the child's inherited descriptor
    copy before reopening; this does not close the parent's descriptor.
    Reads borrow the descriptor, so a pool can only reclaim idle shards.
    """

    def __init__(self, path: os.PathLike | str, *, pool=None):
        self.path = os.fspath(path)
        self.pool = pool
        self._fd = -1
        self._pid = os.getpid()
        self._inflight = 0
        self._state_lock = threading.Lock()
        self._read_lock = None if _HAS_PREAD else threading.Lock()

    def _process(self):
        if self._pid != os.getpid():
            with _PROCESS_LOCK:
                if self._pid == os.getpid():
                    return
                fd, self._fd = self._fd, -1
                self._inflight = 0
                self._state_lock = threading.Lock()
                self._read_lock = None if _HAS_PREAD else threading.Lock()
                self._pid = os.getpid()
                if fd >= 0:
                    os.close(fd)

    def _ensure_fd(self) -> int:
        if self._fd < 0:
            self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        return self._fd

    def _acquire(self) -> int:
        return self.pool.acquire(self) if self.pool is not None else self._acquire_local()

    def _acquire_local(self) -> int:
        self._process()
        with self._state_lock:
            fd = self._ensure_fd()
            self._inflight += 1
            return fd

    def _release_borrow(self) -> None:
        with self._state_lock:
            self._inflight -= 1
        if self.pool is not None:
            self.pool.trim()

    def size(self) -> int:
        fd = self._acquire()
        try:
            return os.fstat(fd).st_size
        finally:
            self._release_borrow()

    def read_at(self, size: int, offset: int) -> bytes:
        if size == 0:
            return b""
        fd = self._acquire()
        try:
            if self._read_lock is None:
                return os.pread(fd, size, offset)
            with self._read_lock:
                os.lseek(fd, offset, os.SEEK_SET)
                return os.read(fd, size)
        finally:
            self._release_borrow()

    def open_stream(self):
        return open(self.path, "rb")

    @property
    def is_open(self) -> bool:
        return self._fd >= 0 and self._pid == os.getpid()

    def release(self) -> bool:
        self._process()
        with self._state_lock:
            if self._inflight or self._fd < 0:
                return False
            os.close(self._fd)
            self._fd = -1
            return True

    def close(self) -> None:
        self._process()
        with self._state_lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1

    def __getstate__(self) -> dict:
        return {"path": self.path, "pool": self.pool}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["path"], pool=state["pool"])


class _TailStream(io.RawIOBase):
    """A seekable read-only view for central-directory parsing.

    Caches the object's tail and grows the cached window downward with one
    ranged read per miss, so zipfile's EOCD scan plus the directory pass
    costs two or three requests -- instead of one readahead block per seek.
    """

    # sized so EOCD (zipfile scans at most max-comment + 22 bytes for it)
    # plus a typical central directory arrive in one request: remote reads
    # are round-trip bound, and 256 KiB of transfer is noise next to one.
    # A larger directory costs exactly one gap fetch.
    INITIAL_WINDOW = 256 << 10

    def __init__(self, source: RangeSource):
        super().__init__()
        self._source = source
        self._size = source.size()
        self._window_start = max(self._size - self.INITIAL_WINDOW, 0)
        self._buffer = source.read_at(self._size - self._window_start, self._window_start)
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._pos, os.SEEK_END: self._size}[whence]
        self._pos = base + offset
        return self._pos

    def readinto(self, buffer) -> int:
        want = min(len(buffer), self._size - self._pos)
        if want <= 0:
            return 0
        if self._pos < self._window_start:
            # one fetch covers the gap; directory reads ascend afterwards
            self._buffer = self._source.read_at(self._window_start - self._pos, self._pos) + self._buffer
            self._window_start = self._pos
        start = self._pos - self._window_start
        chunk = self._buffer[start : start + want]
        buffer[: len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)


class FsspecSource(RangeSource):
    """A remote object read with stateless ranged requests.

    Payload reads go through ``cat_file``/``cat_ranges``, never a shared
    seekable handle: an fsspec file object carries a position and its own
    buffering, which cannot be shared across threads or survive a fork.
    Pickling carries the protocol and the storage options -- including any
    credentials they hold -- and rebuilds the filesystem from them, rather
    than pickling a live client.
    """

    lazy_validation = True

    def __init__(self, fs, path: str, *, storage_options: dict | None = None):
        self.fs = fs
        self.path = path
        self._storage_options = dict(storage_options or {})
        self._verified_ranges = False

    def _check_range_support(self) -> None:
        """Refuse a backend that answers ranged requests with whole objects."""
        if self._verified_ranges:
            return
        total = self.size()
        probe = min(4, total)
        offset = max(total - probe, 0)  # a nonzero start; offset-0 ranges can succeed on backends that ignore starts
        if probe:
            data = self.fs.cat_file(self.path, start=offset, end=offset + probe)
            if len(data) != probe:
                raise BlobPackError(
                    f"{self.path}: backend ignored a byte range (asked {probe} bytes at {offset}, got {len(data)}); "
                    "blobpack needs range-capable storage"
                )
        self._verified_ranges = True

    def size(self) -> int:
        return int(self.fs.info(self.path)["size"])

    def _cat_exact(self, size: int, offset: int) -> bytes:
        """One ranged read that costs what it asks for.

        A synchronous ``cat_file`` fetches a whole readahead block (4 MiB
        on the Hub) per call; ``cache_type="none"`` makes the read exact,
        with a fallback for backends whose open() lacks the kwarg.
        """
        if getattr(self.fs, "async_impl", False):
            return self.fs.cat_file(self.path, start=offset, end=offset + size)
        try:
            return self.fs.cat_file(self.path, start=offset, end=offset + size, cache_type="none")
        except TypeError:
            return self.fs.cat_file(self.path, start=offset, end=offset + size)

    def read_at(self, size: int, offset: int) -> bytes:
        if size == 0:
            return b""
        if self._verified_ranges:
            data = self._cat_exact(size, offset)
            return data[:size] if len(data) > size else data
        # the first read doubles as the range-capability probe: an exact
        # answer to a nonzero-offset request proves range support, and a
        # request at the tail can never coincide with the whole object.
        # Ambiguous shapes (offset 0, short reads) pay the explicit probe.
        if offset == 0:
            self._check_range_support()
            return self.read_at(size, offset)
        data = self._cat_exact(size, offset)
        if len(data) == size:
            self._verified_ranges = True
            return data
        self._check_range_support()  # raises on a range-ignoring backend
        return data[:size] if len(data) > size else data

    #: mean gap below which a batch is served by one sequential sweep
    DENSE_GAP_BYTES = 1 << 20

    def _read_batch_sweep(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        """Serve a dense batch from one sequential pass over the shard,
        instead of paying a round-trip (and a block fetch) per member.
        """
        order = sorted(range(len(ranges)), key=lambda i: ranges[i][1])
        out: list[bytes] = [b""] * len(ranges)
        with self.fs.open(self.path, "rb", cache_type="readahead", block_size=8 << 20) as stream:
            for index in order:
                size, offset = ranges[index]
                stream.seek(offset)
                out[index] = stream.read(size)
        return out

    def read_batch(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        if not ranges:
            return []
        self._check_range_support()
        cat_ranges = getattr(self.fs, "cat_ranges", None)
        if cat_ranges is None or not getattr(self.fs, "async_impl", False):
            offsets = [offset for _, offset in ranges]
            span = max(o + s for s, o in ranges) - min(offsets)
            if span // len(ranges) <= self.DENSE_GAP_BYTES:
                return self._read_batch_sweep(ranges)
            # sparse ranges: a sweep would read the gaps; exact concurrent
            # per-range reads pay one round-trip each instead
            with ThreadPoolExecutor(min(16, len(ranges))) as pool:
                return list(pool.map(lambda r: self.read_at(r[0], r[1]), ranges))
        starts = [offset for _, offset in ranges]
        ends = [offset + size for size, offset in ranges]
        chunks = cat_ranges([self.path] * len(ranges), starts, ends)
        if len(chunks) != len(ranges):
            raise CorruptPackError(f"{self.path}: backend returned {len(chunks)} of {len(ranges)} ranges")
        out = []
        for (size, offset), chunk in zip(ranges, chunks):
            if isinstance(chunk, BaseException):
                raise CorruptPackError(f"{self.path}: ranged read failed at {offset}: {chunk!r}")
            out.append(chunk[:size])
        return out

    def open_stream(self):
        return self.fs.open(self.path, "rb")

    def open_directory_stream(self):
        return _TailStream(self)

    def __getstate__(self) -> dict:
        protocol = getattr(self.fs, "protocol", None)
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0]
        return {
            "path": self.path,
            "protocol": protocol,
            "storage_options": self._storage_options,
        }

    def __setstate__(self, state: dict) -> None:
        import fsspec

        fs = fsspec.filesystem(state["protocol"], **state["storage_options"])
        self.__init__(fs, state["path"], storage_options=state["storage_options"])
