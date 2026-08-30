"""Byte-range sources behind a pack shard.

A shard is read only through positioned range requests, so the same reader
works over a local descriptor and over object storage. Sources own
process-bound state (descriptors, sessions); indices do not, and never
travel inside a pickle.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from ._zip import BlobPackError, CorruptPackError

_HAS_PREAD = hasattr(os, "pread")


class RangeSource:
    """Positioned reads over one shard. Implementations must be thread-safe."""

    #: identifies the shard for error messages and shard-name lookups
    path: str

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

    @property
    def is_open(self) -> bool:
        return False

    def release(self) -> bool:
        """Drop idle process-bound state; return True if anything was freed."""
        return False

    def close(self) -> None:
        pass


class LocalSource(RangeSource):
    """A local file read with ``pread`` (or a lock-guarded seek+read).

    The descriptor is opened lazily and re-opened whenever the owning
    process changes, so a descriptor inherited by a forked child or revived
    by unpickling is never reused or closed by the wrong process. Reads
    borrow the descriptor, so a pool can only reclaim idle shards.
    """

    def __init__(self, path: os.PathLike | str):
        self.path = os.fspath(path)
        self._fd = -1
        self._pid = -1
        self._inflight = 0
        self._state_lock = threading.Lock()
        self._read_lock = None if _HAS_PREAD else threading.Lock()

    def _ensure_fd(self) -> int:
        pid = os.getpid()
        if self._fd >= 0 and self._pid == pid:
            return self._fd
        if self._fd >= 0:
            self._fd = -1  # inherited: owned by another process, never closed here
        self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self._pid = pid
        return self._fd

    def _acquire(self) -> int:
        with self._state_lock:
            fd = self._ensure_fd()
            self._inflight += 1
            return fd

    def _release_borrow(self) -> None:
        with self._state_lock:
            self._inflight -= 1

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
        with self._state_lock:
            if self._inflight or self._fd < 0:
                return False
            if self._pid == os.getpid():
                os.close(self._fd)
            self._fd = -1
            return True

    def close(self) -> None:
        with self._state_lock:
            if self._fd >= 0:
                if self._pid == os.getpid():
                    os.close(self._fd)
                self._fd = -1

    def __getstate__(self) -> dict:
        return {"path": self.path}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["path"])


class FsspecSource(RangeSource):
    """A remote object read with stateless ranged requests.

    Payload reads go through ``cat_file``/``cat_ranges``, never a shared
    seekable handle: an fsspec file object carries a position and its own
    buffering, which cannot be shared across threads or survive a fork.
    Pickling carries the protocol and the storage options -- including any
    credentials they hold -- and rebuilds the filesystem from them, rather
    than pickling a live client.
    """

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

        A synchronous filesystem's ``cat_file`` goes through a buffered
        file whose readahead cache fetches a whole block (4 MiB on the
        Hub) per call -- a 34-byte header read costing 100,000x its size.
        ``cache_type="none"`` makes the read exact; backends whose open()
        does not take the kwarg fall back to the plain call.
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
        self._check_range_support()
        data = self._cat_exact(size, offset)
        if len(data) > size:  # some backends over-deliver; never under-report
            data = data[:size]
        return data

    #: mean gap below which a batch is served by one sequential sweep
    DENSE_GAP_BYTES = 1 << 20

    def _read_batch_sweep(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        """Serve a dense batch from one sequential pass over the shard.

        When members are small and many, their headers sit a few hundred
        kilobytes apart; per-range requests would pay a round-trip (and,
        on buffered backends, a block fetch) per member. One stream with
        forward seeks reads the covered region once, in order.
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
