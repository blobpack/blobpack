"""A disk-backed catalog of a pack set's members.

Opening a pack set normally parses every shard's central directory and keeps
one dict entry per member: fine for thousands of blobs, expensive for
millions. A catalog moves that mapping into SQLite, so a later open costs a
few queries instead of a full parse, and lookups do not hold the whole
index in memory.

A catalog is a cache of validated structure, so it is only reused when every
shard still has the exact identity recorded when it was built. Identity is
deliberately strong: size, mtime, ctime, device and inode locally, so a
shard that was replaced in place cannot be served through stale offsets that
skip the STORED and header checks.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
import weakref
from pathlib import Path

from ._zip import BlobPackError

CATALOG_NAME = ".blobpack-catalog.sqlite"
SCHEMA_VERSION = 1


class _ThreadConnection:
    """One thread's SQLite connection, closed when the thread lets go.

    Held only through the owning thread's local storage, so a transient
    thread -- one of ``read_many``'s pool workers, say -- releases its
    connection the moment it dies, instead of pinning an open descriptor
    for every thread that ever touched the catalog. The catalog itself
    keeps just a weak reference for ``close()``.
    """

    __slots__ = ("__weakref__", "connection", "pid")

    def __init__(self, connection: sqlite3.Connection, pid: int):
        self.connection = connection
        self.pid = pid

    def close(self) -> None:
        if self.pid == os.getpid():  # never close a descriptor a fork inherited
            with contextlib.suppress(sqlite3.Error):
                self.connection.close()

    def __del__(self):
        self.close()


def shard_identity(source) -> str:
    """A string that changes whenever a shard's bytes could have changed.

    Locally: size, mtime, ctime, device and inode. Remotely: size plus
    whatever change marker the backend exposes (ETag, version, mtime).
    A backend exposing none gets ``size=N;weak``, which ``matches`` never
    trusts -- a same-size replacement would otherwise be served through
    stale offsets that skipped every structural check.
    """
    path = Path(source.path)
    try:
        stat = path.stat()
    except OSError:  # remote or otherwise not a local file
        return _remote_identity(source)
    return (
        f"size={stat.st_size};mtime_ns={stat.st_mtime_ns};"
        f"ctime_ns={stat.st_ctime_ns};dev={stat.st_dev};ino={stat.st_ino}"
    )


def _remote_identity(source) -> str:
    size = source.size()
    fs = getattr(source, "fs", None)
    info = {}
    if fs is not None:
        try:
            info = fs.info(source.path) or {}
        except OSError:
            info = {}
    for field in ("ETag", "etag", "version_id", "generation", "mtime", "LastModified", "last_modified"):
        marker = info.get(field)
        if marker:
            return f"size={size};{field.lower()}={marker}"
    return f"size={size};weak"


def _is_weak(identity: str) -> bool:
    return identity.endswith(";weak")


class Catalog:
    """SQLite mapping of ``key -> (shard, offset, size)`` for one pack set."""

    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        # One connection per thread and process: a shared sqlite3 connection
        # interleaves cursors under concurrent reads, which silently returns
        # another query's row (and therefore another blob's bytes).
        self._local = threading.local()
        self._connections: weakref.WeakSet[_ThreadConnection] = weakref.WeakSet()
        self._lock = threading.Lock()
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS shards (
                name TEXT PRIMARY KEY,
                identity TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blobs (
                key TEXT PRIMARY KEY,
                shard TEXT NOT NULL,
                offset INTEGER NOT NULL,
                size INTEGER NOT NULL
            );
            """
        )
        self._db.commit()

    @property
    def _db(self) -> sqlite3.Connection:
        """The calling thread's connection, reopened after a fork."""
        pid = os.getpid()
        held = getattr(self._local, "held", None)
        if held is not None and held.pid == pid:
            return held.connection
        connection = sqlite3.connect(str(self.path))
        connection.execute("PRAGMA busy_timeout=5000")
        held = _ThreadConnection(connection, pid)
        self._local.held = held
        with self._lock:
            self._connections.add(held)
        return connection

    # -- construction ----------------------------------------------------

    def write(self, shards: dict[str, object]) -> None:
        """Record every shard's members and identity, replacing any earlier
        contents in one transaction."""
        with self._db:
            self._db.execute("DELETE FROM blobs")
            self._db.execute("DELETE FROM shards")
            self._db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            for name, shard in shards.items():
                self._db.execute(
                    "INSERT INTO shards VALUES (?, ?)",
                    (name, shard_identity(shard.source)),
                )
                self._db.executemany(
                    "INSERT INTO blobs VALUES (?, ?, ?, ?)",
                    ((key, name, offset, size) for key, (offset, size) in shard.index.items()),
                )

    # -- reuse -----------------------------------------------------------

    def matches(self, shards: dict[str, object]) -> bool:
        """True when the catalog describes exactly these shards, unchanged."""
        row = self._db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if not row or int(row[0]) != SCHEMA_VERSION:
            return False
        recorded = dict(self._db.execute("SELECT name, identity FROM shards"))
        if set(recorded) != set(shards):
            return False
        for name, shard in shards.items():
            identity = shard_identity(shard.source)
            if _is_weak(identity) or recorded[name] != identity:
                return False  # size alone cannot vouch for unchanged bytes
        return True

    def locate(self, key: str) -> tuple[str, int, int] | None:
        row = self._db.execute("SELECT shard, offset, size FROM blobs WHERE key = ?", (key,)).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def count(self, shard: str | None = None) -> int:
        if shard is None:
            return int(self._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0])
        return int(self._db.execute("SELECT COUNT(*) FROM blobs WHERE shard = ?", (shard,)).fetchone()[0])

    def keys(self, shard: str | None = None):
        """Iterate keys lazily; SQLite streams rows, so memory stays bounded."""
        if shard is None:
            cursor = self._db.execute("SELECT key FROM blobs ORDER BY shard, offset")
        else:
            cursor = self._db.execute("SELECT key FROM blobs WHERE shard = ? ORDER BY offset", (shard,))
        for (key,) in cursor:
            yield key

    def entries(self, shard: str):
        cursor = self._db.execute("SELECT key, offset, size FROM blobs WHERE shard = ? ORDER BY offset", (shard,))
        yield from cursor

    def close(self) -> None:
        with self._lock:
            held, self._connections = list(self._connections), weakref.WeakSet()
        for entry in held:  # only this process's descriptors; a fork's parent keeps its own
            entry.close()
        self._local = threading.local()


def open_catalog(pack_dir: Path, catalog: bool | str | os.PathLike) -> Catalog:
    path = pack_dir / CATALOG_NAME if catalog is True else Path(catalog)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return Catalog(path)
    except sqlite3.Error as exc:
        raise BlobPackError(f"cannot open catalog at {path}: {exc}") from exc
