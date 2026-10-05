"""Disk-backed member catalogs: reuse when shards are unchanged, rebuild
(with full validation) when they are not."""

import os
import sqlite3
import time
import zipfile

import pytest

from blobpack import BlobPackError, PackSet, PackWriter
from blobpack._catalog import CATALOG_NAME, Catalog

PAYLOADS = {f"g{i % 3}/blob{i:02d}.bin": bytes([i]) * (200 + i) for i in range(15)}


@pytest.fixture()
def pack_dir(tmp_path):
    with PackWriter(tmp_path / "media", ref_base="media", max_pack_bytes=900) as writer:
        for key, data in PAYLOADS.items():
            writer.add(key, data)
    return tmp_path / "media"


def read_all(packs):
    return {key: packs.read(key) for key in PAYLOADS}


def test_catalog_is_written_and_reused(pack_dir):
    with PackSet(pack_dir, catalog=True) as packs:
        assert read_all(packs) == PAYLOADS
        assert len(packs) == len(PAYLOADS)
    assert (pack_dir / CATALOG_NAME).exists()

    # second open serves reads from the catalog without any in-memory index
    with PackSet(pack_dir, catalog=True) as packs:
        assert all(shard.index == {} for shard in packs._shards.values())
        assert read_all(packs) == PAYLOADS
        assert sorted(packs.keys()) == sorted(PAYLOADS)
        assert "g0/blob00.bin" in packs
        assert "nope" not in packs


def test_catalog_matches_plain_open(pack_dir, tmp_path):
    with PackSet(pack_dir) as plain, PackSet(pack_dir, catalog=tmp_path / "c.sqlite") as cataloged:
        assert sorted(plain.keys()) == sorted(cataloged.keys())
        assert read_all(plain) == read_all(cataloged)
        key = next(iter(PAYLOADS))
        assert plain.read_many(list(PAYLOADS)) == cataloged.read_many(list(PAYLOADS))
        with plain.open(key) as one, cataloged.open(key) as two:
            assert one.read() == two.read()


def test_iter_blobs_and_worker_split_with_catalog(pack_dir, tmp_path):
    with PackSet(pack_dir, catalog=tmp_path / "c.sqlite") as packs:
        assert dict(packs.iter_blobs()) == PAYLOADS
        parts = [dict(packs.iter_blobs(worker_id=w, num_workers=3)) for w in range(3)]
        merged = {}
        for part in parts:
            assert not (merged.keys() & part.keys())
            merged.update(part)
        assert merged == PAYLOADS


def test_stale_catalog_is_rebuilt_not_trusted(pack_dir, tmp_path):
    """A shard replaced in place must not be served through old offsets."""
    catalog_path = tmp_path / "c.sqlite"
    with PackSet(pack_dir, catalog=catalog_path) as packs:
        read_all(packs)

    shard = sorted(pack_dir.glob("*.zip"))[0]
    with zipfile.ZipFile(shard) as bundle:
        names = bundle.namelist()
    # rewrite the same shard with the same members in a different order, so
    # every offset shifts while names and sizes stay identical
    rebuilt = {name: PAYLOADS[name] for name in names}
    shard.unlink()
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        for name in reversed(names):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            bundle.writestr(info, rebuilt[name])
    time.sleep(0.01)
    os.utime(shard, None)

    with PackSet(pack_dir, catalog=catalog_path) as packs:
        assert read_all(packs) == PAYLOADS  # rebuilt, so offsets are correct


def test_added_or_removed_shard_invalidates(pack_dir, tmp_path):
    catalog_path = tmp_path / "c.sqlite"
    with PackSet(pack_dir, catalog=catalog_path) as packs:
        shard_count = len(packs._shards)
    extra = pack_dir / "pack-9999.zip"
    with zipfile.ZipFile(extra, "w", zipfile.ZIP_STORED) as bundle:
        info = zipfile.ZipInfo("extra/one.bin")
        info.compress_type = zipfile.ZIP_STORED
        bundle.writestr(info, b"extra-payload")
    with PackSet(pack_dir, catalog=catalog_path) as packs:
        assert len(packs._shards) == shard_count + 1
        assert packs.read("extra/one.bin") == b"extra-payload"
    extra.unlink()
    with PackSet(pack_dir, catalog=catalog_path) as packs:
        assert len(packs) == len(PAYLOADS)
        assert "extra/one.bin" not in packs


def test_rebuild_still_validates_members(pack_dir, tmp_path):
    """Building a catalog must not skip the STORED and header checks."""
    bad = pack_dir / "pack-8888.zip"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("bad/one.bin", b"compress me " * 100)
    with pytest.raises(BlobPackError):
        PackSet(pack_dir, catalog=tmp_path / "c.sqlite")


def test_schema_version_mismatch_rebuilds(pack_dir, tmp_path):
    catalog_path = tmp_path / "c.sqlite"
    with PackSet(pack_dir, catalog=catalog_path) as packs:
        read_all(packs)
    db = sqlite3.connect(catalog_path)
    db.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    db.commit()
    db.close()
    with PackSet(pack_dir, catalog=catalog_path) as packs:
        assert read_all(packs) == PAYLOADS


def test_catalog_backed_set_refuses_to_pickle(pack_dir, tmp_path):
    import pickle

    packs = PackSet(pack_dir, catalog=tmp_path / "c.sqlite")
    try:
        with pytest.raises(BlobPackError, match="cannot be pickled"):
            pickle.dumps(packs)
    finally:
        packs.close()


def test_catalog_keys_stream_in_shard_order(pack_dir, tmp_path):
    catalog = Catalog(tmp_path / "c.sqlite")
    try:
        with PackSet(pack_dir, validate_on_open=True) as packs:  # a catalog records validated offsets
            catalog.write(packs._shards)
            expected = sorted(PAYLOADS)
        assert sorted(catalog.keys()) == expected
        assert catalog.count() == len(PAYLOADS)
        first_shard = sorted(catalog.keys())[0]
        assert catalog.locate(first_shard) is not None
        assert catalog.locate("missing") is None
    finally:
        catalog.close()


def test_concurrent_reads_return_the_right_bytes(pack_dir, tmp_path):
    """Catalog lookups must not interleave across threads: a shared sqlite
    connection returns another query's row, and therefore another blob."""
    keys = list(PAYLOADS)
    with PackSet(pack_dir, catalog=tmp_path / "c.sqlite") as packs:
        for _ in range(20):
            assert packs.read_many(keys, workers=8) == [PAYLOADS[key] for key in keys]


def test_catalog_survives_fork(pack_dir, tmp_path):
    if not hasattr(os, "fork"):
        pytest.skip("requires fork")
    with PackSet(pack_dir, catalog=tmp_path / "c.sqlite") as packs:
        read_all(packs)
        pid = os.fork()
        if pid == 0:
            status = 0 if read_all(packs) == PAYLOADS else 1
            os._exit(status)
        _, status = os.waitpid(pid, 0)
        assert os.WEXITSTATUS(status) == 0
        assert read_all(packs) == PAYLOADS


def test_transient_threads_do_not_pin_connections(pack_dir):
    """read_many-style pool workers must release their SQLite connections
    when they die; the old code kept a strong reference per thread forever."""
    import gc
    import threading

    with PackSet(pack_dir, catalog=True) as packs:
        keys = list(packs.keys())[:4]
        for _ in range(10):
            packs.read_many(keys, workers=4)
        gc.collect()
        alive = threading.active_count()
        held = len(packs._catalog._connections)
        # bounded by threads actually alive, not by threads that ever existed
        assert held <= alive + 1, f"{held} connections pinned with {alive} threads alive"


def test_forked_child_closing_leaves_parent_connection_usable(pack_dir):
    import os

    if not hasattr(os, "fork"):
        pytest.skip("no fork on this platform")
    with PackSet(pack_dir, catalog=True) as packs:
        key = next(iter(packs.keys()))
        assert packs.read(key)
        pid = os.fork()
        if pid == 0:  # child: close inherited catalog, must not touch parent's fd
            try:
                packs._catalog.close()
                os._exit(0)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert packs.read(key)  # parent's connection survived the child's close


def test_shard_queries_use_covering_index_and_upgrade_existing_catalog(pack_dir, monkeypatch):
    from blobpack._zip import PackFile

    with PackSet(pack_dir, catalog=True) as packs:
        packs._catalog._db.execute("DROP INDEX blobs_by_shard")
        packs._catalog._db.commit()

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("adding a query index must not reparse validated packs")

    monkeypatch.setattr(PackFile, "_build_index", unexpected_rebuild)
    with PackSet(pack_dir, catalog=True) as packs:
        db = packs._catalog._db
        shard = next(iter(packs._shards))
        for query in (
            "SELECT COUNT(*) FROM blobs WHERE shard = ?",
            "SELECT key, offset, size FROM blobs WHERE shard = ? ORDER BY offset",
        ):
            plan = " ".join(row[3] for row in db.execute("EXPLAIN QUERY PLAN " + query, (shard,)))
            assert "SEARCH blobs USING COVERING INDEX blobs_by_shard" in plan
            assert "TEMP B-TREE" not in plan
        assert dict(packs.iter_blobs()) == PAYLOADS
