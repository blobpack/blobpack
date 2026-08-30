#!/usr/bin/env python3
"""Blob Pack benchmark: when and how much does packing loose media into
STORED-zip shards help?

Real data, two size regimes:
  coco_images   COCO train2017, all 118K JPEGs (avg ~160 KB)
  pusht_frames  LeRobot PushT: every episode frame as a 96x96 JPEG (~2-5 KB)
Formats compared on each filesystem (shared FS + node-local NVMe):

  loose/   one file per blob, upstream layout                (baseline)
  pack/    STORED zip shards, creation order, ~512 MiB       (Blob Pack)
  data.tar single uncompressed tar                            (sequential-only ref)

Measured: creation/conversion, disk usage + inodes, sequential read
(cold/warm) over a fixed item prefix, random-access read on a fixed probe
sample (1 and 8 threads; per-call vs cached zip handles), pack-shuffled
chunked-sequential read, decode throughput, and folder ops (find, rsync
relocation, rm). Cold reads approximated with posix_fadvise(DONTNEED).

Sampling protocol: throughput and latency are measured on samples, not the
full dataset — sequential passes read the first SEQ_PREFIX items (sized to
~SEQ_CAP bytes), random passes use RAND_PROBES draws from a full-dataset
shuffle, folder ops relocate an OPS_FILES subset for loose. Sample sizes are
recorded in `params` and results are normalized per item/byte in the report.

Design-space extras (coco variant only): ZIP_STORED vs ZIP_DEFLATE, shard-size
sweep (64 MiB / 512 MiB / 2 GiB), random access via zip central directory vs
uncompressed tar + external offset index, embedding bytes in a Parquet table,
and Lance datasets (inline large_binary column, and Lance's blob storage
class) as the strongest table-format baseline.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SEED = 0
SHARD_TARGET = 512 << 20
SEQ_CAP = int(os.environ.get("BENCH_SEQ_CAP_BYTES", str(4 << 30)))
RAND_PROBES = int(os.environ.get("BENCH_RAND_PROBES", "4000"))
PERCALL_PROBES = int(os.environ.get("BENCH_PERCALL_PROBES", "2000"))
PARQUET_PROBES = int(os.environ.get("BENCH_PARQUET_PROBES", "300"))
DECODE_N = int(os.environ.get("BENCH_DECODE_N", "5000"))
OPS_FILES = int(os.environ.get("BENCH_OPS_FILES", "20000"))
SOURCES = Path(os.environ.get("BENCH_SOURCES", "sources"))
VARIANTS = {
    "coco_images": {"source": "coco", "cap": None, "extras": True},  # all 118K
    "pusht_frames": {"source": "pusht", "cap": None},
}
TARGETS = {
    name: Path(path)
    for name, path in (
        kv.split("=", 1) for kv in os.environ["BENCH_TARGETS"].split(",")
    )
}
OUT_JSON = Path(os.environ.get("BENCH_OUT", "bench_results.json"))

RESULTS = {
    "seed": SEED,
    "variants": {},
    "params": {
        "seq_cap_bytes": SEQ_CAP,
        "rand_probes": RAND_PROBES,
        "percall_probes": PERCALL_PROBES,
        "parquet_probes": PARQUET_PROBES,
        "decode_n": DECODE_N,
        "ops_files": OPS_FILES,
    },
    "notes": [
        "cold reads approximated with posix_fadvise(DONTNEED) on the files a pass "
        "will touch; a shared-FS server may still serve some blocks from its own "
        "cache, so cold figures are upper bounds on cold performance",
        "creation of loose includes JPEG encoding for pusht; pack/tar are "
        "conversions from loose (read+write), reported separately",
        "sequential passes read the same item prefix in every format; random "
        "passes use the same probe sample; seconds are per recorded sample size",
    ],
}


def log(msg):
    print(msg, flush=True)


def ingest_coco(loose: Path, cap: int):
    """First `cap` images of COCO train2017, original names, one flat dir
    (that IS the upstream layout, and the realistic loose baseline)."""
    keys = []
    with zipfile.ZipFile(SOURCES / "train2017.zip") as bundle:
        names = [n for n in bundle.namelist() if n.endswith(".jpg")][:cap]
        (loose / "train2017").mkdir(parents=True)
        for name in names:
            key = name  # train2017/000000xxxxxx.jpg
            (loose / key).write_bytes(bundle.read(name))
            keys.append(key)
    return keys


def pusht_episodes():
    """(episode_index, from_index, to_index) from the LeRobot v3 metadata;
    the release concatenates all episodes into one AV1 MP4."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        next(SOURCES.glob("pusht/meta/episodes/**/*.parquet")),
        columns=["episode_index", "dataset_from_index", "dataset_to_index"],
    )
    return sorted(
        zip(
            *[
                table.column(c).to_pylist()
                for c in ("episode_index", "dataset_from_index", "dataset_to_index")
            ]
        )
    )


def ingest_pusht(loose: Path, cap):
    """Every PushT frame as a JPEG, one directory per episode. Frames are
    dumped from the single concatenated video with ffmpeg (AV1)."""
    import tempfile

    video = next(SOURCES.glob("pusht/videos/**/*.mp4"))
    keys = []
    with tempfile.TemporaryDirectory(dir=loose.parent) as tmp:
        subprocess.run(
            [
                os.environ.get("FFMPEG", "ffmpeg"),
                "-v",
                "error",
                "-i",
                str(video),
                "-q:v",
                "3",
                f"{tmp}/%06d.jpg",
            ],
            check=True,
        )
        for episode, start, stop in pusht_episodes():
            edir = loose / f"episode_{episode:03d}"
            edir.mkdir(parents=True, exist_ok=True)
            for offset, dataset_index in enumerate(range(start, stop)):
                key = f"episode_{episode:03d}/{offset:04d}.jpg"
                os.rename(f"{tmp}/{dataset_index + 1:06d}.jpg", loose / key)
                keys.append(key)
                if cap and len(keys) >= cap:
                    return keys
    return keys


def du_bytes(path):
    out = subprocess.run(["du", "-sB1", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def apparent_bytes(path):
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def inode_count(path):
    out = subprocess.run(
        ["bash", "-c", f"find {path} | wc -l"], capture_output=True, text=True
    )
    return int(out.stdout.strip())


def sync_first():
    """fadvise(DONTNEED) cannot drop dirty pages, so anything this process
    wrote must be flushed before eviction or 'cold' passes read cache."""
    os.sync()


def fadvise_dontneed(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def drop_cache(*paths):
    """fadvise(DONTNEED) exactly the files a pass will touch: file paths as-is,
    directories recursively (packs, lance datasets: few large files)."""
    sync_first()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for dirpath, _, files in os.walk(path):
                for fn in files:
                    try:
                        fadvise_dontneed(os.path.join(dirpath, fn))
                    except OSError:
                        pass
        else:
            try:
                fadvise_dontneed(path)
            except OSError:
                pass


def drop_loose(base, keys, indices=None):
    loose = base / "loose"
    sync_first()
    names = keys if indices is None else [keys[i] for i in indices]
    for k in names:
        try:
            fadvise_dontneed(loose / k)
        except OSError:
            pass


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return time.perf_counter() - t0, value


def build_variant(base: Path, spec):
    loose = base / "loose"
    metrics = {}
    if not (base / ".complete").exists():
        if base.exists():
            shutil.rmtree(base)
        loose.mkdir(parents=True)
        t0 = time.perf_counter()
        ingest = ingest_coco if spec["source"] == "coco" else ingest_pusht
        keys = ingest(loose, spec["cap"])
        metrics["loose_create_s"] = round(time.perf_counter() - t0, 2)
        (base / "keys.json").write_text(json.dumps(keys))
        metrics["n"] = len(keys)
        metrics["payload_bytes"] = sum((loose / k).stat().st_size for k in keys)

        # pack: creation order, STORED, rolling shards
        pack = base / "pack"
        pack.mkdir()
        t0 = time.perf_counter()
        shard, size, writer = (
            0,
            0,
            zipfile.ZipFile(pack / "shard-0000.zip", "w", zipfile.ZIP_STORED),
        )
        for key in keys:
            data = (loose / key).read_bytes()
            if size and size + len(data) > SHARD_TARGET:
                writer.close()
                shard += 1
                writer = zipfile.ZipFile(
                    pack / f"shard-{shard:04d}.zip", "w", zipfile.ZIP_STORED
                )
                size = 0
            writer.writestr(zipfile.ZipInfo(key), data)
            size += len(data)
        writer.close()
        metrics["pack_convert_s"] = round(time.perf_counter() - t0, 2)
        metrics["pack_shards"] = shard + 1

        t0 = time.perf_counter()
        with tarfile.open(base / "data.tar", "w") as tar:
            for key in keys:
                tar.add(loose / key, arcname=key)
        metrics["tar_convert_s"] = round(time.perf_counter() - t0, 2)
        (base / ".complete").write_text(json.dumps(metrics))
    else:
        metrics = json.loads((base / ".complete").read_text())

    for fmt, path in (
        ("loose", loose),
        ("pack", base / "pack"),
        ("tar", base / "data.tar"),
    ):
        metrics[f"{fmt}_du_bytes"] = du_bytes(path)
        metrics[f"{fmt}_apparent_bytes"] = apparent_bytes(path)
        metrics[f"{fmt}_inodes"] = inode_count(path)
    return metrics


def shard_index(base: Path):
    index = {}
    for shard_path in sorted((base / "pack").glob("shard-*.zip")):
        with zipfile.ZipFile(shard_path) as bundle:
            for name in bundle.namelist():
                index[name] = shard_path.name
    return index


def seq_loose(base, keys, limit):
    total = 0
    for key in keys[:limit]:
        total += len((base / "loose" / key).read_bytes())
    return total


def seq_pack_dir(pack: Path, limit):
    total, count = 0, 0
    for shard_path in sorted(pack.glob("shard-*.zip")):
        with zipfile.ZipFile(shard_path) as bundle:
            for name in bundle.namelist():
                total += len(bundle.read(name))
                count += 1
                if count >= limit:
                    return total
    return total


def seq_tar(base, limit):
    total, count = 0, 0
    with tarfile.open(base / "data.tar", "r|") as tar:
        for member in tar:
            total += len(tar.extractfile(member).read())
            count += 1
            if count >= limit:
                break
    return total


def rand_loose(base, keys, order, workers):
    def read(i):
        return len((base / "loose" / keys[i]).read_bytes())

    if workers == 1:
        return sum(read(i) for i in order)
    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(read, order, chunksize=64))


def rand_pack_cached(pack: Path, keys, order, index, workers):
    local = threading.local()

    def handle(shard):
        handles = getattr(local, "handles", None)
        if handles is None:
            handles = local.handles = {}
        if shard not in handles:
            handles[shard] = zipfile.ZipFile(pack / shard)
        return handles[shard]

    def read(i):
        name = keys[i]
        return len(handle(index[name]).read(name))

    if workers == 1:
        return sum(read(i) for i in order)
    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(read, order, chunksize=64))


def rand_pack_percall(base, keys, order, index):
    total = 0
    for i in order:
        name = keys[i]
        with zipfile.ZipFile(base / "pack" / index[name]) as bundle:
            total += len(bundle.read(name))
    return len(order)


def chunked_pack_dir(pack: Path, rng, cap_bytes):
    """The Blob Pack epoch pattern: shuffle shard order, sequential within."""
    shards = sorted(pack.glob("shard-*.zip"))
    rng.shuffle(shards)
    total = 0
    for shard_path in shards:
        with zipfile.ZipFile(shard_path) as bundle:
            for name in bundle.namelist():
                total += len(bundle.read(name))
        if total >= cap_bytes:
            break
    return total


def warm_subset(base, keys, order, index):
    """Pre-read probe payloads so decode passes measure decode, not IO."""
    for i in order:
        (base / "loose" / keys[i]).read_bytes()
    handles = {}
    for i in order:
        name = keys[i]
        shard = index[name]
        if shard not in handles:
            handles[shard] = zipfile.ZipFile(base / "pack" / shard)
        handles[shard].read(name)
    for h in handles.values():
        h.close()


def decode_pass(base, keys, order, source, index, workers):
    def decode_loose(i):
        data = np.fromfile(base / "loose" / keys[i], np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) is not None

    local = threading.local()

    def decode_pack(i):
        handles = getattr(local, "handles", None)
        if handles is None:
            handles = local.handles = {}
        name = keys[i]
        shard = index[name]
        if shard not in handles:
            handles[shard] = zipfile.ZipFile(base / "pack" / shard)
        data = np.frombuffer(handles[shard].read(name), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) is not None

    fn = decode_loose if source == "loose" else decode_pack
    with ThreadPoolExecutor(workers) as pool:
        assert all(pool.map(fn, order, chunksize=64))
    return len(order)


def folder_ops(base: Path, scratch: Path, keys):
    """Relocation and management costs. loose is relocated as an OPS_FILES
    subset (extrapolated in the report); pack is relocated in full."""
    ops = {}
    subset = keys[:OPS_FILES]
    listfile = scratch / "ops_files.txt"
    listfile.write_text("\n".join(subset) + "\n")
    dst = scratch / "copy-loose"
    if dst.exists():
        shutil.rmtree(dst)
    t, _ = timed(
        lambda: subprocess.run(
            [
                "rsync",
                "-a",
                f"--files-from={listfile}",
                str(base / "loose") + "/",
                str(dst) + "/",
            ],
            check=True,
        )
    )
    ops["loose_rsync_s"] = round(t, 2)
    ops["loose_rsync_files"] = len(subset)
    t, _ = timed(lambda: shutil.rmtree(dst))
    ops["loose_rm_s"] = round(t, 2)
    t, _ = timed(
        lambda: subprocess.run(
            ["bash", "-c", f"find {base / 'loose'} | wc -l"],
            capture_output=True,
            check=True,
        )
    )
    ops["loose_find_s"] = round(t, 3)

    dst = scratch / "copy-pack"
    if dst.exists():
        shutil.rmtree(dst)
    t, _ = timed(
        lambda: subprocess.run(
            ["rsync", "-a", str(base / "pack") + "/", str(dst) + "/"], check=True
        )
    )
    ops["pack_rsync_s"] = round(t, 2)
    t, _ = timed(lambda: shutil.rmtree(dst))
    ops["pack_rm_s"] = round(t, 2)
    t, _ = timed(
        lambda: subprocess.run(
            ["bash", "-c", f"find {base / 'pack'} | wc -l"],
            capture_output=True,
            check=True,
        )
    )
    ops["pack_find_s"] = round(t, 3)
    return ops


# ---------------------------------------------------------------- extras


def build_extras(base: Path):
    """Design-space artifacts for the coco variant (built once per fs)."""
    marker = base / ".extras"
    metrics = json.loads(marker.read_text()) if marker.exists() else {}
    loose = base / "loose"

    keys = json.loads((base / "keys.json").read_text())
    if "deflate_convert_s" not in metrics:
        t0 = time.perf_counter()
        with zipfile.ZipFile(base / "deflate.zip", "w", zipfile.ZIP_DEFLATED) as writer:
            for key in keys:
                writer.writestr(zipfile.ZipInfo(key), (loose / key).read_bytes())
        metrics["deflate_convert_s"] = round(time.perf_counter() - t0, 2)
        metrics["deflate_bytes"] = os.path.getsize(base / "deflate.zip")
        marker.write_text(json.dumps(metrics))

    if "tar_index_entries" not in metrics:
        index = {}
        with tarfile.open(base / "data.tar", "r:") as tar:
            for member in tar:
                if member.isfile():
                    index[member.name] = (member.offset_data, member.size)
        (base / "data.tar.index.json").write_text(json.dumps(index))
        metrics["tar_index_entries"] = len(index)
        metrics["tar_index_bytes"] = os.path.getsize(base / "data.tar.index.json")
        marker.write_text(json.dumps(metrics))

    if "parquet_convert_s" not in metrics:
        import pyarrow as pa
        import pyarrow.parquet as pq

        t0 = time.perf_counter()
        writer = None
        batch_keys, batch_data = [], []

        def flush():
            nonlocal writer, batch_keys, batch_data
            table = pa.table(
                {"key": batch_keys, "data": pa.array(batch_data, pa.binary())}
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    base / "embedded.parquet", table.schema, compression="none"
                )
            writer.write_table(table, row_group_size=1000)
            batch_keys, batch_data = [], []

        for key in keys:
            batch_keys.append(key)
            batch_data.append((loose / key).read_bytes())
            if len(batch_keys) == 1000:
                flush()
        if batch_keys:
            flush()
        writer.close()
        metrics["parquet_convert_s"] = round(time.perf_counter() - t0, 2)
        metrics["parquet_bytes"] = os.path.getsize(base / "embedded.parquet")
        marker.write_text(json.dumps(metrics))

    for target, name in ((64 << 20, "pack64m"), (2 << 30, "pack2g")):
        if f"{name}_shards" in metrics:
            continue
        pack = base / name
        if pack.exists():
            shutil.rmtree(pack)
        pack.mkdir()
        t0 = time.perf_counter()
        shard, size = 0, 0
        writer = zipfile.ZipFile(pack / "shard-0000.zip", "w", zipfile.ZIP_STORED)
        for key in keys:
            data = (loose / key).read_bytes()
            if size and size + len(data) > target:
                writer.close()
                shard += 1
                writer = zipfile.ZipFile(
                    pack / f"shard-{shard:04d}.zip", "w", zipfile.ZIP_STORED
                )
                size = 0
            writer.writestr(zipfile.ZipInfo(key), data)
            size += len(data)
        writer.close()
        metrics[f"{name}_convert_s"] = round(time.perf_counter() - t0, 2)
        metrics[f"{name}_shards"] = shard + 1
        marker.write_text(json.dumps(metrics))

    build_lance(base, keys, metrics, marker)
    return metrics


def build_lance(base: Path, keys, metrics, marker):
    """Lance baselines: inline large_binary column, and the blob storage class
    (Lance's intended path for large media). Streaming 1000-row batches."""
    import lance
    import pyarrow as pa

    loose = base / "loose"
    metrics["lance_version"] = lance.__version__
    try:
        from lance import blob_array, blob_field  # blob v2 API

        blob_v2 = True
    except ImportError:
        blob_v2 = False
        blob_field = blob_array = None
    metrics["lance_blob_api"] = "v2" if blob_v2 else "legacy-2.1"

    configs = [("lance_inline", "inline", "stable")]
    if blob_v2:
        configs.append(("lance_blob", "blob_v2", "2.2"))
    else:
        configs.append(("lance_blob", "blob_legacy", "2.1"))

    for name, mode, dsv in configs:
        if f"{name}_convert_s" in metrics or f"{name}_error" in metrics:
            continue
        uri = base / f"{name}.lance"
        if uri.exists():
            shutil.rmtree(uri)
        if mode == "inline":
            schema = pa.schema(
                [pa.field("key", pa.string()), pa.field("data", pa.large_binary())]
            )
        elif mode == "blob_v2":
            schema = pa.schema([pa.field("key", pa.string()), blob_field("data")])
        else:
            schema = pa.schema(
                [
                    pa.field("key", pa.string()),
                    pa.field(
                        "data",
                        pa.large_binary(),
                        metadata={"lance-encoding:blob": "true"},
                    ),
                ]
            )

        def batches():
            for i in range(0, len(keys), 1000):
                chunk = keys[i : i + 1000]
                payload = [(loose / k).read_bytes() for k in chunk]
                if mode == "blob_v2":
                    arr = blob_array(payload)
                else:
                    arr = pa.array(payload, pa.large_binary())
                yield pa.record_batch(
                    [pa.array(chunk, pa.string()), arr], schema=schema
                )

        try:
            t0 = time.perf_counter()
            lance.write_dataset(
                batches(),
                str(uri),
                schema=schema,
                data_storage_version=dsv,
                max_bytes_per_file=2 << 30,
            )
            metrics[f"{name}_convert_s"] = round(time.perf_counter() - t0, 2)
            metrics[f"{name}_du_bytes"] = du_bytes(uri)
            metrics[f"{name}_files"] = inode_count(uri)
            metrics[f"{name}_storage_version"] = dsv
        except Exception as exc:  # record, keep the bench alive
            metrics[f"{name}_error"] = repr(exc)[:300]
            if uri.exists():
                shutil.rmtree(uri)
        marker.write_text(json.dumps(metrics))


def take_blobs_compat(ds, ids):
    try:
        return ds.take_blobs("data", indices=ids)
    except TypeError:
        return ds.take_blobs(ids, "data")


def blob_len(bf):
    try:
        return len(bf.read())
    finally:
        try:
            bf.close()
        except Exception:
            pass


def seq_lance(uri: Path, limit, blob):
    import lance

    ds = lance.dataset(str(uri))
    total = 0
    if blob:
        n = min(limit, ds.count_rows())
        for start in range(0, n, 256):
            for bf in take_blobs_compat(ds, list(range(start, min(start + 256, n)))):
                total += blob_len(bf)
    else:
        seen = 0
        for batch in ds.scanner(
            columns=["data"], batch_size=256, limit=limit
        ).to_batches():
            col = batch.column(0)
            for j in range(len(col)):
                total += len(col[j].as_py())
            seen += len(col)
            if seen >= limit:
                break
    return total


def rand_lance_percall(uri: Path, order, blob):
    import lance

    ds = lance.dataset(str(uri))
    total = 0
    for i in order:
        if blob:
            total += blob_len(take_blobs_compat(ds, [i])[0])
        else:
            total += len(ds.take([i], columns=["data"]).column("data")[0].as_py())
    return len(order)


def rand_lance_batched(uri: Path, order, blob):
    import lance

    ds = lance.dataset(str(uri))
    if blob:
        return sum(blob_len(bf) for bf in take_blobs_compat(ds, list(order)))
    table = ds.take(list(order), columns=["data"])
    return sum(len(v.as_py()) for v in table.column("data"))


def pack_dir_index(pack: Path):
    index = {}
    for shard_path in sorted(pack.glob("shard-*.zip")):
        with zipfile.ZipFile(shard_path) as bundle:
            for name in bundle.namelist():
                index[name] = shard_path.name
    return index


def rand_tar_index(base: Path, keys, order, workers):
    index = json.loads((base / "data.tar.index.json").read_text())
    fd = os.open(base / "data.tar", os.O_RDONLY)
    try:

        def read(i):
            offset, size = index[keys[i]]
            return len(os.pread(fd, size, offset))

        with ThreadPoolExecutor(workers) as pool:
            return sum(pool.map(read, order, chunksize=64))
    finally:
        os.close(fd)


def seq_parquet(base: Path, limit):
    import pyarrow.parquet as pq

    total, rows = 0, 0
    handle = pq.ParquetFile(base / "embedded.parquet")
    for g in range(handle.num_row_groups):
        for buf in handle.read_row_group(g, columns=["data"]).column("data"):
            total += len(buf.as_py())
        rows += handle.metadata.row_group(g).num_rows
        if rows >= limit:
            break
    return total


def rand_parquet(base: Path, order):  # order indexes rows directly
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(base / "embedded.parquet")
    total = 0
    for i in order:
        group = handle.read_row_group(i // 1000, columns=["data"])
        total += len(group.column("data")[i % 1000].as_py())
    return len(order)


# ---------------------------------------------------------------- driver


def bench_fs(base: Path, scratch: Path, spec):
    result = {"build": build_variant(base, spec)}
    keys = json.loads((base / "keys.json").read_text())
    n = len(keys)
    avg = result["build"]["payload_bytes"] / n
    seq_n = min(n, int(SEQ_CAP / avg) + 1)
    index = shard_index(base)
    rng = random.Random(SEED)
    order = list(range(n))
    rng.shuffle(order)
    probes = order[: min(RAND_PROBES, n)]
    percall_probes = order[: min(PERCALL_PROBES, n)]
    decode_probes = order[: min(DECODE_N, n)]

    runs = {"seq_items": seq_n, "rand_items": len(probes)}
    # sequential over the same item prefix, cold then warm
    drop_loose(base, keys[:seq_n])
    t, b = timed(lambda: seq_loose(base, keys, seq_n))
    runs["seq_loose_cold_s"], runs["seq_bytes"] = round(t, 2), b
    t, _ = timed(lambda: seq_loose(base, keys, seq_n))
    runs["seq_loose_warm_s"] = round(t, 2)
    drop_cache(base / "pack")
    t, _ = timed(lambda: seq_pack_dir(base / "pack", seq_n))
    runs["seq_pack_cold_s"] = round(t, 2)
    t, _ = timed(lambda: seq_pack_dir(base / "pack", seq_n))
    runs["seq_pack_warm_s"] = round(t, 2)
    drop_cache(base / "data.tar")
    t, _ = timed(lambda: seq_tar(base, seq_n))
    runs["seq_tar_cold_s"] = round(t, 2)
    t, _ = timed(lambda: seq_tar(base, seq_n))
    runs["seq_tar_warm_s"] = round(t, 2)

    # random access, cold, 1 and 8 threads, same probe sample
    for workers in (1, 8):
        drop_loose(base, keys, probes)
        t, _ = timed(lambda: rand_loose(base, keys, probes, workers))
        runs[f"rand_loose_w{workers}_cold_s"] = round(t, 2)
        drop_cache(base / "pack")
        t, _ = timed(
            lambda: rand_pack_cached(base / "pack", keys, probes, index, workers)
        )
        runs[f"rand_pack_cached_w{workers}_cold_s"] = round(t, 2)
    drop_cache(base / "pack")
    t, n_done = timed(lambda: rand_pack_percall(base, keys, percall_probes, index))
    runs["rand_pack_percall_cold_s"] = round(t, 2)
    runs["rand_pack_percall_items"] = n_done

    # chunked-sequential epoch sample (pack shuffle) vs fully random loose
    drop_cache(base / "pack")
    t, b = timed(lambda: chunked_pack_dir(base / "pack", random.Random(SEED), SEQ_CAP))
    runs["chunked_pack_cold_s"], runs["chunked_pack_bytes"] = round(t, 2), b

    if spec.get("extras"):
        result["extras_build"] = build_extras(base)
        for name in ("pack64m", "pack2g"):
            pack = base / name
            pidx = pack_dir_index(pack)
            drop_cache(pack)
            t, _ = timed(lambda: rand_pack_cached(pack, keys, probes, pidx, 8))
            runs[f"rand_{name}_cached_w8_cold_s"] = round(t, 2)
            drop_cache(pack)
            t, b = timed(lambda: chunked_pack_dir(pack, random.Random(SEED), SEQ_CAP))
            runs[f"chunked_{name}_cold_s"] = round(t, 2)
        drop_cache(base / "data.tar")
        t, _ = timed(lambda: rand_tar_index(base, keys, probes, 8))
        runs["rand_tarindex_w8_cold_s"] = round(t, 2)
        drop_cache(base / "embedded.parquet")
        t, _ = timed(lambda: seq_parquet(base, seq_n))
        runs["seq_parquet_cold_s"] = round(t, 2)
        drop_cache(base / "embedded.parquet")
        t, n_pq = timed(lambda: rand_parquet(base, order[:PARQUET_PROBES]))
        runs["rand_parquet_cold_s"] = round(t, 2)
        runs["rand_parquet_items"] = n_pq

        for name in ("lance_inline", "lance_blob"):
            uri = base / f"{name}.lance"
            if f"{name}_error" in result["extras_build"] or not uri.exists():
                continue
            blob = name == "lance_blob"
            drop_cache(uri)
            t, _ = timed(lambda: seq_lance(uri, seq_n, blob))
            runs[f"seq_{name}_cold_s"] = round(t, 2)
            t, _ = timed(lambda: seq_lance(uri, seq_n, blob))
            runs[f"seq_{name}_warm_s"] = round(t, 2)
            drop_cache(uri)
            t, n_done = timed(lambda: rand_lance_percall(uri, percall_probes, blob))
            runs[f"rand_{name}_percall_cold_s"] = round(t, 2)
            runs[f"rand_{name}_percall_items"] = n_done
            drop_cache(uri)
            t, _ = timed(lambda: rand_lance_batched(uri, probes, blob))
            runs[f"rand_{name}_batched_cold_s"] = round(t, 2)

    # decode throughput, warm payloads, 8 threads
    warm_subset(base, keys, decode_probes, index)
    for source in ("loose", "pack"):
        t, n_dec = timed(
            lambda: decode_pass(base, keys, decode_probes, source, index, 8)
        )
        runs[f"decode_{source}_w8_warm_s"] = round(t, 2)
    runs["decode_items"] = len(decode_probes)

    result["runs"] = runs
    result["ops"] = folder_ops(base, scratch, keys)
    return result


def main():
    for variant, spec in VARIANTS.items():
        RESULTS["variants"][variant] = {}
        for fs_name, root in TARGETS.items():
            base = root / variant
            scratch = root / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)
            log(f"=== {variant} on {fs_name} ({root})")
            RESULTS["variants"][variant][fs_name] = bench_fs(base, scratch, spec)
            log(json.dumps(RESULTS["variants"][variant][fs_name], indent=1))
            OUT_JSON.write_text(json.dumps(RESULTS, indent=1))
    log(f"results -> {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
