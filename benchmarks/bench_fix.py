#!/usr/bin/env python3
"""Follow-up measurements addressing review findings on the main benchmark.

1. STORED vs DEFLATE, corrected: the main run passed ZipInfo(key) to
   writestr, whose compress_type default (STORED) overrides the ZipFile
   compression — so its "deflate.zip" was never compressed. Rebuild both
   from the same 20K-image subset with compress_type set explicitly.
2. Zip pread fast path, measured (not inferred): resolve STORED members'
   data offsets from local headers once (untimed), then time raw
   os.pread() against the unchanged pack shards, 1 and 8 threads. No CRC.
3. Central-directory parse cost vs member count: warm ZipFile-open time on
   a ~3.3K-member coco shard vs the 25,650-member pusht shard.
4. Lance blob batched read with a concurrency-matched consumer: the main
   run consumed take_blobs() handles serially; also read them with 8
   threads.
Runs on the existing cephfs artifacts of the main benchmark.
"""

import json
import os
import random
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH = Path(os.environ.get("BENCH_DIR", "."))
COCO = BENCH / "data" / "coco_images"
PUSHT = BENCH / "data" / "pusht_frames"
OUT = BENCH / "fix_results.json"
SEED = 0
results = {}


def fadvise(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return time.perf_counter() - t0, value


keys = json.loads((COCO / "keys.json").read_text())
n = len(keys)
order = list(range(n))
random.Random(SEED).shuffle(order)
probes = order[:4000]

# 1. corrected STORED vs DEFLATE on the same 20K-image subset
subset = keys[:20000]
scratch = BENCH / "data" / "scratch"
scratch.mkdir(exist_ok=True)
for label, method in (
    ("stored", zipfile.ZIP_STORED),
    ("deflate", zipfile.ZIP_DEFLATED),
):
    dst = scratch / f"fix_{label}.zip"
    if dst.exists():
        dst.unlink()
    t0 = time.perf_counter()
    with zipfile.ZipFile(dst, "w") as writer:
        for key in subset:
            info = zipfile.ZipInfo(key)
            info.compress_type = method
            writer.writestr(info, (COCO / "loose" / key).read_bytes())
    results[f"zip_{label}_convert_s"] = round(time.perf_counter() - t0, 2)
    results[f"zip_{label}_bytes"] = dst.stat().st_size
    with zipfile.ZipFile(dst) as check:  # confirm the method actually applied
        results[f"zip_{label}_member_compress_type"] = check.infolist()[0].compress_type
    dst.unlink()
results["zip_subset_items"] = len(subset)
results["zip_subset_payload_bytes"] = sum(
    (COCO / "loose" / k).stat().st_size for k in subset
)
print(
    "deflate check:",
    {k: v for k, v in results.items() if k.startswith("zip")},
    flush=True,
)

# 2. pread fast path against the unchanged pack shards
index = {}
for shard_path in sorted((COCO / "pack").glob("shard-*.zip")):
    with zipfile.ZipFile(shard_path) as bundle:
        for zinfo in bundle.infolist():
            index[zinfo.filename] = (shard_path, zinfo.header_offset, zinfo.file_size)

fds = {}
resolved = []  # (fd, data_offset, size) per probe — resolved untimed, like a cached index
for i in probes:
    shard_path, header_offset, size = index[keys[i]]
    fd = fds.get(shard_path)
    if fd is None:
        fd = fds[shard_path] = os.open(shard_path, os.O_RDONLY)
    header = os.pread(fd, 30, header_offset)
    nlen, elen = struct.unpack("<HH", header[26:30])
    resolved.append((fd, header_offset + 30 + nlen + elen, size))

expected = sum(size for _, _, size in resolved)
for workers in (1, 8):
    for path in (COCO / "pack").glob("shard-*.zip"):
        fadvise(path)

    def read(job):
        fd, offset, size = job
        return len(os.pread(fd, size, offset))

    if workers == 1:
        t, total = timed(lambda: sum(read(j) for j in resolved))
    else:

        def run():
            with ThreadPoolExecutor(workers) as pool:
                return sum(pool.map(read, resolved, chunksize=64))

        t, total = timed(run)
    assert total == expected, (total, expected)
    results[f"pack_pread_w{workers}_cold_s"] = round(t, 2)
for fd in fds.values():
    os.close(fd)
results["pack_pread_items"] = len(probes)
print("pread:", {k: v for k, v in results.items() if "pread" in k}, flush=True)

# 3. central-directory parse cost vs member count (warm)
for label, shard in (
    ("coco_3k", COCO / "pack" / "shard-0000.zip"),
    ("pusht_25k", PUSHT / "pack" / "shard-0000.zip"),
):
    with zipfile.ZipFile(shard) as bundle:  # warm the tail blocks
        members = len(bundle.infolist())
    t0 = time.perf_counter()
    for _ in range(20):
        zipfile.ZipFile(shard).close()
    results[f"cd_open_{label}_warm_ms"] = round(
        (time.perf_counter() - t0) / 20 * 1000, 2
    )
    results[f"cd_open_{label}_members"] = members
print("cd:", {k: v for k, v in results.items() if k.startswith("cd_")}, flush=True)

# 4. lance blob batched: serial vs 8-thread consumer, cold
import lance

ds = lance.dataset(str(COCO / "lance_blob.lance"))


def take_blobs_compat(ids):
    try:
        return ds.take_blobs("data", indices=ids)
    except TypeError:
        return ds.take_blobs(ids, "data")


def read_blob(bf):
    try:
        return len(bf.read())
    finally:
        try:
            bf.close()
        except Exception:
            pass


def drop_lance():
    for dirpath, _, files in os.walk(COCO / "lance_blob.lance"):
        for fn in files:
            try:
                fadvise(os.path.join(dirpath, fn))
            except OSError:
                pass


drop_lance()
t, _ = timed(lambda: sum(read_blob(bf) for bf in take_blobs_compat(list(probes))))
results["lance_blob_batched_serial_cold_s"] = round(t, 2)
drop_lance()


def run_threaded():
    handles = take_blobs_compat(list(probes))
    with ThreadPoolExecutor(8) as pool:
        return sum(pool.map(read_blob, handles, chunksize=16))


t, _ = timed(run_threaded)
results["lance_blob_batched_w8_cold_s"] = round(t, 2)
results["lance_blob_items"] = len(probes)

OUT.write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1), flush=True)
