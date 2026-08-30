#!/usr/bin/env python3
"""Measure Blob Pack random reads through zipfile and raw pread on NVMe."""

import argparse
import json
import os
import platform
import random
import statistics
import struct
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _env import storage_class

SEED = 0
PROBES = 4000
REPEATS = 5
SHARD_TARGET = 512 << 20
EXPECTED_ITEMS = 118_287
EXPECTED_PAYLOAD_BYTES = 19_314_466_396


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def drop_cache(pack):
    for path in pack.glob("shard-*.zip"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def timed(fn):
    start = time.perf_counter()
    total = fn()
    return time.perf_counter() - start, total


def build_pack(source_zip, pack):
    pack.mkdir(parents=True)
    started = time.perf_counter()
    keys = []
    payload_bytes = 0
    shard = 0
    shard_bytes = 0
    writer = zipfile.ZipFile(pack / "shard-0000.zip", "w", zipfile.ZIP_STORED)
    try:
        with zipfile.ZipFile(source_zip) as source:
            for name in source.namelist():
                if not name.endswith(".jpg"):
                    continue
                data = source.read(name)
                if shard_bytes and shard_bytes + len(data) > SHARD_TARGET:
                    writer.close()
                    shard += 1
                    shard_bytes = 0
                    writer = zipfile.ZipFile(
                        pack / f"shard-{shard:04d}.zip", "w", zipfile.ZIP_STORED
                    )
                writer.writestr(zipfile.ZipInfo(name), data)
                keys.append(name)
                payload_bytes += len(data)
                shard_bytes += len(data)
    finally:
        writer.close()

    assert len(keys) == EXPECTED_ITEMS, len(keys)
    assert payload_bytes == EXPECTED_PAYLOAD_BYTES, payload_bytes
    assert shard + 1 == 36, shard + 1
    return keys, payload_bytes, shard + 1, time.perf_counter() - started


def build_indexes(pack):
    zip_index = {}
    pread_index = {}
    fds = {}
    for shard_path in sorted(pack.glob("shard-*.zip")):
        fd = fds[shard_path.name] = os.open(shard_path, os.O_RDONLY)
        with zipfile.ZipFile(shard_path) as bundle:
            for info in bundle.infolist():
                assert info.compress_type == zipfile.ZIP_STORED
                header = os.pread(fd, 30, info.header_offset)
                assert header[:4] == b"PK\x03\x04"
                name_len, extra_len = struct.unpack("<HH", header[26:30])
                data_offset = info.header_offset + 30 + name_len + extra_len
                zip_index[info.filename] = shard_path.name
                pread_index[info.filename] = (fd, data_offset, info.file_size)
    return zip_index, pread_index, fds


def read_zipfile(pack, keys, probes, index, workers):
    local = threading.local()

    def read(i):
        handles = getattr(local, "handles", None)
        if handles is None:
            handles = local.handles = {}
        name = keys[i]
        shard = index[name]
        if shard not in handles:
            handles[shard] = zipfile.ZipFile(pack / shard)
        return len(handles[shard].read(name))

    if workers == 1:
        return sum(read(i) for i in probes)
    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(read, probes, chunksize=64))


def read_pread(keys, probes, index, workers):
    def read(i):
        fd, offset, size = index[keys[i]]
        return len(os.pread(fd, size, offset))

    if workers == 1:
        return sum(read(i) for i in probes)
    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(read, probes, chunksize=64))


def summarize(samples):
    median_s = statistics.median(samples)
    return {
        "seconds": [round(value, 4) for value in samples],
        "median_s": round(median_s, 4),
        "median_ms_per_item": round(median_s * 1000 / PROBES, 4),
    }


def main():
    args = parse_args()
    pack = args.work_dir / "pack"
    keys, payload_bytes, shards, build_seconds = build_pack(args.source_zip, pack)
    zip_index, pread_index, fds = build_indexes(pack)
    order = list(range(len(keys)))
    random.Random(SEED).shuffle(order)
    probes = order[:PROBES]
    expected = sum(pread_index[keys[i]][2] for i in probes)

    samples = {
        f"{method}_w{workers}": []
        for method in ("zipfile", "pread")
        for workers in (1, 8)
    }
    try:
        for _ in range(REPEATS):
            for method, workers in (
                ("zipfile", 1),
                ("pread", 1),
                ("zipfile", 8),
                ("pread", 8),
            ):
                drop_cache(pack)
                if method == "zipfile":
                    elapsed, total = timed(
                        lambda: read_zipfile(pack, keys, probes, zip_index, workers)
                    )
                else:
                    elapsed, total = timed(
                        lambda: read_pread(keys, probes, pread_index, workers)
                    )
                assert total == expected, (method, workers, total, expected)
                samples[f"{method}_w{workers}"].append(elapsed)
                print(method, workers, round(elapsed, 4), flush=True)
    finally:
        for fd in fds.values():
            os.close(fd)

    results = {
        "environment": {
            "python": platform.python_version(),
            "storage": storage_class(args.work_dir),
        },
        "params": {
            "seed": SEED,
            "probes": PROBES,
            "repeats": REPEATS,
            "workers": [1, 8],
            "cold_method": "posix_fadvise(POSIX_FADV_DONTNEED) before each pass",
            "pread_index_build": "untimed",
        },
        "dataset": {
            "items": len(keys),
            "payload_bytes": payload_bytes,
            "pack_shards": shards,
            "pack_build_s": round(build_seconds, 2),
            "probe_bytes": expected,
        },
        "runs": {name: summarize(values) for name, values in samples.items()},
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
