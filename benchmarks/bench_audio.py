#!/usr/bin/env python3
"""Small-blob storage and read costs on real speech audio.

Speech corpora are the extreme small-file regime: utterance-level clips in
the tens to hundreds of kilobytes, millions of them. This measures what a
corpus costs as loose files, as a Blob Pack, and as uncompressed tar
shards: bytes on disk, inodes, a full sequential pass, random access, and
relocation, plus a decode pass so the numbers sit next to real work.

Point --src at a directory of audio files (e.g. LibriSpeech dev-clean).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from blobpack import PackSet, PackWriter

SEED = 0


def log(message: str) -> None:
    print(message, flush=True)


def evict(root: Path) -> None:
    # fadvise(DONTNEED) is a no-op on dirty pages, so files this process
    # just wrote survive eviction and the next pass reads page cache at
    # memory speed while claiming to be cold. Flush first.
    os.sync()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def du_bytes(path: Path) -> int:
    out = subprocess.run(["du", "-sB1", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


def inode_count(path: Path) -> int:
    out = subprocess.run(["bash", "-c", f"find {path} | wc -l"], capture_output=True, text=True)
    return int(out.stdout.strip())


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def build(src: Path, work: Path, extension: str) -> tuple[list[str], dict]:
    """Copy the corpus into loose/pack/tar layouts under one work dir."""
    files = sorted(p for p in src.rglob(f"*{extension}") if p.is_file())
    if not files:
        raise SystemExit(f"no *{extension} under {src}")
    keys = [str(p.relative_to(src).as_posix()) for p in files]
    loose, pack, tars = work / "loose", work / "pack", work / "tar"

    metrics = {"clips": len(keys)}
    start = time.perf_counter()
    for path, key in zip(files, keys):
        target = loose / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    metrics["loose_build_s"] = round(time.perf_counter() - start, 1)

    start = time.perf_counter()
    with PackWriter(pack, ref_base="pack") as writer:
        for key in keys:
            writer.add_file(key, loose / key)
    metrics["pack_build_s"] = round(time.perf_counter() - start, 1)

    tars.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with tarfile.open(tars / "data.tar", "w") as tar:
        for key in keys:
            tar.add(loose / key, arcname=key)
    metrics["tar_build_s"] = round(time.perf_counter() - start, 1)

    metrics["payload_bytes"] = sum((loose / key).stat().st_size for key in keys)
    metrics["mean_clip_bytes"] = round(metrics["payload_bytes"] / len(keys))
    for name, path in (("loose", loose), ("pack", pack), ("tar", tars)):
        metrics[f"{name}_bytes"] = du_bytes(path)
        metrics[f"{name}_inodes"] = inode_count(path)
    return keys, metrics


def sequential(work: Path, keys: list[str]) -> dict:
    loose, pack, tars = work / "loose", work / "pack", work / "tar"
    out = {}
    evict(loose)
    out["seq_loose_s"] = round(timed(lambda: sum(len((loose / k).read_bytes()) for k in keys))[0], 2)
    evict(pack)

    def read_pack():
        with PackSet(pack) as packs:
            return sum(len(payload) for _, payload in packs.iter_blobs())

    out["seq_pack_s"] = round(timed(read_pack)[0], 2)
    evict(tars)

    def read_tar():
        total = 0
        with tarfile.open(tars / "data.tar", "r|") as tar:
            for member in tar:
                if member.isfile():
                    total += len(tar.extractfile(member).read())
        return total

    out["seq_tar_s"] = round(timed(read_tar)[0], 2)
    return out


def random_access(work: Path, keys: list[str], probes: int, workers: int) -> dict:
    loose, pack = work / "loose", work / "pack"
    order = list(keys)
    random.Random(SEED).shuffle(order)
    order = order[:probes]
    out = {"probes": len(order), "workers": workers}

    evict(loose)

    def read_loose():
        with ThreadPoolExecutor(workers) as pool:
            return sum(pool.map(lambda k: len((loose / k).read_bytes()), order))

    out["rand_loose_ms_per_clip"] = round(timed(read_loose)[0] / len(order) * 1000, 3)

    evict(pack)

    def read_pack():
        with PackSet(pack) as packs:
            return sum(len(payload) for payload in packs.read_many(order, workers=workers))

    out["rand_pack_ms_per_clip"] = round(timed(read_pack)[0] / len(order) * 1000, 3)
    return out


def relocation(work: Path, scratch: Path) -> dict:
    out = {}
    for name in ("loose", "pack"):
        source, destination = work / name, scratch / f"copy-{name}"
        shutil.rmtree(destination, ignore_errors=True)
        elapsed, _ = timed(
            lambda: subprocess.run(["rsync", "-a", f"{source}/", f"{destination}/"], check=True)
        )
        out[f"{name}_rsync_s"] = round(elapsed, 2)
        elapsed, _ = timed(lambda: shutil.rmtree(destination))
        out[f"{name}_rm_s"] = round(elapsed, 2)
    return out


def decode_pass(work: Path, keys: list[str], limit: int) -> dict:
    """Decode a sample of clips from each layout, so read wins are not
    mistaken for end-to-end wins when decoding dominates."""
    try:
        import soundfile
    except ImportError:
        return {"skipped": "soundfile not installed"}
    import io

    loose, pack = work / "loose", work / "pack"
    subset = keys[:limit]
    out = {"decoded": len(subset)}

    def decode(payload: bytes) -> int:
        data, _ = soundfile.read(io.BytesIO(payload))
        return len(data)

    evict(loose)
    out["decode_loose_s"] = round(timed(lambda: sum(decode((loose / k).read_bytes()) for k in subset))[0], 2)
    evict(pack)

    def decode_pack():
        with PackSet(pack) as packs:
            return sum(decode(packs.read(k)) for k in subset)

    out["decode_pack_s"] = round(timed(decode_pack)[0], 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="directory of audio files")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", default=".flac")
    parser.add_argument("--probes", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--decode", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    work = args.work_dir
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    scratch = work / "scratch"
    scratch.mkdir()

    keys, storage = build(args.src, work, args.extension)
    log(json.dumps(storage, indent=1))

    results = {"storage": storage, "reads": {}}
    seq_runs = [sequential(work, keys) for _ in range(args.repeats)]
    # per-pass values too: fadvise drops page cache but not dentry/inode
    # caches, so pass 1 is the only fully cold pass and the only one
    # comparable with bench.py's single-pass numbers
    results["reads"]["sequential_passes_s"] = seq_runs
    results["reads"]["sequential_cold_s"] = seq_runs[0]
    results["reads"]["sequential_median_s"] = {
        key: round(statistics.median(run[key] for run in seq_runs), 2) for key in seq_runs[0]
    }
    log(json.dumps(results["reads"]["sequential_median_s"], indent=1))

    rand_runs = [random_access(work, keys, args.probes, args.workers) for _ in range(args.repeats)]
    results["reads"]["random_median"] = {
        key: round(statistics.median(run[key] for run in rand_runs), 3)
        for key in rand_runs[0]
        if key.endswith("per_clip")
    }
    results["reads"]["random_median"].update({"probes": rand_runs[0]["probes"], "workers": args.workers})
    log(json.dumps(results["reads"]["random_median"], indent=1))

    results["decode"] = decode_pass(work, keys, args.decode)
    results["relocation"] = relocation(work, scratch)
    log(json.dumps({"decode": results["decode"], "relocation": results["relocation"]}, indent=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    log(f"results -> {args.output}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
