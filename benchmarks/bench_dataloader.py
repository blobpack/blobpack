#!/usr/bin/env python3
"""End-to-end training-loader benchmark: blobpack vs the alternatives.

File-read microbenchmarks do not settle whether a storage format helps a
training pipeline, so this measures what a training job actually feels:
time to first batch, steady-state samples/s through a torch DataLoader
that decodes every JPEG, peak RSS, open file descriptors, and how those
scale with worker count.

Formats: loose files, Blob Pack (direct-offset reader), WebDataset tar
shards, and Lance. Run it per filesystem (shared vs node-local) by
pointing --work-dir at each.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from _env import storage_class

import cv2
import numpy as np
import torch

# OpenCV's own thread pool does not survive fork, which crashes DataLoader
# workers; one decode thread per worker is also what a training job wants.
cv2.setNumThreads(0)
torch.set_num_threads(1)
from torch.utils.data import DataLoader, Dataset, IterableDataset

from blobpack import PackSet, PackWriter

SEED = 0
SHARD_TARGET = 512 << 20


def log(message: str) -> None:
    print(message, flush=True)


def drop_cache(root: Path) -> None:
    """Approximate a cold cache: evict every data file we are about to read."""
    for path in sorted(root.rglob("*")):
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


def open_fds() -> int:
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except OSError:
        return -1


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------- build

def build_loose(source_zip: Path, dst: Path, limit: int) -> list[str]:
    keys = []
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as bundle:
        names = [n for n in bundle.namelist() if n.endswith(".jpg")][:limit]
        for name in names:
            target = dst / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))
            keys.append(name)
    return keys


def build_pack(loose: Path, keys: list[str], dst: Path) -> None:
    with PackWriter(dst, ref_base=dst.name, max_pack_bytes=SHARD_TARGET) as writer:
        for key in keys:
            writer.add_file(key, loose / key)


def build_webdataset(loose: Path, keys: list[str], dst: Path) -> None:
    """Plain tar shards with WebDataset's basename convention."""
    import tarfile

    dst.mkdir(parents=True, exist_ok=True)
    shard_index, written = 0, 0
    tar = tarfile.open(dst / f"shard-{shard_index:04d}.tar", "w")
    try:
        for key in keys:
            data = (loose / key).read_bytes()
            if written and written + len(data) > SHARD_TARGET:
                tar.close()
                shard_index += 1
                tar = tarfile.open(dst / f"shard-{shard_index:04d}.tar", "w")
                written = 0
            info = tarfile.TarInfo(Path(key).stem + ".jpg")
            info.size = len(data)
            tar.addfile(info, __import__("io").BytesIO(data))
            written += len(data)
    finally:
        tar.close()


def build_lance(loose: Path, keys: list[str], dst: Path) -> None:
    import lance
    import pyarrow as pa

    schema = pa.schema([pa.field("key", pa.string()), pa.field("data", pa.large_binary())])

    def batches():
        for start in range(0, len(keys), 512):
            chunk = keys[start : start + 512]
            payload = [(loose / key).read_bytes() for key in chunk]
            yield pa.record_batch(
                [pa.array(chunk, pa.string()), pa.array(payload, pa.large_binary())], schema=schema
            )

    lance.write_dataset(batches(), str(dst), schema=schema, max_bytes_per_file=2 << 30)


# ---------------------------------------------------------------- datasets


def decode(payload: bytes) -> int:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("decode failed")
    return int(image.shape[0])


class LooseDataset(Dataset):
    def __init__(self, root: Path, keys: list[str]):
        self.root, self.keys = root, keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> int:
        return decode((self.root / self.keys[index]).read_bytes())


class PackDataset(Dataset):
    """Index-style random access, the pattern blobpack is built for."""

    def __init__(self, pack_dir: Path, keys: list[str], catalog: Path | None = None):
        self.pack_dir, self.keys, self.catalog = pack_dir, keys, catalog
        self._packs = None

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> int:
        if self._packs is None:  # opened per worker process
            self._packs = PackSet(self.pack_dir, catalog=self.catalog)
        return decode(self._packs.read(self.keys[index]))


class LanceDataset(Dataset):
    def __init__(self, uri: Path, count: int):
        self.uri, self.count = uri, count
        self._ds = None

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        if self._ds is None:
            import lance

            self._ds = lance.dataset(str(self.uri))
        table = self._ds.take([index], columns=["data"])
        return decode(table.column("data")[0].as_py())


class TarIterable(IterableDataset):
    """WebDataset-style streaming: shards split across workers, sequential
    within a shard (a tar cannot be opened at an offset)."""

    def __init__(self, shard_dir: Path):
        self.shards = sorted(shard_dir.glob("*.tar"))

    def __iter__(self):
        import tarfile

        info = torch.utils.data.get_worker_info()
        shards = self.shards
        if info is not None:
            shards = shards[info.id :: info.num_workers]
        for shard in shards:
            with tarfile.open(shard, "r|") as tar:
                for member in tar:
                    if member.isfile():
                        yield decode(tar.extractfile(member).read())


class PackIterable(IterableDataset):
    """The same streaming pattern on packs, with member-range worker splits
    so shard count does not have to exceed worker count."""

    def __init__(self, pack_dir: Path, catalog: Path | None = None):
        self.pack_dir, self.catalog = pack_dir, catalog

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        with PackSet(self.pack_dir, catalog=self.catalog) as packs:
            for _, payload in packs.iter_blobs(worker_id=worker_id, num_workers=num_workers):
                yield decode(payload)


# ---------------------------------------------------------------- measure


def _pass(loader, limit: int, batch_size: int) -> dict:
    start = time.perf_counter()
    first_batch = None
    seen = 0
    for batch in loader:
        if first_batch is None:
            first_batch = time.perf_counter() - start
        seen += len(batch)
        if seen >= limit:
            break
    elapsed = time.perf_counter() - start
    after_first = max(elapsed - (first_batch or 0.0), 1e-6)
    return {
        "samples": seen,
        "time_to_first_batch_s": round(first_batch or 0.0, 3),
        "elapsed_s": round(elapsed, 2),
        "samples_per_s": round(max(seen - batch_size, 0) / after_first, 1),
    }


def run_loader(dataset, workers: int, batch_size: int, limit: int) -> dict:
    """Startup and throughput as two separate numbers.

    Workers persist, as they do in a training job, so a format's opening
    cost is paid once and belongs in time_to_first_batch_s. Throughput is
    then measured on a second pass over the already-running loader, with no
    startup mixed into it.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=isinstance(dataset, Dataset) and not isinstance(dataset, IterableDataset),
        persistent_workers=bool(workers),
        prefetch_factor=4 if workers else None,
    )
    startup = _pass(loader, limit, batch_size)
    steady = _pass(loader, limit, batch_size)
    del loader
    return {
        "samples": steady["samples"],
        "time_to_first_batch_s": startup["time_to_first_batch_s"],
        "steady_samples_per_s": steady["samples_per_s"],
        "first_pass_samples_per_s": startup["samples_per_s"],
        "steady_time_to_first_batch_s": steady["time_to_first_batch_s"],
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "open_fds": open_fds(),
    }


def measure_in_subprocess(case: str, workers: int, args, timeout: int) -> dict:
    """Run one measurement in a fresh process.

    A DataLoader worker that segfaults or deadlocks takes its parent with
    it, and neither is catchable in-process; isolating each cell keeps one
    bad combination from costing the whole matrix.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        case,
        "--source-zip",
        str(args.source_zip),
        "--work-dir",
        str(args.work_dir),
        "--output",
        os.devnull,
        "--images",
        str(args.images),
        "--samples",
        str(args.samples),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(workers),
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s"}
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"error": f"exit {done.returncode}: {tail[0][:200]}"}
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("MEASUREMENT "):
            return json.loads(line[len("MEASUREMENT ") :])
    return {"error": "no measurement emitted"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True, help="COCO train2017.zip")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=40000)
    parser.add_argument("--samples", type=int, default=8000, help="samples per measurement")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--single", default=None, help="internal: run one case and print MEASUREMENT")
    parser.add_argument("--cell-timeout", type=int, default=900)
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    loose, pack, tars, lance_uri = work / "loose", work / "pack", work / "wds", work / "data.lance"

    results = {
        "config": {
            "images": args.images,
            "samples_per_measurement": args.samples,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "repeats": args.repeats,
            "storage": storage_class(work),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
        },
        "build": {},
        "runs": {},
    }

    if not (work / ".built").exists():
        log("building loose")
        t0 = time.perf_counter()
        keys = build_loose(args.source_zip, loose, args.images)
        results["build"]["loose_s"] = round(time.perf_counter() - t0, 1)
        (work / "keys.json").write_text(json.dumps(keys))
        for name, builder, target in (
            ("pack", build_pack, pack),
            ("wds", build_webdataset, tars),
            ("lance", build_lance, lance_uri),
        ):
            log(f"building {name}")
            t0 = time.perf_counter()
            builder(loose, keys, target)
            results["build"][f"{name}_s"] = round(time.perf_counter() - t0, 1)
        for name, target in (("loose", loose), ("pack", pack), ("wds", tars), ("lance", lance_uri)):
            usage = subprocess.run(["du", "-sB1", str(target)], capture_output=True, text=True)
            results["build"][f"{name}_bytes"] = int(usage.stdout.split()[0])
            count = subprocess.run(["bash", "-c", f"find {target} -type f | wc -l"], capture_output=True, text=True)
            results["build"][f"{name}_files"] = int(count.stdout.strip())
        (work / ".built").write_text(json.dumps(results["build"]))
    else:
        results["build"] = json.loads((work / ".built").read_text())
        keys = json.loads((work / "keys.json").read_text())
    log(json.dumps(results["build"], indent=1))

    cases = {
        "loose_random": lambda: LooseDataset(loose, keys),
        "pack_random": lambda: PackDataset(pack, keys),
        "pack_random_catalog": lambda: PackDataset(pack, keys, catalog=work / "catalog.sqlite"),
        "lance_random": lambda: LanceDataset(lance_uri, len(keys)),
        "wds_stream": lambda: TarIterable(tars),
        "pack_stream": lambda: PackIterable(pack),
        "pack_stream_catalog": lambda: PackIterable(pack, catalog=work / "catalog.sqlite"),
    }
    roots = {
        "loose_random": loose,
        "pack_random": pack,
        "pack_random_catalog": pack,
        "lance_random": lance_uri,
        "wds_stream": tars,
        "pack_stream": pack,
        "pack_stream_catalog": pack,
    }

    if args.single:
        case = args.single
        workers = args.workers[0]
        drop_cache(roots[case])
        measured = run_loader(cases[case](), workers, args.batch_size, args.samples)
        print("MEASUREMENT " + json.dumps(measured), flush=True)
        return 0

    for case in cases:
        results["runs"][case] = {}
        for workers in args.workers:
            samples = []
            for repeat in range(args.repeats):
                measured = measure_in_subprocess(case, workers, args, args.cell_timeout)
                log(f"{case} w{workers} run{repeat}: {measured}")
                if "error" in measured:
                    results["runs"][case][f"w{workers}"] = measured
                    args.output.write_text(json.dumps(results, indent=1))
                    samples = []
                    break
                samples.append(measured)
            if not samples:
                continue
            results["runs"][case][f"w{workers}"] = {
                "steady_samples_per_s_median": round(
                    statistics.median(m["steady_samples_per_s"] for m in samples), 1
                ),
                "first_pass_samples_per_s_median": round(
                    statistics.median(m["first_pass_samples_per_s"] for m in samples), 1
                ),
                "time_to_first_batch_s_median": round(
                    statistics.median(m["time_to_first_batch_s"] for m in samples), 3
                ),
                "peak_rss_mb_max": max(m["peak_rss_mb"] for m in samples),
                "open_fds_max": max(m["open_fds"] for m in samples),
                "runs": samples,
            }
            args.output.write_text(json.dumps(results, indent=1))
    args.output.write_text(json.dumps(results, indent=1))
    log(f"results -> {args.output}")
    if os.environ.get("BENCH_CLEANUP") == "1":
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
