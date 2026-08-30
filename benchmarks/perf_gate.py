"""Pre-release performance gate: run ``make perf`` before pushing a tag.

Budgets are deliberately loose -- an order of magnitude above healthy
measurements -- so the gate is quiet on slow machines but fails hard on
cost-class regressions (issue #11 shipped because nothing measured the
I/O cost of the remote paths). The remote half injects a fixed latency
per filesystem call and meters transferred bytes, so its budgets are
request-count budgets in disguise and independent of machine speed.
"""

from __future__ import annotations

import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

from blobpack import PackSet, PackWriter

MEMBERS = 5_000
BLOB_BYTES = 4_096
DELAY_S = 0.01  # injected per remote filesystem call

results: list[tuple[bool, str]] = []


def gate(name: str, measured: float, budget: float, unit: str, *, at_least: bool = False) -> None:
    ok = measured >= budget if at_least else measured <= budget
    bound = "≥" if at_least else "≤"
    results.append((ok, f"{'PASS' if ok else 'FAIL'}  {name}: {measured:,.3f} {unit} (budget {bound} {budget:,g})"))


def build_pack(root: Path) -> list[str]:
    with PackWriter(root / "media", ref_base="media") as writer:
        return [writer.add(f"blob/{i:05d}.bin", bytes([i % 251]) * BLOB_BYTES) for i in range(MEMBERS)]


def gate_local(root: Path) -> None:
    start = time.perf_counter()
    with PackSet(root / "media") as packs:
        gate("local: open (eager, 5k members)", time.perf_counter() - start, 10.0, "s")

        keys = random.Random(0).sample([f"blob/{i:05d}.bin" for i in range(MEMBERS)], 1_000)
        laps = []
        for key in keys:
            t = time.perf_counter()
            packs.read(key)
            laps.append(time.perf_counter() - t)
        gate("local: random read p50", statistics.median(laps) * 1e3, 5.0, "ms")

        start = time.perf_counter()
        total = sum(len(data) for _, data in packs.iter_blobs())
        gate("local: sequential throughput", total / (time.perf_counter() - start) / 1e6, 20.0, "MB/s", at_least=True)


class MeteredFS:
    """Delegating fsspec wrapper: fixed latency per call, bytes metered."""

    def __init__(self, inner):
        self._inner = inner
        self.bytes_moved = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def info(self, *args, **kwargs):
        time.sleep(DELAY_S)
        return self._inner.info(*args, **kwargs)

    def cat_file(self, path, start=None, end=None, **kwargs):
        time.sleep(DELAY_S)
        data = self._inner.cat_file(path, start=start, end=end, **kwargs)
        self.bytes_moved += len(data)
        return data

    def open(self, path, *args, **kwargs):
        time.sleep(DELAY_S)
        handle = self._inner.open(path, *args, **kwargs)
        fs = self

        class MeteredFile:
            def __getattr__(self, name):
                return getattr(handle, name)

            def read(self, *a, **kw):
                data = handle.read(*a, **kw)
                fs.bytes_moved += len(data)
                return data

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                handle.close()

        return MeteredFile()


def gate_remote(root: Path, refs: list[str]) -> None:
    import fsspec

    fs = fsspec.filesystem("memory")
    shard = root / "media" / "pack-0000.zip"
    with fs.open("/perfgate/media/pack-0000.zip", "wb") as handle:
        handle.write(shard.read_bytes())
    try:
        metered = MeteredFS(fs)
        start = time.perf_counter()
        with PackSet.from_fs(metered, "/perfgate/media") as packs:
            packs.read(refs[MEMBERS // 2])
            gate(f"remote: open + first read ({DELAY_S * 1e3:g} ms/call injected)", time.perf_counter() - start, 1.5, "s")
            # ~64 KB EOCD window + this shard's ~300 KB directory; a whole-
            # shard sweep (20 MB) breaches this by 40x
            gate("remote: open + first read transfer", metered.bytes_moved / 1e3, 512.0, "KB")

            start = time.perf_counter()
            packs.read(refs[MEMBERS // 2])
            gate("remote: repeat read", time.perf_counter() - start, 0.5, "s")
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        refs = build_pack(root)
        gate_local(root)
        gate_remote(root, refs)
    for _, line in results:
        print(line)
    if all(ok for ok, _ in results):
        print("perf gate: all budgets met")
        return 0
    print("perf gate: BUDGET EXCEEDED -- do not tag a release from this state")
    return 1


if __name__ == "__main__":
    sys.exit(main())
