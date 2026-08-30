#!/usr/bin/env python3
"""Appendix experiment: for temporally-redundant frame sequences, how does a
media-container blob (H.264 MP4, short GOP) compare with per-frame JPEG blobs?

Real data: LeRobot PushT episodes (robot-manipulation video, 96x96, ~25K
frames over ~200 episodes) — genuine temporal consistency. Measures:
  - storage: sum of per-frame JPEGs vs re-encoded MP4 (gop=30 and gop=300)
  - sequential decode of a whole episode (jpg loop vs container decode)
  - random single-frame access latency (jpg read+decode vs seek+decode)
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(os.environ.get("APPENDIX_OUT", "appendix_results.json"))
WORK = Path(os.environ.get("APPENDIX_WORK", "/tmp/blobpack-appendix"))
SOURCES = Path(os.environ.get("BENCH_SOURCES", "sources"))
SEED = 0
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def dump_frames(tmp: Path):
    """All frames of the single concatenated AV1 video, plus episode ranges."""
    import pyarrow.parquet as pq

    video = next(SOURCES.glob("pusht/videos/**/*.mp4"))
    subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(video), "-q:v", "3", f"{tmp}/%06d.jpg"],
        check=True,
    )
    table = pq.read_table(
        next(SOURCES.glob("pusht/meta/episodes/**/*.parquet")),
        columns=["episode_index", "dataset_from_index", "dataset_to_index"],
    )
    episodes = sorted(
        zip(
            *[
                table.column(c).to_pylist()
                for c in ("episode_index", "dataset_from_index", "dataset_to_index")
            ]
        )
    )
    return video, episodes


def main():
    if WORK.exists():
        subprocess.run(["rm", "-rf", str(WORK)], check=True)
    (WORK / "jpg").mkdir(parents=True)
    (WORK / "mp4").mkdir()
    dump = WORK / "dump"
    dump.mkdir()
    source_video, episodes = dump_frames(dump)
    clip_frames = {}
    results = {"clips": len(episodes)}

    jpg_bytes = 0
    t_mp4_30 = 0.0
    mp4_bytes = {30: 0, 300: 0}
    total_frames = 0
    for c, (episode, start, stop) in enumerate(episodes):
        clip_dir = WORK / "jpg" / f"clip{c:03d}"
        clip_dir.mkdir()
        for offset, dataset_index in enumerate(range(start, stop)):
            os.rename(
                dump / f"{dataset_index + 1:06d}.jpg", clip_dir / f"{offset:04d}.jpg"
            )
        clip_frames[c] = stop - start
        total_frames += stop - start
        jpg_bytes += sum(f.stat().st_size for f in clip_dir.iterdir())

        for gop in (30, 300):
            dst = WORK / "mp4" / f"clip{c:03d}_gop{gop}.mp4"
            t0 = time.perf_counter()
            subprocess.run(
                [
                    FFMPEG,
                    "-y",
                    "-v",
                    "error",
                    "-framerate",
                    "10",
                    "-i",
                    str(clip_dir / "%04d.jpg"),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    str(gop),
                    "-movflags",
                    "+faststart",
                    str(dst),
                ],
                check=True,
            )
            if gop == 30:
                t_mp4_30 += time.perf_counter() - t0
            mp4_bytes[gop] += dst.stat().st_size
    sample = cv2.imread(str(WORK / "jpg" / "clip000" / "0000.jpg"))
    results["frames_total"] = total_frames
    results["wh"] = [sample.shape[1], sample.shape[0]]
    results["source_av1_total_bytes"] = source_video.stat().st_size
    results["jpg_total_bytes"] = jpg_bytes
    results["mp4_gop30_total_bytes"] = mp4_bytes[30]
    results["mp4_gop300_total_bytes"] = mp4_bytes[300]
    results["encode_mp4_gop30_s"] = round(t_mp4_30, 1)

    # sequential whole-clip decode
    n_clips = len(episodes)
    t0 = time.perf_counter()
    for c in range(n_clips):
        for t in range(clip_frames[c]):
            data = np.fromfile(WORK / "jpg" / f"clip{c:03d}" / f"{t:04d}.jpg", np.uint8)
            assert cv2.imdecode(data, cv2.IMREAD_COLOR) is not None
    results["seq_decode_jpg_s"] = round(time.perf_counter() - t0, 2)

    for gop in (30, 300):
        t0 = time.perf_counter()
        for c in range(n_clips):
            cap = cv2.VideoCapture(str(WORK / "mp4" / f"clip{c:03d}_gop{gop}.mp4"))
            n = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            cap.release()
            assert n == clip_frames[c], (c, n, clip_frames[c])
        results[f"seq_decode_mp4_gop{gop}_s"] = round(time.perf_counter() - t0, 2)

    # random single-frame access (1000 probes)
    probe_rng = random.Random(SEED)
    probes = []
    while len(probes) < 1000:
        c = probe_rng.randrange(n_clips)
        probes.append((c, probe_rng.randrange(clip_frames[c])))
    t0 = time.perf_counter()
    for c, t in probes:
        data = np.fromfile(WORK / "jpg" / f"clip{c:03d}" / f"{t:04d}.jpg", np.uint8)
        assert cv2.imdecode(data, cv2.IMREAD_COLOR) is not None
    results["rand_frame_jpg_s_per_1000"] = round(time.perf_counter() - t0, 2)

    for gop in (30, 300):
        t0 = time.perf_counter()
        for c, t in probes:
            cap = cv2.VideoCapture(str(WORK / "mp4" / f"clip{c:03d}_gop{gop}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, _ = cap.read()
            assert ok
            cap.release()
        results[f"rand_frame_mp4_gop{gop}_s_per_1000"] = round(
            time.perf_counter() - t0, 2
        )

    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1), flush=True)
    subprocess.run(["rm", "-rf", str(WORK)], check=True)


if __name__ == "__main__":
    sys.exit(main())
