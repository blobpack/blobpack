"""Splitting concatenated episode video, when it can be done losslessly.

A cut is a stream copy exactly when the episode boundary lands on a
keyframe, and a re-encode otherwise -- so boundaries are probed and
reported, and nothing here decides to re-encode on its own.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: a boundary this close to a keyframe counts as landing on it
ALIGNMENT_TOLERANCE_S = 0.001


def ffmpeg_available() -> bool:
    return bool(_tool("ffmpeg")) and bool(_tool("ffprobe"))


def _tool(name: str) -> str | None:
    override = os.environ.get(name.upper())
    if override:
        return override
    return shutil.which(name)


def keyframe_times(path) -> list[float]:
    """Presentation times of the video stream's keyframes, in seconds."""
    ffprobe = _tool("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is not available")
    done = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    frames = json.loads(done.stdout).get("frames", [])
    times = [float(f["pts_time"]) for f in frames if f.get("pts_time") not in (None, "N/A")]
    return sorted(times)


@dataclass
class SplitCheck:
    """Which episode boundaries in one file can be cut without re-encoding."""

    video_path: str
    aligned: list[int]
    unaligned: list[int]
    #: the keyframe each aligned boundary matched; cutting at the metadata
    #: timestamp instead would skip a hair-past keyframe to the next one
    keyframes: dict[int, float] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.aligned) + len(self.unaligned)


def check_alignment(path, boundaries: dict[int, float], tolerance: float = ALIGNMENT_TOLERANCE_S) -> SplitCheck:
    """Compare each episode's start against the file's keyframe times."""
    times = keyframe_times(path)
    aligned, unaligned, matched = [], [], {}
    for episode, start in sorted(boundaries.items(), key=lambda kv: kv[1]):
        nearest = min(times, key=lambda t: abs(start - t), default=None)
        if nearest is not None and abs(start - nearest) <= tolerance:
            aligned.append(episode)
            matched[episode] = nearest
        else:
            unaligned.append(episode)
    return SplitCheck(video_path=str(path), aligned=aligned, unaligned=unaligned, keyframes=matched)


def split_at(source, out_dir, times: list[float], *, prefix: str = "part") -> list:
    """Cut a file at the given seconds, one output per span, in order.

    The segment muxer yields exact spans with timestamps reset to zero
    (``-ss``/``-t`` seeking carries trailing frames), always as a stream
    copy -- which is why it can only cut on keyframes. ``times`` must be
    non-empty: with none, the muxer silently cuts every two seconds.
    """
    ffmpeg = _tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available")
    if not times:
        raise ValueError("split_at needs at least one cut time; copy the file instead")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{prefix}%05d{Path(source).suffix or '.mp4'}")
    command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source), "-c", "copy", "-f", "segment"]
    command += ["-segment_times", ",".join(f"{t:.6f}" for t in times)]
    command += ["-reset_timestamps", "1", pattern]
    subprocess.run(command, check=True, capture_output=True)
    return sorted(out_dir.glob(f"{prefix}*"))
