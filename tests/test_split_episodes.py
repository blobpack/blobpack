"""Cutting concatenated episode video into one container per episode.

These need real video, so they generate it with ffmpeg and skip cleanly
when it is missing.
"""

import json
import subprocess

import pytest

from blobpack import PackSet
from blobpack._video import check_alignment, ffmpeg_available, keyframe_times, split_at
from blobpack.lerobot import INDEX_NAME, convert, plan_conversion

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="needs ffmpeg and ffprobe")

FPS = 10
GOP = 20  # a keyframe every 2 seconds
CAM = "observation.image"


def make_video(path, seconds=6, gop=GOP):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=160x120:rate={FPS}:duration={seconds}",
            "-c:v",
            "libx264",
            "-g",
            str(gop),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def probe(path):
    out = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split(",")
    )
    return float(out[0]), int(out[1])


def build(root, starts=(0.0, 2.0, 4.0), seconds=6, cameras=(CAM,), ends=None, gop=GOP):
    (root / "meta" / "episodes").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": FPS,
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {**{c: {"dtype": "video"} for c in cameras}, "action": {"dtype": "float32"}},
                "total_episodes": len(starts),
                "total_frames": len(starts) * FPS * 2,
            }
        )
    )
    ends = list(ends) if ends is not None else [*starts[1:], float(seconds)]
    lengths = [round((e - s) * FPS) for s, e in zip(starts, ends)]
    columns = {
        "episode_index": list(range(len(starts))),
        "length": lengths,
        "dataset_from_index": [0] * len(starts),
        "dataset_to_index": lengths,
    }
    for camera in cameras:
        columns[f"videos/{camera}/chunk_index"] = [0] * len(starts)
        columns[f"videos/{camera}/file_index"] = [0] * len(starts)
        columns[f"videos/{camera}/from_timestamp"] = list(starts)
        columns[f"videos/{camera}/to_timestamp"] = ends
        make_video(root / "videos" / camera / "chunk-000" / "file-000.mp4", seconds, gop=gop)
    pq.write_table(pa.table(columns), root / "meta" / "episodes" / "file-000.parquet")


def test_keyframes_and_alignment_are_detected(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    assert keyframe_times(video) == [0.0, 2.0, 4.0]
    check = check_alignment(video, {0: 0.0, 1: 2.0, 2: 3.0})
    assert check.aligned == [0, 1] and check.unaligned == [2]


def test_segment_cuts_are_exact_on_keyframes(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    parts = split_at(video, tmp_path / "parts", [2.0, 4.0])
    assert len(parts) == 3
    for part in parts:
        assert probe(part) == (2.0, FPS * 2)


def test_unaligned_cut_drifts_which_is_why_it_is_refused(tmp_path):
    """The muxer can only cut on keyframes; asking otherwise moves the
    boundary silently, so the planner must never ask."""
    video = tmp_path / "v.mp4"
    make_video(video)
    parts = split_at(video, tmp_path / "parts", [3.0])
    assert probe(parts[0])[0] == 4.0  # snapped forward from 3.0


def test_aligned_dataset_is_split_into_one_container_per_episode(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src)
    plan = plan_conversion(src)
    assert "3/3 episode boundaries" in plan.render()
    assert "split into one container per episode" in plan.render()

    convert(src, dst)
    index = json.loads((dst / INDEX_NAME).read_text())
    assert len(index["episodes"]) == 3
    # every episode owns its container, so each span starts at zero
    assert {e["from_ns"] for e in index["episodes"]} == {0}
    assert {e["to_ns"] for e in index["episodes"]} == {2_000_000_000}
    assert len({e["ref"] for e in index["episodes"]}) == 3

    with PackSet(dst / "media") as packs:
        assert len(packs) == 3
        for key in list(packs.keys()):
            (tmp_path / "check.mp4").write_bytes(packs.read(key))
            assert probe(tmp_path / "check.mp4") == (2.0, FPS * 2)


def test_unaligned_dataset_keeps_its_concatenated_layout(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src, starts=(0.0, 3.0), seconds=6)  # 3.0 is not a keyframe
    text = plan_conversion(src).render()
    assert "1/2 episode boundaries" in text
    assert "keep their concatenated layout" in text

    convert(src, dst)
    index = json.loads((dst / INDEX_NAME).read_text())
    refs = {e["ref"] for e in index["episodes"]}
    assert len(refs) == 1  # one shared container, addressed by interval
    assert [e["from_ns"] for e in index["episodes"]] == [0, 3_000_000_000]


def test_no_split_option_keeps_the_source_layout(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src)
    plan = plan_conversion(src, split_episodes=False)
    assert "not split (--no-split-episodes)" in plan.render()
    convert(src, dst, plan=plan)
    index = json.loads((dst / INDEX_NAME).read_text())
    assert len({e["ref"] for e in index["episodes"]}) == 1


def test_split_at_refuses_an_empty_cut_list(tmp_path):
    """With no -segment_times the muxer cuts every two seconds on its own,
    so a caller that meant 'pass this through' must not reach it."""
    video = tmp_path / "v.mp4"
    make_video(video)
    with pytest.raises(ValueError, match="at least one cut time"):
        split_at(video, tmp_path / "parts", [])


def test_single_episode_file_is_packed_whole(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src, starts=(0.0,), seconds=6)
    convert(src, dst)
    index = json.loads((dst / INDEX_NAME).read_text())
    assert len(index["episodes"]) == 1
    with PackSet(dst / "media") as packs:
        assert len(packs) == 1
        (tmp_path / "check.mp4").write_bytes(packs.read(next(iter(packs.keys()))))
    assert probe(tmp_path / "check.mp4") == (6.0, FPS * 6)


def test_boundary_just_past_its_keyframe_still_cuts_there(tmp_path):
    """A boundary within tolerance but after the keyframe is aligned; asking
    the muxer for that timestamp would skip to the following keyframe."""
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src, starts=(0.0, 2.0005, 4.0), seconds=6)
    assert "3/3 episode boundaries" in plan_conversion(src).render()

    convert(src, dst)
    with PackSet(dst / "media") as packs:
        assert len(packs) == 3
        for key in list(packs.keys()):
            (tmp_path / "check.mp4").write_bytes(packs.read(key))
            assert probe(tmp_path / "check.mp4") == (2.0, FPS * 2)


def test_multiple_cameras_of_one_episode_share_a_shard(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src, cameras=("observation.left", "observation.right"))
    convert(src, dst, max_pack_bytes=24_000)  # small enough to force rolls
    index = json.loads((dst / INDEX_NAME).read_text())
    shards = {}
    for entry in index["episodes"]:
        shards.setdefault(entry["episode_index"], set()).add(entry["ref"].split("::")[-1])
    assert all(len(s) == 1 for s in shards.values()), shards


def test_preroll_before_first_episode_is_cut_off(tmp_path):
    """Footage before the first episode must not end up in its container.
    Episodes at 2.0s and 4.0s of a 6s file: 0-2s is nobody's."""
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src, starts=(2.0, 4.0))
    plan = plan_conversion(src)
    assert "footage before their first episode" in plan.render()

    convert(src, dst)
    index = json.loads((dst / INDEX_NAME).read_text())
    assert [(e["from_ns"], e["to_ns"]) for e in index["episodes"]] == [(0, 2_000_000_000)] * 2
    with PackSet(dst / "media") as packs:
        assert len(packs) == 2  # the pre-roll segment was dropped, not packed
        for key in list(packs.keys()):
            (tmp_path / "check.mp4").write_bytes(packs.read(key))
            assert probe(tmp_path / "check.mp4") == (2.0, FPS * 2)


def test_single_episode_with_preroll_is_trimmed(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src, starts=(4.0,))
    convert(src, dst)
    with PackSet(dst / "media") as packs:
        assert len(packs) == 1
        (tmp_path / "check.mp4").write_bytes(packs.read(next(iter(packs.keys()))))
    assert probe(tmp_path / "check.mp4") == (2.0, FPS * 2)


def test_cut_index_spans_match_the_containers_exactly(tmp_path):
    """Boundaries within tolerance of a keyframe: the index must state the
    produced container's actual keyframe-to-keyframe span, not the metadata
    interval, which may sit up to the tolerance away."""
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src, starts=(0.0, 2.0005, 4.0))
    convert(src, dst)
    index = json.loads((dst / INDEX_NAME).read_text())
    spans = sorted((e["episode_index"], e["to_ns"]) for e in index["episodes"])
    # containers are exactly 2.0 s each; metadata said 2.0005/1.9995
    assert spans == [(0, 2_000_000_000), (1, 2_000_000_000), (2, 2_000_000_000)]


def test_first_boundary_within_tolerance_of_zero_is_not_preroll(tmp_path):
    """A metadata start of 0.0005 s maps to the keyframe at 0; the old code
    treated any nonzero start as pre-roll and refused a valid dataset."""
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build(src, starts=(0.0005, 2.0, 4.0))
    convert(src, dst)
    with PackSet(dst / "media") as packs:
        assert len(packs) == 3
        for key in sorted(packs.keys()):
            (tmp_path / "check.mp4").write_bytes(packs.read(key))
            assert probe(tmp_path / "check.mp4") == (2.0, FPS * 2)


def test_gap_between_episodes_stays_out_of_the_claimed_span(tmp_path):
    """Footage between two episodes lands in the preceding container (it has
    to go somewhere in a packet copy), but the index span must end where the
    episode does, not where the container does."""
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    # keyframes every second (gop=FPS); ep0 = [0, 2), ep1 = [3, 6): 2-3 s is a gap
    build(src, starts=(0.0, 3.0), ends=(2.0, 6.0), gop=FPS)
    convert(src, dst)
    index = json.loads((dst / INDEX_NAME).read_text())
    spans = {e["episode_index"]: (e["from_ns"], e["to_ns"]) for e in index["episodes"]}
    assert spans[0] == (0, 2_000_000_000)  # not 3 s, despite a 3 s container
    assert spans[1] == (0, 3_000_000_000)
