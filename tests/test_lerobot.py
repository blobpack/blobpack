"""LeRobot conversion across the layouts that exist in the wild.

Four combinations occur on the Hub: v2.x and v3.x, each with camera
observations either as video files or as frames embedded in the data
table. Nothing here decodes video, so the fixtures use stand-in bytes and
no ffmpeg is needed.
"""

import json
import zipfile

import pytest

from blobpack import BlobPackError, PackSet
from blobpack.cli import main as cli_main
from blobpack.lerobot import INDEX_NAME, convert, plan_conversion

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

FPS = 10
CAM = "observation.images.cam_high"
WRIST = "observation.images.wrist"


def write_info(root, **overrides):
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {CAM: {"dtype": "video"}, "action": {"dtype": "float32"}},
        "total_episodes": 2,
        "total_frames": 20,
    }
    info.update(overrides)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return info


def build_v3_video(root, cameras=(CAM,), episodes=((0, 10, 0, 0.0, 1.0), (1, 10, 0, 1.0, 2.0))):
    write_info(
        root,
        features={c: {"dtype": "video"} for c in cameras} | {"action": {"dtype": "float32"}},
        total_episodes=len(episodes),
        total_frames=sum(e[1] for e in episodes),
    )
    (root / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    columns = {
        "episode_index": [e[0] for e in episodes],
        "length": [e[1] for e in episodes],
        "dataset_from_index": [0] * len(episodes),
        "dataset_to_index": [e[1] for e in episodes],
    }
    for cam in cameras:
        columns[f"videos/{cam}/chunk_index"] = [0] * len(episodes)
        columns[f"videos/{cam}/file_index"] = [e[2] for e in episodes]
        columns[f"videos/{cam}/from_timestamp"] = [e[3] for e in episodes]
        columns[f"videos/{cam}/to_timestamp"] = [e[4] for e in episodes]
    pq.write_table(pa.table(columns), root / "meta" / "episodes" / "file-000.parquet")
    payloads = {}
    for cam in cameras:
        for file_index in sorted({e[2] for e in episodes}):
            path = root / "videos" / cam / "chunk-000" / f"file-{file_index:03d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = f"{cam}|{file_index}".encode() * 200
            path.write_bytes(data)
            payloads[path.relative_to(root).as_posix()] = data
    return payloads


def build_v2_video(root, cameras=(CAM,), episodes=(0, 1, 2)):
    write_info(
        root,
        codebase_version="v2.1",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        features={c: {"dtype": "video"} for c in cameras} | {"action": {"dtype": "float32"}},
        total_episodes=len(episodes),
        total_frames=10 * len(episodes),
    )
    lines = [json.dumps({"episode_index": e, "tasks": ["t"], "length": 10}) for e in episodes]
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    payloads = {}
    for cam in cameras:
        for episode in episodes:
            path = root / "videos" / "chunk-000" / cam / f"episode_{episode:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = f"{cam}|{episode}".encode() * 150
            path.write_bytes(data)
            payloads[path.relative_to(root).as_posix()] = data
    return payloads


def build_embedded(root, version="v3.0", cameras=(CAM,), rows=8, tables=2):
    common = {
        "codebase_version": version,
        "features": {c: {"dtype": "image"} for c in cameras} | {"action": {"dtype": "float32"}},
        "total_episodes": tables,
        "total_frames": rows * tables,
    }
    if version.startswith("v2"):
        common["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    write_info(root, **common)
    if version.startswith("v2"):
        lines = [json.dumps({"episode_index": i, "tasks": ["t"], "length": rows}) for i in range(tables)]
        (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    else:
        (root / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": list(range(tables)),
                    "length": [rows] * tables,
                    "dataset_from_index": [0] * tables,
                    "dataset_to_index": [rows] * tables,
                }
            ),
            root / "meta" / "episodes" / "file-000.parquet",
        )
    frames = {}
    for table_index in range(tables):
        columns = {"action": pa.array([float(i) for i in range(rows)], pa.float32())}
        for cam in cameras:
            payloads = [{"bytes": f"{cam}|{table_index}|{i}".encode() * 20, "path": f"{i}.jpg"} for i in range(rows)]
            columns[cam] = pa.array(payloads, pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
            for i, payload in enumerate(payloads):
                frames[f"frames/{cam}/chunk-000/file-{table_index:03d}/{i:08d}.jpg"] = payload["bytes"]
        path = root / "data" / "chunk-000" / f"file-{table_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), path)
    return frames


# ---------------------------------------------------------------- planning


def test_plan_describes_v3_video_without_writing(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_v3_video(src)
    plan = plan_conversion(src)
    assert not plan.changes_schema
    assert not dst.exists()
    text = plan.render()
    assert "lossless" in text
    assert "copied unchanged" in text
    assert "SCHEMA CHANGES" not in text


def test_plan_flags_schema_change_for_embedded_frames(tmp_path):
    src = tmp_path / "ds"
    src.mkdir()
    build_embedded(src)
    plan = plan_conversion(src)
    assert plan.changes_schema
    text = plan.render()
    assert "SCHEMA CHANGES" in text
    assert "rewritten" in text


def build_mixed(root, rows=8):
    """A dataset with one video camera and one camera embedded in the table."""
    payloads = build_v3_video(root)
    write_info(
        root,
        features={
            CAM: {"dtype": "video"},
            WRIST: {"dtype": "image"},
            "action": {"dtype": "float32"},
        },
        total_episodes=2,
        total_frames=20,
    )
    frames = {}
    columns = {"action": pa.array([float(i) for i in range(rows)], pa.float32())}
    cells = [{"bytes": f"{WRIST}|{i}".encode() * 20, "path": f"{i}.jpg"} for i in range(rows)]
    columns[WRIST] = pa.array(cells, pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    for i, cell in enumerate(cells):
        frames[f"frames/{WRIST}/chunk-000/file-000/{i:08d}.jpg"] = cell["bytes"]
    path = root / "data" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return payloads, frames


def test_plan_honours_skip_options(tmp_path):
    src = tmp_path / "ds"
    src.mkdir()
    build_mixed(src)
    plan = plan_conversion(src, images="skip")
    assert not plan.changes_schema  # only the video is converted
    assert "skipped" in plan.render()


def test_mixed_dataset_converts_both_media_kinds(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    videos, frames = build_mixed(src)
    result = convert(src, dst)
    assert result["video_keys"] == [CAM] and result["image_keys"] == [WRIST]
    with PackSet(dst / "media") as packs:
        for relative, data in videos.items():
            assert packs.read(relative) == data
        for key, payload in frames.items():
            assert packs.read(key) == payload
    assert (dst / INDEX_NAME).exists()


def test_skipping_images_keeps_the_table_untouched(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_mixed(src)
    result = convert(src, dst, images="skip")
    assert result["image_keys"] == []
    # no rewrite happened: the tables travel as byte-for-byte copies
    src_tables = sorted((src / "data").rglob("*.parquet"))
    dst_tables = sorted((dst / "data").rglob("*.parquet"))
    assert [p.name for p in dst_tables] == [p.name for p in src_tables]
    assert all(d.read_bytes() == s.read_bytes() for d, s in zip(dst_tables, src_tables))


# ---------------------------------------------------------------- v3 video


def test_v3_video_is_packed_unchanged(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    payloads = build_v3_video(src)
    result = convert(src, dst)
    assert result["videos"] == 1  # both episodes share one concatenated file
    with PackSet(dst / "media") as packs:
        for relative, data in payloads.items():
            assert packs.read(relative) == data
    index = json.loads((dst / INDEX_NAME).read_text())
    assert [e["from_ns"] for e in index["episodes"]] == [0, 1_000_000_000]


def test_v3_multiple_cameras_get_their_own_blobs(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_v3_video(src, cameras=(CAM, WRIST), episodes=((0, 10, 0, 0.0, 1.0), (1, 10, 1, 0.0, 1.0)))
    result = convert(src, dst)
    assert result["videos"] == 4  # two cameras x two files
    index = json.loads((dst / INDEX_NAME).read_text())
    assert len(index["episodes"]) == 4
    per_camera = {e["video_key"]: e["ref"] for e in index["episodes"] if e["episode_index"] == 0}
    assert len(set(per_camera.values())) == 2  # cameras never share a blob


# ---------------------------------------------------------------- v2 video


def test_v2_video_gives_one_blob_per_episode(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    payloads = build_v2_video(src, cameras=(CAM, WRIST))
    result = convert(src, dst)
    assert result["videos"] == 6  # 3 episodes x 2 cameras, one file each
    with PackSet(dst / "media") as packs:
        for relative, data in payloads.items():
            assert packs.read(relative) == data
    index = json.loads((dst / INDEX_NAME).read_text())
    assert len(index["episodes"]) == 6
    # each episode occupies its whole file, so every interval starts at zero
    assert {e["from_ns"] for e in index["episodes"]} == {0}
    assert {e["to_ns"] for e in index["episodes"]} == {1_000_000_000}


def test_v2_reports_missing_video(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_v2_video(src)
    next((src / "videos").rglob("*.mp4")).unlink()
    with pytest.raises(BlobPackError, match="missing from"):
        convert(src, dst)


# ---------------------------------------------------------------- embedded


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_embedded_frames_move_into_packs(tmp_path, version):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    frames = build_embedded(src, version=version, cameras=(CAM, WRIST))
    result = convert(src, dst)
    assert result["embedded"]["frames"] == len(frames)
    with PackSet(dst / "media") as packs:
        assert sorted(packs.keys()) == sorted(frames)
        for key, payload in frames.items():
            assert packs.read(key) == payload
    table = pq.read_table(next((dst / "data").rglob("*.parquet")))
    assert table.column(CAM).to_pylist()[0].startswith("zip://frames/")
    original = pq.read_table(next((src / "data").rglob("*.parquet")))
    assert table.column("action").to_pylist() == original.column("action").to_pylist()


def test_embedded_references_resolve_to_the_original_bytes(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_embedded(src)
    convert(src, dst)
    table = pq.read_table(next((dst / "data").rglob("*.parquet")))
    original = pq.read_table(next((src / "data").rglob("*.parquet")))
    with PackSet(dst / "media") as packs:
        for ref, cell in zip(table.column(CAM).to_pylist(), original.column(CAM).to_pylist()):
            assert packs.read(ref) == cell["bytes"]


def test_skipping_every_feature_leaves_nothing_to_do(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_embedded(src)
    with pytest.raises(BlobPackError):
        convert(src, dst, images="skip")


# ---------------------------------------------------------------- refusals


def test_unknown_version_is_refused(tmp_path):
    src = tmp_path / "ds"
    src.mkdir()
    build_v3_video(src)
    write_info(src, codebase_version="v4.0")
    with pytest.raises(BlobPackError, match="codebase_version"):
        plan_conversion(src)


def test_non_lerobot_directory_is_refused(tmp_path):
    with pytest.raises(BlobPackError, match="does not look like"):
        plan_conversion(tmp_path)


def test_inconsistent_intervals_abort(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_v3_video(src, episodes=((0, 161, 0, 0.0, 5.0),))  # 5 s covers 50 frames
    with pytest.raises(BlobPackError, match="interval covers"):
        convert(src, dst)


def test_overlapping_intervals_abort(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_v3_video(src, episodes=((0, 10, 0, 0.0, 1.0), (1, 10, 0, 0.5, 1.5)))
    with pytest.raises(BlobPackError, match="overlaps"):
        convert(src, dst)


# ---------------------------------------------------------------- cli


def test_cli_prints_plan_and_stops_on_dry_run(tmp_path, capsys):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_v3_video(src)
    assert cli_main(["convert-lerobot", str(src), str(dst), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "LeRobot v3.0" in out
    assert not dst.exists()


def test_cli_declining_writes_nothing(tmp_path, monkeypatch, capsys):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_v3_video(src)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert cli_main(["convert-lerobot", str(src), str(dst)]) == 1
    assert "aborted" in capsys.readouterr().out
    assert not dst.exists()


def test_cli_converts_after_confirmation(tmp_path, monkeypatch, capsys):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    payloads = build_v3_video(src)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert cli_main(["convert-lerobot", str(src), str(dst)]) == 0
    assert "copied unchanged" in capsys.readouterr().out
    assert cli_main(["verify", str(dst / "media")]) == 0
    shard = next((dst / "media").glob("*.zip"))
    with zipfile.ZipFile(shard) as bundle:
        name = next(iter(payloads))
        assert bundle.read(name) == payloads[name]


def test_cli_yes_skips_the_prompt(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_embedded(src)
    assert cli_main(["convert-lerobot", str(src), str(dst), "--yes"]) == 0
    assert (dst / "data").exists()


def test_converted_dataset_is_self_contained(tmp_path):
    """The plan promises meta/ and untouched data/ are copied; the output
    must be a complete dataset, not a media dump beside the source."""
    src = tmp_path / "src"
    build_v3_video(src)
    (src / "data").mkdir()
    pq.write_table(pa.table({"action": [0.5] * 4}), src / "data" / "part-000.parquet")
    dst = tmp_path / "out"
    convert(src, dst)
    assert (dst / "meta" / "info.json").read_bytes() == (src / "meta" / "info.json").read_bytes()
    assert (dst / "data" / "part-000.parquet").read_bytes() == (src / "data" / "part-000.parquet").read_bytes()


def test_rewritten_tables_are_not_overwritten_by_the_copy(tmp_path):
    src = tmp_path / "src"
    build_embedded(src)
    dst = tmp_path / "out"
    convert(src, dst)
    # data/ was rewritten by extraction: it must hold references, not a copy
    table = pq.read_table(sorted((dst / "data").rglob("*.parquet"))[0])
    assert str(table.column(CAM).type) == "string"
    assert (dst / "meta").is_dir()


def test_rewritten_table_updates_hf_feature_metadata(tmp_path):
    """The source parquet declares the frame column as an HF Image feature;
    keeping that on a string column makes `datasets` decode references as
    images. The rewrite must re-declare it."""
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_embedded(src)
    declared = {
        "info": {
            "features": {
                CAM: {"_type": "Image"},
                "action": {"dtype": "float32", "_type": "Value"},
            }
        }
    }
    for path in sorted((src / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        pq.write_table(table.cast(table.schema.with_metadata({b"huggingface": json.dumps(declared)})), path)

    convert(src, dst)
    out = pq.read_schema(sorted((dst / "data").rglob("*.parquet"))[0])
    features = json.loads(out.metadata[b"huggingface"])["info"]["features"]
    assert features[CAM] == {"dtype": "string", "_type": "Value"}
    assert features["action"] == {"dtype": "float32", "_type": "Value"}  # untouched


def test_path_only_frames_are_packed_from_disk(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_embedded(src, rows=4, tables=1)
    # rewrite one table so every cell is the {bytes: None, path: ...} variant
    frames_dir = src / "images"
    frames_dir.mkdir()
    path = sorted((src / "data").rglob("*.parquet"))[0]
    table = pq.read_table(path)
    payloads = [v["bytes"] for v in table.column(CAM).to_pylist()]
    cells = []
    for i, payload in enumerate(payloads):
        (frames_dir / f"{i}.png").write_bytes(payload)
        cells.append({"bytes": None, "path": f"images/{i}.png"})
    column = pa.array(cells, pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    pq.write_table(table.set_column(table.column_names.index(CAM), CAM, column), path)

    result = convert(src, dst)
    assert result["embedded"]["frames"] == len(payloads)
    refs = pq.read_table(sorted((dst / "data").rglob("*.parquet"))[0]).column(CAM).to_pylist()
    with PackSet(dst / "media") as packs:
        assert [packs.read(r) for r in refs] == payloads


def test_dangling_frame_path_aborts(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_embedded(src, rows=2, tables=1)
    path = sorted((src / "data").rglob("*.parquet"))[0]
    table = pq.read_table(path)
    column = pa.array(
        [{"bytes": None, "path": "images/missing.png"}] * table.num_rows,
        pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    pq.write_table(table.set_column(table.column_names.index(CAM), CAM, column), path)
    with pytest.raises(BlobPackError, match="does not exist"):
        convert(src, dst)


def test_two_frame_columns_of_one_row_share_a_shard(tmp_path):
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_embedded(src, cameras=(CAM, WRIST), rows=16, tables=1)
    convert(src, dst, max_pack_bytes=30_000)
    table = pq.read_table(sorted((dst / "data").rglob("*.parquet"))[0])
    pairs = zip(table.column(CAM).to_pylist(), table.column(WRIST).to_pylist())
    for cam_ref, wrist_ref in pairs:
        assert cam_ref.split("::")[-1] == wrist_ref.split("::")[-1]


def test_frame_path_escaping_the_dataset_root_is_refused(tmp_path):
    """A path inside a table cell is untrusted data; '../' must not read
    files outside the dataset root into the pack."""
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"top secret")
    src, dst = tmp_path / "ds", tmp_path / "out"
    src.mkdir()
    build_embedded(src, rows=2, tables=1)
    path = sorted((src / "data").rglob("*.parquet"))[0]
    table = pq.read_table(path)
    column = pa.array(
        [{"bytes": None, "path": "../secret.png"}] * table.num_rows,
        pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    pq.write_table(table.set_column(table.column_names.index(CAM), CAM, column), path)
    with pytest.raises(BlobPackError, match="outside"):
        convert(src, dst)
