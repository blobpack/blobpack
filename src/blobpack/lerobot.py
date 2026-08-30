"""Convert LeRobot datasets into Blob Packs.

LeRobot stores camera observations two ways, and which one decides what
conversion can promise:

- **video files** (v2.x per episode; v3 concatenated per camera): packet
  copies, addressed by half-open ``[from_ns, to_ns)`` intervals -- never
  decoded or re-encoded.
- **frames embedded in the table**: moved into packs, the column becomes
  references -- bytes lossless, but the schema changes.

Not equivalent, so conversion is planned first and executed only after
the caller sees what each feature will become.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from . import PackWriter
from ._video import ALIGNMENT_TOLERANCE_S, check_alignment, ffmpeg_available, split_at
from ._zip import BlobPackError

INDEX_NAME = "video_index.json"
NS_PER_SECOND = 1_000_000_000

#: what a step does to the data, in the order a reader should worry about it
LOSSLESS = "lossless"
LOSSY = "LOSSY, re-encoded"
SCHEMA = "bytes lossless, SCHEMA CHANGES"
SKIPPED = "skipped"


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise BlobPackError(
            "converting LeRobot datasets needs pyarrow; install the extra: pip install 'blobpack[lerobot]'"
        ) from exc
    return pq


@dataclass
class FeaturePlan:
    """One camera feature and what conversion will do with it."""

    name: str
    mode: str  # "video" or "embedded"
    action: str  # "copy", "extract", "skip"
    fidelity: str
    files: int = 0
    bytes: int = 0
    frames: int = 0
    notes: list[str] = field(default_factory=list)
    #: video paths that will be cut into one container per episode
    split_paths: tuple[str, ...] = ()


@dataclass
class ConversionPlan:
    version: str
    episodes: int
    frames: int
    fps: float
    features: list[FeaturePlan]
    table_files: int
    table_bytes: int
    table_rewritten: bool
    src: Path

    @property
    def changes_schema(self) -> bool:
        return any(f.fidelity == SCHEMA for f in self.features)

    def render(self) -> str:
        lines = [
            f"LeRobot {self.version} - {self.episodes} episodes, {self.frames:,} frames @ {self.fps:g} fps",
            "",
        ]
        for feature in self.features:
            if feature.mode == "video":
                what = f"video    {feature.files} file(s), {_human(feature.bytes)}"
            else:
                what = f"image    {feature.frames:,} frames inside the table"
            lines.append(f"  {feature.name:<34} {what}")
            for note in feature.notes:
                lines.append(f"      -> {note}")
            lines.append(f"      {'':<4}{feature.fidelity}")
        table = "rewritten" if self.table_rewritten else "copied unchanged"
        lines += [
            f"  {'data/*.parquet':<34} {self.table_files} file(s), {_human(self.table_bytes)}",
            f"      -> {table}",
            f"  {'meta/':<34} -> copied unchanged",
        ]
        if self.changes_schema:
            lines += [
                "",
                "  A rewritten table no longer matches the LeRobot schema: readers",
                "  must resolve blob references instead of reading frame bytes.",
            ]
        return "\n".join(lines)


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------- detection


def _load_info(src: Path) -> dict:
    info_path = src / "meta" / "info.json"
    if not info_path.exists():
        raise BlobPackError(f"{src} does not look like a LeRobot dataset ({info_path} missing)")
    return json.loads(info_path.read_text())


def _major_version(info: dict) -> int:
    raw = str(info.get("codebase_version", "")).lstrip("v")
    head = raw.split(".")[0]
    if not head.isdigit():
        raise BlobPackError(f"unrecognized codebase_version {info.get('codebase_version')!r}")
    version = int(head)
    if version not in (2, 3):
        raise BlobPackError(
            f"unsupported LeRobot codebase_version {info.get('codebase_version')!r}; expected v2.x or v3.x"
        )
    return version


def _media_features(info: dict) -> tuple[list[str], list[str]]:
    features = info.get("features", {})
    video = sorted(k for k, v in features.items() if v.get("dtype") == "video")
    image = sorted(k for k, v in features.items() if v.get("dtype") == "image")
    if not video and not image:
        raise BlobPackError("no video or image features in meta/info.json; nothing to convert")
    return video, image


def _episode_rows_v3(src: Path) -> list[dict]:
    pq = _require_pyarrow()
    files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise BlobPackError(f"no episode metadata under {src / 'meta' / 'episodes'}")
    keep = ("episode_index", "dataset_from_index", "dataset_to_index", "length")
    rows: list[dict] = []
    for path in files:
        table = pq.read_table(path)
        wanted = [n for n in table.column_names if n.startswith("videos/") or n in keep]
        rows.extend(table.select(wanted).to_pylist())
    rows.sort(key=lambda row: row["episode_index"])
    return rows


def _episode_rows_v2(src: Path) -> list[dict]:
    path = src / "meta" / "episodes.jsonl"
    if not path.exists():
        raise BlobPackError(f"no episode metadata at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["episode_index"])
    return rows


def _table_files(src: Path) -> list[Path]:
    return sorted((src / "data").rglob("*.parquet"))


def _video_path(info: dict, version: int, video_key: str, episode: int, row: dict) -> str:
    template = info.get("video_path")
    if not template:
        raise BlobPackError("meta/info.json has no video_path template")
    if version == 3:
        prefix = f"videos/{video_key}/"
        return template.format(
            video_key=video_key,
            chunk_index=row[prefix + "chunk_index"],
            file_index=row[prefix + "file_index"],
        )
    chunk_size = int(info.get("chunks_size") or 1000)
    return template.format(
        video_key=video_key,
        episode_chunk=episode // chunk_size,
        episode_index=episode,
    )


def _split_survey(
    src: Path, info: dict, key: str, rows: list[dict], split_episodes: bool
) -> tuple[set[str], list[str]]:
    """Which of this camera's files can be cut on keyframes, and what to say.

    A file is split only when every episode boundary inside it lands on a
    keyframe: the segment muxer can cut nowhere else, and asking it to cut
    elsewhere silently moves the boundary to the next keyframe.
    """
    if not split_episodes:
        return set(), ["not split (--no-split-episodes)"]
    if not ffmpeg_available():
        return set(), ["not split: ffmpeg and ffprobe are not on PATH"]
    prefix = f"videos/{key}/"
    starts: dict[str, dict[int, float]] = {}
    for row in rows:
        if prefix + "from_timestamp" not in row:
            return set(), ["not split: no per-episode timing metadata"]
        relative = _video_path(info, 3, key, row["episode_index"], row)
        starts.setdefault(relative, {})[row["episode_index"]] = float(row[prefix + "from_timestamp"])
    splittable, notes = set(), []
    total_aligned = total = preroll = 0
    for relative, boundaries in sorted(starts.items()):
        try:
            check = check_alignment(src / relative, boundaries)
        except Exception as exc:  # probing failure becomes a plan note, not a crash
            notes.append(f"not split: could not probe {relative} ({type(exc).__name__})")
            return set(), notes
        total_aligned += len(check.aligned)
        total += check.total
        if not check.unaligned:
            splittable.add(relative)
            if min(boundaries.values()) > 0:
                preroll += 1
    notes.append(f"checked: {total_aligned}/{total} episode boundaries land on keyframes")
    if preroll:
        notes.append(f"{preroll} file(s) carry footage before their first episode; it is cut off, not packed")
    if total_aligned != total:
        notes.append(
            f"{len(starts) - len(splittable)} file(s) keep their concatenated layout, "
            "addressed by interval; cutting them would need a re-encode"
        )
    return splittable, notes


def plan_conversion(
    src: Path, *, images: str = "extract", video: str = "copy", split_episodes: bool = True
) -> ConversionPlan:
    """Describe what conversion would do, without writing anything."""
    if images not in ("extract", "skip") or video not in ("copy", "skip"):
        raise ValueError("images must be extract|skip and video must be copy|skip")
    info = _load_info(src)
    version = _major_version(info)
    video_keys, image_keys = _media_features(info)
    rows = _episode_rows_v3(src) if version == 3 else _episode_rows_v2(src)

    features: list[FeaturePlan] = []
    for key in video_keys:
        paths = {_video_path(info, version, key, row["episode_index"], row) for row in rows}
        missing = sorted(p for p in paths if not (src / p).exists())
        if missing:
            raise BlobPackError(f"{key}: {missing[0]} is missing from {src}")
        size = sum((src / p).stat().st_size for p in paths)
        if video == "skip":
            features.append(FeaturePlan(key, "video", "skip", SKIPPED, len(paths), size, notes=["left in place"]))
            continue
        if version == 2:
            features.append(
                FeaturePlan(
                    key,
                    "video",
                    "copy",
                    LOSSLESS,
                    len(paths),
                    size,
                    notes=["one file per episode already, copied unchanged into media/"],
                )
            )
            continue
        splittable, notes = _split_survey(src, info, key, rows, split_episodes)
        if splittable:
            notes.insert(0, "split into one container per episode, copied packet for packet")
        else:
            notes.insert(0, "copied unchanged into media/, episodes indexed by [from_ns, to_ns)")
        features.append(
            FeaturePlan(
                key,
                "video",
                "copy",
                LOSSLESS,
                len(paths),
                size,
                notes=notes,
                split_paths=tuple(sorted(splittable)),
            )
        )

    table_paths = _table_files(src)
    table_bytes = sum(p.stat().st_size for p in table_paths)
    frames_total = int(info.get("total_frames") or sum(r.get("length", 0) for r in rows))
    for key in image_keys:
        if images == "skip":
            features.append(
                FeaturePlan(key, "embedded", "skip", SKIPPED, frames=frames_total, notes=["left inside the table"])
            )
            continue
        features.append(
            FeaturePlan(
                key,
                "embedded",
                "extract",
                SCHEMA,
                frames=frames_total,
                notes=[
                    "extracted to media/, column rewritten to blob references",
                    "random reads stop pulling a whole row group",
                ],
            )
        )

    if all(f.action == "skip" for f in features):
        raise BlobPackError("every media feature is skipped; there is nothing to convert")

    return ConversionPlan(
        version=str(info.get("codebase_version")),
        episodes=int(info.get("total_episodes") or len(rows)),
        frames=frames_total,
        fps=float(info.get("fps") or 0),
        features=features,
        table_files=len(table_paths),
        table_bytes=table_bytes,
        table_rewritten=any(f.action == "extract" for f in features),
        src=src,
    )


# ---------------------------------------------------------------- execution


def _validate_intervals(entries: list[dict], fps: float, tolerance_frames: float) -> list[str]:
    problems: list[str] = []
    by_file: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        by_file.setdefault((entry["video_key"], entry["video_path"]), []).append(entry)
    for (video_key, video_path), grouped in sorted(by_file.items()):
        grouped.sort(key=lambda entry: entry["from_ns"])
        previous_end = None
        for entry in grouped:
            if entry["from_ns"] >= entry["to_ns"]:
                problems.append(f"{video_key} episode {entry['episode_index']}: empty or reversed interval")
            if previous_end is not None and entry["from_ns"] < previous_end:
                problems.append(
                    f"{video_key} {video_path}: episode {entry['episode_index']} overlaps the previous interval"
                )
            previous_end = max(previous_end or 0, entry["to_ns"])
    if fps > 0:
        for entry in entries:
            expected = entry["frames"]
            if not expected:
                continue
            covered = (entry["to_ns"] - entry["from_ns"]) / NS_PER_SECOND * fps
            if abs(covered - expected) > tolerance_frames:
                problems.append(
                    f"{entry['video_key']} episode {entry['episode_index']}: interval covers {covered:.2f} frames "
                    f"but the episode has {expected}"
                )
    return problems


def _video_entries(src: Path, info: dict, version: int, keys: list[str], rows: list[dict]) -> list[dict]:
    entries = []
    for row in rows:
        episode = row["episode_index"]
        frames = row.get("length")
        if frames is None and row.get("dataset_to_index") is not None:
            frames = row["dataset_to_index"] - row["dataset_from_index"]
        for key in keys:
            relative = _video_path(info, version, key, episode, row)
            if version == 3:
                prefix = f"videos/{key}/"
                missing = [c for c in ("from_timestamp", "to_timestamp") if prefix + c not in row]
                if missing:
                    raise BlobPackError(f"episode {episode} has no {key} timing metadata (missing {missing[0]})")
                from_ns = round(float(row[prefix + "from_timestamp"]) * NS_PER_SECOND)
                to_ns = round(float(row[prefix + "to_timestamp"]) * NS_PER_SECOND)
            else:
                # v2.x stores one file per episode, so the blob is the episode
                fps = float(info.get("fps") or 0)
                if fps <= 0 or not frames:
                    raise BlobPackError(
                        f"episode {episode}: cannot derive its interval "
                        f"(fps={info.get('fps')!r}, length={frames!r} in the episode metadata)"
                    )
                from_ns = 0
                to_ns = round(frames / fps * NS_PER_SECOND)
            entries.append(
                {
                    "episode_index": episode,
                    "video_key": key,
                    "video_path": relative,
                    "frames": frames,
                    "from_ns": from_ns,
                    "to_ns": to_ns,
                }
            )
    return entries


def _extension(path_hint: str | None, default: str = ".jpg") -> str:
    if not path_hint:
        return default
    suffix = Path(path_hint).suffix
    return suffix if suffix else default


def _rewrite_hf_metadata(schema, keys: list[str]):
    """Update the parquet's Hugging Face feature metadata for rewritten columns.

    The source table often declares the frame column as an ``Image``
    feature; keeping that on a column that now holds reference strings
    makes ``datasets`` decode the strings as images and fail. The column
    is re-declared as a plain string value; everything else is kept.
    """
    metadata = dict(schema.metadata or {})
    raw = metadata.get(b"huggingface")
    if not raw:
        return schema
    try:
        declared = json.loads(raw)
        features = declared["info"]["features"]
    except (ValueError, KeyError, TypeError):
        return schema  # unrecognized layout: leave it alone
    for key in keys:
        if key in features:
            features[key] = {"dtype": "string", "_type": "Value"}
    metadata[b"huggingface"] = json.dumps(declared).encode()
    return schema.with_metadata(metadata)


def _extract_embedded(src: Path, dst: Path, writer: PackWriter, keys: list[str]) -> dict:
    """Move embedded frame bytes into packs and rewrite the tables."""
    pq = _require_pyarrow()
    import pyarrow as pa

    out_dir = dst / "data"
    stats = {"frames": 0, "bytes": 0, "tables": 0}
    for path in _table_files(src):
        relative = path.relative_to(src / "data")
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = pq.ParquetFile(path)
        writer_out = None
        row_offset = 0
        stem = relative.with_suffix("").as_posix()
        try:
            for group in range(handle.num_row_groups):
                table = handle.read_row_group(group)
                present = [key for key in keys if key in table.column_names]
                columns = {key: table.column(key).combine_chunks() for key in present}
                refs: dict[str, list] = {key: [] for key in present}
                # row-major across the frame columns, so one row's frames are
                # adjacent and the per-row group keeps them in one shard
                for index in range(table.num_rows):
                    for key in present:
                        value = columns[key][index].as_py()
                        payload = value.get("bytes") if isinstance(value, dict) else value
                        hint = value.get("path") if isinstance(value, dict) else None
                        if payload is None and hint:
                            # HF Image path variant: the frame lives beside the
                            # table; the path is table data, keep it under src
                            frame_path = (src / hint).resolve()
                            if not frame_path.is_relative_to(src.resolve()):
                                raise BlobPackError(
                                    f"{path.name} row {row_offset + index}: {key} points outside "
                                    f"the dataset root ({hint})"
                                )
                            if not frame_path.is_file():
                                raise BlobPackError(
                                    f"{path.name} row {row_offset + index}: {key} points at {hint}, "
                                    "which does not exist under the dataset root"
                                )
                            payload = frame_path.read_bytes()
                        if payload is None:
                            refs[key].append("")
                            continue
                        blob_key = f"frames/{key}/{stem}/{row_offset + index:08d}{_extension(hint)}"
                        refs[key].append(writer.add(blob_key, payload, group=f"{stem}-{row_offset + index}"))
                        stats["frames"] += 1
                        stats["bytes"] += len(payload)
                for key in present:
                    table = table.set_column(table.column_names.index(key), key, pa.array(refs[key], pa.string()))
                table = table.cast(_rewrite_hf_metadata(table.schema, keys))
                if writer_out is None:
                    writer_out = pq.ParquetWriter(target, table.schema)
                writer_out.write_table(table)
                row_offset += table.num_rows
        finally:
            if writer_out is not None:
                writer_out.close()
        stats["tables"] += 1
    return stats


def _cut_episodes(src: Path, relative: str, entries: list[dict], tmp) -> dict:
    """Cut one file into per-episode containers.

    Cuts land on the probed keyframes, not the metadata timestamps: handing
    the muxer a hair-past timestamp would slide the cut to the next
    keyframe and mispair every following episode. Returns
    ``(episode, camera) -> (key, path, span_ns)`` with the container's
    actual keyframe-to-keyframe duration.
    """
    spans = sorted(
        (
            (e["episode_index"], e["video_key"], e["from_ns"], e["to_ns"])
            for e in entries
            if e["video_path"] == relative
        ),
        key=lambda span: span[2],
    )
    stem = Path(relative).with_suffix("")
    suffix = Path(relative).suffix or ".mp4"
    check = check_alignment(src / relative, {span[0]: span[2] / NS_PER_SECOND for span in spans})
    if check.unaligned:
        raise BlobPackError(f"{relative}: episode {check.unaligned[0]} does not start on a keyframe; cannot cut")
    # pre-roll is decided by the matched keyframe: a metadata start within
    # tolerance of zero maps to keyframe 0 and there is nothing to trim
    preroll = check.keyframes[spans[0][0]] > 0
    if len(spans) == 1 and not preroll:  # already one container per episode
        episode, video_key, from_ns, to_ns = spans[0]
        return {(episode, video_key): (f"{stem}/episode_{episode:06d}{suffix}", src / relative, to_ns - from_ns)}

    # cutting at the first episode's keyframe too keeps pre-roll footage out
    # of its container; the leading part is dropped rather than packed
    cut_points = [check.keyframes[span[0]] for span in spans[(0 if preroll else 1) :]]
    parts = split_at(src / relative, tmp, cut_points)
    if preroll:
        parts = parts[1:]
    if len(parts) != len(spans):
        raise BlobPackError(
            f"{relative}: cutting produced {len(parts)} parts for {len(spans)} episodes; "
            "refusing to guess which is which"
        )
    # within tolerance the container span is the truth (metadata carried the
    # jitter); a larger excess is inter-episode gap footage, which stays in
    # the container but must not be claimed as part of the episode
    tolerance_ns = round(ALIGNMENT_TOLERANCE_S * NS_PER_SECOND)
    boundaries = [round(check.keyframes[span[0]] * NS_PER_SECOND) for span in spans] + [spans[-1][3]]
    spans_ns = []
    for i, span in enumerate(spans):
        container = boundaries[i + 1] - boundaries[i]
        declared = span[3] - span[2]
        spans_ns.append(container if abs(container - declared) <= tolerance_ns else min(container, declared))
    return {
        (episode, video_key): (f"{stem}/episode_{episode:06d}{suffix}", part, spans_ns[i])
        for i, ((episode, video_key, _, _), part) in enumerate(zip(spans, parts))
    }


def _index_entry(entry: dict, refs: dict, split_refs: dict) -> dict:
    """One index row; a cut episode owns its container, so its span is whole."""
    cut = split_refs.get((entry["episode_index"], entry["video_key"]))
    return {
        "episode_index": entry["episode_index"],
        "video_key": entry["video_key"],
        "ref": cut[0] if cut else refs[entry["video_path"]],
        "from_ns": 0 if cut else entry["from_ns"],
        "to_ns": cut[1] if cut else entry["to_ns"],
        "frames": entry["frames"],
    }


def convert(
    src: Path,
    dst: Path,
    *,
    plan: ConversionPlan | None = None,
    ref_base: str | None = None,
    max_pack_bytes: int | None = None,
    tolerance_frames: float = 1.0,
    images: str = "extract",
    video: str = "copy",
    split_episodes: bool = True,
) -> dict:
    """Execute a conversion, planning it first when no plan is supplied."""
    plan = plan or plan_conversion(src, images=images, video=video, split_episodes=split_episodes)
    info = _load_info(src)
    version = _major_version(info)
    rows = _episode_rows_v3(src) if version == 3 else _episode_rows_v2(src)

    video_keys = [f.name for f in plan.features if f.mode == "video" and f.action == "copy"]
    image_keys = [f.name for f in plan.features if f.mode == "embedded" and f.action == "extract"]
    entries = _video_entries(src, info, version, video_keys, rows) if video_keys else []

    problems = _validate_intervals(entries, plan.fps, tolerance_frames)
    if problems:
        raise BlobPackError(
            "LeRobot video metadata is inconsistent, refusing to write a misleading index:\n  "
            + "\n  ".join(problems[:10])
            + ("\n  ..." if len(problems) > 10 else "")
        )

    dst.mkdir(parents=True, exist_ok=True)
    writer_kwargs = {} if max_pack_bytes is None else {"max_pack_bytes": max_pack_bytes}
    refs: dict[str, str] = {}
    embedded = {"frames": 0, "bytes": 0, "tables": 0}
    split_paths = {p for f in plan.features for p in f.split_paths}
    split_refs: dict[tuple[int, str], tuple[str, int]] = {}
    with ExitStack() as stack:
        # cut first, then write everything in episode order below, so one
        # episode's cameras stay contiguous and land in the same shard
        cuts: dict[tuple[int, str], tuple[str, Path, int]] = {}
        for relative in sorted(split_paths):
            tmp = stack.enter_context(tempfile.TemporaryDirectory())
            cuts.update(_cut_episodes(src, relative, entries, Path(tmp)))

        writer = stack.enter_context(PackWriter(dst / "media", ref_base=ref_base or "media", **writer_kwargs))
        first_use: dict[str, int] = {}
        for entry in sorted(entries, key=lambda e: (e["episode_index"], e["video_key"])):
            if entry["video_path"] not in split_paths:
                first_use.setdefault(entry["video_path"], entry["episode_index"])
        work: list[tuple[int, str, str, Path]] = [
            (episode, relative, relative, src / relative) for relative, episode in first_use.items()
        ]
        work += [(episode, video_key, key, path) for (episode, video_key), (key, path, _) in cuts.items()]
        for episode, video_key, key, path in sorted(work, key=lambda w: (w[0], w[1])):
            ref = writer.add_file(key, path, group=f"episode-{episode}")
            if (episode, video_key) in cuts:
                split_refs[(episode, video_key)] = (ref, cuts[(episode, video_key)][2])
            else:
                refs[key] = ref
        if image_keys:
            embedded = _extract_embedded(src, dst, writer, image_keys)
        blobs, blob_bytes, shards = writer.blobs_written, writer.bytes_written, writer.shards_written

    shutil.copytree(src / "meta", dst / "meta", dirs_exist_ok=True)
    if not image_keys and (src / "data").is_dir():
        # tables were not rewritten, so the promise is a byte-for-byte copy
        shutil.copytree(src / "data", dst / "data", dirs_exist_ok=True)

    if entries:
        index = {
            "format": "blobpack/lerobot-video-index@1",
            "source": {
                "codebase_version": info.get("codebase_version"),
                "fps": plan.fps,
                "video_keys": video_keys,
            },
            "note": (
                "Video is packet-copied, never re-encoded. Each entry addresses one episode as a "
                "half-open [from_ns, to_ns) interval of its blob, per the Blob Pack media-container "
                "rule; an episode cut into its own container spans that container whole."
            ),
            "episodes": [
                _index_entry(e, refs, split_refs)
                for e in sorted(entries, key=lambda e: (e["episode_index"], e["video_key"]))
            ],
        }
        (dst / INDEX_NAME).write_text(json.dumps(index, indent=1) + "\n")

    return {
        "blobs": blobs,
        "bytes": blob_bytes,
        "shards": shards,
        "videos": len(refs) + len(split_refs),
        "episodes": plan.episodes,
        "video_keys": video_keys,
        "image_keys": image_keys,
        "embedded": embedded,
        "index_path": (dst / INDEX_NAME) if entries else None,
    }
