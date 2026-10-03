"""Command-line interface: pack, unpack, ls, verify, manifest, and the convert-* migrations."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path

from . import BlobPackError, PackSet, PackWriter, __version__, _validate_key

MANIFEST_NAME = "pack.manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cmd_pack(args: argparse.Namespace) -> int:
    src = Path(args.src)
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 2
    files = sorted(p for p in src.rglob("*") if p.is_file())
    if not files:
        print(f"error: no files under {src}", file=sys.stderr)
        return 2
    with PackWriter(args.dst, ref_base=args.ref_base, max_pack_bytes=args.max_pack_bytes) as writer:
        for path in files:
            writer.add_file(path.relative_to(src).as_posix(), path)  # streamed, never materialized
        print(
            f"packed {writer.blobs_written} blobs "
            f"({writer.bytes_written:,} bytes) into "
            f"{writer.shards_written} shards under {args.dst}"
        )
    return 0


def _cmd_convert_wds(args: argparse.Namespace) -> int:
    """Stream WebDataset-style tar shards into a pack directory."""
    src = Path(args.src)
    tars = sorted(src.glob("*.tar")) if src.is_dir() else ([src] if src.suffix == ".tar" else [])
    if not tars:
        print(f"error: no .tar shards at {src}", file=sys.stderr)
        return 2
    with PackWriter(args.dst, ref_base=args.ref_base, max_pack_bytes=args.max_pack_bytes) as writer:
        for tar_path in tars:
            prefix = f"{tar_path.stem}/" if args.shard_prefix else ""
            with tarfile.open(tar_path, "r|*") as tar:  # streaming, no seek needed
                for member in tar:
                    if not member.isfile():
                        continue
                    key = member.name
                    while key.startswith("./"):
                        key = key[2:]
                    key = prefix + key
                    stream = tar.extractfile(member)
                    try:
                        writer.add_file(key, stream, size=member.size)
                    except BlobPackError:
                        print(
                            f"error: duplicate key {key!r} (from {tar_path.name}); WebDataset only "
                            "guarantees per-shard uniqueness -- rerun with --shard-prefix",
                            file=sys.stderr,
                        )
                        return 1
                    except ValueError as exc:
                        print(f"error: {tar_path.name}: unusable member name: {exc}", file=sys.stderr)
                        return 1
        print(
            f"converted {len(tars)} tar shard(s): {writer.blobs_written} blobs "
            f"({writer.bytes_written:,} bytes) into {writer.shards_written} pack shard(s) under {args.dst}"
        )
    return 0


def _choose_columns(plan) -> list[str] | None:
    """Ask which measured columns hold payloads; Lance does not say."""
    candidates = [c.name for c in plan.columns]
    print("Which columns hold payloads to extract?")
    for number, column in enumerate(plan.columns, start=1):
        print(f"  {number}) {column.name:<26} {column.evidence}")
    answer = input("Numbers, comma separated, or 'all' (blank aborts): ").strip().lower()
    if not answer:
        return None
    if answer == "all":
        return candidates
    chosen = []
    for piece in answer.split(","):
        piece = piece.strip()
        if not piece.isdigit() or not 1 <= int(piece) <= len(candidates):
            print(f"error: {piece!r} is not one of the listed numbers", file=sys.stderr)
            return None
        chosen.append(candidates[int(piece) - 1])
    return chosen


def _cmd_convert_lance(args: argparse.Namespace) -> int:
    """Move a Lance dataset's payload columns into packs, after confirmation."""
    from .lance import convert, plan_conversion

    plan = plan_conversion(Path(args.src), columns=args.column or None)
    print(f"{plan.render()}\n")
    if args.dry_run:
        return 0
    if not args.column:
        if args.yes:
            print(
                "error: name the columns to extract with --column; Lance does not record which "
                "column holds media, so there is nothing safe to assume",
                file=sys.stderr,
            )
            return 2
        chosen = _choose_columns(plan)
        if not chosen:
            print("aborted; nothing was written")
            return 1
        plan = plan_conversion(Path(args.src), columns=chosen)
        print()
        print(f"{plan.render()}\n")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("aborted; nothing was written")
        return 1
    result = convert(
        Path(args.src), Path(args.dst), plan=plan, ref_base=args.ref_base, max_pack_bytes=args.max_pack_bytes
    )
    print(
        f"wrote {result['values']:,} values ({result['bytes']:,} bytes) into {result['shards']} shard(s); "
        f"rewrote {', '.join(result['columns'])} in {result['table']}"
    )
    return 0


def _cmd_unpack(args: argparse.Namespace) -> int:
    import shutil

    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    out_resolved = out.resolve()
    with PackSet(args.src) as packs:
        keys = packs.keys()  # lazy; a catalog-backed set streams from SQLite
        for key in keys:
            try:
                _validate_key(key)
            except ValueError as exc:
                print(f"error: refusing to extract unsafe member: {exc}", file=sys.stderr)
                return 1
            target = out / key
            if not target.resolve().is_relative_to(out_resolved):
                print(f"error: member escapes destination: {key!r}", file=sys.stderr)
                return 1
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, NotADirectoryError):
                print(f"error: {key!r} collides with an already-extracted file", file=sys.stderr)
                return 1
            with packs.open(key) as blob, open(target, "wb") as sink:
                shutil.copyfileobj(blob, sink)  # bounded memory for multi-GB blobs
        print(f"unpacked {len(packs)} blobs into {out}")
    return 0


def _cmd_ls(args: argparse.Namespace) -> int:
    with PackSet(args.src) as packs:
        len(packs)  # materialize deferred remote directories
        for shard_name, shard in packs._shards.items():
            for key, size in shard.member_sizes():
                print(f"{size}\t{shard_name}\t{key}")
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    """Write a per-shard sha256 manifest next to the shards."""
    src = Path(args.src)
    shard_paths = sorted(src.glob("*.zip"))
    if not shard_paths:
        print(f"error: no *.zip shards under {src}", file=sys.stderr)
        return 2
    target = src / MANIFEST_NAME
    if target.exists() and not args.force:
        print(f"error: {target} exists; packs are immutable, use --force to rewrite", file=sys.stderr)
        return 2
    manifest = {
        "algorithm": "sha256",
        "shards": {path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size} for path in shard_paths},
    }
    target.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {target} for {len(shard_paths)} shard(s)")
    return 0


def _verify_manifest(src: Path, shard_paths: list[Path]) -> int:
    """Check shards against pack.manifest.json if present; returns failures."""
    target = src / MANIFEST_NAME
    if not target.exists():
        return 0
    failures = 0
    recorded = json.loads(target.read_text())["shards"]
    names = {path.name for path in shard_paths}
    for name in sorted(set(recorded) - names):
        print(f"FAIL {name}: listed in manifest but missing")
        failures += 1
    for name in sorted(names - set(recorded)):
        print(f"FAIL {name}: not listed in manifest")
        failures += 1
    for path in shard_paths:
        entry = recorded.get(path.name)
        if entry is None:
            continue
        if path.stat().st_size != entry["bytes"] or _sha256_file(path) != entry["sha256"]:
            print(f"FAIL {path.name}: does not match manifest")
            failures += 1
    return failures


def _cmd_verify(args: argparse.Namespace) -> int:
    """Full at-rest check: STORED members, unique keys, CRC over every byte."""
    shard_paths = sorted(Path(args.src).glob("*.zip"))
    if not shard_paths:
        print(f"error: no *.zip shards under {args.src}", file=sys.stderr)
        return 2
    seen: dict[str, str] = {}
    blobs = total = 0
    failures = _verify_manifest(Path(args.src), shard_paths)
    for path in shard_paths:
        try:
            with zipfile.ZipFile(path) as bundle:
                for info in bundle.infolist():
                    if info.is_dir():
                        continue
                    if info.compress_type != zipfile.ZIP_STORED:
                        print(f"FAIL {path.name}:{info.filename}: not STORED")
                        failures += 1
                    if info.flag_bits & 0x1:
                        print(f"FAIL {path.name}:{info.filename}: encrypted")
                        failures += 1
                    if info.filename in seen:
                        print(f"FAIL {path.name}:{info.filename}: duplicate key (also in {seen[info.filename]})")
                        failures += 1
                    seen[info.filename] = path.name
                    blobs += 1
                    total += info.file_size
                bad = bundle.testzip()  # CRC pass over every member
                if bad is not None:
                    print(f"FAIL {path.name}:{bad}: CRC mismatch")
                    failures += 1
        except Exception as exc:  # unreadable shard is a finding, not a crash
            print(f"FAIL {path.name}: unreadable ({exc})")
            failures += 1
    status = "FAILED" if failures else "OK"
    print(f"{status}: {blobs} blobs, {total:,} bytes, {len(shard_paths)} shards, {failures} problems")
    return 1 if failures else 0


def _cmd_convert_parquet(args):
    from .parquet import convert, plan_conversion

    plan = plan_conversion(args.src, columns=args.column, fields=args.field)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            raise BlobPackError("use --yes to accept the extraction plan")
        if input("Extract selected binary fields? [y/N] ").strip().lower() not in {"y", "yes"}:
            return 0
    convert(args.src, args.dst, columns=args.column, fields=args.field, max_pack_bytes=args.max_pack_bytes)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blobpack", description="Pack dataset media into plain STORED zip shards.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pack", help="pack a directory of files into shards")
    p.add_argument("src", help="source directory (keys = relative paths)")
    p.add_argument("dst", help="output pack directory")
    p.add_argument(
        "--ref-base",
        default=None,
        help="pack dir path as seen from the dataset root (default: dst's final path component)",
    )
    p.add_argument(
        "--max-pack-bytes",
        type=int,
        default=4 << 30,
        help="maximum shard file size (default 4 GiB, the classic-zip "
        "ceiling; use smaller for finer transfer granularity)",
    )
    p.set_defaults(func=_cmd_pack)

    p = sub.add_parser("convert-wds", help="convert WebDataset tar shards into packs")
    p.add_argument("src", help="directory of .tar shards (or a single .tar)")
    p.add_argument("dst", help="output pack directory")
    p.add_argument(
        "--ref-base",
        default=None,
        help="pack dir path as seen from the dataset root (default: dst's final path component)",
    )
    p.add_argument(
        "--max-pack-bytes",
        type=int,
        default=4 << 30,
        help="maximum shard file size (default 4 GiB)",
    )
    p.add_argument(
        "--shard-prefix",
        action="store_true",
        help="prefix keys with the source tar's stem (use when keys repeat across tars)",
    )
    p.set_defaults(func=_cmd_convert_wds)

    p = sub.add_parser("convert-lance", help="move a Lance dataset's payload columns into packs")
    p.add_argument("src", help="Lance dataset (.lance directory)")
    p.add_argument("dst", help="output directory (receives media/ and table.lance)")
    p.add_argument("--column", action="append", help="restrict extraction to this column (repeatable)")
    p.add_argument("--ref-base", default=None, help="pack dir path as seen from the dataset root")
    p.add_argument("--max-pack-bytes", type=int, default=None, help="maximum shard file size (default 4 GiB)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    p.set_defaults(func=_cmd_convert_lance)

    p = sub.add_parser("convert-parquet", help="extract selected Parquet binary fields into packs")
    p.add_argument("src", help="Parquet file or directory of Parquet files")
    p.add_argument("dst", help="new output directory (tables/, media/, extraction.json)")
    p.add_argument("--column", action="append", help="literal top-level binary column name (repeatable)")
    p.add_argument(
        "--field", nargs="+", action="append", help="literal field path components, e.g. --field camera bytes"
    )
    p.add_argument("--max-pack-bytes", type=int, default=4 << 30)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=_cmd_convert_parquet)

    p = sub.add_parser("unpack", help="extract all blobs back to files")
    p.add_argument("src", help="pack directory")
    p.add_argument("dst", help="output directory")
    p.set_defaults(func=_cmd_unpack)

    p = sub.add_parser("ls", help="list blobs (size, shard, key)")
    p.add_argument("src", help="pack directory")
    p.set_defaults(func=_cmd_ls)

    p = sub.add_parser("verify", help="at-rest integrity check (STORED + CRC + manifest if present)")
    p.add_argument("src", help="pack directory")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("manifest", help="write a per-shard sha256 manifest")
    p.add_argument("src", help="pack directory")
    p.add_argument("--force", action="store_true", help="rewrite an existing manifest")
    p.set_defaults(func=_cmd_manifest)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BlobPackError as exc:  # a user-facing condition, not a crash
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # a reader hanging up (`blobpack ls | head`) is not an error; it is
        # EPIPE on POSIX and EINVAL on Windows
        if exc.errno not in (errno.EPIPE, errno.EINVAL):
            raise
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())
