"""Move media out of a Lance dataset's columns and into Blob Packs.

Lance is the other place payload bytes end up inside a table. A dataset
carrying images or video in a binary column keeps its scalar columns
useful, but every consumer needs Lance to reach the bytes, and random
reads are answered by the table engine rather than by the filesystem.

This converter extracts those columns into packs and rewrites them to blob
references, leaving every other column exactly as it was. The bytes
survive unchanged; the schema does not, so conversion is planned first and
executed only after the caller agrees.

Two storage shapes are handled: an ordinary ``binary``/``large_binary``
column, and Lance's blob storage class, whose values are descriptors that
must be fetched with ``take_blobs``.

Unlike LeRobot, Lance has no convention for where media lives: a binary
column may hold images, or it may hold hashes, embeddings, or serialized
structs that belong in the table. Nothing here guesses. Candidate columns
are measured and sniffed, the evidence is shown, and the caller says which
to extract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import PackWriter
from ._zip import BlobPackError

BLOB_EXTENSION_PREFIX = "lance.blob"
LOSSLESS_BYTES = "bytes lossless, SCHEMA CHANGES"


def _require_lance():
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise BlobPackError(
            "converting Lance datasets needs pylance; install the extra: pip install 'blobpack[lance]'"
        ) from exc
    return lance


#: enough of a prefix to name the common payload formats
_MAGIC = (
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"GIF8", "GIF"),
    (b"BM", "BMP"),
    (b"fLaC", "FLAC"),
    (b"OggS", "Ogg"),
    (b"\x1a\x45\xdf\xa3", "Matroska/WebM"),
    (b"%PDF", "PDF"),
    (b"PK\x03\x04", "ZIP"),
)


def _sniff(sample: bytes) -> str:
    """Name a payload from its first bytes, for the operator's benefit."""
    if not sample:
        return "empty"
    for prefix, label in _MAGIC:
        if sample.startswith(prefix):
            return label
    if len(sample) > 11 and sample[4:8] == b"ftyp":
        return "MP4/MOV"
    if sample.startswith(b"RIFF") and len(sample) > 12:
        return {b"WEBP": "WebP", b"WAVE": "WAV"}.get(sample[8:12], "RIFF")
    return "unrecognized"


@dataclass
class ColumnPlan:
    name: str
    storage: str  # "binary" or "blob"
    rows: int
    action: str  # "extract" or "keep"
    mean_bytes: int = 0
    min_bytes: int = 0
    max_bytes: int = 0
    kinds: tuple[str, ...] = ()

    @property
    def fidelity(self) -> str:
        return LOSSLESS_BYTES if self.action == "extract" else "unchanged"

    @property
    def evidence(self) -> str:
        kinds = "/".join(self.kinds) if self.kinds else "unknown"
        return f"{self.rows:,} values, {kinds}, {_human(self.mean_bytes)} mean ({_human(self.min_bytes)}-{_human(self.max_bytes)})"


@dataclass
class LancePlan:
    uri: Path
    rows: int
    columns: list[ColumnPlan]
    scalar_columns: list[str]

    @property
    def extracted(self) -> list[ColumnPlan]:
        return [c for c in self.columns if c.action == "extract"]

    def render(self) -> str:
        lines = [f"Lance dataset at {self.uri} - {self.rows:,} rows", ""]
        for column in self.columns:
            kind = "blob storage class" if column.storage == "blob" else "binary column"
            lines.append(f"  {column.name:<28} {kind}")
            lines.append(f"      {'':<4}{column.evidence}")
            if column.action == "extract":
                lines += [
                    "      -> extracted to media/, column rewritten to blob references",
                    f"      {'':<4}{column.fidelity}",
                ]
            else:
                lines.append("      -> left in the table")
        kept = ", ".join(self.scalar_columns) or "none"
        lines += [
            f"  {'other columns':<28} {kept}",
            "      -> copied unchanged",
            "",
            "  The rewritten dataset stays a Lance dataset: scalar queries keep",
            "  working, while payloads move out from under the table engine.",
        ]
        return "\n".join(lines)


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _blob_columns(schema) -> list[tuple[str, str]]:
    import pyarrow as pa

    found = []
    for field in schema:
        arrow_type = field.type
        name = getattr(arrow_type, "extension_name", "")
        if name and str(name).startswith(BLOB_EXTENSION_PREFIX):
            found.append((field.name, "blob"))
        elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            found.append((field.name, "binary"))
    return found


def _measure(dataset, column: str, storage: str, rows: int, sample: int) -> dict:
    """Sample a column so the operator can tell payloads from small values."""
    if rows == 0:
        return {"mean_bytes": 0, "min_bytes": 0, "max_bytes": 0, "kinds": ()}
    step = max(rows // sample, 1)
    indices = list(range(0, rows, step))[:sample]
    payloads = (
        _read_blob_column(dataset, column, indices)
        if storage == "blob"
        else [v.as_py() for v in dataset.take(indices, columns=[column]).column(column)]
    )
    sizes = [len(p) for p in payloads if p is not None]
    kinds = sorted({_sniff(p[:16]) for p in payloads if p})
    if not sizes:
        return {"mean_bytes": 0, "min_bytes": 0, "max_bytes": 0, "kinds": ("empty",)}
    return {
        "mean_bytes": sum(sizes) // len(sizes),
        "min_bytes": min(sizes),
        "max_bytes": max(sizes),
        "kinds": tuple(kinds[:3]),
    }


def plan_conversion(uri: Path, *, columns: list[str] | None = None, sample: int = 32) -> LancePlan:
    """Describe which columns would move into packs, without writing.

    ``columns=None`` measures the candidates and selects none: Lance does
    not say which column is media, so the choice belongs to the caller.
    """
    lance = _require_lance()
    dataset = lance.dataset(str(uri))
    candidates = _blob_columns(dataset.schema)
    if not candidates:
        raise BlobPackError(f"{uri} has no binary or blob columns; nothing to extract")
    if columns:
        known = {name for name, _ in candidates}
        unknown = sorted(set(columns) - known)
        if unknown:
            raise BlobPackError(f"not a binary or blob column: {unknown[0]} (candidates: {sorted(known)})")
    rows = dataset.count_rows()
    selected = set(columns) if columns is not None else set()
    plans = []
    for name, storage in candidates:
        stats = _measure(dataset, name, storage, rows, sample)
        plans.append(
            ColumnPlan(
                name,
                storage,
                rows,
                "extract" if name in selected else "keep",
                **stats,
            )
        )
    if columns is not None and not any(c.action == "extract" for c in plans):
        raise BlobPackError("no column selected for extraction; there is nothing to convert")
    scalars = [f.name for f in dataset.schema if f.name not in {name for name, _ in candidates}]
    return LancePlan(uri=Path(uri), rows=rows, columns=plans, scalar_columns=scalars)


def _read_blob_column(dataset, column: str, offsets: list[int]) -> list[bytes]:
    """Fetch blob-storage-class values, which read as descriptors otherwise."""
    try:
        handles = dataset.take_blobs(column, indices=offsets)
    except TypeError:  # older signature took indices first
        handles = dataset.take_blobs(offsets, column)
    payloads = []
    for handle in handles:
        try:
            payloads.append(handle.read())
        finally:
            close = getattr(handle, "close", None)
            if close is not None:
                close()
    return payloads


def convert(
    uri: Path,
    dst: Path,
    *,
    plan: LancePlan | None = None,
    columns: list[str] | None = None,
    ref_base: str | None = None,
    max_pack_bytes: int | None = None,
    batch_size: int = 512,
) -> dict:
    """Extract the planned columns into packs and write a rewritten dataset."""
    lance = _require_lance()
    import pyarrow as pa

    plan = plan or plan_conversion(uri, columns=columns)
    extract = [c.name for c in plan.extracted]
    blob_like = {c.name: c.storage for c in plan.columns}
    dataset = lance.dataset(str(uri))
    keep = [f.name for f in dataset.schema if f.name not in extract]

    dst.mkdir(parents=True, exist_ok=True)
    stats = {"values": 0, "bytes": 0, "batches": 0}
    writer_kwargs = {} if max_pack_bytes is None else {"max_pack_bytes": max_pack_bytes}

    with PackWriter(dst / "media", ref_base=ref_base or "media", **writer_kwargs) as writer:

        def batches():
            offset = 0
            scanner = dataset.scanner(columns=keep, batch_size=batch_size)
            for batch in scanner.to_batches():
                arrays = list(batch.columns)
                names = list(batch.schema.names)
                indices = list(range(offset, offset + batch.num_rows))
                payloads = {}
                for column in extract:
                    if blob_like[column] == "blob":
                        payloads[column] = _read_blob_column(dataset, column, indices)
                    else:
                        table = dataset.take(indices, columns=[column])
                        payloads[column] = [v.as_py() for v in table.column(column)]
                # write row-major: a row's payload columns are adjacent, so
                # the per-row group keeps them in one shard
                refs = {column: [] for column in extract}
                for position, index in enumerate(indices):
                    for column in extract:
                        payload = payloads[column][position]
                        if payload is None:
                            refs[column].append(None)  # null stays null
                            continue
                        refs[column].append(writer.add(f"{column}/{index:012d}.bin", payload, group=f"row-{index}"))
                        stats["values"] += 1
                        stats["bytes"] += len(payload)
                for column in extract:
                    names.append(column)
                    arrays.append(pa.array(refs[column], pa.string()))
                offset += batch.num_rows
                stats["batches"] += 1
                yield pa.record_batch(arrays, names=names)

        schema = pa.schema(
            [dataset.schema.field(name) for name in keep] + [pa.field(column, pa.string()) for column in extract]
        )
        lance.write_dataset(batches(), str(dst / "table.lance"), schema=schema)
        blobs, blob_bytes, shards = writer.blobs_written, writer.bytes_written, writer.shards_written

    return {
        "blobs": blobs,
        "bytes": blob_bytes,
        "shards": shards,
        "columns": extract,
        "rows": plan.rows,
        "table": dst / "table.lance",
        **stats,
    }
