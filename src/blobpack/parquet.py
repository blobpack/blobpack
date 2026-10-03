"""Extract explicitly selected binary Parquet fields into Blob Packs."""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from itertools import chain
from pathlib import Path

from . import BlobPackError, PackWriter


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise BlobPackError("install the extra: pip install 'blobpack[parquet]'") from exc
    return pa, pq


def _selection(columns, fields):
    paths = [(column,) for column in columns or ()] + [tuple(path) for path in fields or ()]
    if not paths or any(not path or any(not part for part in path) for path in paths):
        raise BlobPackError("select binary payloads with columns= or fields=")
    if len(paths) != len(set(paths)):
        raise BlobPackError("duplicate field selection")
    return paths


def _field(schema, path, pa):
    current = schema
    for name in path:
        if not isinstance(current, pa.Schema) and not pa.types.is_struct(current):
            raise BlobPackError(f"{path!r}: intermediate field is not a struct")
        if current.names.count(name) != 1:
            raise BlobPackError(f"{path!r}: field must exist and have a unique name")
        field = current.field(name)
        current = field.type
    if not (
        pa.types.is_binary(current) or pa.types.is_large_binary(current) or pa.types.is_fixed_size_binary(current)
    ):
        raise BlobPackError(f"{path!r}: expected a binary field, got {current}")
    return field


def plan_conversion(src, *, columns=None, fields=None):
    """Inspect schemas without reading payloads; field paths are sequences of literal names."""
    pa, pq = _arrow()
    src = Path(src).resolve()
    paths = _selection(columns, fields)
    files = [src] if src.is_file() else sorted(src.rglob("*.parquet"))
    if not files:
        raise BlobPackError(f"{src}: no Parquet files")
    rows = 0
    for file in files:
        if src.is_dir() and not file.resolve().is_relative_to(src):
            raise BlobPackError(f"{file}: source file escapes input directory")
        with pq.ParquetFile(file) as reader:
            for path in paths:
                _field(reader.schema_arrow, path, pa)
            rows += reader.metadata.num_rows
    return {"files": len(files), "rows": rows, "fields": [list(path) for path in paths]}


def _replace(array, path, extract, pa, valid=None, nullable=True):
    present = array.is_valid().to_pylist()
    valid = present if valid is None else [a and b for a, b in zip(valid, present)]
    if not path:
        return pa.array(
            [
                extract(value) if present else (None if nullable else "")
                for value, present in zip(array.to_pylist(), valid)
            ],
            pa.string(),
        )
    index = array.type.get_field_index(path[0])
    children = [array.field(i) for i in range(array.type.num_fields)]
    children[index] = _replace(children[index], path[1:], extract, pa, valid, array.type[index].nullable)
    fields = list(array.type)
    fields[index] = fields[index].with_type(children[index].type)
    return pa.StructArray.from_arrays(children, fields=fields, mask=array.is_null())


def convert(src, dst, *, columns=None, fields=None, max_pack_bytes=4 << 30, batch_size=128):
    """Write table(s), media/ and extraction.json to a new directory, preserving binary bytes.

    Struct leaves are replaced in place; siblings and null masks survive. Original
    Arrow schemas are recorded in extraction.json. Engine-specific HF/Pandas schema
    metadata is archived there rather than attached to the rewritten tables.
    """
    pa, pq = _arrow()
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst.exists() or dst == src or dst.is_relative_to(src) or src.is_relative_to(dst):
        raise BlobPackError("destination must be new and disjoint from the source")
    if batch_size < 1:
        raise BlobPackError("batch_size must be positive")
    plan = plan_conversion(src, columns=columns, fields=fields)
    paths = [tuple(path) for path in plan["fields"]]
    files = [src] if src.is_file() else sorted(src.rglob("*.parquet"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", dir=dst.parent))
    schemas = {}
    try:
        with PackWriter(work / "media", ref_base="media", max_pack_bytes=max_pack_bytes) as packs:
            for file_index, file in enumerate(files):
                relative = Path(file.name) if src.is_file() else file.relative_to(src)
                target = work / "tables" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with pq.ParquetFile(file) as reader:
                    schema = reader.schema_arrow
                    schemas[relative.as_posix()] = base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")
                    metadata = {
                        key: value
                        for key, value in (schema.metadata or {}).items()
                        if key not in (b"huggingface", b"pandas")
                    }
                    offset = 0
                    writer = None
                    try:
                        # Include an empty batch so empty input tables retain the output schema.
                        batches = reader.iter_batches(batch_size=batch_size)
                        first = next(batches, None)
                        for batch in chain(
                            [
                                first
                                if first is not None
                                else pa.RecordBatch.from_arrays(
                                    [pa.array([], type=f.type) for f in schema], schema=schema
                                )
                            ],
                            batches,
                        ):
                            arrays = list(batch.columns)
                            fields_out = list(schema)
                            for field_index, path in enumerate(paths):
                                row = 0

                                def extract(value, file_index=file_index, field_index=field_index, offset=offset):
                                    nonlocal row
                                    key = f"{file_index:06d}/{field_index:04d}/{offset + row:012d}"
                                    row += 1
                                    return packs.add(key, value)

                                index = schema.get_field_index(path[0])
                                arrays[index] = _replace(arrays[index], path[1:], extract, pa)
                                fields_out[index] = fields_out[index].with_type(arrays[index].type)
                            out = pa.Table.from_arrays(arrays, schema=pa.schema(fields_out, metadata=metadata))
                            if writer is None:
                                writer = pq.ParquetWriter(target, out.schema)
                            writer.write_table(out)
                            offset += batch.num_rows
                    finally:
                        if writer is not None:
                            writer.close()
            result = {**plan, "blobs": packs.blobs_written, "bytes": packs.bytes_written, "source_schemas": schemas}
        (work / "extraction.json").write_text(json.dumps(result, indent=2) + "\n")
        os.rename(work, dst)
        return result
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
