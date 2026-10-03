import base64
import json

import pytest

from blobpack import BlobPackError, PackSet
from blobpack.cli import main
from blobpack.parquet import convert

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@pytest.mark.parametrize("batch_size", [1, 2, 100])
def test_binary_and_struct_extraction(tmp_path, batch_size):
    source, out = tmp_path / "source.parquet", tmp_path / "out"
    images = pa.array(
        [
            {"bytes": b"png", "path": "a.png"},
            None,
            {"bytes": b"", "path": "empty.png"},
            {"bytes": None, "path": "external.png"},
        ]
    )
    table = pa.table({"id": [0, 1, 2, 3], "observation.image": images, "audio": [b"wav", None, b"", b"end"]})
    table = table.replace_schema_metadata({b"huggingface": b'{"info":{"features":{}}}', b"custom": b"keep"})
    pq.write_table(table, source)
    convert(source, out, columns=["audio"], fields=[("observation.image", "bytes")], batch_size=batch_size)
    result = pq.read_table(out / "tables/source.parquet")
    assert result["id"] == table["id"]
    assert result.schema.metadata == {b"custom": b"keep"}
    schema = json.loads((out / "extraction.json").read_text())["source_schemas"]["source.parquet"]
    assert pa.ipc.read_schema(pa.BufferReader(base64.b64decode(schema))) == table.schema
    relocated = tmp_path / "relocated"
    out.rename(relocated)
    with PackSet(relocated / "media") as packs:
        for before, after in zip(table.to_pylist(), result.to_pylist()):
            assert (packs.read(after["audio"]) if after["audio"] is not None else None) == before["audio"]
            image = before["observation.image"]
            actual = after["observation.image"]
            if image is None:
                assert actual is None
            else:
                assert actual["path"] == image["path"]
                assert (packs.read(actual["bytes"]) if actual["bytes"] is not None else None) == image["bytes"]


def test_directory_empty_tables_and_names(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "partition=1").mkdir(parents=True)
    pq.write_table(pa.table({"a.b": pa.array([], pa.binary())}), src / "empty.parquet")
    pq.write_table(pa.table({"a.b": [b"data"]}), src / "partition=1/data.parquet")
    result = convert(src, dst, columns=["a.b"])
    assert result["rows"] == 1
    assert pq.ParquetFile(dst / "tables/empty.parquet").schema_arrow.field("a.b").type == pa.string()
    assert (dst / "tables/partition=1/data.parquet").exists()


def test_invalid_selection_and_destination_leave_source_untouched(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table({"value": [1], "data": [b"x"]}), src)
    original = src.read_bytes()
    for selection in ([], ["missing"], ["value"], ["data", "data"]):
        with pytest.raises(BlobPackError):
            convert(src, tmp_path / "out", columns=selection)
        assert not (tmp_path / "out").exists()
    with pytest.raises(BlobPackError):
        convert(src, src, columns=["data"])
    assert src.read_bytes() == original


def test_failed_extraction_cleans_partial_output(tmp_path, monkeypatch):
    import blobpack.parquet as module

    src = tmp_path / "src.parquet"
    pq.write_table(pa.table({"data": [b"x"]}), src)

    def fail(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(module.PackWriter, "add", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        convert(src, tmp_path / "out", columns=["data"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src.parquet"]


def test_cli_explicit_fields_and_dry_run(tmp_path):
    src, dst = tmp_path / "src.parquet", tmp_path / "dst"
    pq.write_table(pa.table({"camera": [{"bytes": b"png", "path": "x"}]}), src)
    assert main(["convert-parquet", str(src), str(dst), "--field", "camera", "bytes", "--dry-run"]) == 0
    assert not dst.exists()
    assert main(["convert-parquet", str(src), str(dst), "--yes"]) == 1
    assert main(["convert-parquet", str(src), str(dst), "--field", "camera", "bytes", "--yes"]) == 0


def test_null_parent_with_required_binary_child(tmp_path):
    src = tmp_path / "src.parquet"
    images = pa.StructArray.from_arrays(
        [pa.array([b"hidden", b"visible"])],
        fields=[pa.field("bytes", pa.binary(), nullable=False)],
        mask=pa.array([True, False]),
    )
    pq.write_table(pa.table({"image": images}), src)
    result = convert(src, tmp_path / "out", fields=[("image", "bytes")])
    assert result["blobs"] == 1
    table = pq.read_table(tmp_path / "out/tables/src.parquet")
    assert table["image"][0].as_py() is None
    assert not table.schema.field("image").type.field("bytes").nullable
    with PackSet(tmp_path / "out/media") as packs:
        assert packs.read(table["image"][1].as_py()["bytes"]) == b"visible"
