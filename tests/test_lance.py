"""Moving payload columns out of a Lance dataset and into packs."""

import pytest

from blobpack import BlobPackError, PackSet
from blobpack.cli import main as cli_main
from blobpack.lance import convert, plan_conversion

lance = pytest.importorskip("lance")
pa = pytest.importorskip("pyarrow")

ROWS = 24


def build_plain(uri, rows=ROWS):
    schema = pa.schema(
        [pa.field("id", pa.int64()), pa.field("label", pa.string()), pa.field("img", pa.large_binary())]
    )
    payloads = [bytes([i % 256]) * (100 + i) for i in range(rows)]
    table = pa.table(
        {"id": list(range(rows)), "label": [f"l{i % 3}" for i in range(rows)], "img": payloads}, schema=schema
    )
    lance.write_dataset(table, str(uri))
    return payloads


def build_blob_storage(uri, rows=8):
    from lance import blob_array, blob_field

    schema = pa.schema([pa.field("id", pa.int64()), blob_field("vid")])
    payloads = [bytes([i]) * (200 + i) for i in range(rows)]
    table = pa.table({"id": list(range(rows)), "vid": blob_array(payloads)}, schema=schema)
    lance.write_dataset(table, str(uri), data_storage_version="2.2")
    return payloads


def test_plan_measures_candidates_and_selects_none(tmp_path):
    """Lance does not record which column is media, so nothing is assumed."""
    build_plain(tmp_path / "in.lance")
    plan = plan_conversion(tmp_path / "in.lance")
    assert plan.extracted == []
    assert [c.name for c in plan.columns] == ["img"]
    assert plan.scalar_columns == ["id", "label"]
    text = plan.render()
    assert "left in the table" in text
    assert "values," in text  # evidence for the operator


def test_plan_reports_size_and_kind_evidence(tmp_path):
    uri = tmp_path / "in.lance"
    jpeg = b"\xff\xd8\xff" + b"x" * 4000
    digest = bytes(range(32))
    schema = pa.schema([pa.field("img", pa.large_binary()), pa.field("sha", pa.large_binary())])
    lance.write_dataset(pa.table({"img": [jpeg] * 8, "sha": [digest] * 8}, schema=schema), str(uri))
    plan = plan_conversion(uri)
    by_name = {c.name: c for c in plan.columns}
    assert by_name["img"].kinds == ("JPEG",)
    assert by_name["img"].mean_bytes == len(jpeg)
    assert by_name["sha"].kinds == ("unrecognized",)
    assert by_name["sha"].mean_bytes == 32  # obviously not a payload column
    text = plan.render()
    assert "JPEG" in text


def test_plan_selects_named_columns(tmp_path):
    build_plain(tmp_path / "in.lance")
    plan = plan_conversion(tmp_path / "in.lance", columns=["img"])
    assert [c.name for c in plan.extracted] == ["img"]
    assert "SCHEMA CHANGES" in plan.render()


def test_plan_rejects_unknown_column(tmp_path):
    build_plain(tmp_path / "in.lance")
    with pytest.raises(BlobPackError, match="not a binary or blob column"):
        plan_conversion(tmp_path / "in.lance", columns=["nope"])


def test_plan_refuses_dataset_without_payloads(tmp_path):
    lance.write_dataset(pa.table({"id": [1, 2]}), str(tmp_path / "scalars.lance"))
    with pytest.raises(BlobPackError, match="no binary or blob columns"):
        plan_conversion(tmp_path / "scalars.lance")


def test_binary_column_moves_into_packs(tmp_path):
    payloads = build_plain(tmp_path / "in.lance")
    result = convert(tmp_path / "in.lance", tmp_path / "out", columns=["img"])
    assert result["values"] == len(payloads)

    table = lance.dataset(str(result["table"])).to_table()
    assert table.schema.field("img").type == pa.string()
    assert table.column("id").to_pylist() == list(range(len(payloads)))
    assert table.column("label").to_pylist()[:3] == ["l0", "l1", "l2"]
    with PackSet(tmp_path / "out" / "media") as packs:
        for ref, expected in zip(table.column("img").to_pylist(), payloads):
            assert packs.read(ref) == expected


def test_blob_storage_class_moves_into_packs(tmp_path):
    payloads = build_blob_storage(tmp_path / "in.lance")
    plan = plan_conversion(tmp_path / "in.lance", columns=["vid"])
    assert [c.storage for c in plan.extracted] == ["blob"]
    result = convert(tmp_path / "in.lance", tmp_path / "out", columns=["vid"])
    table = lance.dataset(str(result["table"])).to_table()
    with PackSet(tmp_path / "out" / "media") as packs:
        for ref, expected in zip(table.column("vid").to_pylist(), payloads):
            assert packs.read(ref) == expected


def test_scalar_queries_still_work_after_conversion(tmp_path):
    build_plain(tmp_path / "in.lance")
    result = convert(tmp_path / "in.lance", tmp_path / "out", columns=["img"])
    dataset = lance.dataset(str(result["table"]))
    filtered = dataset.to_table(filter="label = 'l0'", columns=["id", "img"])
    assert len(filtered) == ROWS // 3
    with PackSet(tmp_path / "out" / "media") as packs:
        assert all(packs.read(ref) for ref in filtered.column("img").to_pylist())


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    build_plain(tmp_path / "in.lance")
    assert cli_main(["convert-lance", str(tmp_path / "in.lance"), str(tmp_path / "out"), "--dry-run"]) == 0
    assert "Lance dataset" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_cli_blank_column_choice_writes_nothing(tmp_path, monkeypatch, capsys):
    build_plain(tmp_path / "in.lance")
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert cli_main(["convert-lance", str(tmp_path / "in.lance"), str(tmp_path / "out")]) == 1
    assert "aborted" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_cli_asks_which_columns_then_converts(tmp_path, monkeypatch, capsys):
    build_plain(tmp_path / "in.lance")
    answers = iter(["1", "y"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert cli_main(["convert-lance", str(tmp_path / "in.lance"), str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "Which columns hold payloads" in out
    assert cli_main(["verify", str(tmp_path / "out" / "media")]) == 0


def test_cli_yes_requires_explicit_columns(tmp_path, capsys):
    build_plain(tmp_path / "in.lance")
    assert cli_main(["convert-lance", str(tmp_path / "in.lance"), str(tmp_path / "out"), "--yes"]) == 2
    assert "name the columns" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_converts_with_named_column_and_yes(tmp_path):
    build_plain(tmp_path / "in.lance")
    assert (
        cli_main(["convert-lance", str(tmp_path / "in.lance"), str(tmp_path / "out"), "--column", "img", "--yes"]) == 0
    )
    assert cli_main(["verify", str(tmp_path / "out" / "media")]) == 0


def test_two_extracted_columns_of_one_row_share_a_shard(tmp_path):
    """Extraction must write row-major: with column-major order, a row's
    payload columns land far apart and the per-row group cannot keep them
    in one shard."""
    lance = pytest.importorskip("lance")
    uri = tmp_path / "two.lance"
    rows = 24
    payload = lambda seed: bytes([seed]) * 4_000  # noqa: E731
    table = pa.table(
        {
            "id": list(range(rows)),
            "rgb": [payload(i) for i in range(rows)],
            "depth": [payload(100 + i) for i in range(rows)],
        }
    )
    lance.write_dataset(table, str(uri))
    out = tmp_path / "out"
    convert(uri, out, columns=["rgb", "depth"], max_pack_bytes=30_000)

    rewritten = lance.dataset(str(out / "table.lance")).to_table()
    shards_per_row = [
        {ref.split("::")[-1] for ref in pair}
        for pair in zip(rewritten.column("rgb").to_pylist(), rewritten.column("depth").to_pylist())
    ]
    assert all(len(shards) == 1 for shards in shards_per_row), shards_per_row


def test_null_payloads_stay_null(tmp_path):
    lance = pytest.importorskip("lance")
    uri = tmp_path / "nulls.lance"
    table = pa.table({"id": [0, 1, 2], "img": [b"aa", None, b"cc"]})
    lance.write_dataset(table, str(uri))
    out = tmp_path / "out"
    convert(uri, out, columns=["img"])
    column = lance.dataset(str(out / "table.lance")).to_table().column("img").to_pylist()
    assert column[1] is None  # not ""
    assert column[0] and column[2]
