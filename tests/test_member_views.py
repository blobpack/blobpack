"""Bounded seekable views over single members (PackSet.open)."""

import io
import os
import zipfile

import pytest

from blobpack import PackSet, PackWriter


@pytest.fixture()
def packed(tmp_path):
    payloads = {
        "a.bin": bytes(range(256)) * 8,
        "nested/b.bin": b"second-blob-payload",
        "c.bin": b"",
    }
    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        refs = {key: writer.add(key, data) for key, data in payloads.items()}
    return tmp_path / "media", payloads, refs


def test_open_reads_whole_blob(packed):
    pack_dir, payloads, refs = packed
    with PackSet(pack_dir) as packs:
        for key, data in payloads.items():
            with packs.open(key) as handle:
                assert handle.read() == data
            with packs.open(refs[key]) as handle:
                assert handle.read() == data


def test_seek_and_partial_reads(packed):
    pack_dir, payloads, _ = packed
    data = payloads["a.bin"]
    with PackSet(pack_dir) as packs, packs.open("a.bin") as handle:
        assert handle.seekable() and handle.readable() and not handle.writable()
        assert handle.read(10) == data[:10]
        assert handle.tell() == 10
        handle.seek(100)
        assert handle.read(5) == data[100:105]
        handle.seek(-4, os.SEEK_END)
        assert handle.read() == data[-4:]
        handle.seek(3, os.SEEK_CUR)  # past the end
        assert handle.read() == b""
        handle.seek(0)
        assert handle.read() == data


def test_reads_are_clamped_to_the_member(packed):
    """A view must never reach neighbouring members or archive structures."""
    pack_dir, payloads, _ = packed
    with PackSet(pack_dir) as packs:
        with packs.open("a.bin", buffered=False) as handle:
            assert len(handle.read(len(payloads["a.bin"]) + 4096)) == len(payloads["a.bin"])
            handle.seek(10**9)
            assert handle.read(1024) == b""
        with packs.open("c.bin") as handle:  # empty member
            assert handle.read() == b""


def test_open_rejects_unknown_and_foreign(packed):
    pack_dir, _, _ = packed
    with PackSet(pack_dir) as packs:
        with pytest.raises(KeyError):
            packs.open("missing.bin")
        with pytest.raises(KeyError):
            packs.open("zip://a.bin::elsewhere/pack-0000.zip")
        with pytest.raises(KeyError):
            packs.open("zip://missing.bin::media/pack-0000.zip")


def test_buffered_and_raw_shapes(packed):
    pack_dir, payloads, _ = packed
    with PackSet(pack_dir) as packs:
        with packs.open("a.bin") as buffered:
            assert isinstance(buffered, io.BufferedReader)
            assert buffered.read(4) == payloads["a.bin"][:4]
        raw = packs.open("a.bin", buffered=False)
        assert len(raw) == len(payloads["a.bin"])
        assert raw.name == "a.bin"
        into = bytearray(6)
        assert raw.readinto(into) == 6
        assert bytes(into) == payloads["a.bin"][:6]
        raw.close()
        with pytest.raises(ValueError):
            raw.read(1)


def test_view_feeds_a_nested_reader(tmp_path):
    """A container blob can be parsed in place: here a zip inside a pack."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("inner.txt", b"payload-from-inner-archive")
    with PackWriter(tmp_path / "media") as writer:
        writer.add("container.zip", inner.getvalue())
    with PackSet(tmp_path / "media") as packs:
        handle = packs.open("container.zip")
        with zipfile.ZipFile(handle) as nested:  # seeks inside the blob only
            assert nested.read("inner.txt") == b"payload-from-inner-archive"
        handle.close()


def test_views_are_independent(packed):
    pack_dir, payloads, _ = packed
    data = payloads["a.bin"]
    with PackSet(pack_dir) as packs:
        one = packs.open("a.bin")
        two = packs.open("a.bin")
        one.seek(50)
        assert two.read(4) == data[:4]
        assert one.read(4) == data[50:54]
        one.close()
        two.close()
