import io
import os
import random
import zipfile

import pytest

from blobpack import (
    BlobPackError,
    NotStoredError,
    PackSet,
    PackWriter,
    make_ref,
    parse_ref,
)
from blobpack.cli import main as cli_main


@pytest.fixture()
def blobs():
    rng = random.Random(0)
    return {f"dir{i % 3}/blob{i:03d}.bin": rng.randbytes(rng.randint(1, 4096)) for i in range(50)}


def write_pack(tmp_path, blobs, **kwargs):
    refs = {}
    with PackWriter(tmp_path / "media", ref_base="media", **kwargs) as writer:
        for key, data in blobs.items():
            refs[key] = writer.add(key, data)
    return refs


def test_roundtrip_by_key_and_ref(tmp_path, blobs):
    refs = write_pack(tmp_path, blobs)
    with PackSet(tmp_path / "media") as packs:
        assert len(packs) == len(blobs)
        for key, data in blobs.items():
            assert packs.read(key) == data
            assert packs.read(refs[key]) == data
            assert key in packs


def test_ref_format(tmp_path, blobs):
    refs = write_pack(tmp_path, blobs)
    key = next(iter(blobs))
    assert refs[key] == f"zip://{key}::media/pack-0000.zip"
    assert parse_ref(refs[key]) == (key, "media/pack-0000.zip")
    assert make_ref(*parse_ref(refs[key])) == refs[key]
    for bad in ("media/a.jpg", "zip://a.jpg", "zip://::media/p.zip"):
        with pytest.raises(ValueError):
            parse_ref(bad)


def test_shard_rolling_and_limits(tmp_path, blobs):
    write_pack(tmp_path, blobs, max_pack_bytes=8192)
    shards = sorted((tmp_path / "media").glob("pack-*.zip"))
    assert len(shards) > 1
    payload = sum(len(v) for v in blobs.values())
    assert len(shards) >= payload // 8192
    with PackSet(tmp_path / "media") as packs:
        assert sorted(packs.keys()) == sorted(blobs)


def test_oversized_blob_becomes_singleton(tmp_path):
    with PackWriter(tmp_path / "media", max_pack_bytes=100) as writer:
        writer.add("small.bin", b"x" * 10)
        writer.add("big.bin", b"y" * 1000)  # exceeds the cap on its own
        writer.add("small2.bin", b"z" * 10)
    with PackSet(tmp_path / "media") as packs:
        assert packs.read("big.bin") == b"y" * 1000
        assert len(packs) == 3


def test_max_blob_count_rolls(tmp_path):
    with PackWriter(tmp_path / "media", max_blob_count=2) as writer:
        for i in range(5):
            writer.add(f"b{i}", b"x")
    assert len(sorted((tmp_path / "media").glob("pack-*.zip"))) == 3


def test_deterministic_output(tmp_path, blobs):
    write_pack(tmp_path / "a", blobs)
    write_pack(tmp_path / "b", blobs)
    for one, two in zip(
        sorted((tmp_path / "a" / "media").glob("*.zip")),
        sorted((tmp_path / "b" / "media").glob("*.zip")),
    ):
        assert one.read_bytes() == two.read_bytes()


def test_key_validation(tmp_path):
    with PackWriter(tmp_path / "media") as writer:
        writer.add("ok/key.bin", b"x")
        for bad in ("/abs.bin", "a/../b.bin", "a//b.bin", "", "a\\b.bin", "./x"):
            with pytest.raises(ValueError):
                writer.add(bad, b"x")
        with pytest.raises(BlobPackError):
            writer.add("ok/key.bin", b"x")  # duplicate


def test_immutability_guard(tmp_path, blobs):
    write_pack(tmp_path, blobs)
    with pytest.raises(BlobPackError):
        PackWriter(tmp_path / "media")


def test_rejects_compressed_members(tmp_path):
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    with zipfile.ZipFile(pack_dir / "pack-0000.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("a.bin", os.urandom(1000) + b"\x00" * 4000)
    with pytest.raises(NotStoredError):
        PackSet(pack_dir, validate_on_open=True)
    with PackSet(pack_dir) as packs, pytest.raises(NotStoredError):  # the shard opens on first touch
        packs.read("a.bin")


def test_duplicate_keys_across_shards_rejected(tmp_path):
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    for name in ("pack-0000.zip", "pack-0001.zip"):
        with zipfile.ZipFile(pack_dir / name, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr(zipfile.ZipInfo("same.bin"), b"x")
    with pytest.raises(BlobPackError):
        PackSet(pack_dir, validate_on_open=True)
    with PackSet(pack_dir) as packs, pytest.raises(BlobPackError):  # bare keys resolve across shards
        packs.read("same.bin")


def test_iter_blobs_epoch(tmp_path, blobs):
    write_pack(tmp_path, blobs, max_pack_bytes=8192)
    with PackSet(tmp_path / "media") as packs:
        plain = dict(packs.iter_blobs())
        shuffled = dict(packs.iter_blobs(shuffle_shards=True, seed=7))
    assert plain == blobs == shuffled


def test_pread_matches_zipfile(tmp_path, blobs):
    write_pack(tmp_path, blobs)
    shard = next((tmp_path / "media").glob("pack-*.zip"))
    with PackSet(tmp_path / "media") as packs, zipfile.ZipFile(shard) as bundle:
        for name in bundle.namelist():
            assert packs.read(name) == bundle.read(name)


def test_cli_roundtrip(tmp_path, blobs, capsys):
    src = tmp_path / "src"
    for key, data in blobs.items():
        path = src / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert cli_main(["pack", str(src), str(tmp_path / "media")]) == 0
    assert cli_main(["verify", str(tmp_path / "media")]) == 0
    assert cli_main(["ls", str(tmp_path / "media")]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "\t" in ln]
    assert len(lines) == len(blobs)
    assert cli_main(["unpack", str(tmp_path / "media"), str(tmp_path / "out")]) == 0
    for key, data in blobs.items():
        assert (tmp_path / "out" / key).read_bytes() == data


def test_cli_verify_catches_deflate(tmp_path):
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    with zipfile.ZipFile(pack_dir / "pack-0000.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("a.bin", b"hello " * 1000)
    assert cli_main(["verify", str(pack_dir)]) == 1


def test_per_shard_file_size_limit(tmp_path, blobs):
    limit = 8192
    write_pack(tmp_path, blobs, max_pack_bytes=limit)
    for shard in sorted((tmp_path / "media").glob("pack-*.zip")):
        with zipfile.ZipFile(shard) as bundle:
            members = len(bundle.infolist())
        # the limit applies to the shard file itself (headers included)
        assert shard.stat().st_size <= limit or members == 1


def test_singleton_shard_holds_only_the_oversized_blob(tmp_path):
    with PackWriter(tmp_path / "media", max_pack_bytes=100) as writer:
        writer.add("small.bin", b"x" * 10)
        writer.add("big.bin", b"y" * 1000)
        writer.add("small2.bin", b"z" * 10)
    memberships = {}
    for shard in (tmp_path / "media").glob("pack-*.zip"):
        with zipfile.ZipFile(shard) as bundle:
            memberships[shard.name] = bundle.namelist()
    (big_shard,) = [names for names in memberships.values() if "big.bin" in names]
    assert big_shard == ["big.bin"]


def test_key_with_ref_separator_rejected(tmp_path):
    with PackWriter(tmp_path / "media") as writer, pytest.raises(ValueError):
        writer.add("a::b.bin", b"x")


def test_intra_shard_duplicate_rejected(tmp_path):
    from blobpack import CorruptPackError

    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    with zipfile.ZipFile(pack_dir / "pack-0000.zip", "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr(zipfile.ZipInfo("same.bin"), b"one")
        bundle.writestr(zipfile.ZipInfo("same.bin"), b"two")
    with pytest.raises(CorruptPackError):
        PackSet(pack_dir, validate_on_open=True)
    with PackSet(pack_dir) as packs, pytest.raises(CorruptPackError):  # the shard opens on first touch
        len(packs)


def test_foreign_ref_rejected(tmp_path, blobs):
    write_pack(tmp_path, blobs)
    with PackSet(tmp_path / "media") as packs:
        key = next(iter(blobs))
        with pytest.raises(KeyError):
            packs.read(f"zip://{key}::other-dataset/media/pack-0000.zip")
        assert packs.read(f"zip://{key}::media/pack-0000.zip") == blobs[key]
        # ref_base="" opts out of the directory check
    with PackSet(tmp_path / "media", ref_base="") as packs:
        assert packs.read(f"zip://{key}::anything/pack-0000.zip") == blobs[key]


def test_unpack_blocks_traversal(tmp_path, capsys):
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    with zipfile.ZipFile(pack_dir / "pack-0000.zip", "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr(zipfile.ZipInfo("../evil.bin"), b"x")
    assert cli_main(["unpack", str(pack_dir), str(tmp_path / "out")]) == 1
    assert not (tmp_path / "evil.bin").exists()


def test_verify_reports_truncated_shard(tmp_path, blobs, capsys):
    write_pack(tmp_path, blobs)
    shard = next((tmp_path / "media").glob("pack-*.zip"))
    shard.write_bytes(shard.read_bytes()[:100])
    assert cli_main(["verify", str(tmp_path / "media")]) == 1
    assert "unreadable" in capsys.readouterr().out


def test_cli_pack_empty_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    assert cli_main(["pack", str(tmp_path / "empty"), str(tmp_path / "media")]) == 2


def test_writer_refuses_dir_with_any_zip(tmp_path):
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    with zipfile.ZipFile(pack_dir / "other.zip", "w") as bundle:
        bundle.writestr(zipfile.ZipInfo("a"), b"x")
    with pytest.raises(BlobPackError):
        PackWriter(pack_dir)


def make_wds_tar(path, samples):
    import io
    import tarfile

    with tarfile.open(path, "w") as tar:
        for name, data in samples.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_convert_wds_roundtrip(tmp_path):
    wds = tmp_path / "wds"
    wds.mkdir()
    a = {"000000.jpg": b"img-a", "000000.json": b"{}", "./000001.jpg": b"img-b"}
    b = {"000002.jpg": b"img-c", "000002.json": b"{2}"}
    make_wds_tar(wds / "shard-000.tar", a)
    make_wds_tar(wds / "shard-001.tar", b)
    assert cli_main(["convert-wds", str(wds), str(tmp_path / "media")]) == 0
    assert cli_main(["verify", str(tmp_path / "media")]) == 0
    with PackSet(tmp_path / "media") as packs:
        assert packs.read("000001.jpg") == b"img-b"  # "./" stripped
        assert packs.read("000002.json") == b"{2}"
        assert len(packs) == 5


def test_convert_wds_collision_hint(tmp_path, capsys):
    wds = tmp_path / "wds"
    wds.mkdir()
    make_wds_tar(wds / "shard-000.tar", {"000000.jpg": b"a"})
    make_wds_tar(wds / "shard-001.tar", {"000000.jpg": b"b"})
    assert cli_main(["convert-wds", str(wds), str(tmp_path / "media")]) == 1
    assert "--shard-prefix" in capsys.readouterr().err


def test_convert_wds_shard_prefix(tmp_path):
    wds = tmp_path / "wds"
    wds.mkdir()
    make_wds_tar(wds / "shard-000.tar", {"000000.jpg": b"a"})
    make_wds_tar(wds / "shard-001.tar", {"000000.jpg": b"b"})
    assert cli_main(["convert-wds", str(wds), str(tmp_path / "media"), "--shard-prefix"]) == 0
    with PackSet(tmp_path / "media") as packs:
        assert packs.read("shard-000/000000.jpg") == b"a"
        assert packs.read("shard-001/000000.jpg") == b"b"


def test_convert_wds_single_tar(tmp_path):
    make_wds_tar(tmp_path / "one.tar", {"x.bin": b"payload"})
    assert cli_main(["convert-wds", str(tmp_path / "one.tar"), str(tmp_path / "media")]) == 0
    with PackSet(tmp_path / "media") as packs:
        assert packs.read("x.bin") == b"payload"


def test_read_many_order_and_equality(tmp_path, blobs):
    refs = write_pack(tmp_path, blobs)
    keys = list(blobs)
    mixed = [refs[k] if i % 2 else k for i, k in enumerate(keys)]
    with PackSet(tmp_path / "media") as packs:
        assert packs.read_many(mixed) == [blobs[k] for k in keys]
        assert packs.read_many(keys, workers=1) == [blobs[k] for k in keys]
        assert packs.read_many([]) == []


def test_iter_blobs_worker_split(tmp_path, blobs):
    write_pack(tmp_path, blobs, max_pack_bytes=8192)
    with PackSet(tmp_path / "media") as packs:
        for num_workers in (1, 2, 3, 7, len(blobs) + 5):
            parts = [
                dict(packs.iter_blobs(worker_id=w, num_workers=num_workers, shuffle_shards=True, seed=3))
                for w in range(num_workers)
            ]
            merged = {}
            for part in parts:
                assert not (merged.keys() & part.keys())  # disjoint
                merged.update(part)
            assert merged == blobs  # complete
        with pytest.raises(ValueError):
            list(packs.iter_blobs(worker_id=2, num_workers=2))
        with pytest.raises(ValueError):
            list(packs.iter_blobs(worker_id=0, num_workers=0))


def test_manifest_roundtrip_and_detection(tmp_path, blobs, capsys):
    write_pack(tmp_path, blobs, max_pack_bytes=8192)
    media = tmp_path / "media"
    assert cli_main(["manifest", str(media)]) == 0
    assert cli_main(["verify", str(media)]) == 0
    # immutable by default
    assert cli_main(["manifest", str(media)]) == 2
    assert cli_main(["manifest", str(media), "--force"]) == 0
    capsys.readouterr()

    # corruption is caught via the manifest
    shard = sorted(media.glob("pack-*.zip"))[0]
    raw = bytearray(shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    shard.write_bytes(bytes(raw))
    assert cli_main(["verify", str(media)]) == 1
    assert "does not match manifest" in capsys.readouterr().out


def test_manifest_missing_and_extra_shards(tmp_path, blobs, capsys):
    write_pack(tmp_path, blobs, max_pack_bytes=8192)
    media = tmp_path / "media"
    assert cli_main(["manifest", str(media)]) == 0
    shards = sorted(media.glob("pack-*.zip"))
    shards[0].rename(media / "pack-9999.zip")  # one missing, one unlisted
    capsys.readouterr()
    assert cli_main(["verify", str(media)]) == 1
    out = capsys.readouterr().out
    assert "listed in manifest but missing" in out
    assert "not listed in manifest" in out


def test_add_file_from_path_and_object(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"a" * 5000)
    (src / "b.bin").write_bytes(b"b" * 1234)
    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        ref_a = writer.add_file("a.bin", src / "a.bin")
        with open(src / "b.bin", "rb") as handle:
            ref_b = writer.add_file("b.bin", handle)
    assert ref_a == "zip://a.bin::media/pack-0000.zip"
    with PackSet(tmp_path / "media") as packs:
        assert packs.read(ref_a) == b"a" * 5000
        assert packs.read(ref_b) == b"b" * 1234


def test_add_file_matches_add_bytes(tmp_path):
    payload = bytes(range(256)) * 40
    (tmp_path / "blob.bin").write_bytes(payload)
    with PackWriter(tmp_path / "one" / "media", ref_base="media") as writer:
        writer.add("blob.bin", payload)
    with PackWriter(tmp_path / "two" / "media", ref_base="media") as writer:
        writer.add_file("blob.bin", tmp_path / "blob.bin")
    one = (tmp_path / "one" / "media" / "pack-0000.zip").read_bytes()
    two = (tmp_path / "two" / "media" / "pack-0000.zip").read_bytes()
    assert one == two  # streaming path stays byte-identical and deterministic


def test_add_file_non_seekable_requires_size(tmp_path):
    class Pipe:
        def __init__(self, data):
            self._buf = io.BytesIO(data)

        def read(self, n=-1):
            return self._buf.read(n)

        def seekable(self):
            return False

    with PackWriter(tmp_path / "media") as writer:
        with pytest.raises(ValueError, match="size is required"):
            writer.add_file("x.bin", Pipe(b"xyz"))
        ref = writer.add_file("x.bin", Pipe(b"xyz"), size=3)
    with PackSet(tmp_path / "media") as packs:
        assert packs.read(ref) == b"xyz"


def test_add_file_size_mismatch_is_reported(tmp_path):
    with PackWriter(tmp_path / "media") as writer, pytest.raises(BlobPackError, match="short or oversized"):
        writer.add_file("x.bin", io.BytesIO(b"1234"), size=99)


def test_add_file_rolls_shards_by_size(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        (src / f"{i}.bin").write_bytes(bytes([i]) * 3000)
    with PackWriter(tmp_path / "media", max_pack_bytes=8192) as writer:
        for i in range(6):
            writer.add_file(f"{i}.bin", src / f"{i}.bin")
    assert len(sorted((tmp_path / "media").glob("pack-*.zip"))) > 1
    with PackSet(tmp_path / "media") as packs:
        assert packs.read("5.bin") == bytes([5]) * 3000


def test_group_keeps_related_blobs_in_one_shard(tmp_path):
    """One episode's cameras should land together, not scatter by size."""
    with PackWriter(tmp_path / "media", max_pack_bytes=4000, group_fill=0.5) as writer:
        for episode in range(6):
            for cam in ("high", "left", "right"):
                writer.add(f"ep{episode}/{cam}.mp4", bytes([episode]) * 500, group=f"ep{episode}")
        assert writer.groups_split == 0
    shards = {}
    for shard in sorted((tmp_path / "media").glob("*.zip")):
        with zipfile.ZipFile(shard) as bundle:
            for name in bundle.namelist():
                shards.setdefault(name.split("/")[0], set()).add(shard.name)
    assert len(shards) == 6
    for episode, used in shards.items():
        assert len(used) == 1, f"{episode} spread across {used}"


def test_oversized_group_is_split_and_counted(tmp_path):
    """A group too large for one shard still spans shards, and says so."""
    with PackWriter(tmp_path / "media", max_pack_bytes=2000) as writer:
        for i in range(6):
            writer.add(f"big/{i}.bin", b"x" * 900, group="one-episode")
        assert writer.groups_split > 0


def test_grouping_does_not_make_a_shard_per_group(tmp_path):
    """Small groups keep sharing a shard; rolling only kicks in when full."""
    with PackWriter(tmp_path / "media", max_pack_bytes=100_000, group_fill=0.5) as writer:
        for episode in range(20):
            writer.add(f"ep{episode}.bin", b"x" * 100, group=f"ep{episode}")
    assert len(sorted((tmp_path / "media").glob("*.zip"))) == 1


def test_group_fill_is_validated(tmp_path):
    with pytest.raises(ValueError, match="group_fill"):
        PackWriter(tmp_path / "media", group_fill=1.5)


def test_blob_between_2_and_4_gib_writes_and_reads(tmp_path):
    """zipfile refuses streamed members past ZIP64_LIMIT (~2 GiB) unless
    zip64 is forced; the old threshold (~4 GiB) made such blobs crash."""

    class Zeros(io.RawIOBase):
        def __init__(self, n):
            self.n, self.p = n, 0

        def readable(self):
            return True

        def readinto(self, b):
            take = min(len(b), self.n - self.p)
            b[:take] = bytes(take)
            self.p += take
            return take

    size = zipfile.ZIP64_LIMIT + 10
    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        ref = writer.add_file("big.bin", Zeros(size), size=size)
    with PackSet(tmp_path / "media") as packs:
        with packs.open(ref) as handle:
            handle.seek(size - 5)
            assert handle.read() == bytes(5)
        assert len(packs) == 1


def test_cli_pack_streams_without_materializing(tmp_path, monkeypatch):
    """pack must go through the streaming path; reading whole payloads
    into memory OOMs on a directory holding a large video."""
    from pathlib import Path

    src = tmp_path / "src"
    src.mkdir()
    (src / "clip.bin").write_bytes(b"x" * 4096)
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("payload materialized")))
    assert cli_main(["pack", str(src), str(tmp_path / "media")]) == 0


def test_cli_reports_blobpack_errors_without_traceback(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"a")
    dst = tmp_path / "media"
    assert cli_main(["pack", str(src), str(dst)]) == 0
    # second run into the same directory: an immutability error, not a crash
    assert cli_main(["pack", str(src), str(dst)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err


def test_writer_prefix_must_be_a_filename(tmp_path):
    with pytest.raises(ValueError, match="filename component"):
        PackWriter(tmp_path / "media", prefix="../escape")


def test_reader_normalizes_ref_base_like_the_writer(tmp_path):
    with PackWriter(tmp_path / "media", ref_base="./media") as writer:
        ref = writer.add("a.bin", b"a")
    with PackSet(tmp_path / "media", ref_base="./media") as packs:
        assert packs.read(ref) == b"a"


def test_writer_close_catches_a_compressed_member(tmp_path, monkeypatch):
    """SPEC.md asks writers to verify STORED after writing; simulate a
    future writer bug by re-compressing the shard behind the writer's back."""
    writer = PackWriter(tmp_path / "media", ref_base="media")
    writer.add("a.bin", b"A" * 512)
    shard = tmp_path / "media" / "pack-0000.zip"
    writer._writer.close()
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("a.bin", b"A" * 512)
    writer._writer = None
    with pytest.raises(NotStoredError, match="written compressed"):
        writer.close()


def test_cli_ls_into_a_closed_pipe_is_not_an_error(tmp_path):
    """`blobpack ls | head` closes stdout early; that must exit quietly."""
    import subprocess
    import sys as _sys

    src = tmp_path / "src"
    src.mkdir()
    # enough members that ls overflows the 64 KiB pipe buffer and blocks,
    # so the hang-up is guaranteed to surface as EPIPE in the child
    for i in range(4000):
        (src / f"member-{i:05d}.bin").write_bytes(b"x")
    assert cli_main(["pack", str(src), str(tmp_path / "media")]) == 0
    proc = subprocess.Popen(
        [_sys.executable, "-m", "blobpack.cli", "ls", str(tmp_path / "media")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdout.read(64)
    proc.stdout.close()  # the `head` side hangs up
    stderr = proc.stderr.read().decode()
    assert proc.wait() == 0, stderr
    assert "Traceback" not in stderr


def test_members_are_validated_on_first_read_unless_validate_on_open(tmp_path, monkeypatch):
    """By default a pack set reads no member headers at open; each member's header is checked on its first read.
    validate_on_open reads them all at open. Both serve the same bytes, also after pickling."""
    import pickle

    from blobpack._sources import LocalSource

    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        refs = [writer.add(f"k{i:03d}.bin", bytes([i]) * (i + 1)) for i in range(40)]
    batches = []
    real = LocalSource.read_batch
    monkeypatch.setattr(
        LocalSource, "read_batch", lambda self, ranges: batches.append(len(ranges)) or real(self, ranges)
    )
    eager = PackSet(tmp_path / "media", validate_on_open=True)
    assert sum(batches) == 40  # one header read per member at open
    batches.clear()
    lazy = PackSet(tmp_path / "media")
    assert lazy.read(refs[7]) == bytes([7]) * 8 == eager.read(refs[7])
    assert sum(batches) == 0
    clone = pickle.loads(pickle.dumps(lazy))
    assert clone.read(refs[39]) == bytes([39]) * 40
    for packs in (eager, lazy, clone):
        packs.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_opens_shards_without_the_parents_lock(tmp_path):
    """A child forked while a parent thread holds the deferred-open lock opens shards with its own lock."""
    import multiprocessing

    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        ref = writer.add("k.bin", b"payload")
    packs = PackSet(tmp_path / "media")
    ctx = multiprocessing.get_context("fork")
    with packs._open_guard():  # a parent thread mid-open at the fork
        child = ctx.Process(target=lambda: os._exit(0 if packs.read(ref) == b"payload" else 1))
        child.start()
        child.join(30)
    if child.is_alive():
        child.kill()
        pytest.fail("the child waited on the parent's lock")
    assert child.exitcode == 0
    packs.close()


def test_a_first_read_finds_a_shard_another_thread_opened_meanwhile(tmp_path):
    """A read that misses an unopened shard while another thread opens it uses that thread's shard."""
    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        ref = writer.add("k.bin", b"payload")
    packs = PackSet(tmp_path / "media")
    raced = []

    class Racing(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if value is None and not raced:
                raced.append(key)
                packs._open_deferred(key)  # the other thread, right after this lookup
            return value

    packs._shards = Racing(packs._shards)
    assert packs.read(ref) == b"payload"
    assert raced
    packs.close()
