"""Zip-format conformance for the direct-offset reader: zip64 extras,
archive prefixes, and forged central-directory metadata."""

import struct
import zipfile

import pytest

from blobpack import CorruptPackError, PackSet, UnsupportedMemberError
from blobpack._zip import PackFile


def test_zip64_local_extras(tmp_path):
    """force_zip64 adds a zip64 extra to the local header; the offset math
    must account for it (local extra length differs from the central one)."""
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    payloads = {f"blob{i:02d}.bin": bytes([i]) * (100 + i) for i in range(20)}
    with zipfile.ZipFile(pack_dir / "pack-0000.zip", "w", zipfile.ZIP_STORED) as bundle:
        for name, data in payloads.items():
            with bundle.open(zipfile.ZipInfo(name), "w", force_zip64=True) as member:
                member.write(data)
    with PackSet(pack_dir) as packs:
        for name, data in payloads.items():
            assert packs.read(name) == data


def test_mixed_zip64_and_classic_members(tmp_path):
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    with zipfile.ZipFile(pack_dir / "pack-0000.zip", "w", zipfile.ZIP_STORED) as bundle:
        info = zipfile.ZipInfo("classic.bin")
        info.compress_type = zipfile.ZIP_STORED
        bundle.writestr(info, b"classic-payload")
        with bundle.open(zipfile.ZipInfo("zip64.bin"), "w", force_zip64=True) as member:
            member.write(b"zip64-payload")
    with PackSet(pack_dir) as packs:
        assert packs.read("classic.bin") == b"classic-payload"
        assert packs.read("zip64.bin") == b"zip64-payload"


def test_archive_prefix(tmp_path):
    """Self-extracting-style archives carry leading junk; zipfile normalizes
    header offsets by the concatenation offset and reads must stay correct."""
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    shard = pack_dir / "pack-0000.zip"
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        info = zipfile.ZipInfo("a.bin")
        info.compress_type = zipfile.ZIP_STORED
        bundle.writestr(info, b"payload-after-prefix")
    shard.write_bytes(b"#!/bin/junk-prefix\n" * 16 + shard.read_bytes())
    with PackSet(pack_dir) as packs:
        assert packs.read("a.bin") == b"payload-after-prefix"


def test_forged_size_reaching_into_central_directory(tmp_path):
    """A forged central-directory size that stays inside the file but
    crosses into the central directory must be rejected, not served."""
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    shard = pack_dir / "pack-0000.zip"
    payload = b"x" * 64
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        info = zipfile.ZipInfo("a.bin")
        info.compress_type = zipfile.ZIP_STORED
        bundle.writestr(info, payload)
    raw = bytearray(shard.read_bytes())
    cd = raw.index(b"PK\x01\x02")
    # forged size crosses into the central directory but stays inside the file,
    # which the old end-of-file bound would have accepted
    forged = len(payload) + 46 + len("a.bin")
    # compressed and uncompressed size fields sit at offsets 20 and 24 of the
    # central-directory entry; CRC (offset 16) is left untouched
    struct.pack_into("<II", raw, cd + 20, forged, forged)
    shard.write_bytes(bytes(raw))
    with pytest.raises(CorruptPackError, match="central directory"):
        PackSet(pack_dir)


def test_truncated_local_header(tmp_path):
    """A shard cut inside a local header must fail cleanly, not with a
    struct error or a short read."""
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    shard = pack_dir / "pack-0000.zip"
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        info = zipfile.ZipInfo("a.bin")
        info.compress_type = zipfile.ZIP_STORED
        bundle.writestr(info, b"y" * 32)
    raw = shard.read_bytes()
    cd = raw.index(b"PK\x01\x02")
    # keep the central directory but overwrite the local header region with a
    # too-short garbage prefix by pointing header_offset past the payload
    forged = bytearray(raw)
    struct.pack_into("<I", forged, cd + 42, len(raw) - 8)  # header_offset field
    shard.write_bytes(bytes(forged))
    with pytest.raises(CorruptPackError):
        PackSet(pack_dir)


def test_overlapping_members_rejected(tmp_path):
    """A forged central directory whose member range reaches into the next
    member would let one key alias another's bytes."""
    path = tmp_path / "overlap.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", b"A" * 64)
        bundle.writestr("b.bin", b"B" * 64)
    data = bytearray(path.read_bytes())
    # grow a.bin: still inside the payload region, but reaching into b.bin
    at = data.rindex(b"PK\x01\x02", 0, data.rindex(b"PK\x01\x02"))
    for field_offset in (20, 24):  # compressed + uncompressed size
        data[at + field_offset : at + field_offset + 4] = (100).to_bytes(4, "little")
    path.write_bytes(bytes(data))
    with pytest.raises(CorruptPackError, match="overlaps"):
        PackFile(path)


def test_strong_encryption_flag_rejected(tmp_path):
    path = tmp_path / "strong.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", b"A" * 32)
    data = bytearray(path.read_bytes())
    for magic in (b"PK\x03\x04", b"PK\x01\x02"):
        at = data.index(magic)
        flag_offset = at + (6 if magic == b"PK\x03\x04" else 8)
        data[flag_offset] |= 0x40
    path.write_bytes(bytes(data))
    with pytest.raises(UnsupportedMemberError, match="encrypted"):
        PackFile(path)


def test_readall_reports_truncated_shard(tmp_path):
    path = tmp_path / "trunc.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", b"A" * 4096)
    shard = PackFile(path)
    try:
        # truncate the payload after indexing; readall must not return short
        with open(path, "r+b") as handle:
            handle.truncate(64)
        with pytest.raises(CorruptPackError, match="short read"):
            shard.open("a.bin").readall()
    finally:
        shard.close()


def test_under_delivering_batch_source_rejected(tmp_path):
    """A source returning fewer header blocks than asked must fail loudly,
    not index a subset of the members."""
    from blobpack._sources import LocalSource

    path = tmp_path / "ok.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", b"A" * 16)
        bundle.writestr("b.bin", b"B" * 16)

    class Stingy(LocalSource):
        def read_batch(self, ranges):
            return super().read_batch(ranges)[:-1]

    with pytest.raises(CorruptPackError, match="of 2 headers"):
        PackFile(Stingy(path))


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "offset, replacement, error, message",
    [
        (0, b"BAD!", CorruptPackError, "bad local header"),
        (6, b"\x01\x00", UnsupportedMemberError, "encrypted"),
        (8, b"\x08\x00", CorruptPackError, "compression method"),
        (26, b"\x04\x00", CorruptPackError, "name length"),
        (30, b"z.bin", CorruptPackError, "name mismatch"),
    ],
)
def test_local_header_checks_agree_for_eager_and_lazy(tmp_path, lazy, offset, replacement, error, message):
    from blobpack._sources import LocalSource

    path = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", b"payload")
    raw = bytearray(path.read_bytes())
    raw[offset : offset + len(replacement)] = replacement
    path.write_bytes(raw)
    source = LocalSource(path)
    source.lazy_validation = lazy
    try:
        if lazy:
            shard = PackFile(source)  # local checks wait until first access
            try:
                with pytest.raises(error, match=message):
                    shard.read("a.bin")
            finally:
                shard.close()
        else:
            with pytest.raises(error, match=message):
                PackFile(source)
    finally:
        source.close()
