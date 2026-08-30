"""Write a pack, read it back three ways."""

import zipfile
from pathlib import Path

from blobpack import PackSet, PackWriter

root = Path("example-dataset")

with PackWriter(root / "media", ref_base="media") as writer:
    refs = [writer.add(f"blobs/{i:04d}.bin", bytes([i % 256]) * (i + 1)) for i in range(1000)]
    print(f"{writer.blobs_written} blobs -> {writer.shards_written} shard(s)")

with PackSet(root / "media") as packs:
    assert packs.read(refs[42]) == packs.read("blobs/0042.bin")

    # epoch pattern: shuffle shard order, read sequentially within each shard
    total = sum(len(data) for _, data in packs.iter_blobs(shuffle_shards=True))
    print(f"epoch read {len(packs)} blobs, {total} bytes")

# no blobpack needed: it is a plain zip
plain = zipfile.ZipFile(root / "media" / "pack-0000.zip").read("blobs/0042.bin")
assert plain == bytes([42]) * 43
print("plain zipfile read matches")
