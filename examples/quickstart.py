"""Read the live demo dataset three ways.

Runs against https://huggingface.co/datasets/MilkClouds/blobpack-demo
(PASS images, LibriSpeech audio, PushT episode video); nothing is
downloaded up front except one small table and, at the end, one shard:

    pip install "blobpack[remote]" huggingface_hub pyarrow
"""

import zipfile

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from blobpack import PackSet

REPO = "MilkClouds/blobpack-demo"

# 1) streamed by reference: the table holds plain strings, and each blob
#    is one ranged request against the Hub
table = pq.read_table(hf_hub_download(REPO, "images/table/train.parquet", repo_type="dataset")).to_pylist()
with PackSet(f"hf://datasets/{REPO}/images/media") as packs:
    row = table[0]
    image = packs.read(row["image_ref"])
    assert len(image) == row["bytes"]
    assert image == packs.read(f"image/{row['id']:06d}.jpg")  # or by bare key
    print(f"streamed {row['filename']}: {len(image):,} bytes")

# 2) seeked into like a file: a robot episode's mp4 as a bounded seekable
#    object -- a video decoder would seek inside it the same way
with PackSet(f"hf://datasets/{REPO}/robot/media") as packs:
    with packs.open("video/000000.mp4") as clip:
        header = clip.read(12)
        assert header[4:8] == b"ftyp"  # it really is an mp4 container
        clip.seek(-4, 2)
        tail = clip.read()
    print(f"seeked episode video: header {header!r}, last bytes {tail!r}")

# 3) no blobpack at all: a shard is a plain zip archive (the 7 MB robot
#    shard keeps this step's download small)
shard = hf_hub_download(REPO, "robot/media/pack-0000.zip", repo_type="dataset")
plain = zipfile.ZipFile(shard).read("video/000000.mp4")
assert plain[:12] == header and plain[-4:] == tail
print("plain zipfile read matches the streamed bytes")
