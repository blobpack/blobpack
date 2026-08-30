"""Publish a Blob Pack dataset to the Hugging Face Hub and read it back.

Run with the extras this needs:

    pip install "blobpack[hf,remote]" huggingface_hub

The layout published here is the ordinary self-contained one, so consumers
can either stream blobs straight from the Hub over range requests or
download the folder and open it locally; both are shown below.
"""

import os
from pathlib import Path

from datasets import Dataset, load_from_disk
from huggingface_hub import HfApi, snapshot_download

from blobpack import PackSet, PackWriter

REPO_ID = os.environ.get("BLOBPACK_DEMO_REPO", "your-name/blobpack-demo")
root = Path("hub-dataset")


def build() -> None:
    """dataset/ holds a table of references plus the packs they point into."""
    with PackWriter(root / "media", ref_base="media") as writer:
        refs = [writer.add(f"img/{i:06d}.bin", bytes([i % 256]) * 4096) for i in range(200)]
    Dataset.from_dict({"image_ref": refs, "label": [i % 4 for i in range(200)]}).save_to_disk(root / "table")


def publish() -> None:
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)  # upload_folder does not create it
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(root),
        commit_message="Add Blob Pack media and reference table",
    )
    print(f"published https://huggingface.co/datasets/{REPO_ID}")


def read_streaming() -> None:
    """No download: shard indices are parsed once, blobs arrive as ranges."""
    packs = PackSet(f"hf://datasets/{REPO_ID}/media")
    try:
        print("streamed", len(packs.read("img/000000.bin")), "bytes from the Hub")
    finally:
        packs.close()


def read_downloaded() -> None:
    local = Path(snapshot_download(REPO_ID, repo_type="dataset"))
    table = load_from_disk(local / "table")
    with PackSet(local / "media") as packs:
        row = table[0]
        print(row["image_ref"], "->", len(packs.read(row["image_ref"])), "bytes")


if __name__ == "__main__":
    build()
    publish()
    read_streaming()
    read_downloaded()
