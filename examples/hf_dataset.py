"""A self-contained Hugging Face dataset with blob references.

Requires the `hf` extra: pip install "blobpack[hf]"
Everything lands under dataset/; copy that folder anywhere and it keeps
working, because references are relative to the dataset root.
"""

import os
from pathlib import Path

from datasets import Dataset, load_from_disk

from blobpack import PackSet, PackWriter

root = Path("dataset")

# build: media into packs, references into the table
with PackWriter(root / "media", ref_base="media") as writer:
    refs = [writer.add(f"img/{i:06d}.bin", os.urandom(2048)) for i in range(500)]
Dataset.from_dict(
    {
        "image_ref": refs,
        "label": [i % 10 for i in range(500)],
    }
).save_to_disk(root / "table")

# consume: table via datasets, payloads via one PackSet
ds = load_from_disk(root / "table")
with PackSet(root / "media") as packs:
    row = ds[123]
    payload = packs.read(row["image_ref"])
    print(row["image_ref"], "->", len(payload), "bytes, label", row["label"])
