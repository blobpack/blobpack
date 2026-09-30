# blobpack

[![CI](https://github.com/blobpack/blobpack/actions/workflows/ci.yml/badge.svg)](https://github.com/blobpack/blobpack/actions/workflows/ci.yml) [![PyPI version](https://img.shields.io/pypi/v/blobpack.svg)](https://pypi.org/project/blobpack/) [![Python versions](https://img.shields.io/pypi/pyversions/blobpack.svg)](https://pypi.org/project/blobpack/) [![License](https://img.shields.io/pypi/l/blobpack.svg)](https://github.com/blobpack/blobpack/blob/main/LICENSE)

**Pack your dataset's media into dumb zip shards. Read them fast anywhere.**

A Blob Pack is a directory of uncompressed (STORED) zip shards plus a
convention for referencing their members from annotation tables. It is not
a new format: any zip tool can read it, no library is required to consume a
dataset built with it, and your media bytes are stored unchanged.

![Blob Pack dataset layout](https://raw.githubusercontent.com/blobpack/blobpack/main/docs/layout.svg)

## Install and quickstart

```
pip install blobpack
```

No dependencies in the core. Python >= 3.9.

```python
import os

from blobpack import PackWriter, PackSet

# write: one directory of rolling shards (4 GiB files by default)
with PackWriter("media") as writer:
    refs = [writer.add(f"images/{i:04d}.jpg", os.urandom(1024)) for i in range(1000)]
    # refs[0] == "zip://images/0000.jpg::media/pack-0000.zip"

# read: opens each shard's index once, then one positioned read per blob
with PackSet("media") as packs:
    data = packs.read(refs[0])  # or packs.read("images/0000.jpg")
    batch = packs.read_many(refs, workers=8)  # concurrent, input order

    # big blobs: a bounded seekable file object, nothing loaded up front
    with packs.open(refs[0]) as handle:
        header = handle.read(2)  # hand to a decoder; seeks stay inside this blob

    # epoch pattern: shuffled shards, sequential within each shard,
    # split across dataloader workers by member range
    for key, data in packs.iter_blobs(shuffle_shards=True, seed=0, worker_id=0, num_workers=4):
        ...
```

`writer.add_file(key, path)` streams an existing file in without holding
it in memory. From the shell:

```
blobpack pack   ./images ./media      # files -> shards
blobpack verify ./media               # STORED + unique keys + full CRC pass
blobpack ls     ./media
blobpack unpack ./media ./restored
```

<details>
<summary><b>How it works</b> — reference anatomy, offsets, integrity</summary>

A reference has two parts, and the table stores it as an ordinary string:

```
zip://images/0000.jpg::media/pack-0000.zip
      ^ member name       ^ shard path, relative to the dataset root
```

The relative shard path is what keeps a dataset folder self-contained when
it moves. `PackSet.read` takes a full reference (which names and checks its
shard) or a bare member name, resolved through the set's unique index.

Every member is contiguous and uncompressed, so `PackSet` indexes offsets
once, then serves each blob with one positioned read locally or one range
request on object storage. The fast path skips per-read CRC checks by
design: `blobpack verify` runs the full at-rest pass, and
`blobpack manifest` records a SHA-256 per shard for distribution.

And because a shard is a plain zip archive, blobpack is never required to
get your bytes back:

```
unzip -p media/pack-0000.zip images/0000.jpg > out.jpg
```

</details>

## One shape for images, audio, and robot episodes

Every modality hits the same wall: millions of small files are miserable to
store, copy, and read, while burying the bytes inside a table makes every
random read expensive. Blob Pack separates the two concerns — tables keep
the annotations, packs keep the payloads:

| domain | what a blob is | what it replaces |
|---|---|---|
| vision | one JPEG or PNG per sample | 100K+ loose files, or bytes embedded in parquet |
| audio | one utterance (FLAC/WAV/OPUS) | millions of tiny clip files |
| robotics | one episode video, addressed by time interval | per-frame images, or a bespoke episode layout |

The mechanism is the same everywhere: a loose-file read costs per file
(milliseconds of metadata, whatever the size — the exact cost varies with
filesystem and layout), while a pack read costs per byte. The gap is
therefore widest where files are small and many:

![Per-file cost across domains](https://raw.githubusercontent.com/blobpack/blobpack/main/benchmarks/plots/domains.png)

## Compared with the alternatives

Six things a dataset team ends up needing, in the regime blobpack targets
(many media blobs, shared or object storage) — starting with holding
millions of blobs as tens of files. ✅ out of the box · ⚠️ with extra
machinery or non-default settings · ❌ not available:

| | few files on disk | random access | sequential epochs | plain-tools read | S3 / Hub ranges | keeps your tables |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **blobpack** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| loose files | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| WebDataset | ✅ | ⚠️¹ | ✅ | ✅ | ⚠️¹ | ⚠️² |
| Lance | ✅ | ✅ | ✅ | ❌ | ✅ | ❌³ |
| bytes in parquet | ✅ | ⚠️⁴ | ✅ | ❌ | ⚠️⁴ | ✅ |

<sub>¹ with a sidecar index (e.g. wids) · ² the WebDataset idiom keeps
annotations inside shards, next to each sample · ³ payloads are only
reachable through the Lance engine, and the table itself becomes Lance ·
⁴ row-group granularity; shrinking row groups to approximate random access
trades away parquet's own scan efficiency</sub>

Every alternative either gives something up or needs extra machinery to
get it back; blobpack needs neither, in exchange for one hard scope line:
it holds immutable blobs and lets anything address them, full stop.
Tables, queries, and mutation stay in whatever you already use — Hugging
Face Datasets, parquet, or Lance when you need column queries and
updates. Skip blobpack when data mutates in place, when a pure byte-pipe
stream with no seeks is all you need (tar's one exclusive trick), or for
millions of tiny scalars, which belong in the table.

## Performance

Measured end to end through a torch DataLoader with persistent workers
decoding every JPEG (COCO train2017, 40,000 images):

![Training-loader throughput](https://raw.githubusercontent.com/blobpack/blobpack/main/benchmarks/plots/throughput.png)

On a shared cluster filesystem, blobpack sustains **2.5x loose files** on
random access and edges past WebDataset when streaming — while keeping
random access, which streaming formats give up. On local NVMe every format
converges because JPEG decoding becomes the bottleneck; there the isolated
storage layer serves blobs at 0.035 ms each, 21.5x faster than stock
`zipfile` on the same shards. Scaling with workers is clean:

![Scaling with workers](https://raw.githubusercontent.com/blobpack/blobpack/main/benchmarks/plots/scaling.png)

Startup is a one-time cost, and the [catalog](#very-large-pack-sets)
removes it: a 40,000-member set opens in 0.35 s instead of 12 s. Full
methodology, audio and video results, startup and shard-size sweeps, and
every caveat: [benchmarks/](benchmarks/).

## With Hugging Face Datasets

References are plain strings, so tables need nothing special:

```python
from datasets import Dataset, load_from_disk
from blobpack import PackWriter, PackSet

with PackWriter("dataset/media") as writer:
    refs = [writer.add(f"img/{i:06d}.jpg", img) for i, img in enumerate(images)]
Dataset.from_dict({"image_ref": refs, "label": labels}).save_to_disk("dataset/table")

# ... later, anywhere the folder was copied to:
ds = load_from_disk("dataset/table")
packs = PackSet("dataset/media")
img = packs.read(ds[0]["image_ref"])
```

Everything under `dataset/` is relative, so the folder is self-contained:
move it across machines or clusters and it keeps working.

## Storage and distribution

### Very large pack sets

Opening a pack set parses every shard's index — fine for thousands of
blobs, wasteful for millions. `catalog=True` keeps the mapping in a SQLite
file next to the shards, so later opens cost a few queries:

```python
packs = PackSet("media", catalog=True)  # builds once, reuses afterwards
```

The catalog is a pure cache: it is reused only while every shard keeps the
exact identity recorded at build time, and anything else rebuilds it with
full validation — a shard swapped in place is never served through stale
offsets.

### Object storage

Packs work unchanged on anything fsspec speaks: the index builds from
batched concurrent range reads, each blob is one ranged request, and a
backend that ignores byte ranges is refused rather than silently
downloading whole shards. Protocols need their backend package
(`pip install "blobpack[remote]" s3fs`):

```python
packs = PackSet("s3://bucket/dataset/media", storage_options={"anon": False})
data = packs.read("images/0000.jpg")
# or bring your own configured filesystem: PackSet.from_fs(fs, "bucket/dataset/media")
```

### Publishing on the Hugging Face Hub

A pack dataset is an ordinary folder: publish with `upload_folder`, and
consumers stream blobs straight off the Hub or download as usual
(`pip install "blobpack[remote]" huggingface_hub`):

```python
from huggingface_hub import HfApi
from blobpack import PackSet

REPO = "your-name/your-dataset"
api = HfApi()
api.create_repo(REPO, repo_type="dataset", exist_ok=True)
api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path="dataset")
packs = PackSet(f"hf://datasets/{REPO}/media")  # no download; ranged reads
```

[examples/hf_hub.py](examples/hf_hub.py) runs the whole loop, and a live
three-domain demo — PASS images, LibriSpeech utterances, and PushT
episodes cut into per-episode containers by `convert-lerobot` — is up at
[MilkClouds/blobpack-demo](https://huggingface.co/datasets/MilkClouds/blobpack-demo).

## Migrating from other formats

Existing datasets convert in one command, and nothing is ever re-encoded:
media bytes come out of a conversion exactly as they went into the source.
Three layouts are supported, each behind its own extra
(`pip install "blobpack[lerobot]"` / `"blobpack[lance]"`; WebDataset needs
none):

| source | command | what happens |
|---|---|---|
| WebDataset | `blobpack convert-wds` | tar members stream into indexed zip shards, keys preserved |
| LeRobot v2/v3 | `blobpack convert-lerobot` | videos packet-copied, one container per episode; embedded frames extracted |
| Lance | `blobpack convert-lance` | chosen binary columns move to packs; the Lance table survives |

WebDataset conversion changes nothing but the container, so it just runs.
The other two can rewrite table schemas, so they print a plan of exactly
what each feature or column will become and wait for your confirmation
(`--dry-run` stops at the plan, `--yes` skips the prompt). Per-source
detail folds out below.

<details>
<summary><b>From WebDataset</b> — streaming pass, keys preserved, random access gained</summary>

WebDataset tar shards (the format behind OpenCLIP/LAION, open-diffusion,
and DataComp pipelines) convert in one streaming pass, keys preserved:

```
blobpack convert-wds ./shards ./media                  # *.tar -> packs
blobpack convert-wds ./shards ./media --shard-prefix   # if keys repeat across tars
```

```
before                              after
shards/                             media/
  shard-0000.tar                      pack-0000.zip
    000000.jpg  000000.json             000000.jpg  000000.json
    000001.jpg  000001.json             000001.jpg  000001.json
  shard-0001.tar                      pack-0001.zip
    000002.jpg  000002.json             000002.jpg  000002.json

sequential only, no member index    same bytes, plus a member index:
                                    random reads, sub-shard worker splits
```

Cold sequential reads measured identical for uncompressed tar and packs,
so streaming loses nothing; the conversion adds random access without a
sidecar `.idx`, sub-shard worker splitting, `verify`, and none of tar's
512-byte framing overhead on small blobs.

</details>

<details>
<summary><b>From LeRobot</b> — v2/v3 video packet-copied into per-episode containers; embedded frames extracted</summary>

LeRobot stores camera observations two ways, and they are not the same
conversion — which is exactly what the plan shows before anything is
written.

**Videos keep every packet they started with.** v2.x already stores one
file per episode. v3 concatenates a camera's episodes into one file, and
the converter cuts them apart again so each episode owns its container,
copying packets rather than re-encoding:

```
before  (v3)                              after
dataset/                                  out/
  videos/observation.image/                 media/
    chunk-000/file-000.mp4  <- 206 eps         pack-0000.zip
  data/…  meta/…                                 …/episode_000000.mp4
                                                 …/episode_000001.mp4  <- one per episode
                                            data/…  meta/…  (copied: self-contained)
                                            video_index.json
                                              ep0  -> ref, [0 ns, 16.1 s)
                                              ep1  -> ref, [0 ns, 11.8 s)
```

A cut is only exact where an episode starts on a keyframe, so the converter
probes keyframes first and reports the count; files whose boundaries miss
stay concatenated, with episodes addressed by interval as before.
`--no-split-episodes` keeps the source layout either way.

**Frames embedded in the table are moved out** and the column becomes
references, which is lossless for the bytes but changes the schema:

```
before  (v2.1, frames in parquet)         after
data/episode_000000.parquet               media/pack-0000.zip
  action            float32[]               frames/observation.images.camera_front/…
  observation.…     struct<bytes, path>   data/episode_000000.parquet
                    ^ 31.6 GB of frames     action          float32[]
                                            observation.…   string  <- "zip://frames/…"
```

```
$ blobpack convert-lerobot ./dataset ./out
LeRobot v2.1 - 120 episodes, 120,000 frames @ 50 fps

  observation.images.camera_front    image    120,000 frames inside the table
      -> extracted to media/, column rewritten to blob references
      -> random reads stop pulling a whole row group
          bytes lossless, SCHEMA CHANGES
  data/*.parquet                     120 file(s), 31.6 GB
      -> rewritten
  meta/                              -> copied unchanged

Proceed? [y/N]
```

`--images extract|skip` / `--video copy|skip` decide per media kind, and
the converter refuses to write an index whose intervals overlap or
disagree with the episode lengths.

</details>

<details>
<summary><b>From Lance</b> — you pick the payload columns; the table survives as Lance</summary>

A Lance dataset carrying payloads in a binary column has the same shape as
frames in parquet: the scalar columns stay useful, but the bytes are only
reachable through the table engine. Extraction moves them out and leaves
everything else alone:

```
before                                    after
data.lance                                out/
  id       int64                            table.lance
  label    string                             id      int64
  img      large_binary   <- payloads         label   string
                                              img     string  <- "zip://img/…"
                                            media/pack-0000.zip
                                              img/000000000000.bin
```

Lance records no convention for which column is media, so nothing is
guessed. Candidates are measured and sniffed, and you choose:

```
$ blobpack convert-lance ./data.lance ./out
Lance dataset at data.lance - 500 rows

  image      binary column    500 values, JPEG, 146.5 KB mean
  sha256     binary column    500 values, unrecognized, 32 B mean
  embedding  binary column    500 values, unrecognized, 3.0 KB mean

Which columns hold payloads to extract?
  1) image       500 values, JPEG, 146.5 KB mean (146.5 KB-146.5 KB)
  2) sha256      500 values, unrecognized, 32 B mean (32 B-32 B)
  3) embedding   500 values, unrecognized, 3.0 KB mean (3.0 KB-3.0 KB)
Numbers, comma separated, or 'all' (blank aborts): 1
```

Scripts name them instead: `--column image --yes`. Running `--yes` without
`--column` is an error rather than a guess, since extracting a column of
hashes would be worse than doing nothing.

The output is still a Lance dataset, so filters and column queries keep
working; only the payloads have moved out from under the engine. Both
storage shapes are handled: an ordinary `binary`/`large_binary` column and
Lance's blob storage class.

</details>

## Specification

[SPEC.md](SPEC.md) defines the format contract in one page: STORED zip
shards as the default container, size and count limits, reference syntax,
and immutability, plus the Blob-Referenced Columnar Dataset (BRCD)
convention for tables that reference blobs. This repository is one reference implementation of that
contract. For timestamped media references and decoding, see
[MediaRef](https://github.com/open-world-agents/MediaRef).

## License

Apache-2.0
