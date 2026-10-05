# blobpack

[![CI](https://github.com/blobpack/blobpack/actions/workflows/ci.yml/badge.svg)](https://github.com/blobpack/blobpack/actions/workflows/ci.yml) [![PyPI version](https://img.shields.io/pypi/v/blobpack.svg)](https://pypi.org/project/blobpack/) [![Python versions](https://img.shields.io/pypi/pyversions/blobpack.svg)](https://pypi.org/project/blobpack/) [![License](https://img.shields.io/pypi/l/blobpack.svg)](https://github.com/blobpack/blobpack/blob/main/LICENSE)

**Blob Packs and the BRCD layout: media in dumb zip shards, referenced from any table. Read them fast anywhere.**

A Blob Pack is a directory of uncompressed (STORED) zip shards. A dataset
built on it is a **Blob-Referenced Columnar Dataset (BRCD)**: columnar
tables hold the structured data, and each media blob stays in a pack,
referenced from a table cell by a plain string. Neither is a new format: any
zip tool can read the packs, any table library can read the tables, no
library is required to consume a BRCD, and your media bytes are stored
unchanged.

![BRCD layout: a table of blob references beside a Blob Pack](https://raw.githubusercontent.com/blobpack/blobpack/main/docs/layout.svg)

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

# read: a shard opens on first touch, then one positioned read per blob
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
request on object storage. Before a member's bytes are served its local
header is checked against the central directory, on the member's first read
and next to the payload it guards; `PackSet(..., validate_on_open=True)`
checks every member at open instead. The fast path skips per-read CRC checks
by design: run `blobpack verify` (the full at-rest pass) after writing or
copying a pack, and `blobpack manifest` records a SHA-256 per shard for
distribution.

And because a shard is a plain zip archive, blobpack is never required to
get your bytes back:

```
unzip -p media/pack-0000.zip images/0000.jpg > out.jpg
```

</details>

## One shape for images, audio, and robot episodes

Every modality hits the same wall: millions of small files are miserable to
store, copy, and read, while burying the bytes inside a table makes every
random read expensive. A BRCD separates the two concerns — tables keep
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

On a shared cluster filesystem, blobpack sustains **2.2x loose files** on
random access and edges past WebDataset when streaming — while keeping
random access, which streaming formats give up. On local NVMe every format
converges because JPEG decoding becomes the bottleneck; there the isolated
storage layer serves blobs at 0.035 ms each, 21.5x faster than stock
`zipfile` on the same shards. With workers, blobpack's random access scales as
the streaming formats do, while loose files flatten on shared storage:

![Scaling with workers](https://raw.githubusercontent.com/blobpack/blobpack/main/benchmarks/plots/scaling.png)

Startup is small by default: members are validated on first read, so a
40,000-member set on shared storage gives its first batch in 0.6 s (11 s
with `validate_on_open=True`). Full
methodology, audio and video results, startup and shard-size sweeps, and
every caveat: [benchmarks/](benchmarks/).

## With Hugging Face Datasets

Hugging Face Datasets is one table interface for BRCD; other table libraries
can store the same references. References are plain strings, so tables need
nothing special:

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

A bare-key read (or `len`, `keys`) on a newly constructed `PackSet` parses
every shard's index — fine for thousands of blobs, wasteful for millions. Read by the
references `PackWriter` returns instead: a reference names its shard, so a
process opens only the shards it reads.

```python
packs.read("zip://images/0000.jpg::media/pack-0000.zip")  # opens pack-0000.zip alone
```

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
episodes — is up at
[MilkClouds/blobpack-demo](https://huggingface.co/datasets/MilkClouds/blobpack-demo).

## Migrating from other formats

Existing datasets convert in one command, and nothing is ever re-encoded:
media bytes come out of a conversion exactly as they went into the source.
Three layouts are supported, each behind its own extra
(`pip install "blobpack[parquet]"` / `"blobpack[lance]"`; WebDataset needs
none):

| source | command | what happens |
|---|---|---|
| WebDataset | `blobpack convert-wds` | tar members stream into indexed zip shards, keys preserved |
| Parquet | `blobpack convert-parquet` | selected binary columns or struct leaves move to packs; other fields stay in the table |
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
<summary><b>From Parquet</b> — explicitly selected binary fields become blob references</summary>

```sh
blobpack convert-parquet ./data ./out --column audio --yes
blobpack convert-parquet ./data ./out --field observation.image bytes --yes
```

`--column` selects a literal top-level binary column; `--field` takes literal
struct-path components. Dots in names are not separators. The second example
replaces only the `bytes` child with a string reference, preserving `path`,
other fields and null masks. No media is decoded and external paths are not
followed. Every selected field must exist and be binary in every input file.
Use `--dry-run` to inspect selected fields and row counts without writing.

Output contains `tables/` (the input Parquet file layout), `media/`, and
`extraction.json`. Non-Parquet files are not copied. Batches contain at most
128 rows; individual large payloads can exceed ordinary memory budgets.
The destination must be new and disjoint from the source; failures remove
partial output.

The receipt archives each original Arrow schema, including metadata, as base64
Arrow IPC. HF/Pandas schema metadata is omitted from rewritten tables because
it describes a different physical representation. Other schema metadata is
retained. Applications read the output's explicit schema and string references.

Dataset-level robot interpretation and BRCD conversion use Epishelf's common
`epishelf convert` path. Column extraction does not interpret episode boundaries,
clocks or actions.

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
and immutability, plus the BRCD layout for tables that reference
blobs. This repository is one reference implementation of that contract. For timestamped media references and decoding, see
[MediaRef](https://github.com/open-world-agents/MediaRef).

## License

Apache-2.0
