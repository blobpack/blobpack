# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org).

## Unreleased

- Name the Blob-Referenced Columnar Dataset convention BRCD in `SPEC.md`; the README introduces a dataset built on
  Blob Packs as a BRCD from its first paragraph.

## 0.1.4

- Add PyPI, Python support, license, and CI badges to the README.
- Declare Python 3.9–3.13 support in package classifiers.
- Rename the test workflow to `ci.yml`, fix release checkout permissions, and streamline agent instructions.

## 0.1.3

Remote reads are round-trip bound (a 4-byte range and a 146 KB payload
cost the same ~0.9 s against the Hub), so the cold path sheds three
requests (#16): the first tail-window read doubles as the
range-capability probe, a member's first read fuses local-header
validation into the payload request, and the directory window grew to
256 KiB so EOCD plus a typical central directory arrive together.

## 0.1.2

Opening a pack set on object storage no longer costs O(shard bytes)
(#11). 0.1.1 bounded each request but kept validating every member's
local header at open; over the Hub that still meant transferring the
whole shard (43 s for a 2,000-member, 250 MB shard).

- On remote sources, open reads central directories only; a member's
  local header is validated on its first read (first read of a member
  costs two ranged requests, later reads one). The integrity model is
  unchanged -- a forged or overlapping member is refused at first read
  instead of at open, and `blobpack verify` remains the at-rest check.
- Shards also open lazily: a full `zip://key::path` reference touches
  only its own shard; the first bare-key read loads the remaining
  directories and enforces cross-shard key uniqueness then.
- A backend that ignores byte ranges is refused at the first actual
  read rather than at open.
- A pre-release performance gate (`make perf`) guards open/read cost
  budgets; request- and byte-count budgets also run in CI. Its first run
  shrank the directory read window from 1 MiB to zipfile's EOCD scan
  bound.

## 0.1.1

Remote reads on synchronous, buffered fsspec backends (the Hugging Face
Hub above all) cost far more than they should: every ranged read went
through a readahead cache that fetched a full block per call, so opening
a 2,000-member shard transferred gigabytes over minutes. Found while
verifying the live demo dataset.

- Dense index batches are served by one sequential sweep over a single
  stream; sparse ranges read concurrently with exact-size requests.
- Payload reads request exactly the bytes they need on such backends.

## 0.1.0

First release. Implements Blob Pack specification 1.0. Before release the
whole repository went through an issue-finding review (13 issues, all
fixed) plus a cross-family review of the fixes themselves; the notable
outcomes are folded into the sections below.

### Format and readers

- `PackWriter` writes STORED zip shards, rolling at 4 GiB or 65,535
  members, and verifies on close that every member it produced is STORED.
  zip64 is forced from `zipfile`'s actual 2 GiB streaming limit, so blobs
  between 2 GiB and 4 GiB write correctly.
- `PackSet` reads them by string reference, building an offset index once
  and serving each blob with a single positioned read. Rejects compressed,
  encrypted (including strong-encryption), overlapping, and out-of-bounds
  members, and refuses short reads and under-delivered batches instead of
  returning partial data.
- Reads from local paths or any fsspec URL, with ranged HTTP requests
  where the server supports them.
- Optional SQLite catalog caches the offset index across processes, keyed
  on shard identity so a changed shard invalidates itself; a remote shard
  with no change marker beyond its size is never trusted, and transient
  reader threads release their connections when they die.
- Safe under `fork` and `spawn`: descriptors are process-owned and
  reopened lazily by dataloader workers.

### Converters

- `convert-wds` from WebDataset tar shards.
- `convert-lerobot` from LeRobot v2.x and v3, covering external video and
  frames embedded in parquet. Video is packet-copied, never re-encoded;
  concatenated per-camera files are cut so each episode owns its container
  where every boundary lands on a keyframe, pre-roll footage is trimmed
  rather than mislabeled, and the output carries `meta/` and untouched
  tables so it is self-contained. Rewritten tables stay loadable through
  `datasets` (feature metadata is updated with the schema).
- `convert-lance` extracts binary and blob-storage-class columns. Lance
  has no convention for which column is media, so candidates are measured
  and the caller chooses.

`convert-lerobot` and `convert-lance` print a plan and wait for
confirmation (`--dry-run` stops at the plan; `--yes` skips the prompt);
`convert-wds` converts directly, since it changes nothing but the container.
