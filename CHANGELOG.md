# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org).

## Unreleased

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
