# Benchmarks

Measurements behind the numbers quoted in the top-level README. Real data,
two blob-size regimes, two filesystems:

- **coco_images**: [COCO train2017](https://cocodataset.org/#download),
  all 118,287 JPEGs, 19.3 GB, avg 163 KB per blob
- **pusht_frames**: [LeRobot PushT](https://huggingface.co/datasets/lerobot/pusht),
  every frame as a 96x96 JPEG, 25,650 blobs, avg 1.3 KB
- filesystems: a shared CephFS filesystem on a GPU cluster, and node-local
  NVMe (`/tmp`); one 16-core node, Python 3.14

Formats compared: loose files, Blob Pack (STORED zip shards, 512 MiB
target), uncompressed tar (with and without an external offset index),
DEFLATE zip, bytes embedded in Parquet (1,000-row groups, uncompressed),
and Lance 10.0 (inline `large_binary` and blob storage class).

Benchmark-only dependencies install with `uv sync --group bench`. Note on
provenance fields: results committed before the identity scrub carry
`work_dir`/`node`; re-runs emit a `storage` class (`local`/`shared`)
instead, so freshly produced files differ from the committed ones in those
fields only.

## Files

| file | contents |
|---|---|
| `bench.py` | main matrix: build costs, storage/inodes, sequential and epoch reads, random access (1/8 threads), decode, folder ops, shard-size sweep, Parquet and Lance baselines |
| `bench_fix.py` | follow-ups: corrected STORED-vs-DEFLATE build, the pread fast path measured on unchanged shards, archive-open cost vs member count, threaded Lance blob consumer |
| `bench_nvme_pread.py` | pread vs stock `zipfile` on node-local NVMe, median of 5 |
| `bench_dataloader.py` | torch DataLoader end to end: startup and steady throughput against loose files, WebDataset and Lance |
| `bench_audio.py` | small-blob storage and read costs on speech audio |
| `appendix_video.py` | per-frame JPEGs vs MP4 container (GOP 30/300) on PushT episodes |
| `results/*.json` | raw outputs of the runs quoted everywhere |

## Headline results (cold, per blob where applicable)

Sequential file-order read of the same 4.3 GB / 26,304-item sample, coco,
shared FS: loose 44 MB/s, pack 734 MB/s, tar 734 MB/s, parquet 487 MB/s,
lance 321 MB/s. For 1.3 KB blobs the loose-vs-pack gap grows to 311x, and
creating the 25,650 loose files cost 3,118 s on the shared FS vs 4 s on
NVMe. Folder-operation ratios extrapolate loose from a 20,000-file subset.

Random access, coco:

| reader | shared FS | NVMe |
|---|---|---|
| loose files, 8 threads | 0.49 ms | 0.08 ms |
| pack, stock `zipfile`, 8 threads | 2.25 ms | 0.74 ms |
| pack, pread fast path, 8 threads | 0.60 ms | 0.035 ms |
| tar + offset index, 8 threads | 0.95 ms | 0.06 ms |
| Parquet embedded, per call | 217 ms | 171 ms |
| Lance, batched take | 1.19 ms | 0.31-0.54 ms |

Other findings: DEFLATE on JPEGs recovered 0.65% of bytes for 55% more
build time; shard sizes 64 MiB-2 GiB were performance-neutral; warm archive
open costs ~1.8 us per member (cache handles); a short-GOP MP4 was 7.6x
smaller than per-frame JPEGs at near-equal sequential decode speed, 64x
slower on single-frame access.

## Training-loader benchmark (`bench_dataloader.py`)

What a training job actually feels, rather than raw file reads: a torch
DataLoader with persistent workers decoding every JPEG. Startup and
throughput are reported separately, because a format's opening cost is
paid once and must not be read as slow reading.

- **time to first batch**: opening the format, paid once per worker
- **steady samples/s**: sustained rate on a second pass, workers already alive

COCO train2017, 40,000 images (6.5 GB), batch 32, median of 3.

**Shared filesystem (CephFS)**

| case | TTFB | 1 worker | 4 workers | 8 workers |
|---|---|---|---|---|
| loose files | 0.21 s | 303 | 1071 | 1702 |
| blobpack, in-memory index | 11.9 s | 442 | 2018 | **4190** |
| blobpack, catalog | 0.35 s | 400 | 1736 | 3594 |
| Lance | 1.21 s | 377 | 1558 | 3289 |
| WebDataset (streaming) | 0.13 s | 513 | 2046 | 3864 |
| blobpack streaming, catalog | 0.26 s | 526 | 2100 | **4167** |

**Node-local NVMe**

| case | TTFB | 1 worker | 4 workers | 8 workers |
|---|---|---|---|---|
| loose files | 0.10 s | 490 | 1959 | 3875 |
| blobpack, in-memory index | 3.49 s | 496 | 1974 | 3934 |
| blobpack, catalog | 0.11 s | 487 | 1942 | 3880 |
| Lance | 0.96 s | 357 | 1487 | 2963 |
| WebDataset (streaming) | 0.10 s | 515 | 2058 | 3936 |
| blobpack streaming, catalog | 0.22 s | 527 | 2098 | **4198** |

![Startup cost and the catalog](plots/startup.png)

Reading these:

- On shared storage blobpack sustains **2.5x** loose files for random access
  and edges past WebDataset for streaming; on local NVMe everything except
  Lance converges, because JPEG decoding becomes the bottleneck.
- The catalog buys startup, not throughput: it removes the 11.9 s index
  build but costs ~14% of random-access throughput (SQLite lookup per read
  instead of a dict hit). Under streaming the difference vanishes, since
  lookups amortize over sequential ranges.
- Startup scales with member count, so it matters for short-lived processes
  and very large sets, and amortizes to nothing across a long run with
  persistent workers.

## Audio benchmark (`bench_audio.py`)

LibriSpeech dev-clean, 2,703 FLAC utterances, mean 133 KB, on CephFS:

| | loose files | blobpack | uncompressed tar |
|---|---|---|---|
| storage vs payload | +0.2% | +0.1% | +1.4% |
| inodes | 2,841 | 2 | 2 |
| full sequential pass (cold) | 5.07 s | 0.77 s | 0.34 s |
| random read, 8 threads | 0.339 ms | 0.350 ms | not addressable |
| decode 500 clips | 1.71 s | 1.49 s | - |
| relocate (rsync) | 4.76 s | 0.26 s | - |

Speech corpora are the small-blob regime at production scale (millions of
utterances); this corpus is small enough that the file-count effects are
directional rather than a scaling measurement. Uncompressed tar wins the
pure sequential pass and cannot address a single utterance without a
side index, which is the trade it makes.

A measurement pitfall this run surfaced, now guarded in both harnesses:
`fadvise(DONTNEED)` cannot drop dirty pages, so a pass measured seconds
after building its files reads this process's own page cache at memory
speed while claiming to be cold (0.11 s instead of 5.07 s). Eviction is
now preceded by `sync`, per-pass values are recorded alongside the
median, and figures use the first fully cold pass.

## Caveats

- Absolute rates differ between runs on a shared cluster; compare within a
  table, not across them. Earlier runs of the same matrix are kept as
  `results/v1_*.json` and `results/v2_*.json`, together with the
  measurement changes that produced them.
- v1 charged each format's index build to a short measurement window with
  non-persistent workers, which made blobpack look ~5x slower than
  WebDataset at streaming. That was the benchmark's configuration, not the
  formats: with workers persistent and startup reported separately, the
  ordering reverses.
- Decode work is included deliberately. Excluding it would overstate every
  storage difference, since decoding dominates on fast local storage.

## Reproducing

```
# sources/: train2017.zip, the lerobot/pusht snapshot, LibriSpeech dev-clean
BENCH_TARGETS=sharedfs=./data,local=/tmp/blobpack-bench python bench.py
BENCH_DIR=. python bench_fix.py
python bench_nvme_pread.py --source-zip sources/train2017.zip \
  --work-dir /tmp/bp-nvme --output results/nvme_pread_results.json
python appendix_video.py

python bench_dataloader.py --source-zip sources/train2017.zip \
  --work-dir /tmp/bp-dl --output results/dataloader_local.json \
  --images 40000 --samples 8000 --workers 1 4 8 --repeats 3
python bench_audio.py --src sources/LibriSpeech \
  --work-dir /tmp/bp-audio --output results/audio_local.json
```

The loader benchmark needs `torch`, `opencv-python-headless`, `pylance` and
`pyarrow`; the audio benchmark needs `soundfile`. Each loader measurement
runs in its own subprocess under `--cell-timeout`, so a crashing or hanging
combination costs one cell instead of the run.
