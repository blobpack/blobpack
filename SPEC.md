# Blob Storage

Specification version 1.0 (2026-08-28). The key words MUST, SHOULD, MAY are
to be interpreted as described in RFC 2119.

- **Blob**: An opaque byte sequence, such as an image, video, audio file, document, or tensor.
- **Blob Reference**: An fsspec-compatible URI reference identifying one blob.
  - Examples include `media/pack/images/a.jpg` and `zip://images/a.jpg::media/pack.zip`.
- **Blob Pack**: A collection of blobs managed as one storage unit, with each blob identified by a Blob Reference.

## Blob Pack Specification

- A pack SHOULD remain unchanged while referenced by a dataset version.
- **Pack Limits**: Writers SHOULD limit both pack size and blob count.
  - The defaults SHOULD be `max_pack_bytes = 4 GiB` and `max_blob_count = 65,535`.
  - An oversized blob MAY form a singleton pack.
  - These values use the [classic ZIP limits](https://en.wikipedia.org/wiki/ZIP_%28file_format%29#ZIP64) as conservative ceilings and are tunable defaults rather than format constraints.
  - Measured read performance was insensitive to pack size between 64 MiB and 2 GiB, so the size cap is an operational choice; packs well below the ceiling (e.g. 512 MiB) MAY be used for finer transfer and retry granularity.
- **Pack Container Format**: ZIP_STORED SHOULD be the default for portable random access.
  - A directory MAY be used when many-small-file behavior is acceptable.
  - Uncompressed TAR MAY be used for primarily sequential access.
  - Other formats MAY be used if accessible through fsspec.
  - Blobs with internal redundancy (e.g. video frames) SHOULD be kept as a single media-container blob (MP4/MKV, re-encoded for seeking) addressed via PTS intervals, not exploded into per-frame blobs; compression belongs to the inner container, which is why the pack itself stays uncompressed.
- **Blob Placement**: Writers SHOULD keep blobs that are read together (one episode's cameras, one sample's modalities) within the same pack, preferring a pack boundary between such groups over one inside them, subject to the size and count limits. A group too large for one pack still spans packs, and a writer SHOULD report when that happens. This is a placement policy only; readers are unaffected and existing packs remain valid.
- **Writer Validation**: Writers SHOULD verify every member's compression method is STORED after writing.
  - Python's `zipfile` takes the method from a passed `ZipInfo` (whose default is STORED) but from the archive-level default otherwise, so a mixed-path writer can silently produce members that break direct-offset reads.
- **Integrity**: Dataset-level at-rest checksums SHOULD cover packs; per-read CRC verification by readers is OPTIONAL.

## Blob-Referenced Columnar Dataset Convention

- **Definition**: A Blob-Referenced Columnar Dataset (BRCD) stores structured data in columnar tables and opaque payloads in Blob Packs, referenced from table cells using Blob References.
- **Principles**:
  - **Separation**: Typed scalars, lists, structs, and ordinary text remain inline, while large opaque payloads are stored in Blob Packs.
  - **Self-Containment**: A self-contained dataset version remains readable and relocatable without data dependencies outside its dataset root.
    - Relative Blob References resolve against the dataset root, defined as the base URI from which the dataset is opened.
    - Absolute Blob References introduce external dependencies and make the dataset non-self-contained.
  - **Interoperability**: Tables and references use established ecosystem interfaces instead of dataset-specific storage APIs.
  - **Media Correctness**: Timestamped media references follow PTS-interval playback semantics.
- **Canonical Representation**:
  - Use Hugging Face Datasets as the table interface.
  - Store Blob References as string-valued features such as `Value("string")` or sequences thereof.
  - Use [MediaRef](https://github.com/open-world-agents/MediaRef) with [TorchCodec](https://github.com/meta-pytorch/torchcodec) for media references and decoding.
    - Ad hoc implementations such as LeRobot's risk mapping timestamps to the wrong frames; see [playback semantics](https://github.com/open-world-agents/MediaRef/blob/main/docs/playback_semantics.md).
- **Rationale**: Columnar operations avoid loading blob payloads, while Blob Packs reduce small-file pressure without sacrificing direct access.

## Note on Recording Formats

- **Principle**: Recording and training impose conflicting requirements (append-only low-overhead capture vs. random access and dataset-wide queries); no single format solves both. Use a recording-specialized format for recording, then convert once for training.
  - The [OWA data pipeline](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-data) is a good example of this conversion.
- **Recording**: An MCAP-like per-episode log, with video in a separate container encoded for capture (long keyframe interval, e.g. 30 s).
- **Training**: A training-specialized format supporting random access and dataset-wide queries, with video re-encoded for seeking (short keyframe interval, e.g. 30 frames / 0.5 s); the [BRCD convention](#blob-referenced-columnar-dataset-convention) above is one such format.
