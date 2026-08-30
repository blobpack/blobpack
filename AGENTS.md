# Agent instructions

Reference implementation of Blob Pack: STORED zip shards + string references.
`SPEC.md` is normative for the format; this repo is normative for nothing
else. Reader performance techniques (offset index, pread) are implementation
concerns and belong in code and README, not in SPEC.md.

## Commands

- `make check` — ruff lint + format check (CI runs the same)
- `make test` — pytest
- `make format` — apply formatting and safe lint fixes

## Rules

- Core stays dependency-free and stdlib-only; anything beyond that goes in
  an extra (`blobpack[hf]`-style).
- Every member a writer produces MUST be STORED; every reader-facing path
  MUST reject non-STORED, encrypted, or out-of-bounds members. Direct-offset
  reads skip per-read CRC by design; `blobpack verify` is the integrity
  tool.
- IO and format-parsing code paths need tests (they fail silently);
  benchmarks/ scripts do not.
- Descriptors are process-owned: never reuse or close one opened by another
  process, keep indices picklable, and leave descriptor state out of
  pickled state so spawned dataloader workers reopen lazily.
- Do not edit `benchmarks/results/*.json`: they are measurement records.
  Provenance fields may be redacted where they name a host or an internal
  path. A measured value may not be touched; a stale one is corrected by
  re-running the benchmark.
- Workflow per the engineering handbook: branch + PR with green CI, squash
  merge, PR title written as the commit message. Do not push to `main`.
- Publishing is the owner's call, never an agent's: do not change the
  repository's visibility, publish to PyPI, or push a dataset anywhere.
