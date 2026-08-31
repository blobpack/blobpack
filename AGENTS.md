# Agent instructions

- `SPEC.md` alone defines the format; keep implementation and performance details in code and README.
- Run `make check` and `make test`; use `make format` for safe fixes.
- Keep the core stdlib-only; put third-party integrations behind extras.
- Writers must produce STORED members. Reader paths must reject compressed, encrypted, and out-of-bounds members. Direct reads skip CRC; use `blobpack verify` for integrity.
- Test all I/O and format-parsing changes; benchmark scripts are exempt.
- File descriptors are process-owned. Keep indices picklable and reopen process-bound state lazily after fork or spawn.
- Never alter measurements in `benchmarks/results/*.json`; only redact host/path provenance or rerun the benchmark.
- Work through a branch and green PR; squash-merge using the PR title. Never push directly to `main`.
- Do not change repository visibility or publish packages or datasets.
