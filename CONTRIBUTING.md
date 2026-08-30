# Contributing

Thanks for looking. Blob Pack is a small library with a written
specification, so changes fall into two kinds and they are handled
differently.

## Two kinds of change

**Format changes** touch `SPEC.md` and affect every implementation, so
they start as an issue describing what a reader or writer can no longer
assume. `SPEC.md` is normative; this repository is a reference
implementation, normative for nothing else. Reader techniques (offset
index, positioned reads) are implementation concerns and stay out of it.

**Library changes** are everything else. Open a pull request.

## Working on it

```sh
uv sync --group dev
make check   # ruff lint + format, exactly what CI runs
make test    # pytest
make format  # apply formatting and safe fixes
```

The core is stdlib-only. Anything that needs a third-party package goes
behind an extra, like `blobpack[lerobot]`, and skips cleanly when the
package is absent — tests included.

## Pull requests

Branch, push, open a PR, and let CI go green before asking for review;
the PR title becomes the squashed commit message, so write it as one.
`main` takes no direct pushes.

Code that parses a format or touches IO needs a test, because it fails
quietly when it fails at all. If you are fixing a bug, the test should
fail without your fix — please check that it does, rather than assuming.

Changes on an IO path also need their cost accounted for, not just
their result: the remote tests assert request and byte budgets, and a
correct change that turns one request into N per member is a regression
those budgets exist to catch.

## Releasing

Versioning is hatch-vcs — the tag is the version. Before pushing a
`vX.Y.Z` tag, from the release commit:

```sh
make check test
make perf    # pre-release workload gate; wall-clock, so not in CI
```

`make perf` runs `benchmarks/perf_gate.py`: local open/read/iteration
against loose thresholds, plus a latency-injected fake remote whose
budgets are effectively request- and byte-count budgets. Do not tag if
it fails. Then `git tag vX.Y.Z && git push origin vX.Y.Z`; the release
workflow checks the tag, re-runs CI, builds, and publishes to PyPI.
