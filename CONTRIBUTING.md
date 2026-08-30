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
