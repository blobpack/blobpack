"""Describing where a benchmark ran without naming the machine it ran on.

Results are committed, so they must record the property that explains a
number -- whether reads hit local NVMe or a shared filesystem -- and not
the host, job, or directory that happens to provide it.
"""

from __future__ import annotations

from pathlib import Path

#: mount prefixes that mean "this is a local disk"
_LOCAL_PREFIXES = ("/tmp", "/var/tmp", "/scratch", "/local")


def storage_class(path) -> str:
    """``"local"``, ``"shared"``, or ``"unknown"`` for a working directory."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return "unknown"
    text = resolved.as_posix()
    if text.startswith(_LOCAL_PREFIXES):
        return "local"
    if text.startswith("/mnt/") or text.startswith("/net/"):
        return "shared"
    return "unknown"
