#!/usr/bin/env python3
"""Render the committed benchmark results as figures for the README.

Reads results/*.json and writes plots/*.png, so every figure in the docs
traces back to a measurement file in this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"

BLOBPACK = "#1f6feb"
OTHER = "#8b949e"
ACCENT = "#d29922"

#: every non-blobpack series gets its own color and marker, so a
#: multi-series figure is readable without tracing the legend by order
CASE_STYLE = {
    "loose_random": {"color": "#8b949e", "marker": "o"},
    "lance_random": {"color": "#bf5af2", "marker": "s"},
    "wds_stream": {"color": "#2da44e", "marker": "^"},
    "pack_random": {"color": BLOBPACK, "marker": "o"},
    "pack_stream": {"color": BLOBPACK, "marker": "^"},
    "pack_stream_catalog": {"color": "#0a3069", "marker": "D"},
    "pack_random_catalog": {"color": "#0a3069", "marker": "D"},
    "pack_random_validate_on_open": {"color": "#54aeff", "marker": "v"},
}

LABELS = {
    "loose_random": "loose files",
    "pack_random": "blobpack",
    "pack_random_catalog": "blobpack + catalog",
    "pack_random_validate_on_open": "blobpack, validate_on_open",
    "lance_random": "Lance",
    "wds_stream": "WebDataset",
    "pack_stream": "blobpack (stream)",
    "pack_stream_catalog": "blobpack (stream) + catalog",
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def steady(data: dict, case: str, workers: str = "w8") -> float | None:
    cell = data["runs"].get(case, {}).get(workers, {})
    return None if "error" in cell else cell.get("steady_samples_per_s_median")


def ttfb(data: dict, case: str, workers: str = "w8") -> float | None:
    cell = data["runs"].get(case, {}).get(workers, {})
    return None if "error" in cell else cell.get("time_to_first_batch_s_median")


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d0d7de", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_throughput() -> None:
    """Sustained samples/s at 8 workers, both filesystems."""
    cephfs, nvme = load("dataloader_cephfs.json"), load("dataloader_nvme.json")
    groups = [
        ("random access", ["loose_random", "lance_random", "pack_random"]),
        ("streaming", ["wds_stream", "pack_stream_catalog"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    for ax, (title, cases) in zip(axes, groups):
        names = [LABELS[c] for c in cases]
        positions = range(len(cases))
        width = 0.38
        for offset, (data, label, alpha) in enumerate(
            ((cephfs, "shared filesystem", 1.0), (nvme, "local NVMe", 0.55))
        ):
            values = [steady(data, c) or 0 for c in cases]
            colors = [BLOBPACK if "pack" in c else OTHER for c in cases]
            bars = ax.bar(
                [p + (offset - 0.5) * width for p in positions],
                values,
                width,
                color=colors,
                alpha=alpha,
                label=label,
                edgecolor="white",
            )
            ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("samples/s (sustained, 8 workers)" if not offset else "")
        style(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="upper center")
    fig.suptitle(
        "Training-loader throughput, COCO 40k with JPEG decode (higher is better)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "throughput.png", bbox_inches="tight")
    plt.close(fig)


def plot_scaling() -> None:
    """How each format scales with dataloader workers on shared storage."""
    data = load("dataloader_cephfs.json")
    cases = ["loose_random", "lance_random", "pack_random", "wds_stream", "pack_stream_catalog"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    workers = [1, 4, 8]
    for case in cases:
        values = [steady(data, case, f"w{w}") for w in workers]
        is_pack = "pack" in case
        style_kwargs = CASE_STYLE[case]
        ax.plot(
            workers,
            values,
            linewidth=2.2 if is_pack else 1.4,
            linestyle="-" if is_pack else "--",
            label=LABELS[case],
            **style_kwargs,
        )
    ax.set_xticks(workers)
    ax.set_xlabel("dataloader workers")
    ax.set_ylabel("samples/s (sustained)")
    ax.set_title("Scaling on a shared filesystem", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def plot_startup() -> None:
    """What opening a pack set costs, and what a catalog removes."""
    cephfs = load("dataloader_cephfs.json")
    cases = ["loose_random", "wds_stream", "lance_random", "pack_random_validate_on_open", "pack_random", "pack_random_catalog"]
    values = [ttfb(cephfs, c) or 0 for c in cases]
    colors = [ACCENT if c == "pack_random" else (BLOBPACK if "pack" in c else OTHER) for c in cases]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=200)
    bars = ax.barh([LABELS[c] for c in cases], values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f s", fontsize=8.5, padding=3)
    ax.set_xlabel("time to first batch (s), 40k-member set on shared storage")
    ax.set_title("Deferred validation removes most of the startup; a catalog the rest", fontsize=11)
    ax.set_xlim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#d0d7de", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(PLOTS / "startup.png", bbox_inches="tight")
    plt.close(fig)


def plot_domains() -> None:
    """Per-file read cost across domains: the mechanism, not just the ratio.

    Loose-file reads cost the same order per file across a 125x size range
    (metadata dominates; the exact cost varies with directory layout),
    while pack reads cost bytes -- which is why packing pays off most
    where files are smallest and most numerous. Speech uses the first
    fully cold pass where the result file records passes; a file carrying
    only the median gets an asterisk for the unknown per-pass provenance.
    """
    audio = load("audio_cephfs.json")
    micro = load("bench_results.json")
    coco = micro["variants"]["coco_images"]["cephfs"]
    pusht = micro["variants"]["pusht_frames"]["cephfs"]

    # the first pass is the only fully cold one (fadvise leaves metadata
    # caches warm), and the only number comparable with bench.py's
    audio_seq = audio["reads"].get("sequential_cold_s")
    comparable = audio_seq is not None
    if not comparable:
        audio_seq = audio["reads"]["sequential_median_s"]

    star = "" if comparable else "*"
    domains = ["images\n(COCO 163 KB)", f"speech\n(LibriSpeech 133 KB){star}", "robot frames\n(PushT 1.3 KB)"]
    loose_s = [
        coco["runs"]["seq_loose_cold_s"],
        audio_seq["seq_loose_s"],
        pusht["runs"]["seq_loose_cold_s"],
    ]
    pack_s = [
        coco["runs"]["seq_pack_cold_s"],
        audio_seq["seq_pack_s"],
        pusht["runs"]["seq_pack_cold_s"],
    ]
    items = [coco["runs"]["seq_items"], audio["storage"]["clips"], pusht["runs"]["seq_items"]]
    loose_us = [s / n * 1e6 for s, n in zip(loose_s, items)]
    pack_us = [s / n * 1e6 for s, n in zip(pack_s, items)]
    files = [micro["variants"]["coco_images"]["cephfs"]["build"]["n"], audio["storage"]["clips"], 25_650]
    shards = [coco["build"]["pack_shards"], audio["storage"]["pack_inodes"] - 1, pusht["build"]["pack_inodes"] - 1]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=200)

    positions = range(len(domains))
    width = 0.36
    for offset, (values, label, color) in enumerate(((loose_us, "loose files", OTHER), (pack_us, "blobpack", BLOBPACK))):
        bars = axes[0].bar(
            [p + (offset - 0.5) * width for p in positions], values, width, color=color,
            label=label, edgecolor="white",
        )
        axes[0].bar_label(bars, fmt="%.0f", fontsize=8.5, padding=2)
    axes[0].set_xticks(list(positions))
    axes[0].set_xticklabels(domains)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("us per file, sequential read")
    axes[0].set_title("loose cost is per file; pack cost is per byte", fontsize=11)
    style(axes[0])

    for offset, (values, label, color) in enumerate(((files, "loose files", OTHER), (shards, "blobpack", BLOBPACK))):
        bars = axes[1].bar(
            [p + (offset - 0.5) * width for p in positions], values, width, color=color,
            label=label, edgecolor="white",
        )
        axes[1].bar_label(bars, fmt="%d", fontsize=8.5, padding=2)
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(domains)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.5, max(files) * 6)
    axes[1].set_title("files on disk (blobpack bars = shards)", fontsize=11)
    style(axes[1])

    # a single legend above both panels: the axes have no bar-free region
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Same mechanism across domains, on a shared filesystem", fontsize=12, y=1.02)
    if not comparable:
        fig.text(
            0.01, 0.01,
            "* speech reported as a median of three passes; per-pass provenance not recorded in this result file",
            fontsize=7.5, color="#57606a",
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1) if not comparable else None)
    fig.savefig(PLOTS / "domains.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    plot_throughput()
    plot_scaling()
    plot_startup()
    plot_domains()
    print(f"wrote {len(list(PLOTS.glob('*.png')))} figures to {PLOTS}")


if __name__ == "__main__":
    main()
