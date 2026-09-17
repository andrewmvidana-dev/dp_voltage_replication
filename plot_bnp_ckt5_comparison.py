"""Plot the saved EPRI Ckt5 BNP/Gaussian comparison.

This script is deliberately measurement-only: it reads the canonical v4
metrics file and never reruns the feeder experiment.  The two seeds are
shown as means with min/max whiskers so the figure summarizes the existing
release without creating a new measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = Path("results/bnp_circuit_regime_20260917_v4/metrics.json")
DEFAULT_OUTPUT = Path("figures/bnp_ckt5_gaussian_comparison")
REGIMES = ((96, 6.0), (24, 3.0))
MECHANISMS = ("bnp", "gaussian")
COLORS = {"bnp": "#c65d2e", "gaussian": "#1f6f8b"}
LABELS = {"bnp": "BNP (ε=0, δ=.02)", "gaussian": "Gaussian (ε=1, δ=.02)"}


def load_rows(metrics_path: Path) -> list[dict]:
    """Load saved experiment rows from *metrics_path*."""

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No measurement rows found in {metrics_path}")
    return rows


def aggregate(rows: Iterable[dict], key: Callable[[dict], float]) -> tuple[float, float, float]:
    """Return mean, minimum, and maximum for a metric over saved seeds."""

    values = np.asarray([key(row) for row in rows], dtype=float)
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty set of rows")
    return float(values.mean()), float(values.min()), float(values.max())


def grouped_panel(
    ax: plt.Axes,
    grouped: dict[tuple[int, float], dict[str, list[dict]]],
    metric: Callable[[dict], float],
    title: str,
    ylabel: str,
    *,
    truth_metric: Callable[[dict], float] | None = None,
    threshold: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draw a mechanism comparison panel with seed-range whiskers."""

    x = np.arange(len(REGIMES))
    width = 0.34
    for index, mechanism in enumerate(MECHANISMS):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for regime in REGIMES:
            mean, low, high = aggregate(grouped[regime][mechanism], metric)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        means_array = np.asarray(means)
        errors = np.vstack((means_array - lows, highs - means_array))
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            means_array,
            width,
            color=COLORS[mechanism],
            edgecolor="white",
            linewidth=0.7,
            label=LABELS[mechanism],
            zorder=3,
        )
        if np.any(errors > 1e-12):
            ax.errorbar(positions, means_array, yerr=errors, fmt="none", ecolor="#4d5663", capsize=3, lw=1, zorder=4)
        label_offset = (ylim[1] if ylim else max(means_array)) * 0.03
        for bar, value in zip(bars, means_array):
            ax.text(bar.get_x() + bar.get_width() / 2, value + label_offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#253247")

    if truth_metric is not None:
        truth = np.asarray([aggregate(grouped[regime]["bnp"], truth_metric)[0] for regime in REGIMES])
        ax.plot(x, truth, "o--", color="#222b39", label="Truth / nonprivate", lw=1.5, ms=5, zorder=5)
        truth_offset = (ylim[1] if ylim else 1.0) * 0.035
        for position, value in zip(x, truth):
            ax.text(position, value + truth_offset, f"{value:.3f}", ha="center", fontsize=8, color="#222b39")
    if threshold is not None:
        ax.axhline(threshold, color="#7f8790", ls=":", lw=1.4, label="Acceptance threshold", zorder=2)

    ax.set_title(title, color="#172f5f", pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, ["15-min\nT=96, C=6", "Hourly\nT=24, C=3"])
    ax.grid(axis="y", color="#d9e0e8", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)


def make_figure(rows: list[dict], output: Path) -> tuple[Path, Path]:
    """Create PNG and SVG comparison figures from saved *rows*."""

    grouped: dict[tuple[int, float], dict[str, list[dict]]] = {}
    for regime in REGIMES:
        grouped[regime] = {}
        for mechanism in MECHANISMS:
            selected = [row for row in rows if (row["T"], float(row["clip_norm"])) == regime and row["mechanism"] == mechanism]
            if not selected:
                raise ValueError(f"Missing {mechanism} rows for regime T={regime[0]}, C={regime[1]}")
            grouped[regime][mechanism] = selected

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), dpi=170)
    figure.patch.set_facecolor("white")
    figure.text(0.06, 0.965, "EPRI CKT5 INPUT-STAGE RESULTS", fontsize=11, weight="bold", color="#2f8f63", va="top", ha="left")
    figure.text(0.06, 0.925, "Hourly BNP preserves temporal structure under a hard bound", fontsize=24, weight="bold", color="#172f5f", ha="left")
    figure.text(0.06, 0.89, "Same δ = 0.02; Gaussian comparison uses ε = 1 while BNP reports ε = 0.", fontsize=10.5, color="#43516a", ha="left")

    grouped_panel(axes[0, 0], grouped, lambda row: row["max_noise_to_median_covariance"], "Parameter noise relative to median covariance", "noise / signal", threshold=1, ylim=(0, 1.05))
    grouped_panel(axes[0, 1], grouped, lambda row: row["mean_private_covariance_lag1"], "Covariance temporal structure (lag-1)", "correlation", truth_metric=lambda row: row["mean_repaired_true_log_lag1"], ylim=(0.5, 1.0))
    grouped_panel(axes[1, 0], grouped, lambda row: row["voltage"]["wasserstein_1"], "Voltage distribution error (Wasserstein-1)", "W1 (p.u.)", ylim=(0, 0.0032))
    grouped_panel(axes[1, 1], grouped, lambda row: row["voltage"]["released_lag1"], "Released voltage temporal structure (lag-1)", "correlation", truth_metric=lambda row: row["voltage"]["true_lag1"], ylim=(0.55, 0.66))

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.855), ncol=3, frameon=False, fontsize=9)
    figure.text(0.06, 0.045, "Bars show mean across two saved seeds; whiskers show the seed range. BNP passes the hourly screen (noise/signal ≤ 1 and covariance lag-1 change ≤ 0.05).", fontsize=9, color="#43516a", ha="left")
    figure.text(0.06, 0.022, "The 15-minute BNP row retains voltage fidelity but fails the covariance lag-1 screen; Gaussian is shown for context, not as a matched-ε comparison.", fontsize=8.5, color="#6d3f35", ha="left")
    figure.tight_layout(rect=(0.04, 0.075, 0.98, 0.82))

    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    svg_path = output.with_suffix(".svg")
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def main() -> None:
    """Parse command-line arguments and write the comparison figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="saved v4 metrics JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output path without extension")
    args = parser.parse_args()
    png_path, svg_path = make_figure(load_rows(args.metrics), args.output)
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
