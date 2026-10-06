"""Plot input and output fidelity against the saved Ckt5 v4 measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = Path("results/bnp_circuit_regime_20260917_v4/metrics.json")
DEFAULT_OUTPUT = Path("figures/bnp_ckt5_input_output_baseline")
REGIMES = ((96, 6.0), (24, 3.0))
MECHANISMS = ("none", "bnp", "gaussian")
COLORS = {"none": "#59636d", "bnp": "#c65d2e", "gaussian": "#1f6f8b"}
LABELS = {"none": "No noise", "bnp": "BNP (ε=0, δ=.02)", "gaussian": "Gaussian (ε=1, δ=.02)"}


def load_rows(metrics_path: Path) -> list[dict]:
    """Load saved experiment rows from *metrics_path*."""

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No measurement rows found in {metrics_path}")
    return rows


def aggregate(rows: list[dict], metric: Callable[[dict], float]) -> tuple[float, float, float]:
    """Return mean, minimum, and maximum for a metric over saved seeds."""

    values = np.asarray([metric(row) for row in rows], dtype=float)
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty set of rows")
    return float(values.mean()), float(values.min()), float(values.max())


def make_figure(rows: list[dict], output: Path) -> tuple[Path, Path]:
    """Create the input/output fidelity figure as PNG and SVG."""

    grouped: dict[tuple[int, float], dict[str, list[dict]]] = {}
    for regime in REGIMES:
        T, C = regime
        grouped[regime] = {}
        for mechanism in ("bnp", "gaussian"):
            selected = [row for row in rows if row["T"] == T and float(row["clip_norm"]) == C and row["mechanism"] == mechanism]
            if not selected:
                raise ValueError(f"Missing {mechanism} rows for regime T={T}, C={C}")
            grouped[regime][mechanism] = selected
        grouped[regime]["none"] = grouped[regime]["bnp"]

    metrics = {
        "ratio": lambda row: row["max_noise_to_median_covariance"],
        "covlag": lambda row: row["mean_private_covariance_lag1"],
        "w1": lambda row: row["voltage"]["wasserstein_1"],
        "vlag": lambda row: row["voltage"]["released_lag1"],
    }
    truth_metrics = {
        "covlag": lambda row: row["mean_repaired_true_log_lag1"],
        "vlag": lambda row: row["voltage"]["true_lag1"],
    }

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), dpi=170)
    figure.patch.set_facecolor("white")
    figure.text(0.06, 0.965, "INPUT AND OUTPUT FIDELITY", fontsize=11, weight="bold", color="#2f8f63", va="top")
    figure.text(0.06, 0.925, "Temporal structure is measured at both sides of the power-flow map", fontsize=24, weight="bold", color="#172f5f")
    figure.text(0.06, 0.89, "No-noise bars are the original release. BNP and Gaussian use the saved Ckt5 v4 measurements.", fontsize=10.5, color="#43516a")

    x = np.arange(len(REGIMES))
    width = 0.25
    xlabels = ["15-min\nT=96, C=6", "Hourly\nT=24, C=3"]
    panel_specs = [
        (axes[0, 0], "ratio", "Input perturbation relative to covariance signal", "noise / signal", (0, 1.05)),
        (axes[0, 1], "covlag", "Input covariance temporal structure (lag-1)", "correlation", (0.5, 1.0)),
        (axes[1, 0], "w1", "Output voltage distance from original", "W1 (p.u.)", (0, 0.0032)),
        (axes[1, 1], "vlag", "Output voltage temporal structure (lag-1)", "correlation", (0.55, 0.66)),
    ]
    for axis, key, panel_title, ylabel, ylim in panel_specs:
        for index, mechanism in enumerate(MECHANISMS):
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for regime in REGIMES:
                if mechanism == "none":
                    value = 0.0 if key in {"ratio", "w1"} else aggregate(grouped[regime]["bnp"], truth_metrics[key])[0]
                    means.append(value)
                    lows.append(value)
                    highs.append(value)
                else:
                    mean, low, high = aggregate(grouped[regime][mechanism], metrics[key])
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
            values = np.asarray(means)
            errors = np.vstack((values - lows, highs - values))
            positions = x + (index - 1) * width
            bars = axis.bar(positions, values, width, color=COLORS[mechanism], label=LABELS[mechanism], edgecolor="white", linewidth=0.7, zorder=3)
            if np.any(errors > 1e-12):
                axis.errorbar(positions, values, yerr=errors, fmt="none", ecolor="#4d5663", capsize=3, lw=1, zorder=4)
            offset = ylim[1] * 0.03
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#253247")
        axis.set_title(panel_title, color="#172f5f", pad=8)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, xlabels)
        axis.grid(axis="y", color="#d9e0e8", lw=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.set_ylim(*ylim)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.855), ncol=3, frameon=False, fontsize=9)
    figure.text(0.06, 0.045, "Bars show means across two saved seeds; whiskers show the seed range. No-noise is the original voltage/load release.", fontsize=9, color="#43516a")
    figure.text(0.06, 0.022, "The hourly BNP row is the viable bounded release: input lag-1 change is −0.034 while output lag-1 remains near 0.60.", fontsize=8.5, color="#6d3f35")
    figure.tight_layout(rect=(0.04, 0.075, 0.98, 0.82))

    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    svg_path = output.with_suffix(".svg")
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def main() -> None:
    """Parse arguments and write the input/output fidelity figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    png_path, svg_path = make_figure(load_rows(args.metrics), args.output)
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
