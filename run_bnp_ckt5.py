"""Measure BNP and Gaussian fidelity on the public EPRI Ckt5 feeder.

This is a circuit-level experiment. A protected record is a load-object-day;
the report makes no customer-population claim. Existing result directories are
never read or overwritten. Run ``python get_ckt5.py`` first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dpvolt.bnp_setup import feeder_load_ratings
from dpvolt.experiments import ansi_violation_rate, voltage_wasserstein
from dpvolt.loads import (
    LoadModel,
    assign_classes,
    fit_load_model,
    make_historical,
    reactive_from_active,
    sample_loads,
)
from dpvolt.powerflow import PowerFlowRunner
from dpvolt.privacy import analytic_gaussian_sigma, bnp_fit_class, dp_fit_class


ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "feeders" / "ckt5" / "Master_ckt5.dss"
DELTA = 0.02
GAUSSIAN_EPSILON = 1.0
HISTORICAL_DAYS = 365
EVALUATION_DAYS = 4
SEEDS = (2026091701, 2026091702)
REGIMES = ((96, 6.0), (24, 3.0))


def lag_one(values: np.ndarray) -> float:
    """Return the mean lag-1 correlation across rows of a time-series array."""
    rows = values.reshape(-1, values.shape[-1])
    correlations = []
    for row in rows:
        if np.std(row[:-1]) > 1e-12 and np.std(row[1:]) > 1e-12:
            correlations.append(np.corrcoef(row[:-1], row[1:])[0, 1])
    return float(np.mean(correlations))


def covariance_lag_one(covariance: np.ndarray) -> float:
    """Return the normalized first temporal covariance diagonal."""
    diagonal = float(np.mean(np.diag(covariance)))
    first_off_diagonal = float(np.mean(np.diag(covariance, k=1)))
    return first_off_diagonal / max(diagonal, 1e-15)


def repair_covariance(covariance: np.ndarray, floor_ratio: float = 0.1) -> np.ndarray:
    """Apply the existing public eigenvalue floor without changing its value."""
    floor = floor_ratio * max(float(np.trace(covariance)) / covariance.shape[0], 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def fit_private(
    archive: np.ndarray,
    classes: dict[int, np.ndarray],
    reference: LoadModel,
    theta: np.ndarray,
    clip_norm: float,
    mechanism: str,
    rng: np.random.Generator,
) -> tuple[LoadModel, dict[int, dict[str, float]]]:
    """Fit a private model and return per-class calibration details."""
    means, covariances, details = {}, {}, {}
    for label, members in classes.items():
        logs = np.log(archive[members].reshape(-1, reference.T))
        lo = float(np.log(reference.p_min[label]))
        hi = float(np.log(reference.p_max[label]))
        if mechanism == "bnp":
            mean, covariance, report = bnp_fit_class(
                logs, lo, hi, DELTA, rng, clip_norm=clip_norm,
                eig_floor_ratio=0.1,
            )
            noise_sd = report.sigma_cov / np.sqrt(3.0)
            epsilon = 0.0
        elif mechanism == "gaussian":
            mean, covariance, report = dp_fit_class(
                logs, lo, hi, GAUSSIAN_EPSILON, DELTA, rng,
                clip_norm=clip_norm, eig_floor_ratio=0.1,
            )
            noise_sd = report.sigma_cov
            epsilon = GAUSSIAN_EPSILON
        else:
            raise ValueError(f"unknown mechanism: {mechanism}")
        means[label], covariances[label] = mean, covariance
        true_covariance = np.cov(logs, rowvar=False)
        signal = float(np.median(np.abs(true_covariance)))
        details[label] = {
            "records": int(logs.shape[0]),
            "noise_sd": float(noise_sd),
            "median_abs_true_covariance": signal,
            "noise_to_signal": float(noise_sd / max(signal, 1e-15)),
            "clipped_records": int(report.records_clipped),
            "epsilon": epsilon,
            "delta": DELTA,
            "clip_norm": clip_norm,
        }
    return LoadModel(
        mu=means, Sigma=covariances, members=classes,
        p_min=reference.p_min, p_max=reference.p_max,
        power_factor=theta, T=reference.T,
    ), details


def evaluate_voltage(
    runner: PowerFlowRunner,
    archive: np.ndarray,
    theta: np.ndarray,
    private_model: LoadModel,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Solve private samples and compare their voltages to the same archive."""
    reference, ok_reference = runner.solve_many(
        archive[:, :EVALUATION_DAYS, :],
        reactive_from_active(archive[:, :EVALUATION_DAYS, :], theta),
    )
    if not np.all(ok_reference):
        raise RuntimeError("Ckt5 reference trajectory did not converge")
    synthetic = sample_loads(
        private_model, EVALUATION_DAYS, rng=rng, sweeps=15,
    )
    released, ok_released = runner.solve_many(
        synthetic, reactive_from_active(synthetic, theta),
    )
    if not np.all(ok_released):
        raise RuntimeError("private Ckt5 trajectory did not converge")
    retained = runner.retained_indices()
    true_flat = reference.reshape(-1, reference.shape[-1])[:, retained]
    released_flat = released.reshape(-1, released.shape[-1])[:, retained]
    return {
        "wasserstein_1": float(voltage_wasserstein(true_flat, released_flat)),
        "ansi_violation_rate": float(ansi_violation_rate(released_flat)),
        "true_lag1": lag_one(np.abs(reference[:, :, retained])),
        "released_lag1": lag_one(np.abs(released[:, :, retained])),
        "reference_steps": int(reference.size // reference.shape[-1]),
    }


def run_experiment() -> dict:
    """Run both Ckt5 resolutions and mechanisms on fresh synthetic archives."""
    if not MASTER.exists():
        raise FileNotFoundError(
            f"{MASTER} is missing; run get_ckt5.py before this experiment"
        )
    kw, theta = feeder_load_ratings(str(MASTER))
    runner = PowerFlowRunner(str(MASTER))
    rows = []
    for T, clip_norm in REGIMES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            classes = assign_classes(kw, L=3)
            archive = make_historical(
                kw, classes, n_days=HISTORICAL_DAYS, T=T, rng=rng,
            )
            reference = fit_load_model(archive, classes, theta)
            for mechanism in ("bnp", "gaussian"):
                runner.reset()
                private_model, details = fit_private(
                    archive, classes, reference, theta, clip_norm, mechanism,
                    rng,
                )
                metrics = evaluate_voltage(
                    runner, archive, theta, private_model, rng,
                )
                ratios = [v["noise_to_signal"] for v in details.values()]
                raw_true_lags = [
                    covariance_lag_one(np.cov(
                        np.log(archive[members].reshape(-1, T)), rowvar=False,
                    ))
                    for members in classes.values()
                ]
                repaired_true_lags = [
                    covariance_lag_one(repair_covariance(np.cov(
                        np.log(archive[members].reshape(-1, T)), rowvar=False,
                    )))
                    for members in classes.values()
                ]
                true_lags = repaired_true_lags
                private_lags = [
                    covariance_lag_one(private_model.Sigma[label])
                    for label in classes
                ]
                rows.append({
                    "T": T,
                    "clip_norm": clip_norm,
                    "seed": seed,
                    "mechanism": mechanism,
                    "privacy": {
                        "epsilon": 0.0 if mechanism == "bnp" else GAUSSIAN_EPSILON,
                        "delta": DELTA,
                    },
                    "records_per_class": {
                        str(label): int(len(members) * HISTORICAL_DAYS)
                        for label, members in classes.items()
                    },
                    "class_details": details,
                    "max_noise_to_median_covariance": float(max(ratios)),
                    "mean_true_log_lag1": float(np.mean(true_lags)),
                    "mean_raw_true_log_lag1": float(np.mean(raw_true_lags)),
                    "mean_repaired_true_log_lag1": float(np.mean(repaired_true_lags)),
                    "mean_private_covariance_lag1": float(np.mean(private_lags)),
                    "lag1_change": float(np.mean(private_lags) - np.mean(true_lags)),
                    "acceptance": {
                        "noise_below_signal": bool(max(ratios) <= 1.0),
                        "lag1_within_0_05": bool(
                            abs(np.mean(private_lags) - np.mean(true_lags)) <= 0.05
                        ),
                        "passed": bool(
                            max(ratios) <= 1.0
                            and abs(np.mean(private_lags) - np.mean(true_lags)) <= 0.05
                        ),
                    },
                    "voltage": metrics,
                })
    return {
        "protocol": {
            "feeder": "EPRI Ckt5",
            "master": str(MASTER.relative_to(ROOT)),
            "historical_days": HISTORICAL_DAYS,
            "evaluation_days": EVALUATION_DAYS,
            "delta": DELTA,
            "seeds": list(SEEDS),
            "regimes": [list(x) for x in REGIMES],
            "adjacency": "one load-object-day record",
            "gaussian_comparison": "epsilon=1, delta=0.02; BNP epsilon=0, delta=0.02",
        },
        "rows": rows,
    }


def make_figure(result: dict) -> bytes:
    """Plot saved circuit-level metrics without rerunning the experiment."""
    rows = result["rows"]
    bnp = [r for r in rows if r["mechanism"] == "bnp"]
    labels = [f"T={r['T']}\nC={r['clip_norm']:g}" for r in bnp]
    x = np.arange(len(bnp))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    colors = ["#b34e17", "#17628f"]
    axes[0].bar(x, [r["max_noise_to_median_covariance"] for r in bnp], color=colors)
    axes[0].axhline(1, color="#333333", ls="--", lw=1)
    axes[0].set_title("BNP noise / median covariance")
    axes[0].set_ylabel("ratio (lower is better)")
    axes[1].bar(x, [r["voltage"]["wasserstein_1"] for r in bnp], color=colors)
    axes[1].set_title("Voltage fidelity")
    axes[1].set_ylabel("Wasserstein-1 (lower is better)")
    axes[2].bar(x, [r["voltage"]["released_lag1"] for r in bnp], color=colors)
    axes[2].plot(x, [r["voltage"]["true_lag1"] for r in bnp], "k--", label="true")
    axes[2].set_title("Voltage lag-1 correlation")
    axes[2].set_ylabel("correlation (higher is better)")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=.25)
    axes[2].legend(frameon=False)
    fig.suptitle("EPRI Ckt5: BNP across declared resolutions and bounds")
    fig.text(.5, .01, "Two fresh seeds per regime; load-object-day records; no customer-count claim.",
             ha="center", fontsize=9)
    buffer = __import__("io").BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    return buffer.getvalue()


def make_report(result: dict) -> str:
    """Render a complete, failure-inclusive markdown report."""
    rows = result["rows"]
    lines = [
        "# BNP circuit-regime experiment: EPRI Ckt5",
        "",
        "This experiment tests whether BNP preserves the original load and voltage",
        "dataset when applied at the load-model stage. It uses load-object-day",
        "records and makes no claim about customer counts.",
        "",
        "## Predeclared protocol",
        "",
        "- Public Ckt5 OpenDSS model at revision `5005c668a72d20775f4c2d060feebb2866ba1d38`.",
        f"- {HISTORICAL_DAYS} historical days and {EVALUATION_DAYS} held-out evaluation days.",
        f"- BNP privacy: `(epsilon=0, delta={DELTA})`; Gaussian: `(epsilon={GAUSSIAN_EPSILON}, delta={DELTA})`.",
        "- Two regimes: `(T=96, C=6)` and hourly `(T=24, C=3)`.",
        "- Two fresh seeds per regime and mechanism; every case is reported.",
        "- One record is one load-object-day. Days are not silently relabeled as independent customers.",
        "",
        "## Acceptance criterion",
        "",
        "A mechanism meets the fidelity screen only if the maximum class noise",
        "standard deviation is",
        "below that class's measured median absolute true covariance entry and the",
        "mean covariance lag-1 changes by no more than 0.05 relative to the",
        "nonprivate covariance after the existing 0.1 eigenvalue floor. Raw and",
        "repaired baselines are both retained in metrics.json. Voltage metrics are",
        "reported separately and are not converted into a privacy guarantee. The",
        "pass column applies the same fidelity screen to both mechanisms; only the",
        "BNP rows are used to claim BNP viability.",
        "",
        "## Results",
        "",
        "| T | C | records/class | mechanism | epsilon | max noise/signal | covariance lag-1 change | voltage W-1 | voltage lag-1 | ANSI violations | pass |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        records = ", ".join(str(v) for v in row["records_per_class"].values())
        lines.append(
            f"| {row['T']} | {row['clip_norm']:g} | {records} | {row['mechanism']} | "
            f"{row['privacy']['epsilon']:g} | {row['max_noise_to_median_covariance']:.3f} | "
            f"{row['lag1_change']:.3f} | {row['voltage']['wasserstein_1']:.6f} | "
            f"{row['voltage']['released_lag1']:.3f} | {row['voltage']['ansi_violation_rate']:.2%} | "
            f"{'yes' if row['acceptance']['passed'] else 'no'} |"
        )
    lines += [
        "",
        "The report distinguishes BNP's `(0, delta)` guarantee from the Gaussian",
        "comparison's `(1, delta)` guarantee. The mechanisms therefore share delta",
        "but not epsilon; the Gaussian row is a stated reference, not a matched",
        "privacy claim. The existing analytic Gaussian calibration and eigenvalue",
        "floor are unchanged.",
        "",
        "## Limits",
        "",
        "- Ckt5 synthetic histories are generated from the repository's model, not real meter data.",
        "- A lower C or lower T can introduce clipping or aggregation bias; those effects are measured per class in `metrics.json`.",
        "- The voltage comparison is empirical. No global voltage sensitivity bound is claimed.",
        "- Existing IEEE 123 measurements and prior result directories were not rerun or modified.",
        "",
        "## Reproduction",
        "",
        "    python get_ckt5.py",
        "    python run_bnp_ckt5.py --output-dir results/bnp_circuit_regime_20260917",
        "",
        "The runner refuses to overwrite an existing output directory.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the experiment and optionally save its new artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            parser.error("output directory already exists; refusing to overwrite measurements")
    result = run_experiment()
    report = make_report(result)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.output_dir / "report.md").write_text(report, encoding="utf-8")
        (args.output_dir / "overview.png").write_bytes(make_figure(result))
    print(report)


if __name__ == "__main__":
    main()
