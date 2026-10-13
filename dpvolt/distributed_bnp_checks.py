"""Opt-in validation of the ideal continuous distributed bounded-noise arm."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from dpvolt.privacy import (bnp_bound_scalar, bnp_distributed_delta,
                            bnp_distributed_half_width)


def distributed_bnp_checks(check, *, include_crypto=True, output=None):
    """Run independent checks without consuming any existing test RNG state.

    The optional verification report is written only when this function is
    explicitly called. Monte Carlo tolerance is fixed before sampling. Crypto
    checks encrypt the same prepared shares that the plaintext reference sums.
    """
    records = []
    metrics = {
        "scope": "One reading of one meter at one time, ideal continuous noise",
        "evidence": {
            "calibration": "ANALYTIC",
            "monte_carlo": "MEASURED",
            "cryptography": "MEASURED",
            "checks": "MEASURED",
        },
        "checks": records,
    }

    def record(name, passed, detail=""):
        records.append({"name": name, "passed": bool(passed), "detail": detail})
        check(name, bool(passed), detail)

    def rejects(call):
        try:
            call()
        except ValueError:
            return True
        return False

    # Fix public units and calibrate the same workload as the comparison.
    sensitivity_kw, delta, n = 20.0, 0.01, 10
    b_kw = bnp_distributed_half_width(sensitivity_kw, delta, n)
    actual_delta = bnp_distributed_delta(sensitivity_kw, b_kw, n)
    metrics["calibration"] = {
        "sensitivity_kw": sensitivity_kw, "delta": delta, "contributors": n,
        "half_width_kw": b_kw, "aggregate_bound_kw": n * b_kw,
        "aggregate_rms_kw": b_kw * np.sqrt(n / 3),
        "aggregate_delta": actual_delta,
        "individual_delta": bnp_distributed_delta(sensitivity_kw, b_kw, 1),
        "individual_attack_accuracy": 0.5 + sensitivity_kw / (4 * b_kw),
        "formula": "delta = F_n(n/2 + S/(4*b)) - F_n(n/2 - S/(4*b))",
    }
    record("distributed BNP exact CDF reaches the target delta",
           abs(actual_delta - delta) < 1e-9,
           f"ANALYTIC delta={actual_delta:.12g}, target={delta:.12g}")

    roundtrips = []
    for count in (1, 2, 10, 64):
        for target in (1e-5, 0.01, 0.1, 0.8):
            width = bnp_distributed_half_width(sensitivity_kw, target, count)
            achieved = bnp_distributed_delta(sensitivity_kw, width, count)
            roundtrips.append(abs(achieved - target))
    record("distributed BNP calibration round-trips across counts and budgets",
           max(roundtrips) < 1e-9,
           f"ANALYTIC maximum delta residual={max(roundtrips):.3g}")
    record("one distributed BNP share equals central uniform calibration",
           np.isclose(bnp_distributed_half_width(20, 0.01, 1),
                      bnp_bound_scalar(20, 0.01), rtol=1e-13)
           and np.isclose(bnp_distributed_delta(20, 1000, 1), 0.01))
    record("two-share CDF agrees with the triangular density integral",
           np.isclose(bnp_distributed_delta(1, 2, 2), 1 / 4 - 1 / 64,
                      rtol=0, atol=1e-14))
    record("distributed BNP calibration preserves a change of units",
           np.isclose(bnp_distributed_half_width(0.02, delta, n) * 1000,
                      b_kw, rtol=1e-13))
    record("distributed BNP more noise decreases distinguishability",
           bnp_distributed_delta(sensitivity_kw, 2 * b_kw, n)
           < actual_delta
           < bnp_distributed_delta(sensitivity_kw, b_kw / 2, n))
    record("distributed BNP zero sensitivity and zero noise are explicit",
           bnp_distributed_half_width(0, delta, n) == 0
           and bnp_distributed_delta(0, 0, n) == 0
           and bnp_distributed_delta(20, 0, n) == 1)
    record("distributed BNP disjoint supports return total variation one",
           bnp_distributed_delta(2 * n * b_kw, b_kw, n) == 1
           and bnp_distributed_delta(3 * n * b_kw, b_kw, n) == 1)
    invalid = [
        lambda: bnp_distributed_half_width(-1, delta, n),
        lambda: bnp_distributed_half_width(np.nan, delta, n),
        lambda: bnp_distributed_half_width(20, 0, n),
        lambda: bnp_distributed_half_width(20, 1, n),
        lambda: bnp_distributed_half_width(20, np.inf, n),
        lambda: bnp_distributed_half_width(20, 1e-20, n),
        lambda: bnp_distributed_delta(20, -1, n),
        lambda: bnp_distributed_delta(20, np.inf, n),
        lambda: bnp_distributed_delta(20, 1, 0),
        lambda: bnp_distributed_delta(20, 1, True),
        lambda: bnp_distributed_delta(20, 1, 1.5),
    ]
    record("distributed BNP rejects invalid public parameters",
           all(rejects(call) for call in invalid))

    # Estimate the defining central interval probability in bounded chunks.
    samples, seed, chunk_size = 1_000_000, 2026101303, 100_000
    tolerance = 6 * np.sqrt(delta * (1 - delta) / samples) + 1 / samples
    rng = np.random.default_rng(seed)
    count_inside = 0
    max_share = 0.0
    max_sum = 0.0
    sum_squares = 0.0
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        shares = rng.uniform(-b_kw, b_kw, size=(count, n))
        summed = shares.sum(axis=1)
        count_inside += int(np.count_nonzero(np.abs(summed) < sensitivity_kw / 2))
        max_share = max(max_share, float(np.abs(shares).max()))
        max_sum = max(max_sum, float(np.abs(summed).max()))
        sum_squares += float(np.dot(summed, summed))
    estimate = count_inside / samples
    ci95 = [
        float(beta.ppf(0.025, count_inside, samples - count_inside + 1))
        if count_inside else 0.0,
        float(beta.ppf(0.975, count_inside + 1, samples - count_inside))
        if count_inside < samples else 1.0,
    ]
    metrics["monte_carlo"] = {
        "samples": samples, "seed": seed, "inside_count": count_inside,
        "tv_estimate": estimate, "ci95_clopper_pearson": ci95,
        "predeclared_absolute_tolerance": tolerance,
        "tolerance_formula": "6*sqrt(delta*(1-delta)/samples) + 1/samples",
        "maximum_absolute_share_kw": max_share,
        "maximum_absolute_sum_kw": max_sum,
        "aggregate_rms_kw": float(np.sqrt(sum_squares / samples)),
        "meaning": "Each sample is one sum of independent uniform shares",
    }
    record("distributed BNP Monte Carlo TV agrees within predeclared tolerance",
           abs(estimate - delta) <= tolerance,
           f"MEASURED TV={estimate:.6f}, CI95={ci95}; "
           f"ANALYTIC target={delta}, tolerance={tolerance:.6f}")
    record("every sampled distributed BNP share respects its hard bound",
           max_share <= b_kw,
           f"MEASURED maximum={max_share:.6f} kW; ANALYTIC bound={b_kw:.6f} kW")
    record("every sampled distributed BNP sum respects its hard bound",
           max_sum <= n * b_kw,
           f"MEASURED maximum={max_sum:.6f} kW; ANALYTIC bound={n*b_kw:.6f} kW")

    if include_crypto:
        from dpvolt.secure_agg import PaillierSimulation
        from dpvolt.threshold_agg import ThresholdBGVSimulation

        # Use per-unit active power on the project's 1 MVA base.
        crypto_rng = np.random.default_rng(2026101304)
        raw = crypto_rng.uniform(0, sensitivity_kw / 1000, size=(n, 8))
        noise = crypto_rng.uniform(-b_kw / 1000, b_kw / 1000, size=raw.shape)
        prepared = raw + noise
        reference = prepared.sum(axis=0)
        metrics["cryptography"] = {}
        for name, constructor in (("Paillier", PaillierSimulation),
                                  ("Threshold BGV", ThresholdBGVSimulation)):
            try:
                session = constructor()
                encrypted = session.sum(prepared, 0, n, n,
                                        np.random.default_rng(2026101305))
                error_kw = float(np.max(np.abs(encrypted - reference)) * 1000)
                # Bound the sum of individual fixed-point encoding errors.
                bound_pu = (n * session.precision if name == "Paillier"
                            else n / (2 * session.config.scale))
                bound_kw = bound_pu * 1000 + 1e-9
                metrics["cryptography"][name] = {
                    "maximum_sum_difference_kw": error_kw,
                    "encoding_error_bound_kw": bound_kw,
                    "timing": asdict(session.timing),
                    "releases": getattr(session, "releases", []),
                }
                record(f"{name} preserves prepared distributed BNP sums",
                       error_kw <= bound_kw,
                       f"MEASURED max difference={error_kw:.9g} kW; "
                       f"ANALYTIC encoding bound={bound_kw:.9g} kW")
            except Exception as exc:
                metrics["cryptography"][name] = {"error": str(exc)}
                record(f"{name} preserves prepared distributed BNP sums", False,
                       f"Could not validate: {exc}")
    else:
        metrics["cryptography"] = {"status": "Not run in this invocation"}

    metrics["passed"] = sum(row["passed"] for row in records)
    metrics["total"] = len(records)
    if output is None:
        output = (Path(__file__).resolve().parents[1] / "results"
                  / "threat_model_20261013" / "distributed_verification.json")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
