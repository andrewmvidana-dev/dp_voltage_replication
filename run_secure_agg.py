"""Benchmark real 2048-bit Paillier on the complete IEEE 123 load roster.

python run_secure_agg.py --steps 3
python run_secure_agg.py --fit --steps 4 --output results/secure_agg_fit.json
python run_secure_agg.py --dropout 0.3 --compensate

The scalar benchmark measures one active-power sum per bus per timestep.
Optional fitting measures both model rounds separately from power flow.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import opendssdirect as dss

from dpvolt.experiments import (ModelFitConfig, fit_private_load_model,
                                voltage_utility_metrics)
from dpvolt.loads import (assign_classes, make_historical, fit_load_model,
                          sample_loads, reactive_from_active)
from dpvolt.powerflow import PowerFlowRunner
from dpvolt.secure_agg import (SecureAggConfig, PaillierSimulation,
                               aggregate_bus_loads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--meters", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--compensate", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--output", default="results/secure_agg_benchmark.json")
    args = parser.parse_args()
    if args.steps < (2 if args.fit else 1):
        parser.error("need positive steps, and at least two for model fitting")
    settings = SecureAggConfig(n_meters=args.meters, dropout=args.dropout,
                               compensate_dropout=args.compensate)
    root = Path(__file__).resolve().parent
    runner = PowerFlowRunner(str(root / "feeders/IEEE123Master.dss"))
    kw, theta = [], []
    for name in runner.load_names:
        dss.Loads.Name(name)
        kw.append(dss.Loads.kW())
        theta.append(np.arccos(np.clip(dss.Loads.PF(), -1, 1)))
    kw, theta = np.array(kw), np.array(theta)
    classes = assign_classes(kw)
    archive = make_historical(kw, classes, 45, T=max(args.steps, 2),
                              rng=np.random.default_rng(100))
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "settings": asdict(settings), "load_rows": len(kw),
              "steps": args.steps, "key_bits": 2048}
    try:
        import gmpy2
        report["gmpy2"] = gmpy2.version()
    except ImportError:
        report["gmpy2"] = None

    # Sigma=0 isolates fixed-point error without changing encryption cost.
    session = PaillierSimulation(settings.precision)
    elapsed, errors = [], []
    for step in range(args.steps):
        start = perf_counter()
        released, variance, _ = aggregate_bus_loads(
            archive[:, 0, step], 0.0, np.random.default_rng(1000 + step),
            settings, session=session)
        elapsed.append(perf_counter() - start)
        errors.append(float(np.max(np.abs(released - archive[:, 0, step]))))
        print(f"Step {step + 1}/{args.steps}: {elapsed[-1]:.3f} s", flush=True)
    timing = asdict(session.timing)
    report["bus_sum_benchmark"] = {
        "timing_totals": timing,
        "encryption_s_per_step": timing["encryption_s"] / args.steps,
        "decryption_s_per_step": timing["decryption_s"] / args.steps,
        "wall_s_per_step": float(np.mean(elapsed)),
        "max_difference_from_full_load_pu": max(errors),
        "unit_sigma_variance_ratio": (
            1.0 if settings.compensate_dropout else
            variance[0]["reporters"] / settings.n_meters),
    }

    if args.fit:
        # Public margins for this synthetic experiment, not private extrema.
        model = fit_load_model(archive, classes, theta)
        model.p_min = {label: 1e-5 for label in classes}
        model.p_max = {label: 1.0 for label in classes}
        runner.reset()
        truth, ok = runner.solve_many(archive[:, :8], reactive_from_active(archive[:, :8], theta))
        if not ok.all():
            raise RuntimeError("reference power flow failed")
        report["model_fit"] = {}
        for mode in ("trusted_curator", "secure_aggregation"):
            print(f"Fitting {mode}, T={model.T} on {len(kw)} load rows...", flush=True)
            config = ModelFitConfig(mode=mode, secure=settings)
            start = perf_counter()
            private = fit_private_load_model(
                archive, classes, model, theta, 50.0, 1e-5,
                np.random.default_rng(2000), config=config,
                clip_norm=6.0, eig_floor_ratio=0.1)
            fit_s = perf_counter() - start
            synth = sample_loads(private, 8, rng=np.random.default_rng(3000))
            runner.reset()
            voltages, ok = runner.solve_many(synth, reactive_from_active(synth, theta))
            if not ok.all():
                raise RuntimeError(f"{mode} power flow failed")
            item = {"fit_wall_s": fit_s,
                    "metrics": voltage_utility_metrics(truth, voltages)}
            if config.timing is not None:
                item["timing_totals"] = asdict(config.timing)
                item["amortized_encryption_s_per_model_time_coordinate"] = config.timing.encryption_s / model.T
                item["amortized_decryption_s_per_model_time_coordinate"] = config.timing.decryption_s / model.T
                item["classes"] = config.reports
            report["model_fit"][mode] = item

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
