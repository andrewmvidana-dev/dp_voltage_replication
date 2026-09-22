# Every correctness invariant in the project. Run after any change; takes about
# 30 seconds and prints PASS/FAIL per check plus a summary.
#
# This exists because every serious bug here was SILENT -- the code ran and
# produced plausible but wrong numbers. Deleting stub rows instead of
# eliminating them produced 238,225 kVA of phantom power; the slack offset b
# evaluating to zero blew ||M~^-1|| up to 1e16; capturing Y with loads disabled
# re-tapped the regulators and read -216 MW at the substation. None threw an
# exception. Each was caught by an invariant with a known answer.

import os
import sys
import warnings

import numpy as np
import opendssdirect as dss

from dpvolt.network import load_feeder, kron_reduce, injection_check
from dpvolt.loads import (assign_classes, make_historical, fit_load_model,
                          sample_loads, reactive_from_active, ar1_covariance,
                          diurnal_shape)
from dpvolt.powerflow import (PowerFlowRunner, to_per_unit, bnp_bound,
                              bnp_delta, add_bounded_voltage_noise,
                              add_voltage_noise)
from dpvolt.privacy import (gaussian_sigma, dp_fit_class, theorem1,
                            calibrate_M_inv, normalised_jacobian, solve_for_r,
                            bnp_fit_class, bnp_bound_scalar,
                            analytic_gaussian_sigma, analytic_gaussian_delta,
                            zcdp_rho_from_eps_delta,
                            bnp_fit_class_eigen_oracle, OracleFitReport,
                            DPFitReport)
from dpvolt.experiments import (voltage_wasserstein, build_masked_dataset,
                                Standardizer, train_and_curve,
                                ansi_violation_rate, mean_autocorrelation)


HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "feeders", "IEEE123Master.dss")

RESULTS = []


def check(name, condition, detail=""):
    """Record one invariant and print the outcome."""
    status = "PASS" if condition else "FAIL"
    RESULTS.append((name, bool(condition)))
    print(f"  [{status}]  {name}")
    if detail:
        print(f"           {detail}")


def _raises(fn, exc=ValueError):
    """True if fn() raises `exc`. For invariants about rejected input."""
    try:
        fn()
    except exc:
        return True
    return False


def section(title):
    print()
    print("-" * 74)
    print(title)
    print("-" * 74)


def secure_dropout_fit_checks():
    """Check dropout at the actual two-round model-fitting boundary."""
    from dpvolt.secure_agg import SecureAggConfig, PaillierSimulation
    from dpvolt.experiments import ModelFitConfig, fit_private_load_model

    archive = np.array([[[0.02, 0.04], [0.03, 0.06]],
                        [[0.08, 0.05], [0.07, 0.04]]])
    classes = {0: np.arange(2)}
    theta = np.full(2, 0.2)
    model = fit_load_model(archive, classes, theta)
    model.p_min, model.p_max = {0: 0.001}, {0: 1.0}
    session = PaillierSimulation()
    _, _, reference = dp_fit_class(np.log(archive.reshape(-1, 2)), np.log(0.001),
                                    0, 50, 1e-5, np.random.default_rng(42), clip_norm=1)
    for compensate in (False, True):
        config = ModelFitConfig(mode="secure_aggregation", secure=SecureAggConfig(
            n_meters=4, dropout=0.25, compensate_dropout=compensate))
        fitted = fit_private_load_model(
            archive, classes, model, theta, 50, 1e-5, np.random.default_rng(42),
            config=config, session=session, clip_norm=1)
        report = config.reports[0]
        ratio = 1 if compensate else 0.75
        check(f"encrypted fitting reports actual dropout variance, compensate={compensate}",
              report["enrolled"] == 8 and report["reporters"] == 6
              and report["nominal_noise_preserved"] == compensate
              and report["sigma_mu"] == reference.sigma_mu
              and report["sigma_cov"] == reference.sigma_cov
              and np.isclose(report["effective_mean_variance"], ratio * reference.sigma_mu**2)
              and np.isclose(report["effective_covariance_offdiag_variance"], ratio * reference.sigma_cov**2)
              and np.isclose(report["effective_covariance_diagonal_variance"], 2 * ratio * reference.sigma_cov**2)
              and np.linalg.eigvalsh(fitted.Sigma[0]).min() > 0)
    check("secure fit reports do not disclose clean-data diagnostics",
          not any(key in report for key in ("kl_to_true", "records_clipped", "mu_true", "cov_true")))
    check("secure model fitting rejects data outside public bounds",
          _raises(lambda: fit_private_load_model(
              archive * 100, classes, model, theta, 50, 1e-5, np.random.default_rng(42),
              config=config, session=session, clip_norm=1)))
    check("unknown aggregation mode is rejected",
          _raises(lambda: fit_private_load_model(
              archive, classes, model, theta, 50, 1e-5, np.random.default_rng(42),
              config=ModelFitConfig(mode="typo"))))


def secure_aggregation_checks():
    """Opt-in checks using real 2048-bit encryption, not mocked ciphertexts.

    Run python verify.py --secure-agg to include these slower checks, or use
    --secure-agg-only while developing this layer. No secrets enter reports.
    """
    from dataclasses import asdict
    import json
    from scipy.stats import chi2, t as student_t
    from dpvolt.secure_agg import (SecureAggConfig, PaillierSimulation,
                                  split_loads, noise_share, participation,
                                  aggregate_bus_loads, SimulatedMeter)
    from dpvolt.experiments import (ModelFitConfig, fit_private_load_model,
                                    voltage_utility_metrics)
    from dpvolt.loads import LoadModel

    section("10. Secure aggregation: arithmetic, noise, and utility")
    secure_dropout_fit_checks()
    settings = SecureAggConfig()
    session = PaillierSimulation()
    check("Paillier key is 2048 bits and collector holds no private key",
          session.collector.public_key.n.bit_length() == 2048
          and not hasattr(session.collector, "_private_key"))
    p = np.array([[0.04, 0.07], [0.12, 0.09]])
    pieces, weights = split_loads(p)
    check("Dirichlet customer loads sum to bus loads and reproduce",
          np.allclose(pieces.sum(axis=1), p, atol=1e-15, rtol=0)
          and np.array_equal(pieces, split_loads(p)[0]))
    check("explicit customer weights are respected",
          np.allclose(split_loads(p, weights=np.full((2, 10), 0.1))[0],
                      p[:, None, :] / 10))
    result, _, _ = aggregate_bus_loads(p, 0.0, np.random.default_rng(10), session=session)
    error = float(np.max(np.abs(result - p)))
    check("noise-free decrypted bus sums agree within 1e-9 pu", error < 1e-9,
          f"maximum error {error:.3e} pu")
    signed = np.array([0.0, -0.3, 0.8])
    result, _, _ = aggregate_bus_loads(signed, 0.0, np.random.default_rng(11), session=session)
    check("fixed-point sums handle zero and negative readings",
          np.allclose(result, signed, atol=1e-9, rtol=0))
    check("invalid settings, weights and nonfinite loads are rejected",
          all(_raises(fn) for fn in (
              lambda: SecureAggConfig(n_meters=1),
              lambda: SecureAggConfig(n_meters=2.5),
              lambda: SecureAggConfig(dropout=1),
              lambda: SecureAggConfig(dropout=-0.1),
              lambda: SecureAggConfig(dirichlet_alpha=0),
              lambda: split_loads(p, weights=np.zeros((2, 10))),
              lambda: split_loads(np.array([np.nan])),
              lambda: noise_share(-1, 10, 10),
              lambda: noise_share(1, 10, 0),
              lambda: participation(2, SecureAggConfig(dropout=0.9), np.random.default_rng(0)),
          )))
    check("committed roster mismatch is rejected before decryption",
          _raises(lambda: session.sum([np.zeros(1)], 0, 2, 2, np.random.default_rng(0))))

    # Verify the statistics that the two model-fitting rounds will sum.
    history = np.array([[[0.02, 0.04], [0.03, 0.06]],
                        [[0.08, 0.05], [0.07, 0.04]]])
    readings, shares = split_loads(history)
    meters = [SimulatedMeter(readings[b, i], shares[b, i])
              for b in range(2) for i in range(10)]
    logs = np.log(history.reshape(-1, 2))
    mean = logs.mean(axis=0)
    local_mean = sum(m.mean_contribution(4) for m in meters)
    centred = logs - mean
    centred *= np.minimum(1, 0.2 / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12))
    expected = (centred.T @ centred / 4)[np.triu_indices(2)]
    local_cov = sum(m.covariance_contribution(mean, 0.2, 4) for m in meters)
    check("meter contributions reproduce mean and clipped covariance",
          np.allclose(local_mean, mean, atol=1e-14, rtol=0)
          and np.allclose(local_cov, expected, atol=1e-14, rtol=0))
    encrypted_mean = session.sum((m.mean_contribution(4) for m in meters),
                                 0, 20, 20, np.random.default_rng(0))
    encrypted_cov = session.sum((m.covariance_contribution(mean, 0.2, 4) for m in meters),
                                0, 20, 20, np.random.default_rng(0))
    check("encrypted model sufficient statistics match plaintext",
          np.allclose(encrypted_mean, mean, atol=1e-9, rtol=0)
          and np.allclose(encrypted_cov, expected, atol=1e-9, rtol=0))

    # A tight many-trial check plus an independent end-to-end ciphertext check.
    sigma = 0.03
    noise_results = {}
    for k, compensate in ((10, False), (7, False), (7, True)):
        sd, variance = noise_share(sigma, 10, k, compensate)
        samples = np.random.default_rng(120 + k + compensate).normal(
            0, sd, size=(50000, k)).sum(axis=1)
        ratio = float(np.var(samples, ddof=1) / variance)
        interval = chi2.ppf([0.0005, 0.9995], len(samples) - 1) / (len(samples) - 1)
        name = f"k={k}, compensated={compensate}"
        check(f"summed Gaussian variance, {name}", interval[0] < ratio < interval[1],
              f"empirical/expected {ratio:.5f}; expected variance {variance:.6g}")
        noise_results[name] = {"empirical_variance": float(np.var(samples, ddof=1)),
                               "expected_variance": float(variance)}
    # Zero contributions isolate 256 independent summed noise draws, all
    # encrypted by ten meters and decrypted only after addition.
    draws = session.sum((np.zeros(256) for _ in range(10)), sigma / np.sqrt(10),
                        10, 10, np.random.default_rng(131))
    variance_ratio = float(np.var(draws, ddof=1) / sigma**2)
    limits = chi2.ppf([0.0005, 0.9995], 255) / 255
    check("encrypted Gaussian sums pass a 99.9% variance interval",
          limits[0] < variance_ratio < limits[1],
          f"empirical/target {variance_ratio:.4f}, interval {limits}")
    dropped = SecureAggConfig(dropout=0.3)
    compensated = SecureAggConfig(dropout=0.3, compensate_dropout=True)
    v_drop, rep_drop, _ = aggregate_bus_loads(
        np.array([0.1]), 0, np.random.default_rng(44), dropped, session=session)
    v_comp, rep_comp, _ = aggregate_bus_loads(
        np.array([0.1]), 0, np.random.default_rng(44), compensated, session=session)
    check("dropout removes readings; noise compensation does not impute data",
          0 < v_drop[0] < 0.1 and np.array_equal(v_drop, v_comp)
          and rep_drop[0]["reporters"] == rep_comp[0]["reporters"] == 7)

    # Learn a representative class from three simulated bus histories, then
    # use that model at every feeder load row. Small T makes repeated REAL
    # encrypted fits feasible. The separate benchmark fits all feeder rows.
    T = 6
    classes = {0: np.arange(3)}
    theta_small = np.full(3, np.arccos(0.95))
    archive = make_historical(np.full(3, 40.0), classes, 360, T=T,
                              rng=np.random.default_rng(140))
    model = fit_load_model(archive, classes, theta_small)
    model.p_min, model.p_max = {0: 1e-5}, {0: 1.0}
    options = dict(clip_norm=6.0, eig_floor_ratio=0.1)
    trusted = fit_private_load_model(archive, classes, model, theta_small, 50, 1e-5,
                                    np.random.default_rng(141),
                                    config=ModelFitConfig(), **options)
    old_mu, old_cov, _ = dp_fit_class(np.log(archive.reshape(-1, T)),
                                     np.log(1e-5), 0, 50, 1e-5,
                                     np.random.default_rng(141), **options)
    check("flag off preserves original fitter outputs exactly",
          np.array_equal(trusted.mu[0], old_mu)
          and np.array_equal(trusted.Sigma[0], old_cov))

    runner = PowerFlowRunner(MASTER)
    n_loads = len(runner.load_names)
    theta = np.full(n_loads, np.arccos(0.95))

    def whole_feeder(fitted):
        return LoadModel(fitted.mu, fitted.Sigma, {0: np.arange(n_loads)},
                         fitted.p_min, fitted.p_max, theta, T=T)

    truth_p = sample_loads(whole_feeder(model), 12, rng=np.random.default_rng(142))
    truth_v, ok = runner.solve_many(truth_p, reactive_from_active(truth_p, theta))
    if not ok.all():
        raise RuntimeError("utility reference did not converge")
    values = {mode: [] for mode in ("trusted_curator", "secure_aggregation")}
    secure_settings = SecureAggConfig(n_meters=2)
    trials = 32
    for trial in range(trials):
        for mode in values:
            config = ModelFitConfig(mode=mode, secure=secure_settings)
            # Separate fit seeds, paired downstream sampling to reduce variance.
            seed = 2000 + 2 * trial + (mode == "secure_aggregation")
            fitted = fit_private_load_model(
                archive, classes, model, theta_small, 50, 1e-5,
                np.random.default_rng(seed), config=config,
                session=session, **options)
            p_synth = sample_loads(whole_feeder(fitted), 12,
                                   rng=np.random.default_rng(3000 + trial))
            runner.reset()
            volts, ok = runner.solve_many(p_synth, reactive_from_active(p_synth, theta))
            if not ok.all():
                raise RuntimeError(f"{mode}, trial {trial}: power flow did not converge")
            values[mode].append(voltage_utility_metrics(truth_v, volts))
        if (trial + 1) % 4 == 0:
            print(f"  Encrypted utility fits: {trial + 1}/{trials}", flush=True)

    # Equivalence requires the entire 90% CI inside predeclared margins.
    # Merely obtaining p > .05 in a difference test would not establish this.
    margins = {"r2": 0.02, "ansi_exceedance": 0.005, "lag1": 0.05}
    comparisons = {}
    for metric, margin in margins.items():
        a = np.array([v[metric] for v in values["trusted_curator"]])
        b = np.array([v[metric] for v in values["secure_aggregation"]])
        differences = b - a
        half_width = float(student_t.ppf(0.95, trials - 1)
                           * differences.std(ddof=1) / np.sqrt(trials))
        interval = [float(differences.mean() - half_width),
                    float(differences.mean() + half_width)]
        equivalent = interval[0] > -margin and interval[1] < margin
        check(f"voltage {metric} is statistically equivalent within {margin}", equivalent,
              f"90% CI for secure minus trusted: {interval}")
        comparisons[metric] = {"trusted_mean": float(a.mean()),
                               "secure_mean": float(b.mean()),
                               "difference_ci90": interval, "margin": margin,
                               "equivalent": equivalent}
    output = {"noise": noise_results, "encrypted_variance_ratio": variance_ratio,
              "noise_free_max_error_pu": error,
              "utility_trials": trials, "utility_training_buses": 3,
              "utility_training_days": 360, "utility_T": T,
              "utility_meters_per_bus": 2, "voltage_load_rows": n_loads,
              "utility": comparisons, "crypto_timing": asdict(session.timing)}
    path = os.path.join(HERE, "results", "secure_agg_verification.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")


def main():
    if not os.path.exists(MASTER):
        print("Could not find the feeder files. Run get_feeder.py first.")
        return

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)

    print("=" * 74)
    print("VERIFICATION SUITE")
    print("=" * 74)

    # =====================================================================
    section("1. Network model and Kron reduction")
    # =====================================================================

    feeder = load_feeder(MASTER)
    kron = kron_reduce(feeder)
    inj = injection_check(feeder)

    check("feeder solves and node bookkeeping adds up",
          len(feeder.load_nodes) + len(feeder.zero_inj_nodes)
          + len(feeder.slack_nodes) == feeder.N,
          f"{len(feeder.load_nodes)} load + {len(feeder.zero_inj_nodes)} "
          f"zero-inj + {len(feeder.slack_nodes)} slack = {feeder.N}")

    # This one caught the stub-deletion bug. A bus we call zero-injection must
    # genuinely inject nothing.
    check("zero-injection buses inject no power",
          inj["max_|S|_zero_inj_kVA"] < 10.0,
          f"largest is {inj['max_|S|_zero_inj_kVA']:.3f} kVA against "
          f"3615 kW throughput; was 238225 kVA when pruning was wrong")

    check("total substation power matches IEEE 123's documented load",
          3400 < inj["total_slack_kW"] < 3700,
          f"{inj['total_slack_kW']:.1f} kW, documented 3490 kW; "
          f"read -216 MW before the regulator freeze was added")

    check("Kron reduction reproduces the true eliminated voltages",
          kron.residual < 1e-5,
          f"relative residual {kron.residual:.2e}; was 1.06 with naive pruning")

    check("Y_ZZ is well conditioned after cleanup",
          kron.cond_Y_ZZ < 1e8,
          f"cond = {kron.cond_Y_ZZ:.3e}, was 9.2e12 before merge and prune")

    check("network cleanup actually reduced kappa_Kron",
          kron.kappa_kron < 1e10,
          f"kappa_Kron = {kron.kappa_kron:.3e}, was 2.1e25 uncleaned")

    # The reduction must be an identity, not an approximation, on the block
    # it claims to reproduce.
    R, Z, S = kron.retained, kron.zero_inj, kron.slack
    Y = feeder.Y_full
    lhs = kron.Y_red
    rhs = (Y[np.ix_(R, R)]
           - Y[np.ix_(R, Z)] @ np.linalg.solve(Y[np.ix_(Z, Z)], Y[np.ix_(Z, R)]))
    check("Y_red equals the Schur complement exactly",
          np.allclose(lhs, rhs),
          f"max difference {np.abs(lhs - rhs).max():.2e}")

    # =====================================================================
    section("2. Per-unit conversion and the slack offset")
    # =====================================================================

    Vb = feeder.Vbase[kron.retained]
    Y_pu = to_per_unit(kron.Y_red, Vb)
    b_pu = kron.b * Vb / 1e6

    check("the slack offset b is non-zero",
          np.abs(b_pu).max() > 1e-6,
          f"max |b_pu| = {np.abs(b_pu).max():.4f}; evaluated to exactly 0 "
          f"before the elimination path was included")

    check("Y_pu is well conditioned",
          np.linalg.cond(Y_pu) < 1e6,
          f"cond = {np.linalg.cond(Y_pu):.3e}")

    # The strongest single check in the project. It exercises Y_red, b and the
    # per-unit conversion simultaneously against a number we know independently.
    v0 = feeder.V_node[kron.retained] / Vb
    s0 = v0 * (np.conj(Y_pu) @ np.conj(v0) + np.conj(b_pu))
    implied_kw = -np.real(s0).sum() * 1000.0
    check("load reconstructed from the reduced model matches the feeder",
          3200 < implied_kw < 3800,
          f"{implied_kw:.0f} kW against the feeder's 3490 kW "
          f"({abs(implied_kw - 3490) / 3490:.1%} error)")

    # Per-unit conversion must be reversible.
    D = np.diag(Vb)
    back = np.linalg.inv(D) @ (Y_pu * 1e6) @ np.linalg.inv(D)
    check("per-unit conversion is invertible",
          np.allclose(back, kron.Y_red, rtol=1e-9),
          f"max relative difference "
          f"{np.abs(back - kron.Y_red).max() / np.abs(kron.Y_red).max():.2e}")

    # =====================================================================
    section("3. Load model")
    # =====================================================================

    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {MASTER}")
    dss.Text.Command("Solve")
    kw, pf = [], []
    i = dss.Loads.First()
    while i > 0:
        kw.append(dss.Loads.kW())
        pf.append(dss.Loads.PF())
        i = dss.Loads.Next()
    kw = np.array(kw)
    theta = np.arccos(np.clip(np.array(pf), -1.0, 1.0))

    classes = assign_classes(kw, L=3)
    check("every bus lands in exactly one class",
          sum(len(v) for v in classes.values()) == len(kw)
          and len(set().union(*[set(v) for v in classes.values()])) == len(kw),
          f"{len(kw)} buses across {len(classes)} classes")

    # A daily shape must average to 1, or it silently rescales demand.
    for kind in ("residential", "commercial", "industrial"):
        shape = diurnal_shape(96, kind)
        check(f"diurnal shape '{kind}' averages to 1.0",
              abs(shape.mean() - 1.0) < 1e-12,
              f"mean = {shape.mean():.12f}")

    cov = ar1_covariance(96, sigma=0.3, rho=0.93)
    evals = np.linalg.eigvalsh(cov)
    check("AR(1) covariance is symmetric positive definite",
          np.allclose(cov, cov.T) and evals.min() > 0,
          f"smallest eigenvalue {evals.min():.3e}")

    archive = make_historical(kw, classes, n_days=45, rng=rng)
    total = archive.sum(axis=0).mean() * 1000.0
    check("synthetic history reproduces the feeder's total demand",
          abs(total - kw.sum()) / kw.sum() < 0.05,
          f"{total:.0f} kW against {kw.sum():.0f} kW "
          f"({abs(total - kw.sum()) / kw.sum():.2%})")

    check("all generated loads are strictly positive",
          archive.min() > 0,
          f"minimum {archive.min():.3e} per-unit")

    model = fit_load_model(archive, classes, theta)
    check("fitted covariances are positive definite",
          all(np.linalg.eigvalsh(model.Sigma[l]).min() > 0 for l in classes))

    check("load margins bracket the observed data",
          all(model.p_min[l] <= archive[classes[l]].min()
              and model.p_max[l] >= archive[classes[l]].max()
              for l in classes))

    synth, repairs = sample_loads(model, 4, rng=rng, sweeps=15, report=True)
    check("synthetic loads respect the truncation box",
          all(np.all(synth[classes[l]] >= model.p_min[l] * 0.999)
              and np.all(synth[classes[l]] <= model.p_max[l] * 1.001)
              for l in classes),
          f"repair rates " + ", ".join(f"{k}:{v:.1%}" for k, v in repairs.items()))

    # The property the whole method exists to preserve.
    real_c = np.log(archive[classes[0]].reshape(-1, 96))
    syn_c = np.log(synth[classes[0]].reshape(-1, 96))
    lag_r = np.corrcoef(real_c.T)[0, 1:6]
    lag_s = np.corrcoef(syn_c.T)[0, 1:6]
    check("temporal correlation survives sampling",
          np.abs(lag_r - lag_s).max() < 0.08,
          f"max lag-1..5 difference {np.abs(lag_r - lag_s).max():.4f}")

    q = reactive_from_active(synth, theta)
    expected = synth[0, 0, 0] * np.tan(theta[0])
    check("reactive power follows the fixed power factor",
          np.isclose(q[0, 0, 0], expected),
          f"q = {q[0,0,0]:.6f}, expected {expected:.6f}")

    # =====================================================================
    section("4. Differential privacy")
    # =====================================================================

    # More privacy must mean more noise. If this is backwards, everything is.
    s_lo = gaussian_sigma(1.0, 1.0, 1e-5)
    s_hi = gaussian_sigma(1.0, 100.0, 1e-5)
    check("smaller epsilon gives larger noise",
          s_lo > s_hi,
          f"sigma(eps=1) = {s_lo:.3f} > sigma(eps=100) = {s_hi:.3f}")

    check("noise scales inversely with epsilon",
          np.isclose(s_lo / s_hi, 100.0),
          f"ratio {s_lo / s_hi:.4f}, expected exactly 100")

    data = np.log(archive[classes[0]].reshape(-1, 96))
    lo_b, hi_b = np.log(model.p_min[0]), np.log(model.p_max[0])

    mu_dp, cov_dp, rep = dp_fit_class(data, lo_b, hi_b, 50.0, 1e-5, rng,
                                      clip_norm=6.0)
    check("DP covariance is symmetric",
          np.allclose(cov_dp, cov_dp.T),
          f"max asymmetry {np.abs(cov_dp - cov_dp.T).max():.2e}")

    check("DP covariance is positive definite after repair",
          np.linalg.eigvalsh(cov_dp).min() > 0,
          f"smallest eigenvalue {np.linalg.eigvalsh(cov_dp).min():.3e}, "
          f"{rep.eig_clipped} of 96 needed lifting")

    # Two runs at the same epsilon must differ, or the noise is not being added.
    mu_a, _, _ = dp_fit_class(data, lo_b, hi_b, 50.0, 1e-5,
                              np.random.default_rng(1), clip_norm=6.0)
    mu_b, _, _ = dp_fit_class(data, lo_b, hi_b, 50.0, 1e-5,
                              np.random.default_rng(2), clip_norm=6.0)
    check("the mechanism is actually random",
          not np.allclose(mu_a, mu_b),
          f"two draws differ by {np.abs(mu_a - mu_b).max():.3e}")

    # A tighter budget must move the fit further from the truth.
    _, _, r_tight = dp_fit_class(data, lo_b, hi_b, 5.0, 1e-5,
                                 np.random.default_rng(3), clip_norm=6.0)
    _, _, r_loose = dp_fit_class(data, lo_b, hi_b, 200.0, 1e-5,
                                 np.random.default_rng(3), clip_norm=6.0)
    check("tighter privacy degrades the fit",
          r_tight.kl_to_true > r_loose.kl_to_true,
          f"KL {r_tight.kl_to_true:.1f} at eps=5 vs "
          f"{r_loose.kl_to_true:.1f} at eps=200")

    # ---- calibration audit -------------------------------------------------
    # The whole point of this block: a sigma is only correct if the delta it
    # ACTUALLY achieves is at or under the delta claimed. The classical formula
    # passes every structural check above while failing this one, which is how
    # the under-noising went unnoticed.

    # The analytic mechanism must hit its target delta exactly, at every
    # epsilon -- including the 25-200 range this project runs in.
    exact_ok, exact_detail = True, []
    for eps_t in (0.5, 1.0, 5.0, 25.0, 50.0, 200.0):
        for delta_t in (1e-3, 1e-5, 5e-6):
            sig = analytic_gaussian_sigma(0.0267, eps_t, delta_t)
            hit = analytic_gaussian_delta(0.0267, sig, eps_t)
            if hit > delta_t * (1.0 + 1e-6):
                exact_ok = False
                exact_detail.append(f"eps={eps_t} delta={delta_t:g} hit {hit:.3e}")
    check("analytic Gaussian achieves its target delta at every epsilon",
          exact_ok,
          "18 (epsilon, delta) pairs checked against Balle-Wang Thm 8"
          if exact_ok else "; ".join(exact_detail))

    # More noise must buy a smaller delta -- the monotonicity that makes the
    # bisection in analytic_gaussian_sigma valid in the first place.
    check("delta decreases as sigma grows",
          analytic_gaussian_delta(0.0267, 0.02, 50.0)
          > analytic_gaussian_delta(0.0267, 0.2, 50.0),
          f"delta {analytic_gaussian_delta(0.0267, 0.02, 50.0):.3e} at "
          f"sigma=0.02 vs {analytic_gaussian_delta(0.0267, 0.2, 50.0):.3e} "
          f"at sigma=0.2")

    # THE finding. The paper's stated formula does not deliver its guarantee
    # above epsilon = 1; this pins the failure so a future edit cannot quietly
    # reintroduce it as the default.
    d_classical = analytic_gaussian_delta(
        0.0267, gaussian_sigma(0.0267, 50.0, 1e-5), 50.0)
    check("classical formula is confirmed to under-noise at eps=50",
          d_classical > 1e-5,
          f"claims delta=1e-5, actually achieves {d_classical:.3e} "
          f"({d_classical / 1e-5:.0f}x worse) -- why 'analytic' is the default")

    # The default path must be the correct one. A regression here is silent.
    _, _, r_default = dp_fit_class(data, lo_b, hi_b, 50.0, 1e-5,
                                   np.random.default_rng(4), clip_norm=6.0)
    check("dp_fit_class defaults to the analytic calibration",
          r_default.calibration == "analytic",
          f"calibration = {r_default.calibration!r}")

    check("the default fit's audited delta meets its claim",
          r_default.delta_achieved <= 1e-5 * (1.0 + 1e-3),
          f"claimed 1e-5, audited {r_default.delta_achieved:.3e}")

    # And the classical path, when explicitly requested, must FAIL that audit
    # -- otherwise the comparison it exists to support is meaningless.
    _, _, r_cls = dp_fit_class(data, lo_b, hi_b, 50.0, 1e-5,
                               np.random.default_rng(4), clip_norm=6.0,
                               calibration="classical")
    check("the classical path is measurably worse than it claims",
          r_cls.delta_achieved > 1e-5,
          f"claimed 1e-5, audited {r_cls.delta_achieved:.3e}")

    # zCDP conversion must round-trip, or the budget split is not accounted.
    rho_rt = zcdp_rho_from_eps_delta(50.0, 1e-5)
    eps_rt = rho_rt + 2.0 * np.sqrt(rho_rt * np.log(1e5))
    check("zCDP conversion round-trips",
          np.isclose(eps_rt, 50.0),
          f"rho {rho_rt:.4f} converts back to eps {eps_rt:.6f}")

    # Shifting budget toward the covariance must hold delta while changing the
    # noise split -- the tunable knob is only sound if delta is unaffected.
    _, _, r_even = dp_fit_class(data, lo_b, hi_b, 50.0, 1e-5,
                                np.random.default_rng(5), clip_norm=6.0,
                                rho_split=0.5)
    _, _, r_cov = dp_fit_class(data, lo_b, hi_b, 50.0, 1e-5,
                               np.random.default_rng(5), clip_norm=6.0,
                               rho_split=0.1)
    check("shifting budget to the covariance lowers its noise",
          r_cov.sigma_cov < r_even.sigma_cov
          and r_cov.sigma_mu > r_even.sigma_mu,
          f"sigma_cov {r_even.sigma_cov:.3e} -> {r_cov.sigma_cov:.3e}, "
          f"sigma_mu {r_even.sigma_mu:.3e} -> {r_cov.sigma_mu:.3e}")

    check("the budget split does not change the delta achieved",
          r_cov.delta_achieved <= 1e-5 * (1.0 + 1e-3),
          f"audited {r_cov.delta_achieved:.3e} at rho_split=0.1")

    check("dp_fit_class rejects an unknown calibration",
          _raises(lambda: dp_fit_class(data, lo_b, hi_b, 50.0, 1e-5, rng,
                                       calibration="nonsense")),
          "unknown calibration must raise, not silently pick a default")

    # =====================================================================
    section("5. Theorem 1")
    # =====================================================================

    adj = np.abs(Y_pu) > 1e-10
    np.fill_diagonal(adj, False)
    d_max = int(adj.sum(axis=1).max())
    n = len(kron.retained)

    Sigma, sizes, pmins = {}, {}, {}
    for l in classes:
        d = np.log(archive[classes[l]].reshape(-1, 96))
        _, c, _ = dp_fit_class(d, np.log(model.p_min[l]), np.log(model.p_max[l]),
                               50.0, 1e-5, rng, clip_norm=6.0,
                               eig_floor_ratio=0.1)
        Sigma[l], sizes[l], pmins[l] = c, len(classes[l]), model.p_min[l]

    common = dict(Sigma_by_class=Sigma, size_by_class=sizes,
                  p_min_by_class=pmins, n=n, T=96, d_max=d_max,
                  kappa_kron=kron.kappa_kron, delta=1e-5, M_inv_norm=5.8)

    b_small = theorem1(r=1e-14, **common)
    b_large = theorem1(r=1e-12, **common)

    check("epsilon increases with the adjacency radius",
          b_small.epsilon < b_large.epsilon,
          f"{b_small.epsilon:.2f} at r=1e-14 vs "
          f"{b_large.epsilon:.2f} at r=1e-12")

    check("alpha scales linearly with r",
          np.isclose(b_large.alpha / b_small.alpha, 100.0),
          f"ratio {b_large.alpha / b_small.alpha:.4f}, expected 100")

    check("the admissibility condition is alpha < 1/4",
          b_small.admissible == (b_small.alpha < 0.25))

    check("an inadmissible radius returns infinity, not a plausible number",
          not np.isfinite(theorem1(r=1.0, **common).epsilon))

    # The paper's core mechanism: a noisier load model must buy more privacy.
    Sigma_noisy = {l: Sigma[l] * 25.0 for l in Sigma}
    b_noisy = theorem1(r=1e-13, **{**common, "Sigma_by_class": Sigma_noisy})
    b_sharp = theorem1(r=1e-13, **common)
    check("a noisier load model gives a smaller epsilon",
          b_noisy.epsilon < b_sharp.epsilon,
          f"{b_noisy.epsilon:.2f} with 25x covariance vs "
          f"{b_sharp.epsilon:.2f} baseline")

    check("epsilon decomposes into its three reported terms",
          np.isclose(b_sharp.epsilon,
                     b_sharp.bias_term + b_sharp.tail_term),
          f"bias {b_sharp.bias_term:.3f} + tail {b_sharp.tail_term:.3f} "
          f"= {b_sharp.epsilon:.3f}")

    r_star = solve_for_r(target_epsilon=200.0, **common)
    eps_at_r = theorem1(r=r_star, **common).epsilon
    check("solve_for_r inverts theorem1 correctly",
          abs(eps_at_r - 200.0) / 200.0 < 0.02,
          f"r = {r_star:.3e} gives epsilon = {eps_at_r:.2f}, targeted 200")

    # =====================================================================
    section("6. Jacobian and Monte Carlo calibration")
    # =====================================================================

    M = normalised_jacobian(v0, Y_pu, b_pu)
    check("the normalised Jacobian has the right shape",
          M.shape == (2 * n, 2 * n),
          f"{M.shape} for n = {n}")

    check("the normalised Jacobian is invertible",
          np.linalg.cond(M) < 1e12,
          f"cond = {np.linalg.cond(M):.3e}")

    runner = PowerFlowRunner(MASTER)
    sel_idx = runner.retained_indices()
    V_pf, ok_pf = runner.solve_many(synth[:, :2, :],
                                    reactive_from_active(synth[:, :2, :], theta))
    check("power flow converges on synthetic loads",
          ok_pf.mean() > 0.95,
          f"{ok_pf.mean():.1%} of timesteps converged")

    name_to_idx = {nm: i for i, nm in enumerate(runner.node_names)}
    sel_k = [name_to_idx[nm] for nm in kron.names_retained]
    Vf = V_pf.reshape(-1, V_pf.shape[-1])[:, sel_k]

    cal = calibrate_M_inv(Vf[:60], Y_pu, b_pu)
    check("||M~^-1|| calibrates to a sane magnitude",
          1.0 < cal["mu_0"] < 1e3,
          f"mu_0 = {cal['mu_0']:.3f}; read 1.26e16 when b was zero")

    check("Clopper-Pearson upper bound exceeds the point estimate",
          cal["delta_M_upper"] >= cal["delta_M"],
          f"delta_M {cal['delta_M']:.4f}, upper {cal['delta_M_upper']:.4f}")

    check("the calibration quantile behaves monotonically",
          calibrate_M_inv(Vf[:60], Y_pu, b_pu, quantile=0.5)["mu_0"]
          <= cal["mu_0"])

    # =====================================================================
    section("7. Experiments")
    # =====================================================================

    A = np.abs(Vf[:40])
    check("Wasserstein distance of a set with itself is zero",
          voltage_wasserstein(Vf[:40], Vf[:40]) < 1e-12)

    check("Wasserstein distance grows with added noise",
          voltage_wasserstein(Vf[:40], Vf[:40] + 0.05)
          > voltage_wasserstein(Vf[:40], Vf[:40] + 0.01))

    X, Yt = build_masked_dataset(V_pf[:, :, sel_idx])
    check("masked dataset has consistent shapes",
          X.shape[0] == Yt.shape[0] and X.shape[1] == 96 and Yt.shape[1] == 24,
          f"X{X.shape} Y{Yt.shape}")

    check("the mask flags exactly the hidden entries",
          np.all(X[:, 48:72] == 1.0) and np.all(X[:, 72:96] == 0.0),
          "first 24 of the window visible, last 24 hidden")

    st = Standardizer(X)
    Z = st.transform(X)
    # Only the VARYING columns can be standardised to unit variance. The mask
    # flags are constant by construction, so they correctly become all-zero
    # and must be excluded from this check.
    varying = X.std(axis=0) > 1e-6
    check("standardiser gives varying columns zero mean and unit variance",
          abs(Z[:, varying].mean()) < 1e-10
          and abs(Z[:, varying].std() - 1.0) < 1e-6,
          f"{varying.sum()} varying columns: mean "
          f"{Z[:, varying].mean():.2e}, std {Z[:, varying].std():.6f}")

    check("standardiser leaves constant columns at zero",
          np.allclose(Z[:, ~varying], 0.0),
          f"{(~varying).sum()} constant columns (the mask flags)")

    curve = train_and_curve(X, Yt, X, Yt, epochs=10)
    check("training reduces the error",
          curve[-1] < curve[0],
          f"MSE {curve[0]:.3e} -> {curve[-1]:.3e}")

    check("the error curve is finite throughout",
          np.all(np.isfinite(curve)))

    # =====================================================================
    section("8. Bounded-Noise Privacy baseline")
    # =====================================================================

    # The defining guarantee: no released value may ever exceed B.
    B_test = 0.02
    V_small = Vf[:40]
    V_bnp = add_bounded_voltage_noise(V_small, B_test, np.random.default_rng(7))
    dev_re = np.abs(np.real(V_bnp - V_small)).max()
    dev_im = np.abs(np.imag(V_bnp - V_small)).max()
    check("BNP noise never exceeds the bound B",
          dev_re <= B_test and dev_im <= B_test,
          f"max deviation {max(dev_re, dev_im):.6f} against B = {B_test}")

    # Corollary 1 round-trip.
    check("bnp_bound inverts bnp_delta",
          np.isclose(bnp_delta(0.67, bnp_bound(0.67, 1e-5)), 1e-5),
          f"delta {bnp_delta(0.67, bnp_bound(0.67, 1e-5)):.3e}, targeted 1e-5")

    check("larger B gives a smaller delta",
          bnp_delta(0.67, 100.0) < bnp_delta(0.67, 10.0),
          f"{bnp_delta(0.67, 100.0):.3e} at B=100 vs "
          f"{bnp_delta(0.67, 10.0):.3e} at B=10")

    # Corollary 1 requires S <= 2B; a delta above 1 is not a probability.
    raised = False
    try:
        bnp_delta(10.0, 1.0)          # S = 10, 2B = 2
    except ValueError:
        raised = True
    check("bnp_delta rejects S > 2B (Corollary 1 precondition)",
          raised,
          "S = 10 with 2B = 2 raises rather than returning delta = 5")

    # The honest limitation: bounding controls magnitude, not time structure.
    # Both output-perturbation baselines must flatten autocorrelation.
    V_days = V_pf[:, :, sel_idx]
    ac_true = mean_autocorrelation(V_days)
    ac_bnp = mean_autocorrelation(
        add_bounded_voltage_noise(V_days, 0.02, np.random.default_rng(8)))
    ac_gauss = mean_autocorrelation(
        add_voltage_noise(V_days, 0.02, np.random.default_rng(9)))
    check("bounded noise still destroys temporal correlation",
          ac_bnp[0] < ac_true[0] and ac_gauss[0] < ac_true[0],
          f"lag-1: true {ac_true[0]:.3f}, bnp {ac_bnp[0]:.3f}, "
          f"gaussian {ac_gauss[0]:.3f}")

    # Violation rate: a known-good and a known-bad case.
    check("violation rate is zero on in-band voltages",
          ansi_violation_rate(np.ones((5, 5), dtype=complex)) == 0.0)

    check("violation rate is one on out-of-band voltages",
          ansi_violation_rate(np.full((5, 5), 2.0, dtype=complex)) == 1.0)

    # The two bnp_bound implementations must agree -- privacy.py keeps its own
    # copy so it need not import the OpenDSS-dependent powerflow module.
    check("the two bnp_bound implementations agree",
          np.isclose(bnp_bound_scalar(0.67, 1e-3), bnp_bound(0.67, 1e-3)),
          f"privacy {bnp_bound_scalar(0.67, 1e-3):.4f} vs "
          f"powerflow {bnp_bound(0.67, 1e-3):.4f}")

    # The bounded load fit must still return a usable covariance.
    mu_b, cov_b, rep_b = bnp_fit_class(data, lo_b, hi_b, 0.02,
                                       np.random.default_rng(11),
                                       clip_norm=6.0, eig_floor_ratio=0.1)
    check("bounded load fit returns a positive-definite covariance",
          np.linalg.eigvalsh(cov_b).min() > 0,
          f"smallest eigenvalue {np.linalg.eigvalsh(cov_b).min():.3e}")

    check("bounded load fit reports epsilon = 0",
          rep_b.epsilon == 0.0,
          "uniform BNP is (0, delta)-private by Corollary 1")

    # The mean perturbation must respect its own bound.
    B_mu_expected = bnp_bound_scalar(np.sqrt(96) * (hi_b - lo_b) / len(data),
                                     0.02 / 2)
    check("bounded mean noise never exceeds its bound",
          np.abs(mu_b - data.mean(axis=0)).max() <= B_mu_expected + 1e-12,
          f"max deviation {np.abs(mu_b - data.mean(axis=0)).max():.4f} "
          f"against B = {B_mu_expected:.4f}")

    # The finding: at matched delta, BNP's bound dwarfs the Gaussian sigma,
    # and the ratio does not improve with more records.
    ratios = []
    for m_test in (270, 27000):
        s = np.sqrt(96) * 4.6 / m_test
        ratios.append(bnp_bound_scalar(s, 5e-6) / gaussian_sigma(s, 25, 5e-6))
    check("BNP/Gaussian noise ratio is independent of sample size",
          np.isclose(ratios[0], ratios[1]),
          f"{ratios[0]:.1f} at m=270 vs {ratios[1]:.1f} at m=27000 "
          f"-- averaging cannot close the gap")

    # reset() must recover the runner after absurd loads have been written.
    # Without it a single degenerate operating point makes every LATER solve
    # return nan, which is silent and corrupts whole parameter sweeps.
    absurd = np.full((len(runner.load_names), 1, 3), 1e6)
    runner.solve_many(absurd, absurd * 0.3)
    runner.reset()
    V_after, ok_after = runner.solve_many(synth[:, :1, :],
                                          reactive_from_active(synth[:, :1, :], theta))
    check("reset() recovers the runner after a degenerate solve",
          ok_after.mean() > 0.95 and np.isfinite(V_after).all(),
          f"{ok_after.mean():.0%} converged, all finite "
          f"{bool(np.isfinite(V_after).all())}")

    check("bigger bound gives more ANSI violations",
          ansi_violation_rate(
              add_bounded_voltage_noise(V_small, 0.5, np.random.default_rng(10)))
          > ansi_violation_rate(
              add_bounded_voltage_noise(V_small, 0.001, np.random.default_rng(10))))

    # =====================================================================
    section("9. The eigenvalue-space oracle -- NOT a mechanism")
    # =====================================================================

    # bnp_fit_class_eigen_oracle exists to be MEASURED, never reported. It
    # releases the true eigenvectors in the clear, so it has no guarantee at
    # any parameter. The protection is structural -- its report type has no
    # guarantee fields at all -- and these checks pin that down, because a
    # future edit re-adding them would otherwise be silent.

    mu_o, cov_o, rep_o = bnp_fit_class_eigen_oracle(
        data, lo_b, hi_b, 0.02, np.random.default_rng(12),
        clip_norm=6.0, eig_floor_ratio=0.1, verbose=False)

    check("the eigen-oracle labels itself NOT PRIVATE",
          rep_o.calibration == "bnp-eigen-oracle-NOTPRIVATE",
          f"calibration = {rep_o.calibration!r}")

    check("the eigen-oracle report has NO guarantee fields",
          isinstance(rep_o, OracleFitReport)
          and not isinstance(rep_o, DPFitReport)
          and not any(hasattr(rep_o, f)
                      for f in ("epsilon", "delta", "delta_achieved")),
          "epsilon/delta/delta_achieved must not exist on an oracle report -- "
          "reading one is an AttributeError, not a number")

    check("reading a delta off the oracle raises",
          _raises(lambda: rep_o.delta, AttributeError),
          "the invalid state is unrepresentable, not merely detected")

    check("real mechanisms still carry their guarantee fields",
          isinstance(r_default, DPFitReport)
          and r_default.delta_achieved is not None
          and rep_b.epsilon == 0.0,
          "the report split must not strip dp_fit_class or bnp_fit_class")

    # The defining property, stated as a test: the released eigenvectors ARE
    # the true ones. If this ever passes on a real mechanism, that mechanism
    # is broken.
    centred_chk = data - mu_o
    nrm = np.linalg.norm(centred_chk, axis=1, keepdims=True)
    centred_chk = centred_chk * np.minimum(1.0, 6.0 / np.maximum(nrm, 1e-12))
    cov_chk = (centred_chk.T @ centred_chk) / len(data) + 1e-12 * np.eye(96)
    ev_true = np.linalg.eigh(cov_chk)[1]
    ev_rel = np.linalg.eigh(cov_o)[1]

    # NOT a projector comparison of the leading columns. B_lambda (1.38) is
    # larger than the gaps between the true eigenvalues, so the noise REORDERS
    # the spectrum -- and eigh sorts its output by eigenvalue, so the released
    # matrix's "leading k" columns are a different SUBSET of the same true
    # eigenbasis, not a rotation of it. Comparing leading blocks therefore
    # fails (measured: 1.3e-1 at k=8) even though nothing was rotated.
    #
    # The right statement of the leak: EVERY released eigenvector is one of the
    # true ones up to sign. So match each released column to its best true
    # partner by |inner product| and require that to be 1.
    overlap = np.abs(ev_rel.T @ ev_true)        # (released, true)
    best = overlap.max(axis=1)
    # The floor pins 54 of 96 eigenvalues to one degenerate value, and any
    # orthonormal basis of a degenerate eigenspace is valid, so only the
    # non-degenerate modes carry a well-defined individual eigenvector.
    floor_val = 0.1 * float(np.trace(cov_chk)) / 96
    nondegenerate = ~np.isclose(np.linalg.eigvalsh(cov_o), floor_val, rtol=1e-9)
    check("the eigen-oracle leaks the true eigenvectors exactly",
          bool(np.all(best[nondegenerate] > 1.0 - 1e-6)),
          f"all {int(nondegenerate.sum())} non-degenerate released "
          f"eigenvectors match a TRUE eigenvector to "
          f"|<u,v>| >= {best[nondegenerate].min():.9f} -- no rotation, no "
          f"noise on V: this is the leak, measured")

    # Disabling the floor must be a real change, or the 'floor disabled' row of
    # the peer-review table is measuring nothing.
    _, cov_nofloor, rep_nofloor = bnp_fit_class_eigen_oracle(
        data, lo_b, hi_b, 0.02, np.random.default_rng(12),
        clip_norm=6.0, eig_floor_ratio=None, verbose=False)
    check("disabling the eigenvalue floor changes the result",
          not np.allclose(cov_o, cov_nofloor),
          f"eig_clipped {rep_o.eig_clipped} with floor vs "
          f"{rep_nofloor.eig_clipped} without")

    # =====================================================================
    print()
    print("=" * 74)
    passed = sum(1 for _, ok in RESULTS if ok)
    total_n = len(RESULTS)
    print(f"RESULT: {passed} of {total_n} checks passed")
    if passed < total_n:
        print()
        print("Failing checks:")
        for nm, ok in RESULTS:
            if not ok:
                print(f"   - {nm}")
    print("=" * 74)
    print()


if __name__ == "__main__":
    if "--secure-agg-only" not in sys.argv:
        main()
    if "--secure-agg" in sys.argv or "--secure-agg-only" in sys.argv:
        secure_aggregation_checks()
        print(f"RESULT INCLUDING SECURE AGGREGATION: {sum(ok for _, ok in RESULTS)} of {len(RESULTS)} checks passed")
    sys.exit(1 if any(not ok for _, ok in RESULTS) else 0)
