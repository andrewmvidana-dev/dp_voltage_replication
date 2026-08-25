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
import warnings

import numpy as np
import opendssdirect as dss

from dpvolt.network import load_feeder, kron_reduce, injection_check
from dpvolt.loads import (assign_classes, make_historical, fit_load_model,
                          sample_loads, reactive_from_active, ar1_covariance,
                          diurnal_shape)
from dpvolt.powerflow import PowerFlowRunner, to_per_unit
from dpvolt.privacy import (gaussian_sigma, dp_fit_class, theorem1,
                            calibrate_M_inv, normalised_jacobian, solve_for_r)
from dpvolt.experiments import (voltage_wasserstein, build_masked_dataset,
                                Standardizer, train_and_curve)


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


def section(title):
    print()
    print("-" * 74)
    print(title)
    print("-" * 74)


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
    main()
