# Figures 2 and 3. Run after run_days2_4.py; takes about two minutes and writes
# two PNGs into figures/ alongside the numbers behind them.
#
#   figure2_wasserstein.png   statistical fidelity against privacy budget
#   figure3_recovery.png      does the released data train a useful model?
#
# Our historical data is manufactured rather than OEDI, and our DP mechanism
# substitutes for DP-GMM, so expect the ORDERING and shape of the curves to
# match the paper but not the absolute numbers.

import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")          # render to file; no interactive window needed
import matplotlib.pyplot as plt
import opendssdirect as dss

from dpvolt.loads import (assign_classes, make_historical, fit_load_model,
                          sample_loads, reactive_from_active, LoadModel)
from dpvolt.powerflow import (PowerFlowRunner, add_voltage_noise,
                              add_bounded_voltage_noise, bnp_bound, bnp_delta)
from dpvolt.privacy import dp_fit_class, gaussian_sigma
from dpvolt.experiments import (voltage_wasserstein, empirical_voltage_sensitivity,
                                build_masked_dataset, run_seeds,
                                ansi_violation_rate, mean_autocorrelation)


HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "feeders", "IEEE123Master.dss")
FIGDIR = os.path.join(HERE, "figures")

EPSILONS = [25.0, 30.0, 50.0, 100.0, 200.0]    # the paper's sweep
DELTA = 1e-5
CLIP_NORM = 6.0
COV_FLOOR = 0.1        # puts our substitute mechanism in the paper's regime
N_HIST_DAYS = 90
N_EVAL_DAYS = 12
N_SEEDS = 20
EPOCHS = 30
SEED = 0


def banner(text):
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def feeder_load_ratings():
    """Each Load element's kW rating and power factor angle, from the feeder."""
    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {MASTER}")
    dss.Text.Command("Solve")
    kw, pf = [], []
    i = dss.Loads.First()
    while i > 0:
        kw.append(dss.Loads.kW())
        pf.append(dss.Loads.PF())
        i = dss.Loads.Next()
    return np.array(kw), np.arccos(np.clip(np.array(pf), -1.0, 1.0))


def dp_model(archive, classes, model, theta, epsilon, rng):
    """Refit the load model under DP at a given epsilon.

    Returns a LoadModel carrying the PRIVATE mean and covariance, so that
    sampling from it produces private synthetic loads.
    """
    mu, Sigma = {}, {}
    for label, members in classes.items():
        data = np.log(archive[members].reshape(-1, model.T))
        m, cov, _ = dp_fit_class(
            data, np.log(model.p_min[label]), np.log(model.p_max[label]),
            epsilon, DELTA, rng, clip_norm=CLIP_NORM,
            eig_floor_ratio=COV_FLOOR,
        )
        mu[label], Sigma[label] = m, cov

    return LoadModel(
        mu=mu, Sigma=Sigma, members=classes,
        p_min=model.p_min, p_max=model.p_max,
        power_factor=theta, T=model.T,
    )


def main():
    if not os.path.exists(MASTER):
        print("Feeder files not found. Run get_feeder.py first.")
        return

    warnings.filterwarnings("ignore")
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- shared setup -----------------------------------------------------
    kw, theta = feeder_load_ratings()
    classes = assign_classes(kw, L=3)
    archive = make_historical(kw, classes, n_days=N_HIST_DAYS, rng=rng)
    model = fit_load_model(archive, classes, theta)

    runner = PowerFlowRunner(MASTER)
    sel = runner.retained_indices()

    print(f"  feeder: {len(kw)} loads, {kw.sum():.0f} kW, "
          f"{len(sel)} monitored nodes")

    # The reference: real loads through real physics, no privacy anywhere.
    V_true, ok = runner.solve_many(
        archive[:, :N_EVAL_DAYS, :],
        reactive_from_active(archive[:, :N_EVAL_DAYS, :], theta),
    )
    print(f"  reference voltages {V_true.shape}, converged {ok.mean():.1%}")
    V_true_flat = V_true.reshape(-1, V_true.shape[-1])[:, sel]

    # Sensitivity for the output-perturbation baseline.
    t0 = time.time()
    delta_V = empirical_voltage_sensitivity(runner, model, theta, rng, n_trials=10)
    print(f"  empirical voltage sensitivity {delta_V:.4f} "
          f"({time.time() - t0:.1f}s)")

    # =====================================================================
    banner("Figure 2: Wasserstein-1 against privacy budget")
    # =====================================================================

    # ---- BNP: why we cannot match it to delta = 1e-5 ----------------------
    #
    # The uniform mechanism is (0, S/2B)-private, so it has no epsilon to match
    # on and delta is the only common axis. But delta = S/2B falls as 1/B, not
    # exponentially, so a cryptographically small delta demands an enormous
    # bound: at delta = 1e-5 we would need B = S/(2 delta) = 33497 per-unit,
    # against voltages of ~1.0. That is not a privacy mechanism, it is erasure.
    #
    # So we sweep B instead and report what each buys. This IS the finding:
    # bounded uniform noise cannot reach the Gaussian mechanism's delta regime
    # at any usable bound.
    B_valid = delta_V / 2.0                    # smallest B with S <= 2B
    B_match = bnp_bound(delta_V, DELTA)

    print(f"  Voltage sensitivity S = {delta_V:.4f} per-unit.\n")
    print("  Corollary 1 holds only for S <= 2B, so the SMALLEST admissible")
    print(f"  bound is B = S/2 = {B_valid:.4f} pu -- already a third of nominal")
    print("  voltage, and it buys only delta = 0.5. Tighter bounds have no")
    print("  valid (0, S/2B) guarantee at all.\n")

    print(f"  {'B (pu)':>10} {'delta':>12} {'W-1':>12} {'ANSI viol':>11}")
    for B in [B_valid, 0.5, 1.0, 5.0, 50.0]:
        V_tmp = add_bounded_voltage_noise(V_true_flat, B, rng)
        print(f"  {B:10.4f} {bnp_delta(delta_V, B):12.3e} "
              f"{voltage_wasserstein(V_true_flat, V_tmp):12.6f} "
              f"{ansi_violation_rate(V_tmp):10.2%}")

    print(f"\n  And delta = S/2B falls only as 1/B, so matching the Gaussian")
    print(f"  path's delta = {DELTA:g} would need B = {B_match:,.0f} pu against")
    print("  voltages of ~1.0. There is no usable operating point: every")
    print("  admissible bound already destroys the data, and every bound that")
    print("  preserves it is outside Corollary 1. On per-unit voltages, uniform")
    print("  BNP is not a viable substitute for the Gaussian mechanism.")
    print()

    # Curves use the most favourable admissible point -- the smallest B that
    # Corollary 1 permits. Anything tighter has no guarantee; anything looser
    # is strictly worse. This is BNP at its best on this data.
    B_bnp = B_valid
    print(f"  Curves below use B = {B_bnp:.4f} pu "
          f"(delta {bnp_delta(delta_V, B_bnp):.2f}) -- the smallest admissible")
    print("  bound, i.e. BNP shown at its most favourable.")
    print()

    print(f"  {'epsilon':>9} {'proposed':>12} {'gaussian':>12} {'bnp':>12} "
          f"{'g/p':>7} {'b/p':>7}")

    w_proposed, w_gaussian, w_bnp = [], [], []
    viol = {"proposed": [], "gaussian": [], "bnp": []}

    for eps in EPSILONS:
        # Proposed: fit privately, sample, push through the TRUE power flow.
        priv = dp_model(archive, classes, model, theta, eps, rng)
        synth = sample_loads(priv, N_EVAL_DAYS, rng=rng, sweeps=15)
        V_p, ok_p = runner.solve_many(synth, reactive_from_active(synth, theta))
        V_p_flat = V_p.reshape(-1, V_p.shape[-1])[:, sel]

        # Baseline: take the TRUE voltages and add calibrated noise.
        sigma = gaussian_sigma(delta_V, eps, DELTA)
        V_g_flat = add_voltage_noise(V_true_flat, sigma, rng)

        # BNP: same true voltages, bounded uniform noise. B does not depend on
        # epsilon, so this arm is flat by construction -- that is the point.
        V_b_flat = add_bounded_voltage_noise(V_true_flat, B_bnp, rng)

        wp = voltage_wasserstein(V_true_flat, V_p_flat)
        wg = voltage_wasserstein(V_true_flat, V_g_flat)
        wb = voltage_wasserstein(V_true_flat, V_b_flat)
        w_proposed.append(wp)
        w_gaussian.append(wg)
        w_bnp.append(wb)

        viol["proposed"].append(ansi_violation_rate(V_p_flat))
        viol["gaussian"].append(ansi_violation_rate(V_g_flat))
        viol["bnp"].append(ansi_violation_rate(V_b_flat))

        print(f"  {eps:9.0f} {wp:12.6f} {wg:12.6f} {wb:12.6f} "
              f"{wg / wp:6.1f}x {wb / wp:6.1f}x")

    plt.figure(figsize=(7, 4.5))
    plt.plot(EPSILONS, w_proposed, "o-", lw=2, label="Proposed (DP loads, true power flow)")
    plt.plot(EPSILONS, w_gaussian, "s--", lw=2, label="Gaussian output perturbation")
    plt.plot(EPSILONS, w_bnp, "^:", lw=2,
             label=f"BNP bounded (B={B_bnp}, $\\delta$={bnp_delta(delta_V, B_bnp):.2f})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("privacy budget $\\varepsilon$ of the load model")
    plt.ylabel("Wasserstein-1 distance to true voltages (per-unit)")
    plt.title("Figure 2: statistical fidelity of released voltages")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    f2 = os.path.join(FIGDIR, "figure2_wasserstein.png")
    plt.savefig(f2, dpi=150)
    plt.close()
    print(f"\n  saved {f2}")

    # ---- the physical-feasibility axis ------------------------------------
    print("\n  Voltages outside ANSI C84.1 [0.95, 1.05] -- the axis Wasserstein")
    print("  and masked recovery both miss:\n")
    print(f"  {'epsilon':>9} {'proposed':>12} {'gaussian':>12} {'bnp':>12}")
    for i, eps in enumerate(EPSILONS):
        print(f"  {eps:9.0f} {viol['proposed'][i]:11.2%} "
              f"{viol['gaussian'][i]:11.2%} {viol['bnp'][i]:11.2%}")

    print("\n  The proposed method's violations come from freezing the regulator")
    print("  taps (needed to keep Y fixed), not from noise -- so they do not")
    print("  shrink as epsilon grows. The Gaussian baseline's violations are")
    print("  intrinsic: an unbounded tail can put a released voltage anywhere,")
    print("  and at tight epsilon most of them land outside the band. BNP's are")
    print("  capped by B -- bounded by construction, at the cost of a delta")
    print("  orders of magnitude weaker than the Gaussian path's.")

    print("\n  The proposed curve is nearly FLAT in epsilon while the baseline")
    print("  falls steeply -- the paper's central claim made visible. The")
    print("  proposed method never adds noise to the voltages, so its error")
    print("  comes only from the load model being slightly off. The baseline's")
    print("  error is noise sprayed onto the published quantity itself, so it")
    print("  scales as 1/epsilon.")

    # =====================================================================
    banner("Figure 3: does the released data train a useful model?")
    # =====================================================================

    # Test set is ALWAYS real voltages. That is the point of the experiment.
    X_test, Y_test = build_masked_dataset(V_true[:, :, sel][-4:])
    print(f"  test set (real voltages): X{X_test.shape} Y{Y_test.shape}")

    eps_fig3 = 50.0
    curves = {}

    # 1. noise-free: the unattainable ceiling
    X_nf, Y_nf = build_masked_dataset(V_true[:, :, sel][:8])
    curves["Noise-free (not private)"] = run_seeds(
        X_nf, Y_nf, X_test, Y_test, n_seeds=N_SEEDS, epochs=EPOCHS)

    # 2. proposed
    priv = dp_model(archive, classes, model, theta, eps_fig3, rng)
    synth = sample_loads(priv, 8, rng=rng, sweeps=15)
    V_p, _ = runner.solve_many(synth, reactive_from_active(synth, theta))
    X_pr, Y_pr = build_masked_dataset(V_p[:, :, sel])
    curves[f"Proposed ($\\varepsilon$={eps_fig3:.0f})"] = run_seeds(
        X_pr, Y_pr, X_test, Y_test, n_seeds=N_SEEDS, epochs=EPOCHS)

    # 3. gaussian output perturbation at the same epsilon
    sigma = gaussian_sigma(delta_V, eps_fig3, DELTA)
    V_g = add_voltage_noise(V_true[:, :, sel][:8], sigma, rng)
    X_g, Y_g = build_masked_dataset(V_g)
    curves[f"Gaussian output pert. ($\\varepsilon$={eps_fig3:.0f})"] = run_seeds(
        X_g, Y_g, X_test, Y_test, n_seeds=N_SEEDS, epochs=EPOCHS)

    # 4. BNP: bounded, but still independent per timestep
    V_b = add_bounded_voltage_noise(V_true[:, :, sel][:8], B_bnp, rng)
    X_b, Y_b = build_masked_dataset(V_b)
    curves[f"BNP bounded (B={B_bnp:.3f})"] = run_seeds(
        X_b, Y_b, X_test, Y_test, n_seeds=N_SEEDS, epochs=EPOCHS)

    print(f"\n  trained {N_SEEDS} seeds x {EPOCHS} epochs per method")
    var_test = float(Y_test.var())
    print(f"\n  {'method':<38} {'test MSE':>12} {'R^2':>8}")
    for name, (mean, std) in curves.items():
        clean = name.replace("$\\varepsilon$", "eps")
        r2 = 1.0 - mean[-1] / var_test
        print(f"  {clean:<38} {mean[-1]:>10.3e} {r2:>8.3f}")
    print(f"\n  (R^2 = 1 means perfect recovery; R^2 = 0 means no better than")
    print(f"   predicting the average voltage. Test target variance {var_test:.3e}.)")

    plt.figure(figsize=(7, 4.5))
    for name, (mean, std) in curves.items():
        ep = np.arange(len(mean))
        line, = plt.plot(ep, mean, lw=2, label=name)
        plt.fill_between(ep, mean - std, mean + std,
                         alpha=0.2, color=line.get_color())
    plt.yscale("log")
    plt.xlabel("training epoch")
    plt.ylabel("test MSE on REAL voltages")
    plt.title("Figure 3: masked recovery, trained on released data")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    f3 = os.path.join(FIGDIR, "figure3_recovery.png")
    plt.savefig(f3, dpi=150)
    plt.close()
    print(f"\n  saved {f3}")

    print("\n  The gap between 'proposed' and 'noise-free' is the true cost of")
    print("  privacy on the task you actually care about. The gap to the")
    print("  Gaussian baseline is what the paper's method buys at equal eps.")

    # ---- why the ordering comes out that way ------------------------------
    print("\n  Mean voltage-magnitude autocorrelation, lags 1-5:\n")
    priv_f3 = dp_model(archive, classes, model, theta, eps_fig3, rng)
    synth_f3 = sample_loads(priv_f3, 8, rng=rng, sweeps=15)
    V_p_f3, _ = runner.solve_many(synth_f3, reactive_from_active(synth_f3, theta))

    for name, arr in (
        ("true (reference)", V_true[:, :, sel][:8]),
        ("proposed", V_p_f3[:, :, sel]),
        ("gaussian", V_g),
        ("bnp", V_b),
    ):
        ac = mean_autocorrelation(arr)
        print(f"    {name:<18} {np.round(ac, 3)}")

    print("\n  This is the mechanism behind Figure 3. Both output-perturbation")
    print("  baselines add noise independently per timestep, so both flatten")
    print("  the autocorrelation -- bounding the noise caps how far a voltage")
    print("  can stray, but does nothing for temporal structure. The proposed")
    print("  method never touches the voltages, so the correlation induced by")
    print("  the load model and the network survives.")

    banner("DONE")
    print(f"  Both figures are in {FIGDIR}/")
    print("  Next: run verify.py to check every correctness invariant.")
    print()


if __name__ == "__main__":
    main()
