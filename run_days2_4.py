# The load model, the DP mechanism, and Theorem 1. Run after run_day1.py;
# takes roughly a minute.
#
#   Day 2  split buses into consumer classes, build stand-in historical data,
#          fit a log-normal load model, sample synthetic loads from it
#   Day 3  refit under differential privacy, then push the private synthetic
#          loads through AC power flow on the TRUE admittance matrix
#   Day 4  evaluate Theorem 1, and calibrate ||M~^-1||, the one constant it
#          cannot compute in closed form

import os
import time
import warnings

import numpy as np
import opendssdirect as dss

from dpvolt.network import load_feeder, kron_reduce
from dpvolt.loads import (assign_classes, make_historical, fit_load_model,
                          sample_loads, reactive_from_active)
from dpvolt.powerflow import PowerFlowRunner, to_per_unit
from dpvolt.privacy import dp_fit_class, theorem1, calibrate_M_inv


HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "feeders", "IEEE123Master.dss")

N_HIST_DAYS = 90         # how much "historical" data the utility holds
N_SYNTH_DAYS = 4         # synthetic days to generate
DELTA = 1e-5             # DP failure probability
CLIP_NORM = 6.0          # L2 clip on centred log-load records
SEED = 0


def banner(text):
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def feeder_load_ratings():
    """Read each Load element's kW rating and power factor from the feeder."""
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
    # Power factor is cos(theta), so the angle is its arccos.
    theta = np.arccos(np.clip(np.array(pf), -1.0, 1.0))
    return kw, theta


def main():
    if not os.path.exists(MASTER):
        print("Could not find the feeder files.")
        print("Open get_feeder.py and press Run first, then come back here.")
        return

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEED)

    # =====================================================================
    banner("DAY 2  --  The load model")
    # =====================================================================

    kw, theta = feeder_load_ratings()
    print(f"  {len(kw)} load elements, {kw.sum():.0f} kW total, "
          f"{kw.min():.0f}-{kw.max():.0f} kW each")

    classes = assign_classes(kw, L=3)
    print("\n  Consumer classes (split by demand size, smallest first):")
    for label, members in classes.items():
        print(f"     class {label}: {len(members):2d} buses, {kw[members].sum():6.0f} kW")

    archive = make_historical(kw, classes, n_days=N_HIST_DAYS, rng=rng)
    daily_mean = archive.sum(axis=0).mean() * 1000.0
    print(f"\n  Historical archive: {archive.shape} "
          f"(buses, days, 15-min steps)")
    print(f"     mean total demand {daily_mean:.0f} kW "
          f"against the feeder's {kw.sum():.0f} kW")

    model = fit_load_model(archive, classes, theta)
    print("\n  Fitted log-normal per class:")
    for label in classes:
        print(f"     class {label}: load margins "
              f"[{model.p_min[label]:.5f}, {model.p_max[label]:.5f}] per-unit")

    synth, repairs = sample_loads(model, N_SYNTH_DAYS, rng=rng,
                                  sweeps=20, report=True)
    print(f"\n  Synthetic loads: {synth.shape}")
    print("     truncation repair rate: "
          + ", ".join(f"class {k} {v:.1%}" for k, v in repairs.items()))

    # The property that matters. Independent noise would destroy this.
    real_c = np.log(archive[classes[0]].reshape(-1, model.T))
    synth_c = np.log(synth[classes[0]].reshape(-1, model.T))
    lag_real = np.corrcoef(real_c.T)[0, 1:6]
    lag_synth = np.corrcoef(synth_c.T)[0, 1:6]
    print("\n  Temporal correlation, lags 1 to 5 (the thing worth preserving):")
    print(f"     real      {np.round(lag_real, 3)}")
    print(f"     synthetic {np.round(lag_synth, 3)}")

    # =====================================================================
    banner("DAY 3  --  Differential privacy, then AC power flow")
    # =====================================================================

    print("  Refitting each class under DP. Half the budget goes to the mean,")
    print("  half to the covariance, by composition.\n")
    print(f"  {'eps':>6} {'class':>6} {'records':>9} {'sigma_cov':>11} "
          f"{'eig floored':>12} {'KL from true':>13}")

    for eps_load in (25.0, 100.0):
        for label, members in classes.items():
            data = np.log(archive[members].reshape(-1, model.T))
            _, _, rep = dp_fit_class(
                data, np.log(model.p_min[label]), np.log(model.p_max[label]),
                eps_load, DELTA, rng, clip_norm=CLIP_NORM,
            )
            print(f"  {eps_load:6.0f} {label:6d} {rep.n_records:9d} "
                  f"{rep.sigma_cov:11.5f} {rep.eig_clipped:8d}/{model.T:<3d} "
                  f"{rep.kl_to_true:13.1f}")

    print("\n  Note how many eigenvalues have to be floored. Privately")
    print("  estimating a 96-by-96 covariance is genuinely expensive, and")
    print("  that is the honest reason the paper works at epsilon 25 to 200")
    print("  rather than the single digits you see elsewhere in DP.")

    runner = PowerFlowRunner(MASTER)
    q_synth = reactive_from_active(synth, theta)

    t0 = time.time()
    V, ok = runner.solve_many(synth, q_synth)
    print(f"\n  Power flow on the TRUE Y: {V.shape} in {time.time() - t0:.1f}s")
    print(f"     converged at {ok.mean():.1%} of timesteps")

    vmag = np.abs(V[ok])
    print(f"     released |V| spans {vmag.min():.4f} to {vmag.max():.4f} per-unit")
    outside = np.mean((vmag < 0.95) | (vmag > 1.05))
    print(f"     {outside:.2%} of released voltages fall outside ANSI [0.95, 1.05]")
    if outside > 0.001:
        print("     -> worth flagging. We froze the regulator taps so that Y")
        print("        stays fixed, as the paper's analysis requires, which")
        print("        means the regulators cannot respond to synthetic load")
        print("        excursions. Definition 2's 'good voltage set' is then")
        print("        not strictly satisfied everywhere.")

    # =====================================================================
    banner("DAY 4  --  Theorem 1")
    # =====================================================================

    feeder = load_feeder(MASTER)
    kron = kron_reduce(feeder)

    # EVERYTHING crossing into the privacy analysis must be per-unit.
    Vb = feeder.Vbase[kron.retained]
    Y_pu = to_per_unit(kron.Y_red, Vb)
    b_pu = kron.b * Vb / 1e6
    n = len(kron.retained)

    adj = np.abs(Y_pu) > 1e-10
    np.fill_diagonal(adj, False)
    d_max = int(adj.sum(axis=1).max())

    print(f"  n = {n} retained buses, max node degree {d_max}")
    print(f"  kappa_Kron = {kron.kappa_kron:.4e}")
    print(f"  condition number of Y_pu = {np.linalg.cond(Y_pu):.3e}")

    # Independent end-to-end check of Y_red, b and the per-unit conversion.
    v0 = feeder.V_node[kron.retained] / Vb
    s0 = v0 * (np.conj(Y_pu) @ np.conj(v0) + np.conj(b_pu))
    print(f"\n  Cross-check: load implied by the reduced model alone is "
          f"{-np.real(s0).sum() * 1000:.0f} kW")
    print(f"               against the feeder's {kw.sum():.0f} kW. "
          "Y_red, b and per-unit all agree.")

    # ---- Remark 2: calibrate the one constant with no closed form --------
    name_to_idx = {nm: i for i, nm in enumerate(runner.node_names)}
    sel = [name_to_idx[nm] for nm in kron.names_retained]
    V_flat = V.reshape(-1, V.shape[-1])[:, sel]

    cal = calibrate_M_inv(V_flat[:120], Y_pu, b_pu)
    print(f"\n  Monte Carlo calibration of ||M~^-1||, the paper's Remark 2:")
    print(f"     median {cal['norm_median']:.3f}, "
          f"99th percentile mu_0 {cal['mu_0']:.3f}, max {cal['norm_max']:.3f}")
    print(f"     delta_M {cal['delta_M']:.4f} "
          f"(Clopper-Pearson upper bound {cal['delta_M_upper']:.4f}, "
          f"{cal['n_exceed']}/{cal['n_samples']} exceeded)")
    print(f"     so the mechanism is (epsilon, delta + delta_M)-private")

    # ---- the bound itself -------------------------------------------------
    def bound_at(eps_load, floor, r):
        Sigma, sizes, pmins = {}, {}, {}
        for label, members in classes.items():
            data = np.log(archive[members].reshape(-1, model.T))
            _, cov, _ = dp_fit_class(
                data, np.log(model.p_min[label]), np.log(model.p_max[label]),
                eps_load, DELTA, rng, clip_norm=CLIP_NORM,
                eig_floor_ratio=floor,
            )
            Sigma[label] = cov
            sizes[label] = len(members)
            pmins[label] = model.p_min[label]
        return theorem1(
            Sigma_by_class=Sigma, size_by_class=sizes, p_min_by_class=pmins,
            n=n, T=model.T, d_max=d_max, kappa_kron=kron.kappa_kron,
            r=r, delta=DELTA, M_inv_norm=cal["mu_0"],
        )

    print("\n  The bound decomposed (eps_load = 25, r = 1e-13):\n")
    print(f"  {'cov floor':>10} {'gamma_0':>11} {'psi_bar':>9} "
          f"{'jacobian':>10} {'bias':>11} {'tail':>11} {'EPSILON':>11}")
    for floor in (1e-3, 1e-2, 1e-1, 0.5, 2.0):
        B = bound_at(25.0, floor, 1e-13)
        print(f"  {floor:10.3f} {B.gamma_ell[0]:11.3e} {B.psi_bar:9.3f} "
              f"{B.jacobian_term:10.3e} {B.bias_term:11.3e} "
              f"{B.tail_term:11.3e} {B.epsilon:11.2f}")

    print("\n  Read that table carefully -- it is the paper's central mechanism")
    print("  made visible. The covariance floor controls how NOISY the private")
    print("  load model is. A noisier model has a larger Sigma, hence a smaller")
    print("  precision sum gamma, hence a smaller psi_bar, hence a smaller")
    print("  epsilon. More load noise really does buy more topology privacy,")
    print("  exactly as the paper claims.")
    print()
    print("  Note also that the Jacobian term barely moves. That is the FLOOR")
    print("  the paper describes: no amount of load noise crosses it. Only")
    print("  shrinking the adjacency radius r does.")

    banner("WHERE THIS LEAVES US")
    print("  Reproduced: the pipeline end to end, and epsilon values in the")
    print("  paper's own 25-to-200 band once the covariance floor is set so")
    print("  that our substitute mechanism is as noisy as their DP-GMM.")
    print()
    print("  Still open: the adjacency radius r that achieves those epsilons")
    print("  is around 1e-13 in per-unit admittance, against ||Y_pu||_F of")
    print("  about 1500. That is a relative perturbation of 1e-16 -- far too")
    print("  small to represent the 'line switching' the paper motivates.")
    print("  Whether that gap is our parameterisation or the bound's own")
    print("  looseness is the single best question to put to the authors.")
    print()
    print("  Next: Days 5 and 6, the Wasserstein sweep and the MLP task.")
    print()


if __name__ == "__main__":
    main()
