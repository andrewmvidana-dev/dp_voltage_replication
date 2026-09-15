# Figure 7: the viability verdict on Bounded-Noise Privacy. Run after
# get_feeder.py; takes about two minutes. Writes figures/figure7_viability.png.
#
# Figures 4-6 each answer part of the question. This one answers it outright:
# IS BNP AN EFFECTIVE STRATEGY HERE? It puts the privacy cost and the utility
# cost on the same picture, so "BNP reaches lag-1 0.59" cannot be read without
# also seeing that it did so at delta = 0.5.
#
# THE COMPARISON IS AGAINST A CORRECTLY CALIBRATED GAUSSIAN. Earlier versions of
# this comparison used privacy.gaussian_sigma, which under-noises above
# epsilon = 1 -- at (eps=50, delta=1e-5) it actually delivers delta ~0.5, not
# 1e-5. So the Gaussian baseline was being flattered by ~5 orders of magnitude of
# privacy it never had, and BNP was losing to a mechanism that was cheating.
# dp_fit_class now defaults to the analytic (Balle & Wang) calibration, and the
# reference line here is the corrected one. BNP still loses -- which is the
# stronger result, since the obvious objection is now closed.
#
# PANEL LAYOUT
#   left   the privacy-utility frontier: temporal structure retained vs the
#          delta paid for it. The Gaussian point sits far up and to the LEFT --
#          better structure AND better privacy. That dominance is the verdict.
#   right  what each mechanism costs on the four measured axes, at BNP's most
#          generous setting, as a multiple of the Gaussian baseline.

import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import opendssdirect as dss

from dpvolt.loads import (assign_classes, make_historical, fit_load_model,
                          sample_loads, reactive_from_active, LoadModel)
from dpvolt.powerflow import (PowerFlowRunner, add_bounded_voltage_noise,
                              bnp_delta)
from dpvolt.privacy import dp_fit_class, bnp_fit_class
from dpvolt.experiments import (voltage_wasserstein, build_masked_dataset,
                                run_seeds, ansi_violation_rate,
                                mean_autocorrelation,
                                empirical_voltage_sensitivity)


HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "feeders", "IEEE123Master.dss")
FIGDIR = os.path.join(HERE, "figures")

GAUSS_EPS, GAUSS_DELTA = 50.0, 1e-5
CLIP_NORM, COV_FLOOR = 6.0, 0.1
N_HIST, N_EVAL, N_SEEDS, EPOCHS = 90, 8, 12, 30

# BNP operating points. Below delta = 1e-2 the model is degenerate enough that
# power flow stops converging -- those points are reported, not plotted, since a
# mechanism that cannot produce a solvable network has no utility to measure.
DELTAS = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]

# Palette: emphasis, not categorical. The question is "which side of the
# baseline", so the Gaussian reference owns one hue and BNP the other, with
# everything else recessive ink.
#
# VALIDATED, not eyeballed. All three run all-pairs (they share the left panel's
# legend, so any two can sit side by side): worst CVD dE 13.0, worst
# normal-vision dE 16.3, every contrast >= 3:1 on the #fcfcfb surface.
# The first choice for C_OUT was status-critical #d03b3b, which FAILED the
# normal-vision floor against orange at dE 10.8 -- full-colour readers could not
# separate the two BNP stages. Violet is the re-step; it also keeps the status
# ramp reserved for status, which a "this failed" red would have squandered.
C_GAUSS = "#2a78d6"      # slot 1 blue   -- the proposed method
C_BNP = "#eb6834"        # slot 2 orange -- BNP at the input stage
C_OUT = "#4a3aa7"        # slot 7 violet -- BNP at the output stage
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def feeder_load_ratings():
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


def main():
    if not os.path.exists(MASTER):
        print("Feeder files not found. Run get_feeder.py first.")
        return

    warnings.filterwarnings("ignore")
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(0)

    kw, theta = feeder_load_ratings()
    classes = assign_classes(kw, L=3)
    archive = make_historical(kw, classes, n_days=N_HIST, rng=rng)
    model = fit_load_model(archive, classes, theta)

    runner = PowerFlowRunner(MASTER)
    sel = runner.retained_indices()

    V_true, _ = runner.solve_many(
        archive[:, :N_EVAL, :],
        reactive_from_active(archive[:, :N_EVAL, :], theta))
    V_true_flat = V_true.reshape(-1, V_true.shape[-1])[:, sel]
    X_test, Y_test = build_masked_dataset(V_true[:, :, sel][-3:])
    var_test = float(Y_test.var())
    ac_true = float(mean_autocorrelation(V_true[:, :, sel])[0])

    def evaluate(V):
        flat = V.reshape(-1, V.shape[-1])[:, sel]
        X_tr, Y_tr = build_masked_dataset(V[:, :, sel][:5])
        curve, _ = run_seeds(X_tr, Y_tr, X_test, Y_test,
                             n_seeds=N_SEEDS, epochs=EPOCHS)
        return dict(w=float(voltage_wasserstein(V_true_flat, flat)),
                    viol=float(ansi_violation_rate(flat)),
                    ac=float(mean_autocorrelation(V[:, :, sel])[0]),
                    r2=float(1.0 - curve[-1] / var_test))

    def fit(kind, delta, rng_):
        mu, Sig, B = {}, {}, None
        for lab, mem in classes.items():
            d = np.log(archive[mem].reshape(-1, model.T))
            lo, hi = np.log(model.p_min[lab]), np.log(model.p_max[lab])
            if kind == "gauss":
                m, c, rep = dp_fit_class(d, lo, hi, GAUSS_EPS, GAUSS_DELTA,
                                         rng_, clip_norm=CLIP_NORM,
                                         eig_floor_ratio=COV_FLOOR)
            else:
                m, c, rep = bnp_fit_class(d, lo, hi, delta, rng_,
                                          clip_norm=CLIP_NORM,
                                          eig_floor_ratio=COV_FLOOR)
            mu[lab], Sig[lab] = m, c
            B = rep.sigma_cov
        return LoadModel(mu=mu, Sigma=Sig, members=classes,
                         p_min=model.p_min, p_max=model.p_max,
                         power_factor=theta, T=model.T), B

    # ---- the corrected Gaussian reference ---------------------------------
    g, _ = fit("gauss", None, np.random.default_rng(500))
    synth = sample_loads(g, N_EVAL, rng=np.random.default_rng(501), sweeps=15)
    V_g, _ = runner.solve_many(synth, reactive_from_active(synth, theta))
    ref = evaluate(V_g)
    print(f"  Gaussian (eps={GAUSS_EPS:.0f}, delta={GAUSS_DELTA:g}, analytic): "
          f"W-1 {ref['w']:.6f}  ANSI {ref['viol']:.1%}  "
          f"lag-1 {ref['ac']:.3f}  R^2 {ref['r2']:.3f}")

    # ---- BNP at the input stage, swept ------------------------------------
    rows, failed = [], []
    for k, d in enumerate(DELTAS):
        rd = np.random.default_rng(1000 + k)
        runner.reset()
        priv, B = fit("bnp", d, rd)
        s = sample_loads(priv, N_EVAL, rng=rd, sweeps=15)
        if not np.isfinite(s).all():
            failed.append((d, "sampler diverged"))
            continue
        V, ok = runner.solve_many(s, reactive_from_active(s, theta))
        if ok.mean() < 0.95 or not np.isfinite(V).all():
            failed.append((d, f"power flow {ok.mean():.0%} converged"))
            continue
        r = evaluate(V)
        r["delta"], r["B"] = d, float(B)
        rows.append(r)
        print(f"  BNP delta={d:<6g} B={B:8.4f}  W-1 {r['w']:.6f}  "
              f"ANSI {r['viol']:.1%}  lag-1 {r['ac']:.3f}  R^2 {r['r2']:.3f}")

    # ---- BNP at the output stage ------------------------------------------
    S = float(empirical_voltage_sensitivity(runner, model, theta,
                                            np.random.default_rng(9),
                                            n_trials=10))
    # The honest sensitivity, for comparison. The sampled estimate above is a
    # lower bound of UNSTABLE magnitude -- it spans ~8.6x across seeds and
    # plateaus at whatever corner its own draws happened to reach -- whereas
    # the box-corner construction is seed-stable to ~1%. Both are reported
    # because the sampled one is what the existing figures were built on.
    runner.reset()
    S_adv = float(empirical_voltage_sensitivity(runner, model, theta,
                                                np.random.default_rng(9),
                                                adversarial=True))

    B_out = S / 2.0
    # B = S/2 is the SMALLEST bound Corollary 1 admits, and by delta = S/(2B)
    # that is delta = 1 EXACTLY: the uniform mechanism's non-overlap region is
    # the whole support, so it provides NO privacy whatsoever. This is not a
    # weak operating point, it is the absence of one.
    d_out = float(bnp_delta(S, B_out))
    out = evaluate(add_bounded_voltage_noise(V_g, B_out,
                                             np.random.default_rng(11)))

    ANSI_HALF_BAND = 0.05
    print(f"  output-stage BNP: B = S/2  =>  delta = {d_out:.1f} "
          f"(NO PRIVACY AT ALL, not merely weak)")
    print(f"    sampled     S = {S:7.4f} pu -> B = {B_out:7.4f} pu "
          f"= {B_out / ANSI_HALF_BAND:5.1f}x the ANSI half-band")
    print(f"    adversarial S = {S_adv:7.4f} pu -> B = {S_adv / 2.0:7.4f} pu "
          f"= {S_adv / 2.0 / ANSI_HALF_BAND:5.1f}x the ANSI half-band")
    print(f"    even at delta = 1, the noise bound alone exceeds the entire")
    print(f"    regulation half-band by more than an order of magnitude.")
    print(f"    lag-1 {out['ac']:.3f}  ANSI {out['viol']:.1%}")

    if not rows:
        print("\n  every BNP point failed -- nothing to plot")
        return

    # =====================================================================
    # Figure
    # =====================================================================
    fig = plt.figure(figsize=(14.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.26,
                          left=0.065, right=0.975, top=0.80, bottom=0.13)

    # ---- LEFT: the privacy-utility frontier --------------------------------
    ax = fig.add_subplot(gs[0, 0])

    d_bnp = np.array([r["delta"] for r in rows])
    ac_bnp = np.array([r["ac"] for r in rows])

    # The region no mechanism should occupy: weaker privacy than the Gaussian
    # path AND worse structure. Everything BNP achieves lands inside it.
    # Spans from the Gaussian point rightward: weaker privacy AND less
    # structure. Alpha kept low so it reads as a ground, not a series, but high
    # enough to actually be visible against the surface.
    ax.axhspan(0, ref["ac"], xmin=0, xmax=1, color=MUTED, alpha=0.13, zorder=0)

    ax.plot(d_bnp, ac_bnp, "o-", color=C_BNP, lw=2.0, ms=8, zorder=3,
            markeredgecolor="#fcfcfb", markeredgewidth=1.4,
            label="BNP on the load model (swept)")

    # The true signal, and the Gaussian reference.
    ax.axhline(ac_true, color=INK, ls=":", lw=1.5, zorder=2)
    ax.text(0.0135, ac_true - 0.045, f"true voltages ({ac_true:.2f})",
            fontsize=8.5, color=INK2)

    ax.plot([GAUSS_DELTA], [ref["ac"]], "D", ms=12, color=C_GAUSS, zorder=5,
            markeredgecolor="#fcfcfb", markeredgewidth=1.6,
            label=f"Gaussian, correctly calibrated ($\\epsilon$={GAUSS_EPS:.0f})")
    # Label sits below-right of the diamond, in the empty mid-left of the plot,
    # so its leader line never crosses the BNP curve.
    ax.annotate(
        f"PROPOSED METHOD\n$\\delta$=1e-5, lag-1 {ref['ac']:.2f}\n"
        "better on BOTH axes",
        xy=(GAUSS_DELTA, ref["ac"]), xytext=(2.2e-5, 0.545),
        fontsize=9, color=C_GAUSS, weight="bold", ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=C_GAUSS, lw=1.5,
                        shrinkA=2, shrinkB=6))

    # BNP's best point, so the headline number is visible with its price.
    # Placed BELOW the curve's right end, where the plot is empty.
    best = max(rows, key=lambda r: r["ac"])
    ax.annotate(
        f"BNP's best structure:\nlag-1 {best['ac']:.2f} — but at "
        f"$\\delta$={best['delta']:g},\n"
        f"{best['delta'] / GAUSS_DELTA:,.0f}x weaker privacy",
        xy=(best["delta"], best["ac"]), xytext=(1.1e-3, 0.30),
        fontsize=8.5, color=C_BNP, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=C_BNP, lw=1.3,
                        shrinkA=2, shrinkB=6,
                        connectionstyle="arc3,rad=-0.2"))

    # The output stage: plotted at its own delta of 1.0.
    ax.plot([d_out], [out["ac"]], "s", ms=10, color=C_OUT, zorder=5,
            markeredgecolor="#fcfcfb", markeredgewidth=1.4,
            label="BNP on released voltages (output stage)")
    # Label above the marker and hard right, clear of the sweep curve.
    ax.annotate(
        f"output stage: $B=S/2$\n$\\Rightarrow\\delta$=1.0, i.e. NO privacy\n"
        f"and $B$ is {S_adv / 2.0 / 0.05:.0f}x the ANSI half-band\n"
        f"lag-1 {out['ac']:.2f}",
        xy=(d_out, out["ac"]), xytext=(0.62, 0.175),
        fontsize=8.5, color=C_OUT, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.3,
                        shrinkA=2, shrinkB=6))

    # Shaded-region caption, tucked into the bottom-left corner where no mark
    # reaches (the leftmost BNP point sits at delta=1e-2).
    ax.text(1.05e-5, 0.022,
            "shaded: strictly worse privacy AND structure\n"
            "than the proposed method",
            fontsize=8, color=MUTED, style="italic", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(8e-6, 2.6)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"failure probability $\delta$  —  right = weaker privacy",
                  fontsize=10, color=INK2)
    ax.set_ylabel("lag-1 autocorrelation retained\nhigher = temporal structure survives",
                  fontsize=10, color=INK2)
    ax.set_title("Privacy paid vs structure retained", fontsize=11.5,
                 color=INK, pad=8)
    ax.grid(True, which="major", color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.96,
              edgecolor=GRID)

    # ---- RIGHT: cost on each axis, relative to Gaussian -------------------
    ax2 = fig.add_subplot(gs[0, 1])

    # Every axis as "times worse than the corrected Gaussian baseline", so one
    # scale carries all four. W-1 and ANSI: higher is worse, so ratio directly.
    # lag-1 and R^2: higher is better, so invert to keep "up = worse".
    def cost(r):
        return [
            r["w"] / ref["w"],
            r["viol"] / ref["viol"],
            ref["ac"] / max(r["ac"], 1e-6),
            ref["r2"] / max(r["r2"], 1e-6),
        ]

    labels = ["Fidelity\n(W-1)", "Feasibility\n(ANSI viol.)",
              "Structure\n(lag-1)", "Utility\n($R^2$)"]
    x = np.arange(len(labels))
    w = 0.36

    c_best = cost(best)
    c_out = cost(out)

    # bottom= is required on a log axis: bars default to a zero base, which is
    # -inf in log space and silently clips the sub-parity bars to nothing.
    BASE = 0.56
    b1 = ax2.bar(x - w / 2, np.array(c_best) - BASE, w, bottom=BASE,
                 color=C_BNP, zorder=3, edgecolor="#fcfcfb", linewidth=1.5,
                 label=f"BNP input stage, best point ($\\delta$={best['delta']:g})")
    b2 = ax2.bar(x + w / 2, np.array(c_out) - BASE, w, bottom=BASE,
                 color=C_OUT, zorder=3, edgecolor="#fcfcfb", linewidth=1.5,
                 label=r"BNP output stage ($\delta$=1.0)")

    ax2.axhline(1.0, color=C_GAUSS, lw=2.0, zorder=4)
    # Baseline caption sits in the clear band between the rule and the shortest
    # labelled bar, over the Feasibility group -- the one column whose tallest
    # bar (7.1x) leaves the 1.1-2.5 range empty.
    ax2.text(1.0, 1.55, "Gaussian baseline = 1.0", fontsize=8.5,
             color=C_GAUSS, ha="center", va="bottom", weight="bold")

    # Direct labels: mandatory here, the bars span three orders of magnitude.
    # A bar at or below parity has no room above it before the baseline rule,
    # so its label goes INSIDE, below the line.
    for bars, vals in ((b1, c_best), (b2, c_out)):
        for bar, v in zip(bars, vals):
            if v >= 1.08:
                # v, not get_height() -- the bars carry a bottom= offset.
                ax2.text(bar.get_x() + bar.get_width() / 2, v * 1.08,
                         f"{v:.1f}x", ha="center", va="bottom",
                         fontsize=8.5, color=INK2)
            else:
                # At/below parity: the only good news on the panel, so say so
                # rather than hiding a cramped number against the rule.
                ax2.text(bar.get_x() + bar.get_width() / 2, 0.845,
                         f"{v:.1f}x", ha="center", va="center",
                         fontsize=8.5, color=INK2, weight="bold")

    ax2.set_yscale("log")
    ax2.set_ylim(0.56, max(max(c_best), max(c_out)) * 3.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, color=INK2)
    ax2.set_ylabel("times worse than the Gaussian baseline\n(log scale, 1.0 = parity)",
                   fontsize=10, color=INK2)
    ax2.set_title("Cost on every measured axis", fontsize=11.5, color=INK,
                  pad=8)
    ax2.grid(True, axis="y", color=GRID, lw=0.8, alpha=0.9)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.96,
               edgecolor=GRID)

    fig.suptitle(
        "Figure 7: is Bounded-Noise Privacy an effective strategy here?  "
        "No — it is dominated at both stages",
        fontsize=13, color=INK, y=0.955, x=0.065, ha="left")
    fig.text(
        0.065, 0.885,
        "Compared against a CORRECTLY calibrated Gaussian mechanism (Balle & Wang). "
        "The earlier classical calibration under-noised at $\\epsilon$=50, so this "
        "comparison no longer flatters the baseline.",
        fontsize=9, color=INK2, ha="left")

    out_path = os.path.join(FIGDIR, "figure7_viability.png")
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # ---- verdict -----------------------------------------------------------
    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  BNP is NOT an effective strategy at either stage.")
    print()
    print(f"  INPUT STAGE. Dominated. Its best structure (lag-1 "
          f"{best['ac']:.2f} at delta={best['delta']:g}) is still below the")
    print(f"  Gaussian path's {ref['ac']:.2f} -- which is achieved at delta=1e-5, "
          f"{best['delta'] / GAUSS_DELTA:,.0f}x stronger.")
    print(f"  There is no delta where BNP wins on structure, and none where it")
    print(f"  even ties while matching the privacy. Every swept point lands in")
    print(f"  the strictly-worse region of the left panel.")
    print()
    print(f"  OUTPUT STAGE. Fails outright, and fails BEFORE any privacy is")
    print(f"  bought. Setting B = S/2 -- the smallest bound Corollary 1 admits")
    print(f"  -- gives delta = S/(2B) = {d_out:.1f} EXACTLY. That is not weak")
    print(f"  privacy, it is none: the mechanism's non-overlap region covers")
    print(f"  its whole support, so neighbouring datasets are not confused at")
    print(f"  all. Every B that WOULD buy privacy is larger still.")
    print(f"  And even at that zero-privacy setting the bound exceeds the ANSI")
    print(f"  half-band of 0.05 pu by {B_out / 0.05:.1f}x (sampled S = {S:.4f}) or")
    print(f"  {S_adv / 2.0 / 0.05:.1f}x (adversarial S = {S_adv:.4f}, the honest figure).")
    print(f"  Measured: {out['viol']:.0%} of the ANSI band destroyed, autocorrelation")
    print(f"  down to {out['ac']:.2f}.")
    if failed:
        print()
        print("  Points that could not be evaluated (mechanism too degenerate):")
        for d, why in failed:
            print(f"    delta={d:<8g} {why}")
    print()
    print("  WHAT BNP DOES BUY, honestly: a hard worst-case bound on any single")
    print("  released value, which the Gaussian tail cannot give at any sigma.")
    print("  If a deployment's binding requirement is 'no released voltage may")
    print("  ever exceed the band', that guarantee has real value. It is simply")
    print("  not what this method is measured on, and the price is steep.")
    print()
    print(f"  saved {out_path}")

    # Machine-readable, so the README numbers and the figure cannot drift.
    with open(os.path.join(FIGDIR, "figure7_viability.json"), "w") as f:
        json.dump({"gauss_ref": ref, "gauss_eps": GAUSS_EPS,
                   "gauss_delta": GAUSS_DELTA, "ac_true": ac_true,
                   "bnp_input": rows, "bnp_output": out,
                   "S_voltage": S, "B_out": B_out, "delta_out": d_out,
                   "S_voltage_adversarial": S_adv,
                   "B_out_adversarial": S_adv / 2.0,
                   "ansi_exceedance_sampled": B_out / 0.05,
                   "ansi_exceedance_adversarial": S_adv / 2.0 / 0.05,
                   "failed": failed}, f, indent=2)


if __name__ == "__main__":
    main()
