# Figure 4: why uniform Bounded-Noise Privacy has no usable operating point on
# per-unit voltages. Run any time (no power flow needed, ~1 second); writes
# figures/figure4_bnp_wall.png.
#
# BNP (Severtson & Khajenejad) adds uniform noise on [-B, B] instead of drawing
# from an unbounded distribution, so a released value can never be more than B
# from the truth -- no Gaussian tail, no physically impossible output. On scalar
# vehicle speeds that works. On per-unit voltages it does not, and this figure
# shows why: the region that is both admissible and useful is empty.

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dpvolt.powerflow import bnp_delta, bnp_bound


HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

# Empirical L2 voltage sensitivity, from empirical_voltage_sensitivity() in
# run_days5_6.py: the largest observed ||V1 - V0|| when one bus's whole daily
# trajectory is replaced.
S = 0.6699

DELTA_TARGET = 1e-5      # the delta the Gaussian path achieves
V_NOMINAL = 1.0          # per-unit
ANSI_HALF_BAND = 0.05    # ANSI C84.1 allows [0.95, 1.05]


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    B_min = S / 2.0                      # smallest B satisfying S <= 2B
    B_for_target = bnp_bound(S, DELTA_TARGET)

    B = np.logspace(-3, 5, 400)
    delta = S / (2.0 * B)                # Corollary 1, plotted for all B

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # ---- the two forbidden regions ----------------------------------------
    # They OVERLAP between the ANSI half-band and B_min, and that overlap is
    # the whole point: there is no B that is both admissible and useful.
    #
    # Left of B_min: Corollary 1 does not hold, so no (0, S/2B) guarantee.
    ax.axvspan(B[0], B_min, color="#c0392b", alpha=0.10, zorder=0)
    # Right of the ANSI half-band: noise alone can push a released voltage out
    # of the regulation band regardless of the true value.
    ax.axvspan(ANSI_HALF_BAND, B[-1], color="#7f8c8d", alpha=0.10, zorder=0)
    # The overlap, hatched: forbidden on BOTH counts.
    ax.axvspan(ANSI_HALF_BAND, B_min, facecolor="#8e44ad", alpha=0.20, zorder=1,
               hatch="///", edgecolor="#6c3483", linewidth=0.0)

    # ---- delta(B) ----------------------------------------------------------
    # Solid only where Corollary 1 applies; dashed where it does not.
    ok = B >= B_min
    ax.plot(B[~ok], delta[~ok], color="#2c3e50", lw=2, ls=":", zorder=3)
    ax.plot(B[ok], delta[ok], color="#2c3e50", lw=2.5, zorder=3,
            label=r"$\delta = S/2B$  (Corollary 1)")

    # ---- reference lines ---------------------------------------------------
    ax.axhline(DELTA_TARGET, color="#2980b9", lw=1.6, ls="--", zorder=2,
               label=r"$\delta=10^{-5}$ (Gaussian path)")
    ax.axhline(1.0, color="#95a5a6", lw=1.2, ls="-", zorder=1)
    ax.text(B[0] * 1.4, 1.35, r"$\delta=1$: no privacy",
            fontsize=8, color="#7f8c8d")

    ax.axvline(B_min, color="#c0392b", lw=1.6, zorder=2)
    ax.axvline(ANSI_HALF_BAND, color="#7f8c8d", lw=1.6, ls="-.", zorder=2)

    # ---- the two operating points that matter ------------------------------
    ax.plot([B_min], [1.0], "o", ms=10, color="#c0392b", zorder=5,
            markeredgecolor="white", markeredgewidth=1.4)
    ax.annotate(
        f"best admissible point\nB = S/2 = {B_min:.3f} pu,  " + r"$\delta=1.0$" +
        "\n(no privacy at all, and\n83% of voltages outside ANSI)",
        xy=(B_min, 1.0), xytext=(3.0, 1.6),
        fontsize=8.5, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )

    ax.plot([B_for_target], [DELTA_TARGET], "s", ms=9, color="#2980b9", zorder=5,
            markeredgecolor="white", markeredgewidth=1.4)
    ax.annotate(
        f"to match the Gaussian path:\nB = {B_for_target:,.0f} pu"
        "\n" + r"$3\times10^{4}\!\times$ nominal voltage",
        xy=(B_for_target, DELTA_TARGET), xytext=(15.0, 3.5e-4),
        fontsize=8.5, color="#2980b9", ha="left",
        arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.2),
    )

    # ---- region labels -----------------------------------------------------
    # Each label sits over the region it describes.
    ax.text(np.sqrt(B[0] * ANSI_HALF_BAND), 2.5e-5,
            "NO GUARANTEE\n" + r"$S > 2B$" + "\nCorollary 1 fails",
            ha="center", va="center", fontsize=9, color="#c0392b", weight="bold")
    ax.text(np.sqrt(B_min * B[-1]), 2.5e-5,
            "DATA DESTROYED\n" + r"$B$ > ANSI half-band" +
            "\nnoise alone leaves the band",
            ha="center", va="center", fontsize=9, color="#5d6d7e", weight="bold")

    # The overlap between the two thresholds -- forbidden twice over. That
    # this interval is non-empty (0.05 < 0.335) IS the finding.
    ax.annotate("", xy=(ANSI_HALF_BAND, 2.2e-2), xytext=(B_min, 2.2e-2),
                arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=2.0))
    ax.text(np.sqrt(B_min * ANSI_HALF_BAND), 3.0e-2,
            "FORBIDDEN TWICE OVER\n"
            "every admissible $B$ already\nexceeds the ANSI band\n"
            r"$\Rightarrow$ no usable operating point",
            ha="center", va="bottom", fontsize=8.5, color="#8e44ad",
            weight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(B[0], B[-1])
    ax.set_ylim(1e-6, 30)

    ax.set_xlabel("noise bound $B$ (per-unit voltage)  —  larger = more noise added",
                  fontsize=10)
    ax.set_ylabel(r"failure probability $\delta = S/2B$" "\nsmaller = stronger privacy",
                  fontsize=10)
    ax.set_title("Figure 4: uniform BNP has no usable operating point on per-unit voltages\n"
                 f"(voltage sensitivity $S$ = {S:.4f} pu, IEEE 123-bus)",
                 fontsize=11.5)

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()

    out = os.path.join(FIGDIR, "figure4_bnp_wall.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"  voltage sensitivity S      : {S:.4f} pu")
    print(f"  smallest admissible B      : {B_min:.4f} pu  (delta = 1.0)")
    print(f"  ANSI half-band             : {ANSI_HALF_BAND:.4f} pu")
    print(f"  B needed for delta = {DELTA_TARGET:g}  : {B_for_target:,.0f} pu")
    print(f"  admissible-and-useful range: EMPTY "
          f"({B_min:.3f} > {ANSI_HALF_BAND:.3f})")
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
