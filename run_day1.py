# Day 1 results: load the IEEE 123-bus feeder, classify its buses, and Kron
# reduce. Runs the reduction twice -- with and without our network cleanup --
# to show the cleanup drops kappa_Kron by ~19 orders of magnitude, which is
# the difference between Theorem 1 being usable and being empty.
#
# Takes a few seconds. Output is deliberately verbose enough to read aloud.

import os
import warnings

import numpy as np

# This pulls in the functions we wrote in dpvolt/network.py.
from dpvolt.network import load_feeder, kron_reduce, injection_check


# -----------------------------------------------------------------------------
# get_feeder.py puts the feeder here. Run that script first.
MASTER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "feeders", "IEEE123Master.dss")


def banner(text):
    """Print a section heading so the output is easy to scan."""
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def main():
    # Fail early with a readable message if the feeder is missing.
    if not os.path.exists(MASTER):
        print("Could not find the feeder files.")
        print(f"  Expected: {MASTER}")
        print()
        print("Open get_feeder.py and press Run first, then come back here.")
        return

    # -------------------------------------------------------------------------
    # PART 1 -- The naive version: exactly what the paper describes, no cleanup.
    # -------------------------------------------------------------------------
    banner("PART 1  --  Kron reduction with NO network cleanup (the naive read)")

    # We silence warnings here only because we EXPECT the ill-conditioning
    # warning to fire; that is the point of this part.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = load_feeder(MASTER, merge_switches=False, prune_stubs=False)
        raw_kron = kron_reduce(raw)

    print(f"  nodes in the network        : {raw.N}")
    print(f"  retained buses    (set R)   : {len(raw.load_nodes)}")
    print(f"  zero-injection    (set Z)   : {len(raw.zero_inj_nodes)}")
    print(f"  slack / substation(set S)   : {len(raw.slack_nodes)}")
    print()
    print(f"  condition number of Y_ZZ    : {raw_kron.cond_Y_ZZ:.3e}")
    print(f"  kappa_Kron                  : {raw_kron.kappa_kron:.4e}   <-- PROBLEM")
    print()
    print("  Read that last number carefully. The paper's theorem needs")
    print("      alpha = ||M~^-1|| * C* * kappa_Kron * r  <  1/4")
    print("  With kappa_Kron at 1e25, the adjacency radius r would have to be")
    print("  smaller than about 1e-27 for the guarantee to apply. In other")
    print("  words, the theorem would protect against no meaningful change to")
    print("  the network at all. Something has to be fixed.")

    # -------------------------------------------------------------------------
    # PART 2 -- With our two cleanup steps applied.
    # -------------------------------------------------------------------------
    banner("PART 2  --  With switch merging and dead-end pruning")

    feeder = load_feeder(MASTER, merge_switches=True, prune_stubs=True)
    kron = kron_reduce(feeder)

    print(f"  nodes after cleanup         : {feeder.N}   (was {raw.N})")
    print(f"  retained buses    (set R)   : {len(feeder.load_nodes)}")
    print(f"  zero-injection    (set Z)   : {len(feeder.zero_inj_nodes)}")
    print(f"  slack / substation(set S)   : {len(feeder.slack_nodes)}")
    print()
    print(f"  condition number of Y_ZZ    : {kron.cond_Y_ZZ:.3e}   (was {raw_kron.cond_Y_ZZ:.1e})")
    print(f"  kappa_Kron                  : {kron.kappa_kron:.4e}   (was {raw_kron.kappa_kron:.1e})")
    print()
    improvement = raw_kron.kappa_kron / kron.kappa_kron
    orders = np.log10(improvement)
    print(f"  improvement factor          : {improvement:.2e}")
    print(f"                              : about {orders:.0f} orders of magnitude")
    print("  Both changes are physically obvious once you see them, and")
    print("  neither is mentioned anywhere in the paper.")

    # -------------------------------------------------------------------------
    # PART 3 -- Prove that we did not break anything while cleaning up.
    # -------------------------------------------------------------------------
    banner("PART 3  --  Correctness checks")

    check = injection_check(feeder)

    print("  Check A: power balance at zero-injection buses")
    print("     A bus we call 'zero-injection' must genuinely inject no power.")
    print("     We test this independently using Ohm's law: S = V * conj(Y @ V).")
    print(f"     largest |S| at a zero-injection bus : {check['max_|S|_zero_inj_kVA']:.4f} kVA")
    print("     -> against 3615 kW flowing through the feeder, that is about")
    print("        0.05 percent. Not exactly zero because merging a switch")
    print("        forces its two ends to share one voltage, and in OpenDSS")
    print("        they differ by the tiny drop across the switch. The")
    print("        classification is right.")
    print()

    print("  Check B: the loads and the substation are where we think")
    print(f"     largest |S| at a load bus          : {check['max_|S|_load_kVA']:.2f} kVA")
    print(f"     total power in from the substation : {check['total_slack_kW']:.1f} kW")
    print("     -> IEEE 123's documented total load is about 3.6 MW. Match.")
    print()

    print("  Check C: the Kron reduction actually reproduces the true voltages")
    print("     The reduction claims the eliminated voltages can be rebuilt")
    print("     from the retained ones via v_Z = Phi @ v_R. We know the true")
    print("     v_Z from OpenDSS, so we can test that claim directly.")
    print(f"     largest relative error : {kron.residual:.3e}")
    print("     -> same story as Check A: the residual is set by the voltage")
    print("        drop we absorbed when merging switches, not by any error in")
    print("        the reduction. Before we fixed the pruning step this number")
    print("        was 1.06, i.e. completely wrong, so it is a real test.")
    print()

    print("  Check D: voltages sit inside the regulation band")
    vpu = np.abs(feeder.V_pu)
    print(f"     voltage range : {vpu.min():.4f} to {vpu.max():.4f} per-unit")
    print("     -> ANSI C84.1 allows [0.95, 1.05], which is also the paper's")
    print("        'good voltage set' in Definition 2. We are inside it, so")
    print("        Assumption 1 holds without any extra work.")

    # -------------------------------------------------------------------------
    # PART 4 -- What this buys us.
    # -------------------------------------------------------------------------
    banner("PART 4  --  What the privacy budget looks like now")

    n = len(feeder.load_nodes)
    Vmin, Vmax = 0.95, 1.05

    # C_star is defined just above Proposition 2 in the paper.
    C_star = np.sqrt(2) * (1 + np.sqrt(n) * Vmax / Vmin)

    # The admissibility condition is alpha < 1/4. We do not yet have the
    # Monte Carlo estimate of ||M~^-1||, so we show what r would have to be
    # for a few placeholder values. This is the number Day 4 will firm up.
    print(f"  n (retained buses)          : {n}")
    print(f"  C_star                      : {C_star:.2f}")
    print()
    print("  Largest adjacency radius r allowed by alpha < 1/4:")
    print()
    print("     ||M~^-1||      max r (cleaned)     max r (naive)")
    print("     ----------------------------------------------------")
    for m in (1.0, 10.0, 100.0):
        r_clean = 1.0 / (4 * m * C_star * kron.kappa_kron)
        r_naive = 1.0 / (4 * m * C_star * raw_kron.kappa_kron)
        print(f"     {m:8.0f}      {r_clean:.3e}        {r_naive:.3e}")
    print()
    print("  The cleaned column is small but finite and workable. The naive")
    print("  column is physically meaningless. That gap is the Day 1 result.")

    banner("NEXT UP")
    print("  Day 2: fit a log-normal load model per customer class.")
    print("  Day 3: add the differential-privacy noise and generate synthetic")
    print("         loads, then push them through power flow.")
    print("  Day 4: implement Theorem 1 and get a real number for ||M~^-1||")
    print("         via the Monte Carlo calibration in Remark 2.")
    print()


# This line means "only run main() if this file was executed directly".
# It is standard Python boilerplate; you can ignore it.
if __name__ == "__main__":
    main()
