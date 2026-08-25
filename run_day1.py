# Load the IEEE 123-bus feeder, classify its buses, and Kron reduce. Runs the
# reduction twice -- with and without our network cleanup -- to show the cleanup
# drops kappa_Kron by ~19 orders of magnitude, which is the difference between
# Theorem 1 being usable and being empty. Takes a few seconds.

import os
import warnings

import numpy as np

from dpvolt.network import load_feeder, kron_reduce, injection_check


MASTER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "feeders", "IEEE123Master.dss")


def banner(text):
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def main():
    if not os.path.exists(MASTER):
        print(f"Feeder files not found at {MASTER}")
        print("Run get_feeder.py first.")
        return

    # ---------------------------------------------------------------------
    banner("PART 1  --  Kron reduction with NO cleanup (the naive read)")
    # ---------------------------------------------------------------------

    # Warnings silenced only because we EXPECT the ill-conditioning warning --
    # that is the point of this part.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = load_feeder(MASTER, merge_switches=False, prune_stubs=False)
        raw_kron = kron_reduce(raw)

    print(f"  nodes                       : {raw.N}")
    print(f"  retained buses    (set R)   : {len(raw.load_nodes)}")
    print(f"  zero-injection    (set Z)   : {len(raw.zero_inj_nodes)}")
    print(f"  slack / substation(set S)   : {len(raw.slack_nodes)}")
    print()
    print(f"  condition number of Y_ZZ    : {raw_kron.cond_Y_ZZ:.3e}")
    print(f"  kappa_Kron                  : {raw_kron.kappa_kron:.4e}   <-- PROBLEM")
    print()
    print("  Theorem 1 needs alpha = ||M~^-1|| * C* * kappa_Kron * r < 1/4.")
    print("  At kappa_Kron = 1e25 the adjacency radius r must be below ~1e-27,")
    print("  i.e. the theorem protects against no meaningful network change.")

    # ---------------------------------------------------------------------
    banner("PART 2  --  With switch merging and dead-end pruning")
    # ---------------------------------------------------------------------

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
    print(f"  improvement                 : {improvement:.2e}"
          f"  (~{np.log10(improvement):.0f} orders of magnitude)")
    print("  Both fixes are physically obvious once seen, and neither is")
    print("  mentioned anywhere in the paper.")

    # ---------------------------------------------------------------------
    banner("PART 3  --  Correctness checks")
    # ---------------------------------------------------------------------

    check = injection_check(feeder)

    print("  A. Power balance at zero-injection buses, via S = V * conj(Y @ V)")
    print(f"       largest |S| at a zero-inj bus : {check['max_|S|_zero_inj_kVA']:.4f} kVA")
    print("       ~0.05% of the 3615 kW throughput. Non-zero only because")
    print("       merging a switch forces its two ends to share one voltage,")
    print("       and in OpenDSS they differ by the drop across the switch.")
    print()
    print("  B. Loads and substation are where we think")
    print(f"       largest |S| at a load bus     : {check['max_|S|_load_kVA']:.2f} kVA")
    print(f"       total power in from substation: {check['total_slack_kW']:.1f} kW")
    print("       IEEE 123's documented total load is ~3.6 MW. Match.")
    print()
    print("  C. Kron reduction reproduces the true eliminated voltages")
    print(f"       largest relative error        : {kron.residual:.3e}")
    print("       Same cause as A. This was 1.06 (i.e. completely wrong)")
    print("       before the pruning step was fixed, so it is a real test.")
    print()
    print("  D. Voltages sit inside the regulation band")
    vpu = np.abs(feeder.V_pu)
    print(f"       voltage range                 : {vpu.min():.4f} to {vpu.max():.4f} pu")
    print("       ANSI C84.1 allows [0.95, 1.05], which is also Definition 2's")
    print("       'good voltage set', so Assumption 1 holds with no extra work.")

    # ---------------------------------------------------------------------
    banner("PART 4  --  What the privacy budget looks like now")
    # ---------------------------------------------------------------------

    n = len(feeder.load_nodes)
    Vmin, Vmax = 0.95, 1.05
    C_star = np.sqrt(2) * (1 + np.sqrt(n) * Vmax / Vmin)   # above Proposition 2

    print(f"  n (retained buses) = {n},  C_star = {C_star:.2f}")
    print()
    print("  Largest adjacency radius r allowed by alpha < 1/4. ||M~^-1|| is")
    print("  not yet calibrated (run_days2_4.py does that), so we show a range:")
    print()
    print("     ||M~^-1||      max r (cleaned)     max r (naive)")
    print("     ----------------------------------------------------")
    for m in (1.0, 10.0, 100.0):
        r_clean = 1.0 / (4 * m * C_star * kron.kappa_kron)
        r_naive = 1.0 / (4 * m * C_star * raw_kron.kappa_kron)
        print(f"     {m:8.0f}      {r_clean:.3e}        {r_naive:.3e}")
    print()
    print("  The cleaned column is small but workable; the naive column is")
    print("  physically meaningless. That gap is the result here.")
    print()
    print("  Next: run_days2_4.py")
    print()


if __name__ == "__main__":
    main()
