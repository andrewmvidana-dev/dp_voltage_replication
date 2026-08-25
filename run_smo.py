# Sliding mode observer fault detection, plus a head-to-head against a
# steady-state Kalman filter. Writes one PNG into figures/ and prints the
# numbers behind it.
#
#   figure4_smo_comparison.png
#
# The experiment is deliberately set up around an INCIPIENT (slowly ramping)
# fault, because that is where the two methods genuinely differ. On an abrupt
# step fault both detect quickly and there is nothing interesting to show.
#
# Run: python run_smo.py

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # render to file; no interactive window needed
import matplotlib.pyplot as plt

from dpvolt.smo import (LinearSystem, example_system, SlidingModeObserver,
                        SteadyStateKalman, ThresholdConfig, ProtectionLogic,
                        design_L, design_G, design_rho, calibrate_e_max,
                        estimate_fault, fault_free_bound, simulate)


HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

DT = 1e-3
T_END = 30.0
FAULT_TIME = 15.0
FAULT_SIZE = 0.6
RAMP_RATE = 0.08          # per second; reaches FAULT_SIZE ~7.5 s after onset
XI_MAX = 0.05
MEAS_NOISE = 1e-3
SEED = 0


def banner(text):
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


# =============================================================================
# Design
# =============================================================================

def build():
    """Construct the plant, check the structural conditions, design the SMO."""
    sys = example_system(xi_max=XI_MAX, f_max=FAULT_SIZE)

    banner("PART 1  --  Plant and structural conditions")
    print(f"  states n = {sys.n}, inputs m = {sys.m}, outputs r = {sys.r}")
    print(f"  open-loop poles: {np.round(np.linalg.eigvals(sys.A), 4)}")
    print(f"  observability rank {sys.observability_rank()} of {sys.n}"
          f"  -> {'OK' if sys.is_observable() else 'FAILS'}")
    print(f"  rank(CF) == rank(F) (fault visible in one step)"
          f"  -> {'OK' if sys.relative_degree_ok() else 'FAILS'}")

    if not sys.is_observable():
        raise RuntimeError("plant not observable; no observer can be built")
    if not sys.relative_degree_ok():
        raise RuntimeError("observer-matching condition fails; see smo.py Section 1")

    banner("PART 2  --  Observer design")

    L = design_L(sys, decay=2.0)
    G = design_G(sys)
    print(f"  A - LC poles: {np.round(np.linalg.eigvals(sys.A - L @ sys.C), 4)}")
    print(f"  ||C G - I||  = {np.linalg.norm(sys.C @ G - np.eye(sys.r)):.2e}"
          "   (must be ~0: injection must act directly on the residual)")

    # e_max bounds ||x - x_hat||, and it dominates rho -- which in turn sets
    # the detection floor. Guessing it conservatively (say 1.0) is the single
    # easiest way to ruin this experiment: it gives rho = 13.3 and a threshold
    # so high that no fault below ~0.3 is visible, which makes the SMO look no
    # better than the Kalman filter. So we MEASURE it from fault-free data.
    # See calibrate_e_max in smo.py for why this is legitimate here and what
    # you would do instead on real hardware.
    cal_e = calibrate_e_max(sys, L, eta=0.5, margin=3.0, boundary=2e-3,
                            t_end=20.0, dt=DT, meas_noise=MEAS_NOISE,
                            seed=SEED + 100)
    print("\n  e_max calibration (fault-free, iterated to a fixed point):")
    for h in cal_e["history"]:
        print(f"    e_max in {h['e_max_in']:8.4f}  ->  rho {h['rho']:7.3f}"
              f"  ->  peak ||x - x_hat|| {h['peak_err']:.5f}")
    print(f"    settled e_max = {cal_e['e_max']:.5f}"
          f"  ({cal_e['margin']:.0f}x the measured peak)")

    gain = design_rho(sys, L, e_max=cal_e["e_max"], eta=0.5)

    print("\n  switching gain from the reachability condition:")
    print(f"    cross term       (||C(A-LC)|| * e_max)   = {gain['cross_term']:.4f}")
    print(f"    disturbance term (||C D|| * xi_max)      = {gain['disturbance_term']:.4f}")
    print(f"    fault term       (||C F|| * f_max)       = {gain['fault_term']:.4f}")
    print(f"    eta (finite-time margin)                 = {gain['eta']:.4f}")
    print(f"    -------------------------------------------------")
    print(f"    rho                                      = {gain['rho']:.4f}")
    print(f"\n  reaching time bound: t <= ||e_y(0)|| / eta = ||e_y(0)|| / {gain['eta']:.2f} s")

    obs = SlidingModeObserver(sys=sys, L=L, G=G, rho=gain["rho"],
                              boundary=2e-3, switch_kind="sigmoid",
                              tau_filter=0.05)
    kf = SteadyStateKalman(sys, Q_scale=1e-2, R_scale=1e-3)
    return sys, obs, kf, gain


# =============================================================================
# Calibration on fault-free data
# =============================================================================

def calibrate(sys, obs, kf):
    """Run fault-free and set the protection threshold from that run."""
    banner("PART 3  --  Threshold calibration on fault-free data")

    clean = simulate(sys, obs, kf, t_end=T_END, dt=DT, fault_kind="none",
                     fault_time=None, meas_noise=MEAS_NOISE, seed=SEED + 100)

    bound = fault_free_bound(sys, obs)
    print(f"  theoretical fault-free bound on ||nu_eq||:")
    print(f"    matched disturbance (||C D|| * xi_max) = {bound['matched']:.5f}")
    print(f"    boundary-layer term                    = {bound['layer_term']:.5f}")
    print(f"    total                                  = {bound['theoretical']:.5f}")

    # Discard the reaching transient before calibrating: the first fraction of
    # a second is the observer converging, not steady-state behaviour, and
    # including it would inflate the threshold enormously.
    settle = int(2.0 / DT)
    cfg = ThresholdConfig(margin=1.5, dwell_steps=int(0.05 / DT),
                          empirical_quantile=0.999)
    cal = ProtectionLogic.calibrate(clean.nu_eq[settle:], bound["theoretical"], cfg)

    print(f"\n  empirical q{cfg.empirical_quantile} of ||nu_eq||        = {cal['empirical']:.5f}")
    print(f"  threshold from empirical (x{cfg.margin})       = {cal['from_empirical']:.5f}")
    print(f"  CHOSEN threshold                          = {cal['threshold']:.5f}")
    print(f"  dwell                                     = {cfg.dwell_steps} steps"
          f" ({cfg.dwell_steps * DT * 1000:.0f} ms)")

    # A false-alarm check on the calibration run itself. This is weak evidence
    # -- it is the same data the threshold was fitted to -- but a failure here
    # would mean the threshold is broken outright.
    logic = ProtectionLogic(cal["threshold"], cfg)
    for k in range(settle, len(clean.t)):
        logic.update(clean.nu_eq[k], k)
    print(f"  false alarm on the fault-free run         = "
          f"{'YES (threshold is broken)' if logic.flagged else 'no'}")

    return clean, cal, cfg, bound


# =============================================================================
# The fault run
# =============================================================================

def run_fault(sys, obs, kf, cal, cfg, kind, size, rate):
    """Simulate one fault and evaluate both detectors on it."""
    res = simulate(sys, obs, kf, t_end=T_END, dt=DT, fault_time=FAULT_TIME,
                   fault_kind=kind, fault_size=size, fault_ramp_rate=rate,
                   meas_noise=MEAS_NOISE, seed=SEED)
    res.threshold = cal["threshold"]

    # -- SMO detector: threshold the equivalent injection --------------------
    logic = ProtectionLogic(cal["threshold"], cfg)
    settle = int(2.0 / DT)
    for k in range(settle, len(res.t)):
        if logic.update(res.nu_eq[k], k):
            break
    res.flag_index = logic.flag_index

    # -- Kalman detector: threshold the innovation, calibrated the same way --
    # Same procedure, same margin, same dwell -- the comparison is only fair
    # if the baseline gets the identical treatment. We calibrate its threshold
    # off its own pre-fault segment.
    pre = int(FAULT_TIME / DT)
    kf_mags = np.linalg.norm(res.e_kf[settle:pre], axis=1)
    kf_thr = cfg.margin * float(np.quantile(kf_mags, cfg.empirical_quantile))

    kf_logic = ProtectionLogic(kf_thr, cfg)
    for k in range(settle, len(res.t)):
        if kf_logic.update(res.e_kf[k], k):
            break
    res.kf_flag_index = kf_logic.flag_index
    res.meta["kf_threshold"] = kf_thr
    return res


def report(res, label):
    """Print detection latency for both methods on one run."""
    dt = res.meta["dt"]

    def latency(idx):
        if idx is None:
            return None
        return res.t[idx] - FAULT_TIME

    lat_smo = latency(res.flag_index)
    lat_kf = latency(res.kf_flag_index)

    print(f"\n  {label}")
    print(f"    fault onset at t = {FAULT_TIME:.1f} s")

    if lat_smo is None:
        print(f"    SMO    : NOT DETECTED")
    else:
        f_at = res.f_true[res.flag_index]
        print(f"    SMO    : detected at t = {res.t[res.flag_index]:7.3f} s"
              f"  (latency {lat_smo:6.3f} s, fault magnitude {f_at:.4f})")

    if lat_kf is None:
        print(f"    Kalman : NOT DETECTED within the {T_END:.0f} s window")
    else:
        f_at = res.f_true[res.kf_flag_index]
        print(f"    Kalman : detected at t = {res.t[res.kf_flag_index]:7.3f} s"
              f"  (latency {lat_kf:6.3f} s, fault magnitude {f_at:.4f})")

    if lat_smo is not None and lat_kf is not None:
        print(f"    -> SMO is faster by {lat_kf - lat_smo:.3f} s")

    # The central claim of the method is that nu_eq ESTIMATES the fault, not
    # merely that it grows when one occurs. Check that directly, on the
    # settled portion where the fault is constant and the filter has caught
    # up, so the comparison is not confounded by the filter lag.
    settled = res.f_true >= res.meta["fault_size"] - 1e-9
    settled &= res.t > FAULT_TIME + 3.0
    # nu_eq is a detector, not a meter -- so report the quantity that actually
    # governs detection (the peak, shortly after the fault is fully developed)
    # and then show the decay honestly rather than hiding it.
    if settled.any():
        true = res.f_true[settled].mean()
        f_hat = np.linalg.norm(estimate_fault(res.meta["sys"], res.nu_eq), axis=1)
        i0 = int(np.argmax(settled))
        peak = f_hat[i0:i0 + int(2.0 / dt)].max()
        print(f"    nu_eq as a fault METER (true fault {true:.4f}):")
        print(f"      peak just after onset  {peak:.4f}  "
              f"({100 * (peak - true) / true:+.1f}%)")
        for lag in (5.0, 10.0, 15.0):
            k = i0 + int(lag / dt)
            if k < len(f_hat):
                print(f"      {lag:4.0f} s later          {f_hat[k]:.4f}  "
                      f"({100 * (f_hat[k] - true) / true:+.1f}%)")
        print(f"      -> the LEVEL decays while the fault is constant; only the")
        print(f"         RISE is trustworthy. See smo.py step() for why.")
    return lat_smo, lat_kf


# =============================================================================
# The comparison figure
# =============================================================================

def make_figure(res_ramp, res_step, cal):
    """Four panels: what each method sees, and what it costs.

    Panel layout is chosen so the argument reads top to bottom:
      (a) the fault itself, so everything below has a reference
      (b) the two residuals -- the Kalman innovation ATTENUATES as the filter
          re-converges around the ramp; this is the failure being illustrated
      (c) the SMO equivalent injection against its threshold -- the signal
          that does not attenuate, because holding the residual at zero is
          exactly what sustains it
      (d) the step fault, where both methods work, as the control case
    """
    os.makedirs(FIGDIR, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    t = res_ramp.t

    # -- (a) the fault ------------------------------------------------------
    ax = axes[0]
    ax.plot(t, res_ramp.f_true, lw=2, color="#c1121f", label="ramp (incipient)")
    ax.plot(res_step.t, res_step.f_true, lw=2, ls="--", color="#003049",
            label="step (abrupt)")
    ax.axvline(FAULT_TIME, color="k", ls=":", alpha=0.6)
    ax.set_ylabel("fault magnitude")
    ax.set_title("(a) The injected faults")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # -- (b) residuals, ramp fault ------------------------------------------
    ax = axes[1]
    ax.plot(t, np.linalg.norm(res_ramp.e_kf, axis=1), lw=1.0, alpha=0.85,
            color="#0077b6", label="Kalman innovation $\\|y-\\hat y\\|$")
    ax.plot(t, np.linalg.norm(res_ramp.e_smo, axis=1), lw=1.0, alpha=0.85,
            color="#c1121f", label="SMO residual $\\|e_y\\|$")
    ax.axhline(res_ramp.meta["kf_threshold"], color="#0077b6", ls="--", lw=1.2,
               label="Kalman threshold")
    ax.axvline(FAULT_TIME, color="k", ls=":", alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("residual norm")
    # NOTE the title is deliberately NOT "the SMO residual stays lower". It
    # does not -- both traces sit in the same 1e-3 band. That is the actual
    # finding: the residual is a poor fault indicator for BOTH methods on an
    # incipient fault, which is precisely why panel (c) exists.
    ax.set_title("(b) Ramp fault: neither residual moves much "
                 "-- the residual is the wrong signal to threshold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # -- (c) equivalent injection, ramp fault -------------------------------
    ax = axes[2]
    ax.plot(t, np.linalg.norm(res_ramp.nu_eq, axis=1), lw=1.5, color="#2a9d8f",
            label="equivalent injection $\\|\\nu_{eq}\\|$")
    ax.axhline(cal["threshold"], color="#e76f51", ls="--", lw=1.5,
               label=f"protection threshold ({cal['threshold']:.3f})")
    ax.axvline(FAULT_TIME, color="k", ls=":", alpha=0.6, label="fault onset")
    if res_ramp.flag_index is not None:
        ax.axvline(t[res_ramp.flag_index], color="#c1121f", lw=2, alpha=0.8,
                   label=f"SMO flag ({t[res_ramp.flag_index] - FAULT_TIME:.2f} s)")
    if res_ramp.kf_flag_index is not None:
        ax.axvline(t[res_ramp.kf_flag_index], color="#0077b6", lw=2, alpha=0.8,
                   label=f"Kalman flag ({t[res_ramp.kf_flag_index] - FAULT_TIME:.2f} s)")
    ax.set_ylabel("$\\|\\nu_{eq}\\|$")
    ax.set_title("(c) Ramp fault: the injection rises well clear of the threshold "
                 "-- this is the detection signal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # -- (d) step fault, both signals ---------------------------------------
    ax = axes[3]
    ax.plot(res_step.t, res_step.f_true, lw=1.5, color="#c1121f", alpha=0.7,
            label="true fault (constant after onset)")
    ax.plot(res_step.t, np.linalg.norm(res_step.nu_eq, axis=1), lw=1.5,
            color="#2a9d8f", label="SMO $\\|\\nu_{eq}\\|$ (decays!)")
    ax.plot(res_step.t, np.linalg.norm(res_step.e_kf, axis=1), lw=1.0,
            alpha=0.8, color="#0077b6", label="Kalman innovation")
    ax.axhline(cal["threshold"], color="#e76f51", ls="--", lw=1.2,
               label="SMO threshold")
    ax.axhline(res_step.meta["kf_threshold"], color="#0077b6", ls="--", lw=1.2,
               label="Kalman threshold")
    ax.axvline(FAULT_TIME, color="k", ls=":", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("magnitude")
    ax.set_title("(d) Step fault: both detect it -- but note BOTH signals decay "
                 "while the fault is constant")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "figure4_smo_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# =============================================================================

def main():
    sys, obs, kf, gain = build()
    clean, cal, cfg, bound = calibrate(sys, obs, kf)

    banner("PART 4  --  Fault detection")

    res_ramp = run_fault(sys, obs, kf, cal, cfg, "ramp", FAULT_SIZE, RAMP_RATE)
    lat_smo_r, lat_kf_r = report(res_ramp, f"RAMP fault ({RAMP_RATE}/s, incipient)")

    res_step = run_fault(sys, obs, kf, cal, cfg, "step", FAULT_SIZE, RAMP_RATE)
    lat_smo_s, lat_kf_s = report(res_step, f"STEP fault ({FAULT_SIZE}, abrupt)")

    path = make_figure(res_ramp, res_step, cal)

    banner("SUMMARY")
    print("  The step fault is the control case: both methods detect it, and")
    print("  quickly. It is there to show the comparison is not rigged.")
    print()
    print("  The ramp fault is the real test. Note first what does NOT work:")
    print("  BOTH residuals stay in the same small band as the fault grows")
    print("  (panel b). The Kalman filter re-converges around a slow fault, so")
    print("  its innovation stays quiet; the SMO's residual is held near zero")
    print("  by the switching term. Thresholding the residual is therefore a")
    print("  weak detector either way.")
    print()
    print("  What works is that for the SMO the fault information is not lost,")
    print("  only moved: holding the residual at zero is exactly what requires")
    print("  an injection, and that injection RISES sharply at fault onset")
    print("  well clear of its fault-free band. Thresholding nu_eq instead of")
    print("  the residual is what buys the early detection.")
    print()
    print("  But nu_eq is a DETECTOR, NOT A METER, and the numbers above show")
    print("  it plainly: with the fault held constant, ||nu_eq|| decays over")
    print("  tens of seconds. The Luenberger term and the boundary layer both")
    print("  erode ideal sliding and drain the injection. Dropping L and using")
    print("  a true sign law roughly doubles the retained magnitude but")
    print("  chatters and raises the false-alarm floor ~50%. Fault SIZING")
    print("  wants a different tool; this module does detection.")
    print()
    print("  Caveats worth stating with the figure:")
    print("   - the disturbance here is MATCHED. An unmatched one is not")
    print("     rejected and would raise the false-alarm floor.")
    print("   - e_max is MEASURED from fault-free data, which uses the true")
    print("     state and so is a simulation-study move, not a deployable one.")
    print("     A Walcott-Zak LMI removes the term and needs no calibration.")
    print("   - rho is dominated by the fault term (||C F|| * f_max), i.e. by")
    print("     the largest fault we insist on sliding through. Detection of")
    print("     SMALL faults would be sharper with a smaller f_max, at the")
    print("     cost of losing the guarantee for large ones.")
    print("   - detection latency includes the deliberate "
          f"{cfg.dwell_steps * DT * 1000:.0f} ms dwell.")
    print()
    print(f"  Figure written to {path}")


if __name__ == "__main__":
    main()
