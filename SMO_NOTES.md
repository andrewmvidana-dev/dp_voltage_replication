# Sliding Mode Observer — Fault Detection

Handoff notes for `dpvolt/smo.py` and `run_smo.py`.

This file is written to be self-contained: you can paste it into a fresh Claude
session (or read it yourself) without having seen the conversation that produced
the code. It covers what the code does, why it is built the way it is, what was
measured, and — importantly — the three places where the first attempt was wrong
and testing corrected it.

**Run it:** `python run_smo.py` (needs `numpy`, `scipy`, `matplotlib`)
**Produces:** `figures/figure4_smo_comparison.png` plus a verbose console report

---

## 1. The plain-English version

You have a machine — a motor, a generator, some piece of equipment. Sensors tell
you *some* of what it is doing, not all of it. You want to know two things: what
is happening in the parts you cannot measure, and whether something is **broken**.

### What an observer is

An observer is a **simulation of the machine running alongside the real one**, in
software. Feed it the same commands the real machine gets, and it predicts what
the sensors should read. Then compare:

```
real sensor says:      5.2
simulation says:       5.0
difference:            0.2     <-- this is the RESIDUAL
```

If the simulation is good, that difference stays near zero. If it grows,
something is wrong. The observer continuously nudges its own simulation to stay
matched to reality.

### Two ways to nudge

**Kalman filter** (the standard choice) nudges *gently and proportionally*. Off
by a little, correct a little. It is mathematically optimal against random sensor
noise.

**Sliding Mode Observer** nudges *aggressively*. It ignores how far off it is and
slams the correction to full strength in whichever direction is needed, flipping
back and forth very fast. That is the "switching" or "discontinuous" part — like
steering by yanking hard left, hard right, hundreds of times a second instead of
making smooth adjustments.

Crude-sounding, but it buys two real things: it locks onto the truth in **finite
time** (not "eventually"), and it **fully cancels** bounded disturbances rather
than merely shrinking them. That locked-on condition is the **sliding surface** —
the error is pinned at zero and *held* there.

### The clever bit

Once the observer is locked on, the error is zero. So the residual tells you
nothing — it is zero whether or not there is a fault. Useless?

No. Ask instead: **how hard is the observer working to hold it at zero?** If a
fault pushes the machine one way, the observer must push back continuously to
stay matched. **The amount of push-back is the fault.** Average out the fast
flip-flopping and look at the net effort. That average is the **equivalent
injection**, written `nu_eq`.

The fault information does not vanish. It moves out of the residual and into
the effort.

### Why this beats a Kalman filter here

The interesting test case is an **incipient fault** — something slowly degrading,
like a bearing wearing out, not a sudden break.

This is exactly where a Kalman filter fails, and the failure is subtle: it is
*designed* to adapt. Faced with a slow drift, it quietly re-adjusts and concludes
the machine simply behaves that way now. It **absorbs the fault** and reports
that everything is fine.

### The honest catch

`nu_eq` is a **smoke detector, not a thermometer**. It reliably says *something
is wrong, now* — the jump at fault onset is sharp and unmistakable. It does
**not** reliably say *how bad*. With a constant fault the signal starts near the
right value and then fades over tens of seconds. Section 5 has the measurements
and the cause. The code is scoped to detection and says so throughout.

---

## 2. The math, briefly

Plant (continuous time):

```
x_dot = A x + B u + D xi(t) + F f(t)
y     = C x
```

| symbol | meaning |
|---|---|
| `x` (n) | state |
| `u` (m) | known control input |
| `xi` (q) | unknown but **bounded** disturbance, `\|\|xi\|\| <= xi_max` |
| `f` (p) | the fault we want to detect |
| `y` (r) | measured output |

Observer:

```
x_hat_dot = A x_hat + B u + L (y - y_hat) + G nu
y_hat     = C x_hat
e_y       = y - y_hat                       <-- residual
nu        = -rho * e_y / (||e_y|| + boundary)   <-- switching term
```

Error dynamics, with `e = x - x_hat`:

```
e_dot = (A - L C) e + D xi + F f - G nu
```

Two design requirements fall straight out of that equation:

1. **`A - L C` must be Hurwitz** (stable), or the error grows in directions the
   switching term does not act in. Handled by `design_L`.
2. **The surface must be reachable in finite time.** This constrains `rho`.

### Where the switching gain comes from

Lyapunov candidate `V = (1/2) e_y' e_y`. Differentiating along the error
dynamics, using `C G = I`:

```
V_dot = e_y' C (A - L C) e  +  e_y' C (D xi + F f)  -  rho ||e_y||
```

Bound the first two terms and require the whole thing to be negative:

```
rho >= ||C(A-LC)|| * e_max  +  ||C D|| * xi_max  +  ||C F|| * f_max  +  eta
```

Then `V_dot <= -eta ||e_y||`, which gives **finite-time** reaching:

```
t_reach <= ||e_y(0)|| / eta
```

`eta` is not slack — it is precisely what converts "eventually converges" into
"converges by time T". `design_rho` returns all four terms separately, because
when `rho` is too big you need to see which term is responsible.

### Why the equivalent injection carries the fault

On the surface, `e_y = 0` and `e_y_dot = 0` both hold. Set `e_y_dot = 0`:

```
0     = C(A-LC)e + C D xi + C F f - nu_eq
nu_eq = C D xi + C F f              (since C e = 0 on the surface)
```

So the injection needed to *maintain* sliding equals the matched disturbance plus
the fault. Low-pass filtering the fast switching recovers it.

**This identity is an idealisation.** It assumes a true discontinuous sign law,
no boundary layer, and no other correction acting on the same error. This
implementation violates all three deliberately — see Section 5.

---

## 3. Structural conditions (checked at runtime, and they matter)

**Observability:** `rank([C; CA; ...; CA^(n-1)]) == n`. Otherwise no observer of
any kind can reconstruct the state.

**Observer matching:** `rank(C F) == rank(F)`. The fault must be visible in `y`
in one step. If `C F` loses rank, the fault only reaches the output *through the
dynamics* (relative degree >= 2). The switching term sees only the output error,
so it cannot cancel a fault that is not yet visible there — sliding is never
attained and `nu_eq` carries no fault information.

Both are checked in `run_smo.py` and the script refuses to run if either fails.
That check earned its keep — see Section 5.

---

## 4. Results

Plant: 3 states, 2 outputs, lightly damped oscillatory mode (~3 rad/s, damping
~0.13) coupled to a slow first-order mode. Fault injected at t = 15 s.

### Detection latency

| Fault type | SMO detects at | Kalman detects at | Advantage |
|---|---|---|---|
| **Ramp** (0.08/s, incipient) | fault size **0.093**, +1.16 s | fault size **0.322**, +4.03 s | **2.87 s earlier** |
| **Step** (0.6, abrupt) | +0.011 s | +0.033 s | 0.022 s |

The ramp is the real test: the SMO catches it at roughly **one third** the fault
magnitude. The step case is the **control** — both detect it almost instantly,
included specifically to show the comparison is not rigged.

### Design numbers as printed

```
A - LC poles:          -3.99 +/- 3.42j,  -3.36        (stable)
||C G - I||            0.00e+00                       (injection acts on residual)
e_max (calibrated)     0.00618
rho                    1.2249
  cross term           0.0749
  disturbance term     0.0500
  fault term           0.6000    <-- dominates
  eta                  0.5000
threshold              0.0929
dwell                  50 steps (50 ms)
false alarm (no fault) none
```

Note `rho` is dominated by the **fault term** `||C F|| * f_max` — by the largest
fault you insist on sliding through. Lowering `f_max` sharpens small-fault
detection but drops the guarantee for large ones.

### The figure

`figures/figure4_smo_comparison.png`, four stacked panels:

- **(a)** the two injected faults, for reference
- **(b)** both residuals — deliberately shows that **neither** moves much, which
  is *why* you cannot just threshold the residual
- **(c)** the equivalent injection crossing its threshold, both detection moments
  marked — the money panel
- **(d)** step fault, with the flat true-fault line drawn next to the decaying
  `nu_eq`, so the "not a thermometer" limitation is **visible**, not just described

---

## 5. Three things that were wrong, and how they got caught

This section is the most useful part of the file. In each case the first version
was plausible, and measurement disagreed.

### 5.1 The first example plant was structurally invalid

The obvious sensor choice — measure position (state 1) and the slow mode
(state 3) — gives `C F = 0`. The fault enters on state 2, which neither output
sees directly, so relative degree is 2 and **sliding is unattainable**.

The `rank(C F) == rank(F)` check caught it and the script refused to run. Fixed
by measuring the *velocity* state instead.

This is not a cosmetic fix. It is a real constraint on **sensor placement**: an
SMO fault detector needs a measurement in the fault's own channel. If your
instrumentation does not provide one, the answer is a higher-order sliding mode
or a cascaded observer — not a retuned gain.

### 5.2 The comparison was initially a tie, and that was a bug in the setup

First run: SMO 4.030 s, Kalman 4.025 s. Effectively identical. It would have been
easy to report that as a legitimate negative result.

**Cause:** `e_max` bounds `||x - x_hat||` in the reachability condition. It was
*guessed* at 1.0. The actual peak error was **0.0094** — a 100x over-estimate.

The damage chain is worth understanding, because it is not obvious:

```
e_max too big  ->  rho too big (13.3 instead of 1.2)
               ->  boundary-layer noise floor rises
               ->  detection threshold rises (0.236 instead of 0.093)
               ->  blind to any fault below ~0.32
```

The observer went blind to exactly the small incipient faults it is supposed to
be good at. Meanwhile the Kalman baseline calibrated its threshold empirically
from its own data, so it was **not** handicapped the same way. The tie was an
artifact of one bad constant, not a property of either method.

**Fix:** `calibrate_e_max()` measures it from fault-free data, iterating to a
fixed point (`rho` depends on `e_max`, the error depends on `rho`, and the map
contracts):

```
e_max in 1.0000  ->  rho 13.270  ->  peak err 0.00749
e_max in 0.0225  ->  rho  1.422  ->  peak err 0.00225
e_max in 0.0068  ->  rho  1.232  ->  peak err 0.00206
settled: e_max = 0.00618  (3x the measured peak)
```

**Caveat, stated in the code:** this uses the *true state*, which a real
deployment does not have. Legitimate here because this is a simulation study
characterising the observer. On hardware you would get `e_max` from a
model-validation campaign, a hardware-in-the-loop rig, or a Walcott-Zak LMI
design (which eliminates the term entirely and needs no calibration).

### 5.3 The fault-estimation claim was overstated

The original text said `nu_eq` is "a direct estimate of the fault itself."
Testing that directly: with the fault held constant at 0.6,

| time (onset at t=15 s) | `\|\|nu_eq\|\|` | true fault |
|---|---|---|
| t = 16 s | 0.54 | 0.6 |
| t = 20 s | 0.35 | 0.6 |
| t = 30 s | 0.19 | 0.6 |
| t = 42 s | 0.07 | 0.6 |

The main script reports the same effect on its own 30 s window, corrected for
the `C F` scaling: peak **0.468** (-22%) just after onset, **0.286** (-52%) five
seconds later, **0.183** (-70%) ten seconds later.

It **decays to near zero while the fault is still fully present.**

Two causes, both isolated by ablation (turning one piece off at a time):

1. **The Luenberger term.** `L e_y` and the switching term correct the *same*
   error. `L` integrates the persistent bias a fault creates and gradually takes
   over the cancelling work, draining it from `nu_eq`. Setting `L = 0` raises the
   t=42 s value from 0.07 to 0.26.
2. **The boundary layer.** Inside it the law is finite-gain, so sustained output
   needs a sustained `e_y` — but the observer is simultaneously driving `e_y`
   down. A true `sign` law recovers some of the loss (0.07 -> 0.23 with `L`
   intact). `L = 0` **and** `sign` together give 0.43.

A false lead worth recording: the first hypothesis was that the gap was the
`C F` scaling, and `estimate_fault()` was written to invert it with a
pseudo-inverse. Measured effect: **0.3237 -> 0.3236**, i.e. under 0.1%. On this
plant `C F = [1, 0]'`, so `pinv(C F)` is norm-preserving. The function is kept
because it matters on plants where `C F` is poorly scaled, but its docstring now
states plainly that it does **not** rescue the estimate here.

**Resolution:** the module is scoped as a **detector**. Detection depends only on
the *rise* clearing the fault-free band, which is sharp and reliable. Fault
*sizing* needs a different tool — a higher-order sliding-mode differentiator, or
an adaptive / unknown-input observer.

---

## 6. Code map

`dpvolt/smo.py` — standalone, imports nothing else from `dpvolt`:

| Section | Contents |
|---|---|
| 1 | `LinearSystem`, `example_system` — plant, structural checks |
| 2 | `switching` — `sign` / `sat` / `sigmoid`, chattering trade-off |
| 3 | `SlidingModeObserver` — the observer, RK4, equivalent injection |
| 4 | `design_L`, `design_G`, `design_rho`, `estimate_fault`, `calibrate_e_max` |
| 5 | `fault_free_bound`, `ThresholdConfig`, `ProtectionLogic` |
| 6 | `SteadyStateKalman` — the baseline |
| 7 | `SimResult`, `simulate` — the driver |

`run_smo.py` — design, calibrate, run both faults, plot, report.

### Protection logic — three stages

1. **Threshold** — compare `||nu_eq||` (not the residual) against a calibrated bound
2. **Dwell** — require N *consecutive* exceedances; a single sample over threshold
   is a transient, not a fault. This is a relay's definite-time delay and is the
   main defence against false trips.
3. **Latch** — once flagged, stay flagged

The dwell costs `dwell_steps * dt` of detection latency (50 ms here). That cost is
deliberate and is **included** in every reported latency figure rather than
quietly excluded.

### Threshold calibration

Two numbers, and neither alone is sufficient:

- **Theoretical floor:** `||C D|| * xi_max` + boundary-layer term. Worst-case over
  adversarial `xi`, ignores measurement noise. Can sit far above the real
  fault-free level (missing small faults) or, if `xi_max` was underestimated,
  below it (constant tripping).
- **Empirical:** a high quantile of `||nu_eq||` on fault-free data, times a margin.
  Overfits whatever disturbance realisation happened to occur.

The code takes the max of the empirical value and a fraction of the theoretical
one. The theory protects against a worse disturbance than the one you happened to
record.

---

## 7. Standing caveats

- **The disturbance is MATCHED** — it enters through the same channel the injection
  acts in. An unmatched disturbance is *not* rejected, shows up in the residual,
  and is indistinguishable from a fault. Genuine limitation of the method.
- **`e_max` is measured using the true state** — simulation-study move. See 5.2.
- **`rho` is dominated by `f_max`** — the largest fault you insist on sliding through.
- **`nu_eq` detects, it does not measure.** See 5.3.
- **Reported latencies include the 50 ms dwell.**
- The plant is a synthetic 3-state example, not a model of the power-flow work in
  the rest of this repo. `smo.py` shares no code with the other `dpvolt` modules.

---

## 8. Useful things to ask a fresh session

- Swap the sigmoid for a true `sign` law and quantify the chattering / fidelity trade
- Implement the Walcott-Zak LMI design so `e_max` is eliminated rather than calibrated
- Add an **unmatched** disturbance and measure how much the false-alarm floor rises
- Sweep ramp rate vs. detection latency — where does the SMO advantage disappear?
- Add sensor-fault cases (currently only actuator faults, `F = B`)
- Try a higher-order sliding-mode differentiator to get fault *sizing* working
- Multi-fault isolation: which fault, not just whether one occurred
