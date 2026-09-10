# Differentially Private Synthetic Voltage Phasor Release — Replication

An independent replication of Campbell, Zhang, Scaglione, Kerr, Chesler & Peisert,
*"Differentially Private Synthetic Voltage Phasor Release for Distribution Grids"*
([arXiv:2605.02390](https://arxiv.org/abs/2605.02390)), on the IEEE 123-bus test feeder.

The pipeline runs end to end: feeder model → admittance matrix → Kron reduction →
load model fitting → DP mechanism → AC power flow → evaluation figures. A 74-check
verification suite passes cleanly.

Beyond reproducing the paper, this repo does two things the paper does not: it
tests whether **Bounded-Noise Privacy** can be substituted for the Gaussian
mechanism at either noise-injection point (figures 4–7), and it audits the
paper's own noise calibration, which turns out not to deliver the privacy it
states at the epsilons the paper runs at.

---

## What the paper proposes

Utilities want to publish voltage phasor data for research, but raw phasors leak
information about individual customers' consumption. The paper's approach is to
avoid releasing measured voltages at all. Instead:

1. Fit a statistical model of load behaviour per consumer class.
2. Add calibrated Gaussian noise to the *load model parameters*, not to the voltages.
3. Draw synthetic loads from the privatised model.
4. Run those synthetic loads through the true network physics via AC power flow.

The output is a set of voltage phasors that are physically consistent — they satisfy
the real power flow equations on the real network — but carry a formal differential
privacy guarantee inherited from the noise added upstream. Theorem 1 in the paper
gives the sensitivity bound that makes this calibration possible, expressed in terms
of the power flow Jacobian norm and the Kron-reduced admittance matrix.

---

## Findings

**The mechanism reproduces.** The fidelity/privacy tradeoff has the shape the paper
reports — Wasserstein-1 distance degrades smoothly as epsilon tightens, and the
masked-recovery attack loses accuracy in the expected direction.

### The calibration formula does not deliver its stated guarantee

The paper's Table III calibration is the classical Gaussian mechanism,
`sigma = S * sqrt(2 ln(1.25/delta)) / eps`. That derivation **requires eps ≤ 1**.
Above that it does not merely get loose — it under-noises, returning a sigma that
does not achieve the `(eps, delta)` it claims. The paper runs at eps 25–200.

Feeding the returned sigma back through the exact Balle & Wang condition
(`analytic_gaussian_delta`), at sensitivity 0.0267:

| eps | sigma | delta claimed | delta actually achieved |
|-----|-------|---------------|-------------------------|
| 5   | 2.66e-2 | 5e-6 | 6.3e-7 ✓ |
| 25  | 5.32e-3 | 5e-6 | **4.2e-3** |
| 50  | 2.66e-3 | 5e-6 | **0.47** |
| 200 | 6.65e-4 | 5e-6 | **1.0** (no privacy at all) |

At eps=50 — the operating point used as the Gaussian reference throughout — the
real delta is **0.47, not 5e-6**. Five orders of magnitude.

`dp_fit_class` now defaults to `calibration="analytic"`: the Balle & Wang analytic
Gaussian mechanism, with the two releases (mean, covariance) composed in zCDP
rather than by naive eps/2 halving. It hits the target delta exactly at every
epsilon tested. The old path remains available as `calibration="classical"` purely
to quantify the gap, and `verify.py` pins it as failing so it cannot silently
return as the default.

The cost is smaller than expected, because the improved composition partly pays
for the tighter calibration:

| eps | sigma_cov multiplier | KL to true fit |
|-----|---------------------|----------------|
| 25  | **0.86x** (less noise) | 116.9 → 97.4 |
| 50  | 1.05x | 50.4 → 53.4 |
| 200 | 1.73x | 8.4 → 17.2 |

`rho_split` exposes the budget division as a tunable: shifting budget toward the
covariance (which carries the temporal structure) cuts KL from 53.4 to 37.5 at
eps=50, at identical delta.

### Bounded-Noise Privacy is not a viable substitute — at either stage

BNP (Severtson & Khajenejad) replaces unbounded noise with uniform noise on
`[-B, B]`, so a released value can never be more than B from the truth. `B = S/2δ`
by their Corollary 1. Tested as a drop-in replacement at both injection points,
against the **corrected** Gaussian baseline:

**Output stage (on released voltages) — fails outright.** Two constraints bracket
B from opposite sides: Corollary 1 needs `B ≥ S/2`, and usefulness needs B under
the ANSI C84.1 half-band of 0.05 pu. On this feeder those intervals do not
intersect. The smallest admissible bound buys `delta = 1.0` — no privacy — while
putting ~60% of released voltages outside the regulation band and taking lag-1
autocorrelation to 0.00. Figure 4 shows the empty region.

**Input stage (on the load model) — dominated.** This is the promising place:
the per-record sensitivity of a mean over m≈2700 records carries a 1/m factor, so
B lands small relative to the log-load range. It still loses. Its best temporal
structure (lag-1 0.59) arrives at `delta = 0.5`, which is **50,000x weaker
privacy** than the Gaussian path's 1e-5 — and still below that path's lag-1 0.78.
Every swept operating point lands in the region of strictly worse privacy *and*
strictly worse structure.

The shape to notice: every utility axis improves monotonically as B falls, but
delta rises exactly as fast, because `delta = S/2B` ties them together. There is
no knee. Buying utility costs privacy one-for-one.

**This conclusion survives the calibration fix**, which is what makes it worth
stating. The obvious objection — that BNP was losing to a baseline flattered by
five orders of magnitude of privacy it never had — is now closed.

**What BNP does buy, honestly:** a hard worst-case bound on any single released
value, which no Gaussian sigma can provide. If a deployment's binding requirement
is "no released voltage may ever leave the band", that guarantee has real value.
It is simply not what this method is measured on, and the price is steep.

### Two open questions the paper does not address

Both are questions rather than claimed errors:

1. **Preprocessing sensitivity.** The paper does not specify how closed switches and
   dangling stub nodes are handled before Kron reduction. These choices are routine
   and easy to leave undocumented, but here they move the condition number of the
   Kron-reduced admittance matrix by roughly **18 orders of magnitude**. Since κ feeds
   the Theorem 1 sensitivity bound, the privacy calibration depends heavily on an
   unstated modelling decision.

2. **Adjacency radius.** Working backwards from the epsilon values reported in the
   paper, the implied adjacency radius corresponds to a relative load perturbation on
   the order of **10⁻¹⁶** — comparable to floating-point epsilon, and far below any
   physically meaningful change in a customer's consumption. Either the adjacency
   relation is defined differently than assumed here, or the reported epsilons
   correspond to a weaker notion of neighbouring datasets than the natural one.

### Four silent bugs

Each produced plausible numbers while being wrong, which is the dangerous kind.
The first three were in this reimplementation; the fourth is in the paper's own
stated calibration:

- Deleting stub nodes outright instead of eliminating them created phantom power
  injection — the network no longer conserved power, but nothing errored.
- An incomplete elimination path zeroed the slack offset term in Theorem 1, quietly
  removing a contribution to the sensitivity bound.
- Capturing the admittance matrix while loads were still enabled let OpenDSS regulator
  action corrupt it, producing a negative substation power reading.
- **The classical Gaussian formula under-noising above eps=1** (above). Every
  structural check passed — the covariance was symmetric, positive definite,
  correctly shaped, and degraded monotonically with epsilon. Only auditing the
  delta actually achieved exposed it.

The verification suite exists because of these. Every check in `verify.py`
corresponds to a property that should hold if the physics — or the privacy
accounting — is right.

---

## Figures

| Figure | Question | Script |
|--------|----------|--------|
| 2, 3 | Wasserstein-1 fidelity and masked-recovery attack | `run_days5_6.py` |
| 4 | Why no admissible BNP bound exists on voltages | `run_bnp_figure.py` |
| 5 | 2×2 mechanism grid: {Gaussian, BNP} in × {none, BNP} out | `run_bnp_grid.py` |
| 6 | Sweeping the BNP bound across five orders of magnitude | `run_bnp_sweep.py` |
| 7 | **The viability verdict**: privacy paid vs structure retained | `run_bnp_viability.py` |

Figure 7 is the summary: the left panel puts every mechanism on a privacy/utility
plane so dominance is visible directly, and the right panel gives the cost on all
four measured axes as a multiple of the corrected Gaussian baseline.

## Repository layout

```
dp-voltage-replication/
├── dpvolt/                  package with the core modules
│   ├── __init__.py
│   ├── network.py           feeder loading, admittance matrix, Kron reduction
│   ├── loads.py             log-normal load model fitting and sampling
│   ├── privacy.py           DP mechanisms, calibration, Theorem 1 bound
│   ├── powerflow.py         OpenDSS driver for AC power flow
│   └── experiments.py       Wasserstein-1 metric, MLP recovery attack
├── get_feeder.py            downloads the IEEE 123-bus feeder (run once)
├── run_day1.py              builds and validates the network model
├── run_days2_4.py           privacy analysis and Theorem 1 evaluation
├── run_days5_6.py           figures 2 and 3
├── run_bnp_figure.py        figure 4 — BNP feasibility on voltages (~1s)
├── run_bnp_grid.py          figure 5 — the 2x2 mechanism grid (~3 min)
├── run_bnp_sweep.py         figure 6 — sweeping the BNP bound (~4 min)
├── run_bnp_viability.py     figure 7 — the viability verdict (~2 min)
├── verify.py                74 correctness checks
├── requirements.txt
└── RUN_GUIDE.md             step-by-step setup, no terminal required
```

### Key entry points in `privacy.py`

| Function | Purpose |
|----------|---------|
| `analytic_gaussian_sigma` | Balle & Wang exact calibration — correct at every eps |
| `analytic_gaussian_delta` | Audits any sigma; this is what found the under-noising |
| `zcdp_rho_from_eps_delta` | zCDP composition, for splitting budget across releases |
| `dp_fit_class` | The DP load-model fit (`calibration=`, `rho_split=`) |
| `bnp_fit_class` | Bounded-noise counterpart, identical except the distribution |
| `gaussian_sigma` | The paper's classical formula — **do not use above eps=1** |

## Running it

1. Create the environment from `requirements.txt`.
2. Run `get_feeder.py` once to download the feeder model.
3. Run `verify.py` — should report **74 of 74** checks passing.
4. Run `run_day1.py`, `run_days2_4.py`, `run_days5_6.py` in that order.
5. For the BNP analysis, run the four `run_bnp_*.py` scripts in any order.

Total runtime after setup is about four minutes for the core pipeline, plus
roughly nine for the full BNP suite. Figures are written to `figures/`.

Dependencies: `numpy`, `scipy`, `opendssdirect.py`, `matplotlib`.

---

## Limitations

- Single feeder (IEEE 123-bus). The paper's conclusions may not transfer to feeders
  with different topology or a different mix of consumer classes.
- The load model is fitted to synthetic class profiles rather than metered data, so
  absolute fidelity numbers are not directly comparable to the paper's.
- Jacobian norm calibration is done by Monte Carlo sampling; the bound is empirical
  rather than analytic.
- **The voltage sensitivity S is empirical**, taken as the largest observed
  `||V1 - V0||` over sampled trajectories, and it is seed-dependent — runs here
  produced values from 0.26 to 0.67 pu. A true worst case would be larger, which
  only widens figure 4's empty region, so the direction is safe; but the BNP
  output-stage numbers rest on a measured rather than proven quantity.
- The BNP conclusions are for uniform bounded noise specifically. A different
  bounded mechanism with better tail behaviour might land differently.
- The two open questions above remain unresolved and are the subject of questions
  submitted to the authors.
