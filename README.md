# Differentially Private Synthetic Voltage Phasor Release — Replication

An independent replication of Campbell, Zhang, Scaglione, Kerr, Chesler & Peisert,
*"Differentially Private Synthetic Voltage Phasor Release for Distribution Grids"*
([arXiv:2605.02390](https://arxiv.org/abs/2605.02390)), on the IEEE 123-bus test feeder.

The pipeline runs end to end: feeder model → admittance matrix → Kron reduction →
load model fitting → Gaussian mechanism → AC power flow → both evaluation figures.
A 50-check verification suite passes cleanly.

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

## What this replication does

Reimplements the method from the paper description, using OpenDSS as the power flow
engine and the IEEE 123-bus feeder as the test network. Both evaluation experiments
from the paper are reproduced: the Wasserstein-1 fidelity sweep across privacy budgets,
and the MLP masked-recovery attack used to test whether individual load information
survives the mechanism.

---

## Findings

**The mechanism reproduces.** The fidelity/privacy tradeoff has the shape the paper
reports — Wasserstein-1 distance degrades smoothly as epsilon tightens, and the
masked-recovery attack loses accuracy in the expected direction.

**Two discrepancies emerged that the paper does not address.** Both are open questions
rather than claimed errors:

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

**Three silent bugs were found and fixed during development.** Each produced plausible
numbers while being wrong, which is the dangerous kind:

- Deleting stub nodes outright instead of eliminating them created phantom power
  injection — the network no longer conserved power, but nothing errored.
- An incomplete elimination path zeroed the slack offset term in Theorem 1, quietly
  removing a contribution to the sensitivity bound.
- Capturing the admittance matrix while loads were still enabled let OpenDSS regulator
  action corrupt it, producing a negative substation power reading.

The verification suite exists because of these. Every check in `verify.py` corresponds
to a property that should hold if the physics is right.

---

## Repository layout

```
dp-voltage-replication/
├── dpvolt/                  package with the core modules
│   ├── __init__.py
│   ├── network.py           feeder loading, admittance matrix, Kron reduction
│   ├── loads.py             log-normal load model fitting and sampling
│   ├── privacy.py           Gaussian mechanism, Theorem 1 sensitivity bound
│   ├── powerflow.py         OpenDSS driver for AC power flow
│   └── experiments.py       Wasserstein-1 metric, MLP recovery attack
├── get_feeder.py            downloads the IEEE 123-bus feeder (run once)
├── run_day1.py              builds and validates the network model
├── run_days2_4.py           privacy analysis and Theorem 1 evaluation
├── run_days5_6.py           produces both evaluation figures
├── verify.py                50 correctness checks
├── requirements.txt
└── RUN_GUIDE.md             step-by-step setup, no terminal required
```

## Running it

1. Create the environment from `requirements.txt`.
2. Run `get_feeder.py` once to download the feeder model.
3. Run `verify.py` — should report **50 of 50** checks passing.
4. Run `run_day1.py`, `run_days2_4.py`, `run_days5_6.py` in that order.

Total runtime after setup is about four minutes. Figures are written to the project
folder.

Dependencies: `numpy`, `scipy`, `opendssdirect.py`, `matplotlib`.

---

## Limitations

- Single feeder (IEEE 123-bus). The paper's conclusions may not transfer to feeders
  with different topology or a different mix of consumer classes.
- The load model is fitted to synthetic class profiles rather than metered data, so
  absolute fidelity numbers are not directly comparable to the paper's.
- Jacobian norm calibration is done by Monte Carlo sampling; the bound is empirical
  rather than analytic.
- The two discrepancies above remain unresolved and are the subject of questions
  submitted to the authors.
