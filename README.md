# DP voltage replication

Replication of the differentially private voltage-release method in
arXiv:2506.03467, on the IEEE 123-bus test feeder.

**The idea.** Instead of adding noise to voltages, privatise the *load model*
and push private synthetic loads through the real AC power flow on the true
admittance matrix. Released voltages then satisfy the power flow equations
exactly. Theorem 1 says the private loads alone also hide the topology `Y`, with
no extra noise on the admittance matrix.

## Setup

```
pip install -r requirements.txt
python get_feeder.py        # downloads ./feeders/ (~39 kB, once)
```

## Run, in order

| script | what it does | time |
|---|---|---|
| `run_day1.py` | Feeder load, bus classification, Kron reduction, `kappa_Kron` | seconds |
| `run_days2_4.py` | Load model, DP mechanism, Theorem 1, `\|\|M~^-1\|\|` calibration | ~1 min |
| `run_days5_6.py` | Figures 2 and 3 into `figures/` | ~2 min |
| `verify.py` | All correctness invariants, PASS/FAIL | ~30 s |

## Layout

```
dpvolt/network.py      feeder loading, network cleanup, Kron reduction
dpvolt/loads.py        log-normal load model, fitting, truncated sampling
dpvolt/privacy.py      DP mechanism, Theorem 1, Monte Carlo calibration
dpvolt/powerflow.py    OpenDSS driver, per-unit conversion, noise baseline
dpvolt/experiments.py  Wasserstein metric, masked-recovery task, MLP
```

## Where this departs from the paper

1. **Network cleanup (`network.py`, section 3).** Not in the paper. IEEE 123's
   zero-impedance switches and zero-injection dead ends make the Kron reduction
   numerically catastrophic: `kappa_Kron` comes out at 2e25, which forces the
   adjacency radius `r` below ~1e-27 and makes Theorem 1 vacuous. Merging closed
   switches and eliminating dead-end stubs brings it to 3e6. Both fixes are
   physically obvious once seen.

2. **Gaussian mechanism instead of DP-GMM.** The paper's DP-GMM has no public
   implementation. Legitimate under their Section III-C: Theorem 1 only needs
   the released loads log-normal with a known `Sigma`. We lose fit quality, not
   validity — expect the same *ordering* of methods in Figures 2 and 3, not the
   same values.

3. **Manufactured historical data instead of OEDI**, anchored to the feeder's
   real per-bus kW ratings so total demand still matches IEEE 123's 3.6 MW.

4. **Regulator taps frozen**, so `Y` stays fixed as the analysis requires. The
   cost is that regulators cannot respond to synthetic load excursions, so
   Definition 2's "good voltage set" is not strictly satisfied everywhere.

## Open question

The adjacency radius `r` achieving the paper's own epsilon band (25–200) is
~1e-13 in per-unit admittance, against `||Y_pu||_F` of ~1500 — a relative
perturbation of 1e-16, far too small to represent the "line switching" the paper
motivates. Whether that gap is our parameterisation or the bound's own
looseness is the best question to put to the authors.

## A note on `verify.py`

Every serious bug in this project was **silent** — plausible numbers, no
exception. Deleting stub rows instead of eliminating them produced 238,225 kVA
of phantom power; the slack offset `b` evaluating to zero blew `||M~^-1||` up to
1e16; capturing `Y` with loads disabled re-tapped the regulators and read
-216 MW at the substation. Each was caught by an invariant with a known answer.
Run it after any change.
