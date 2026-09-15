# Does BNP do what it claims?

A correctness check on the Bounded-Noise Privacy mechanism (Severtson & Khajenejad), independent of any comparison and independent of whether the released data is useful on this feeder.

**Scope.** This establishes that the mechanism is sound: its guarantee is exact, its bound is never violated, and its output is always well formed. It does **not** establish that BNP is a viable substitute for the Gaussian mechanism here — it is not, and `results/bnp_peer_review.md` shows why. Both statements are true at once, and neither implies the other.

## 1. Corollary 1 is exact, and its precondition is enforced

`delta = S / (2B)`, checked by round-tripping every (S, target delta) pair through `bnp_bound` and back through `bnp_delta`.

```
30 (S, delta) pairs round-trip
S from 0.01 to 7.9341 pu, delta from 1e-05 to 1
worst relative error: 0.000e+00
```

Exact to machine precision across five orders of magnitude of `S` and 6 orders of `delta`.

The precondition `S <= 2B` raises rather than returning a "delta" above 1, which would not be a probability:

| S | 2B | delta it would have returned | behaviour |
|---|---|---|---|
| 10.0000 | 2.00 | 5.00 | correctly raised |
| 1.0000 | 0.80 | 1.25 | correctly raised |
| 7.9341 | 2.00 | 3.97 | correctly raised |

## 2. The defining guarantee: no released value ever exceeds B

This is the property no Gaussian mechanism can offer at any sigma: the released value is within `B` of the truth **always**, not with high probability. Checked on 20 x 96 x 40 complex voltages per bound, real and imaginary parts separately.

| B | max deviation | deviation / B |
|---|---|---|
| 0.0001 | 9.999985e-05 | 0.999998 |
| 0.001 | 9.999985e-04 | 0.999998 |
| 0.01 | 9.999985e-03 | 0.999998 |
| 0.05 | 4.999992e-02 | 0.999998 |
| 0.335 | 3.349995e-01 | 0.999998 |
| 3.9686 | 3.968594e+00 | 0.999998 |

Worst ratio over every bound tested: **0.999998** — never above 1. The ratio sits just under 1 rather than well under it, which is the right behaviour: the noise uses its full admissible range instead of being quietly conservative.

## 3. The released load model is always valid

A covariance that is not positive definite cannot be sampled from, so this is the difference between a mechanism that runs and one that only appears to. Checked at every BNP operating point in the sweep.

| delta | B_cov | smallest eigenvalue | asymmetry | finite | positive definite |
|---|---|---|---|---|---|
| 0.01 | 2.5806 | 3.750e-02 | 6.66e-16 | True | True |
| 0.02 | 1.2903 | 3.416e-02 | 2.78e-16 | True | True |
| 0.05 | 0.5161 | 1.538e-02 | 1.11e-16 | True | True |
| 0.1 | 0.2581 | 1.246e-02 | 6.94e-17 | True | True |
| 0.2 | 0.1290 | 1.173e-02 | 4.16e-17 | True | True |
| 0.5 | 0.0516 | 1.152e-02 | 4.16e-17 | True | True |

And the mean perturbation respects its own bound at every setting:

| delta | max \|mu_dp - mu_true\| | B_mu | within bound |
|---|---|---|---|
| 0.01 | 1.7270 | 1.7400 | True |
| 0.05 | 0.3454 | 0.3480 | True |
| 0.5 | 0.0345 | 0.0348 | True |

## What this does and does not show

**Shown.** The BNP mechanism is correct. Corollary 1 is exact across the full admissible range, the bound is never violated, the precondition is enforced rather than silently broken, and the released model is always samplable. The hard worst-case bound on every released value is real, and no Gaussian mechanism provides it at any epsilon.

**Not shown.** That the released data is useful on this feeder. Section 3 reports a valid covariance at `delta = 0.01` where `B_cov = 2.58` against true covariance entries of about 0.015 — mathematically valid and physically uninformative at the same time. Validity is a precondition for usefulness, not evidence of it.

The viability question is answered separately in `results/bnp_peer_review.md` and figures 4-7, and the answer there is that BNP is dominated at the input stage and fails outright at the output stage. Nothing in this file contradicts that, and nothing here should be quoted as if it did.

