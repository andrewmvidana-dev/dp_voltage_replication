# BNP peer-review comparison

IEEE 123-bus. 12 mechanism seeds, mean ± sd. CLIP_NORM=6.0, COV_FLOOR=0.1, N_HIST_DAYS=90, N_EVAL_DAYS=8, SEED=0.

Gaussian input at (eps=50, delta=1e-05), analytic (Balle & Wang) calibration. BNP input at delta=0.02 -- its own operating point, not matched to the Gaussian, because uniform BNP buys delta as 1/B while the Gaussian buys it exponentially.

| condition | W-1 | ANSI viol | lag-1 ac | R^2 | eig_clipped | KL to true |
|---|---|---|---|---|---|---|
| **true reference** | — | 6.85% | 0.973 | 0.895 | — | — |
| **Gaussian input, no out** | 0.004033 ± 0.000490 | 8.31% ± 0.95% | 0.787 ± 0.012 | 0.877 ± 0.007 | 44.9 ± 0.4 | 6.229e+02 ± 1.034e+01 |
| **BNP input entrywise, no out** | 0.035794 ± 0.001428 | 24.91% ± 1.09% | 0.079 ± 0.030 | 0.673 ± 0.028 | 47.0 ± 0.5 | 5.759e+04 ± 1.086e+03 |
| **BNP in eigen-oracle, no out** <br> **:warning: NOT PRIVATE** | 0.018136 ± 0.001219 | 13.63% ± 0.73% | 0.173 ± 0.048 | 0.730 ± 0.020 | 47.1 ± 2.0 | 8.642e+03 ± 7.594e+02 |
| **BNP in eigen-oracle, NO floor** <br> **:warning: NOT PRIVATE** | 0.022187 ± 0.002080 | 20.43% ± 1.68% | 0.139 ± 0.058 | 0.699 ± 0.026 | 46.2 ± 1.9 | 8.873e+03 ± 7.694e+02 |
| **BNP output only (B = S_adv/2)** <br> delta = 1, i.e. NO privacy | 2.166409 ± 0.003211 | 98.99% ± 0.04% | -0.011 ± 0.004 | -9351.073 ± 190.254 | — | — |

**NOT PRIVATE rows** release the true eigenvectors of the class covariance in the clear. `V` is a function of the raw data, so post-processing does not apply and there is no delta at which these are private. They are ablations that isolate cause, never candidate mechanisms.

**On the output row's `R^2`.** A large negative value is not a measurement of degree. With `B` several times nominal voltage the released series is noise, the probe's standardiser is fitted on that noise, and `R^2` against the true targets diverges — the magnitude reflects the probe, not the mechanism. Read it as "no predictive content"; the informative numbers in that row are `ANSI viol` and `lag-1 ac`.

## Noise magnitudes at the input stage

```
m (records per class)      = 2790
sens_cov = 2 C^2 / m       = 0.025806   (C = 6.0)
B_cov at delta=0.02        = 1.2903
  implied uniform sd B/sqrt(3) = 0.7450
analytic Gaussian sigma_cov  = 0.003919   (eps=50, delta=1e-05)
  ratio  BNP sd / Gaussian sd  = 190.1x
typical |true covariance entry| = 0.0154
  ratio  BNP sd / covariance   = 48.3x
```

The BNP noise sd is **190x the Gaussian sd** and **48x a typical covariance entry**. The eigenvalue floor was never the binding constraint: the noise magnitude alone exceeds the signal it is added to.

## Output-stage sensitivity

```
sampled S (n_trials=10)   = 0.2613 pu   LOWER BOUND, seed-unstable
adversarial S (box corner) = 7.9372 pu   the honest figure
B = S/2 (sampled)          = 0.1306 pu  -> delta = 1.0, 2.6x the ANSI half-band
B = S/2 (adversarial)      = 3.9686 pu  -> delta = 1.0, 79.4x the ANSI half-band   <- the table row uses this
```

The sampled estimator is not merely a lower bound, it is a lower bound of **unstable magnitude**: measured over seeds 0-5 it spans 0.18 to 1.56 pu (sd 0.47, an 8.6x spread) and plateaus by `n_trials=50` at whatever corner its own draws reached, so more trials do not fix it. The box-corner construction spans 1.03x over the same seeds. Since the output bound is `B = S/2`, an under-estimated `S` means less noise for the same claimed delta -- the seed silently sets how favourable this baseline looks.

`B = S/2` is the smallest bound Corollary 1 admits, and `delta = S/(2B) = 1` exactly -- **no privacy at all**, not merely weak privacy. Even there the bound exceeds the entire ANSI regulation half-band by more than an order of magnitude.

