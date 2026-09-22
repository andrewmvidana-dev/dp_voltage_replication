# Secure aggregation validation

Measured on 2026-09-22 using Python 3.14.3, Windows 11, `phe` 1.5.0,
`gmpy2` 2.3.1, and real 2048-bit Paillier keys.

## Checks

The existing suite passed all 80 checks. The combined run passed 98/98 checks;
five subsequently added checks of compensated/uncompensated encrypted fitting,
private diagnostics, public bounds, and invalid modes also passed. All 23 new
checks are now included by `python verify.py --secure-agg`.

The original Git version and the refactored trusted fitter produced bitwise
identical arrays and identical reports for analytic, zCDP, and classical
calibration on a seeded comparison. Secure mode is opt-in.

Noise-free encrypted sums had maximum error `1.20e-12` pu in the focused tests
and `2.49e-12` pu across the full feeder benchmark, below `1e-9` pu.

| Noise test, target sigma = 0.03 | Expected variance | Empirical variance |
|---|---:|---:|
| Ten reporters | 0.00090000 | 0.00090083 |
| Seven of ten, uncompensated | 0.00063000 | 0.00062907 |
| Seven of ten, compensated | 0.00090000 | 0.00089524 |

Each row uses 50,000 noise trials. Another 256 draws used ten meters and real
encryption throughout; empirical variance divided by target variance was
`1.00020`, inside the 99.9% chi-square acceptance interval `[0.73400, 1.31732]`.

## Voltage utility

Thirty-two encrypted fits of a representative three-bus class, with two meters
per bus, 360 history days, and six time samples, drove all 91 feeder load rows.
The comparison uses independently noised fits and paired downstream sampling.

| Metric | Trusted mean | Secure mean | 90% CI, secure minus trusted | Equivalence margin |
|---|---:|---:|---|---:|
| Magnitude prediction R^2 | 0.90714 | 0.90914 | [-0.00089, 0.00490] | 0.02 |
| ANSI exceedance fraction | 0.26236 | 0.26390 | [0.00015, 0.00293] | 0.005 |
| Mean lag-1 correlation | 0.08375 | 0.08580 | [-0.00650, 0.01060] | 0.05 |

All confidence intervals lie within the predeclared practical equivalence
margins. The ANSI interval excludes zero: its small increase should not be
described as proof of identical distributions. These results concern this
operating point. The encrypted fitter uses a private covariance trace for
eigenvalue repair; the original trusted fitter uses an unnoised trace.

## Full-feeder runtime

The full-feeder experiment used 91 OpenDSS load elements, ten meters per row,
and four time samples. These are serial wall-clock measurements, excluding
network transport. Two benchmark processes overlapped for part of this run,
so timing reflects the machine's workload during measurement.

| Operation | Measured time |
|---|---:|
| Scalar bus sums: encryption per feeder timestep | 13.652 s |
| Scalar bus sums: decryption per feeder timestep | 0.401 s |
| Scalar bus sums: total wall time per timestep | 14.207 s |
| Scalar benchmark: key generation once | 0.028 s |
| Full encrypted model fit, T = 4 | 215.544 s |
| Model-fit encryption, total | 214.254 s |
| Model-fit decryption, total | 0.186 s |
| Model-fit key generation once | 0.071 s |

The model fit encrypted 12,740 scalars and decrypted 42 aggregate scalars,
covering both model rounds for all three classes. Its encryption time divided
by four is 53.564 seconds per model time coordinate; this is an amortized
offline fitting cost, not streaming latency. Both fitted models sampled loads
that converged through the existing power-flow path.

At T = 96, the same unbatched implementation would encrypt 4,324,320 scalars.
Scaling the measured per-scalar encryption time projects roughly 20 hours per
fit on this machine. That projection is not a measured 96-step run. Start with
the small-horizon benchmark before enabling the full original sweep.

## Interpretation

This implements encrypted **model fitting**, as clarified in the task, and
retains the original calibration scales. Direct bus-noise aggregation is a
separate primitive used for testing and runtime measurement.

The proportional-meter assumption, finite-precision Gaussian approximation,
public model metadata, dropout bias, and key-holder trust remain material.
Ordinary Paillier does not prevent its private-key holder from decrypting an
individual ciphertext. This implementation is not a proof that the trusted
curator has been removed against a malicious utility. See
[the integration guide](../SECURE_AGGREGATION.md) for the threat model.

Raw evidence: [full check log](secure_agg_checks.txt),
[variance and utility data](secure_agg_verification.json), and
[full-feeder benchmark](secure_agg_fit.json).
