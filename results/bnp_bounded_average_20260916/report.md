# Bounded-average BNP: fresh synthetic experiment

## Scope and fixed protocol

- Five synthetic archives, seeds 2026091601 through 2026091605.
- 14 training days, 7 held-out days; 96 readings/day.
- Counts: 91, 250, 1250, 2500, 5000 synthetic customers, using nested prefixes.
- One residential archetype from make_historical; nominal 10 kW/customer.
- Additional persistent customer multipliers: lognormal with log-sd 0.25, mean 1.
- Public caps 20 and 30 kW, chosen before evaluation; these are experimental assumptions.
- Each customer contributes one bounded vector summarizing their entire training history.
- Fixed-size replacement adjacency; no multiplication of privacy sample size by days.
- Uniform BNP: epsilon=0, delta=0.02; 1000 noise trials per archive/configuration.
- K=1 is a scalar average; K=4/8 are block means with periodic linear interpolation.
- Model configurations and repeated releases are separate experiments, not a joint private release.

## Acceptance criteria, set before evaluation

- Scalar: added-error bound <=1% cap, maximum clipping bias <=0.5% cap,
  empirical 95th-percentile held-out absolute error <=2% cap.
- Structured: added-error bound <=5% cap, maximum clipping bias <=0.5% cap,
  empirical 95th-percentile held-out profile RMSE <=5% cap.
- Percentages use the public cap, not average demand. A larger cap permits a larger absolute error.
- Maximum clipping bias means the largest per-coordinate aggregate bias across the five archives.
- The empirical 95th percentile uses 5000 noise trials on five datasets; it is not a confidence bound.

## Privacy and error accounting

For K block summaries per customer in [0,1], coordinate sensitivity is 1/n.
Allocate delta/K per coordinate: B=K/(2*n*delta). The joint uniform support
overlap gives worst-case delta=1-(1-delta/K)^K <= delta.
Projection to [0,1] and public convex interpolation do not enlarge added error.
This bounds deviation from the clipped fitted query/profile, not from held-out
truth. Clipping, sampling, distribution shift and model error remain.

Gaussian epsilon=0 uses L2 sensitivity sqrt(K)/n and exact sigma
=sqrt(K)/(2*n*Phi_inverse((1+delta)/2)). Gaussian epsilon=1 uses the
existing analytic_gaussian_sigma unchanged. The epsilon=1 row is not matched privacy.

## Every BNP case

| Customers | Cap kW | Blocks | Guaranteed added bound kW | Held-out error p95 kW | Max clipping bias kW | Pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 91 | 20 | 1 | 5.4945 | 5.1970 | 0.0163 | no |
| 91 | 20 | 4 | 21.9780 | 9.8433 | 0.4591 | no |
| 91 | 20 | 8 | 43.9560 | 10.2325 | 1.3765 | no |
| 91 | 30 | 1 | 8.2418 | 7.7969 | 0.0000 | no |
| 91 | 30 | 4 | 32.9670 | 15.6160 | 0.0196 | no |
| 91 | 30 | 8 | 65.9341 | 16.2565 | 0.0712 | no |
| 250 | 20 | 1 | 2.0000 | 1.8996 | 0.0166 | no |
| 250 | 20 | 4 | 8.0000 | 5.6994 | 0.3898 | no |
| 250 | 20 | 8 | 16.0000 | 8.3176 | 1.1748 | no |
| 250 | 30 | 1 | 3.0000 | 2.8642 | 0.0000 | no |
| 250 | 30 | 4 | 12.0000 | 7.9019 | 0.0234 | no |
| 250 | 30 | 8 | 24.0000 | 12.3802 | 0.0810 | no |
| 1250 | 20 | 1 | 0.4000 | 0.3879 | 0.0060 | no |
| 1250 | 20 | 4 | 1.6000 | 2.4744 | 0.2584 | no |
| 1250 | 20 | 8 | 3.2000 | 2.3394 | 0.9009 | no |
| 1250 | 30 | 1 | 0.6000 | 0.5709 | 0.0000 | no |
| 1250 | 30 | 4 | 2.4000 | 2.7231 | 0.0085 | no |
| 1250 | 30 | 8 | 4.8000 | 3.1004 | 0.0450 | no |
| 2500 | 20 | 1 | 0.2000 | 0.1999 | 0.0045 | yes |
| 2500 | 20 | 4 | 0.8000 | 2.2407 | 0.2667 | no |
| 2500 | 20 | 8 | 1.6000 | 1.5552 | 0.9084 | no |
| 2500 | 30 | 1 | 0.3000 | 0.2910 | 0.0000 | yes |
| 2500 | 30 | 4 | 1.2000 | 2.3079 | 0.0070 | no |
| 2500 | 30 | 8 | 2.4000 | 1.7930 | 0.0429 | no |
| 5000 | 20 | 1 | 0.1000 | 0.1003 | 0.0047 | yes |
| 5000 | 20 | 4 | 0.4000 | 2.1484 | 0.2607 | no |
| 5000 | 20 | 8 | 0.8000 | 1.2276 | 0.9043 | no |
| 5000 | 30 | 1 | 0.1500 | 0.1471 | 0.0000 | yes |
| 5000 | 30 | 4 | 0.6000 | 2.1532 | 0.0069 | no |
| 5000 | 30 | 8 | 1.2000 | 1.2148 | 0.0381 | yes |

The scalar passes at n=2500 and 5000 for both caps. The only structured
pass is n=5000, cap=30 kW, K=8. All other BNP cases fail at least one criterion.

## Gaussian comparison

| Customers | Cap kW | Blocks | Mechanism | Noise sd kW | Held-out error p95 kW | Added error >1% cap, fraction |
|---:|---:|---:|---|---:|---:|---:|
| 2500 | 20 | 1 | uniform | 0.1155 | 0.1999 | 0.0000 |
| 2500 | 20 | 1 | gaussian_eps0 | 0.1596 | 0.3210 | 0.2122 |
| 2500 | 20 | 1 | gaussian_eps1 | 0.0132 | 0.0473 | 0.0000 |
| 5000 | 30 | 8 | uniform | 0.6928 | 1.2148 | 1.0000 |
| 5000 | 30 | 8 | gaussian_eps0 | 0.3385 | 0.9723 | 0.9766 |
| 5000 | 30 | 8 | gaussian_eps1 | 0.0280 | 0.8142 | 0.0000 |

Uniform improves this scalar comparison at matched epsilon=0, delta=0.02.
The 8-block Gaussian baseline has lower shape error. BNP offers a strict
small added-error bound, not general superiority. Gaussian is projected
to the same public domain but has no comparable small guaranteed query-error cap.

## Exploratory voltage transfer

Only IEEE123 was solved. No additional candidate feeders were evaluated.
A synthetic 5000-customer cohort is NOT a verified population on IEEE123.
Map the first predeclared archive's 8-block, 30 kW-cap profile onto all
91 public load ratings using cohort mean/10 kW as a common multiplier.
This is a scenario transfer, not a spatially resolved customer-load reconstruction.
Noise seeds 2026091700..2026091711; controls frozen; circuit reset per trajectory.

- All 1344 time-step solves converged.
- Nonprivate structured voltage-magnitude RMSE: 0.004761 p.u.
- BNP voltage-magnitude RMSE range: 0.004958 to 0.007081 p.u.
- Largest observed pointwise magnitude error: 0.030344 p.u.
- Held-out mean-profile peak: 18.0739 kW/customer; nonprivate coarse fit peak: 17.0441.

These are empirical voltage errors, not guaranteed bounds. The model
represents a population mean curve, not daily variability or covariance.
It does not establish topology privacy or preserve the original paper's
truncated-lognormal assumptions. Real-meter data, nonstationarity and
spatial diversity remain untested.

## Reproduction and checks

Run from the repo using the existing Python environment:

    python run_bnp_bounded_average.py --output-dir <NEW_DIRECTORY>

An existing output directory is rejected. Existing results are not read
or overwritten. The runner includes bound, interpolation and joint-overlap assertions.
The original verification suite passed 80/80 after the in-memory experiments.
No original mechanism, calibration, eigenvalue floor or numerical default changed.
The figure's shaded band surrounds the clipped fitted curve, not held-out truth.

Dagan-Kur is a different bounded-noise mechanism and was not implemented:
https://proceedings.mlr.press/v178/dagan22a.html
