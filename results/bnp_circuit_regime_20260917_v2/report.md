# BNP circuit-regime experiment: EPRI Ckt5

This experiment tests whether BNP preserves the original load and voltage
dataset when applied at the load-model stage. It uses load-object-day
records and makes no claim about customer counts.

## Predeclared protocol

- Public Ckt5 OpenDSS model at revision `5005c668a72d20775f4c2d060feebb2866ba1d38`.
- 365 historical days and 4 held-out evaluation days.
- BNP privacy: `(epsilon=0, delta=0.02)`; Gaussian: `(epsilon=1.0, delta=0.02)`.
- Two regimes: `(T=96, C=6)` and hourly `(T=24, C=3)`.
- Two fresh seeds per regime and mechanism; every case is reported.
- One record is one load-object-day. Days are not silently relabeled as independent customers.

## Acceptance criterion

A BNP regime passes only if the maximum class noise standard deviation is
below that class's measured median absolute true covariance entry and the
mean covariance lag-1 changes by no more than 0.05. Voltage metrics are
reported separately and are not converted into a privacy guarantee.

## Results

| T | C | records/class | mechanism | epsilon | max noise/signal | covariance lag-1 change | voltage W-1 | voltage lag-1 | ANSI violations | pass |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|
| 96 | 6 | 167900, 167900, 167535 | bnp | 0 | 0.839 | -0.350 | 0.002303 | 0.606 | 4.90% | no |
| 96 | 6 | 167900, 167900, 167535 | gaussian | 1 | 0.068 | -0.098 | 0.002378 | 0.601 | 4.00% | no |
| 96 | 6 | 167900, 167900, 167535 | bnp | 0 | 0.842 | -0.357 | 0.002246 | 0.596 | 4.62% | no |
| 96 | 6 | 167900, 167900, 167535 | gaussian | 1 | 0.068 | -0.099 | 0.002143 | 0.602 | 4.47% | no |
| 24 | 3 | 167900, 167900, 167535 | bnp | 0 | 0.074 | -0.103 | 0.002397 | 0.599 | 4.50% | no |
| 24 | 3 | 167900, 167900, 167535 | gaussian | 1 | 0.006 | -0.073 | 0.002567 | 0.603 | 3.96% | no |
| 24 | 3 | 167900, 167900, 167535 | bnp | 0 | 0.074 | -0.103 | 0.002515 | 0.609 | 4.25% | no |
| 24 | 3 | 167900, 167900, 167535 | gaussian | 1 | 0.006 | -0.073 | 0.002468 | 0.601 | 3.90% | no |

The report distinguishes BNP's `(0, delta)` guarantee from the Gaussian
comparison's `(1, delta)` guarantee. The mechanisms therefore share delta
but not epsilon; the Gaussian row is a stated reference, not a matched
privacy claim. The existing analytic Gaussian calibration and eigenvalue
floor are unchanged.

## Limits

- Ckt5 synthetic histories are generated from the repository's model, not real meter data.
- A lower C or lower T can introduce clipping or aggregation bias; those effects are measured per class in `metrics.json`.
- The voltage comparison is empirical. No global voltage sensitivity bound is claimed.
- Existing IEEE 123 measurements and prior result directories were not rerun or modified.

## Reproduction

    python get_ckt5.py
    python run_bnp_ckt5.py --output-dir results/bnp_circuit_regime_20260917

The runner refuses to overwrite an existing output directory.
