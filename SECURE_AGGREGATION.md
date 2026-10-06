# Encrypted load-model fitting

This is a research adaptation of [Shi et al., NDSS 2011](https://www.ndss-symposium.org/ndss2011/privacy-preserving-aggregation-of-time-series-data/).
Their scheme uses per-user keys and geometric noise. This implementation uses
the `phe` library, 2048-bit Paillier keys, fixed-point encoding, and independent
Gaussian noise shares to match the Gaussian releases in this repository.

## What is integrated

The checked-out Method B fits **class-level log-load means and covariances**.
It does not add one Gaussian perturbation directly to each bus's active power.
After clarification, the integration therefore replaces those two model-fitting
releases. The result is the same `LoadModel` type consumed by `sample_loads`,
`reactive_from_active`, and `PowerFlowRunner`. No additional noise is added to
sampled loads or voltages.

`experiments.py` exposes `MODEL_FIT_CONFIG`. Its default `trusted_curator` mode
keeps the original fitter, options, random-number consumption, and outputs.
`run_days5_6.py` uses that config through its existing `dp_model` function.
Other scripts that directly call `dp_fit_class` retain their original behavior.

```python
from dpvolt.experiments import MODEL_FIT_CONFIG
from dpvolt.secure_agg import SecureAggConfig

MODEL_FIT_CONFIG.mode = "secure_aggregation"
MODEL_FIT_CONFIG.secure = SecureAggConfig(
    n_meters=10,
    split_seed=0,
    dropout=0.0,
    compensate_dropout=False,
)

# The existing entry point now uses encrypted model fitting.
# A full 96-step sweep is very expensive; start with the small benchmark below.
from run_days5_6 import main
main()
```

For a single fit, pass `ModelFitConfig(mode="secure_aggregation", secure=...)`
to `fit_private_load_model`, or to `run_days5_6.dp_model`. The config's
`reports` contains per-class scales and realized dropout variances; `timing`
contains cryptographic runtime. Reuse a `PaillierSimulation` session across
fits when measuring steady-state costs. Its counters are cumulative.

Install dependencies in the repository environment:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install gmpy2
```

`gmpy2` is optional but substantially accelerates integer arithmetic. Existing
trusted experiments do not import `phe` unless secure mode is selected.

## Meter simulation and the two fitting rounds

The pipeline's rows are OpenDSS **Load elements**, which the existing code
calls buses. They are not all 123 physical feeder buses; empty buses require
no meters. Ordering stays identical to `PowerFlowRunner.load_names`.

1. Each row is split among ten customer meters by default. Positive Dirichlet
   weights have a fixed seed and remain constant over days and time. Explicit
   positive weights with row sums of one can be supplied instead.
2. A meter with weight `w` receives `w * bus_load`. It computes
   `log(own_reading / w)` locally and contributes `w` times its sufficient
   statistic. Summing over that bus's meters recovers the original bus record's
   contribution. Records retain the original bus-day adjacency, not a newly
   established arbitrary-customer adjacency.
3. For the mean, each meter contributes its weighted sum of log records divided
   by the public total record count. With `N` enrolled meters in a class, it
   adds independent noise with SD `sigma_mu / sqrt(N)`, encrypts each entry,
   and sends only ciphertexts to the collector.
4. The collector adds ciphertexts. A separate decryptor decrypts just the
   aggregate. The released noisy mean is broadcast to meters.
5. Each meter centers its own records on that mean, applies the original L2
   clipping rule, and forms its weighted covariance contribution. The upper
   triangle is encrypted with independent noise SD `sigma_cov / sqrt(N)`
   off the diagonal and `sqrt(2) * sigma_cov / sqrt(N)` on the diagonal.
   This matches the original symmetrized Gaussian matrix distribution.
6. The covariance sum is decrypted, symmetrized, and repaired. The **noisy**
   trace determines the eigenvalue floor. The original fitter's unnoised trace
   would disclose another private statistic, so exact covariance equality with
   that fitter is not promised. No clean KL or clipping-count diagnostic is
   released by secure mode.

The normalized-reading construction is specific to proportional simulated
customers. An actual customer's varying load share generally does not reveal
the bus's total. Paillier addition cannot compute `log(sum(customer_loads))`.
Real independent profiles would require a different record/model definition or
a protocol supporting that nonlinear computation. This is not implemented.

## Noise, dropout, and accounting

`gaussian_fit_scales` factors out the original fitter's calibration arithmetic.
Both paths use its `sigma_mu` and `sigma_cov`, calibration choice, and budget
split. This work does not independently certify the existing privacy accountant.
The mathematical sum of independent Gaussian shares has the target Gaussian
distribution; finite-precision encryption approximates it. Each encoded scalar
uses explicit precision `1e-12`, so quantization error accumulates with the
number of contributors. Noise-free bus sums are checked to `1e-9` pu.

For each release, let `sigma` denote its original scalar noise SD. With `N`
enrolled meters and `K` reporting meters:

| Setting | Per-report noise variance | Aggregate variance |
|---|---:|---:|
| No dropout | `sigma^2 / N` | `sigma^2` |
| Dropout, no compensation | `sigma^2 / N` | `(K/N) * sigma^2` |
| Dropout, compensated | `sigma^2 / K` | `sigma^2` |

For ten meters with 30% dropout, the variance ratio is 0.7 without compensation
and 1 with compensation. Covariance diagonals have twice the tabulated variance.
The requested fraction removes `floor(N * dropout)` randomly selected identities;
reports contain the actual fraction. Both fitting rounds use the same committed
roster. Fewer than two reporters is rejected. Compensation assumes that this
roster is known before meters draw noise and remains fixed through both rounds.

The full-record denominator and original split weights stay fixed when meters
drop out. Thus missing signal contributions create bias. Noise compensation
does not impute or rescale those contributions. Restoring noise variance alone
does not restore the no-dropout fitted distribution or establish unchanged
privacy under data-dependent dropout, collusion, or a new adjacency definition.
No privacy amplification from dropout is claimed. Without compensation, the
original noise budget is explicitly marked as not preserved.

`aggregate_bus_loads` separately implements direct per-bus noisy summation for
arithmetic tests and streaming benchmarks. Its sigma has units of pu and must
not be substituted for either log-model scale. It is not inserted into Method B.

## Trust boundary

The collector stores only a public key. The decryptor receives aggregate
ciphertexts through the intended interface. The simulation hosts both roles,
the synthetic archive, and meter computations in one Python process.

**Ordinary Paillier does not enforce aggregate-only decryption.** Whoever holds
the private key can decrypt an individual ciphertext if they obtain it. This
code consequently does not, by itself, remove the trusted-curator assumption
against a malicious utility or key holder. The separated roles model an honest,
non-colluding collector and key service. Python object encapsulation is not a
security boundary. Threshold decryption or an appropriate private stream
aggregation protocol with enforceable release rules is needed for that stronger
deployment claim.

This prototype does not implement authenticated meters, verifiable shares,
replay protection, release authorization, or resistance to subset/differencing
queries. NumPy seeds are for reproducible simulation, never for deployment
noise. Paillier keys and encryption randomness are not seeded. Published
membership, bounds, weights, power factors, and rosters are assumed public;
estimating them from private data needs separate accounting. In particular,
do not reuse `fit_load_model`'s observed extrema as public bounds on real data.

## Verification and runtime

```powershell
# Existing checks plus encrypted checks; failures give a nonzero exit code.
.venv/Scripts/python.exe verify.py --secure-agg

# Only the slower encrypted checks.
.venv/Scripts/python.exe verify.py --secure-agg-only

# Complete feeder, ten meters per load row, three scalar timesteps.
.venv/Scripts/python.exe run_secure_agg.py --steps 3

# Also fit both models on every feeder load row, then solve voltages.
.venv/Scripts/python.exe run_secure_agg.py --fit --steps 4 --output results/secure_agg_fit.json

# Dropout demonstration.
.venv/Scripts/python.exe run_secure_agg.py --dropout 0.3 --compensate
```

The variance tests use 50,000 inexpensive independent noise trials for each
dropout setting and 256 fully encrypted draws with ten meters. Chi-square
intervals give a 99.9% statistical tolerance.

The utility test performs 32 independent encrypted model fits on a representative
three-bus class, with two meters per bus, 360 history days, and six time samples.
Each learned class model drives **all** feeder load rows through OpenDSS. Matched
downstream sampling reduces comparison variance. A 90% confidence interval for
the mean metric difference must lie inside predeclared equivalence margins:
0.02 for magnitude prediction R^2, 0.005 for ANSI exceedance fraction, and 0.05
for lag-1 autocorrelation. R^2 is direct magnitude prediction against aligned
reference voltages, not the repository's masked-recovery training metric.
This tests a defined operating point, not equivalence for all privacy budgets,
dropout rates, or 96-step models. Nonconverged trajectories fail the test.

Results are written to `results/secure_agg_verification.json`. The full-feeder
benchmark records key generation, encryption, ciphertext addition, and
decryption separately. Its scalar per-timestep timings are measured serially
over the entire feeder, not parallel meter wall-clock predictions. Optional
model fitting reports total time and time divided by the model horizon; this
amortized figure is **not** an online per-timestep latency. For horizon `T`,
each meter encrypts `T + T*(T+1)/2` values per fit, so a full 96-step fit is
expensive. Network transfer and key-service latency are excluded.

The [measured validation report](results/secure_agg_report.md) records 13.65 s
encryption and 0.40 s decryption per full-feeder scalar timestep, and 216 s
for a four-step fit. A 96-step fit is projected at roughly 20 hours for this
unbatched implementation on the measured machine, not yet benchmarked.

The API follows the [phe documentation](https://python-paillier.readthedocs.io/en/stable/).
The original model-fitting context is described by
[Campbell et al.](https://arxiv.org/abs/2605.02390).
# Optional 2023 threshold BGV extension

An additional backend now implements Lattigo's Thresholdize/Combine operations
with a 3-of-5 authority quorum and packed BGV vectors. Select
`ModelFitConfig(mode="secure_aggregation", encryption_backend="threshold_bgv")`
to use it in the existing Gaussian mean/covariance fitter. Paillier remains the
default. See [THRESHOLD_ENCRYPTION.md](THRESHOLD_ENCRYPTION.md) for build steps,
matched comparisons, and the research security limits. The BNP encrypted
direct-sum experiment uses an independent trusted noise service; dealer-free
BNP model fitting remains unimplemented.
