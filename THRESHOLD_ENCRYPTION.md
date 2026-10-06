# Threshold BGV extension

This implements the **Thresholdize** and **Combine** operations of Christian
Mouchet, Elliott Bertrand and Jean-Pierre Hubaux, *An Efficient Threshold
Access-Structure for RLWE-Based Multiparty Homomorphic Encryption*, Journal
of Cryptology (2023), through pinned Lattigo v6.2.0.
[Paper](https://doi.org/10.1007/s00145-023-09452-8),
[library API](https://pkg.go.dev/github.com/tuneinsight/lattigo/v6/multiparty).

## What changes

The existing 2048-bit Paillier backend stays available. The new optional BGV
backend packs signed fixed-point values into an RLWE ciphertext. Five simulated
authorities each generate a secret component and privately reshare it with
degree-two Shamir polynomials. They collectively generate the encryption key.
Any committed three-authority subset converts its threshold shares into additive
shares, then supplies one masked partial decryption each. The implementation
never reconstructs or sums the complete secret key. Collective switching to
the public zero key releases only the calculated sum in the honest harness.

This changes **who must cooperate to decrypt** and **how many values one
ciphertext carries**. It does not improve epsilon or delta by itself. Three
colluding authorities can decrypt. Two unavailable authorities are tolerated
only when the active quorum is chosen before the operation. Losing a party
mid-protocol requires aborting this research run, not retrying that ciphertext.

`dpvolt/threshold_agg.py` has the same `sum(...)` interface as Paillier, so the
existing Gaussian private mean/covariance fitter accepts it. For example:

```python
from dpvolt.experiments import ModelFitConfig, fit_private_load_model
from dpvolt.secure_agg import SecureAggConfig

config = ModelFitConfig(mode="secure_aggregation",
    secure=SecureAggConfig(n_meters=10), encryption_backend="threshold_bgv")
# Supply the same archive, public bounds/classes and other arguments as before.
fitted = fit_private_load_model(archive, classes, model, theta,
    epsilon, delta, rng, config=config)
```

To choose a different valid quorum or encryption worker count, pass a
`ThresholdBGVSimulation(ThresholdConfig(active=(1,3,5), workers=4))` session.
Authorities and customer meters are different populations.

## Parameters and safety limits

- Lattigo v6.2.0, Go 1.25.1, BGV, ring dimension 16384, ciphertext modulus
  chain 55+55 bits, auxiliary modulus 55 bits, plaintext modulus
  1099511922689, scale 1,000,000. Only additions are evaluated.
- Each ciphertext has 16,384 available slots. The program splits longer
  vectors across ciphertexts and refuses a conservative plaintext overflow
  bound. Small six-coordinate vectors leave almost all slots unused.
- Each rounded contribution has at most 0.5/scale absolute representation
  error, so the aggregate error is at most contributors/(2*scale), assuming
  correct cryptographic decoding. This is an error bound, **not a DP proof**.
- Every `sum` creates fresh keys and ciphertexts. No endpoint accepts a
  previously used ciphertext or asks for a second share under the same key.
- Decryption uses experimental Gaussian flooding with sigma=2^40. This value
  has **not** been derived as a protocol leakage/security bound. Correct
  decoding in tests does not establish cryptographic security. This parameter
  set and summed-secret distribution need independent estimator/protocol
  review before any 128-bit or post-quantum security claim.
- All meter and authority roles share a trusted local machine. Plaintexts
  enter the Go subprocess over standard input. This demonstrates real
  cryptographic arithmetic, not separation of adversarial services.
- No malicious-share proofs, authenticated network, persistent release budget,
  or cross-process aggregate authorization are implemented. Application
  checks do not cryptographically enforce aggregate-only access.

These limitations matter because Lattigo's [security guidance](https://github.com/tuneinsight/lattigo/security)
warns about leakage from multiparty decryption and repeated protocol shares.
Fresh epochs reduce reuse exposure but do not replace a full security analysis.

## Comparison and privacy meaning

`run_threshold_comparison.py` evaluates the 3x3 design: no encryption,
Paillier, threshold BGV; crossed with no noise, Gaussian DP, bounded uniform
noise (BNP). All data are synthetic. The main utility comparison fixes
epsilon=0 and delta=0.01, sensitivity=20 kW, for replacement of **one meter's
one-time reading**, clipped to public [0,20] kW. This is an illustrative
stress setting, not an endorsed privacy budget.

For a scalar sensitivity S, uniform noise on [-B,B] has total variation
min(1,S/(2B)), hence (0,delta)-DP for B=S/(2 delta). At epsilon=0,
Gaussian sigma=S/[2 Phi^-1((1+delta)/2)]. Here B=1000 kW and sigma is about
797.86 kW. Both are large relative to the actual loads. A hard noise bound
does not automatically mean small error. Each trial's full output vector has
the same event-level ideal guarantee because this adjacency changes only one
coordinate. A whole six-reading trajectory needs different accounting.

BNP is already a possible DP mechanism; it is not an independent third
guarantee. Adding Gaussian noise on top preserves appropriate DP but removes
the hard bound on the total perturbation. This extension keeps the two
mechanisms as alternatives so the comparison is interpretable.

Gaussian noise is split across ten independent honest reporting meters.
Colluding meters that reveal their noise reduce the unknown noise variance.
For BNP, an **independent trusted noise service** contributes one encrypted
uniform draw per output. Ten independent uniform shares would not sum to a
uniform draw, so this prototype does not make that substitution. The service
knows its draw and could subtract it from a public release; it is inside the
trusted boundary and must not collude. A dealer-free bounded-noise protocol
and BNP private model fitting are **not implemented**. BNP encryption is
implemented for the direct-sum experiment only. The fitter uses Gaussian DP.

Every encrypted condition runs actual cryptography over the whole feeder on
the first trial. Thirty reproducible noise trials estimate utility through
equivalent plaintext calculations; they are not thirty encryption timings.
Encryption error is checked separately. Trials are offline evaluations, not
composition-free repeated publications. Gaussian and BNP raw outputs are
clipped to public [0,200] kW before OpenDSS. Both raw error and clipping rate
are reported so clipping cannot conceal the noise cost. Confidence intervals
cover random noise on this fixed synthetic feeder, not population uncertainty.

## Standards interpretation

[NIST SP 800-226 (March 2025)](https://csrc.nist.gov/pubs/sp/800/226/final)
provides guidance for evaluating DP guarantees. We use it to organize the
privacy unit, adjacency, contribution bounds, mechanism parameters, trust
boundary and repeated-release accounting. This is not NIST certification and
does not establish a universally acceptable epsilon/delta. Ideal continuous
noise calculations do not certify the NumPy sampler, floating-point arithmetic,
overflow refusals or fixed-point encoding as a finite-precision DP mechanism.
Data-dependent release failures also need accounting in deployment.

Meter dropout has two effects: missing signal biases the sum, and missing
Gaussian noise shares reduce privacy. Rescaling surviving noise can restore
the intended variance when the roster and honest count are valid, but cannot
restore missing load. A 3-of-5 decryption quorum addresses authority dropout,
not missing customer data. Fixed-grid downstream power flow is post-processing
for the stated customer-data release; no topology privacy is inferred.

## Reproduce

Build with an installed Go 1.25.1+ toolchain, from `threshold_mhe`:

```powershell
go mod download
go build -o threshold_mhe.exe .
```

The local workspace also contains a portable official Go download in `.tools/go`.
For that toolchain, use `../.tools/go/bin/go.exe` from `threshold_mhe` and set
GOPATH/GOCACHE to writable workspace directories. `go.mod` and `go.sum` pin
dependencies. Downloaded toolchains, caches and the binary are ignored by Git.

```powershell
.venv/Scripts/python.exe verify_threshold.py
.venv/Scripts/python.exe run_threshold_comparison.py
.venv/Scripts/python.exe validate_threshold_results.py
.venv/Scripts/python.exe make_threshold_figures.py
```

Results are in `results/threshold_20261005`. The full-feeder Gaussian model fit
is an integration test with T=4 and 45 days. The earlier Paillier fit used a
different archive; their wall times must not be presented as a matched speedup.
The separate 10-meter, 96-value, three-repeat microbenchmark is matched on
input work, but the schemes have different precision/security parameters.
Its wall time includes a fresh threshold setup in each call; Paillier key
generation occurs before the timed sum and is reported separately elsewhere.
No network latency is measured. Serialized message sizes are payload estimates,
not measured network traffic.

This is an engineering extension and experimental evaluation. Establishing a
first-of-its-kind research claim requires a systematic literature review and
a new, proved contribution beyond combining existing tools.
