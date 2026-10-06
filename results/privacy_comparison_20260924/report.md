# BNP, Gaussian and Paillier comparison

Direct bus active-power sums; not model-fitting or topology privacy

91 IEEE 123 load elements, 6 time samples, ten meters per row, 30 noise trials.
Public meter bounds: [0,20] kW. Gaussian epsilon=1, delta=0.01. BNP epsilon=0, delta=0.01.

## Results

| Method | Raw load RMSE (kW) | Load RMSE after clipping (kW) | Voltage RMSE (pu) | Raw-output attack accuracy |
|---|---:|---:|---:|---:|
| BNP | 574.17 | 113.17 | 0.094192 | 50.50% |
| Gaussian | 37.644 | 31.92 | 0.016167 | 60.55% |
| Encryption | 8.5948e-10 | 8.5948e-10 | 4.2173e-13 | 100.00% |
| Encryption + Gaussian | 37.644 | 31.92 | 0.016167 | 60.55% |

Attack accuracy is a synthetic equal-prior test of two neighboring sums. 50% is chance. It is not a general attack-resistance or ciphertext-breaking score.

## Interpretation

Encryption preserves exact sums up to encoding error and hides inputs from the public-key collector under the simulated trust assumptions. Decryption still exposes the exact aggregate, so encryption alone supplies no output DP.
Gaussian and hybrid releases share the same ideal output distribution. Paired trials isolate encoding error. BNP achieves a stricter epsilon at the same delta but uses substantially more noise in this setup.

## Method and limitations

- Synthetic proportional meters, six time samples, fixed feeder controls, no dropouts.
- BNP has the stricter epsilon=0 at the same delta; this is not an equal-epsilon comparison.
- For the full vector, one-event adjacency changes only one sum. Whole-customer trajectories are outside this guarantee.
- Ideal continuous-noise DP accounting; floating-point sampling/encoding is not a certified finite-precision DP implementation.
- Cryptographic roles share one process and assume an honest, separate non-colluding decryptor. Plain Paillier cannot enforce aggregate-only decryption.
- Raw noisy outputs are clipped to public [0,200] kW bounds before power flow. This changes utility, while preserving ideal DP by post-processing.
- Repeated trials are independent offline evaluations, not a composition-free repeated publishing policy.
- Paired Gaussian/hybrid trials use identical noise shares to isolate encryption error. They are not independent estimates.

The reference uses the same public input clipping for every method. Input-clipping distortion is recorded separately in metrics.json. Voltage RMSE averages retained nodes and converged snapshots. Trial-level 95% Student-t confidence intervals are in metrics.json.

Crypto timing covers one actual full-feeder run per encryption method and excludes key generation. Later trials evaluate equivalent plaintext aggregation. Encryption timing is a single-machine observation, not a scalability estimate.

Checks passed: 15/15.

Reproduce: `.venv/Scripts/python.exe run_privacy_comparison.py`

Sources: [Balle and Wang](https://proceedings.mlr.press/v80/balle18a.html), [python-paillier](https://python-paillier.readthedocs.io/en/latest/phe.html), and the local dpvolt modules.
## Equal-budget extension

For an apples-to-apples test, Gaussian and BNP both use epsilon=0 and delta=0.01 on the same scalar query. Gaussian is calibrated by exact total variation. This additional Gaussian run uses the same inputs and 30 fresh noise seeds.

Gaussian theoretical raw RMSE: 797.864 kW. BNP theoretical raw RMSE: 577.350 kW. Both have theoretical neighboring-sum attack accuracy 50.5%. BNP has lower raw squared error at this equal budget.

Measured Gaussian raw RMSE: 797.093 kW. Measured voltage RMSE after clipping: 0.09539 pu. Public output clipping affects 90.23% of outputs. Neither strict-budget setting has good utility here. See matched_budget.json for intervals.
