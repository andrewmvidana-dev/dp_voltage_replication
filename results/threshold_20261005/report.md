# Threshold BGV comparison results

Actual Lattigo v6.2.0 Thresholdize/Combine implementation of the 2023 Mouchet, Bertrand and Hubaux paper. Research simulation with colocated roles. See [implementation and limits](../../THRESHOLD_ENCRYPTION.md).

## Matched utility

All noise arms: epsilon=0, delta=0.01, replace one meter reading at one time, sensitivity20 kW. Not full-customer privacy. Thirty offline noise trials on fixed synthetic IEEE123 data. One real full-feeder encrypted trial for each encrypted condition; remaining utility trials use equivalent plaintext computation.

| Mechanism | Raw RMSE kW (95% CI) | Voltage RMSE pu (95% CI) | Outputs clipped |
|---|---:|---:|---:|
| None | 0.000 (0.000, 0.000) | 0.00000 (0.00000, 0.00000) | 0.00% |
| Gaussian DP | 794.275 (786.567, 801.984) | 0.09654 (0.09427, 0.09881) | 89.89% |
| BNP | 580.672 (577.017, 584.327) | 0.09640 (0.09361, 0.09918) | 90.33% |

No-noise output has no DP guarantee. Paillier and BGV preserve these mechanisms within measured encoding error. Actual encrypted outputs also passed AC power-flow convergence and matched plaintext voltages within the errors in encrypted_powerflow_validation.json.

Paired BNP minus Gaussian raw_rmse_kw: -213.604; 95% CI [-222.078, -205.129].
Paired BNP minus Gaussian voltage_rmse_pu: -0.000142448; 95% CI [-0.00380657, 0.00352168].

BNP lowers raw noise error significantly in this fixed synthetic experiment, but no clear voltage improvement is established. About90% of outputs are clipped and both noisy voltage RMSEs are around0.096pu. These stress-budget settings have poor utility.

## Local encryption cost

| Condition | Elapsed seconds | Maximum encoding error kW |
|---|---:|---:|
| Paillier / None | 97.873 | 3.07189e-09 |
| Threshold BGV / None | 21.940 | 0.00252701 |
| Paillier / Gaussian DP | 92.496 | 2.39497e-09 |
| Threshold BGV / Gaussian DP | 21.921 | 0.00296982 |
| Paillier / BNP | 95.594 | 3.25773e-09 |
| Threshold BGV / BNP | 21.683 | 0.00240002 |

BGV includes a fresh setup for each bus. Paillier key generation is outside timed sum and separately recorded. No network or security-equivalent parameter matching. Timings are single observations.

BGV ciphertext payload: 500.82 MiB; additional decryption shares: 68.26 MiB. Setup traffic excluded. Paillier fixed-width ciphertext-integer estimate: 2.93 MiB, excluding exponent metadata. Sparse BGV packing therefore trades much larger messages for faster local computation.

## Packing and parallelism

- Paillier: mean 15.0049s; three runs [15.151837400044315, 15.09007920010481, 14.77280969999265].
- BGV serial: mean 0.2481s; three runs [0.2637838999507949, 0.24166110006626695, 0.23880629998166114].
- BGV 4 workers: mean 0.1989s; three runs [0.19117550004739314, 0.20760999992489815, 0.19800890004262328].

Same10-meter x96-value workload. Paillier960 ciphertexts versus packed BGV10. Parallel mode uses4 encryption workers; no distributed parallelism.

## Integration and verification

Full Gaussian model fit: 16.037s, 91 load rows,45days,T4,910 simulated meters,3classes,epsilon50/delta1e-5. Maximum mean difference 8.45139e-06; covariance difference 9.13855e-06, versus same-noise plaintext reference. This is a separate integration test, not the epsilon0 utility experiment or a matched comparison with the older Paillier fit.

Verification: all10 valid 3-of-5 quorums; insufficient/duplicate/invalid quorums rejected; signed values; overflow refusal; vectors spanning multiple ciphertexts; parallel encryption; two-round fitter; backend selection. Existing verification suite:80/80 checks. Threshold suite:5/5 test groups.

## Interpretation under privacy guidance

NIST SP800-226 guides evaluation; this prototype is not certified. Ideal continuous mechanism bounds do not certify NumPy/fixed-point DP. Gaussian shares assume honest unknown noise. BNP uses a trusted independent noise service, not a dealer-free joint sampler. BNP model fitting remains future work. Lattigo flooding and parameters need independent security analysis, authenticated distributed services and safe release accounting.

Sources: [2023 paper](https://doi.org/10.1007/s00145-023-09452-8), [NIST SP800-226](https://csrc.nist.gov/pubs/sp/800/226/final), [Lattigo security guidance](https://github.com/tuneinsight/lattigo/security).