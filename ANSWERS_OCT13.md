# Answers for the meeting

M = MEASURED in this session; R = RECORDED from the historical results; A = ANALYTIC from the stated formula; P = PREDICTED by the task prompt. Brackets in accuracy and utility tables are MEASURED confidence intervals at the predeclared 95% confidence level.

**Can the voltage set be decrypted, or reveal loads?** The voltage set is already plaintext. With complete complex phasors and the public feeder/load model, the validated inversion recovers nominal load-element totals. MEASURED no-noise RMS residual is 0.0013187 kW. Actual individual Paillier and BGV ciphertexts were also decrypted with the compromised key/quorum. Without those capabilities, interface and quorum checks refuse decryption; this is not proof of a cryptographic security level.

**Why add noise?** Encryption does not protect an exact public aggregate. MEASURED known-neighbor accuracy without noise is 100%, including for the encrypted curious aggregator. Local noise remains after decryption and reduces information in public outputs.

**What does combining them add under key theft?** Encryption hides contributions before key compromise; local noise limits what the thief obtains afterward. Gaussian key-thief accuracy is M 51.385% [50.692%, 52.077%]; distributed BNP is M 51.255% [50.562%, 51.947%]. Central-service BNP remains MEASURED 100% because its individual meter messages are unperturbed. An observer who sees only the released sum benefits from central BNP; the failed second layer concerns the stronger message-and-key observer.

**What does protection cost?** MEASURED raw RMSE is 794.275 kW (Gaussian DP), 580.672 kW (BNP), 786.374 kW (Distributed BNP). Voltage RMSE is MEASURED 0.096538 pu (Gaussian DP), 0.096396 pu (BNP), 0.097364 pu (Distributed BNP). Heavy public clipping limits utility. The report separates paired utility-error passes from strict maximum-encoding diagnostic failures and provides repeated runtime intervals. Exact timing equality is not established.

**Which contributions survive the prior-work review?** The supported deliverable is a reproducible threat-model and physics evaluation, including actual individual-ciphertext decryption, validated inversion and exact scalar bounded-noise calibration. Encryption plus DP and bounded distributed perturbations already appear in published work. A broad new-construction claim is not supported. The Severtson-Khajenejad citation is still unverified; see [PRIOR_WORK.md](PRIOR_WORK.md).

[Full evidence, predictions, cost decisions and reproduction commands](results/threat_model_20261013/report.md).
