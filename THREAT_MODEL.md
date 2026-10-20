# Direct-load-sum threat model

M = MEASURED in this session; R = RECORDED from the historical results; A = ANALYTIC from the stated formula; P = PREDICTED by the task prompt. Brackets in accuracy and utility tables are MEASURED confidence intervals at the predeclared 95% confidence level.

The evaluator compares no protection, noise only, encryption only, and both. The noise alternatives are Gaussian meter shares, central BNP from a trusted service, and distributed uniform meter shares. Encryption alternatives are Paillier and threshold BGV. The model-fitting pipeline is outside this experiment.

The public release is a raw bus-sum vector plus retained complex voltage phasors from publicly clipped sums. The feeder topology, per-node bases, frozen controls, source and load-law parameters are public. The public analyst sees this release; the curious aggregator also sees transit messages; the key thief additionally holds the private key or sufficient authority shares. All comparable cells use the same information rule and neighboring datasets.

Evaluator noise seeds, clean synthetic data, and private random draws are never passed to an adversary classifier. The complete experiment is reproducible, so these simulation seeds must not be used for real private publishing. Known-neighbor classification assumes the adversary knows both candidate datasets. It does not imply arbitrary individual-meter reconstruction from an aggregate.

Gaussian and distributed uniform noise is attached to each meter before encryption. The central BNP service perturbs only the aggregate. Its individual meter messages therefore offer no noise protection after key compromise. For locally noised messages, the attacker receives a weaker per-meter privacy guarantee than the aggregate guarantee.

The voltage attacker uses only retained phasors and a separately constructed public model. It reconstructs clipped nominal load-element setpoints using a validated load-law inversion. A raw YV identity instead identifies actual nodal injections. Capacitors remain in the network matrix, and the source contribution uses public Norton parameters, not private solved slack voltages.

Read [the measured report](results/threat_model_20261013/report.md) for the complete evidence tables, attack intervals, inversion residuals and cost-hypothesis decisions. [Machine-readable metrics](results/threat_model_20261013/metrics.json) retain both predictions and observations. [Prior work](PRIOR_WORK.md) records verified references and unresolved citation gaps.

Limitations: colocated roles, honest noise contributors, synthetic data, no authenticated network, no malicious-party security proof, ideal continuous-noise accounting, fixed-point approximation, one-reading privacy unit and no repeated-release composition policy. The BGV parameters and flooding are not independently certified. Ciphertext refusals are capability demonstrations, not measured security levels.
