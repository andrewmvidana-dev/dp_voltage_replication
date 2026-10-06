"""Check actual decrypted releases through AC power flow and summarize evidence."""
import json
from pathlib import Path
import numpy as np
import opendssdirect as dss
from dpvolt.powerflow import PowerFlowRunner
from dpvolt.loads import reactive_from_active

root=Path(__file__).resolve().parent
out=root/'results/threshold_20261005'
d=json.loads((out/'metrics.json').read_text())
assert d['complete']
original=np.load(root/'results/privacy_comparison_20260924/synthetic_inputs.npz')
meters,truth=original['bounded_meters'],original['reference_sum']
runner=PowerFlowRunner(str(root/'feeders/IEEE123Master.dss'))
theta=[]
for name in runner.load_names:
    dss.Loads.Name(name);theta.append(np.arccos(np.clip(dss.Loads.PF(),-1,1)))
theta=np.array(theta);retained=runner.retained_indices();validation={}
for mechanism in ['None','Gaussian DP','BNP']:
    assert sum(r['mechanism']==mechanism for r in d['trials'])==30
    rng=np.random.default_rng(61005);prepared=meters.copy();extra=np.zeros_like(truth)
    if mechanism=='Gaussian DP': prepared+=rng.normal(0,d['settings']['gaussian_sd_kw']/1000/np.sqrt(10),prepared.shape)
    if mechanism=='BNP': extra=rng.uniform(-1,1,truth.shape)
    reference=np.concatenate([prepared,extra[:,None,:]],axis=1).sum(axis=1)
    clipped=np.clip(reference,0,.2);runner.reset()
    vref,ok=runner.solve_trajectory(clipped,reactive_from_active(clipped,theta));assert ok.all()
    for backend in ['Paillier','Threshold BGV']:
        actual=np.load(out/(backend.replace(' ','_')+'_'+mechanism.replace(' ','_')+'.npz'))['release']
        error=float(np.max(np.abs(actual-reference))*1000)
        allowed=.0055 if backend=='Threshold BGV' else 1e-5
        assert error<=allowed
        clipped=np.clip(actual,0,.2);runner.reset()
        vactual,ok=runner.solve_trajectory(clipped,reactive_from_active(clipped,theta));assert ok.all()
        verror=float(np.max(np.abs(np.abs(vactual[:,retained])-np.abs(vref[:,retained]))))
        assert verror<1e-4
        validation[backend+' / '+mechanism]=dict(max_load_error_kw=error,max_voltage_difference_pu=verror,all_converged=True)
(out/'encrypted_powerflow_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
messages=d['crypto']['Threshold BGV / None']['releases']
payload=sum(r['ciphertext_bytes'] for r in messages)/2**20
shares=sum(r['decryption_share_bytes'] for r in messages)/2**20
lines=['# Threshold BGV comparison results','',
 'Actual Lattigo v6.2.0 Thresholdize/Combine implementation of the 2023 Mouchet, Bertrand and Hubaux paper. Research simulation with colocated roles. See [implementation and limits](../../THRESHOLD_ENCRYPTION.md).','',
 '## Matched utility','',
 'All noise arms: epsilon=0, delta=0.01, replace one meter reading at one time, sensitivity20 kW. Not full-customer privacy. Thirty offline noise trials on fixed synthetic IEEE123 data. One real full-feeder encrypted trial for each encrypted condition; remaining utility trials use equivalent plaintext computation.','',
 '| Mechanism | Raw RMSE kW (95% CI) | Voltage RMSE pu (95% CI) | Outputs clipped |','|---|---:|---:|---:|']
for m,u in d['utility'].items():
    r,v=u['raw_rmse_kw'],u['voltage_rmse_pu']
    lines.append(f"| {m} | {r['mean']:.3f} ({r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}) | {v['mean']:.5f} ({v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}) | {100*u['clipped_fraction']['mean']:.2f}% |")
lines+=['','No-noise output has no DP guarantee. Paillier and BGV preserve these mechanisms within measured encoding error. Actual encrypted outputs also passed AC power-flow convergence and matched plaintext voltages within the errors in encrypted_powerflow_validation.json.','']
for k,v in d['paired_bnp_minus_gaussian'].items():
    lines.append(f"Paired BNP minus Gaussian {k}: {v['mean']:.6g}; 95% CI [{v['ci95'][0]:.6g}, {v['ci95'][1]:.6g}].")
lines+=['','BNP lowers raw noise error significantly in this fixed synthetic experiment, but no clear voltage improvement is established. About90% of outputs are clipped and both noisy voltage RMSEs are around0.096pu. These stress-budget settings have poor utility.','',
 '## Local encryption cost','',
 '| Condition | Elapsed seconds | Maximum encoding error kW |','|---|---:|---:|']
for k,v in d['crypto'].items():lines.append(f"| {k} | {v['wall_s']:.3f} | {v['max_error_kw']:.6g} |")
lines+=['','BGV includes a fresh setup for each bus. Paillier key generation is outside timed sum and separately recorded. No network or security-equivalent parameter matching. Timings are single observations.','',
 f'BGV ciphertext payload: {payload:.2f} MiB; additional decryption shares: {shares:.2f} MiB. Setup traffic excluded. Paillier fixed-width ciphertext-integer estimate: {6006*512/2**20:.2f} MiB, excluding exponent metadata. Sparse BGV packing therefore trades much larger messages for faster local computation.','',
 '## Packing and parallelism','']
for k,v in d['batching'].items():lines.append(f"- {k}: mean {v['mean_s']:.4f}s; three runs {v['wall_s']}.")
lines+=['','Same10-meter x96-value workload. Paillier960 ciphertexts versus packed BGV10. Parallel mode uses4 encryption workers; no distributed parallelism.','',
 '## Integration and verification','',
 f"Full Gaussian model fit: {d['model_fit']['wall_s']:.3f}s, 91 load rows,45days,T4,910 simulated meters,3classes,epsilon50/delta1e-5. Maximum mean difference {d['model_fit']['max_mean_error']:.6g}; covariance difference {d['model_fit']['max_covariance_error']:.6g}, versus same-noise plaintext reference. This is a separate integration test, not the epsilon0 utility experiment or a matched comparison with the older Paillier fit.",
 '', 'Verification: all10 valid 3-of-5 quorums; insufficient/duplicate/invalid quorums rejected; signed values; overflow refusal; vectors spanning multiple ciphertexts; parallel encryption; two-round fitter; backend selection. Existing verification suite:80/80 checks. Threshold suite:5/5 test groups.','',
 '## Interpretation under privacy guidance','',
 'NIST SP800-226 guides evaluation; this prototype is not certified. Ideal continuous mechanism bounds do not certify NumPy/fixed-point DP. Gaussian shares assume honest unknown noise. BNP uses a trusted independent noise service, not a dealer-free joint sampler. BNP model fitting remains future work. Lattigo flooding and parameters need independent security analysis, authenticated distributed services and safe release accounting.','',
 'Sources: [2023 paper](https://doi.org/10.1007/s00145-023-09452-8), [NIST SP800-226](https://csrc.nist.gov/pubs/sp/800/226/final), [Lattigo security guidance](https://github.com/tuneinsight/lattigo/security).']
(out/'report.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(validation,indent=2))
