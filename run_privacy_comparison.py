"""Matched scalar bus-sum privacy comparison, separate from model-fitting DP.

Run: .venv/Scripts/python.exe run_privacy_comparison.py
The privacy unit is ONE meter reading at ONE time, with public [0,20] kW
bounds. This does not claim customer-trajectory or network-topology privacy.
All generated source data and random seeds are synthetic and public.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import opendssdirect as dss
from scipy.stats import norm, t

from dpvolt.loads import assign_classes, make_historical, reactive_from_active
from dpvolt.powerflow import PowerFlowRunner, bnp_bound, bnp_delta
from dpvolt.privacy import analytic_gaussian_sigma, analytic_gaussian_delta
from dpvolt.secure_agg import SecureAggConfig, PaillierSimulation, split_loads


def summary(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
    return {"mean": mean, "ci95": [mean-half, mean+half]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--output', default='results/privacy_comparison_20260924')
    parser.add_argument('--matched-only', action='store_true', help='Extend an existing run with equal-budget Gaussian testing')
    args = parser.parse_args()
    if args.trials < 2 or args.steps < 2:
        parser.error('need at least two trials and time samples')
    root = Path(__file__).resolve().parent
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.matched_only:
        matched_budget(root, out)
        return
    runner = PowerFlowRunner(str(root/'feeders/IEEE123Master.dss'))
    kw, theta = [], []
    for name in runner.load_names:
        dss.Loads.Name(name)
        kw.append(dss.Loads.kW())
        theta.append(np.arccos(np.clip(dss.Loads.PF(), -1, 1)))
    kw, theta = np.array(kw), np.array(theta)
    raw = make_historical(kw, assign_classes(kw), 1, T=args.steps,
                          rng=np.random.default_rng(20260924))[:, 0, :]
    cfg = SecureAggConfig(n_meters=10, split_seed=17)
    meters, weights = split_loads(raw, cfg)
    sensitivity = .020  # fixed public cap, pu on the project's 1 MVA base
    bounded = np.clip(meters, 0, sensitivity)
    truth = bounded.sum(axis=1)
    retained = runner.retained_indices()
    runner.reset()
    vtruth, ok = runner.solve_trajectory(truth, reactive_from_active(truth, theta))
    assert ok.all(), 'reference failed'
    eps, delta = 1., .01
    sigma = analytic_gaussian_sigma(sensitivity, eps, delta)
    bound = bnp_bound(sensitivity, delta)
    checks = {}
    checks['gaussian_calibration'] = analytic_gaussian_delta(sensitivity, sigma, eps) <= delta*(1+1e-8)
    checks['bnp_calibration'] = abs(bnp_delta(sensitivity, bound)-delta) < 1e-12
    checks['meter_bounds'] = bool(((bounded>=0)&(bounded<=sensitivity)).all())
    report = {
        'scope': 'Direct bus active-power sums; not model-fitting or topology privacy',
        'adjacency': 'Replace one bounded meter reading at one time. Changing a whole trajectory needs composition.',
        'python': platform.python_version(), 'platform': platform.platform(),
        'settings': {'trials': args.trials, 'steps': args.steps, 'meters_per_load': 10,
                     'load_rows': len(kw), 'retained_nodes': len(retained),
                     'sensitivity_kw': sensitivity*1000, 'epsilon_gaussian': eps,
                     'epsilon_bnp': 0., 'delta': delta, 'key_bits': 2048,
                     'gaussian_sd_kw': sigma*1000, 'bnp_half_width_kw': bound*1000,
                     'input_clipped_fraction': float(np.mean(bounded != meters)),
                     'input_clipping_load_rmse_kw': float(np.sqrt(np.mean((truth-raw)**2))*1000),
                     'data_seed': 20260924, 'split_seed': 17, 'noise_seed_start': 5000},
        'limitations': [
            'Synthetic proportional meters, six time samples, fixed feeder controls, no dropouts.',
            'BNP has the stricter epsilon=0 at the same delta; this is not an equal-epsilon comparison.',
            'For the full vector, one-event adjacency changes only one sum. Whole-customer trajectories are outside this guarantee.',
            'Ideal continuous-noise DP accounting; floating-point sampling/encoding is not a certified finite-precision DP implementation.',
            'Cryptographic roles share one process and assume an honest, separate non-colluding decryptor. Plain Paillier cannot enforce aggregate-only decryption.',
            'Raw noisy outputs are clipped to public [0,200] kW bounds before power flow. This changes utility, while preserving ideal DP by post-processing.',
            'Repeated trials are independent offline evaluations, not a composition-free repeated publishing policy.',
            'Paired Gaussian/hybrid trials use identical noise shares to isolate encryption error. They are not independent estimates.'
        ]
    }
    print('Running actual Paillier on all load rows for encryption and hybrid...', flush=True)
    session = PaillierSimulation()
    encrypted = {}
    encrypted_reference = {}
    for method, sd in [('Encryption', 0.), ('Encryption + Gaussian', sigma)]:
        start = perf_counter()
        before = asdict(session.timing)
        noise = np.random.default_rng(5000).normal(0., sd/np.sqrt(10), bounded.shape)
        released = []
        for i, bus in enumerate(bounded):
            # Supplying noise locally and zero extra noise allows an exact
            # paired plaintext audit of the cryptographic operation.
            released.append(session.sum(iter(bus+noise[i]), 0., 10, 10, np.random.default_rng(0)))
        released = np.array(released)
        expected = (bounded+noise).sum(axis=1)
        error = float(np.max(np.abs(released-expected)))
        checks[method+'_quantization'] = error < 1e-9
        encrypted[method] = released
        encrypted_reference[method] = expected
        after = asdict(session.timing)
        report.setdefault('cryptography', {})[method] = {
            'wall_s': perf_counter()-start, 'max_plaintext_difference_kw': error*1000,
            'timing': {k: after[k]-before[k] for k in before if k != 'key_generation_s'},
            'coverage': 'Every feeder load row and time sample, first trial only'}
        print(method, report['cryptography'][method], flush=True)
    report['key_generation_s'] = session.timing.key_generation_s
    # One full encrypted trial; subsequent trials use mathematically identical
    # plaintext sums to avoid confusing crypto throughput with noise sampling.
    rows = []
    for trial in range(args.trials):
        noise = np.random.default_rng(5000+trial).normal(0., sigma/np.sqrt(10), bounded.shape)
        gaussian = (bounded+noise).sum(axis=1)
        uniform = np.random.default_rng(15000+trial).uniform(-bound, bound, truth.shape)
        checks['bnp_support'] = checks.get('bnp_support', True) and bool((np.abs(uniform)<=bound).all())
        outputs = {'BNP': truth+uniform, 'Gaussian': gaussian,
                   'Encryption': encrypted['Encryption'],
                   'Encryption + Gaussian': encrypted['Encryption + Gaussian'] if trial==0 else gaussian}
        for method, released in outputs.items():
            clipped = np.clip(released, 0., sensitivity*10)
            runner.reset()
            volts, converged = runner.solve_trajectory(clipped, reactive_from_active(clipped, theta))
            valid = volts[converged][:, retained]
            ref = vtruth[converged][:, retained]
            row = {'trial': trial, 'method': method,
                'raw_load_rmse_kw': float(np.sqrt(np.mean((released-truth)**2))*1000),
                'clipped_load_rmse_kw': float(np.sqrt(np.mean((clipped-truth)**2))*1000),
                'output_clipped_fraction': float(np.mean(clipped != released)),
                'convergence_fraction': float(converged.mean()),
                'voltage_magnitude_rmse_pu': float(np.sqrt(np.mean((np.abs(valid)-np.abs(ref))**2))) if valid.size else None}
            rows.append(row)
        if (trial+1)%10 == 0:
            print(f'Utility trials {trial+1}/{args.trials}', flush=True)
    report['utility'] = {}
    for method in outputs:
        selected = [r for r in rows if r['method']==method]
        report['utility'][method] = {k: summary([r[k] for r in selected if r[k] is not None])
            for k in selected[0] if k not in ('trial','method')}
    checks['all_powerflows_converged'] = all(r['convergence_fraction']==1 for r in rows)
    # Equal-prior, optimal binary test between sums differing by sensitivity.
    # This diagnostic uses the raw output, before public clipping/power flow.
    count = 200000
    rng = np.random.default_rng(719)
    labels = rng.integers(0, 2, count)
    xg = labels*sensitivity+rng.normal(0,sigma,count)
    xb = labels*sensitivity+rng.uniform(-bound,bound,count)
    pred_b = rng.integers(0,2,count)  # random decision on overlap
    pred_b[xb < sensitivity-bound] = 0
    pred_b[xb > bound] = 1
    empirical_g = float(np.mean((xg>sensitivity/2)==labels))
    empirical_b = float(np.mean(pred_b==labels))
    report['privacy'] = {
        'BNP': {'epsilon': 0., 'delta': delta, 'attack_theory': .5+delta/2,
                'attack_empirical': empirical_b, 'collector_sees_raw': True},
        'Gaussian': {'epsilon': eps, 'delta': analytic_gaussian_delta(sensitivity,sigma,eps),
                'attack_theory': float(norm.cdf(sensitivity/(2*sigma))),
                'attack_empirical': empirical_g, 'collector_sees_raw': True},
        'Encryption': {'epsilon': None, 'delta': None, 'attack_theory': 1.,
                'attack_empirical': 1., 'collector_sees_raw': False},
        'Encryption + Gaussian': {'epsilon': eps, 'delta': analytic_gaussian_delta(sensitivity,sigma,eps),
                'attack_theory': float(norm.cdf(sensitivity/(2*sigma))),
                'attack_empirical': empirical_g, 'collector_sees_raw': False}}
    report['attack_samples'] = count
    for method in ['BNP','Gaussian']:
        r = report['privacy'][method]
        checks[method+'_attack_agrees'] = abs(r['attack_empirical']-r['attack_theory']) < .005
    report['delta_sweep'] = []
    for d in [1e-5,1e-4,1e-3,.01,.05,.1]:
        sg = analytic_gaussian_sigma(sensitivity,eps,d)
        b = bnp_bound(sensitivity,d)
        report['delta_sweep'].append({'delta':d,'gaussian_theoretical_rmse_kw':sg*1000,
            'bnp_theoretical_rmse_kw':b/np.sqrt(3)*1000,
            'gaussian_delta_audit':analytic_gaussian_delta(sensitivity,sg,eps)})
        checks[f'sweep_{d}'] = analytic_gaussian_delta(sensitivity,sg,eps)<=d*(1+1e-8)
    report['checks'] = checks
    with (out/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'synthetic_inputs.npz',raw=raw,bounded_meters=bounded,weights=weights,
                        reference_sum=truth,reference_voltages=vtruth)
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['# BNP, Gaussian and Paillier comparison','',report['scope'], '',
        f'{len(kw)} IEEE 123 load elements, {args.steps} time samples, ten meters per row, {args.trials} noise trials.',
        f'Public meter bounds: [0,20] kW. Gaussian epsilon=1, delta=0.01. BNP epsilon=0, delta=0.01.',
        '', '## Results', '', '| Method | Raw load RMSE (kW) | Load RMSE after clipping (kW) | Voltage RMSE (pu) | Raw-output attack accuracy |',
        '|---|---:|---:|---:|---:|']
    for method,u in report['utility'].items():
        lines.append(f"| {method} | {u['raw_load_rmse_kw']['mean']:.5g} | {u['clipped_load_rmse_kw']['mean']:.5g} | {u['voltage_magnitude_rmse_pu']['mean']:.5g} | {report['privacy'][method]['attack_empirical']:.2%} |")
    lines += ['', 'Attack accuracy is a synthetic equal-prior test of two neighboring sums. 50% is chance. It is not a general attack-resistance or ciphertext-breaking score.',
        '', '## Interpretation', '',
        'Encryption preserves exact sums up to encoding error and hides inputs from the public-key collector under the simulated trust assumptions. Decryption still exposes the exact aggregate, so encryption alone supplies no output DP.',
        'Gaussian and hybrid releases share the same ideal output distribution. Paired trials isolate encoding error. BNP achieves a stricter epsilon at the same delta but uses substantially more noise in this setup.',
        '', '## Method and limitations', '']
    lines += ['- '+x for x in report['limitations']]
    lines += ['', 'The reference uses the same public input clipping for every method. Input-clipping distortion is recorded separately in metrics.json. Voltage RMSE averages retained nodes and converged snapshots. Trial-level 95% Student-t confidence intervals are in metrics.json.',
        '', 'Crypto timing covers one actual full-feeder run per encryption method and excludes key generation. Later trials evaluate equivalent plaintext aggregation. Encryption timing is a single-machine observation, not a scalability estimate.',
        '', f"Checks passed: {sum(checks.values())}/{len(checks)}.", '',
        'Reproduce: `.venv/Scripts/python.exe run_privacy_comparison.py`', '',
        'Sources: [Balle and Wang](https://proceedings.mlr.press/v80/balle18a.html), [python-paillier](https://python-paillier.readthedocs.io/en/latest/phe.html), and the local dpvolt modules.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'checks':checks,'utility':report['utility'],'privacy':report['privacy']},indent=2),flush=True)
    assert all(checks.values()), 'comparison checks failed; inspect metrics'
    matched_budget(root, out)


def matched_budget(root, out):
    """Gaussian epsilon=0 can be calibrated directly by total variation.

    This independent extension reuses the saved synthetic reference, but does
    not repeat the already measured Paillier run or change its measurements.
    """
    original = json.loads((out/'metrics.json').read_text())
    inputs = np.load(out/'synthetic_inputs.npz')
    truth, vtruth = inputs['reference_sum'], inputs['reference_voltages']
    settings = original['settings']
    sensitivity, delta = settings['sensitivity_kw']/1000, settings['delta']
    sigma = sensitivity/(2*norm.ppf((1+delta)/2))
    audited_delta = 2*norm.cdf(sensitivity/(2*sigma))-1
    assert abs(audited_delta-delta) < 1e-12
    runner = PowerFlowRunner(str(root/'feeders/IEEE123Master.dss'))
    retained = runner.retained_indices()
    theta=[]
    for name in runner.load_names:
        dss.Loads.Name(name)
        theta.append(np.arccos(np.clip(dss.Loads.PF(),-1,1)))
    theta=np.array(theta)
    rows=[]
    for trial in range(settings['trials']):
        released=truth+np.random.default_rng(25000+trial).normal(0,sigma,truth.shape)
        clipped=np.clip(released,0,.2)
        runner.reset()
        volts,ok=runner.solve_trajectory(clipped,reactive_from_active(clipped,theta))
        assert ok.all()
        rows.append({'trial':trial,'raw_load_rmse_kw':float(np.sqrt(np.mean((released-truth)**2))*1000),
            'clipped_load_rmse_kw':float(np.sqrt(np.mean((clipped-truth)**2))*1000),
            'output_clipped_fraction':float(np.mean(released!=clipped)),
            'voltage_magnitude_rmse_pu':float(np.sqrt(np.mean((np.abs(volts[:,retained])-np.abs(vtruth[:,retained]))**2)))})
    result={'epsilon':0,'delta':delta,'gaussian_sd_kw':sigma*1000,
        'gaussian_delta_audit':float(audited_delta),'noise_seed_start':25000,
        'gaussian_theoretical_raw_rmse_kw':sigma*1000,
        'bnp_theoretical_raw_rmse_kw':settings['bnp_half_width_kw']/np.sqrt(3),
        'theoretical_attack_accuracy':float(norm.cdf(sensitivity/(2*sigma))),
        'utility':{key:summary([r[key] for r in rows]) for key in rows[0] if key!='trial'},
        'all_powerflows_converged':True,
        'calibration':'At epsilon=0, total variation is 2*Phi(S/(2*sigma))-1. Invert this to match delta exactly.'}
    (out/'matched_budget.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    with (out/'trials_matched.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report=(out/'report.md').read_text(encoding='utf8').split('\n## Equal-budget extension')[0]
    report+='\n## Equal-budget extension\n\n'
    report+='For an apples-to-apples test, Gaussian and BNP both use epsilon=0 and delta=0.01 on the same scalar query. Gaussian is calibrated by exact total variation. This additional Gaussian run uses the same inputs and 30 fresh noise seeds.\n\n'
    report+=f"Gaussian theoretical raw RMSE: {sigma*1000:.3f} kW. BNP theoretical raw RMSE: {result['bnp_theoretical_raw_rmse_kw']:.3f} kW. Both have theoretical neighboring-sum attack accuracy 50.5%. BNP has lower raw squared error at this equal budget.\n\n"
    report+=f"Measured Gaussian raw RMSE: {result['utility']['raw_load_rmse_kw']['mean']:.3f} kW. Measured voltage RMSE after clipping: {result['utility']['voltage_magnitude_rmse_pu']['mean']:.5f} pu. Public output clipping affects {100*result['utility']['output_clipped_fraction']['mean']:.2f}% of outputs. Neither strict-budget setting has good utility here. See matched_budget.json for intervals.\n"
    (out/'report.md').write_text(report,encoding='utf8')
    print('Equal-budget Gaussian extension:',json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
