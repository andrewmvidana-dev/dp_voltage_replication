"""Opt-in direct-sum experiments; historical scripts and results stay intact.

Utility uses the saved feeder inputs and original seeds. Timing is a separate
stage so it can run without competing research processes on this machine.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import opendssdirect as dss
from scipy.stats import norm

from dpvolt.loads import reactive_from_active
from dpvolt.powerflow import PowerFlowRunner
from dpvolt.privacy import bnp_distributed_half_width
from dpvolt.secure_agg import PaillierSimulation
from dpvolt.threshold_agg import ThresholdBGVSimulation
from run_privacy_comparison import summary

ROOT = Path(__file__).resolve().parent
MECHANISMS = ['None', 'Gaussian DP', 'BNP', 'Distributed BNP']
BACKENDS = ['None', 'Paillier', 'Threshold BGV']
SENSITIVITY = .020
DELTA = .01


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def prepare(meters, mechanism, seed):
    """Use pu on the public power base; keep a matched service-message slot."""
    rng = np.random.default_rng(seed)
    prepared = meters.copy()
    extra = np.zeros(meters.shape[:1] + meters.shape[2:])
    if mechanism == 'Gaussian DP':
        sigma = SENSITIVITY / (2 * norm.ppf((1 + DELTA) / 2))
        prepared += rng.normal(0, sigma / np.sqrt(meters.shape[1]), meters.shape)
    elif mechanism == 'BNP':
        extra = rng.uniform(-SENSITIVITY / (2 * DELTA),
                            SENSITIVITY / (2 * DELTA), extra.shape)
    elif mechanism == 'Distributed BNP':
        b = bnp_distributed_half_width(SENSITIVITY, DELTA, meters.shape[1])
        prepared += rng.uniform(-b, b, meters.shape)
    elif mechanism != 'None':
        raise ValueError(mechanism)
    return np.concatenate([prepared, extra[:, None, :]], axis=1)


def solve(runner, loads, theta):
    clipped = np.clip(loads, 0, .2)
    runner.reset()
    voltages, converged = runner.solve_trajectory(clipped, reactive_from_active(clipped, theta))
    if not converged.all():
        raise RuntimeError('Nonconverged trajectory; do not silently drop its error')
    return clipped, voltages


def verify_recorded(out):
    """Recompute recorded summaries and encoding residuals without overwriting them."""
    old = json.loads((ROOT / 'results/threshold_20261005/metrics.json').read_text())
    with np.load(ROOT / 'results/privacy_comparison_20260924/synthetic_inputs.npz') as f:
        meters, truth = f['bounded_meters'], f['reference_sum']
    checks, recomputed, crypto = {}, {}, {}
    for m in MECHANISMS[:3]:
        rows = [r for r in old['trials'] if r['mechanism'] == m]
        recomputed[m] = {}
        for key in ['raw_rmse_kw', 'clipped_rmse_kw', 'clipped_fraction', 'voltage_rmse_pu']:
            value = summary([r[key] for r in rows])
            recomputed[m][key] = value
            checks[m + '/' + key] = bool(np.allclose(
                [value['mean'], *value['ci95']],
                [old['utility'][m][key]['mean'], *old['utility'][m][key]['ci95']], rtol=0, atol=1e-12))
        expected = prepare(meters, m, 61005).sum(axis=1)
        for backend in BACKENDS[1:]:
            name = backend + ' / ' + m
            file = (backend + '_' + m).replace(' ', '_') + '.npz'
            actual = np.load(ROOT / 'results/threshold_20261005' / file)['release']
            error = float(np.max(np.abs(actual - expected)) * 1000)
            crypto[name] = dict(max_error_kw=error, label='RECORDED')
            checks[name] = bool(np.isclose(error, old['crypto'][name]['max_error_kw'], rtol=0, atol=1e-10))
    data = dict(label='RECORDED', interpretation='Recomputed from saved trials and ciphertext outputs; no fresh encryption.',
                utility=recomputed, crypto=crypto, checks=checks, all_passed=all(checks.values()))
    dump(out / 'recorded_audit.json', data)
    if not data['all_passed']:
        raise AssertionError('Recorded data audit failed')


def utility(out, trials, run_crypto):
    verify_recorded(out)
    original = np.load(ROOT / 'results/privacy_comparison_20260924/synthetic_inputs.npz')
    meters, truth, vtruth = (original[k] for k in ['bounded_meters', 'reference_sum', 'reference_voltages'])
    runner = PowerFlowRunner(str(ROOT / 'feeders/IEEE123Master.dss'))
    theta = []
    for name in runner.load_names:
        dss.Loads.Name(name)
        theta.append(np.arccos(np.clip(dss.Loads.PF(), -1, 1)))
    theta = np.array(theta)
    retained = runner.retained_indices()
    b = bnp_distributed_half_width(SENSITIVITY, DELTA, 10)
    result = dict(label='MEASURED', settings=dict(trials=trials, steps=truth.shape[1],
        load_rows=truth.shape[0], meters_per_row=10, service_slots=1, seed_start=61005,
        epsilon=0, delta=DELTA, sensitivity_kw=20, clip_kw=[0, 200],
        privacy_unit='Replace one meter reading at one time; synthetic fixed feeder'),
        analytic=dict(label='ANALYTIC', half_width_kw=b*1000, aggregate_bound_kw=10*b*1000,
            aggregate_rms_kw=b*np.sqrt(10/3)*1000, local_delta=SENSITIVITY/(2*b),
            formula='TV=F_IH(n/2+S/(4b))-F_IH(n/2-S/(4b)); RMS=b*sqrt(n/3)'),
        acceptance=dict(label='PROPOSED predeclared before measurement', max_load_error_kw=.003,
            max_voltage_difference_pu=1.5e-6,
            note='Observed historical maxima are acceptance criteria, not universal encoding bounds.'),
        trials=[], utility={}, paired={}, encrypted_validation={})
    trial0 = {}
    for mechanism in MECHANISMS:
        for trial in range(trials):
            prepared = prepare(meters, mechanism, 61005 + trial)
            clean = prepared.sum(axis=1)
            clipped, volts = solve(runner, clean, theta)
            row = dict(mechanism=mechanism, trial=trial,
                raw_rmse_kw=float(np.sqrt(np.mean((clean-truth)**2))*1000),
                clipped_rmse_kw=float(np.sqrt(np.mean((clipped-truth)**2))*1000),
                clipped_fraction=float(np.mean(clean != clipped)),
                voltage_rmse_pu=float(np.sqrt(np.mean((np.abs(volts[:,retained])-np.abs(vtruth[:,retained]))**2))))
            result['trials'].append(row)
            if trial == 0:
                trial0[mechanism] = clean
                for backend in BACKENDS[1:]:
                    name = backend + ' / ' + mechanism
                    if mechanism == 'Distributed BNP':
                        if not run_crypto:
                            result['encrypted_validation'][name] = dict(status='NOT RUN', reason='--skip-new-crypto requested')
                            continue
                        session = PaillierSimulation() if backend == 'Paillier' else ThresholdBGVSimulation()
                        start = perf_counter()
                        actual = np.array([session.sum(bus, 0, 11, 11, np.random.default_rng(0)) for bus in prepared])
                        wall = perf_counter()-start
                        np.savez_compressed(out / ((backend+'_'+mechanism).replace(' ', '_')+'.npz'), release=actual)
                        evidence = dict(label='MEASURED', wall_s=wall, timing=asdict(session.timing),
                            releases=getattr(session, 'releases', []))
                    else:
                        file = (backend+'_'+mechanism).replace(' ', '_')+'.npz'
                        actual = np.load(ROOT / 'results/threshold_20261005' / file)['release']
                        evidence = dict(label='RECORDED ciphertext output; MEASURED power-flow rerun')
                    _, va = solve(runner, actual, theta)
                    le = float(np.max(np.abs(actual-clean))*1000)
                    ve = float(np.max(np.abs(np.abs(va[:,retained])-np.abs(volts[:,retained]))))
                    evidence.update(max_load_error_kw=le, max_voltage_difference_pu=ve,
                        load_test='PASS' if le <= .003 else 'FAIL',
                        voltage_test='PASS' if ve <= 1.5e-6 else 'FAIL',
                        analytic_encoding_bound_kw=11/(2*1e6)*1000 if backend=='Threshold BGV' else None)
                    result['encrypted_validation'][name] = evidence
        selected = [r for r in result['trials'] if r['mechanism']==mechanism]
        result['utility'][mechanism] = {k:summary([r[k] for r in selected]) for k in selected[0] if k not in ('mechanism','trial')}
        # Checkpoint before the longer cryptographic/timing work.
        dump(out/'comparison.json', result)
        np.savez_compressed(out/'trial0_releases.npz', **trial0)
        print('Completed utility:', mechanism, flush=True)
    for reference in ['Gaussian DP', 'BNP']:
        ref = [r for r in result['trials'] if r['mechanism']==reference]
        rows = [r for r in result['trials'] if r['mechanism']=='Distributed BNP']
        result['paired']['Distributed BNP minus '+reference] = {
            k:summary([a[k]-b[k] for a,b in zip(rows,ref)]) for k in result['utility'][reference]}
    dump(out/'comparison.json', result)
    audit_error_differences(out)


def audit_error_differences(out):
    """Separate utility-cost differences from maximum elementwise encoding error."""
    result = json.loads((out/'comparison.json').read_text())
    original = np.load(ROOT/'results/privacy_comparison_20260924/synthetic_inputs.npz')
    meters, truth, vtruth = (original[k] for k in ['bounded_meters','reference_sum','reference_voltages'])
    runner = PowerFlowRunner(str(ROOT/'feeders/IEEE123Master.dss'))
    theta = []
    for name in runner.load_names:
        dss.Loads.Name(name)
        theta.append(np.arccos(np.clip(dss.Loads.PF(),-1,1)))
    theta = np.array(theta)
    retained = runner.retained_indices()
    for mechanism in MECHANISMS:
        reference = prepare(meters, mechanism, 61005).sum(axis=1)
        _, vr = solve(runner,reference,theta)
        raw_ref = float(np.sqrt(np.mean((reference-truth)**2))*1000)
        voltage_ref = float(np.sqrt(np.mean((np.abs(vr[:,retained])-np.abs(vtruth[:,retained]))**2)))
        for backend in BACKENDS[1:]:
            item = result['encrypted_validation'][backend+' / '+mechanism]
            if item.get('status') == 'NOT RUN':
                continue
            folder = out if mechanism=='Distributed BNP' else ROOT/'results/threshold_20261005'
            actual = np.load(folder/((backend+'_'+mechanism).replace(' ','_')+'.npz'))['release']
            _, va = solve(runner,actual,theta)
            raw_diff = float(np.sqrt(np.mean((actual-truth)**2))*1000)-raw_ref
            vdiff = float(np.sqrt(np.mean((np.abs(va[:,retained])-np.abs(vtruth[:,retained]))**2)))-voltage_ref
            item['paired_raw_rmse_difference_kw'] = raw_diff
            item['paired_voltage_rmse_difference_pu'] = vdiff
            item['paired_utility_test'] = 'PASS' if abs(raw_diff)<=.003 and abs(vdiff)<=1.5e-6 else 'FAIL'
            item['paired_utility_scope'] = 'One saved encrypted trial paired with exactly the same prepared plaintext contributions; not a universal equivalence proof.'
    dump(out/'comparison.json',result)


def timings(out, repeats):
    """End-to-end microbenchmark includes setup for both encryption backends."""
    settings = dict(label='PROPOSED predeclared', repeats=repeats, meter_shape=[10,96], service_slots=1,
        time_equivalence_margin_relative=.10,
        criterion='PASS if paired relative-difference 95% CI lies inside +/-10%; FAIL if wholly outside; otherwise INCONCLUSIVE.',
        scope='Noise preparation + fresh key setup + serialization + aggregation/decryption; local, no network.',
        message_scope='BGV serialized ciphertext and decryption shares exclude key-setup traffic; Paillier fixed-width ciphertext integers exclude exponent metadata.')
    result = dict(settings=settings, samples=[], summary={}, comparisons={})
    dump(out/'timing.json', result)
    meters = np.random.default_rng(99).uniform(0, .020, (1,10,96))
    for rep in range(repeats):
        order = [(b,m) for b in BACKENDS for m in MECHANISMS]
        np.random.default_rng(301 + rep).shuffle(order)
        for backend, mechanism in order:
            start = perf_counter()
            prepared = prepare(meters, mechanism, 93000+rep)[0]
            noise_s = perf_counter()-start
            session = None
            if backend == 'None':
                actual = prepared.sum(axis=0)
            else:
                session = PaillierSimulation() if backend == 'Paillier' else ThresholdBGVSimulation()
                actual = session.sum(prepared, 0, 11, 11, np.random.default_rng(0))
            wall = perf_counter()-start
            if np.max(np.abs(actual-prepared.sum(axis=0))) > 11/(2*1e6)+1e-10:
                raise AssertionError('Encoding error exceeds the analytic BGV bound')
            if backend == 'Threshold BGV':
                release = session.releases[-1]
                byte_count = release['ciphertext_bytes']+release['decryption_share_bytes']
                size_label = 'MEASURED serialized payload bytes, setup excluded'
            else:
                byte_count = prepared.size * (512 if backend == 'Paillier' else 8)
                size_label = 'ANALYTIC fixed-width message estimate'
            row = dict(backend=backend, mechanism=mechanism, repetition=rep, label='MEASURED',
                wall_s=wall, noise_preparation_s=noise_s, message_mib=byte_count/2**20,
                message_label=size_label, crypto_timing=asdict(session.timing) if session else None)
            result['samples'].append(row)
            dump(out/'timing.json', result)
            print('Timing',rep,backend,mechanism,round(wall,3),flush=True)
    for backend in BACKENDS:
        for mechanism in MECHANISMS:
            rows = sorted([r for r in result['samples'] if r['backend']==backend and r['mechanism']==mechanism], key=lambda r:r['repetition'])
            vals = np.array([r['wall_s'] for r in rows])
            result['summary'][backend+' / '+mechanism] = dict(**summary(vals), std_s=float(vals.std(ddof=1)),
                min_s=float(vals.min()), max_s=float(vals.max()), values_s=vals.tolist(),
                message_mib=rows[0]['message_mib'],message_label=rows[0]['message_label'],label='MEASURED')
        base = result['summary'][backend+' / None']
        for mechanism in MECHANISMS[1:]:
            arm = result['summary'][backend+' / '+mechanism]
            relative = summary(np.array(arm['values_s'])/np.array(base['values_s'])-1)
            lo,hi = relative['ci95']
            status = 'PASS' if lo>=-.1 and hi<=.1 else ('FAIL' if lo>.1 or hi<-.1 else 'INCONCLUSIVE')
            spread = float(np.hypot(arm['std_s'],base['std_s']))
            difference = arm['mean']-base['mean']
            result['comparisons'][backend+' / '+mechanism] = dict(status=status,relative_difference=relative,
                mean_difference_s=difference,combined_run_std_s=spread,within_run_spread=abs(difference)<=spread,
                note='Operational +/-10% equivalence margin; not proof of exact equality.')
    dump(out/'timing.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='results/threat_model_20261013')
    p.add_argument('--stage', choices=['utility','timing','errors','all'], default='all')
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--skip-new-crypto', action='store_true')
    a = p.parse_args()
    if a.trials < 2 or a.repeats < 3:
        p.error('need at least two utility trials and three timing repetitions')
    output = ROOT/a.output
    output.mkdir(parents=True, exist_ok=True)
    if a.stage in ['utility','all']:
        utility(output,a.trials,not a.skip_new_crypto)
    if a.stage in ['timing','all']:
        timings(output,a.repeats)
    if a.stage == 'errors':
        audit_error_differences(output)
