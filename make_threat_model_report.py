"""Build evidence-labeled reports and figures from completed experiment files."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MECHANISMS = ['None','Gaussian DP','BNP','Distributed BNP']
BACKENDS = ['None','Paillier','Threshold BGV']
ADVERSARIES = ['Public analyst','Curious aggregator','Key thief']


def case_name(backend, mechanism):
    if backend == 'None':
        return 'None' if mechanism=='None' else 'Noise only: '+mechanism
    return ('Encryption only: ' if mechanism=='None' else 'Both: '+mechanism+' / ')+backend


def main(out):
    def read(name):
        return json.loads((out/name).read_text(encoding='utf-8-sig'))
    c, k, v, t, d = [read(name) for name in ['comparison.json','key_compromise.json',
        'voltage_attack.json','timing.json','distributed_verification.json']]
    if not k['complete'] or v.get('status')!='complete' or not v['validation']['gate_passed']:
        raise RuntimeError('Experiments are incomplete; refuse to publish a completed report')
    if len(t['samples']) != len(BACKENDS)*len(MECHANISMS)*t['settings']['repeats']:
        raise RuntimeError('Timing repetitions are incomplete')
    if any(m not in v['voltage_only_attacks'] for m in MECHANISMS):
        raise RuntimeError('Missing voltage attack mechanism')
    old = json.loads((ROOT/'results/threshold_20261005/metrics.json').read_text())
    cases = [(b,m) for b in BACKENDS for m in MECHANISMS]
    def cell(b,m,a):
        return next(x for x in k['cells'] if x['backend']==b and x['mechanism']==m and
                    x['adversary']==a and x['channel']=='full_view' and x['target']=='target_meter')
    def vc(b,m):
        return v['cases'][m if b=='None' else (b+'_'+m).replace(' ','_')]
    def pct(x):
        return f"{100*x:.3f}%"
    def measured(x, digits=5):
        return f'M {x:.{digits}g}'
    def accuracy(x):
        lo,hi=x['measured']['ci95']
        return f"M {pct(x['measured']['accuracy'])} [{pct(lo)}, {pct(hi)}]"
    legend = 'M = MEASURED in this session; R = RECORDED from the historical results; A = ANALYTIC from the stated formula; P = PREDICTED by the task prompt. Brackets in accuracy and utility tables are MEASURED confidence intervals at the predeclared 95% confidence level.'
    regression = read('default_regression_audit.json')
    measurements = dict(scope='Direct load sums, not the model-fitting pipeline',
        comparison=c, key_compromise=k, voltage_attack=v, timing=t,
        distributed_verification=d, recorded_audit=read('recorded_audit.json'),
        default_regression_audit=regression, validation_summary=read('validation_summary.json'),
        provenance={name:name for name in ['comparison.json','key_compromise.json','voltage_attack.json','timing.json','distributed_verification.json']},
        missing=['Exact Severtson-Khajenejad BNP citation remains unverified; user asked.',
                 'Optional additional-delta sweep was not run.'],
        interpretation='Ideal continuous-noise theory, actual crypto demonstrations and AC solves are distinct evidence types.')
    # Connect the separately measured voltage channel without changing its scope.
    measurements['key_compromise']['voltage_channel'] = dict(status='complete',evidence='MEASURED',
        source='voltage_attack.json',cells=v['voltage_only_attacks'],
        note='Same public-voltage channel for all observers. Separate ideal plaintext-noise AC trials, not repeated encrypted AC trials.')
    (out/'key_compromise.json').write_text(json.dumps(k,indent=2,allow_nan=False),encoding='utf-8')
    measurements['hypotheses'] = {
        'paired_error': {name:r['paired_utility_test'] for name,r in c['encrypted_validation'].items() if not name.endswith(' / None')},
        'maximum_encoding_diagnostic': {name:dict(load=r['load_test'],voltage=r['voltage_test']) for name,r in c['encrypted_validation'].items()},
        'time':{name:r['status'] for name,r in t['comparisons'].items() if not name.startswith('None / ')}}
    (out/'metrics.json').write_text(json.dumps(measurements,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Encryption and noise: direct-load threat-model results','',legend,'',
        'Noise added at each meter survives key theft. A central noise service does not protect the unperturbed individual messages that the compromised key can decrypt. Encryption alone leaves exact public aggregates exposed.',
        '', '## Scope and evidence', '',
        f"MEASURED workload: {c['settings']['load_rows']} load elements, {c['settings']['meters_per_row']} meters per element, {c['settings']['steps']} times, {c['settings']['trials']} utility trials. ANALYTIC calibration: epsilon=0, delta=0.01, sensitivity=20 kW, one meter reading at one time. Public clipping is [0,200] kW. This is a synthetic IEEE 123-bus test feeder, not field customer data.",
        '', 'The experiment publishes raw noisy sums for each load element and complex retained-node phasors calculated from clipped sums. The code calls these bus sums, but a physical bus can host multiple load-element rows. A separate clipped-sum channel is also evaluated. A public analyst sees the release and public network. A curious aggregator additionally sees transit messages but has no key. A key thief sees the same messages and owns the Paillier key or a committed BGV quorum. Noise seeds and honest meter random draws are evaluator-only information.',
        '', 'The two known neighboring datasets differ only in the target reading. Other target-row meters are fixed at ANALYTIC 10 kW each; the target is ANALYTIC 0 or 20 kW. Guessing this binary world is not recovering an arbitrary customer reading from a bus sum. Full-view bus and meter classification share the same hypothesis; channel-restricted attacks are reported separately.',
        '',f"MEASURED: {k['settings']['trials']} independent noise trials per attack cell, paired across backend/observer cells. Real cryptography is demonstrated separately on each backend/mechanism. The voltage-only experiment performs MEASURED {sum(x['completed_trials'] for x in v['voltage_only_attacks'].values())} actual AC solves, with MEASURED {sum(x['failed_powerflows'] for x in v['voltage_only_attacks'].values())} failures.",
        '', '## Utility and protection cost', '',
        'Utility columns are MEASURED plaintext-equivalent mechanism averages. Encrypted differences are checked below. Runtime is the separately matched microbenchmark, with MEASURED '+str(t['settings']['repeats'])+' repetitions, ANALYTIC workload of ten meters by ninety-six values plus one service slot. It includes fresh key setup for both encryption backends. Message estimates omit setup traffic and are not network measurements. Zero service messages keep the contributor count matched outside central BNP.',
        '', '| Case | Raw RMSE kW | Voltage RMSE pu | Clipped | Micro time s, mean +/- SD | Message MiB | Public accuracy | Aggregator accuracy | Key-thief accuracy |',
        '|---|---:|---:|---:|---:|---:|---|---|---|']
    for b,m in cases:
        u=c['utility'][m]; timing=t['summary'][b+' / '+m]
        size_label='M' if b=='Threshold BGV' else 'A'
        lines.append(f"| {case_name(b,m)} | {measured(u['raw_rmse_kw']['mean'])} | {measured(u['voltage_rmse_pu']['mean'])} | M {pct(u['clipped_fraction']['mean'])} | M {timing['mean']:.5g} +/- {timing['std_s']:.3g} | {size_label} {timing['message_mib']:.5g} | "+' | '.join(accuracy(cell(b,m,a)) for a in ADVERSARIES)+' |')
    lines+=['','The no-encryption microtimings include local noise preparation and array summation. End-to-end voltage computation and networking are excluded for every microbenchmark arm. Numerical attack cells do not certify ciphertext security or finite-precision DP.',
        '', '## Paired distributed-BNP utility differences', '',
        '| Difference | Raw RMSE kW | Clipped RMSE kW | Clipping fraction | Voltage RMSE pu |', '|---|---|---|---|---|']
    for name,values in c['paired'].items():
        row=[]
        for key in ['raw_rmse_kw','clipped_rmse_kw','clipped_fraction','voltage_rmse_pu']:
            x=values[key]; row.append(f"M {x['mean']:.6g} [{x['ci95'][0]:.6g}, {x['ci95'][1]:.6g}]")
        lines.append('| '+name+' | '+' | '.join(row)+' |')
    lines+=['','Distributed BNP has substantially more raw error than central BNP. Its raw and voltage differences from Gaussian include zero in these paired intervals. Heavy clipping dominates this operating point; the result does not establish a useful deployment budget.',
        '', '## Voltage inversion and prediction audit', '',
        'The naive identity yields actual nodal power, not nominal load-element setpoints. The implementation uses the full named, unmerged network with loads disabled in Y; capacitors, transformers and finite source impedance remain. Public Norton source currents supply the offset. All omitted nodes, including source terminals, are eliminated. The attacker receives only retained phasors and public model metadata, never solved private slack voltages.',
        '', 'ANALYTIC units: V_volts=V_pu*Vbase; I_load=J_source-Y_network*V_volts; S_load_kVA=V_volts*conj(I_load)/1000. A public branch-current map handles wye/delta connections, power factor, voltage breakpoints and constant-power/current/impedance load laws. Real and reactive equations jointly identify nominal active-power setpoints. Capacitor power is not subtracted a second time.',
        '',f"MEASURED baseline nominal-setpoint residual: RMS {v['validation']['setpoint_reconstruction_error_kw']['rms']:.9g} kW, maximum {v['validation']['setpoint_reconstruction_error_kw']['max_abs']:.9g} kW. MEASURED nodal power residual maximum {v['validation']['nodal_power_residual_kva']['max_abs']:.9g} kVA. MEASURED matrix rank {v['validation']['minimum_rank']}. Solver voltage-iteration tolerance is MEASURED {v['validation']['solver_voltage_iteration_tolerance_pu']:.6g} pu; the predeclared engineering power gate is ANALYTIC {v['validation']['predeclared_power_residual_limit_kw_or_kva']:.6g} kW/kVA. This normalized gate is not a theorem converting solver tolerance into power error.",
        '',f"MEASURED actual terminal power differs from nominal inputs by up to {v['validation']['actual_terminal_power_minus_nominal_kw']['max_abs']:.6g} kW. Thus the literal exact-setpoint interpretation of V*conj(YV) is false; model-aware reconstruction works within the reported residuals. Recovered loads follow clipped noisy setpoints, not the raw noise before clipping.",
        '', '| Mechanism | PREDICTED voltage-only accuracy | MEASURED accuracy and confidence interval | ANALYTIC ideal classifier |', '|---|---:|---|---:|']
    for m,x in v['voltage_only_attacks'].items():
        lo,hi=x['ci95_wilson']
        lines.append(f"| {m} | P {pct(x['predicted_accuracy'])} | M {pct(x['accuracy'])} [{pct(lo)}, {pct(hi)}] | A {pct(x['analytic_accuracy'])} |")
    lines+=['','The analytic value is for exact recovery of the clipped release in this selected midpoint test. It is not a claim that every voltage post-processing preserves total variation. Measured inversion error and quantization can alter decisions. The AC statistical trials use plaintext-equivalent releases; actual encrypted feeder releases are independently inverted in the case table below.',
        '', '| Case | Recovered vs true load RMSE kW | Recovered vs clipped release RMSE kW |', '|---|---:|---:|']
    for b,m in cases:
        x=vc(b,m);lines.append(f"| {case_name(b,m)} | M {x['recovered_vs_true_load_rmse_kw']:.7g} | M {x['recovered_vs_clipped_release_rmse_kw']:.7g} |")
    lines+=['', '## All task-prompt quantitative predictions', '',
        '| Quantity | PREDICTED | ANALYTIC | MEASURED |', '|---|---:|---:|---:|']
    for m in MECHANISMS:
        x=cell('Threshold BGV',m,'Key thief')
        lines.append(f"| Key-thief accuracy, {m} (same observation law with or without encryption) | P {pct(x['predicted']['accuracy'])} | A {pct(x['analytic']['accuracy'])} | {accuracy(x)} |")
    for m in MECHANISMS:
        x=cell('Threshold BGV',m,'Public analyst');y=cell('Threshold BGV',m,'Curious aggregator')
        lines.append(f"| Public raw-sum accuracy, {m} | P {pct(x['predicted']['accuracy'])} | A {pct(x['analytic']['accuracy'])} | {accuracy(x)} |")
        lines.append(f"| Encrypted curious-aggregator accuracy, {m} | P {pct(y['predicted']['accuracy'])} | A {pct(y['analytic']['accuracy'])} | {accuracy(y)} |")
    cal,mc=d['calibration'],d['monte_carlo']
    lines += [f"| Distributed share half-width, kW | P 430.4 | A {cal['half_width_kw']:.9f} | M largest sampled absolute share {mc['maximum_absolute_share_kw']:.6f}; not an estimate of the hard bound |",
        f"| Distributed aggregate hard bound, kW | P 4304 | A {cal['aggregate_bound_kw']:.9f} | M largest sampled absolute sum {mc['maximum_absolute_sum_kw']:.6f}; finite sampling cannot prove the bound |",
        f"| Distributed aggregate RMS noise, kW | P about 786 | A {cal['aggregate_rms_kw']:.6f} | M {mc['aggregate_rms_kw']:.6f} from independent sums; M {c['utility']['Distributed BNP']['raw_rmse_kw']['mean']:.6f} feeder RMSE |",
        f"| Distributed per-meter effective delta | P 0.0232 | A {cal['individual_delta']:.9f} | M local attack accuracy above; not a direct TV estimate |",
        f"| Central BNP aggregate RMS, kW | P about 577 | A {1000/np.sqrt(3):.6f} | M {c['utility']['BNP']['raw_rmse_kw']['mean']:.6f} feeder RMSE |",
        f"| Gaussian aggregate RMS, kW | P about 798 | A {k['settings']['gaussian_sd_kw']:.6f} | M {c['utility']['Gaussian DP']['raw_rmse_kw']['mean']:.6f} feeder RMSE |",
        '', 'The key-thief accuracy predictions agree with the analytic formulas at their stated rounding and lie inside the measured intervals. The curious-aggregator chance prediction is structurally wrong: it also sees the public release. Exact sums identify the known neighbor. Noisy aggregate accuracy is ANALYTIC 50.5%, not exactly chance, though these sample sizes do not reliably separate such a small advantage.',
        '',f"MEASURED distributed aggregate TV estimate {mc['tv_estimate']:.6f}, interval {mc['ci95_clopper_pearson']}, from MEASURED {mc['samples']} independent sums; ANALYTIC target {cal['delta']}. All sampled shares and sums respected their analytic bounds. The exact-CDF calibration and bounds are in distributed_verification.json.",
        '', '## Do the costs compound?', '',
        'Paired error criteria use absolute differences in raw-load RMSE and voltage-magnitude RMSE on the same prepared draw. The predeclared acceptance settings are 0.003 kW and 1.5e-6 pu, chosen from the task prompt, rather than calculated privacy bounds. Maximum elementwise deviations are separate, stricter diagnostics.',
        '', '| Backend / mechanism | Paired raw RMSE difference kW | Paired voltage RMSE difference pu | Paired utility result | Maximum load / voltage diagnostic |', '|---|---:|---:|---|---|']
    for name,x in c['encrypted_validation'].items():
        lines.append(f"| {name} | M {x['paired_raw_rmse_difference_kw']:.9g} | M {x['paired_voltage_rmse_difference_pu']:.9g} | {x['paired_utility_test']} | {x['load_test']} / {x['voltage_test']} |")
    lines += ['', 'The distributed-BNP BGV maximum elementwise load error exceeds the strict empirical cutoff but remains below the ANALYTIC fixed-point sum bound of 0.0055 kW. The historical encryption-only BGV maximum voltage error also slightly exceeds the rounded cutoff. Neither diagnostic alone establishes an increase in paired utility RMSE. The paired noisy-arm results above support negligible added error on the tested encrypted draw, not universal equivalence.',
        '', 'Time uses a predeclared practical equivalence setting of +/-10% and MEASURED paired intervals. The margin is a chosen test setting, not a calculated privacy bound. A pass requires the entire interval inside that margin. Failure requires the entire interval outside it; other outcomes are inconclusive. Exact equality cannot be established by a finite timing sample.',
        '', '| Backend / mechanism vs no noise | Mean time difference s | Relative difference interval | Within combined run SD? | Result |', '|---|---:|---|---|---|']
    for name,x in t['comparisons'].items():
        if name.startswith('None / '):continue
        lo,hi=x['relative_difference']['ci95']
        lines.append(f"| {name} | M {x['mean_difference_s']:.6g} | M [{pct(lo)}, {pct(hi)}] | {'Yes' if x['within_run_spread'] else 'No'} | {x['status']} |")
    lines+=['','These are local serialized workloads with fixed payload counts, no authenticated network, and different cryptographic parameters. They do not establish security-equivalent throughput or production latency.',
        '', '## Historical full-feeder cost context', '', '| Backend / mechanism | Wall time s | Ciphertext MiB | Decryption-share MiB |', '|---|---:|---:|---:|']
    for b in BACKENDS[1:]:
        for m in MECHANISMS:
            label='M' if m=='Distributed BNP' else 'R'
            x=c['encrypted_validation'][b+' / '+m] if m=='Distributed BNP' else old['crypto'][b+' / '+m]
            if b=='Threshold BGV':
                ct=sum(r['ciphertext_bytes'] for r in x['releases'])/2**20
                sh=sum(r['decryption_share_bytes'] for r in x['releases'])/2**20
                size=f'{label} {ct:.3f} | {label} {sh:.3f}'
            else:size=f'A {91*11*6*512/2**20:.3f} | Not applicable'
            lines.append(f"| {b} / {m} | {label} {x['wall_s']:.5g} | {size} |")
    lines += ['', 'Historical Paillier setup is excluded while BGV setup is included. These single full-feeder observations are context only; the new microbenchmark includes setup consistently. New full-feeder crypto ran alongside other validation work and is not used for timing inference.',
        '', '## Limits and unrun work', '',
        '- All roles are colocated; meters are honest about sampling, and no malicious-meter robustness is demonstrated.',
        '- No authenticated network, safe release ledger, subset-query protection, secure deployment RNG, or certified finite-precision DP is supplied.',
        '- Keys are compromised; meter random draws are not. Compromising both would permit noise subtraction.',
        '- Complete complex phasors, public source parameters, frozen controls and exact load-law metadata are assumed. Voltage magnitudes alone are a different observation.',
        '- Only one reading changes. Repeated releases and whole-customer trajectories need different accounting.',
        '- The optional additional-budget sweep was not run. The exact Severtson-Khajenejad citation remains unverified.',
        '- Prior-work review supports an implementation and empirical audit; it does not support a broad new-construction claim. See [PRIOR_WORK.md](../../PRIOR_WORK.md).',
        '', '## Figures', '',
        '![Measured attack accuracy by adversary](accuracy_by_adversary.png)', '',
        '![Measured recovery from released voltages](voltage_load_recovery.png)', '',
        '![Measured noise utility](noise_utility.png)', '',
        '![Measured cost versus protection](cost_vs_protection.png)', '',
        'Editable SVG counterparts are saved beside each PNG.',
        '', '## Verification', '',
        'MEASURED: core checks 80/80; threshold groups 5/5; distributed-BNP checks 14/14; key-compromise tests 6/6; voltage tests 4/4. Baseline and final logs are saved alongside these results. The source/AST audit confirms existing privacy functions and default verification bodies are unchanged. Every original-mechanism trial metric reproduces the historical data exactly.',
        '', '## Reproduction', '', 'Run from the repository root after building the Go backend as described in THRESHOLD_ENCRYPTION.md.', '', '```powershell',
        '.venv/Scripts/python.exe -B verify.py',
        '.venv/Scripts/python.exe -B verify_threshold.py',
        '.venv/Scripts/python.exe -B verify.py --distributed-bnp-only',
        '.venv/Scripts/python.exe -B verify_key_compromise.py',
        '.venv/Scripts/python.exe -B verify_voltage_set_attack.py',
        '.venv/Scripts/python.exe -B run_threat_model_comparison.py --stage utility',
        '.venv/Scripts/python.exe -B run_key_compromise.py --trials 20000',
        '.venv/Scripts/python.exe -B run_voltage_set_attack.py --attack-trials 20000',
        '# Run timing after the other experiments stop.',
        '.venv/Scripts/python.exe -B run_threat_model_comparison.py --stage timing --repeats 3',
        '.venv/Scripts/python.exe -B make_threat_model_report.py', '```','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    figures(out,c,k,v,t,cases,cell,vc)
    (ROOT/'THREAT_MODEL.md').write_text('\n'.join([
        '# Direct-load-sum threat model','',legend,'',
        'The evaluator compares no protection, noise only, encryption only, and both. The noise alternatives are Gaussian meter shares, central BNP from a trusted service, and distributed uniform meter shares. Encryption alternatives are Paillier and threshold BGV. The model-fitting pipeline is outside this experiment.',
        '', 'The public release is a raw bus-sum vector plus retained complex voltage phasors from publicly clipped sums. The feeder topology, per-node bases, frozen controls, source and load-law parameters are public. The public analyst sees this release; the curious aggregator also sees transit messages; the key thief additionally holds the private key or sufficient authority shares. All comparable cells use the same information rule and neighboring datasets.',
        '', 'Evaluator noise seeds, clean synthetic data, and private random draws are never passed to an adversary classifier. The complete experiment is reproducible, so these simulation seeds must not be used for real private publishing. Known-neighbor classification assumes the adversary knows both candidate datasets. It does not imply arbitrary individual-meter reconstruction from an aggregate.',
        '', 'Gaussian and distributed uniform noise is attached to each meter before encryption. The central BNP service perturbs only the aggregate. Its individual meter messages therefore offer no noise protection after key compromise. For locally noised messages, the attacker receives a weaker per-meter privacy guarantee than the aggregate guarantee.',
        '', 'The voltage attacker uses only retained phasors and a separately constructed public model. It reconstructs clipped nominal load-element setpoints using a validated load-law inversion. A raw YV identity instead identifies actual nodal injections. Capacitors remain in the network matrix, and the source contribution uses public Norton parameters, not private solved slack voltages.',
        '', 'Read [the measured report](results/threat_model_20261013/report.md) for the complete evidence tables, attack intervals, inversion residuals and cost-hypothesis decisions. [Machine-readable metrics](results/threat_model_20261013/metrics.json) retain both predictions and observations. [Prior work](PRIOR_WORK.md) records verified references and unresolved citation gaps.',
        '', 'Limitations: colocated roles, honest noise contributors, synthetic data, no authenticated network, no malicious-party security proof, ideal continuous-noise accounting, fixed-point approximation, one-reading privacy unit and no repeated-release composition policy. The BGV parameters and flooding are not independently certified. Ciphertext refusals are capability demonstrations, not measured security levels.','']),encoding='utf-8')
    g=cell('Threshold BGV','Gaussian DP','Key thief');db=cell('Threshold BGV','Distributed BNP','Key thief')
    answers=['# Answers for the meeting','',legend,'',
        '**Can the voltage set be decrypted, or reveal loads?** The voltage set is already plaintext. With complete complex phasors and the public feeder/load model, the validated inversion recovers nominal load-element totals. MEASURED no-noise RMS residual is '+f"{v['validation']['setpoint_reconstruction_error_kw']['rms']:.6g} kW. Actual individual Paillier and BGV ciphertexts were also decrypted with the compromised key/quorum. Without those capabilities, interface and quorum checks refuse decryption; this is not proof of a cryptographic security level.",
        '', '**Why add noise?** Encryption does not protect an exact public aggregate. MEASURED known-neighbor accuracy without noise is 100%, including for the encrypted curious aggregator. Local noise remains after decryption and reduces information in public outputs.',
        '', '**What does combining them add under key theft?** Encryption hides contributions before key compromise; local noise limits what the thief obtains afterward. Gaussian key-thief accuracy is '+accuracy(g)+'; distributed BNP is '+accuracy(db)+'. Central-service BNP remains MEASURED 100% because its individual meter messages are unperturbed. An observer who sees only the released sum benefits from central BNP; the failed second layer concerns the stronger message-and-key observer.',
        '', '**What does protection cost?** MEASURED raw RMSE is '+', '.join(f"{c['utility'][m]['raw_rmse_kw']['mean']:.3f} kW ({m})" for m in MECHANISMS[1:])+'. Voltage RMSE is MEASURED '+', '.join(f"{c['utility'][m]['voltage_rmse_pu']['mean']:.6f} pu ({m})" for m in MECHANISMS[1:])+'. Heavy public clipping limits utility. The report separates paired utility-error passes from strict maximum-encoding diagnostic failures and provides repeated runtime intervals. Exact timing equality is not established.',
        '', '**Which contributions survive the prior-work review?** The supported deliverable is a reproducible threat-model and physics evaluation, including actual individual-ciphertext decryption, validated inversion and exact scalar bounded-noise calibration. Encryption plus DP and bounded distributed perturbations already appear in published work. A broad new-construction claim is not supported. The Severtson-Khajenejad citation is still unverified; see [PRIOR_WORK.md](PRIOR_WORK.md).',
        '', '[Full evidence, predictions, cost decisions and reproduction commands](results/threat_model_20261013/report.md).','']
    (ROOT/'ANSWERS_OCT13.md').write_text('\n'.join(answers),encoding='utf-8')
    print('Built reports and figures:',out)


def figures(out,c,k,v,t,cases,cell,vc):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#67778a','#2879ab','#dd9a35','#3b966d']
    labels=[case_name(b,m).replace('Distributed BNP','Distributed').replace('Gaussian DP','Gaussian') for b,m in cases]
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,7),sharey=True)
    for ax,a in zip(axs,ADVERSARIES):
        data=[cell(b,m,a)['measured'] for b,m in cases]
        means=np.array([x['accuracy'] for x in data])*100
        ci=np.array([x['ci95'] for x in data])*100
        ax.barh(np.arange(len(cases)),means,xerr=np.maximum(0,np.vstack([means-ci[:,0],ci[:,1]-means])),color=colors*3,capsize=2)
        ax.set_xlim(48,102);ax.set_title(a);ax.set_xlabel('MEASURED attack accuracy (%)')
        ax.axvline(50,color='gray',linestyle=':',linewidth=1)
    axs[0].set_yticks(np.arange(len(cases)),labels);axs[0].invert_yaxis()
    fig.suptitle('Full observer information: known-neighbor attack with confidence intervals')
    fig.tight_layout();save(fig,'accuracy_by_adversary')
    fig,ax=plt.subplots(figsize=(11,7))
    values=[vc(b,m)['recovered_vs_true_load_rmse_kw'] for b,m in cases]
    ax.barh(np.arange(len(cases)),values,color=colors*3)
    ax.set_yticks(np.arange(len(cases)),labels);ax.invert_yaxis();ax.set_xlabel('MEASURED recovered-load RMSE against original truth (kW)')
    ax.set_title('Voltage inversion recovers clipped released setpoints; one paired feeder trial')
    fig.tight_layout();save(fig,'voltage_load_recovery')
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,key,unit in zip(axs,['raw_rmse_kw','voltage_rmse_pu'],['Raw load RMSE (kW)','Voltage magnitude RMSE (pu)']):
        vals=[c['utility'][m][key] for m in MECHANISMS[1:]]
        ax.bar(['Gaussian','Central BNP','Distributed BNP'],[x['mean'] for x in vals],
            yerr=[x['ci95'][1]-x['mean'] for x in vals],color=colors[1:],capsize=4)
        ax.set_ylabel('MEASURED '+unit);ax.tick_params(axis='x',labelrotation=15)
    fig.suptitle('Matched scalar privacy; trial-level confidence intervals')
    fig.tight_layout();save(fig,'noise_utility')
    fig,ax=plt.subplots(figsize=(10,5))
    for b,marker in zip(BACKENDS,['o','s','^']):
        for m,color in zip(MECHANISMS,colors):
            tm=t['summary'][b+' / '+m]
            y=cell(b,m,'Key thief')['measured']['accuracy']*100
            ax.scatter(tm['mean'],y,color=color,marker=marker,s=65,label=b+' / '+m)
    ax.set_xscale('log');ax.set_xlabel('MEASURED local microbenchmark wall time (s, log scale)')
    ax.set_ylabel('MEASURED key-thief accuracy (%)');ax.set_ylim(49,102)
    ax.set_title('Computation cost versus known-neighbor protection')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    fig.tight_layout();save(fig,'cost_vs_protection')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/threat_model_20261013')
    main(ROOT/parser.parse_args().output)
