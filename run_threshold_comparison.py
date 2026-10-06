"""Nine matched direct-sum conditions plus real encrypted fitting and dropout.

First trial of all encrypted conditions executes real cryptography. Remaining
noise trials evaluate the identical plaintext mechanism, avoiding 30 expensive
Paillier reruns. No timing confidence intervals are inferred from one run.
"""
import json
from pathlib import Path
from dataclasses import asdict
from time import perf_counter
import platform
import numpy as np
from scipy.stats import norm
import opendssdirect as dss
from dpvolt.threshold_agg import ThresholdBGVSimulation, ThresholdConfig
from dpvolt.secure_agg import PaillierSimulation, SecureAggConfig
from dpvolt.loads import reactive_from_active, assign_classes, make_historical, fit_load_model
from dpvolt.powerflow import PowerFlowRunner
from dpvolt.experiments import ModelFitConfig, fit_private_load_model
from run_privacy_comparison import summary

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/threshold_20261005'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(ROOT / 'results/privacy_comparison_20260924/synthetic_inputs.npz')
    meters, truth, vtruth = saved['bounded_meters'], saved['reference_sum'], saved['reference_voltages']
    runner = PowerFlowRunner(str(ROOT / 'feeders/IEEE123Master.dss'))
    theta, kw = [], []
    for name in runner.load_names:
        dss.Loads.Name(name)
        theta.append(np.arccos(np.clip(dss.Loads.PF(), -1, 1))); kw.append(dss.Loads.kW())
    theta = np.asarray(theta)
    retained = runner.retained_indices()
    delta, sensitivity = .01, .02
    sigma = sensitivity / (2 * norm.ppf((1 + delta) / 2))
    bound = sensitivity / (2 * delta)
    result = dict(settings=dict(trials=30,epsilon=0,delta=delta,sensitivity_kw=20,
        privacy_unit='One reading of one meter at one time; not a full customer trajectory',
        load_rows=91,meters_per_row=10,steps=6,gaussian_sd_kw=sigma*1000,
        uniform_half_width_kw=bound*1000,noise_seed=61005,scale=1e6,
        authorities=5,threshold=3,platform=platform.platform()), utility={}, crypto={}, trials=[],
        sources=['https://doi.org/10.1007/s00145-023-09452-8',
                 'https://csrc.nist.gov/pubs/sp/800/226/final',
                 'https://github.com/tuneinsight/lattigo/security'])
    def save():
        (OUT/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    # Each bus release has 10 meter contributors plus one independent noise
    # service. Zero service input in no-noise/Gaussian arms keeps cost matched.
    for mechanism in ['None', 'Gaussian DP', 'BNP']:
        outputs = []
        for trial in range(30):
            rng = np.random.default_rng(61005 + trial)
            prepared = meters.copy()
            extra = np.zeros_like(truth)
            if mechanism == 'Gaussian DP':
                prepared += rng.normal(0, sigma/np.sqrt(10), prepared.shape)
            if mechanism == 'BNP':
                extra = rng.uniform(-bound, bound, truth.shape)
            prepared = np.concatenate([prepared, extra[:,None,:]], axis=1)
            clean = prepared.sum(axis=1)
            outputs.append(clean)
            if trial == 0:
                for backend in ['Paillier', 'Threshold BGV']:
                    print('Actual full-feeder encryption:',backend,mechanism,flush=True)
                    s = PaillierSimulation() if backend=='Paillier' else ThresholdBGVSimulation()
                    start=perf_counter()
                    encrypted=np.array([s.sum(iter(bus),0,11,11,np.random.default_rng(0)) for bus in prepared])
                    wall=perf_counter()-start
                    error=float(np.max(np.abs(encrypted-clean))*1000)
                    assert error < (.006 if backend=='Threshold BGV' else 1e-5)
                    result['crypto'][backend+' / '+mechanism] = dict(wall_s=wall,max_error_kw=error,
                        timing=asdict(s.timing),releases=getattr(s,'releases',[]))
                    np.savez_compressed(OUT/(backend.replace(' ','_')+'_'+mechanism.replace(' ','_')+'.npz'),release=encrypted)
                    save()
            clipped=np.clip(clean,0,.2)
            runner.reset(); volts,ok=runner.solve_trajectory(clipped,reactive_from_active(clipped,theta))
            assert ok.all()
            row=dict(mechanism=mechanism,trial=trial,
                raw_rmse_kw=float(np.sqrt(np.mean((clean-truth)**2))*1000),
                clipped_rmse_kw=float(np.sqrt(np.mean((clipped-truth)**2))*1000),
                clipped_fraction=float(np.mean(clean!=clipped)),
                voltage_rmse_pu=float(np.sqrt(np.mean((np.abs(volts[:,retained])-np.abs(vtruth[:,retained]))**2))),
                convergence=float(ok.mean()))
            result['trials'].append(row)
        rows=[x for x in result['trials'] if x['mechanism']==mechanism]
        result['utility'][mechanism]={k:summary([r[k] for r in rows]) for k in rows[0] if k not in ['mechanism','trial']}
        save()
    # Paired seed utility differences: both mechanisms have the same budget.
    gs=[r for r in result['trials'] if r['mechanism']=='Gaussian DP']
    bs=[r for r in result['trials'] if r['mechanism']=='BNP']
    result['paired_bnp_minus_gaussian']={k:summary([b[k]-g[k] for b,g in zip(bs,gs)]) for k in ['raw_rmse_kw','voltage_rmse_pu']}
    # A pure public-key operation microbenchmark: same 10x96 vector in both.
    data=np.random.default_rng(7).normal(size=(10,96))*.01
    result['batching']={}
    for label,workers in [('Paillier',1),('BGV serial',1),('BGV 4 workers',4)]:
        print('Packing benchmark:',label,flush=True)
        times=[];payloads=[]
        for rep in range(3):
            s=PaillierSimulation() if label=='Paillier' else ThresholdBGVSimulation(ThresholdConfig(workers=workers))
            start=perf_counter();out=s.sum(data,0,10,10,np.random.default_rng(1));times.append(perf_counter()-start)
            assert np.max(np.abs(out-data.sum(0)))<6e-6
            if hasattr(s,'releases'): payloads.append(s.releases[-1])
        result['batching'][label]=dict(wall_s=times,mean_s=float(np.mean(times)),releases=payloads)
        save()
    # Real full feeder model fitting (two rounds per class) with new backend.
    classes=assign_classes(np.array(kw));archive=make_historical(np.array(kw),classes,45,T=4,rng=np.random.default_rng(30))
    model=fit_load_model(archive,classes,theta)
    # Declare public synthetic bounds, rather than reuse empirical min/max.
    for c in classes: model.p_min[c]=.0001;model.p_max[c]=1.
    archive=np.clip(archive,.0001,1.)
    cfg=ModelFitConfig(mode='secure_aggregation',secure=SecureAggConfig(n_meters=10),encryption_backend='threshold_bgv')
    session=ThresholdBGVSimulation();start=perf_counter()
    fitted=fit_private_load_model(archive,classes,model,theta,50,1e-5,np.random.default_rng(42),config=cfg,session=session)
    wall=perf_counter()-start
    class Plain:
        timing=None
        def sum(self,contributions,sigma,enrolled,reporters,rng):
            return sum(x+rng.normal(0,sigma,size=x.shape) for x in contributions)
    reference=fit_private_load_model(archive,classes,model,theta,50,1e-5,np.random.default_rng(42),config=cfg,session=Plain())
    result['model_fit']=dict(wall_s=wall,timing=asdict(session.timing),releases=session.releases,
        max_mean_error=max(float(np.max(np.abs(fitted.mu[c]-reference.mu[c]))) for c in classes),
        max_covariance_error=max(float(np.max(np.abs(fitted.Sigma[c]-reference.Sigma[c]))) for c in classes),
        scope='91 rows, 45 synthetic days, T=4, 3 classes, 910 meters; epsilon=50, delta=1e-5. Integration smoke test, not same workload as old 360-day timing.')
    # Meter loss is separate from authority loss; missing signal remains absent.
    result['dropout']=[]
    for lost in [0,1,2,3,4,5,6,7,8]:
        remaining=10-lost
        signal=meters[:,:remaining].sum(1)
        effective_sigma=sigma*np.sqrt(remaining/10)
        result['dropout'].append(dict(lost_percent=10*lost,remaining=remaining,
            signal_rmse_kw=float(np.sqrt(np.mean((signal-truth)**2))*1000),
            variance_fraction=remaining/10,uncompensated_delta=float(2*norm.cdf(sensitivity/(2*effective_sigma))-1),
            compensated_delta=delta))
    result['delta_sweep']=[dict(delta=d,bnp_rmse_kw=sensitivity/(2*d*np.sqrt(3))*1000,
        gaussian_rmse_kw=sensitivity/(2*norm.ppf((1+d)/2))*1000) for d in [1e-5,1e-4,1e-3,.01,.05,.1]]
    result['complete']=True;save()
    print('Finished',OUT,flush=True)


if __name__=='__main__': main()
