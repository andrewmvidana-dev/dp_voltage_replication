"""Opt-in, synthetic key-compromise experiments for direct meter sums.

Each backend/mechanism gets a real individual-ciphertext demonstration.
Monte Carlo attack trials then use its decoded-message mathematical model;
they are not repeated cryptographic security trials. Public observers receive
the raw noisy bus sum and voltages derived from the clipped sum. Noise seeds
and evaluator-only originals are excluded from every adversary view.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import norm

from dpvolt.privacy import bnp_distributed_half_width, bnp_distributed_delta
from dpvolt.secure_agg import CiphertextCollector, PaillierSimulation
from dpvolt.threshold_agg import ThresholdBGVSimulation


MECHANISMS = ('None', 'Gaussian DP', 'BNP', 'Distributed BNP')
BACKENDS = ('None', 'Paillier', 'Threshold BGV')
ADVERSARIES = ('Public analyst', 'Curious aggregator', 'Key thief')
ROOT = Path(__file__).resolve().parent


def parameters(sensitivity_kw=20.0, delta=0.01, meters=10):
    """Calibrate independent alternatives with the same scalar privacy budget."""
    return dict(sensitivity_kw=float(sensitivity_kw), delta=float(delta),
                meters=int(meters), gaussian_sd_kw=float(
                    sensitivity_kw / (2 * norm.ppf((1 + delta) / 2))),
                central_half_width_kw=float(sensitivity_kw / (2 * delta)),
                distributed_half_width_kw=float(bnp_distributed_half_width(
                    sensitivity_kw, delta, meters)))


def prepare_contributions(readings, mechanism, rng, config):
    """Return meter messages followed by one independent noise-service slot.

    Input shape is (trials, meters). The service sends zero for mechanisms
    that have no service noise, so message counts remain comparable.
    """
    prepared = np.asarray(readings, dtype=float).copy()
    service = np.zeros((len(prepared), 1))
    if mechanism == 'Gaussian DP':
        prepared += rng.normal(0, config['gaussian_sd_kw'] / np.sqrt(config['meters']),
                               prepared.shape)
    elif mechanism == 'BNP':
        bound = config['central_half_width_kw']
        service[:, 0] = rng.uniform(-bound, bound, len(prepared))
    elif mechanism == 'Distributed BNP':
        bound = config['distributed_half_width_kw']
        prepared += rng.uniform(-bound, bound, prepared.shape)
    elif mechanism != 'None':
        raise ValueError('unknown mechanism: ' + mechanism)
    return np.concatenate((prepared, service), axis=1)


def wilson(correct, trials):
    """A binomial confidence interval, including all-correct attack cells."""
    z = float(norm.ppf(0.975))
    p = correct / trials
    divisor = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / divisor
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / divisor
    return [0.0 if correct == 0 else float(max(0, centre - half)),
            1.0 if correct == trials else float(min(1, centre + half))]


def analytic_accuracy(mechanism, channel, config):
    """Known-neighbor optimal success under the ideal continuous mechanisms."""
    sensitivity = config['sensitivity_kw']
    if mechanism == 'None' or (channel == 'individual' and mechanism == 'BNP'):
        return 1.0
    if channel == 'individual':
        if mechanism == 'Gaussian DP':
            sd = config['gaussian_sd_kw'] / np.sqrt(config['meters'])
            return float(norm.cdf(sensitivity / (2 * sd)))
        return float((1 + min(1, sensitivity / (
            2 * config['distributed_half_width_kw']))) / 2)
    if mechanism == 'Gaussian DP':
        return float(norm.cdf(sensitivity / (2 * config['gaussian_sd_kw'])))
    if mechanism == 'BNP':
        return float((1 + min(1, sensitivity / (2 * config['central_half_width_kw']))) / 2)
    return float((1 + bnp_distributed_delta(sensitivity,
        config['distributed_half_width_kw'], config['meters'])) / 2)


@dataclass(frozen=True)
class AttackView:
    """Only information that this observer is allowed to see."""
    public_raw_sum: np.ndarray
    public_clipped_sum: np.ndarray
    individual_messages: np.ndarray | None = None


def observer_view(backend, adversary, messages, raw_sum):
    """Project evaluator state onto the specified information boundary.

    In statistical trials, messages are the ideal decoded-message model.
    Real cryptographic decoding is verified separately. An encrypted no-key
    observer never receives these decoded messages or a decryption object.
    """
    visible = messages if (adversary == 'Key thief' or
                           (backend == 'None' and adversary == 'Curious aggregator')) else None
    return AttackView(raw_sum, np.clip(raw_sum, 0, 200), visible)


def prediction(backend, mechanism, adversary, channel):
    """Preserve the prompt's hypotheses, including its incorrect no-key row."""
    if channel == 'full_view':
        if backend != 'None' and adversary == 'Curious aggregator':
            return 0.5
        if adversary == 'Key thief' or (backend == 'None' and adversary == 'Curious aggregator'):
            return {'None': 1.0, 'Gaussian DP': 0.516, 'BNP': 1.0,
                    'Distributed BNP': 0.512}[mechanism]
    return 1.0 if mechanism == 'None' else 0.505


def run_attack_trials(trials, seed, config):
    """Test two public neighboring hypotheses using fixed, optimal thresholds.

    Only the target changes, from zero to S kW. All other meters equal S/2.
    Thus the bus sums differ by S as well. Meter and bus identification are
    separate targets with the same binary hypothesis label. This is not an
    arbitrary-reading reconstruction claim from a bus total.
    """
    if trials < 20000:
        raise ValueError('need at least 20000 trials per cell')
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, trials)
    sensitivity, meters = config['sensitivity_kw'], config['meters']
    known_other_sum = (meters - 1) * sensitivity / 2
    readings = np.full((trials, meters), sensitivity / 2)
    readings[:, 0] = labels * sensitivity
    truth_sum = readings.sum(axis=1)
    cells, reconstruction = [], []
    for index, mechanism in enumerate(MECHANISMS):
        messages = prepare_contributions(readings, mechanism,
                                         np.random.default_rng(seed + index + 1), config)
        raw_sum = messages.sum(axis=1)
        for backend in BACKENDS:
            for adversary in ADVERSARIES:
                view = observer_view(backend, adversary, messages, raw_sum)
                # Each channel classifier receives only its visible observations.
                observations = {'raw_sum': (view.public_raw_sum, known_other_sum + sensitivity / 2),
                                'clipped_sum': (view.public_clipped_sum, known_other_sum + sensitivity / 2)}
                if view.individual_messages is None:
                    observations['full_view'] = observations['raw_sum']
                    local_channel = 'sum'
                else:
                    observations['full_view'] = (view.individual_messages[:, 0], sensitivity / 2)
                    local_channel = 'individual'
                    reconstruction.append(dict(backend=backend, mechanism=mechanism,
                        adversary=adversary, label='MEASURED', trials=trials,
                        meter_direct_rmse_kw=float(np.sqrt(np.mean(
                            (view.individual_messages[:, 0] - readings[:, 0])**2))),
                        decoded_meter_sum_rmse_kw=float(np.sqrt(np.mean(
                            (view.individual_messages[:, :meters].sum(axis=1) - truth_sum)**2))),
                        note='Meter sum excludes the observed noise-service message; it is not the optimal known-neighbor classifier.'))
                for channel, (observed, threshold) in observations.items():
                    estimated_label = observed > threshold
                    correct = int(np.count_nonzero(estimated_label == labels))
                    ci = wilson(correct, trials)
                    analytic = analytic_accuracy(mechanism,
                        local_channel if channel == 'full_view' else 'sum', config)
                    predicted = prediction(backend, mechanism, adversary, channel)
                    for target in ('target_meter', 'neighboring_bus_sum'):
                        cells.append(dict(backend=backend, mechanism=mechanism,
                            adversary=adversary, target=target, channel=channel,
                            measured=dict(label='MEASURED', accuracy=correct / trials,
                                ci95=ci, correct=correct, trials=trials,
                                method='Independent ideal-noise trials, paired across backend/adversary views'),
                            analytic=dict(label='ANALYTIC', accuracy=analytic),
                            predicted=dict(label='PREDICTED', accuracy=predicted,
                                source='Adopted task prompt; rounded values retained'),
                            prediction_minus_analytic=predicted - analytic,
                            analytic_inside_ci=bool(ci[0] <= analytic <= ci[1]),
                            threshold_kw=float(threshold),
                            observation=('Target meter message and public release' if
                                channel == 'full_view' and local_channel == 'individual'
                                else 'Public ' + ('raw bus sum' if channel != 'clipped_sum' else 'clipped bus sum')),
                            caveat='Two known neighbors, not general individual reconstruction.'))
    return cells, reconstruction


class CapturingCollector(CiphertextCollector):
    """Copy real meter ciphertext references without changing addition."""
    def __init__(self, public_key):
        super().__init__(public_key)
        self.messages = []

    def add(self, total, message):
        self.messages.append(tuple(message))
        return super().add(total, message)


def paillier_demo(contributions):
    """Use the stolen private-key role to decrypt each captured ciphertext."""
    start = perf_counter()
    session = PaillierSimulation()
    capture = CapturingCollector(session.collector.public_key)
    session.collector = capture
    aggregate = session.sum(contributions, 0, len(contributions), len(contributions),
                            np.random.default_rng(0))
    recovered = np.array([session.decryptor.decrypt_sum(message) for message in capture.messages])
    wall = perf_counter() - start
    fixed_bytes = (capture.public_key.nsquare.bit_length() + 7) // 8
    fingerprints = [hashlib.sha256(b''.join(
        value.ciphertext().to_bytes(fixed_bytes, 'big') for value in message)).hexdigest()
        for message in capture.messages]
    # A collector has no decryption API. This is a capability check only.
    no_key_check = not hasattr(capture, 'decrypt') and not hasattr(capture, '_private_key')
    # phe rejects a private key paired with a different public modulus.
    other = PaillierSimulation()
    try:
        other.decryptor.decrypt_sum(capture.messages[0])
        wrong_key = 'unexpectedly accepted'
    except ValueError:
        wrong_key = 'refused mismatched public/private key'
    if wrong_key == 'unexpectedly accepted' or not no_key_check:
        raise AssertionError('Paillier capability check failed')
    return aggregate, recovered, dict(label='MEASURED', wall_s=wall,
        timing=asdict(session.timing), ciphertext_messages=len(capture.messages),
        ciphertext_sha256=fingerprints,
        ciphertext_bytes_fixed_integer_estimate=sum(map(len, capture.messages)) * fixed_bytes,
        size_scope='Ciphertext integers only, excluding exponent metadata and setup traffic',
        no_key_capability_check=no_key_check, wrong_key_capability_check=wrong_key,
        security_scope='Interface/key-consistency checks do not measure ciphertext security.')


def bgv_demo(contributions):
    """A compromised committed quorum decrypts originals, including the service."""
    session = ThresholdBGVSimulation()
    result = session.demonstrate_key_compromise(contributions)
    aggregate, recovered = result.pop('values'), result.pop('recovered_contributions')
    # Exercise the real subprocess refusal, rather than only Python validation.
    invalid = dict(contributions=np.asarray(contributions).tolist(), authorities=5,
                   threshold=3, active=[1, 2], scale=1e6, workers=1,
                   compromise_individuals=True)
    import subprocess
    refused = subprocess.run([str(session.executable)], input=json.dumps(invalid),
        text=True, capture_output=True, timeout=60, check=False)
    if refused.returncode == 0:
        raise AssertionError('insufficient-quorum request was accepted')
    result.update(label='MEASURED', insufficient_quorum_capability_check=refused.stderr.strip(),
                  security_scope='Quorum refusal checks are not a cryptographic security proof; wrong-key decoding is not tested as a security metric.')
    return aggregate, recovered, result


def classify_real_demo(backend, aggregate_kw, recovered_kw, config):
    """Run each observer's classifier on the actual decoded demonstration.

    This single demonstration used the upper public neighbor. Only the
    evaluator knows that label. Monte Carlo intervals are reported separately.
    """
    rows = []
    aggregate_kw = np.asarray(aggregate_kw).reshape(-1)
    messages = np.asarray(recovered_kw).reshape(config['meters'] + 1, -1).T
    midpoint = config['meters'] * config['sensitivity_kw'] / 2
    for adversary in ADVERSARIES:
        view = observer_view(backend, adversary, messages, aggregate_kw)
        observations = {'raw_sum': (view.public_raw_sum, midpoint),
                        'clipped_sum': (view.public_clipped_sum, midpoint)}
        observations['full_view'] = ((view.individual_messages[:, 0], config['sensitivity_kw'] / 2)
            if view.individual_messages is not None else observations['raw_sum'])
        for channel, (visible, threshold) in observations.items():
            guessed = bool(visible[0] > threshold)
            for target in ('target_meter', 'neighboring_bus_sum'):
                rows.append(dict(label='MEASURED', adversary=adversary,
                    channel=channel, target=target, guessed_upper_neighbor=guessed,
                    correct=guessed, trials=1,
                    note='Actual plaintext/cryptographic demonstration; no confidence claim from this single trial.'))
    return rows


def run_crypto_demonstrations(config, seed, output):
    """Real encryption for every mechanism, with errors and failures retained."""
    demos = {backend: {} for backend in BACKENDS}
    readings = np.full((1, config['meters']), config['sensitivity_kw'] / 2)
    readings[0, 0] = config['sensitivity_kw']
    for index, mechanism in enumerate(MECHANISMS):
        prepared = prepare_contributions(readings, mechanism,
                        np.random.default_rng(seed + index), config).T
        # Match the existing feeder encoding: inputs are pu on a 1000 kW base.
        prepared_pu = prepared / 1000
        for backend in BACKENDS:
            print('Individual-message demonstration:', backend, mechanism, flush=True)
            try:
                if backend == 'None':
                    aggregate, recovered = prepared_pu.sum(axis=0), prepared_pu.copy()
                    metadata = dict(label='MEASURED', cryptography='not applicable; messages are plaintext')
                elif backend == 'Paillier':
                    aggregate, recovered, metadata = paillier_demo(prepared_pu)
                else:
                    aggregate, recovered, metadata = bgv_demo(prepared_pu)
                individual_error = float(np.max(np.abs(recovered - prepared_pu)) * 1000)
                aggregate_error = float(np.max(np.abs(aggregate - prepared_pu.sum(axis=0))) * 1000)
                tolerance = len(prepared_pu) / (2 * 1e6) * 1000 if backend == 'Threshold BGV' else 1e-5
                if individual_error > tolerance or aggregate_error > tolerance:
                    raise AssertionError('actual ciphertext decoding exceeded the encoding bound')
                metadata.update(status='complete', recovered_individual_kw=(recovered * 1000).tolist(),
                    prepared_individual_kw=prepared.tolist(), decoded_sum_kw=(aggregate * 1000).tolist(),
                    service_index=config['meters'], max_individual_error_kw=individual_error,
                    max_sum_error_kw=aggregate_error,
                    evaluator_only_original_meter_kw=readings[0].tolist(),
                    privacy_scope='Original readings and seeds are evaluator-only. The stolen key reveals prepared messages, not local noise draws separately.')
                metadata['adversary_attacks'] = classify_real_demo(
                    backend, aggregate * 1000, recovered * 1000, config)
                filename = 'demo_' + backend.replace(' ', '_') + '_' + mechanism.replace(' ', '_') + '.npz'
                np.savez_compressed(output / filename, prepared_kw=prepared,
                    recovered_kw=recovered * 1000, aggregate_kw=aggregate * 1000)
                metadata['artifact'] = filename
                demos[backend][mechanism] = metadata
            except (ImportError, FileNotFoundError, RuntimeError, ValueError, AssertionError) as exc:
                demos[backend][mechanism] = dict(status='failed', label='MEASURED', error=str(exc))
    return demos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--output', default='results/threat_model_20261013')
    args = parser.parse_args()
    if args.trials < 20000:
        parser.error('at least 20000 trials are required per attack cell')
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = parameters()
    cells, reconstruction = run_attack_trials(args.trials, args.seed, config)
    demos = run_crypto_demonstrations(config, args.seed + 100, output)
    result = dict(settings=dict(label='MEASURED', calibration_label='ANALYTIC', **config, trials=args.trials, seed=args.seed,
        privacy_unit='One meter reading at one time', prior='Independent fair Bernoulli hypotheses',
        neighbors='Target reading is 0 or 20 kW; nine other readings are fixed at 10 kW',
        release='Public raw bus sum; voltage derives from the same sum clipped to [0,200] kW',
        seed_scope='Evaluation reproducibility only; seeds and noise draws are excluded from adversary views',
        finite_precision='ANALYTIC values are ideal continuous-noise guarantees. Numeric Monte Carlo is not a finite-precision DP certification.'),
        formulas=dict(label='ANALYTIC', accuracy='(1 + TV) / 2',
            gaussian='TV = 2 Phi(S/(2 sigma)) - 1; local sigma = aggregate sigma/sqrt(n)',
            uniform='TV = min(1, S/(2 B))',
            distributed='TV = P(abs(sum_i Uniform(-b,b)) < S/2), using exact Irwin-Hall CDF',
            clipped='Midpoint thresholds 100 kW lie inside [0,200], so clipping preserves these specific binary decisions.'),
        cells=cells, transcript_reconstruction=reconstruction, crypto_demonstrations=demos,
        voltage_channel=dict(status='pending_inversion_validation', label='PROPOSED',
                             note='Voltage-only recovery and attack cells are supplied by run_voltage_set_attack.py after validation.'),
        limitations=['Paillier and BGV roles run on one local machine.',
            'Known-neighbor discrimination is not arbitrary-load reconstruction.',
            'Trials share draws across backends and adversaries; cells are not independent.',
            'Observed ciphertext security is not tested or certified.',
            'A compromised Gaussian meter RNG is outside this key-only compromise scenario.',
            'The public-release assumption corrects the prompt: an encrypted curious aggregator can exploit that release.'],
        complete=all(value['status'] == 'complete' for backend in demos.values() for value in backend.values()))
    path = output / 'key_compromise.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Saved', path, flush=True)
    if not result['complete']:
        raise SystemExit('Some cryptographic demonstrations failed; see labeled errors in key_compromise.json')


if __name__ == '__main__':
    main()
