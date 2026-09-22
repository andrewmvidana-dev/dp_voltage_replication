"""Research simulation of Paillier aggregation with distributed Gaussian noise.

Adaptation of Shi et al., NDSS 2011: their per-user-key construction uses
geometric noise. Here we use 2048-bit Paillier and split Gaussian noise to
match this repository's model-fitting releases. This is not their protocol.

The collector has only a public key. A separate decryptor receives sums only.
Plain Paillier does NOT enforce aggregate-only decryption: a private-key holder
can decrypt any ciphertext it obtains. These in-process roles simulate honest,
non-colluding parties, not a cryptographic boundary or a production deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class SecureAggConfig:
    """Public simulation settings. Only load splits are seeded by default."""

    n_meters: int = 10
    split_seed: int = 0
    dirichlet_alpha: float = 1.0
    dropout: float = 0.0
    compensate_dropout: bool = False
    precision: float = 1e-12

    def __post_init__(self):
        if isinstance(self.n_meters, bool) or not isinstance(self.n_meters, int):
            raise ValueError("n_meters must be an integer")
        if self.n_meters < 2:
            raise ValueError("need at least two meters per bus")
        if not np.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive and finite")
        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not np.isfinite(self.precision) or not 0 < self.precision <= 1e-12:
            raise ValueError("precision must be positive and at most 1e-12")


def split_loads(loads, config=SecureAggConfig(), weights=None):
    """Split (buses, ...) pu loads into (buses, meters, ...) readings.

    Supply positive weights shaped (buses, meters), each row summing to one,
    or use reproducible Dirichlet weights. Weights remain fixed over time.
    Zero weights are excluded because fitting uses log(reading / weight).
    """
    loads = np.asarray(loads, dtype=float)
    if loads.ndim < 1 or loads.size == 0 or not np.isfinite(loads).all():
        raise ValueError("loads must be a nonempty finite array")
    shape = (len(loads), config.n_meters)
    if weights is None:
        weights = np.random.default_rng(config.split_seed).dirichlet(
            np.full(config.n_meters, config.dirichlet_alpha), size=len(loads))
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != shape or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-14, rtol=0)):
        raise ValueError("weights must be positive with shape (buses, meters) and row sums 1")
    expanded = weights.reshape(shape + (1,) * (loads.ndim - 1))
    return loads[:, None, ...] * expanded, weights.copy()


def participation(n, config, rng):
    """Drop floor(n * fraction) identities, independently of their readings.

    A committed roster is shared by both fitting rounds. Compensation uses
    the actual number of survivors, not an expected Bernoulli count.
    """
    k = n - int(np.floor(n * config.dropout))
    if k < 2:
        raise ValueError("fewer than two reporting meters; refuse this release")
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=k, replace=False)] = True
    return mask


def noise_share(sigma, n, k, compensate=False):
    """Return each survivor's SD and the effective variance of the sum.

    Without compensation: k * sigma**2 / n. With compensation: sigma**2.
    These formulas assume independent noise from every reporting meter.
    """
    sigma = np.asarray(sigma, dtype=float)
    if not 0 < k <= n or not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("need finite sigma >= 0 and 0 < reporters <= enrolled")
    sd = sigma / np.sqrt(k if compensate else n)
    return sd, k * sd**2


@dataclass
class CryptoTiming:
    """Wall time in seconds, with key generation separate from releases."""

    key_generation_s: float = 0.0
    encryption_s: float = 0.0
    addition_s: float = 0.0
    decryption_s: float = 0.0
    encrypted_values: int = 0
    decrypted_values: int = 0


class CiphertextCollector:
    """Sum encrypted vectors without a private key or plaintext readings."""

    def __init__(self, public_key):
        self.public_key = public_key

    def add(self, total, message):
        if any(x.public_key != self.public_key for x in message):
            raise ValueError("ciphertext belongs to a different key")
        if total is None:
            return list(message)
        if len(total) != len(message):
            raise ValueError("ciphertext vector shapes disagree")
        return [a + b for a, b in zip(total, message)]


class SumDecryptor:
    """Simulated separate key holder, called only on collector output.

    The Python interface is not an enforcement mechanism against a malicious
    collector. Deployment requires authenticated fixed rosters, one release
    per round, and a separate service or a suitable threshold/PSA protocol.
    """

    def __init__(self, private_key):
        self._private_key = private_key

    def decrypt_sum(self, ciphertexts):
        return np.array([self._private_key.decrypt(x) for x in ciphertexts])


class PaillierSimulation:
    """Test harness that hosts both roles and measures real cryptography.

    Do not give this harness to an untrusted utility. Only the collector
    belongs there; the decryptor must be operated separately.
    """

    def __init__(self, precision=1e-12):
        from phe import paillier

        if not np.isfinite(precision) or not 0 < precision <= 1e-12:
            raise ValueError("precision must be positive and at most 1e-12")
        self.precision = precision
        start = perf_counter()
        public, private = paillier.generate_paillier_keypair(n_length=2048)
        self.timing = CryptoTiming(key_generation_s=perf_counter() - start)
        self.collector = CiphertextCollector(public)
        self.decryptor = SumDecryptor(private)

    def sum(self, contributions, sigma, enrolled, reporters, rng):
        """Simulate local noise and encryption, then decrypt a vector sum.

        `sigma` is the per-meter SD, broadcastable to each contribution.
        The generator yields each meter's local statistic, never a public log.
        NumPy RNGs make research runs reproducible; they are not secure noise
        sources for deployment. Paillier always uses its own secure randomness.
        """
        if not 2 <= reporters <= enrolled:
            raise ValueError("need at least two reporters")
        sigma = np.asarray(sigma, dtype=float)
        if not np.isfinite(sigma).all() or np.any(sigma < 0):
            raise ValueError("sigma must be finite and nonnegative")
        total, count, shape = None, 0, None
        for contribution in contributions:
            # This block runs at a simulated meter, with only the public key.
            contribution = np.asarray(contribution, dtype=float)
            if contribution.size == 0 or not np.isfinite(contribution).all():
                raise ValueError("meter contribution must be nonempty and finite")
            if shape is not None and shape != contribution.shape:
                raise ValueError("meter contribution shapes disagree")
            shape = contribution.shape
            noisy = contribution + rng.normal(0.0, sigma, size=shape)
            start = perf_counter()
            message = [self.collector.public_key.encrypt(float(x), precision=self.precision)
                       for x in noisy.flat]
            self.timing.encryption_s += perf_counter() - start
            self.timing.encrypted_values += len(message)

            # Only ciphertexts cross into the collector role.
            start = perf_counter()
            total = self.collector.add(total, message)
            self.timing.addition_s += perf_counter() - start
            count += 1
        if count != reporters:
            raise ValueError("committed roster does not match received messages")

        # Only the completed sum crosses into the decryptor role.
        start = perf_counter()
        result = self.decryptor.decrypt_sum(total).reshape(shape)
        self.timing.decryption_s += perf_counter() - start
        self.timing.decrypted_values += result.size
        return result


def aggregate_bus_loads(loads, sigma, rng, config=SecureAggConfig(),
                        weights=None, session=None):
    """Primitive for bus-sum validation and per-timestep feeder benchmarking.

    This is NOT the model-fitting DP mechanism. Sigma here is in pu, whereas
    fitting scales describe log means and covariance entries. Missing loads
    remain missing: no rescaling or imputation hides dropout bias.
    """
    readings, _ = split_loads(loads, config, weights)
    session = session or PaillierSimulation(config.precision)
    outputs, reports = [], []
    for bus in readings:
        mask = participation(config.n_meters, config, rng)
        k = int(mask.sum())
        sd, variance = noise_share(sigma, config.n_meters, k, config.compensate_dropout)
        outputs.append(session.sum(iter(bus[mask]), sd, config.n_meters, k, rng))
        reports.append({"enrolled": config.n_meters, "reporters": k,
                        "effective_variance": np.asarray(variance).tolist()})
    return np.array(outputs), reports, session.timing


@dataclass
class SimulatedMeter:
    """One customer's proportional load trajectory, kept on the meter side.

    Normalizing by the assigned public share recovers the bus-level record.
    This is specific to fixed proportional simulated customers. For real
    independent customer profiles, log(sum(loads)) is not additive and this
    construction does not reproduce the bus-level model.
    """

    readings: np.ndarray = field(repr=False)
    weight: float

    def logs(self):
        return np.log(self.readings / self.weight)

    def mean_contribution(self, n_records):
        return self.weight * self.logs().sum(axis=0) / n_records

    def covariance_contribution(self, mean, clip_norm, n_records):
        centred = self.logs() - mean
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        centred *= np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
        cov = self.weight * (centred.T @ centred) / n_records
        return cov[np.triu_indices(len(mean))]


def fit_class(meters, n_records, T, lo, hi, epsilon, delta, rng,
              config=SecureAggConfig(), session=None, eig_floor_ratio=1e-3,
              clip_norm=None, calibration="analytic", rho_split=0.5):
    """Two encrypted releases: class log mean, then clipped covariance.

    Shares sum to one for each bus, so without dropout the plaintext statistics
    equal dp_fit_class's statistics. No clean mean, covariance, clipping count,
    or KL diagnostic is released. A private trace replaces its nonprivate
    eigenvalue floor; the original trusted path is intentionally unchanged.
    """
    from dpvolt.privacy import gaussian_fit_scales

    if not meters or n_records <= 0 or T < 2 or not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("need meters, records, T >= 2 and finite lo < hi")
    if not np.isfinite(eig_floor_ratio) or eig_floor_ratio <= 0:
        raise ValueError("eig_floor_ratio must be positive and finite")
    C = np.sqrt(T) * (hi - lo) / 2 if clip_norm is None else float(clip_norm)
    if not np.isfinite(C) or C <= 0:
        raise ValueError("clip_norm must be positive and finite")
    sigma_mu, sigma_cov, _ = gaussian_fit_scales(
        n_records, T, hi - lo, epsilon, delta, C, calibration, rho_split)
    n = len(meters)
    mask = participation(n, config, rng)
    active = [meter for meter, present in zip(meters, mask) if present]
    k = len(active)
    sd_mu, var_mu = noise_share(sigma_mu, n, k, config.compensate_dropout)
    sd_cov, var_cov = noise_share(sigma_cov, n, k, config.compensate_dropout)
    session = session or PaillierSimulation(config.precision)

    # Round 1: weighted local means, each with an independent noise share.
    mean = session.sum((meter.mean_contribution(n_records) for meter in active),
                       sd_mu, n, k, rng)

    # Round 2: broadcast the noisy mean, then clip records locally. Sending
    # only the upper triangle halves the work while preserving the exact
    # distribution of the original symmetrized Gaussian covariance noise.
    upper = np.triu_indices(T)
    scales = np.full(len(upper[0]), sd_cov)
    scales[upper[0] == upper[1]] *= np.sqrt(2.0)
    packed = session.sum(
        (meter.covariance_contribution(mean, C, n_records) for meter in active),
        scales, n, k, rng)
    cov = np.zeros((T, T))
    cov[upper] = packed
    cov[(upper[1], upper[0])] = packed
    cov += 1e-12 * np.eye(T)

    # All repair inputs are released values, so repair is post-processing.
    floor = eig_floor_ratio * max(float(np.trace(cov)) / T, 1e-12)
    evals, evecs = np.linalg.eigh(cov)
    clipped = int((evals < floor).sum())
    cov = (evecs * np.maximum(evals, floor)) @ evecs.T
    report = {
        "enrolled": n, "reporters": k, "dropout_realized": 1 - k / n,
        "compensated": config.compensate_dropout,
        "sigma_mu": sigma_mu, "sigma_cov": sigma_cov,
        "effective_mean_variance": float(var_mu),
        "effective_covariance_offdiag_variance": float(var_cov),
        "effective_covariance_diagonal_variance": float(2 * var_cov),
        "nominal_noise_preserved": bool(k == n or config.compensate_dropout),
        "eig_clipped": clipped, "calibration": calibration,
        "n_records": n_records,
    }
    return mean, cov, report
