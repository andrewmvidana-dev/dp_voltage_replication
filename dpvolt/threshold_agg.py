"""Packed BGV threshold research backend, compatible with secure_agg.fit_class.

Actual Lattigo cryptography, simulated colocated roles. Each sum starts a fresh
process/key epoch. Privacy noise is sampled before fixed-point encryption.
No arbitrary ciphertext/decryption endpoint or same-key retry is exposed.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from dpvolt.secure_agg import CryptoTiming


@dataclass(frozen=True)
class ThresholdConfig:
    authorities: int = 5
    threshold: int = 3
    active: tuple[int, ...] = (1, 2, 3)
    scale: float = 1e6
    workers: int = 1

    def __post_init__(self):
        if not (2 <= self.threshold <= self.authorities <= 16):
            raise ValueError('need 2 <= threshold <= authorities <= 16')
        if (len(self.active) != self.threshold or len(set(self.active)) != self.threshold
                or any(type(i) is not int or not 1 <= i <= self.authorities for i in self.active)):
            raise ValueError('commit exactly threshold distinct authority IDs')
        if not np.isfinite(self.scale) or self.scale <= 0 or not 1 <= self.workers <= 32:
            raise ValueError('invalid scale or workers')


class ThresholdBGVSimulation:
    """Do not expose this synthetic-data harness to real customer readings.

    The subprocess receives all plaintexts. Separating actual meter/authority
    services, authenticated channels and robust malicious-party security remain
    deployment work. The CSPRNG inside Lattigo is independent of NumPy noise.
    """
    def __init__(self, config=None, executable=None):
        self.config = config or ThresholdConfig()
        self.executable = Path(executable or Path(__file__).resolve().parents[1]
                               / 'threshold_mhe' / 'threshold_mhe.exe')
        if not self.executable.is_file():
            raise FileNotFoundError('Build threshold_mhe first; see THRESHOLD_ENCRYPTION.md')
        self.timing = CryptoTiming()
        self.releases = []

    def sum(self, contributions, sigma, enrolled, reporters, rng):
        if not 2 <= reporters <= enrolled:
            raise ValueError('need at least two reporters')
        sigma = np.asarray(sigma, dtype=float)
        if not np.isfinite(sigma).all() or np.any(sigma < 0):
            raise ValueError('sigma must be finite and nonnegative')
        inputs = [np.asarray(x, dtype=float) for x in contributions]
        if (len(inputs) != reporters or not inputs or inputs[0].size == 0
                or any(x.shape != inputs[0].shape or not np.isfinite(x).all() for x in inputs)):
            raise ValueError('invalid contributions or committed roster')
        noisy = [x + rng.normal(0, sigma, size=x.shape) for x in inputs]
        return self.sum_prepared(noisy)

    def sum_prepared(self, contributions):
        """Encrypt already prepared contributions (including optional noise).

        No privacy guarantee is attached to this low-level primitive. An
        independent trusted uniform-noise contributor is used by the BNP
        experiment, explicitly not a dealer-free bounded-noise protocol.
        """
        inputs = [np.asarray(x, dtype=float) for x in contributions]
        if len(inputs) < 2 or not inputs[0].size or any(
                x.shape != inputs[0].shape or not np.isfinite(x).all() for x in inputs):
            raise ValueError('need finite contributions with identical shape')
        cfg = self.config
        payload = dict(contributions=[x.ravel().tolist() for x in inputs],
                       authorities=cfg.authorities, threshold=cfg.threshold,
                       active=cfg.active, scale=cfg.scale, workers=cfg.workers)
        start = perf_counter()
        process = subprocess.run([str(self.executable)], input=json.dumps(payload),
                                 text=True, capture_output=True, timeout=600, check=False)
        wall = perf_counter() - start
        if process.returncode:
            raise RuntimeError('Threshold release refused: ' + process.stderr.strip())
        result = json.loads(process.stdout)
        result['wall_s'] = wall
        values = np.asarray(result.pop('values')).reshape(inputs[0].shape)
        for name in ('key_generation_s', 'encryption_s', 'addition_s', 'decryption_s'):
            setattr(self.timing, name, getattr(self.timing, name) + result['timing'][name])
        self.timing.encrypted_values += len(inputs) * inputs[0].size
        self.timing.decrypted_values += values.size
        result['scalar_values'] = len(inputs) * inputs[0].size
        self.releases.append(result)
        return values
