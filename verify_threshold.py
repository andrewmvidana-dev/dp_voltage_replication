"""Correctness and refusal checks for the real threshold BGV backend."""
from itertools import combinations
import unittest
import numpy as np
from dpvolt.threshold_agg import ThresholdBGVSimulation, ThresholdConfig
from dpvolt.secure_agg import SimulatedMeter, SecureAggConfig, fit_class


class ThresholdTests(unittest.TestCase):
    def test_every_quorum(self):
        data = [np.array([1.25, -2.01, 0, .123456789]), np.array([3, 4, -5, .7])]
        for active in combinations(range(1, 6), 3):
            # Fresh key/ciphertext each time: never retry a shared decryption.
            session = ThresholdBGVSimulation(ThresholdConfig(active=active))
            actual = session.sum_prepared(data)
            np.testing.assert_allclose(actual, np.sum(data, axis=0), atol=1e-6, rtol=0)

    def test_refusals(self):
        for active in [(1, 2), (1, 1, 3), (1, 2, 6)]:
            with self.assertRaises(ValueError):
                ThresholdConfig(active=active)
        s = ThresholdBGVSimulation()
        with self.assertRaises(RuntimeError):
            s.sum_prepared([[1e9], [1e9]])
        with self.assertRaises(ValueError):
            s.sum([[1], [2]], 0, 3, 3, np.random.default_rng(0))

    def test_multiple_batches_and_parallel(self):
        data = np.random.default_rng(2).normal(size=(3, 16390))
        s = ThresholdBGVSimulation(ThresholdConfig(workers=2))
        out = s.sum_prepared(data)
        np.testing.assert_allclose(out, data.sum(0), atol=1.5e-6, rtol=0)
        self.assertEqual(s.releases[-1]['ciphertexts'], 6)

    def test_gaussian_and_fit(self):
        class PlainReference:
            def sum(self, contributions, sigma, enrolled, reporters, rng):
                return sum(x + rng.normal(0, sigma, size=x.shape) for x in contributions)
        rng = np.random.default_rng(1)
        readings = np.exp(rng.uniform(-4, -3, (30, 4)))
        meters = [SimulatedMeter(readings / 2, .5) for _ in range(2)]
        args = (meters, 30, 4, -4, -3, 5, 1e-5)
        ref = fit_class(*args, np.random.default_rng(55), session=PlainReference())
        session = ThresholdBGVSimulation()
        enc = fit_class(*args, np.random.default_rng(55), session=session)
        np.testing.assert_allclose(enc[0], ref[0], atol=1e-6, rtol=0)
        np.testing.assert_allclose(enc[1], ref[1], atol=5e-6, rtol=0)
        self.assertEqual(len(session.releases), 2)

    def test_backend_selector(self):
        from dpvolt.experiments import ModelFitConfig, fit_private_load_model
        from dpvolt.loads import fit_load_model
        archive = np.array([[[.02, .03], [.04, .05]]])
        classes, theta = {0: np.arange(1)}, np.array([.2])
        model = fit_load_model(archive, classes, theta)
        model.p_min, model.p_max = {0: .001}, {0: 1.}
        config = ModelFitConfig(mode='secure_aggregation',
            secure=SecureAggConfig(n_meters=2), encryption_backend='threshold_bgv')
        fitted = fit_private_load_model(archive, classes, model, theta, 50, 1e-5,
            np.random.default_rng(1), config=config)
        self.assertTrue(np.isfinite(fitted.mu[0]).all())
        self.assertGreater(config.timing.encrypted_values, 0)
        config.encryption_backend = 'invalid'
        with self.assertRaises(ValueError):
            fit_private_load_model(archive, classes, model, theta, 50, 1e-5,
                np.random.default_rng(1), config=config)


if __name__ == '__main__':
    unittest.main(verbosity=2)
