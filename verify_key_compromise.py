"""Focused tests of observer boundaries and opt-in individual decryption."""
import unittest

import numpy as np

from dpvolt.threshold_agg import ThresholdBGVSimulation, ThresholdConfig
from run_key_compromise import (observer_view, parameters, prepare_contributions,
    run_attack_trials, paillier_demo, wilson)


class KeyCompromiseTests(unittest.TestCase):
    def test_encrypted_no_key_view_excludes_meter_plaintexts(self):
        messages = np.array([[20., 10., 999.]])
        raw = messages.sum(axis=1)
        for backend in ('Paillier', 'Threshold BGV'):
            for adversary in ('Public analyst', 'Curious aggregator'):
                view = observer_view(backend, adversary, messages, raw)
                self.assertIsNone(view.individual_messages)
                np.testing.assert_array_equal(view.public_raw_sum, raw)
        self.assertIsNotNone(observer_view('Paillier', 'Key thief', messages, raw).individual_messages)

    def test_service_is_separate_from_bnp_meter_messages(self):
        config = parameters()
        readings = np.full((3, 10), 10.)
        prepared = prepare_contributions(readings, 'BNP', np.random.default_rng(7), config)
        np.testing.assert_array_equal(prepared[:, :10], readings)
        self.assertTrue(np.all(np.abs(prepared[:, 10]) <= config['central_half_width_kw']))
        gaussian = prepare_contributions(readings, 'Gaussian DP', np.random.default_rng(7), config)
        np.testing.assert_array_equal(gaussian[:, 10], 0)

    def test_public_release_prevents_chance_only_aggregator_claim(self):
        cells, _ = run_attack_trials(20000, 20261013, parameters())
        for row in cells:
            if row['backend'] == 'Threshold BGV' and row['mechanism'] == 'None' and row['adversary'] == 'Curious aggregator' and row['channel'] == 'full_view':
                self.assertEqual(row['measured']['accuracy'], 1)
                self.assertEqual(row['analytic']['accuracy'], 1)
                self.assertEqual(row['predicted']['accuracy'], .5)
        for row in cells:
            if row['channel'] == 'clipped_sum':
                raw = next(other for other in cells if all(other[key] == row[key]
                    for key in ('backend', 'mechanism', 'adversary', 'target')) and other['channel'] == 'raw_sum')
                self.assertEqual(row['measured'], raw['measured'])

    def test_wilson_handles_boundaries(self):
        self.assertGreater(wilson(20000, 20000)[0], .999)
        self.assertEqual(wilson(20000, 20000)[1], 1)
        self.assertEqual(wilson(0, 20000)[0], 0)

    def test_paillier_originals_and_service_are_actually_decrypted(self):
        inputs = np.array([[.02001, -.03], [.01, .022], [.75, -.123]])
        aggregate, recovered, metadata = paillier_demo(inputs)
        np.testing.assert_allclose(recovered, inputs, rtol=0, atol=1e-10)
        np.testing.assert_allclose(aggregate, inputs.sum(axis=0), rtol=0, atol=1e-10)
        self.assertEqual(metadata['ciphertext_messages'], len(inputs))
        self.assertTrue(metadata['no_key_capability_check'])

    def test_bgv_copies_preserve_original_first_meter(self):
        inputs = np.array([[.02001, -.03], [.01, .022], [.75, -.123]])
        session = ThresholdBGVSimulation(ThresholdConfig(active=(1, 3, 5)))
        evidence = session.demonstrate_key_compromise(inputs)
        np.testing.assert_allclose(evidence['recovered_contributions'], inputs,
                                   rtol=0, atol=.5e-6)
        np.testing.assert_allclose(evidence['values'], inputs.sum(axis=0), rtol=0, atol=1.5e-6)
        self.assertEqual(len(evidence['compromise_ciphertext_digests']), len(inputs))
        self.assertGreater(evidence['compromise_decryption_share_bytes'], 0)
        self.assertEqual(session.releases, [])
        self.assertEqual(session.timing.decrypted_values, 0)
        normal = session.sum_prepared(inputs)
        np.testing.assert_array_equal(normal, evidence['values'])
        self.assertNotIn('recovered_contributions', session.releases[-1])
        self.assertNotIn('compromise_wall_s', session.releases[-1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
