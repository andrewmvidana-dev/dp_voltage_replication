"""Focused opt-in checks for the public voltage attack, using real OpenDSS."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import run_voltage_set_attack as attack
from dpvolt.powerflow import PowerFlowRunner


class NoSimulatorAccess:
    def __getattr__(self, name):
        raise AssertionError("The attacker attempted to read simulator state")


class VoltageAttackChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master = attack.ROOT / "feeders/IEEE123Master.dss"
        cls.model = attack.capture_public_voltage_model(master)
        cls.runner = PowerFlowRunner(str(master))
        with np.load(attack.ROOT / "results/privacy_comparison_20260924/synthetic_inputs.npz") as data:
            cls.truth = data["reference_sum"]

    def test_independent_terminal_and_setpoint_validation(self):
        # This compares different equations against the actual DSS engine.
        result = attack.validate_voltage_inversion(self.runner, self.model, self.truth)
        self.assertTrue(result["gate_passed"], result)
        self.assertEqual(result["minimum_rank"], len(self.model.laws))
        # Mixed voltage-dependent load models make nominal and actual power differ.
        self.assertGreater(result["actual_terminal_power_minus_nominal_kw"]["max_abs"], .1)

    def test_recovery_has_no_live_simulator_or_truth_access(self):
        p = self.truth[:, :1]
        q = p * np.array([law.q_over_p for law in self.model.laws])[:, None]
        self.runner.reset()
        v, ok = self.runner.solve_trajectory(p, q)
        self.assertTrue(ok.all())
        released = v[0, self.model.retained].copy()
        # No DSS calls, omitted phasors or true loads are available to this call.
        with patch.object(attack, "dss", NoSimulatorAccess()):
            estimate, diagnostic, _, _ = self.model.recover_load_setpoints(released)
        self.assertEqual(diagnostic["rank"], len(self.model.laws))
        self.assertLess(np.max(np.abs(estimate - p[:, 0] * 1000)), .1)

    def test_identical_colocated_loads_are_not_claimed_identifiable(self):
        # Two indistinguishable public laws on the same nodes cannot be separated.
        ambiguous = replace(self.model, laws=self.model.laws + [self.model.laws[0]])
        p = self.truth[:, :1]
        q = p * np.array([law.q_over_p for law in self.model.laws])[:, None]
        self.runner.reset()
        v, _ = self.runner.solve_trajectory(p, q)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            ambiguous.recover_load_setpoints(v[0, self.model.retained])

    def test_malformed_phasors_are_rejected(self):
        for values in (np.ones(len(self.model.retained) - 1),
                       np.full(len(self.model.retained), np.nan)):
            with self.assertRaisesRegex(ValueError, "finite complex phasor"):
                self.model.recover_nodal_injections(values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
