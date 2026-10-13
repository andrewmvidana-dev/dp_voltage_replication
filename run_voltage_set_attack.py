"""Validate what a public retained-voltage release reveals about load setpoints.

This is an opt-in experiment. It does not change the original feeder or runner.
The attacker only receives retained complex phasors and a public network model.
Actual load powers and omitted voltages are used exclusively to score validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import opendssdirect as dss
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.stats import norm

from dpvolt.powerflow import PowerFlowRunner


ROOT = Path(__file__).resolve().parent
LOAD_SOURCE = "https://github.com/dss-extensions/dss_capi/blob/master/src/PCElements/Load.pas"


def _complex(values):
    values = np.asarray(values)
    return values[::2] + 1j * values[1::2]


def errors(values):
    values = np.asarray(values)
    return {"max_abs": float(np.max(np.abs(values))),
            "rms": float(np.sqrt(np.mean(np.abs(values) ** 2)))}


@dataclass
class LoadLaw:
    name: str
    bus: str
    model: int
    branches: list[tuple[int, int]]
    base_volts: float
    q_over_p: float
    vlow: float
    vmin: float
    vmax: float
    delta: bool

    def unit_branch_currents(self, volts):
        """Return each branch's current per nominal kW, from public data only.

        These are independent equations, checked against OpenDSS terminal values.
        Below the normal-voltage range DSS interpolates current, not power.
        """
        out = []
        per_branch_va = 1000 * complex(1, self.q_over_p) / len(self.branches)
        for a, b in self.branches:
            v = (volts[a] if a >= 0 else 0) - (volts[b] if b >= 0 else 0)
            u = abs(v) / self.base_volts
            if self.model == 2 or u <= self.vlow:
                factor = 1.0
            elif u <= self.vmin:
                # Current magnitude varies linearly from the impedance endpoint.
                high = 1 / self.vmin if self.model == 1 else 1.0
                relative_current = self.vlow + (high - self.vlow) * (
                    (u - self.vlow) / (self.vmin - self.vlow))
                factor = relative_current / u
            elif u > self.vmax:
                factor = 1 / self.vmax ** (2 if self.model == 1 else 1)
            elif self.model == 1:
                factor = 1 / u ** 2
            else:
                factor = 1 / u
            out.append(np.conj(per_branch_va) * v / self.base_volts ** 2 * factor)
        return np.asarray(out)


@dataclass
class PublicVoltageModel:
    names: list[str]
    vbase: np.ndarray
    retained: np.ndarray
    omitted: np.ndarray
    y: csc_matrix
    source: np.ndarray
    reconstruction: np.ndarray
    offset: np.ndarray
    reduced_y: np.ndarray
    reduced_source: np.ndarray
    laws: list[LoadLaw]
    solver_tolerance: float
    metadata: dict

    def recover_nodal_injections(self, retained_v_pu):
        """Use the public Norton source drive; never read a private slack voltage."""
        values = np.asarray(retained_v_pu)
        if values.shape != (len(self.retained),) or not np.all(np.isfinite(values)):
            raise ValueError("Expected one finite complex phasor for each retained node")
        vr = values * self.vbase[self.retained]
        volts = np.empty(len(self.names), dtype=complex)
        volts[self.retained] = vr
        volts[self.omitted] = self.offset - self.reconstruction @ vr
        consumed_current = self.reduced_source - self.reduced_y @ vr
        consumed_kva = vr * np.conj(consumed_current) / 1000
        return volts, consumed_current, consumed_kva

    def current_map(self, volts, indices=None):
        """Each real coefficient is one load element's nominal kW setpoint."""
        indices = list(range(len(self.laws))) if indices is None else list(indices)
        index = {int(n): k for k, n in enumerate(self.retained)}
        matrix = np.zeros((len(self.retained), len(indices)), dtype=complex)
        for column, k in enumerate(indices):
            law = self.laws[k]
            for (a, b), value in zip(law.branches, law.unit_branch_currents(volts)):
                if a >= 0:
                    matrix[index[a], column] += value
                if b >= 0:
                    matrix[index[b], column] -= value
        return matrix

    def recover_load_setpoints(self, retained_v_pu, target=None):
        volts, current, power = self.recover_nodal_injections(retained_v_pu)
        # Independent buses give small real least-squares systems. Delta branches
        # are solved together because the branch powers share terminal nodes.
        groups = {}
        for k, law in enumerate(self.laws):
            if target is None or law.bus == self.laws[target].bus:
                groups.setdefault(law.bus, []).append(k)
        result = np.full(len(self.laws), np.nan)
        ranks, conditions = [], []
        map_residuals = []
        for indices in groups.values():
            matrix = self.current_map(volts, indices)
            rows = np.flatnonzero(np.any(np.abs(matrix) > 0, axis=1))
            a = np.vstack([matrix[rows].real, matrix[rows].imag])
            y = np.r_[current[rows].real, current[rows].imag]
            estimate, _, rank, singular = np.linalg.lstsq(a, y, rcond=None)
            if rank != len(indices):
                raise ValueError("Public load-current map is rank deficient")
            result[indices] = estimate
            ranks.append(int(rank))
            conditions.append(float(singular[0] / singular[-1]))
            map_residuals.extend((a @ estimate - y).tolist())
        return result, {"rank": sum(ranks), "max_bus_condition": max(conditions),
                        "current_map_residual_a": errors(map_residuals)}, volts, power


def capture_public_voltage_model(master_path):
    """Compile a separate public model so the attacker cannot read live secrets."""
    public = dss.NewContext()
    public.Text.Command(f'Redirect "{master_path}"')
    public.Text.Command("Solve")
    if not public.Solution.Converged():
        raise RuntimeError("Public feeder initialization did not converge")
    public.Text.Command("Set ControlMode=OFF")
    names = [s.lower() for s in public.Circuit.YNodeOrder()]
    lookup = {name: i for i, name in enumerate(names)}
    bases = {}
    for bus in public.Circuit.AllBusNames():
        public.Circuit.SetActiveBus(bus)
        bases[bus.lower()] = public.Bus.kVBase() * 1000
    vbase = np.array([bases[name.split('.')[0]] for name in names])
    laws = []
    i = public.Loads.First()
    while i:
        phases = public.Loads.Phases()
        delta = public.Loads.IsDelta()
        bus = public.CktElement.BusNames()[0].split('.')[0].lower()
        nodes = public.CktElement.NodeOrder()
        terminal = [-1 if node == 0 else lookup[f"{bus}.{node}"] for node in nodes]
        if public.Loads.Model() not in (1, 2, 5):
            raise ValueError("This validation only implements feeder load models 1, 2 and 5")
        if not delta and terminal[-1] != -1:
            raise ValueError("Ungrounded wye neutral requires a separate public neutral model")
        pairs = [(terminal[j], terminal[(j + 1) % len(terminal)] if delta
                  else terminal[-1]) for j in range(phases)]
        rated = public.Loads.kV() * 1000
        if not delta and phases > 1:
            rated /= np.sqrt(3)
        pf = public.Loads.PF()
        laws.append(LoadLaw(public.Loads.Name(), bus, public.Loads.Model(), pairs,
                            rated, float(np.tan(np.arccos(np.clip(pf, -1, 1)))),
                            float(public.Properties.Value("vlowpu")),
                            public.Loads.Vminpu(), public.Loads.Vmaxpu(), delta))
        i = public.Loads.Next()
    retained = np.array(sorted({n for law in laws for pair in law.branches
                                for n in pair if n >= 0}))
    omitted = np.setdiff1d(np.arange(len(names)), retained)
    taps = {}
    i = public.Transformers.First()
    while i:
        for winding in range(1, public.Transformers.NumWindings() + 1):
            public.Transformers.Wdg(winding)
            taps[f"{public.Transformers.Name()}:{winding}"] = public.Transformers.Tap()
        i = public.Transformers.Next()
    capacitors = {}
    i = public.Capacitors.First()
    while i:
        capacitors[public.Capacitors.Name()] = public.Capacitors.States()
        i = public.Capacitors.Next()
    tolerance = public.Solution.Convergence()
    # Load admittances must be absent. Capacitors, transformers and finite source
    # impedance remain in Y. Frozen controls preserve the published circuit.
    public.Text.Command("BatchEdit Load..* enabled=no")
    public.Text.Command("Solve")
    if [s.lower() for s in public.Circuit.YNodeOrder()] != names:
        raise ValueError("Node ordering changed during public Y capture")
    y = csc_matrix(public.YMatrix.getYsparse(), shape=(len(names), len(names)))
    public.YMatrix.ZeroInjCurr()
    public.YMatrix.GetSourceInjCurrents()
    source = _complex(public.YMatrix.getI())[1:].copy()
    # Unlike the older Kron b, this drive comes from public source parameters.
    # All omitted nodes, including source terminals, are eliminated together.
    yz = y[omitted][:, omitted]
    lu = splu(yz)
    reconstruction = lu.solve(y[omitted][:, retained].toarray())
    offset = lu.solve(source[omitted])
    yz_residual = float(np.max(np.abs(yz @ reconstruction - y[omitted][:, retained].toarray())))
    yrz = y[retained][:, omitted]
    reduced_y = y[retained][:, retained].toarray() - yrz @ reconstruction
    reduced_source = source[retained] - yrz @ offset
    metadata = {"node_count": len(names), "retained_count": len(retained),
                "load_elements": len(laws), "source_nodes": [names[j] for j in
                np.flatnonzero(np.abs(source) > 0)], "frozen_transformer_taps": taps,
                "fixed_capacitor_states": capacitors,
                "elimination_residual_siemens": yz_residual,
                "source_offset": "Public Norton source currents, not solved private slack phasors",
                "load_models": {str(m): sum(l.model == m for l in laws) for m in (1, 2, 5)},
                "delta_elements": sum(l.delta for l in laws),
                "load_law_reference": LOAD_SOURCE}
    return PublicVoltageModel(names, vbase, retained, omitted, y, source,
                              reconstruction, offset, reduced_y, reduced_source,
                              laws, tolerance, metadata)


def _actual_terminal_currents(runner):
    """Scoring oracle only. This function is never called by the attacker."""
    current = np.zeros(len(runner.node_names), dtype=complex)
    lookup = {name: k for k, name in enumerate(runner.node_names)}
    actual_kw = []
    for name in runner.load_names:
        dss.Loads.Name(name)
        bus = dss.CktElement.BusNames()[0].split('.')[0].lower()
        for node, value in zip(dss.CktElement.NodeOrder(), _complex(dss.CktElement.Currents())):
            if node:
                current[lookup[f"{bus}.{node}"]] += value
        actual_kw.append(float(np.sum(np.asarray(dss.CktElement.Powers())[::2])))
    return current, np.asarray(actual_kw)


def evaluate_release(runner, model, release, truth, validate=False):
    clipped = np.clip(release, 0, .2)
    q = np.array([law.q_over_p for law in model.laws])[:, None] * clipped
    recovered = []
    setpoint_residual, current_residual, power_residual, map_residual = [], [], [], []
    omitted_residual, actual_vs_nominal = [], []
    rank, conditioning = [], []
    runner.reset()
    for t in range(clipped.shape[1]):
        v, ok = runner.solve_trajectory(clipped[:, t:t + 1], q[:, t:t + 1])
        if not ok.all():
            raise RuntimeError(f"Power flow failed at snapshot {t}")
        estimate, diagnostic, reconstructed, inferred_kva = model.recover_load_setpoints(
            v[0, model.retained])
        recovered.append(estimate)
        setpoint_residual.extend((estimate - clipped[:, t] * 1000).tolist())
        rank.append(diagnostic["rank"])
        conditioning.append(diagnostic["max_bus_condition"])
        if validate:
            actual_i, actual_kw = _actual_terminal_currents(runner)
            volts = v[0] * model.vbase
            inferred_i = model.source - model.y @ reconstructed
            current_residual.extend((inferred_i[model.retained] - actual_i[model.retained]).tolist())
            actual_kva = volts[model.retained] * np.conj(actual_i[model.retained]) / 1000
            power_residual.extend((inferred_kva - actual_kva).tolist())
            map_i = model.current_map(volts) @ (clipped[:, t] * 1000)
            map_residual.extend((map_i - actual_i[model.retained]).tolist())
            # Floating unloaded delta nodes can have an arbitrary common mode.
            # Report their diagnostic error without giving them to the attacker.
            omitted_residual.extend(((reconstructed - volts) / model.vbase)[model.omitted].tolist())
            actual_vs_nominal.extend((actual_kw - clipped[:, t] * 1000).tolist())
    recovered = np.asarray(recovered).T
    result = {"evidence": "MEASURED", "all_converged": True,
              "snapshots": clipped.shape[1], "minimum_rank": min(rank),
              "max_bus_condition": max(conditioning),
              "setpoint_reconstruction_error_kw": errors(setpoint_residual),
              "recovered_vs_true_load_rmse_kw": errors(recovered - truth * 1000)["rms"],
              "recovered_vs_raw_release_rmse_kw": errors(recovered - release * 1000)["rms"],
              "recovered_vs_clipped_release_rmse_kw": errors(recovered - clipped * 1000)["rms"]}
    if validate:
        result.update({"nodal_current_residual_a": errors(current_residual),
                       "nodal_power_residual_kva": errors(power_residual),
                       "load_law_current_residual_a": errors(map_residual),
                       "omitted_voltage_diagnostic_pu": errors(omitted_residual),
                       "actual_terminal_power_minus_nominal_kw": errors(actual_vs_nominal)})
    return result


def validate_voltage_inversion(runner, model, truth):
    if runner.node_names != model.names or runner.load_names != [law.name for law in model.laws]:
        raise ValueError("Public and simulation node/load names do not match")
    if not np.array_equal(runner.retained_indices(), model.retained):
        raise ValueError("Public retained set does not match the released set")
    result = evaluate_release(runner, model, truth, truth, validate=True)
    # Fixed before experiments: normalize power by the repository's 1 MVA base.
    # OpenDSS tolerance is a voltage iteration tolerance, so this is an explicit
    # engineering gate, not a theorem equating voltage and power errors.
    threshold_kw = 1000 * model.solver_tolerance
    result.update({"solver_voltage_iteration_tolerance_pu": model.solver_tolerance,
                   "predeclared_power_residual_limit_kw_or_kva": threshold_kw,
                   "predeclared_load_law_current_limit_a": model.solver_tolerance * 1e6 / np.min(model.vbase[model.retained]),
                   "gate_definition": "Rank complete; maximum nominal and nodal power mismatch no greater than 1 MVA times solver tolerance; unit load law matches independent terminal currents",
                   "naive_identity_hypothesis": "YV recovers actual nodal injections, not nominal element kW for voltage-dependent loads"})
    law_limit = model.solver_tolerance * 1e6 / np.min(model.vbase[model.retained])
    result["gate_passed"] = bool(
        result["minimum_rank"] == len(model.laws)
        and result["setpoint_reconstruction_error_kw"]["max_abs"] <= threshold_kw
        and result["nodal_power_residual_kva"]["max_abs"] <= threshold_kw
        and result["load_law_current_residual_a"]["max_abs"] <= law_limit)
    if not result["gate_passed"]:
        result["failure_reason"] = "Public-model inversion exceeds a predeclared residual or identifiability gate. Downstream voltage attacks are withheld."
    return result


def _wilson(success, n):
    z = norm.ppf(.975)
    p = success / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def voltage_neighboring_attacks(runner, model, truth, trials, distributed_b_kw):
    """Every trial here actually runs AC power flow, then reads only voltages.

    Encryption conditions share this same public-voltage experiment because
    their validated release differs only by encoding. It is not a ciphertext
    or key-compromise attack, which belongs in the separate attack harness.
    """
    answer = {}
    target = 0
    sensitivity_kw, delta = 20., .01
    sigma_kw = sensitivity_kw / (2 * norm.ppf((1 + delta)/2))
    center_kw = 100.
    qratio = np.array([law.q_over_p for law in model.laws])
    for mechanism in ("None", "Gaussian DP", "BNP", "Distributed BNP"):
        rng = np.random.default_rng(731013)
        successes, failures, completed = 0, 0, 0
        max_inverse_error = 0.
        started = perf_counter()
        runner.reset()
        for t in range(trials):
            # The other nine target meters total 90 kW. The changed meter is
            # either 0 or 20 kW. Both neighboring datasets satisfy public bounds.
            bit = int(rng.integers(2))
            clean_kw = truth[:, 0] * 1000
            clean_kw[target] = center_kw + (2*bit-1) * sensitivity_kw/2
            if mechanism == "None":
                noise = np.zeros(len(clean_kw))
            elif mechanism == "Gaussian DP":
                # Sum all meter shares, even though the aggregate is Gaussian.
                noise = rng.normal(0, sigma_kw/np.sqrt(10), (len(clean_kw), 10)).sum(axis=1)
            elif mechanism == "BNP":
                noise = rng.uniform(-1000, 1000, len(clean_kw))
            else:
                noise = rng.uniform(-distributed_b_kw, distributed_b_kw,
                                    (len(clean_kw), 10)).sum(axis=1)
            p = np.clip(clean_kw + noise, 0, 200)[:, None]/1000
            v, ok = runner.solve_trajectory(p, p*qratio[:, None])
            if not ok.all():
                failures += 1
                continue
            estimate, _, _, _ = model.recover_load_setpoints(v[0, model.retained], target=target)
            max_inverse_error = max(max_inverse_error, abs(estimate[target] - p[target, 0]*1000))
            guess = int(estimate[target] > center_kw)
            successes += guess == bit
            completed += 1
            if (t + 1) % 2000 == 0:
                print(f"Voltage-only {mechanism}: {t+1}/{trials} real AC solves", flush=True)
        answer[mechanism] = {"evidence": "MEASURED", "requested_trials": trials,
            "completed_trials": completed, "failed_powerflows": failures,
            "accuracy": float(successes/completed) if completed else None,
            "ci95_wilson": _wilson(successes, completed) if completed else None,
            "accuracy_conditioning": ("All generated trials" if not failures else
                "Conditional on power-flow convergence; not an unconditional attack-accuracy estimate"),
            "analytic_accuracy": 1. if mechanism == "None" else .505,
            "analytic_evidence": "ANALYTIC",
            "analytic_formula": "(1 + TV)/2; TV=1 without noise or delta=0.01 for matched additive noise. Clipping at 0 and 200 kW preserves this chosen midpoint decision.",
            "predicted_accuracy": 1. if mechanism == "None" else .505,
            "prediction_evidence": "PREDICTED", "wall_s": perf_counter()-started,
            "max_target_reconstruction_error_kw": float(max_inverse_error),
            "classifier": "Threshold the nominal target-element kW recovered from retained phasors at 100 kW",
            "target_load": model.laws[target].name,
            "privacy_unit": "One bounded meter reading at one time",
            "scope": "Voltage-only public observation, common to all adversaries. Plaintext mechanism without encrypted quantization.",
            "status": "complete" if completed == trials else "incomplete: no accuracy claim for failed solves"}
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/threat_model_20261013")
    parser.add_argument("--releases", default=None)
    parser.add_argument("--attack-trials", type=int, default=20000)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.attack_trials < 0:
        parser.error("attack-trials must be nonnegative")
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    path = out / "voltage_attack.json"
    result = {"evidence": "MEASURED", "settings": {"attack_trials": args.attack_trials,
        "s_base_kw": 1000., "voltage_bases": "Per-node nominal line-to-neutral volts",
        "voltage_input": "Retained complex per-unit phasors, including angle",
        "power_formula": "V_volts = V_pu * Vbase; load_I = J_source - Y_network @ V_volts; load_kVA = V_volts * conj(load_I) / 1000",
        "release_level": "One row is an OpenDSS load element, not necessarily a unique bus",
        "scope": "Direct load sums only"}, "cases": {}, "voltage_only_attacks": {},
        "limitations": ["Complete noiseless complex phasors and exact public topology, controls, source and load laws are assumed.",
            "Only load models 1, 2 and 5 with the captured connections are implemented.",
            "Recovering a load-element total does not uniquely identify individual meter readings.",
            "Published raw noisy sums already dominate voltage information by post-processing; the voltage attack reconstructs clipped setpoints.",
            "The tolerance gate is an explicit engineering criterion, not a power-error bound implied by the OpenDSS voltage iteration tolerance."]}
    def save():
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    try:
        saved = np.load(ROOT / "results/privacy_comparison_20260924/synthetic_inputs.npz")
        truth = saved["reference_sum"]
        master = ROOT / "feeders/IEEE123Master.dss"
        model = capture_public_voltage_model(master)
        runner = PowerFlowRunner(str(master))
        result["public_model"] = model.metadata
        result["validation"] = validate_voltage_inversion(runner, model, truth)
        save()
        print(json.dumps(result["validation"], indent=2), flush=True)
        if not result["validation"]["gate_passed"]:
            result["status"] = "validation failed; downstream voltage recovery/attacks withheld"
            save()
            return
        if args.validation_only:
            result["status"] = "validation complete; additional experiments not requested"
            save()
            return
        release_path = Path(args.releases) if args.releases else out / "trial0_releases.npz"
        if not release_path.exists():
            raise FileNotFoundError(f"Matched release archive is missing: {release_path}")
        mechanisms = ["None", "Gaussian DP", "BNP", "Distributed BNP"]
        with np.load(release_path) as releases:
            if set(releases.files) != set(mechanisms):
                raise ValueError("The matched release archive must contain all four mechanisms; run the matched comparison to completion before this script")
            for mechanism in mechanisms:
                if releases[mechanism].shape != truth.shape:
                    raise ValueError(f"Wrong release shape for {mechanism}")
                result["cases"][mechanism] = evaluate_release(runner, model, releases[mechanism], truth)
                save()
        crypto_archives = [ROOT / "results/threshold_20261005" / f"{backend}_{mechanism}.npz"
                           for backend in ("Paillier", "Threshold_BGV")
                           for mechanism in ("None", "Gaussian_DP", "BNP")]
        crypto_archives += [out / "Paillier_Distributed_BNP.npz", out / "Threshold_BGV_Distributed_BNP.npz"]
        for archive in sorted(crypto_archives):
            if not archive.exists():
                raise FileNotFoundError(f"Expected encrypted comparison output is missing: {archive}")
            with np.load(archive) as loaded:
                if "release" not in loaded or loaded["release"].shape != truth.shape:
                    raise ValueError(f"Encrypted archive has no correctly shaped release: {archive}")
                case = evaluate_release(runner, model, loaded["release"], truth)
                case["release_evidence"] = "RECORDED" if "threshold_20261005" in str(archive) else "MEASURED"
                case["release_archive"] = str(archive.relative_to(ROOT))
                result["cases"][archive.stem] = case
                save()
        if args.attack_trials:
            from dpvolt.privacy import bnp_distributed_half_width
            b = bnp_distributed_half_width(20., .01, 10)
            result["voltage_only_attacks"] = voltage_neighboring_attacks(
                runner, model, truth, args.attack_trials, b)
        all_trials = all(row["status"] == "complete" for row in result["voltage_only_attacks"].values())
        result["status"] = ("complete" if args.attack_trials >= 20000 and all_trials
                            else "incomplete or diagnostic run: required attack trials not all completed")
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    save()


if __name__ == "__main__":
    main()
