# Private synthetic loads solved through the REAL power flow on the TRUE
# admittance matrix. No noise is ever added to the resulting voltages -- that is
# what keeps the released data physically consistent.
#
# Everything crossing into privacy.py is per-unit. Theorem 1 mixes voltages,
# admittances and loads in one formula, so a volts/per-unit slip changes epsilon
# by orders of magnitude with no warning. IEEE 123 has two voltage levels
# (4.16 kV and 480 V), so Vbase is per-node, not a scalar.

from __future__ import annotations

import numpy as np
import opendssdirect as dss


S_BASE_KW = 1000.0        # 1 MVA base, matching loads.py


def to_per_unit(Y: np.ndarray, Vbase: np.ndarray,
                s_base_kw: float = S_BASE_KW) -> np.ndarray:
    """Convert an admittance matrix from siemens into per-unit."""
    D = np.diag(Vbase)
    return D @ Y @ D / (s_base_kw * 1000.0)


class PowerFlowRunner:
    """Drives OpenDSS to solve power flow for a sequence of load vectors.

    The feeder is compiled once and re-solved many times, which is far faster
    than recompiling. Load element names are recorded up front so we can
    address them directly.
    """

    def __init__(self, master_path: str, freeze_controls: bool = True):
        dss.Text.Command("Clear")
        dss.Text.Command(f"Redirect {master_path}")
        dss.Text.Command("Solve")

        if not dss.Solution.Converged():
            raise RuntimeError("base case did not converge")

        # Fixed order, matching the load vectors produced by loads.py.
        self.load_names = []
        i = dss.Loads.First()
        while i > 0:
            self.load_names.append(dss.Loads.Name())
            i = dss.Loads.Next()

        self.node_names = [s.lower() for s in dss.Circuit.YNodeOrder()]

        lookup = {}
        for bus in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus)
            lookup[bus.lower()] = dss.Bus.kVBase() * 1000.0
        self.Vbase = np.array(
            [lookup[nm.split(".")[0]] for nm in self.node_names]
        )

        # Freeze the regulator taps. Leaving controls active re-taps every
        # timestep, which changes Y between timesteps -- and the paper's whole
        # analysis assumes a FIXED Y. A modelling choice worth stating aloud.
        self.controls_frozen = freeze_controls
        if freeze_controls:
            dss.Text.Command("Set ControlMode=OFF")

    def solve_trajectory(
        self,
        p_pu: np.ndarray,
        q_pu: np.ndarray,
        s_base_kw: float = S_BASE_KW,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve power flow at every timestep of one day.

        p_pu, q_pu   (n_loads, T) active and reactive power, per-unit.
        Returns V (T, n_nodes) complex per-unit voltages, and ok (T,) flags.

        ALWAYS CHECK ok. Real feeders under extreme synthetic loads do fail to
        converge, and a failed solve returns whatever the solver had reached
        when it gave up. Silently averaging those in corrupts every downstream
        statistic, so we return the flags and let the caller decide.
        """
        n_loads, T = p_pu.shape
        if n_loads != len(self.load_names):
            raise ValueError(
                f"got {n_loads} load rows but the feeder has "
                f"{len(self.load_names)} Load elements"
            )

        V = np.zeros((T, len(self.node_names)), dtype=complex)
        ok = np.zeros(T, dtype=bool)

        for t in range(T):
            for k, name in enumerate(self.load_names):
                dss.Loads.Name(name)
                dss.Loads.kW(p_pu[k, t] * s_base_kw)
                dss.Loads.kvar(q_pu[k, t] * s_base_kw)

            dss.Text.Command("Solve")
            ok[t] = dss.Solution.Converged()

            v = np.asarray(dss.Circuit.YNodeVArray())
            V[t] = (v[0::2] + 1j * v[1::2]) / self.Vbase

        return V, ok

    def solve_many(
        self,
        p_pu: np.ndarray,
        q_pu: np.ndarray,
        s_base_kw: float = S_BASE_KW,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve several days. p_pu, q_pu are (n_loads, n_days, T); returns
        V (n_days, T, n_nodes) and ok (n_days, T)."""
        n_days = p_pu.shape[1]
        out_V, out_ok = [], []

        for d in range(n_days):
            V, ok = self.solve_trajectory(p_pu[:, d, :], q_pu[:, d, :], s_base_kw)
            out_V.append(V)
            out_ok.append(ok)

        return np.array(out_V), np.array(out_ok)

    def retained_indices(self) -> np.ndarray:
        """Indices of load-carrying nodes -- the "retained" buses of the Kron
        reduction, over which the privacy analysis is stated."""
        name_to_idx = {nm: i for i, nm in enumerate(self.node_names)}
        hits = set()

        i = dss.Loads.First()
        while i > 0:
            spec = dss.CktElement.BusNames()[0].lower().split(".")
            bus = spec[0]
            phases = [p for p in spec[1:] if p != "0"] or ["1", "2", "3"]
            for ph in phases:
                nm = f"{bus}.{ph}"
                if nm in name_to_idx:
                    hits.add(name_to_idx[nm])
            i = dss.Loads.Next()

        return np.array(sorted(hits))


def add_voltage_noise(
    V: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian output perturbation -- the baseline the paper argues against.

    Noise goes independently onto the real and imaginary parts of every phasor,
    as in the paper's Table III baselines. The result no longer satisfies the
    power flow equations at all; that is precisely the damage the proposed
    method avoids, and quantifying it is what Figures 2 and 3 are for.
    """
    noise = rng.normal(0.0, sigma, size=V.shape) + \
        1j * rng.normal(0.0, sigma, size=V.shape)
    return V + noise
