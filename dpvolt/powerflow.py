# Private synthetic loads solved through the REAL power flow on the TRUE
# admittance matrix. No noise is ever added to the resulting voltages -- that
# is what keeps the released data physically consistent.
#
# Everything crossing into privacy.py is per-unit. Theorem 1 mixes voltages,
# admittances and loads in one formula, so a volts/per-unit slip changes
# epsilon by orders of magnitude and nothing warns you. IEEE 123 has two
# voltage levels (4.16 kV and 480 V), so Vbase is per-node, not a scalar.

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

    The feeder is compiled once and then re-solved many times with different
    load values, which is far faster than recompiling. We also record the
    names of the Load elements up front so we can address them directly.
    """

    def __init__(self, master_path: str, freeze_controls: bool = True):
        dss.Text.Command("Clear")
        dss.Text.Command(f"Redirect {master_path}")
        dss.Text.Command("Solve")

        if not dss.Solution.Converged():
            raise RuntimeError("base case did not converge")

        # Record every Load element's name, in a fixed order that matches the
        # load vectors produced by loads.py.
        self.load_names = []
        i = dss.Loads.First()
        while i > 0:
            self.load_names.append(dss.Loads.Name())
            i = dss.Loads.Next()

        self.node_names = [s.lower() for s in dss.Circuit.YNodeOrder()]

        # Base voltages, needed to report results in per-unit.
        lookup = {}
        for bus in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus)
            lookup[bus.lower()] = dss.Bus.kVBase() * 1000.0
        self.Vbase = np.array(
            [lookup[nm.split(".")[0]] for nm in self.node_names]
        )

        # Freeze the regulator taps. If we leave the controls active, each
        # timestep re-taps the regulators, which changes Y between timesteps.
        # The paper's whole analysis assumes a FIXED Y, so letting the taps
        # wander would quietly violate its central assumption. This is a
        # modelling choice worth stating out loud when you present.
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

        Parameters
        ----------
        p_pu, q_pu   (n_loads, T) active and reactive power in per-unit.

        Returns
        -------
        V        (T, n_nodes) complex node voltages in PER-UNIT
        ok       (T,) booleans, True where the solver converged

        A note on convergence. Real feeders under extreme synthetic loads do
        sometimes fail to converge, and a failed solve returns whatever the
        solver had reached when it gave up. Silently averaging those in would
        corrupt every downstream statistic, so we return the flags and let the
        caller decide. Always check them.
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
            # Write this timestep's loads into the circuit.
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
        """Solve several days at once.

        p_pu, q_pu have shape (n_loads, n_days, T).
        Returns V of shape (n_days, T, n_nodes) and ok of shape (n_days, T).
        """
        n_days = p_pu.shape[1]
        out_V, out_ok = [], []

        for d in range(n_days):
            V, ok = self.solve_trajectory(p_pu[:, d, :], q_pu[:, d, :], s_base_kw)
            out_V.append(V)
            out_ok.append(ok)

        return np.array(out_V), np.array(out_ok)

    def retained_indices(self) -> np.ndarray:
        """Indices of the nodes that carry a load.

        These are the "retained" buses of the Kron reduction, and the ones the
        privacy analysis is stated over.
        """
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

    Noise is added independently to the real and imaginary parts of every
    voltage phasor, which is what the paper's Table III baselines do.

    The resulting voltages no longer satisfy the power flow equations at all.
    That is precisely the damage the proposed method avoids, and quantifying
    it is what Figures 2 and 3 are for.
    """
    noise = rng.normal(0.0, sigma, size=V.shape) + \
        1j * rng.normal(0.0, sigma, size=V.shape)
    return V + noise
