# Feeder loading, network cleanup, and the Kron reduction producing the reduced
# admittance matrix Y that everything else is built on.
#
# The cleanup step (merge switches, prune stubs) is NOT in the paper. Without it
# kappa_Kron comes out at 2e25 instead of 3e6, which makes Theorem 1 vacuous.

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import opendssdirect as dss      # OpenDSS, the power-flow engine used by utilities


# ---------------------------------------------------------------------------
# 1. Containers
# ---------------------------------------------------------------------------

@dataclass
class Feeder:
    """Everything extracted from one solved OpenDSS feeder, before reduction."""

    Y_full: np.ndarray          # (N, N) complex admittance matrix
    node_names: list[str]       # e.g. "83.2" == bus 83, phase 2
    bus_of_node: list[str]      # e.g. "83"
    V_node: np.ndarray          # (N,) solved complex voltages, in volts
    Vbase: np.ndarray           # (N,) nominal voltage per node, for per-unit
    load_nodes: np.ndarray      # indices of nodes with a load attached
    slack_nodes: np.ndarray     # substation / source nodes
    zero_inj_nodes: np.ndarray  # everything else: no load, no generation

    @property
    def N(self) -> int:
        return len(self.node_names)

    @property
    def V_pu(self) -> np.ndarray:
        """Voltages in per-unit. ANSI C84.1 (and the paper's Definition 2)
        wants every one inside [0.95, 1.05]."""
        return self.V_node / self.Vbase


@dataclass
class KronResult:
    """The reduced network model -- the actual input to the paper's theorem."""

    Y_red: np.ndarray           # (n, n) reduced admittance matrix, eq. (3)
    Phi: np.ndarray             # recovery map: v_Z = Phi @ v_R, eq. (2)
    b: np.ndarray               # (n,) constant-current slack offset, eq. (7)
    kappa_kron: float           # Kron amplification factor, eq. (4) -- the
                                #   number that matters most
    retained: np.ndarray        # nodes that survived (the "R" set)
    zero_inj: np.ndarray        # nodes eliminated (the "Z" set)
    slack: np.ndarray
    names_retained: list[str]
    residual: float = np.nan    # correctness check; should be around 1e-10
    cond_Y_ZZ: float = np.nan   # conditioning diagnostic; want well below 1e12


# ---------------------------------------------------------------------------
# 2. Loading the feeder
# ---------------------------------------------------------------------------

def load_feeder(
    master_path: str,
    network_only_Y: bool = True,
    merge_switches: bool = True,
    prune_stubs: bool = True,
) -> Feeder:
    """Compile an OpenDSS feeder, solve it, and extract the model.

    network_only_Y
        OpenDSS quietly folds an approximate *load* admittance into its system
        Y. The paper's Y is the wiring ONLY, with loads appearing separately as
        power injections s_k, so we rebuild Y with loads switched off. The
        difference is small (~0.06%) but it is a difference from the paper.
    merge_switches, prune_stubs
        See the two functions in section 3. Together they take kappa_Kron from
        2e25 down to 3e6.
    """

    # "Clear" wipes any circuit left from a previous run -- forgetting it is a
    # classic source of baffling bugs when a script is run twice.
    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {master_path}")
    dss.Text.Command("Solve")

    if not dss.Solution.Converged():
        raise RuntimeError(
            "The base-case power flow did not converge. Check master_path."
        )

    # YNodeOrder is the authoritative node ordering, matching Y's rows and
    # columns. Everything downstream depends on it, so grab it once.
    node_names = [s.lower() for s in dss.Circuit.YNodeOrder()]
    N = len(node_names)
    bus_of_node = [nm.split(".")[0] for nm in node_names]

    # OpenDSS returns voltages as flat alternating real/imag.
    v = np.asarray(dss.Circuit.YNodeVArray())
    V_node = v[0::2] + 1j * v[1::2]

    Vbase = _node_base_voltages(node_names)
    slack_nodes = _slack_nodes(node_names)
    load_nodes = _load_nodes(node_names)

    # Record closed-switch node pairs BEFORE disabling anything, since the
    # element list changes as we edit.
    switch_pairs = _switch_node_pairs(node_names) if merge_switches else []

    Y_full = _extract_Y(N, network_only_Y)

    # ---- cleanup -----------------------------------------------------------
    if merge_switches:
        (Y_full, node_names, bus_of_node, V_node, Vbase,
         keep_map) = _merge_switch_nodes(
            Y_full, node_names, bus_of_node, V_node, Vbase, switch_pairs
        )
        # Node indices have shifted; translate the bookkeeping across.
        load_nodes = np.array(sorted({keep_map[i] for i in load_nodes}))
        slack_nodes = np.array(sorted({keep_map[i] for i in slack_nodes}))

    if prune_stubs:
        (Y_full, node_names, bus_of_node, V_node, Vbase,
         keep_map) = _prune_dangling_stubs(
            Y_full, node_names, bus_of_node, V_node, Vbase,
            protected=set(load_nodes) | set(slack_nodes),
        )
        load_nodes = np.array(
            sorted({keep_map[i] for i in load_nodes if i in keep_map})
        )
        slack_nodes = np.array(
            sorted({keep_map[i] for i in slack_nodes if i in keep_map})
        )

    # Whatever is left over is zero-injection.
    mask = np.ones(len(node_names), dtype=bool)
    mask[slack_nodes] = False
    mask[load_nodes] = False
    zero_inj_nodes = np.flatnonzero(mask)

    return Feeder(
        Y_full=Y_full,
        node_names=node_names,
        bus_of_node=bus_of_node,
        V_node=V_node,
        Vbase=Vbase,
        load_nodes=load_nodes,
        slack_nodes=slack_nodes,
        zero_inj_nodes=zero_inj_nodes,
    )


def _extract_Y(N: int, network_only: bool) -> np.ndarray:
    """Pull the admittance matrix out of OpenDSS as a dense array.

    TWO TRAPS, both of which cost real debugging time.

    1. REGULATOR TAPS. Disabling loads and re-solving makes OpenDSS's regulator
       controls notice the feeder is now unloaded and re-tap. On IEEE 123 taps
       swing from (6,0,2,0,10,4,6) to (-1,-1,0,0,2,2,2), so the captured Y
       describes a different transformer configuration from the voltages we
       solved for, and every downstream check fails -- we saw total slack power
       come out as -216 MW instead of +3.6 MW. "Set ControlMode=OFF" freezes
       the taps.

    2. MEMORY VIEWS. getYsparse() returns arrays pointing into OpenDSS's own
       internal memory, which the next Solve() overwrites underneath you.
       np.array(..., copy=True) takes a private copy first.
    """
    if network_only:
        dss.Text.Command("Set ControlMode=OFF")            # trap 1
        dss.Text.Command("BatchEdit Load..* enabled=no")
        dss.Text.Command("Solve")
        raw = dss.YMatrix.getYsparse()
        data, indices, indptr = (np.array(a, copy=True) for a in raw)  # trap 2
        # Put the circuit back exactly as we found it.
        dss.Text.Command("BatchEdit Load..* enabled=yes")
        dss.Text.Command("Set ControlMode=STATIC")
        dss.Text.Command("Solve")
    else:
        raw = dss.YMatrix.getYsparse()
        data, indices, indptr = (np.array(a, copy=True) for a in raw)

    # OpenDSS hands back compressed sparse column format. Dense is fine and
    # easier to reason about at a few hundred rows.
    return sp.csc_matrix((data, indices, indptr), shape=(N, N)).toarray()


def _node_base_voltages(node_names: list[str]) -> np.ndarray:
    """Each node's nominal line-to-neutral voltage in volts, for per-unit."""
    lookup = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        lookup[bus.lower()] = dss.Bus.kVBase() * 1000.0    # kV -> V

    base = np.array([lookup.get(nm.split(".")[0], np.nan) for nm in node_names])
    if np.any(~np.isfinite(base)) or np.any(base <= 0):
        raise RuntimeError("Could not resolve a base voltage for every node.")
    return base


def _slack_nodes(node_names: list[str]) -> np.ndarray:
    """The substation nodes, where the transmission system feeds in.

    The power-flow 'slack' bus: voltage held fixed, supplies whatever the
    feeder needs. On IEEE 123 this is bus 150.
    """
    buses = set()
    i = dss.Vsources.First()
    while i > 0:
        buses.add(dss.CktElement.BusNames()[0].split(".")[0].lower())
        i = dss.Vsources.Next()
    return np.array(
        [k for k, nm in enumerate(node_names) if nm.split(".")[0] in buses],
        dtype=int,
    )


def _terminal_nodes(bus_spec: str) -> list[str]:
    """Expand an OpenDSS bus spec into node names.

    "83.1.2" means bus 83, phases 1 and 2. A bare "83" implies all three
    phases. ".0" is neutral/ground, which is not a node in Y, so we drop it.
    """
    parts = bus_spec.lower().split(".")
    bus = parts[0]
    phases = [p for p in parts[1:] if p != "0"]
    if not phases:
        phases = ["1", "2", "3"]
    return [f"{bus}.{p}" for p in phases]


def _load_nodes(node_names: list[str]) -> np.ndarray:
    """Nodes carrying a load, PV system, generator or storage unit.

    Capacitors are deliberately EXCLUDED: a capacitor is a fixed shunt
    admittance, so it already lives inside Y and its external injection is
    zero. Under Definition 1 that makes it zero-injection, not retained.
    Getting this wrong silently changes n.
    """
    name_to_idx = {nm: i for i, nm in enumerate(node_names)}
    hits = set()

    for getter in (dss.Loads, dss.PVsystems, dss.Generators, dss.Storages):
        try:
            i = getter.First()
        except Exception:
            continue                       # this element class is absent
        while i > 0:
            bus_spec = dss.CktElement.BusNames()[0]     # terminal 1
            for nm in _terminal_nodes(bus_spec):
                if nm in name_to_idx:
                    hits.add(name_to_idx[nm])
            i = getter.Next()

    return np.array(sorted(hits), dtype=int)


def _switch_node_pairs(node_names: list[str]) -> list[tuple[int, int]]:
    """Every pair of nodes joined by a closed switch."""
    name_to_idx = {nm: i for i, nm in enumerate(node_names)}
    pairs = []
    i = dss.Lines.First()
    while i > 0:
        # A "switch" is a Line flagged as such, or one short enough that its
        # impedance is effectively zero.
        if dss.Lines.IsSwitch() or dss.Lines.Length() <= 1e-3:
            b1, b2 = dss.CktElement.BusNames()[:2]
            for a, b in zip(_terminal_nodes(b1), _terminal_nodes(b2)):
                if a in name_to_idx and b in name_to_idx:
                    pairs.append((name_to_idx[a], name_to_idx[b]))
        i = dss.Lines.Next()
    return pairs


# ---------------------------------------------------------------------------
# 3. Network cleanup  (NOT in the paper -- our addition)
# ---------------------------------------------------------------------------

def _merge_switch_nodes(Y, node_names, bus_of_node, V, Vbase, pairs):
    """Collapse each closed switch so its two ends become ONE node.

    IEEE 123 models its 8 switches as lines with essentially zero impedance.
    Admittance is 1/impedance, so those nodes get diagonal entries around 1e6
    against a median of 44. When the Kron reduction then inverts the
    zero-injection block, that 5-order spread makes the inversion numerically
    catastrophic and kappa_Kron -- which depends on ||Y_ZZ^-1|| -- explodes.

    The fix is physical, not numerical: a CLOSED switch means the two buses
    genuinely ARE the same electrical point, so we merge them topologically
    rather than asking linear algebra to eliminate a near-infinite admittance.

    Grouping is union-find: each node starts alone, each switch merges two
    groups, then we sum rows and columns within each group.
    """
    N = len(node_names)

    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]      # path compression
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in pairs:
        union(a, b)

    # P[i, j] = 1 means "old node i belongs to new merged node j". P.T @ Y @ P
    # then sums the rows and columns of each group -- the right way to merge
    # nodes in an admittance matrix, since Kirchhoff's law adds currents.
    groups = sorted({find(k) for k in range(N)})
    gmap = {g: j for j, g in enumerate(groups)}
    P = np.zeros((N, len(groups)))
    for k in range(N):
        P[k, gmap[find(k)]] = 1.0

    keep_map = {k: gmap[find(k)] for k in range(N)}   # old index -> new index
    reps = list(groups)

    return (
        P.T @ Y @ P,
        [node_names[r] for r in reps],
        [bus_of_node[r] for r in reps],
        V[reps],                   # merged nodes share a voltage, so pick one
        Vbase[reps],
        keep_map,
    )


def _prune_dangling_stubs(Y, node_names, bus_of_node, V, Vbase,
                          protected, tol=1e-8):
    """Remove dead-end buses with no load and no generation.

    On IEEE 123, buses 151, 250, 300, 450, 61 and 610 each have one element
    attached and carry no load. A zero-injection dead end breaks the Kron
    reduction: its block looks like [[y, -y], [-y, y]], which is SINGULAR --
    the rows are copies up to sign. Inverting it is impossible exactly and
    disastrous in floating point; we saw a smallest singular value of 2e-7
    against a largest of 2.7e4.

    Physically, no current flows into a dead end that consumes nothing, so its
    voltage just equals its parent's. It carries no information, so delete it
    rather than eliminate it.

    SUBTLETY WE GOT WRONG FIRST TIME: a dead end is a leaf at the BUS level,
    not the NODE level. Three-phase lines couple phases, so node "250.2"
    connects to 250.1, 250.3, 251.1, 251.2, 251.3 -- degree 5, not 1. Collapse
    the node graph to a bus graph before testing. Pruning repeats until nothing
    changes, since removing one stub can expose another behind it.
    """
    keep = list(range(len(node_names)))
    removed_buses = []

    while True:
        A = np.abs(Y[np.ix_(keep, keep)]) > tol
        np.fill_diagonal(A, False)          # a node is not its own neighbour

        # Collapse node adjacency into BUS adjacency.
        bus_positions = {}
        for pos, k in enumerate(keep):
            bus_positions.setdefault(bus_of_node[k], []).append(pos)

        neighbours = {b: set() for b in bus_positions}
        for bus, positions in bus_positions.items():
            for pos in positions:
                for other in np.flatnonzero(A[pos]):
                    neighbours[bus].add(bus_of_node[keep[other]])
        for bus in neighbours:
            neighbours[bus].discard(bus)

        protected_buses = {
            bus_of_node[k] for k in protected if k < len(bus_of_node)
        }

        # A stub is a bus with at most one neighbour and nothing attached.
        drop = [b for b, nb in neighbours.items()
                if len(nb) <= 1 and b not in protected_buses]
        if not drop:
            break

        removed_buses += drop
        drop_set = set(drop)

        # IMPORTANT: ELIMINATE the stub, do not merely delete its rows.
        #
        # A stub connects to its parent by a branch of admittance y. Deleting
        # its row and column leaves the parent with +y on its diagonal and
        # nothing on the other end -- an invented shunt to ground. Power balance
        # then fails by hundreds of MVA, which is exactly what verify.py's
        # zero-injection check caught the first time we wrote this.
        #
        # The correct operation is the same Schur complement as the Kron
        # reduction, applied just to the stub:
        #     Y_new = Y_kk - Y_kd @ inv(Y_dd) @ Y_dk
        # For a leaf this subtracts precisely y from the parent's diagonal.
        # One stub at a time is well-conditioned (Y_dd is 1x1 or 3x3), whereas
        # folding these nodes into the big Y_ZZ inversion is what was singular.
        d_pos = [j for j, k in enumerate(keep) if bus_of_node[k] in drop_set]
        k_pos = [j for j, k in enumerate(keep) if bus_of_node[k] not in drop_set]

        Y_kk = Y[np.ix_([keep[j] for j in k_pos], [keep[j] for j in k_pos])]
        Y_kd = Y[np.ix_([keep[j] for j in k_pos], [keep[j] for j in d_pos])]
        Y_dk = Y[np.ix_([keep[j] for j in d_pos], [keep[j] for j in k_pos])]
        Y_dd = Y[np.ix_([keep[j] for j in d_pos], [keep[j] for j in d_pos])]

        Y_reduced = Y_kk - Y_kd @ np.linalg.solve(Y_dd, Y_dk)

        surviving = [keep[j] for j in k_pos]
        Y = _reindex(Y, Y_reduced, surviving)
        keep = surviving

    keep_map = {k: j for j, k in enumerate(keep)}
    return (
        Y[np.ix_(keep, keep)] if Y.shape[0] != len(keep) else Y,
        [node_names[k] for k in keep],
        [bus_of_node[k] for k in keep],
        V[keep],
        Vbase[keep],
        keep_map,
    )


def _reindex(Y_old, Y_small, surviving):
    """Write the reduced block back into a full-size matrix.

    The pruning loop keeps original node indices for readability, so after each
    elimination we place the smaller matrix back at full size with removed rows
    zeroed. Only surviving indices are ever read afterwards.
    """
    Y_new = np.zeros_like(Y_old)
    Y_new[np.ix_(surviving, surviving)] = Y_small
    return Y_new


# ---------------------------------------------------------------------------
# 4. The Kron reduction  (Section II-A of the paper)
# ---------------------------------------------------------------------------

def kron_reduce(feeder: Feeder, verify: bool = True) -> KronResult:
    """Eliminate zero-injection buses via the Schur complement.

    Split the buses into R (retained: load or generator), Z (zero-injection,
    nothing attached) and S (slack / substation). Because Z buses inject no
    power their voltages are fully determined by the R buses, so we solve them
    out algebraically and work with a smaller system of identical physics.
    Eliminating the Z rows gives eq. (3):

        Y_red = Y_RR - Y_RZ @ inv(Y_ZZ) @ Y_ZR

    (the Schur complement; in power systems, Kron reduction).

    WHY kappa_Kron MATTERS. The privacy guarantee only holds when
        alpha = ||M~^-1|| * C_star * kappa_Kron * r  <  1/4
    where r is how large a network change the mechanism must hide. kappa_Kron
    multiplies r directly, so at kappa_Kron = 1e25, r must be below ~1e-27 for
    the theorem to apply at all -- protecting against essentially nothing.
    Getting it down to 3e6 is what makes this replication viable.
    """
    Y = feeder.Y_full
    R = feeder.load_nodes
    Z = feeder.zero_inj_nodes
    S = feeder.slack_nodes

    if len(Z) == 0:
        raise RuntimeError("No zero-injection nodes found -- check classification.")

    Y_RR = Y[np.ix_(R, R)]
    Y_RZ = Y[np.ix_(R, Z)]
    Y_ZR = Y[np.ix_(Z, R)]
    Y_ZZ = Y[np.ix_(Z, Z)]

    # You lose roughly log10(cond) digits in an inversion; above 1e12 nothing
    # a 64-bit float offers survives.
    cond = float(np.linalg.cond(Y_ZZ))
    if cond > 1e10:
        warnings.warn(
            f"Y_ZZ is ill-conditioned (cond = {cond:.2e}). kappa_Kron will be "
            "unreliable. Are merge_switches and prune_stubs both switched on?"
        )

    # solve(A, B) computes inv(A) @ B more accurately than forming the inverse.
    Y_ZZ_inv_Y_ZR = np.linalg.solve(Y_ZZ, Y_ZR)

    Y_red = Y_RR - Y_RZ @ Y_ZZ_inv_Y_ZR      # eq. (3)
    Phi = -Y_ZZ_inv_Y_ZR                     # eq. (2)

    # eq. (7): the substation acts as a fixed current source on the retained
    # buses.
    #
    # SUBTLETY THAT COST US A BUG. The naive reading b = Y[R, S] @ v_S
    # evaluates to exactly ZERO on IEEE 123, because no load bus is directly
    # adjacent to the substation -- bus 150 reaches the network only through
    # the regulator and a chain of zero-injection buses. The slack does drive
    # the retained buses, just along paths through Z, so the coupling must be
    # pushed through the same Schur complement:
    #
    #     b = (Y_RS - Y_RZ @ inv(Y_ZZ) @ Y_ZS) @ v_S
    #
    # With b = 0 the reduced system has no voltage reference, Y_red is nearly
    # singular, ||M~^-1|| blows up to ~1e16, and Theorem 1 returns infinity for
    # every adjacency radius.
    Y_RS = Y[np.ix_(R, S)]
    Y_ZS = Y[np.ix_(Z, S)]
    b = (Y_RS - Y_RZ @ np.linalg.solve(Y_ZZ, Y_ZS)) @ feeder.V_node[S]

    # eq. (4): the Kron amplification factor.
    op_RZ = np.linalg.norm(Y_RZ, 2)
    op_ZZinv = np.linalg.norm(np.linalg.inv(Y_ZZ), 2)
    kappa = 1.0 + 2.0 * op_RZ * op_ZZinv + (op_RZ * op_ZZinv) ** 2

    # ---- correctness check -------------------------------------------------
    # The reduction claims v_Z is recoverable from v_R, and we know the true
    # v_Z from OpenDSS, so check it directly.
    residual = np.nan
    if verify:
        vR, vZ = feeder.V_node[R], feeder.V_node[Z]
        drive = np.linalg.solve(Y_ZZ, Y[np.ix_(Z, S)] @ feeder.V_node[S])
        vZ_hat = Phi @ vR - drive
        residual = float(np.abs(vZ - vZ_hat).max() / np.abs(vZ).max())

    return KronResult(
        Y_red=Y_red, Phi=Phi, b=b, kappa_kron=float(kappa),
        retained=R, zero_inj=Z, slack=S,
        names_retained=[feeder.node_names[i] for i in R],
        residual=residual, cond_Y_ZZ=cond,
    )


def injection_check(feeder: Feeder) -> dict:
    """Independent sanity check on the node classification.

    I = Y @ V and S = V * conj(I). If we labelled nodes correctly, |S| must be
    essentially zero at every node we called zero-injection; if not, the
    classification is wrong and every number after this point is meaningless.
    """
    I = feeder.Y_full @ feeder.V_node
    S = feeder.V_node * np.conj(I)                # complex power, in VA

    return {
        "max_|S|_zero_inj_kVA": float(np.abs(S[feeder.zero_inj_nodes]).max() / 1e3),
        "max_|S|_load_kVA": float(np.abs(S[feeder.load_nodes]).max() / 1e3),
        "total_slack_kW": float(np.real(S[feeder.slack_nodes]).sum() / 1e3),
    }
