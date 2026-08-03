# Feeder loading, network cleanup, and the Kron reduction that produces the
# reduced admittance matrix Y everything else is built on.
#
# The cleanup step (merge switches, prune stubs) is NOT in the paper. Without
# it kappa_Kron comes out at 2e25 instead of 3e6, which makes Theorem 1 vacuous.

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np              # fast numerical arrays and linear algebra
import scipy.sparse as sp       # "sparse" matrices: mostly-zero matrices
import opendssdirect as dss     # Python control of OpenDSS, the power-flow
                                # engine used by real utilities


# =============================================================================
# SECTION 1 -- Containers
#
# These two classes are just labelled boxes. They hold results so we can pass
# many related values around as a single object instead of twelve separate
# ones.
# =============================================================================

@dataclass
class Feeder:
    """Everything extracted from one solved OpenDSS feeder.

    Think of this as the raw ingredients, before any reduction happens.
    """

    Y_full: np.ndarray          # (N, N) complex admittance matrix. Entry
                                #   [i,j] says how strongly nodes i and j are
                                #   electrically connected.
    node_names: list[str]       # e.g. "83.2" == bus 83, phase 2
    bus_of_node: list[str]      # e.g. "83"   == just the bus part
    V_node: np.ndarray          # (N,) solved complex voltages, in volts
    Vbase: np.ndarray           # (N,) nominal voltage per node, for per-unit
    load_nodes: np.ndarray      # indices of nodes with a load attached
    slack_nodes: np.ndarray     # indices of the substation / source nodes
    zero_inj_nodes: np.ndarray  # everything else: no load, no generation

    @property
    def N(self) -> int:
        """Total number of nodes.

        'property' means you read it like a value: write feeder.N, not
        feeder.N().
        """
        return len(self.node_names)

    @property
    def V_pu(self) -> np.ndarray:
        """Voltages in per-unit, where 1.0 == nominal.

        ANSI C84.1 (and the paper's Definition 2) wants every voltage inside
        the band [0.95, 1.05].
        """
        return self.V_node / self.Vbase


@dataclass
class KronResult:
    """The reduced network model -- the actual input to the paper's theorem."""

    Y_red: np.ndarray           # (n, n) reduced admittance matrix, eq. (3)
    Phi: np.ndarray             # recovery map: v_Z = Phi @ v_R, eq. (2)
    b: np.ndarray               # (n,) constant-current slack offset, eq. (7)
    kappa_kron: float           # Kron amplification factor, eq. (4).
                                #   THIS IS THE NUMBER THAT MATTERS MOST.
    retained: np.ndarray        # which nodes survived (the "R" set)
    zero_inj: np.ndarray        # which nodes were eliminated (the "Z" set)
    slack: np.ndarray
    names_retained: list[str]
    residual: float = np.nan    # correctness check; should be around 1e-10
    cond_Y_ZZ: float = np.nan   # conditioning diagnostic; want well below 1e12


# =============================================================================
# SECTION 2 -- Loading the feeder from disk
# =============================================================================

def load_feeder(
    master_path: str,
    network_only_Y: bool = True,
    merge_switches: bool = True,
    prune_stubs: bool = True,
) -> Feeder:
    """Compile an OpenDSS feeder, solve it, and extract the model.

    Parameters
    ----------
    master_path
        Path to the feeder's master .dss file on your computer.
    network_only_Y
        OpenDSS quietly folds an approximate *load* admittance into its system
        Y. The paper's Y is the wiring ONLY, with loads appearing separately as
        power injections s_k. So we rebuild Y with the loads switched off. The
        difference is small (about 0.06 percent) but it is a difference between
        our model and the paper's, so we remove it.
    merge_switches
        See _merge_switch_nodes below. Fixes a 5-orders-of-magnitude problem.
    prune_stubs
        See _prune_dangling_stubs below. Fixes a further 14 orders of
        magnitude. Together these take kappa_Kron from 2e25 down to 3e6.
    """

    # ---- Solve the feeder in its normal, fully-loaded state ---------------
    # "Clear" wipes any circuit left over from a previous run. Forgetting this
    # is a classic source of baffling bugs when you run a script twice.
    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {master_path}")   # load the feeder definition
    dss.Text.Command("Solve")                     # run the power flow

    if not dss.Solution.Converged():
        raise RuntimeError(
            "The base-case power flow did not converge. Check master_path."
        )

    # YNodeOrder is the authoritative list of node names, in exactly the same
    # order as the rows and columns of Y. Everything downstream depends on
    # this ordering, so we grab it once and never re-derive it.
    node_names = [s.lower() for s in dss.Circuit.YNodeOrder()]
    N = len(node_names)
    bus_of_node = [nm.split(".")[0] for nm in node_names]

    # Solved voltages. OpenDSS returns these as a flat list of alternating
    # real, imaginary, real, imaginary... so we de-interleave into complex
    # numbers. In NumPy, [0::2] means "every 2nd element starting at index 0".
    v = np.asarray(dss.Circuit.YNodeVArray())
    V_node = v[0::2] + 1j * v[1::2]

    Vbase = _node_base_voltages(node_names)
    slack_nodes = _slack_nodes(node_names)
    load_nodes = _load_nodes(node_names)

    # Record which node pairs are joined by a closed switch BEFORE we start
    # disabling elements, because the element list changes as we edit.
    switch_pairs = _switch_node_pairs(node_names) if merge_switches else []

    # ---- Extract Y --------------------------------------------------------
    Y_full = _extract_Y(N, network_only_Y)

    # ---- Clean the network up ---------------------------------------------
    if merge_switches:
        (Y_full, node_names, bus_of_node, V_node, Vbase,
         keep_map) = _merge_switch_nodes(
            Y_full, node_names, bus_of_node, V_node, Vbase, switch_pairs
        )
        # Node indices have shifted, so translate our bookkeeping across.
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

    # ---- Whatever is left over is a zero-injection node -------------------
    # np.ones(N, dtype=bool) makes an array full of True. We set False for
    # every node already accounted for; whatever stays True is the remainder.
    mask = np.ones(len(node_names), dtype=bool)
    mask[slack_nodes] = False
    mask[load_nodes] = False
    zero_inj_nodes = np.flatnonzero(mask)     # positions where mask is True

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
    """Pull the admittance matrix out of OpenDSS as a dense NumPy array.

    TWO TRAPS ARE HANDLED HERE. Both cost real debugging time.

    TRAP 1 -- Regulator taps.
        If you disable the loads and re-solve, OpenDSS's regulator controls
        notice the feeder is now unloaded and re-tap every regulator. On IEEE
        123 the taps swing from (6,0,2,0,10,4,6) to (-1,-1,0,0,2,2,2). The Y
        you capture then describes a DIFFERENT transformer configuration from
        the voltages you solved for, and every downstream check fails. We saw
        total slack power come out as -216 MW instead of +3.6 MW.
        Fix: "Set ControlMode=OFF" freezes the taps where they are.

    TRAP 2 -- Memory views.
        getYsparse() hands back arrays pointing directly into OpenDSS's own
        internal memory. The next Solve() overwrites that memory underneath
        you. Fix: np.array(..., copy=True) takes a private copy first.
    """
    if network_only:
        dss.Text.Command("Set ControlMode=OFF")            # TRAP 1
        dss.Text.Command("BatchEdit Load..* enabled=no")
        dss.Text.Command("Solve")
        raw = dss.YMatrix.getYsparse()
        data, indices, indptr = (np.array(a, copy=True) for a in raw)  # TRAP 2
        # Put the circuit back exactly as we found it.
        dss.Text.Command("BatchEdit Load..* enabled=yes")
        dss.Text.Command("Set ControlMode=STATIC")
        dss.Text.Command("Solve")
    else:
        raw = dss.YMatrix.getYsparse()
        data, indices, indptr = (np.array(a, copy=True) for a in raw)

    # OpenDSS gives Y in "compressed sparse column" format: three plain arrays
    # that SciPy knows how to reassemble. We then convert to a normal dense
    # array, because at a few hundred rows it is small and dense linear
    # algebra is easier to reason about.
    return sp.csc_matrix((data, indices, indptr), shape=(N, N)).toarray()


def _node_base_voltages(node_names: list[str]) -> np.ndarray:
    """Look up each node's nominal line-to-neutral voltage, in volts.

    Needed to convert volts into per-unit, which is how power engineers (and
    the paper's Definition 2) express voltage limits.
    """
    lookup = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        lookup[bus.lower()] = dss.Bus.kVBase() * 1000.0    # kV -> V

    base = np.array([lookup.get(nm.split(".")[0], np.nan) for nm in node_names])
    if np.any(~np.isfinite(base)) or np.any(base <= 0):
        raise RuntimeError("Could not resolve a base voltage for every node.")
    return base


def _slack_nodes(node_names: list[str]) -> np.ndarray:
    """Find the substation nodes, where the transmission system feeds in.

    In power-flow terms this is the 'slack' or 'swing' bus: its voltage is
    held fixed and it supplies whatever power the rest of the feeder needs.
    On IEEE 123 this is bus 150.
    """
    buses = set()
    i = dss.Vsources.First()             # loop over every voltage source
    while i > 0:
        buses.add(dss.CktElement.BusNames()[0].split(".")[0].lower())
        i = dss.Vsources.Next()
    return np.array(
        [k for k, nm in enumerate(node_names) if nm.split(".")[0] in buses],
        dtype=int,
    )


def _terminal_nodes(bus_spec: str) -> list[str]:
    """Turn an OpenDSS bus specification into a list of node names.

    OpenDSS writes connections like "83.1.2", meaning bus 83, phases 1 and 2.
    A bare "83" implicitly means all three phases. A ".0" means neutral or
    ground, which is not a node in Y, so we drop it.
    """
    parts = bus_spec.lower().split(".")
    bus = parts[0]
    phases = [p for p in parts[1:] if p != "0"]
    if not phases:
        phases = ["1", "2", "3"]
    return [f"{bus}.{p}" for p in phases]


def _load_nodes(node_names: list[str]) -> np.ndarray:
    """Find nodes carrying a load, PV system, generator or storage unit.

    Capacitors are deliberately EXCLUDED. A capacitor is a fixed shunt
    admittance, so it already lives inside Y and its external power injection
    is zero. Under the paper's Definition 1 that makes it a zero-injection
    bus, not a retained one. Getting this wrong silently changes n.
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
    """List every pair of nodes joined by a closed switch."""
    name_to_idx = {nm: i for i, nm in enumerate(node_names)}
    pairs = []
    i = dss.Lines.First()
    while i > 0:
        # A "switch" in OpenDSS is a Line flagged as such, or any line short
        # enough that its impedance is effectively zero.
        if dss.Lines.IsSwitch() or dss.Lines.Length() <= 1e-3:
            b1, b2 = dss.CktElement.BusNames()[:2]
            # zip() pairs phase 1 with phase 1, phase 2 with phase 2, and so on
            for a, b in zip(_terminal_nodes(b1), _terminal_nodes(b2)):
                if a in name_to_idx and b in name_to_idx:
                    pairs.append((name_to_idx[a], name_to_idx[b]))
        i = dss.Lines.Next()
    return pairs


# =============================================================================
# SECTION 3 -- Network cleanup  (NOT in the paper -- this is our addition)
# =============================================================================

def _merge_switch_nodes(Y, node_names, bus_of_node, V, Vbase, pairs):
    """Collapse each closed switch so its two ends become ONE node.

    WHY THIS IS NECESSARY
    ---------------------
    IEEE 123 models its 8 switches as lines with essentially zero impedance.
    Admittance is 1 divided by impedance, so "essentially zero impedance"
    becomes "enormous admittance": those nodes get diagonal entries around
    1,000,000 against a median of 44 for a normal node.

    When the Kron reduction then tries to invert the zero-injection block of
    Y, that 5-orders-of-magnitude spread makes the inversion numerically
    catastrophic, and kappa_Kron -- which depends on the size of Y_ZZ inverse
    -- explodes.

    The fix is physical, not numerical. A CLOSED switch means the two buses
    genuinely ARE the same electrical point. So we merge them topologically
    rather than asking linear algebra to eliminate a near-infinite admittance.

    HOW IT WORKS (union-find)
    -------------------------
    "Union-find" is a standard algorithm for grouping things. Every node
    starts in its own group. Each switch merges two groups. At the end every
    node knows which group it belongs to, and we add up the rows and columns
    of Y within each group.
    """
    N = len(node_names)

    # parent[i] records who represents node i's group. Initially: itself.
    parent = list(range(N))

    def find(a):
        """Walk up to the group's representative, flattening the path."""
        while parent[a] != a:
            parent[a] = parent[parent[a]]      # path compression, a speedup
            a = parent[a]
        return a

    def union(a, b):
        """Merge the groups containing a and b."""
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in pairs:
        union(a, b)

    # Build a projection matrix P, where P[i, j] = 1 means "old node i belongs
    # to new merged node j". Then P.T @ Y @ P adds together the rows and
    # columns of each group. That is exactly the right way to merge nodes in
    # an admittance matrix, because Kirchhoff's current law adds currents at
    # a node.
    groups = sorted({find(k) for k in range(N)})
    gmap = {g: j for j, g in enumerate(groups)}
    P = np.zeros((N, len(groups)))
    for k in range(N):
        P[k, gmap[find(k)]] = 1.0

    keep_map = {k: gmap[find(k)] for k in range(N)}   # old index -> new index
    reps = list(groups)                               # representative old node

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
    """Remove dead-end buses that have no load and no generation.

    WHY THIS IS NECESSARY
    ---------------------
    On IEEE 123, buses 151, 250, 300, 450, 61 and 610 each have exactly one
    element attached and carry no load. They are dead ends: electrical
    cul-de-sacs.

    A zero-injection dead end breaks the Kron reduction. Its block of the
    admittance matrix looks like

            [  y   -y ]
            [ -y    y ]

    which is SINGULAR: the two rows are copies of each other up to sign.
    Inverting it is impossible in exact arithmetic and disastrous in floating
    point. On our feeder this showed up as a smallest singular value of 2e-7
    against a largest of 2.7e4.

    Physically the resolution is obvious. No current flows into a dead end
    that consumes nothing, so its voltage simply equals its parent's. It
    carries no information, so we delete it rather than eliminate it.

    ONE SUBTLETY WE GOT WRONG FIRST TIME
    ------------------------------------
    A dead end is a leaf at the BUS level, not the NODE level. A three-phase
    line has mutual coupling between its phases, so node "250.2" connects to
    250.1, 250.3, 251.1, 251.2 and 251.3 -- degree 5, not degree 1. You have
    to collapse the node graph down to a bus graph before testing.

    Pruning repeats until nothing changes, because removing one stub can
    expose another sitting behind it.
    """
    keep = list(range(len(node_names)))
    removed_buses = []

    while True:
        # Adjacency: A[i,j] is True when nodes i and j are directly connected.
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
            neighbours[bus].discard(bus)             # ignore self-connections

        protected_buses = {
            bus_of_node[k] for k in protected if k < len(bus_of_node)
        }

        # A stub is a bus with at most one neighbour and nothing attached.
        drop = [b for b, nb in neighbours.items()
                if len(nb) <= 1 and b not in protected_buses]
        if not drop:
            break                                    # nothing left to prune

        removed_buses += drop
        drop_set = set(drop)

        # IMPORTANT: we must ELIMINATE the stub, not merely delete its rows.
        #
        # A stub is connected to its parent by a branch of admittance y. If we
        # just delete the stub's row and column, the parent keeps +y on its
        # diagonal with nothing on the other end of it -- an invented shunt to
        # ground. Power balance then fails by hundreds of MVA, which is
        # exactly what our Check A caught the first time we wrote this.
        #
        # The correct operation is the same Schur complement used by the Kron
        # reduction, applied just to the stub:
        #
        #     Y_new = Y_kk  -  Y_kd @ inverse(Y_dd) @ Y_dk
        #
        # For a leaf this subtracts precisely y from the parent's diagonal,
        # removing the branch cleanly. Doing it one stub at a time is
        # well-conditioned (Y_dd is a small 1x1 or 3x3 block), whereas folding
        # these nodes into the big Y_ZZ inversion is what was singular.
        d_pos = [j for j, k in enumerate(keep) if bus_of_node[k] in drop_set]
        k_pos = [j for j, k in enumerate(keep) if bus_of_node[k] not in drop_set]

        Y_kk = Y[np.ix_([keep[j] for j in k_pos], [keep[j] for j in k_pos])]
        Y_kd = Y[np.ix_([keep[j] for j in k_pos], [keep[j] for j in d_pos])]
        Y_dk = Y[np.ix_([keep[j] for j in d_pos], [keep[j] for j in k_pos])]
        Y_dd = Y[np.ix_([keep[j] for j in d_pos], [keep[j] for j in d_pos])]

        Y_reduced = Y_kk - Y_kd @ np.linalg.solve(Y_dd, Y_dk)

        # Rebuild the bookkeeping around the surviving nodes.
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

    The pruning loop keeps working with original node indices for readability,
    so after each elimination we place the smaller matrix back into a matrix of
    the original size, leaving the removed rows as zeros. Only the surviving
    indices are ever read afterwards.
    """
    Y_new = np.zeros_like(Y_old)
    Y_new[np.ix_(surviving, surviving)] = Y_small
    return Y_new


# =============================================================================
# SECTION 4 -- The Kron reduction  (Section II-A of the paper)
# =============================================================================

def kron_reduce(feeder: Feeder, verify: bool = True) -> KronResult:
    """Eliminate zero-injection buses using the Schur complement.

    THE IDEA
    --------
    Split the buses into three groups:
        R  retained   -- buses with a load or generator on them
        Z  zero-inj   -- buses with nothing attached
        S  slack      -- the substation

    Because the Z buses inject no power, their voltages are completely
    determined by the R buses. So we can solve them out algebraically and work
    with a smaller system describing exactly the same physics.

    Writing Y in blocks and eliminating the Z rows gives the paper's eq. (3):

        Y_red = Y_RR  -  Y_RZ @ inverse(Y_ZZ) @ Y_ZR

    Mathematicians call this the "Schur complement". In power systems
    specifically it is called Kron reduction, after Gabriel Kron.

    WHY kappa_Kron MATTERS SO MUCH
    ------------------------------
    The paper's privacy guarantee only holds when

        alpha = ||M_tilde inverse|| * C_star * kappa_Kron * r  <  1/4

    where r is how large a change to the network the mechanism is supposed to
    hide. kappa_Kron multiplies r directly, so if kappa_Kron is 1e25 then r
    must be below about 1e-27 for the theorem to apply at all -- meaning it
    protects against essentially nothing. Getting kappa_Kron down to 3e6 is
    what makes this whole replication viable.
    """
    Y = feeder.Y_full
    R = feeder.load_nodes
    Z = feeder.zero_inj_nodes
    S = feeder.slack_nodes

    if len(Z) == 0:
        raise RuntimeError("No zero-injection nodes found -- check classification.")

    # np.ix_ pulls out a rectangular sub-block: it means "take these rows AND
    # these columns", not "take these individual cells".
    Y_RR = Y[np.ix_(R, R)]
    Y_RZ = Y[np.ix_(R, Z)]
    Y_ZR = Y[np.ix_(Z, R)]
    Y_ZZ = Y[np.ix_(Z, Z)]

    # The condition number measures how trustworthy an inversion is. Roughly,
    # you lose log10(cond) digits of accuracy. Above 1e12 you have lost
    # everything a 64-bit float can offer.
    cond = float(np.linalg.cond(Y_ZZ))
    if cond > 1e10:
        warnings.warn(
            f"Y_ZZ is ill-conditioned (cond = {cond:.2e}). kappa_Kron will be "
            "unreliable. Are merge_switches and prune_stubs both switched on?"
        )

    # np.linalg.solve(A, B) computes inverse(A) @ B, but more accurately and
    # faster than forming the inverse explicitly. Prefer it wherever possible.
    Y_ZZ_inv_Y_ZR = np.linalg.solve(Y_ZZ, Y_ZR)

    Y_red = Y_RR - Y_RZ @ Y_ZZ_inv_Y_ZR      # eq. (3)
    Phi = -Y_ZZ_inv_Y_ZR                     # eq. (2)

    # eq. (7): the substation acts as a fixed current source on the retained
    # buses.
    #
    # SUBTLETY THAT COST US A BUG. The naive reading is b = Y[R, S] @ v_S.
    # On IEEE 123 that evaluates to exactly ZERO, because no load bus is
    # directly adjacent to the substation -- bus 150 reaches the network only
    # through the regulator and a chain of zero-injection buses.
    #
    # But the slack DOES drive the retained buses; it just does so along paths
    # that run through the eliminated set Z. So the coupling has to be pushed
    # through the same Schur complement as everything else:
    #
    #     b = (Y_RS  -  Y_RZ @ inverse(Y_ZZ) @ Y_ZS) @ v_S
    #
    # Getting this wrong is quietly catastrophic. With b = 0 the reduced
    # system has no voltage reference at all, Y_red is nearly singular, and
    # the normalised Jacobian's inverse blows up to about 1e16 -- which then
    # makes alpha astronomically large and Theorem 1 return infinity for
    # every adjacency radius.
    Y_RS = Y[np.ix_(R, S)]
    Y_ZS = Y[np.ix_(Z, S)]
    b = (Y_RS - Y_RZ @ np.linalg.solve(Y_ZZ, Y_ZS)) @ feeder.V_node[S]

    # eq. (4): the Kron amplification factor.
    #   np.linalg.norm(M, 2) is the "operator norm" -- the largest factor by
    #   which the matrix M can stretch any vector.
    op_RZ = np.linalg.norm(Y_RZ, 2)
    op_ZZinv = np.linalg.norm(np.linalg.inv(Y_ZZ), 2)
    kappa = 1.0 + 2.0 * op_RZ * op_ZZinv + (op_RZ * op_ZZinv) ** 2

    # ---- Correctness check ------------------------------------------------
    # The reduction claims v_Z can be recovered from v_R. We already know the
    # true v_Z from OpenDSS, so we can check that claim directly. If this
    # residual is not tiny, something above is wrong.
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
    """Independent sanity check on our node classification.

    Ohm's law in matrix form says  I = Y @ V.  Complex power is S = V * conj(I).

    If we labelled the nodes correctly then S must be essentially ZERO at
    every node we called zero-injection. If it is not, the classification is
    wrong and every number after this point is meaningless.
    """
    I = feeder.Y_full @ feeder.V_node
    S = feeder.V_node * np.conj(I)                # complex power, in VA

    return {
        "max_|S|_zero_inj_kVA": float(np.abs(S[feeder.zero_inj_nodes]).max() / 1e3),
        "max_|S|_load_kVA": float(np.abs(S[feeder.load_nodes]).max() / 1e3),
        "total_slack_kW": float(np.real(S[feeder.slack_nodes]).sum() / 1e3),
    }
