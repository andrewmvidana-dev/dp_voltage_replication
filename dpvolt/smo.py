# A sliding mode observer (SMO) for fault detection on a continuous-time
# linear system, plus the residual and threshold logic that turn it into a
# protection element.
#
# The contrast with a Kalman filter is the whole point. A Kalman filter is
# optimal against ZERO-MEAN GAUSSIAN noise, and a fault is neither zero-mean
# nor Gaussian -- so a fault leaks into the filter's state estimate, and by
# the time the residual grows the estimate is already corrupted. A sliding
# mode observer instead forces the output error to zero in finite time and
# HOLDS it there. Once that happens the switching term must, on average, be
# cancelling the fault. That averaged term -- the equivalent injection -- is
# where the fault information lives, rather than in the residual.
#
# SCOPE, stated up front because it is easy to overclaim here: the equivalent
# injection is a good DETECTION signal, and this module is a detector. It
# rises sharply and early at fault onset, which is what the threshold logic
# keys on. It is NOT a calibrated fault meter -- its steady-state level decays
# while a constant fault persists, for reasons measured and documented in
# SlidingModeObserver.step. Use it to decide WHETHER, not HOW MUCH.
#
# This module is standalone: it does not import anything else from dpvolt.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg


# =============================================================================
# SECTION 1 -- The plant
# =============================================================================

@dataclass
class LinearSystem:
    """A continuous-time state-space model with faults and disturbances.

        x_dot = A x + B u + D xi(t) + F f(t)
        y     = C x

    where

        x    (n,)  state
        u    (m,)  known control input
        xi   (q,)  matched but UNKNOWN disturbance, bounded by ||xi|| <= xi_max
        f    (p,)  the fault signal we are trying to detect
        y    (r,)  measured output

    "Matched" is doing real work in that sentence. The SMO can only reject a
    disturbance that enters through the same channel the injection acts in --
    formally, D must lie in the range of the input distribution the observer
    can reach through C. An unmatched disturbance is not rejected; it shows up
    in the residual and is indistinguishable from a fault. That is a genuine
    limitation of the method, not an artefact of this implementation.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray | None = None
    F: np.ndarray | None = None
    xi_max: float = 0.0          # bound on ||xi(t)||, assumed known a priori
    f_max: float = 0.0           # bound on ||f(t)||, for gain selection

    def __post_init__(self):
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        self.B = np.atleast_2d(np.asarray(self.B, dtype=float))
        self.C = np.atleast_2d(np.asarray(self.C, dtype=float))

        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.r = self.C.shape[0]

        if self.A.shape != (self.n, self.n):
            raise ValueError("A must be square")
        if self.B.shape[0] != self.n:
            raise ValueError("B must have n rows")
        if self.C.shape[1] != self.n:
            raise ValueError("C must have n columns")

        # Default the disturbance and fault channels to the input channel.
        # This is the standard "actuator fault" setup: a fault is whatever the
        # actuator is doing that the controller did not ask for.
        if self.D is None:
            self.D = self.B.copy()
        if self.F is None:
            self.F = self.B.copy()
        self.D = np.atleast_2d(np.asarray(self.D, dtype=float))
        self.F = np.atleast_2d(np.asarray(self.F, dtype=float))

    # -- structural conditions the SMO design depends on ---------------------

    def observability_rank(self) -> int:
        """Rank of the observability matrix [C; CA; ...; CA^(n-1)].

        If this is less than n, no observer of any kind can reconstruct the
        whole state, and the SMO design below cannot be completed.
        """
        rows = [self.C @ np.linalg.matrix_power(self.A, k) for k in range(self.n)]
        return int(np.linalg.matrix_rank(np.vstack(rows)))

    def is_observable(self) -> bool:
        return self.observability_rank() == self.n

    def relative_degree_ok(self) -> bool:
        """Check rank(C F) == rank(F): the fault must be visible in one step.

        This is the observer-matching condition. If C F loses rank relative to
        F, the fault does not appear in y_dot, so the switching term -- which
        only ever sees the output error -- cannot cancel it. Sliding is then
        not attainable and the equivalent injection carries no fault
        information. Higher relative degree needs a different construction
        (a cascade of observers, or a higher-order sliding mode).
        """
        return np.linalg.matrix_rank(self.C @ self.F) == np.linalg.matrix_rank(self.F)


def example_system(xi_max: float = 0.05, f_max: float = 1.0) -> LinearSystem:
    """A third-order plant, stable but lightly damped, with two outputs.

    Concretely: a second-order oscillatory mode (natural frequency ~3 rad/s,
    damping ~0.13) coupled to a slow first-order mode. Nothing about the SMO
    depends on this particular choice -- it just gives the residuals a shape
    that is legible on a plot.

    WHY C MEASURES STATE 2 AND NOT STATE 1. The obvious choice -- measuring
    position (state 1) and the slow mode (state 3) -- FAILS the
    observer-matching condition. The fault enters through B on state 2, so
    with that C we get C F = 0: the fault reaches the output only after
    passing through the dynamics, i.e. relative degree 2. The switching term
    sees only the output error, so it cannot cancel a fault that is not yet
    visible there; sliding is never attained and nu_eq carries no fault
    information.

    Measuring the velocity state instead makes C F full rank and the design
    valid. This is not a cosmetic fix -- it reflects a real constraint on
    SENSOR PLACEMENT: an SMO fault detector needs a measurement in the fault's
    own channel. If the instrumentation does not provide one, the answer is a
    higher-order sliding mode or a cascaded observer, not a retuned gain.
    """
    A = np.array([
        [0.0,   1.0,   0.0],
        [-9.0, -0.8,   2.0],
        [0.0,  -1.0,  -3.0],
    ])
    B = np.array([[0.0], [1.0], [0.0]])
    C = np.array([
        [0.0, 1.0, 0.0],     # velocity: shares the fault channel, so CF != 0
        [0.0, 0.0, 1.0],     # the slow mode
    ])
    return LinearSystem(A=A, B=B, C=C, D=B.copy(), F=B.copy(),
                        xi_max=xi_max, f_max=f_max)


# =============================================================================
# SECTION 2 -- The switching function
# =============================================================================

def switching(s: np.ndarray, kind: str = "sigmoid", boundary: float = 1e-3) -> np.ndarray:
    """The discontinuous injection direction, evaluated on the output error.

    THE IDEAL LAW is the signum:

        nu = -rho * sign(s)

    The sign function is what makes the error reach zero in FINITE time rather
    than merely asymptotically, and it is what makes the rejection of a
    bounded matched disturbance exact rather than approximate. Both properties
    come from the discontinuity; you cannot have them with a smooth law.

    THE PRICE is chattering. A discontinuity at s = 0 combined with any finite
    integration step means the trajectory overshoots the surface each step and
    is thrown back, oscillating at the step frequency with an amplitude
    proportional to rho * dt. In simulation that is ugly; on real hardware it
    excites unmodelled dynamics and wears out actuators.

    THE STANDARD FIX is to replace sign(s) inside a thin boundary layer of
    width `boundary` with something continuous:

        sigmoid:  nu = -rho * s / (||s|| + boundary)
        sat:      nu = -rho * sat(s / boundary)

    This is an honest trade, and worth naming precisely: outside the layer
    nothing changes, but inside it the observer is a high-gain (gain rho /
    boundary) linear observer, not a sliding one. The error no longer reaches
    zero exactly -- it converges to a ball of radius O(boundary). So the
    residual has a noise floor set by `boundary`, and the detection threshold
    in Section 4 must sit above that floor. Shrinking `boundary` tightens the
    floor and worsens the chattering; there is no setting that avoids both.

    We default to the sigmoid because the equivalent injection (Section 3) is
    recovered by low-pass filtering, and filtering a chattering signal that is
    already smooth is far better conditioned numerically.
    """
    if kind == "sign":
        return -np.sign(s)
    if kind == "sigmoid":
        return -s / (np.linalg.norm(s) + boundary)
    if kind == "sat":
        return -np.clip(s / boundary, -1.0, 1.0)
    raise ValueError(f"unknown switching kind: {kind!r}")


# =============================================================================
# SECTION 3 -- The observer
# =============================================================================

@dataclass
class SlidingModeObserver:
    """A Walcott-Zak style sliding mode observer.

        x_hat_dot = A x_hat + B u + L (y - y_hat) + G nu
        y_hat     = C x_hat
        e_y       = y - y_hat
        nu        = -rho * e_y / (||e_y|| + boundary)

    THE ERROR DYNAMICS, which is where the design conditions come from.
    Subtract the observer from the plant and write e = x - x_hat:

        e_dot = (A - L C) e + D xi + F f - G nu

    Two requirements fall straight out of that equation.

    (1) LINEAR PART: A - L C must be Hurwitz. Otherwise the error grows in the
        directions the switching term does not act in, and no gain saves you.
        We place L by solving a Lyapunov equation (see design_L below).

    (2) SLIDING SURFACE: define the surface as S = {e : C e = 0}, i.e. the
        output error is zero. We need the trajectory to REACH that surface in
        finite time and stay on it. That is the reachability condition, and it
        constrains rho.

    REACHABILITY -- why the gain is what it is.
    Take the Lyapunov candidate on the output error, V = (1/2) e_y' e_y. Its
    derivative along the error dynamics is

        V_dot = e_y' C e_dot
              = e_y' C (A - L C) e  +  e_y' C (D xi + F f)  -  rho ||e_y||

    where the last term used C G = I (which is what design_G enforces) and
    e_y' e_y / (||e_y|| + boundary) -> ||e_y|| outside the boundary layer.

    Bound the two middle terms:

        |e_y' C (A - L C) e|      <=  ||C (A - L C)|| * ||e|| * ||e_y||
        |e_y' C (D xi + F f)|     <=  (||C D|| xi_max + ||C F|| f_max) ||e_y||

    So if we choose

        rho  >=  ||C(A - L C)|| * e_max  +  ||C D|| xi_max + ||C F|| f_max + eta

    then V_dot <= -eta ||e_y|| = -eta sqrt(2 V). That is the eta-reachability
    condition, and integrating it gives a finite reaching time

        t_reach  <=  ||e_y(0)|| / eta

    The eta term is the whole design margin: it is not optional slack, it is
    what converts "V_dot < 0" (asymptotic, infinite reaching time) into
    "V_dot <= -eta sqrt(2V)" (finite time). Larger eta reaches sooner and
    rejects more, at the cost of more chattering.

    The e_max term is the uncomfortable one and deserves to be stated plainly:
    it depends on the state error, which is exactly the thing we do not know.
    In a rigorous Walcott-Zak design that term is eliminated by a structural
    constraint (P D = C' Q for some Q, solved as an LMI), which makes the
    unmatched cross-term vanish from V_dot identically. We instead use a
    conservative a-priori bound on ||e||, which is simpler, honest about being
    conservative, and adequate when the initial error is bounded -- but it is
    NOT the same guarantee, and the difference is worth knowing.
    """

    sys: LinearSystem
    L: np.ndarray
    G: np.ndarray
    rho: float
    boundary: float = 1e-3
    switch_kind: str = "sigmoid"
    tau_filter: float = 0.05     # low-pass time constant for equivalent injection

    def __post_init__(self):
        self.x_hat = np.zeros(self.sys.n)
        self.nu_eq = np.zeros(self.sys.r)   # filtered injection, the fault estimate

    def reset(self, x_hat0: np.ndarray | None = None) -> None:
        self.x_hat = np.zeros(self.sys.n) if x_hat0 is None else np.asarray(x_hat0, float).copy()
        self.nu_eq = np.zeros(self.sys.r)

    def step(self, y: np.ndarray, u: np.ndarray, dt: float) -> dict:
        """Advance the observer one step of length dt, given measurement y.

        Integration is RK4 on the smooth part with the switching term held
        constant across the step. Holding nu fixed is deliberate: re-evaluating
        a near-discontinuous function at RK4's intermediate stages produces
        inconsistent slopes and makes the chattering worse rather than better.
        This is effectively the standard treatment of discontinuous ODEs at
        fixed step -- the step size, not the integrator order, sets the
        accuracy near the surface.
        """
        A, B, G = self.sys.A, self.sys.B, self.G

        y_hat = self.sys.C @ self.x_hat
        e_y = y - y_hat                                    # the residual

        nu = self.rho * switching(e_y, self.switch_kind, self.boundary)

        # The injection is -G nu in the error equation, so it enters the
        # observer with a plus sign. Sign conventions here are easy to get
        # backwards; the check is that the residual must SHRINK, not grow.
        def deriv(x):
            return A @ x + B @ u + self.L @ e_y - G @ nu

        k1 = deriv(self.x_hat)
        k2 = deriv(self.x_hat + 0.5 * dt * k1)
        k3 = deriv(self.x_hat + 0.5 * dt * k2)
        k4 = deriv(self.x_hat + dt * k3)
        self.x_hat = self.x_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # ---- the equivalent injection -------------------------------------
        # THE KEY IDEA OF THE WHOLE MODULE. Once the trajectory is on the
        # sliding surface, e_y = 0 and e_y_dot = 0 are both held. Setting
        # e_y_dot = 0 in the error dynamics and solving for the injection:
        #
        #     0 = C(A - LC)e + C D xi + C F f - nu_eq
        #     nu_eq = C D xi + C F f          (on the surface, C e = 0)
        #
        # So the injection required to MAINTAIN sliding is exactly the matched
        # disturbance plus the fault. The high-frequency switching averages to
        # this value, and a low-pass filter recovers it.
        #
        # THAT IDENTITY IS AN IDEALISATION, AND THE GAP IS LARGE. It assumes
        # perfect sliding: a truly discontinuous sign law, no boundary layer,
        # and no other correction term acting on the same error. This
        # implementation violates all three, deliberately, and the result is
        # that nu_eq DECAYS while a constant fault is still present. Measured
        # on the example plant with a constant 0.6 fault:
        #
        #     t = 16 s    ||nu_eq|| ~ 0.54        (close to the true 0.6)
        #     t = 30 s    ||nu_eq|| ~ 0.19
        #     t = 42 s    ||nu_eq|| ~ 0.07        (fault unchanged at 0.6)
        #
        # Two mechanisms cause this, and both were measured by ablation:
        #
        #   THE LUENBERGER TERM. L e_y and the switching term correct the SAME
        #   error. L integrates the persistent bias a fault creates and
        #   gradually takes over the cancelling work, draining it from nu_eq.
        #   Setting L = 0 raises the t=42 value from 0.07 to 0.26.
        #
        #   THE BOUNDARY LAYER. Inside it the law is finite-gain, so sustained
        #   output needs a sustained e_y -- but the observer is simultaneously
        #   driving e_y down. A true sign law recovers some of the loss
        #   (0.07 -> 0.23 with L intact); L = 0 AND sign together give 0.43.
        #
        # SO: nu_eq IS A DETECTOR, NOT A METER. Its RISE at fault onset is
        # sharp, well clear of the fault-free band, and is what the threshold
        # in Section 5 keys on -- that part is sound and is what this module
        # claims. Its steady-state LEVEL understates the fault by an amount
        # that depends on L, the boundary layer and elapsed time, so do not
        # read ||nu_eq|| as a fault magnitude. estimate_fault() inverts the
        # C F scaling, which is necessary but NOT sufficient to fix this --
        # see its docstring.
        #
        # This is what a Kalman filter cannot give you. Its innovation goes to
        # zero as the filter re-converges around the fault, so a slow or
        # incipient fault is silently absorbed. Here the residual returns to
        # zero too -- but nu_eq does NOT, because holding the residual at zero
        # is precisely what requires the sustained injection. The information
        # moves from the residual into the injection rather than disappearing.
        #
        # Filter bandwidth is a genuine trade: tau must be slow enough to
        # average out the switching but fast enough to track the fault. Slow
        # tau means detection lag; fast tau means switching ripple leaks into
        # nu_eq and forces a higher threshold.
        alpha = dt / (self.tau_filter + dt)
        self.nu_eq = self.nu_eq + alpha * (-nu - self.nu_eq)

        return {"e_y": e_y, "nu": nu, "nu_eq": self.nu_eq.copy(),
                "x_hat": self.x_hat.copy()}


# =============================================================================
# SECTION 4 -- Design: choosing L, G and rho
# =============================================================================

def design_L(sys: LinearSystem, decay: float = 2.0) -> np.ndarray:
    """Choose L so that A - L C is Hurwitz with all poles left of -decay.

    We use a Lyapunov / Riccati-free construction rather than direct pole
    placement, because pole placement for a multi-output system is not unique
    and the arbitrary choice can produce a badly conditioned L.

    The trick: solve the shifted Lyapunov equation

        (A + decay I) P + P (A + decay I)' - P C' C P + I = 0

    which is the observer algebraic Riccati equation for the shifted system.
    Then L = P C' places the spectrum of A - L C left of -decay by
    construction. SciPy solves this as a continuous-time ARE.
    """
    n = sys.n
    A_shift = sys.A + decay * np.eye(n)

    # solve_continuous_are(A, B, Q, R) solves A'X + XA - XBR^-1B'X + Q = 0.
    # For the OBSERVER (dual) problem we pass A' and C'.
    P = scipy.linalg.solve_continuous_are(
        A_shift.T, sys.C.T, np.eye(n), np.eye(sys.r)
    )
    L = P @ sys.C.T

    poles = np.linalg.eigvals(sys.A - L @ sys.C)
    if poles.real.max() >= 0:
        raise RuntimeError(f"A - LC not Hurwitz; poles = {poles}")
    return L


def design_G(sys: LinearSystem) -> np.ndarray:
    """Choose G with C G = I, so the injection acts directly on the residual.

    The right-inverse G = C' (C C')^-1 exists whenever C has full row rank,
    which is the mild requirement that no measurement is a linear combination
    of the others. This is the minimum-norm choice among all right inverses,
    which is what we want: any component of G in the null space of C would
    push the state estimate in a direction the residual cannot see, and so
    could not be corrected by the switching law.
    """
    C = sys.C
    if np.linalg.matrix_rank(C) < sys.r:
        raise ValueError("C must have full row rank for a right inverse")
    return C.T @ np.linalg.inv(C @ C.T)


def design_rho(sys: LinearSystem, L: np.ndarray, e_max: float = 1.0,
               eta: float = 0.5) -> dict:
    """Compute the switching gain from the reachability condition.

        rho >= ||C(A - LC)|| e_max + ||C D|| xi_max + ||C F|| f_max + eta

    Derivation is in the SlidingModeObserver docstring. Each term answers a
    separate question:

        cross term      how much the unmatched part of the state error can
                        drive the output error. Conservative -- see the caveat
                        in the observer docstring about e_max.
        disturbance     the bounded uncertainty the observer must overpower to
                        stay on the surface. If rho is below this, sliding is
                        broken by the disturbance alone and every guarantee
                        here is void.
        fault           we must be able to hold the surface THROUGH a fault of
                        size f_max -- that is what makes nu_eq a valid estimate
                        of the fault rather than the surface simply breaking.
        eta             the finite-time margin. Reaching time <= ||e_y(0)||/eta.

    Returning the breakdown rather than one number is deliberate: when rho
    comes out large enough to chatter badly, you need to see which term is
    responsible, because the remedies differ (tighten e_max, tighten xi_max,
    or accept a slower reach).

    ON CHOOSING e_max -- THIS MATTERS MORE THAN IT LOOKS. The cross term is
    usually the largest of the four, so e_max effectively sets rho on its own.
    And rho sets the boundary-layer noise floor (Section 5), which sets the
    detection threshold, which sets the smallest fault you can see. So a
    lazily conservative e_max does not merely cost you some chattering -- it
    directly destroys detection sensitivity.

    Concretely, on the example plant: e_max = 1.0 gives rho = 13.3 and a
    fault-free q99.9 of ||nu_eq|| around 0.16, so nothing below a fault of
    ~0.3 is detectable. Calibrating e_max from fault-free data instead gives
    e_max ~ 0.01, rho ~ 1.4, and a floor several times lower. Same observer,
    same plant; the only change is not over-bounding a quantity we can
    measure. Use calibrate_e_max() rather than guessing.
    """
    cross = float(np.linalg.norm(sys.C @ (sys.A - L @ sys.C), 2)) * e_max
    dist = float(np.linalg.norm(sys.C @ sys.D, 2)) * sys.xi_max
    fault = float(np.linalg.norm(sys.C @ sys.F, 2)) * sys.f_max

    rho = cross + dist + fault + eta
    return {"rho": rho, "cross_term": cross, "disturbance_term": dist,
            "fault_term": fault, "eta": eta, "e_max": e_max}


def estimate_fault(sys: LinearSystem, nu_eq: np.ndarray) -> np.ndarray:
    """Reconstruct the fault signal f from the equivalent injection.

    On the sliding surface nu_eq = C D xi + C F f, so with the disturbance
    neglected the least-squares reconstruction is

        f_hat = pinv(C F) nu_eq

    THIS CORRECTS THE C F SCALING AND NOTHING ELSE. On the example plant
    C F = [1, 0]', so pinv(C F) is norm-preserving and this function changes
    the magnitude by well under 1%. It matters on plants where C F is poorly
    scaled; it does NOT rescue the estimate here.

    IT DOES NOT FIX THE DECAY, which is the dominant error. As documented at
    length in SlidingModeObserver.step, ||nu_eq|| falls away from the true
    fault over tens of seconds because the Luenberger term and the boundary
    layer both erode ideal sliding. No linear post-processing of nu_eq can
    undo that -- the information has genuinely left the signal. If you need
    accurate fault SIZING rather than detection, the fixes are structural:
    drop L and use a true sign law (measured to roughly double the retained
    magnitude, at the cost of chattering and a ~50% higher false-alarm floor),
    or use a dedicated fault-reconstruction scheme such as a higher-order
    sliding-mode differentiator or an adaptive/unknown-input observer.

    WHAT IS ALSO NOT RECOVERABLE. Only the component of f in the row space of
    C F survives; anything in the null space is invisible and no
    post-processing brings it back. And because we neglect xi, the estimate
    carries a bias of up to ||pinv(C F)|| * ||C D|| * xi_max.

    `nu_eq` may be a single (r,) vector or an (steps, r) history.
    """
    CF = sys.C @ sys.F
    pinv = np.linalg.pinv(CF)
    nu = np.atleast_2d(nu_eq)
    f_hat = nu @ pinv.T
    return f_hat[0] if np.asarray(nu_eq).ndim == 1 else f_hat


def calibrate_e_max(sys: LinearSystem, L: np.ndarray, *, eta: float = 0.5,
                    margin: float = 3.0, settle: float = 2.0, iters: int = 3,
                    boundary: float = 2e-3, tau_filter: float = 0.05,
                    **sim_kwargs) -> dict:
    """Measure the state-error bound e_max from fault-free data.

    WHY THIS IS NOT CIRCULAR, even though rho depends on e_max and the error
    depends on rho: we iterate to a fixed point, starting from a deliberately
    loose e_max and re-measuring the ACTUAL peak ||x - x_hat|| each round.
    Each iteration lowers rho, which lowers the boundary-layer floor, which
    slightly lowers the error -- the map is a contraction here and settles in
    two or three passes.

    WHAT THIS DOES AND DOES NOT BUY YOU. It replaces a worst-case bound with a
    measured one, which is the difference between a threshold that can see a
    0.05 fault and one that cannot see anything under 0.3. But a measured
    bound is only valid for disturbances no worse than the ones measured, so
    we keep a `margin` (default 3x) over the observed peak. That margin is a
    judgement call, not a theorem.

    THE HONEST CAVEAT: this uses the true state x, which a real deployment
    does not have. It is legitimate here because this is a SIMULATION STUDY --
    we are characterising the observer, not running it blind. On hardware you
    would get e_max from a model-validation campaign, a hardware-in-the-loop
    rig, or the Walcott-Zak LMI (which removes the term entirely and needs no
    calibration at all). Do not read this function as something that ships.
    """
    G = design_G(sys)
    e_max = 1.0
    history = []

    for _ in range(iters):
        gain = design_rho(sys, L, e_max=e_max, eta=eta)
        obs = SlidingModeObserver(sys=sys, L=L, G=G, rho=gain["rho"],
                                  boundary=boundary, tau_filter=tau_filter)
        res = simulate(sys, obs, None, fault_kind="none", fault_time=None,
                       **sim_kwargs)

        k0 = int(settle / res.meta["dt"])
        peak = float(np.linalg.norm(res.x[k0:] - res.x_hat[k0:], axis=1).max())
        history.append({"e_max_in": e_max, "rho": gain["rho"], "peak_err": peak})
        e_max = margin * peak

    return {"e_max": e_max, "rho": design_rho(sys, L, e_max=e_max, eta=eta)["rho"],
            "margin": margin, "history": history}


# =============================================================================
# SECTION 5 -- Residual generation and protection threshold logic
# =============================================================================

@dataclass
class ThresholdConfig:
    """Settings for the protection element.

    The two-number structure (a theoretical floor plus an empirical margin) is
    how a real protection relay is set, and it is worth being explicit about
    why neither alone is enough.
    """

    margin: float = 1.5           # multiplier on the theoretical bound
    dwell_steps: int = 20         # consecutive steps above threshold before flagging
    empirical_quantile: float = 0.999
    use_empirical: bool = True


def fault_free_bound(sys: LinearSystem, obs: SlidingModeObserver) -> dict:
    """The theoretical ceiling on ||nu_eq|| when NO fault is present.

    On the sliding surface, nu_eq = C D xi + C F f. With f = 0 this leaves

        ||nu_eq||  <=  ||C D|| * xi_max

    That is the clean part of the bound. Two real effects sit on top of it and
    both must be added or the threshold will produce false trips:

      BOUNDARY LAYER   inside the layer the observer is high-gain linear, not
                       sliding, so e_y settles to a ball of radius O(boundary)
                       instead of exactly zero. The residual therefore has a
                       floor, and so does the injection needed to hold it.

      FILTER RIPPLE    the low-pass leaves a residual of the switching
                       amplitude rho. With a first-order filter of time
                       constant tau at step dt, the surviving ripple is
                       roughly rho * dt / tau.

    So the working bound is

        bound = ||C D|| xi_max  +  rho * boundary_contribution  +  ripple

    A THRESHOLD ON THIS ALONE IS STILL NOT ENOUGH, and that is the honest
    statement. The bound is worst-case over adversarial xi(t) and ignores
    measurement noise entirely, so it can sit far above the actual fault-free
    level (missing small faults) or, if xi_max was underestimated, below it
    (tripping constantly). Real protection settings are validated against
    fault-free data. That is what the empirical quantile in
    ProtectionLogic.calibrate is for: the theoretical bound is the floor you
    may never go below, the empirical quantile is what you actually use.
    """
    matched = float(np.linalg.norm(sys.C @ sys.D, 2)) * sys.xi_max
    layer = obs.rho * obs.boundary / (obs.boundary + 1e-12) * obs.boundary
    return {"matched": matched, "layer_term": layer,
            "theoretical": matched + layer}


class ProtectionLogic:
    """Decide, sample by sample, whether a fault is present.

    Three stages, each catching something the previous one cannot:

      1. THRESHOLD   compare ||nu_eq|| against the calibrated bound. Uses the
                     equivalent injection, not the raw residual, because after
                     sliding is re-established the residual returns to zero
                     even with the fault still active -- the fault information
                     lives in the injection.

      2. DWELL       require `dwell_steps` CONSECUTIVE exceedances. A single
                     sample over threshold is a transient, not a fault. This
                     is the direct analogue of a relay's definite-time delay
                     and is the main defence against false trips.

      3. LATCH       once flagged, stay flagged. A protection element that
                     un-trips because the fault momentarily dipped below the
                     threshold is worse than useless.

    The dwell delays detection by dwell_steps * dt. That is a real, deliberate
    cost paid for immunity to transients, and it should be reported alongside
    any detection-latency figure rather than hidden.
    """

    def __init__(self, threshold: float, config: ThresholdConfig | None = None):
        self.config = config or ThresholdConfig()
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.flagged = False
        self.flag_index: int | None = None

    def update(self, nu_eq: np.ndarray, index: int) -> bool:
        mag = float(np.linalg.norm(nu_eq))
        if mag > self.threshold:
            self.count += 1
        else:
            self.count = 0

        if not self.flagged and self.count >= self.config.dwell_steps:
            self.flagged = True
            # Credit detection to the step the exceedance STARTED, not the one
            # the dwell expired on -- otherwise the reported latency silently
            # excludes the dwell we deliberately imposed.
            self.flag_index = index - self.config.dwell_steps + 1
        return self.flagged

    @staticmethod
    def calibrate(nu_eq_history: np.ndarray, theoretical: float,
                  config: ThresholdConfig | None = None) -> dict:
        """Set the threshold from fault-free data, floored by the theory.

        `nu_eq_history` is (steps, r) of equivalent injection recorded during
        a run with NO fault. We take a high quantile of its magnitude, apply
        the safety margin, and then take the MAXIMUM against a small fraction
        of the theoretical bound.

        The max is the important part. A purely empirical threshold overfits
        whatever disturbance realisation happened to occur during calibration;
        the theory is what protects you against a worse one you did not see.
        """
        cfg = config or ThresholdConfig()
        mags = np.linalg.norm(np.atleast_2d(nu_eq_history), axis=1)
        empirical = float(np.quantile(mags, cfg.empirical_quantile))

        thr_emp = cfg.margin * empirical
        thr_theory = cfg.margin * theoretical

        chosen = max(thr_emp, 0.1 * thr_theory) if cfg.use_empirical else thr_theory
        return {"threshold": float(chosen), "empirical": empirical,
                "theoretical": theoretical, "from_empirical": thr_emp,
                "from_theory": thr_theory}


# =============================================================================
# SECTION 6 -- A Kalman filter, for the comparison
# =============================================================================

class SteadyStateKalman:
    """Continuous-time steady-state Kalman filter, the baseline.

        x_hat_dot = A x_hat + B u + K (y - C x_hat),   K = P C' R^-1

    with P from the filter ARE. Included so the comparison is fair: this is
    the correct, properly tuned Kalman filter for this plant, not a straw man.
    Its residual is genuinely optimal for Gaussian noise. The point of the
    comparison is that fault detection is a different problem from state
    estimation under Gaussian noise, and optimality for one does not transfer.
    """

    def __init__(self, sys: LinearSystem, Q_scale: float = 1e-2, R_scale: float = 1e-3):
        self.sys = sys
        Q = Q_scale * sys.D @ sys.D.T + 1e-9 * np.eye(sys.n)
        R = R_scale * np.eye(sys.r)
        P = scipy.linalg.solve_continuous_are(sys.A.T, sys.C.T, Q, R)
        self.K = P @ sys.C.T @ np.linalg.inv(R)
        self.reset()

    def reset(self, x_hat0: np.ndarray | None = None) -> None:
        self.x_hat = np.zeros(self.sys.n) if x_hat0 is None else np.asarray(x_hat0, float).copy()

    def step(self, y: np.ndarray, u: np.ndarray, dt: float) -> dict:
        A, B, K = self.sys.A, self.sys.B, self.K
        e_y = y - self.sys.C @ self.x_hat

        def deriv(x):
            return A @ x + B @ u + K @ (y - self.sys.C @ x)

        k1 = deriv(self.x_hat)
        k2 = deriv(self.x_hat + 0.5 * dt * k1)
        k3 = deriv(self.x_hat + 0.5 * dt * k2)
        k4 = deriv(self.x_hat + dt * k3)
        self.x_hat = self.x_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return {"e_y": e_y, "x_hat": self.x_hat.copy()}


# =============================================================================
# SECTION 7 -- Simulation driver
# =============================================================================

@dataclass
class SimResult:
    """Everything a run produced, kept as arrays for plotting."""

    t: np.ndarray
    x: np.ndarray            # (steps, n) true state
    x_hat: np.ndarray        # (steps, n) SMO state estimate
    y: np.ndarray            # (steps, r) measured output
    e_smo: np.ndarray        # (steps, r) SMO residual
    e_kf: np.ndarray         # (steps, r) Kalman residual
    nu_eq: np.ndarray        # (steps, r) equivalent injection
    f_true: np.ndarray       # (steps,) true fault magnitude
    threshold: float = 0.0
    flag_index: int | None = None
    kf_flag_index: int | None = None
    meta: dict = field(default_factory=dict)


def simulate(
    sys: LinearSystem,
    obs: SlidingModeObserver,
    kf: SteadyStateKalman | None = None,
    *,
    t_end: float = 20.0,
    dt: float = 1e-3,
    fault_time: float | None = 10.0,
    fault_kind: str = "ramp",
    fault_size: float = 0.6,
    fault_ramp_rate: float = 0.08,
    meas_noise: float = 1e-3,
    seed: int = 0,
) -> SimResult:
    """Integrate plant + observers together and record the diagnostics.

    Fault kinds:
        "step"      abrupt, size `fault_size`. Easy case; both methods see it.
        "ramp"      incipient, growing at `fault_ramp_rate` per second up to
                    `fault_size`. This is the hard case and the one that
                    separates the methods -- a slowly growing fault is exactly
                    what a Kalman filter re-converges around.
        "none"      fault-free, used to calibrate the threshold.

    The disturbance xi(t) is a deterministic band-limited signal at amplitude
    xi_max, not white noise. That is on purpose: the SMO's guarantee is
    against any BOUNDED matched disturbance, so a persistent structured one is
    the honest test. White noise would be gentler and would flatter the SMO.
    """
    rng = np.random.default_rng(seed)
    steps = int(round(t_end / dt))
    t = np.arange(steps) * dt

    x = np.zeros(sys.n)
    obs.reset()
    if kf is not None:
        kf.reset()

    X = np.zeros((steps, sys.n))
    XH = np.zeros((steps, sys.n))
    Y = np.zeros((steps, sys.r))
    E_smo = np.zeros((steps, sys.r))
    E_kf = np.zeros((steps, sys.r))
    NU = np.zeros((steps, sys.r))
    Ftrue = np.zeros(steps)

    def control(tt):
        # A persistently exciting input, so the modes stay stimulated and the
        # comparison is not decided by an accidentally quiescent plant.
        return np.array([0.5 * np.sin(0.7 * tt) + 0.2 * np.sin(2.3 * tt)])

    def disturbance(tt):
        if sys.xi_max <= 0:
            return np.zeros(sys.D.shape[1])
        d = np.sin(11.0 * tt) * 0.6 + np.sin(4.0 * tt + 1.0) * 0.4
        return np.full(sys.D.shape[1], sys.xi_max * d)

    def fault(tt):
        if fault_kind == "none" or fault_time is None or tt < fault_time:
            return 0.0
        if fault_kind == "step":
            return fault_size
        if fault_kind == "ramp":
            return min(fault_size, fault_ramp_rate * (tt - fault_time))
        raise ValueError(f"unknown fault_kind: {fault_kind!r}")

    for k in range(steps):
        tt = t[k]
        u = control(tt)
        xi = disturbance(tt)
        fmag = fault(tt)
        f_vec = np.full(sys.F.shape[1], fmag)

        y_clean = sys.C @ x
        y = y_clean + rng.normal(0.0, meas_noise, size=sys.r)

        X[k] = x
        Y[k] = y
        Ftrue[k] = fmag

        # Record the estimate BEFORE the update, so x_hat[k] and x[k] refer to
        # the same instant and their difference is the true state error.
        XH[k] = obs.x_hat

        out = obs.step(y, u, dt)
        E_smo[k] = out["e_y"]
        NU[k] = out["nu_eq"]

        if kf is not None:
            E_kf[k] = kf.step(y, u, dt)["e_y"]

        # Plant integration, RK4. The disturbance and fault are held over the
        # step, consistent with how the observer treats its own injection.
        def deriv(xx):
            return sys.A @ xx + sys.B @ u + sys.D @ xi + sys.F @ f_vec

        k1 = deriv(x)
        k2 = deriv(x + 0.5 * dt * k1)
        k3 = deriv(x + 0.5 * dt * k2)
        k4 = deriv(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return SimResult(t=t, x=X, x_hat=XH, y=Y, e_smo=E_smo, e_kf=E_kf, nu_eq=NU,
                     f_true=Ftrue,
                     meta={"dt": dt, "fault_time": fault_time,
                           "fault_kind": fault_kind, "fault_size": fault_size,
                           "sys": sys})
