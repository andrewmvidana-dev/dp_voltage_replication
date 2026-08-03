# The DP mechanism that fits the load model, plus Theorem 1 -- the bound
# saying private loads alone are enough to hide the topology Y, with no extra
# noise on the admittance matrix.
#
# We substitute a plain Gaussian mechanism for the paper's DP-GMM
# (arXiv:2506.03467), which has no public implementation. Legitimate under
# their Section III-C: Theorem 1 only needs the released loads to be
# log-normal with a known Sigma. We lose fit quality, not validity, so expect
# the same ORDERING of methods in Figures 2 and 3, not the same values.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# SECTION 1 -- The DP mechanism that fits the load model
# =============================================================================

def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Noise scale for the classical Gaussian mechanism.

        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Read it as: more sensitivity means more noise; more privacy (smaller
    epsilon) means more noise. This is the standard formula from Dwork & Roth
    and is what the paper uses for its baselines (their Table III).

    It is slightly loose. Balle & Wang's "analytic Gaussian mechanism" gets
    the same guarantee with less noise. Swapping it in would help every
    method equally, so it does not change any comparison -- but it is an easy
    improvement to mention.
    """
    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("need epsilon > 0 and 0 < delta < 1")
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


@dataclass
class DPFitReport:
    """Diagnostics from privately fitting one class's load model."""

    epsilon: float
    delta: float
    n_records: int
    log_range: float          # width of the log-load box, per coordinate
    sigma_mu: float           # noise standard deviation added to the mean
    sigma_cov: float          # noise standard deviation added to the covariance
    eig_clipped: int          # eigenvalues that had to be lifted to stay valid
    records_clipped: int      # records whose L2 norm exceeded the clip bound
    clip_norm: float          # the L2 clipping bound actually used
    kl_to_true: float         # KL divergence from the true fit, for reference


def dp_fit_class(
    log_data: np.ndarray,
    lo: float,
    hi: float,
    epsilon: float,
    delta: float,
    rng: np.random.Generator,
    eig_floor_ratio: float = 1e-3,
    clip_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, DPFitReport]:
    """Privately estimate the mean and covariance of one class's log-loads.

    Parameters
    ----------
    log_data   (m, T) array. Each row is one bus-day's log-load trajectory.
    lo, hi     the log-load bounds, from the class margins.
    epsilon    total privacy budget for this class.
    delta      total failure probability for this class.
    clip_norm  L2 bound applied to each centred record. See below. If None,
               the worst-case value sqrt(T) * R / 2 is used, which is correct
               but very loose.

    HOW THE BUDGET IS SPLIT
    -----------------------
    We release two things -- a mean and a covariance -- so by composition each
    gets half the budget. That halving is real and unavoidable: it is why a
    method that releases fewer quantities can afford less noise.

    THE SENSITIVITIES
    -----------------
    We use bounded-record adjacency: one row of log_data may be replaced by
    any other row inside the box. Write R = hi - lo for the box width.

    For the MEAN, replacing one record moves each coordinate by at most R/m,
    so the L2 sensitivity across T coordinates is sqrt(T) * R / m.

    For the COVARIANCE, we first CENTRE each record and clip its L2 norm to
    at most `clip_norm` = C. Replacing one clipped record then changes the
    matrix by at most 2 * C^2 / m in Frobenius norm.

    WHY CLIPPING IS NECESSARY, NOT OPTIONAL
    ---------------------------------------
    Without it, C is forced to its worst case sqrt(T) * R / 2. On our data
    that is about 22.8, so C^2 is about 519, and at epsilon = 1 the noise
    standard deviation lands around 14.8 -- against true covariance entries of
    roughly 0.1. The result is total destruction: we measured 49 of 96
    eigenvalues clipped and a KL divergence of 363,589.

    In practice a centred record has norm around 3, not 22.8, because the
    worst case assumes every one of the 96 coordinates sits at an extreme
    simultaneously. Clipping at a realistic C cuts the sensitivity by the
    ratio squared -- more than an order of magnitude.

    This is also the honest explanation for why the paper's experiments use
    epsilon between 25 and 200 rather than the single digits you see in the
    DP literature. Privately estimating a 96-by-96 covariance is simply
    expensive, and no amount of cleverness makes it cheap.

    A CAVEAT TO STATE OUT LOUD
    --------------------------
    C is a hyperparameter and must be chosen WITHOUT looking at the private
    data, or its selection leaks too. Picking it from a public load study, or
    spending a small slice of budget to estimate it, are both defensible.
    Tuning it against your own data until the figures look good is not.

    THE EIGENVALUE REPAIR
    ---------------------
    Adding symmetric noise to a covariance matrix can push some of its
    eigenvalues negative, which is not a valid covariance -- you cannot sample
    from it. So we clip the eigenvalues up to a small positive floor.

    This repair is FREE under differential privacy, because of the
    post-processing property: we only touch the already-noised matrix, never
    the real data. It is worth saying explicitly, because it looks like
    cheating and is not.
    """
    m, T = log_data.shape
    R = float(hi - lo)

    eps_half, delta_half = epsilon / 2.0, delta / 2.0

    # ---- private mean -----------------------------------------------------
    sens_mu = np.sqrt(T) * R / m
    sigma_mu = gaussian_sigma(sens_mu, eps_half, delta_half)
    mu_true = log_data.mean(axis=0)
    mu_dp = mu_true + rng.normal(0.0, sigma_mu, size=T)

    # ---- private covariance ------------------------------------------------
    if clip_norm is None:
        clip_norm = np.sqrt(T) * R / 2.0          # worst case, very loose

    # Centre on the ALREADY-PRIVATE mean, so this step consumes no extra
    # budget (post-processing), then clip each record's L2 norm to C.
    centred = log_data - mu_dp
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    centred = centred * scale
    n_clipped_records = int((scale < 1.0).sum())

    cov_true = (centred.T @ centred) / m + 1e-12 * np.eye(T)

    sens_cov = 2.0 * clip_norm ** 2 / m
    sigma_cov = gaussian_sigma(sens_cov, eps_half, delta_half)

    # Symmetric Gaussian noise: draw a full matrix, then average it with its
    # own transpose so the result is symmetric, as a covariance must be.
    noise = rng.normal(0.0, sigma_cov, size=(T, T))
    cov_dp = cov_true + (noise + noise.T) / np.sqrt(2.0)

    # ---- repair the covariance (post-processing, free) --------------------
    cov_dp = (cov_dp + cov_dp.T) / 2.0            # enforce exact symmetry
    evals, evecs = np.linalg.eigh(cov_dp)
    floor = eig_floor_ratio * max(float(np.trace(cov_true)) / T, 1e-12)
    n_clipped = int((evals < floor).sum())
    evals = np.maximum(evals, floor)
    cov_dp = evecs @ np.diag(evals) @ evecs.T

    report = DPFitReport(
        epsilon=epsilon, delta=delta, n_records=m, log_range=R,
        sigma_mu=sigma_mu, sigma_cov=sigma_cov, eig_clipped=n_clipped,
        records_clipped=n_clipped_records, clip_norm=float(clip_norm),
        kl_to_true=gaussian_kl(mu_dp, cov_dp, mu_true, cov_true),
    )
    return mu_dp, cov_dp, report


def gaussian_kl(mu0, cov0, mu1, cov1) -> float:
    """KL divergence from Normal(mu0, cov0) to Normal(mu1, cov1).

    A single number measuring how far the private model has drifted from the
    true one. Lower is better. The paper's DP-GMM is built specifically to
    minimise this quantity, so it is the fair yardstick for how much our
    simpler substitute costs us.
    """
    T = len(mu0)
    cov1_inv = np.linalg.inv(cov1)
    diff = mu1 - mu0

    _, logdet0 = np.linalg.slogdet(cov0)
    _, logdet1 = np.linalg.slogdet(cov1)

    return float(
        0.5 * (np.trace(cov1_inv @ cov0)
               + diff @ cov1_inv @ diff
               - T
               + logdet1 - logdet0)
    )


# =============================================================================
# SECTION 2 -- Theorem 1
# =============================================================================

@dataclass
class PrivacyBound:
    """Every intermediate quantity in Theorem 1, kept for inspection.

    Keeping the pieces separate rather than returning one number is
    deliberate: when epsilon comes out large you need to see WHICH term is
    responsible, and the paper's discussion turns entirely on that.
    """

    epsilon: float           # the headline result, eq. (32)
    delta: float
    r: float                 # adjacency radius on Y_full
    alpha: float             # admissibility parameter, eq. (30)
    admissible: bool         # is alpha < 1/4? if not, the theorem says nothing
    C_star: float
    jacobian_term: float     # the floor that load noise cannot cross
    psi_bar: float           # whitened-shift bound, eq. (34)
    tail_term: float         # psi_bar * tau(delta)
    bias_term: float         # everything deterministic, eq. (33)
    tau: float               # chi-squared tail factor, eq. (35)
    d_ell: dict              # per-class sensitivity constant, eq. (23)
    gamma_ell: dict          # per-class precision sum, eq. (24)


def theorem1(
    *,
    Sigma_by_class: dict,        # class -> (T, T) covariance of the DP model
    size_by_class: dict,         # class -> number of buses in it
    p_min_by_class: dict,        # class -> lower load margin, per-unit
    n: int,                      # number of retained buses
    T: int,                      # time steps per release
    d_max: int,                  # maximum node degree in the network
    kappa_kron: float,
    r: float,                    # adjacency radius on Y_full, Frobenius
    delta: float,
    M_inv_norm: float,           # ||M~^-1||, from Monte Carlo calibration
    V_min: float = 0.95,
    V_max: float = 1.05,
) -> PrivacyBound:
    """Evaluate the paper's (epsilon, delta) bound, eq. (32) to (35).

    EVERY INPUT MUST BE IN PER-UNIT. The formulas mix voltages, admittances
    and loads together, so a stray volts-versus-per-unit error silently
    changes epsilon by orders of magnitude and nothing will warn you.

    THE STRUCTURE OF THE BOUND
    --------------------------
    epsilon has three pieces, and they behave very differently:

      JACOBIAN TERM   how much a change in Y distorts the volume element of
                      the power-flow map. Depends on r but NOT on the load
                      noise. This is a FLOOR: no amount of extra noise in the
                      load model can push epsilon below it. The only lever is
                      to shrink r.

      BIAS TERM       deterministic, scales with (kappa_Kron * r)^2.

      TAIL TERM       the random part, psi_bar * tau(delta). This is the piece
                      the load noise controls. A noisier load model has a
                      larger Sigma, hence a smaller precision sum gamma, hence
                      a smaller psi_bar. So MORE load noise buys MORE topology
                      privacy -- which is the paper's central mechanism.
    """
    # ---- eq. (30): the admissibility parameter ----------------------------
    C_star = np.sqrt(2.0) * (1.0 + np.sqrt(n) * V_max / V_min)
    alpha = M_inv_norm * C_star * kappa_kron * r
    admissible = alpha < 0.25

    # ---- eq. (31): the Jacobian term --------------------------------------
    # If alpha >= 1/4 this expression goes negative or explodes, which is the
    # theorem's way of saying it does not apply. We return infinity so that a
    # caller who ignores `admissible` still gets an obviously wrong number
    # rather than a plausible-looking one.
    if admissible:
        jac = T * np.sqrt(n) * alpha * (2.0 + alpha) / (2.0 * (1.0 - 4.0 * alpha))
    else:
        jac = np.inf

    # ---- eq. (23) and (24): per-class constants ---------------------------
    d_ell, gamma_ell = {}, {}
    for label, Sigma in Sigma_by_class.items():
        # eq. (23). Note p_min in the DENOMINATOR: lightly loaded buses
        # inflate the bound, which is why the paper tells you to set the
        # margins as tight as feasibility allows.
        d_ell[label] = V_max ** 2 * np.sqrt(d_max) / p_min_by_class[label]

        # eq. (24): the "precision sum", the sum of the absolute values of
        # every entry of the inverse covariance. A NOISIER load model has a
        # bigger Sigma, so a smaller inverse, so a smaller gamma, so a
        # smaller epsilon. This is where private loads buy topology privacy.
        Sigma_inv = np.linalg.inv(Sigma)
        gamma_ell[label] = float(np.abs(Sigma_inv).sum())

    # ---- eq. (34): the uniform whitened-shift bound -----------------------
    psi_sq = (kappa_kron * r) ** 2 * sum(
        d_ell[l] ** 2 * gamma_ell[l] for l in Sigma_by_class
    )
    psi_bar = float(np.sqrt(psi_sq))

    # ---- eq. (35): the chi-squared tail factor ----------------------------
    # This comes from a concentration inequality (Laurent-Massart) on the
    # whitened load draws. For nT much larger than log(1/delta) it is
    # approximately sqrt(nT).
    nT = n * T
    log_inv_delta = np.log(1.0 / delta)
    tau = float(np.sqrt(nT + 2.0 * np.sqrt(nT * log_inv_delta) + 2.0 * log_inv_delta))

    # ---- eq. (33): the deterministic bias ---------------------------------
    beta_sum = 0.0
    for label, Sigma in Sigma_by_class.items():
        beta_sum += (
            d_ell[label]
            * np.sqrt(gamma_ell[label] * size_by_class[label])
            * np.sqrt(np.ones(T) @ Sigma @ np.ones(T))
        )
    bias = jac + 0.5 * psi_sq + kappa_kron * r * beta_sum

    # ---- eq. (32): put it together ----------------------------------------
    epsilon = float(bias + psi_bar * tau)

    return PrivacyBound(
        epsilon=epsilon, delta=delta, r=r, alpha=float(alpha),
        admissible=bool(admissible), C_star=float(C_star),
        jacobian_term=float(jac), psi_bar=psi_bar,
        tail_term=float(psi_bar * tau), bias_term=float(bias), tau=tau,
        d_ell=d_ell, gamma_ell=gamma_ell,
    )


def solve_for_r(target_epsilon: float, *, bracket=(1e-14, 1e-2), **kwargs) -> float:
    """Find the largest adjacency radius r that still meets a target epsilon.

    epsilon increases monotonically with r, so we can bisect. This is the
    inverse question to theorem1() and usually the more useful one: a utility
    knows what epsilon it is willing to spend and wants to know how broad a
    class of network changes that actually protects.
    """
    lo, hi = bracket

    def eps_at(r):
        bound = theorem1(r=r, **kwargs)
        return bound.epsilon if bound.admissible else np.inf

    if eps_at(lo) > target_epsilon:
        return 0.0                     # unreachable even at the smallest r

    for _ in range(200):               # bisection, plenty of iterations
        mid = np.sqrt(lo * hi)         # geometric midpoint, since r spans decades
        if eps_at(mid) <= target_epsilon:
            lo = mid
        else:
            hi = mid
    return lo


# =============================================================================
# SECTION 3 -- Monte Carlo calibration of ||M~^-1||   (the paper's Remark 2)
# =============================================================================

def normalised_jacobian(v: np.ndarray, Y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build M~, the normalised power-flow Jacobian of eq. (29).

        M~ = [[ diag(s / v^2),   conj(Y)      ],
              [ Y,               diag(conj(s) / conj(v)^2) ]]

    where s_i = v_i * (conj(Y) conj(v) + conj(b))_i is the complex power
    injected at bus i.

    This comes from factoring a diagonal voltage matrix out of the full
    Jacobian, which is what makes the paper's bound depend on the network
    rather than on the particular operating point.

    Everything here is per-unit.
    """
    s = v * (np.conj(Y) @ np.conj(v) + np.conj(b))
    n = len(v)

    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[:n, :n] = np.diag(s / v ** 2)
    M[:n, n:] = np.conj(Y)
    M[n:, :n] = Y
    M[n:, n:] = np.diag(np.conj(s) / np.conj(v) ** 2)
    return M


def calibrate_M_inv(
    voltages: np.ndarray,
    Y: np.ndarray,
    b: np.ndarray,
    quantile: float = 0.99,
    confidence: float = 0.95,
) -> dict:
    """Estimate ||M~^-1|| empirically, following the paper's Remark 2.

    WHY NOT USE THE CLOSED FORM?
    ----------------------------
    Appendix E gives an analytical upper bound on ||M~^-1||, but it is a
    worst case over EVERY voltage in the whole admissible set, so it is very
    conservative. The paper says plainly that its own evaluation uses this
    Monte Carlo calibration instead.

    HOW IT WORKS
    ------------
    Draw many trajectories from the private load model, solve power flow for
    each, and evaluate ||M~^-1|| at every resulting operating point. Take a
    high quantile as the working value mu_0, and treat the probability of
    exceeding it as an extra failure probability delta_M that gets ADDED to
    delta. The mechanism is then (epsilon, delta + delta_M)-private.

    That is an honest trade: a smaller mu_0 gives a tighter epsilon but a
    larger delta_M. We report both so the exchange is visible.

    THE CLOPPER-PEARSON INTERVAL
    ----------------------------
    delta_M is estimated from a finite sample, so it carries uncertainty.
    Clopper-Pearson gives an exact confidence interval for a proportion --
    exact in the sense that it never under-covers, unlike the usual normal
    approximation, which is unreliable when the count is small. Since our
    count of exceedances is small by construction, using it matters.

    We report the UPPER end of the interval, because for a privacy claim you
    want the pessimistic figure.
    """
    from scipy.stats import beta as beta_dist

    norms = np.array([
        np.linalg.norm(np.linalg.inv(normalised_jacobian(v, Y, b)), 2)
        for v in voltages
    ])

    mu_0 = float(np.quantile(norms, quantile))

    n_samples = len(norms)
    n_exceed = int((norms > mu_0).sum())

    # Clopper-Pearson upper limit for a binomial proportion.
    if n_exceed == n_samples:
        delta_M_upper = 1.0
    else:
        delta_M_upper = float(
            beta_dist.ppf(confidence, n_exceed + 1, n_samples - n_exceed)
        )

    return {
        "mu_0": mu_0,
        "delta_M": float(n_exceed / n_samples),
        "delta_M_upper": delta_M_upper,
        "n_samples": n_samples,
        "n_exceed": n_exceed,
        "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
        "quantile": quantile,
    }
