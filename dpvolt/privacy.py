# DP mechanism for the load model, plus Theorem 1 -- the bound saying private
# loads alone hide the topology Y, with no extra noise on the admittance matrix.
#
# We substitute a plain Gaussian mechanism for the paper's DP-GMM
# (arXiv:2506.03467), which has no public implementation. Legitimate under their
# Section III-C: Theorem 1 only needs the released loads to be log-normal with a
# known Sigma. We lose fit quality, not validity -- so expect the same ORDERING
# of methods in Figures 2 and 3, not the same values.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 1. The DP mechanism
# ---------------------------------------------------------------------------

def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Classical Gaussian mechanism noise scale (Dwork & Roth), as used by the
    paper's baselines in Table III:

        sigma = sensitivity * sqrt(2 ln(1.25 / delta)) / epsilon

    Slightly loose -- Balle & Wang's analytic Gaussian mechanism achieves the
    same guarantee with less noise. It would help every method equally, so it
    changes no comparison here.
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
    sigma_mu: float           # noise sd added to the mean
    sigma_cov: float          # noise sd added to the covariance
    eig_clipped: int          # eigenvalues lifted to keep the matrix valid
    records_clipped: int      # records whose L2 norm exceeded the clip bound
    clip_norm: float
    kl_to_true: float         # KL from the true fit, for reference


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

    log_data   (m, T); each row is one bus-day's log-load trajectory.
    lo, hi     log-load bounds, from the class margins.
    clip_norm  L2 bound on each centred record. Defaults to the worst case
               sqrt(T) * R / 2, which is correct but very loose -- see below.

    BUDGET SPLIT. We release a mean and a covariance, so by composition each
    gets half. That halving is unavoidable, and it is why a method releasing
    fewer quantities can afford less noise.

    SENSITIVITIES. Bounded-record adjacency: one row may be replaced by any
    other row inside the box. With R = hi - lo:
      mean -- replacing a record moves each coordinate by <= R/m, so the L2
              sensitivity over T coordinates is sqrt(T) * R / m.
      cov  -- centre each record and clip its L2 norm to C; replacing one
              clipped record then moves the matrix by <= 2 C^2 / m (Frobenius).

    WHY CLIPPING IS MANDATORY. Without it C is forced to sqrt(T) * R / 2 ~ 22.8
    on our data, so C^2 ~ 519 and at eps = 1 the noise sd lands near 14.8 --
    against true covariance entries of ~0.1. We measured 49 of 96 eigenvalues
    clipped and KL = 363,589. Real centred records have norm ~3, not 22.8; the
    worst case assumes all 96 coordinates sit at an extreme at once. Clipping
    cuts sensitivity by the ratio squared.

    This is also the honest reason the paper runs at eps 25-200 rather than the
    single digits usual in DP: privately estimating a 96x96 covariance is
    expensive, and no cleverness makes it cheap.

    CAVEAT. C is a hyperparameter and must be chosen WITHOUT looking at the
    private data, or its selection leaks. A public load study, or a small slice
    of budget spent estimating it, are both defensible; tuning it against your
    own data until the figures look good is not.

    EIGENVALUE REPAIR. Symmetric noise can push eigenvalues negative, which is
    not a samplable covariance, so we clip them up to a small floor. This is
    FREE under DP by post-processing -- we only touch the already-noised matrix,
    never the real data. Worth stating, because it looks like cheating.
    """
    m, T = log_data.shape
    R = float(hi - lo)

    eps_half, delta_half = epsilon / 2.0, delta / 2.0

    # ---- private mean -----------------------------------------------------
    sens_mu = np.sqrt(T) * R / m
    sigma_mu = gaussian_sigma(sens_mu, eps_half, delta_half)
    mu_true = log_data.mean(axis=0)
    mu_dp = mu_true + rng.normal(0.0, sigma_mu, size=T)

    # ---- private covariance -----------------------------------------------
    if clip_norm is None:
        clip_norm = np.sqrt(T) * R / 2.0          # worst case, very loose

    # Centre on the ALREADY-PRIVATE mean, so this costs no budget
    # (post-processing), then clip each record's L2 norm to C.
    centred = log_data - mu_dp
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    centred = centred * scale
    n_clipped_records = int((scale < 1.0).sum())

    cov_true = (centred.T @ centred) / m + 1e-12 * np.eye(T)

    sens_cov = 2.0 * clip_norm ** 2 / m
    sigma_cov = gaussian_sigma(sens_cov, eps_half, delta_half)

    # Symmetrised Gaussian noise, as a covariance must be symmetric.
    noise = rng.normal(0.0, sigma_cov, size=(T, T))
    cov_dp = cov_true + (noise + noise.T) / np.sqrt(2.0)

    # ---- repair (post-processing, free) -----------------------------------
    cov_dp = (cov_dp + cov_dp.T) / 2.0
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
    """KL from Normal(mu0, cov0) to Normal(mu1, cov1). Lower is better.

    The paper's DP-GMM is built to minimise exactly this, so it is the fair
    yardstick for what our simpler substitute costs.
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


# ---------------------------------------------------------------------------
# 2. Theorem 1
# ---------------------------------------------------------------------------

@dataclass
class PrivacyBound:
    """Every intermediate quantity in Theorem 1.

    Kept separate rather than collapsed to one number: when epsilon comes out
    large you need to see WHICH term is responsible.
    """

    epsilon: float           # the headline result, eq. (32)
    delta: float
    r: float                 # adjacency radius on Y_full
    alpha: float             # admissibility parameter, eq. (30)
    admissible: bool         # alpha < 1/4? if not, the theorem says nothing
    C_star: float
    jacobian_term: float     # the floor load noise cannot cross
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
    """Evaluate the paper's (epsilon, delta) bound, eq. (32)-(35).

    EVERY INPUT MUST BE PER-UNIT. The formulas mix voltages, admittances and
    loads, so a volts/per-unit slip changes epsilon by orders of magnitude with
    no warning.

    epsilon has three pieces that behave very differently:
      JACOBIAN  distortion of the power-flow volume element by a change in Y.
                Depends on r but NOT on load noise -- a FLOOR that no amount of
                extra load noise crosses. The only lever is shrinking r.
      BIAS      deterministic, scales with (kappa_Kron * r)^2.
      TAIL      psi_bar * tau(delta), the piece load noise controls. Noisier
                loads -> larger Sigma -> smaller precision sum gamma -> smaller
                psi_bar. This is the paper's central mechanism.
    """
    # ---- eq. (30): admissibility ------------------------------------------
    C_star = np.sqrt(2.0) * (1.0 + np.sqrt(n) * V_max / V_min)
    alpha = M_inv_norm * C_star * kappa_kron * r
    admissible = alpha < 0.25

    # ---- eq. (31): Jacobian term ------------------------------------------
    # At alpha >= 1/4 this goes negative or explodes -- the theorem's way of
    # saying it does not apply. Return infinity so a caller who ignores
    # `admissible` gets an obviously wrong number, not a plausible one.
    if admissible:
        jac = T * np.sqrt(n) * alpha * (2.0 + alpha) / (2.0 * (1.0 - 4.0 * alpha))
    else:
        jac = np.inf

    # ---- eq. (23) and (24): per-class constants ---------------------------
    d_ell, gamma_ell = {}, {}
    for label, Sigma in Sigma_by_class.items():
        # p_min in the DENOMINATOR: lightly loaded buses inflate the bound,
        # which is why the paper wants margins as tight as feasibility allows.
        d_ell[label] = V_max ** 2 * np.sqrt(d_max) / p_min_by_class[label]

        # "Precision sum": sum of |entries| of the inverse covariance. Noisier
        # load model -> bigger Sigma -> smaller inverse -> smaller gamma ->
        # smaller epsilon. This is where private loads buy topology privacy.
        Sigma_inv = np.linalg.inv(Sigma)
        gamma_ell[label] = float(np.abs(Sigma_inv).sum())

    # ---- eq. (34): uniform whitened-shift bound ---------------------------
    psi_sq = (kappa_kron * r) ** 2 * sum(
        d_ell[l] ** 2 * gamma_ell[l] for l in Sigma_by_class
    )
    psi_bar = float(np.sqrt(psi_sq))

    # ---- eq. (35): chi-squared tail factor --------------------------------
    # Laurent-Massart concentration on the whitened load draws. For
    # nT >> log(1/delta) it is approximately sqrt(nT).
    nT = n * T
    log_inv_delta = np.log(1.0 / delta)
    tau = float(np.sqrt(nT + 2.0 * np.sqrt(nT * log_inv_delta) + 2.0 * log_inv_delta))

    # ---- eq. (33): deterministic bias -------------------------------------
    beta_sum = 0.0
    for label, Sigma in Sigma_by_class.items():
        beta_sum += (
            d_ell[label]
            * np.sqrt(gamma_ell[label] * size_by_class[label])
            * np.sqrt(np.ones(T) @ Sigma @ np.ones(T))
        )
    bias = jac + 0.5 * psi_sq + kappa_kron * r * beta_sum

    # ---- eq. (32) ----------------------------------------------------------
    epsilon = float(bias + psi_bar * tau)

    return PrivacyBound(
        epsilon=epsilon, delta=delta, r=r, alpha=float(alpha),
        admissible=bool(admissible), C_star=float(C_star),
        jacobian_term=float(jac), psi_bar=psi_bar,
        tail_term=float(psi_bar * tau), bias_term=float(bias), tau=tau,
        d_ell=d_ell, gamma_ell=gamma_ell,
    )


def solve_for_r(target_epsilon: float, *, bracket=(1e-14, 1e-2), **kwargs) -> float:
    """Largest adjacency radius r still meeting a target epsilon.

    epsilon is monotone in r, so we bisect. Usually the more useful direction:
    a utility knows what epsilon it will spend and wants to know how broad a
    class of network changes that actually protects.
    """
    lo, hi = bracket

    def eps_at(r):
        bound = theorem1(r=r, **kwargs)
        return bound.epsilon if bound.admissible else np.inf

    if eps_at(lo) > target_epsilon:
        return 0.0                     # unreachable even at the smallest r

    for _ in range(200):
        mid = np.sqrt(lo * hi)         # geometric midpoint, since r spans decades
        if eps_at(mid) <= target_epsilon:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# 3. Monte Carlo calibration of ||M~^-1||  (the paper's Remark 2)
# ---------------------------------------------------------------------------

def normalised_jacobian(v: np.ndarray, Y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build M~, the normalised power-flow Jacobian of eq. (29):

        M~ = [[ diag(s / v^2),  conj(Y)                    ],
              [ Y,              diag(conj(s) / conj(v)^2)  ]]

    with s_i = v_i * (conj(Y) conj(v) + conj(b))_i the complex power injected
    at bus i. Factoring a diagonal voltage matrix out of the full Jacobian is
    what makes the bound depend on the network rather than the operating point.
    All per-unit.
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

    Appendix E gives a closed form, but it is a worst case over every voltage
    in the admissible set and so very conservative; the paper says plainly that
    its own evaluation uses this Monte Carlo calibration instead.

    Solve power flow at many sampled trajectories, evaluate ||M~^-1|| at each
    operating point, take a high quantile as the working value mu_0, and treat
    the exceedance probability as an extra failure probability delta_M ADDED to
    delta. The mechanism is then (epsilon, delta + delta_M)-private. Smaller
    mu_0 gives a tighter epsilon but a larger delta_M; we report both.

    delta_M is estimated from a finite sample, so we bound it with a
    Clopper-Pearson interval -- exact in the sense of never under-covering,
    unlike the normal approximation, which is unreliable when the exceedance
    count is small (as it is here by construction). We report the UPPER end,
    since a privacy claim wants the pessimistic figure.
    """
    from scipy.stats import beta as beta_dist

    norms = np.array([
        np.linalg.norm(np.linalg.inv(normalised_jacobian(v, Y, b)), 2)
        for v in voltages
    ])

    mu_0 = float(np.quantile(norms, quantile))

    n_samples = len(norms)
    n_exceed = int((norms > mu_0).sum())

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
