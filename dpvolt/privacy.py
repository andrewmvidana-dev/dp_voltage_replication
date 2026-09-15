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
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. The DP mechanism
# ---------------------------------------------------------------------------

def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Classical Gaussian mechanism noise scale (Dwork & Roth), as used by the
    paper's baselines in Table III:

        sigma = sensitivity * sqrt(2 ln(1.25 / delta)) / epsilon

    DO NOT USE THIS FOR epsilon > 1. The derivation requires epsilon <= 1, and
    above that it does not merely get loose -- it UNDER-NOISES, returning a
    sigma that does not achieve the (epsilon, delta) it claims. Audited against
    the exact condition (analytic_gaussian_delta) at sensitivity 0.0267:

        eps   sigma      delta claimed   delta ACTUALLY achieved
          5   2.66e-2    5e-6            6.3e-7   ok
         25   5.32e-3    5e-6            4.2e-3   1000x worse
         50   2.66e-3    5e-6            0.47     ~10^5x worse
        200   6.65e-4    5e-6            1.0      no privacy at all

    This project runs at epsilon 25-200, so the formula is outside its regime
    everywhere it matters. Kept only to reproduce the paper's stated
    calibration for comparison (calibration="classical" in dp_fit_class);
    analytic_gaussian_sigma is the default and the correct choice.
    """
    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("need epsilon > 0 and 0 < delta < 1")
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


def analytic_gaussian_sigma(
    sensitivity: float,
    epsilon: float,
    delta: float,
    tol: float = 1e-12,
) -> float:
    """Balle & Wang (ICML 2018) analytic Gaussian mechanism, Algorithm 1.

    Same (epsilon, delta) guarantee as gaussian_sigma, strictly less noise. The
    classical formula is not just loose, it is only VALID for epsilon <= 1 --
    and this project runs at epsilon 25-200, well outside that regime. So this
    is the correct calibration here, not merely the tighter one.

    The exact condition (their Theorem 8) is

        Phi(A/2 - eps/A) - e^eps * Phi(-A/2 - eps/A) <= delta,   A = Delta/sigma

    which has no closed-form inverse, so we bisect on A. The function is
    monotone in A, which is what makes bisection valid.

    We bisect on sigma directly against that condition rather than reproducing
    their alpha-substitution: delta_achieved(sigma) is monotone DECREASING in
    sigma (more noise -> smaller failure probability), so bisection on sigma is
    valid and needs no case analysis. Slower than their closed-form bracketing
    by a few dozen evaluations of Phi, which is irrelevant here -- we call this
    a handful of times per run, not in a loop. The payoff is that the thing we
    verify is the guarantee itself, and analytic_gaussian_delta below lets a
    caller (and verify.py) check any returned sigma independently.
    """
    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("need epsilon > 0 and 0 < delta < 1")
    if sensitivity == 0:
        return 0.0

    # The classical formula is a valid upper bracket only for eps <= 1; above
    # that it can UNDER-shoot, so bracket by doubling until the condition holds
    # instead of trusting it.
    hi = sensitivity / np.sqrt(2.0 * epsilon)
    for _ in range(200):
        if analytic_gaussian_delta(sensitivity, hi, epsilon) <= delta:
            break
        hi *= 2.0
    else:
        raise RuntimeError("analytic Gaussian bracketing failed")

    lo = 0.0
    for _ in range(200):
        if hi - lo <= tol * max(1.0, hi):
            break
        mid = (lo + hi) / 2.0
        if analytic_gaussian_delta(sensitivity, mid, epsilon) <= delta:
            hi = mid                    # mid is feasible, try smaller
        else:
            lo = mid                    # mid is infeasible
    return float(hi)                    # the feasible end, never the open one


def analytic_gaussian_delta(sensitivity: float, sigma: float,
                            epsilon: float) -> float:
    """delta actually achieved by a Gaussian of scale sigma at this epsilon.

    Balle & Wang (2018) Theorem 8, the exact expression:

        delta = Phi(Delta/2sigma - eps sigma/Delta)
                - e^eps * Phi(-Delta/2sigma - eps sigma/Delta)

    Exposed separately so a privacy claim can be CHECKED rather than trusted:
    verify.py feeds back every sigma from analytic_gaussian_sigma and confirms
    the delta it actually buys is at or under target.
    """
    from scipy.stats import norm

    if sigma <= 0:
        raise ValueError("need sigma > 0")
    if sensitivity == 0:
        return 0.0

    a = sensitivity / sigma
    return float(norm.cdf(a / 2.0 - epsilon / a)
                 - np.exp(epsilon) * norm.cdf(-a / 2.0 - epsilon / a))


# ---------------------------------------------------------------------------
# 1b. zCDP composition -- a better way to split the budget across releases
# ---------------------------------------------------------------------------

def zcdp_rho_from_eps_delta(epsilon: float, delta: float) -> float:
    """Largest zCDP parameter rho whose conversion still fits (epsilon, delta).

    Bun & Steinke (2016), Proposition 1.3: rho-zCDP implies
    (rho + 2 sqrt(rho log(1/delta)), delta)-DP. We invert that in rho.

    WHY BOTHER. Gaussian mechanisms compose additively in rho, not in epsilon.
    Splitting epsilon in half twice (the current path) pays a conversion penalty
    at each release; accounting in rho and converting ONCE at the end is
    strictly better. Same guarantee, less noise, no new assumptions.

    Solving rho + 2 sqrt(rho L) - eps = 0 with L = log(1/delta): substitute
    u = sqrt(rho) to get u^2 + 2 u sqrt(L) - eps = 0, so
    u = -sqrt(L) + sqrt(L + eps) and rho = u^2.
    """
    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("need epsilon > 0 and 0 < delta < 1")
    L = np.log(1.0 / delta)
    u = -np.sqrt(L) + np.sqrt(L + epsilon)
    return float(u * u)


def zcdp_sigma(sensitivity: float, rho: float) -> float:
    """Noise scale for one Gaussian release under a rho-zCDP allocation.

    Bun & Steinke: the Gaussian mechanism with sigma = Delta / sqrt(2 rho)
    satisfies rho-zCDP. Composition is then just addition of rho across
    releases, which is what makes the budget split tunable -- see the
    `rho_split` argument of dp_fit_class.

    NOTE. Measured against a CORRECT analytic-Gaussian baseline this is not a
    win at epsilon 25-200; it saves noise only where the naive epsilon/2 split
    is compared against the invalid classical formula. Provided because the
    tunable split is independently useful (the covariance carries the temporal
    structure and can be given more than half the budget), not as a free lunch.
    """
    if rho <= 0:
        raise ValueError("need rho > 0")
    return float(sensitivity / np.sqrt(2.0 * rho))


@dataclass(kw_only=True)
class FitReport:
    """Diagnostics common to every load-model fit, private or not.

    Deliberately carries NO epsilon, delta or delta_achieved. Those live on
    DPFitReport, the subclass returned by mechanisms that actually have a
    guarantee. The split is structural on purpose: bnp_fit_class_eigen_oracle
    releases the true eigenvectors in the clear and has no guarantee at any
    parameter, so the fields that would state one DO NOT EXIST on its report.
    Reaching for `.delta` on an oracle result is an AttributeError naming the
    type, not a number that reads like a promise.

    Making the invalid state unrepresentable beats detecting it: a guard only
    fires where someone remembers to call it, and the failure mode this exists
    for -- a future edit wiring the oracle into a runner that prints a delta --
    is exactly the edit that would forget to call it.
    """

    n_records: int
    log_range: float          # width of the log-load box, per coordinate
    sigma_mu: float           # noise sd added to the mean
    sigma_cov: float          # noise sd added to the covariance
    eig_clipped: int          # eigenvalues lifted to keep the matrix valid
    records_clipped: int      # records whose L2 norm exceeded the clip bound
    clip_norm: float
    kl_to_true: float         # KL from the true fit, for reference
    calibration: str


@dataclass(kw_only=True)
class DPFitReport(FitReport):
    """A FitReport from a mechanism that HAS a privacy guarantee.

    Adds the three fields that state one. kw_only keeps epsilon and delta
    MANDATORY despite following defaulted fields in the base -- a report that
    claims a guarantee must say what it is, not inherit a default.
    """

    epsilon: float
    delta: float

    # Calibration audit. The classical formula silently under-noises above
    # epsilon = 1, so we record which calibration was used and -- for Gaussian
    # paths -- the delta the returned sigmas ACTUALLY achieve. If
    # delta_achieved > delta, the stated guarantee does not hold.
    delta_achieved: float | None = None


@dataclass(kw_only=True)
class OracleFitReport(FitReport):
    """A FitReport from something that is NOT a mechanism.

    Exists solely so an ablation's diagnostics can be collected and tabulated
    without ever being mistakable for a privacy claim. It has no epsilon, no
    delta and no delta_achieved, because there is no value those could honestly
    take. See bnp_fit_class_eigen_oracle.

    `nominal_delta` records the delta the noise was CALIBRATED with, so the
    ablation can be swept and labelled. It is not a guarantee, and it is named
    so that it cannot be read as one.
    """

    nominal_delta: float = float("nan")


def dp_fit_class(
    log_data: np.ndarray,
    lo: float,
    hi: float,
    epsilon: float,
    delta: float,
    rng: np.random.Generator,
    eig_floor_ratio: float = 1e-3,
    clip_norm: float | None = None,
    calibration: str = "analytic",
    rho_split: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, DPFitReport]:
    """Privately estimate the mean and covariance of one class's log-loads.

    log_data   (m, T); each row is one bus-day's log-load trajectory.
    lo, hi     log-load bounds, from the class margins.
    clip_norm  L2 bound on each centred record. Defaults to the worst case
               sqrt(T) * R / 2, which is correct but very loose -- see below.

    CALIBRATION. One of:
      "analytic"  (default) Balle & Wang exact calibration, split by zCDP.
                  Correct at every epsilon. USE THIS.
      "zcdp"      same as "analytic" -- kept as an alias because the split is
                  what zCDP provides and the per-release scale is Gaussian.
      "classical" the paper's Table III formula with the naive epsilon/2 split.
                  REPRODUCES A BROKEN GUARANTEE above epsilon = 1: at
                  (eps=50, delta=1e-5) the real delta is ~0.47, not 1e-5. Only
                  for showing what the stated calibration actually buys.

    BUDGET SPLIT. We release a mean and a covariance, so the budget divides
    across two Gaussian releases. Under zCDP that division is additive in rho,
    and `rho_split` sets the mean's share (default 0.5, an even split). Because
    the covariance carries the temporal structure that this whole method exists
    to preserve, giving it MORE than half -- rho_split < 0.5 -- is a defensible
    tuning knob, and unlike the clip norm it leaks nothing: the split is chosen
    without reference to the data.

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

    if calibration not in ("analytic", "zcdp", "classical"):
        raise ValueError(f"unknown calibration {calibration!r}")
    if not (0.0 < rho_split < 1.0):
        raise ValueError("need 0 < rho_split < 1")

    sens_mu = np.sqrt(T) * R / m

    if clip_norm is None:
        clip_norm = np.sqrt(T) * R / 2.0          # worst case, very loose
    sens_cov = 2.0 * clip_norm ** 2 / m

    # ---- calibrate both releases together ---------------------------------
    # Done up front so the two paths are visibly the same decision, and so the
    # audit below can re-check whichever one was taken.
    if calibration == "classical":
        # The paper's stated calibration: split epsilon and delta evenly, then
        # apply the classical formula to each half. Under-noises above
        # epsilon = 1; kept only to quantify that gap.
        eps_mu = eps_cov = epsilon / 2.0
        sigma_mu = gaussian_sigma(sens_mu, eps_mu, delta / 2.0)
        sigma_cov = gaussian_sigma(sens_cov, eps_cov, delta / 2.0)
    else:
        # Split the budget in rho (additive under zCDP), then convert each
        # share back to an epsilon and calibrate that release exactly. The two
        # epsilons do NOT sum to `epsilon` -- that is the point of composing in
        # rho -- but the pair of releases together still satisfies
        # (epsilon, delta), because rho_mu + rho_cov = rho_total and
        # rho_total converts to exactly epsilon at delta.
        rho_total = zcdp_rho_from_eps_delta(epsilon, delta)
        eps_mu = _eps_from_rho(rho_total * rho_split, delta / 2.0)
        eps_cov = _eps_from_rho(rho_total * (1.0 - rho_split), delta / 2.0)
        sigma_mu = analytic_gaussian_sigma(sens_mu, eps_mu, delta / 2.0)
        sigma_cov = analytic_gaussian_sigma(sens_cov, eps_cov, delta / 2.0)

    # Audit each release at the epsilon it was actually calibrated FOR -- not
    # at epsilon/2, which is only the right question on the classical path.
    # Summing the two deltas is the basic-composition accounting: the pair is
    # (eps_mu + eps_cov, d_mu + d_cov)-DP, and for the analytic path the
    # tighter zCDP accounting above certifies the stronger (epsilon, delta).
    d_mu = analytic_gaussian_delta(sens_mu, sigma_mu, eps_mu)
    d_cov = analytic_gaussian_delta(sens_cov, sigma_cov, eps_cov)
    delta_achieved = float(d_mu + d_cov)

    # ---- private mean -----------------------------------------------------
    mu_true = log_data.mean(axis=0)
    mu_dp = mu_true + rng.normal(0.0, sigma_mu, size=T)

    # ---- private covariance -----------------------------------------------

    # Centre on the ALREADY-PRIVATE mean, so this costs no budget
    # (post-processing), then clip each record's L2 norm to C.
    centred = log_data - mu_dp
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    centred = centred * scale
    n_clipped_records = int((scale < 1.0).sum())

    cov_true = (centred.T @ centred) / m + 1e-12 * np.eye(T)

    # Symmetrised Gaussian noise, as a covariance must be symmetric. The
    # /sqrt(2) keeps each entry's marginal sd at sigma_cov: off-diagonal
    # entries are (n_ij + n_ji)/sqrt(2), a sum of two independent N(0, sigma^2)
    # scaled to variance sigma^2. Diagonal entries get 2 n_ii / sqrt(2), i.e.
    # sd sigma*sqrt(2) -- conservative (more noise than calibrated), so the
    # guarantee holds.
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
        calibration=calibration, delta_achieved=delta_achieved,
    )
    return mu_dp, cov_dp, report


def _eps_from_rho(rho: float, delta: float) -> float:
    """Convert a rho-zCDP allocation to an (eps, delta)-DP epsilon.

    Bun & Steinke Prop. 1.3, forward direction:
        eps = rho + 2 sqrt(rho log(1/delta))
    The inverse of zcdp_rho_from_eps_delta, used to turn each release's share
    of rho back into the epsilon its own calibration needs.
    """
    if rho <= 0 or not (0 < delta < 1):
        raise ValueError("need rho > 0 and 0 < delta < 1")
    return float(rho + 2.0 * np.sqrt(rho * np.log(1.0 / delta)))


def bnp_fit_class(
    log_data: np.ndarray,
    lo: float,
    hi: float,
    delta: float,
    rng: np.random.Generator,
    eig_floor_ratio: float = 1e-3,
    clip_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, DPFitReport]:
    """Bounded-Noise Privacy counterpart of dp_fit_class.

    Same estimator, same sensitivities, same budget split -- only the noise
    distribution changes, from Gaussian to uniform on [-B, B] with B chosen by
    Corollary 1 to hit the target delta:

        B = sensitivity / (2 * delta)

    WHY THIS IS THE PROMISING PLACE FOR BNP. On the released voltages (see
    add_bounded_voltage_noise and figure 4) there is no admissible bound: the
    voltage sensitivity is comparable to the signal itself, so every valid B
    already exceeds the ANSI band. Here the picture is different -- the
    per-record sensitivity of a mean over m ~ 2700 records carries a 1/m
    factor, so B comes out small relative to the log-load range R. Bounding is
    cheap exactly where the estimator averages over many records.

    The mechanism is (0, delta)-private per released quantity, and we release
    two (mean and covariance), so delta splits in half across them just as
    epsilon does in the Gaussian path.

    The returned report reuses DPFitReport. Its `sigma_mu` and `sigma_cov`
    fields hold the BOUNDS B, not standard deviations -- for uniform noise on
    [-B, B] the standard deviation is B/sqrt(3), so the two are not
    interchangeable when comparing against a Gaussian fit.
    """
    m, T = log_data.shape
    R = float(hi - lo)

    delta_half = delta / 2.0

    # ---- private mean, bounded --------------------------------------------
    sens_mu = np.sqrt(T) * R / m
    B_mu = bnp_bound_scalar(sens_mu, delta_half)
    mu_true = log_data.mean(axis=0)
    mu_dp = mu_true + rng.uniform(-B_mu, B_mu, size=T)

    # ---- private covariance, bounded --------------------------------------
    if clip_norm is None:
        clip_norm = np.sqrt(T) * R / 2.0

    centred = log_data - mu_dp
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    centred = centred * scale
    n_clipped_records = int((scale < 1.0).sum())

    cov_true = (centred.T @ centred) / m + 1e-12 * np.eye(T)

    sens_cov = 2.0 * clip_norm ** 2 / m
    B_cov = bnp_bound_scalar(sens_cov, delta_half)

    noise = rng.uniform(-B_cov, B_cov, size=(T, T))
    cov_dp = cov_true + (noise + noise.T) / 2.0      # symmetrise

    # ---- repair (post-processing, free) -----------------------------------
    cov_dp = (cov_dp + cov_dp.T) / 2.0
    evals, evecs = np.linalg.eigh(cov_dp)
    floor = eig_floor_ratio * max(float(np.trace(cov_true)) / T, 1e-12)
    n_clipped = int((evals < floor).sum())
    evals = np.maximum(evals, floor)
    cov_dp = evecs @ np.diag(evals) @ evecs.T

    report = DPFitReport(
        epsilon=0.0,                    # uniform BNP is (0, delta)-private
        delta=delta, n_records=m, log_range=R,
        sigma_mu=B_mu, sigma_cov=B_cov,   # BOUNDS, not standard deviations
        eig_clipped=n_clipped, records_clipped=n_clipped_records,
        clip_norm=float(clip_norm),
        kl_to_true=gaussian_kl(mu_dp, cov_dp, mu_true, cov_true),
        # Uniform BNP is exactly (0, delta)-private by Corollary 1, so its
        # delta needs no audit -- the bound IS the guarantee. delta_achieved is
        # left None to mark "not a Gaussian calibration" rather than "unchecked".
        calibration="bnp-uniform", delta_achieved=None,
    )
    return mu_dp, cov_dp, report


def bnp_fit_class_eigen_oracle(
    log_data: np.ndarray,
    lo: float,
    hi: float,
    delta: float,
    rng: np.random.Generator,
    eig_floor_ratio: float | None = 1e-3,
    clip_norm: float | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, OracleFitReport]:
    """NOT PRIVATE. NOT A MECHANISM. DO NOT REPORT A GUARANTEE FROM THIS.

    This releases the TRUE EIGENVECTORS V of the class covariance in the clear.
    V is computed from the raw data, so the entire correlation structure -- the
    one thing this project exists to protect -- leaks exactly, with no noise on
    it at all. Perturbing only the eigenvalues does not repair that.

    POST-PROCESSING DOES NOT APPLY. The usual argument ("we only touch the
    already-noised matrix, so it is free") is what makes the eigenvalue repair
    in bnp_fit_class legitimate. It does NOT transfer here: post-processing is
    closed under functions of the RELEASED quantity, and V is a function of the
    RAW data, not of any released quantity. There is no delta, however large,
    that makes this output (0, delta)-private.

    Enforced structurally: this returns an OracleFitReport, which HAS NO
    epsilon, delta or delta_achieved field. Code that tries to report a
    guarantee from it raises AttributeError rather than printing a number.

    WHY IT EXISTS ANYWAY. A reviewer proposed it as a fix for BNP's utility
    collapse. It is an ablation and nothing more: bnp_fit_class destroys the
    covariance through two mechanisms at once -- the magnitude of the uniform
    noise, and the eigenvalue floor that repairs what that noise breaks -- and
    those two cannot be separated while the noise is applied entrywise. Here
    the noise lands on the spectrum directly and the floor can be switched off
    (eig_floor_ratio=None), so the two causes come apart and can be measured
    against each other. The measurement is the deliverable. The mechanism is
    not a candidate.

    Argument signature matches bnp_fit_class, with two additions:
    eig_floor_ratio may be None to disable the floor, and verbose prints the
    bound accounting. The RETURN TYPE deliberately differs -- see above.

    B_lambda FROM WEYL. Under bounded-record replacement with each centred
    record clipped to C, replacing one record moves the covariance by at most
    2 C^2 / m in Frobenius norm, hence by at most that in spectral norm. Weyl's
    inequality then bounds the movement of EVERY eigenvalue by the same
    2 C^2 / m, so that is the per-eigenvalue sensitivity and Corollary 1 gives

        B_lambda = sens / (2 * delta)

    THE HIDDEN COMPOSITION COST. That delta is PER EIGENVALUE. The release is
    all T of them, so naive composition costs T * delta -- at T = 96 and
    delta = 0.02 that is 1.92, which is not a probability. Even granting the
    eigenvectors for free, the eigenvalue release alone has no admissible
    operating point. Printed below so it is visible rather than buried.
    """
    m, T = log_data.shape
    R = float(hi - lo)

    delta_half = delta / 2.0

    # ---- mean: identical to bnp_fit_class, so the rows differ in one thing --
    sens_mu = np.sqrt(T) * R / m
    B_mu = bnp_bound_scalar(sens_mu, delta_half)
    mu_true = log_data.mean(axis=0)
    mu_dp = mu_true + rng.uniform(-B_mu, B_mu, size=T)

    # ---- covariance ---------------------------------------------------------
    if clip_norm is None:
        clip_norm = np.sqrt(T) * R / 2.0

    centred = log_data - mu_dp
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    centred = centred * scale
    n_clipped_records = int((scale < 1.0).sum())

    cov_true = (centred.T @ centred) / m + 1e-12 * np.eye(T)

    # Weyl sensitivity -- the SAME 2C^2/m the entrywise path uses. That it is
    # unchanged is itself the point: moving the noise into eigenvalue space
    # does not reduce how much one record can move the estimate.
    sens_lambda = 2.0 * clip_norm ** 2 / m
    B_lambda = bnp_bound_scalar(sens_lambda, delta_half)

    if verbose:
        print(f"    [NOT PRIVATE oracle] per-eigenvalue B_lambda = "
              f"{B_lambda:.4f} at delta = {delta_half:g}")
        print(f"    [NOT PRIVATE oracle] naive composition over T = {T} "
              f"eigenvalues costs delta = {T * delta_half:.3f}"
              + ("  -- ABOVE 1, not a probability"
                 if T * delta_half > 1.0 else ""))

    # THE NON-PRIVATE STEP. evecs comes from the true (clipped, centred) data
    # and is released unperturbed.
    evals_true, evecs_true = np.linalg.eigh(cov_true)

    evals_hat = evals_true + rng.uniform(-B_lambda, B_lambda, size=T)

    # max(0, .) is part of the reviewer's proposal, not the floor: a negative
    # eigenvalue is not samplable at all. The FLOOR is the separate, optional
    # lift to eig_floor_ratio * mean(diag) that bnp_fit_class applies.
    evals_hat = np.maximum(evals_hat, 0.0)

    if eig_floor_ratio is None:
        # Floor disabled. Zeros are still not invertible and gaussian_kl needs
        # a nonsingular covariance, so lift only to the numerical minimum --
        # enough to be well posed, far below any structural floor. The row this
        # feeds measures "floor at numerical minimum", not "no floor at all".
        floor = 1e-12
    else:
        floor = eig_floor_ratio * max(float(np.trace(cov_true)) / T, 1e-12)
    n_clipped = int((evals_hat < floor).sum())
    evals_hat = np.maximum(evals_hat, floor)

    cov_dp = evecs_true @ np.diag(evals_hat) @ evecs_true.T
    cov_dp = (cov_dp + cov_dp.T) / 2.0        # kill eigh round-off asymmetry

    report = OracleFitReport(
        n_records=m, log_range=R,
        sigma_mu=B_mu, sigma_cov=B_lambda,    # BOUNDS, not standard deviations
        eig_clipped=n_clipped, records_clipped=n_clipped_records,
        clip_norm=float(clip_norm),
        kl_to_true=gaussian_kl(mu_dp, cov_dp, mu_true, cov_true),
        calibration="bnp-eigen-oracle-NOTPRIVATE",
        nominal_delta=float(delta),
    )
    return mu_dp, cov_dp, report


def bnp_bound_scalar(sensitivity: float, delta: float) -> float:
    """Noise bound B giving a target delta: B = S / (2 delta), Corollary 1.

    Duplicated from powerflow.bnp_bound so privacy.py does not import the
    OpenDSS-dependent module. Kept in sync by a check in verify.py.
    """
    if delta <= 0:
        raise ValueError("need delta > 0")
    return sensitivity / (2.0 * delta)


def gaussian_kl(
    mu0: np.ndarray,
    cov0: np.ndarray,
    mu1: np.ndarray,
    cov1: np.ndarray,
) -> float:
    """KL from Normal(mu0, cov0) to Normal(mu1, cov1). Lower is better.

    Means have shape (T,) and positive-definite covariances shape (T, T).
    Returns the scalar divergence in nats.

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
    d_ell: dict[int, float]              # per-class sensitivity constant, eq. (23)
    gamma_ell: dict[int, float]          # per-class precision sum, eq. (24)


def theorem1(
    *,
    Sigma_by_class: dict[int, np.ndarray],        # class -> (T, T) covariance of the DP model
    size_by_class: dict[int, int],         # class -> number of buses in it
    p_min_by_class: dict[int, float],        # class -> lower load margin, per-unit
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


def solve_for_r(
    target_epsilon: float,
    *,
    bracket: tuple[float, float] = (1e-14, 1e-2),
    **kwargs: Any,
) -> float:
    """Largest adjacency radius r still meeting a target epsilon.

    bracket gives the lower and upper radii; kwargs forwards the remaining
    keyword arguments to theorem1. Returns zero if the lower radius fails.

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
) -> dict[str, float | int]:
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
