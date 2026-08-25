# The load model -- Section II-C of the paper and Algorithm 1.
#
# Buses are grouped into L consumer classes. Within a class, the log of one
# day's active power (T = 96 steps at 15-minute resolution) is jointly Gaussian,
# eq. (10). Working in logs keeps load positive and captures its skew; Sigma_l's
# off-diagonal structure preserves the time correlation that independent noise
# destroys. Reactive power follows from a fixed power factor.
#
# We do not use the paper's OEDI dataset. make_historical() manufactures the
# archive from a known log-normal ground truth, anchored to the real per-bus kW
# ratings in the feeder file, so total demand still matches IEEE 123's
# documented 3.6 MW. Theorem 1 doesn't care where the data came from -- it only
# needs the RELEASED loads to be log-normal with a known covariance.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


T_DEFAULT = 96        # one day at 15-minute resolution, as in the paper


@dataclass
class LoadModel:
    """A fitted log-normal load model, one entry per consumer class.

    Per-unit active power throughout unless stated otherwise.
    """

    mu: dict[int, np.ndarray]        # class -> (T,) mean of the log-load
    Sigma: dict[int, np.ndarray]     # class -> (T, T) covariance of the log-load
    members: dict[int, np.ndarray]   # class -> indices of the buses in it
    p_min: dict[int, float]          # class -> lower load margin (per-unit)
    p_max: dict[int, float]          # class -> upper load margin (per-unit)
    power_factor: np.ndarray         # (n,) power factor angle per bus, radians
    T: int = T_DEFAULT

    @property
    def L(self) -> int:
        """Number of consumer classes."""
        return len(self.mu)

    def class_of(self, bus: int) -> int:
        """Which class a given bus belongs to."""
        for lab, idx in self.members.items():
            if bus in idx:
                return lab
        raise KeyError(f"bus {bus} is not in any class")


# ---------------------------------------------------------------------------
# 1. Consumer classes
# ---------------------------------------------------------------------------

def assign_classes(kw_per_bus: np.ndarray, L: int = 3) -> dict[int, np.ndarray]:
    """Split buses into L equal-sized classes by demand, smallest first."""
    if L < 1:
        raise ValueError("need at least one class")

    order = np.argsort(kw_per_bus)
    chunks = np.array_split(order, L)
    return {label: np.sort(chunk) for label, chunk in enumerate(chunks)}


# ---------------------------------------------------------------------------
# 2. Stand-in historical data
# ---------------------------------------------------------------------------

def diurnal_shape(T: int, kind: str) -> np.ndarray:
    """Daily load archetype, normalised to average 1.0 over the day."""
    hours = np.arange(T) * (24.0 / T)

    if kind == "residential":
        shape = (
            0.55
            + 0.25 * np.exp(-0.5 * ((hours - 7.5) / 1.4) ** 2)    # breakfast
            + 0.85 * np.exp(-0.5 * ((hours - 19.5) / 2.2) ** 2)   # evening peak
        )
    elif kind == "commercial":
        shape = 0.45 + 0.75 * np.exp(-0.5 * ((hours - 13.0) / 4.0) ** 2)
    else:  # industrial
        shape = 0.90 + 0.20 * np.exp(-0.5 * ((hours - 14.0) / 6.0) ** 2)

    return shape / shape.mean()


def ar1_covariance(T: int, sigma: float, rho: float) -> np.ndarray:
    """AR(1) covariance, cov[i,j] = sigma^2 * rho^|i-j|.

    The exponential lag decay is what gives synthetic loads realistic temporal
    correlation instead of white noise.
    """
    lag = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    return (sigma ** 2) * (rho ** lag)


def make_historical(
    kw_per_bus: np.ndarray,
    classes: dict[int, np.ndarray],
    n_days: int,
    T: int = T_DEFAULT,
    s_base_kw: float = 1000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Manufacture the historical archive, (n_buses, n_days, T) in per-unit.

    Stands in for the paper's OEDI data, anchored to the feeder's real per-bus
    kW ratings so total demand matches IEEE 123.
    """
    if rng is None:
        rng = np.random.default_rng(0)      # fixed seed: results reproduce

    n = len(kw_per_bus)
    archive = np.zeros((n, n_days, T))

    # Archetype and variability per class. Bigger customers have flatter,
    # steadier demand -- the empirical pattern.
    profiles = ["residential", "commercial", "industrial"]
    sigmas = [0.34, 0.24, 0.15]
    rhos = [0.93, 0.94, 0.96]

    for label, members in classes.items():
        pick = min(label, len(profiles) - 1)
        shape = diurnal_shape(T, profiles[pick])
        cov = ar1_covariance(T, sigmas[pick], rhos[pick])

        # Set the log-mean so load AVERAGES to the bus rating: for a log-normal
        # E[exp(X)] = exp(mean + var/2), so subtract var/2 to stop the
        # exponential inflating the average.
        base_pu = kw_per_bus[members] / s_base_kw            # (m,)
        log_mean = np.log(shape)[None, :] - 0.5 * np.diag(cov)[None, :]

        draws = rng.multivariate_normal(
            np.zeros(T), cov, size=(len(members), n_days)
        )                                                    # (m, n_days, T)
        archive[members] = base_pu[:, None, None] * np.exp(log_mean[:, None, :] + draws)

    return archive


# ---------------------------------------------------------------------------
# 3. Fitting
# ---------------------------------------------------------------------------

def fit_load_model(
    archive: np.ndarray,
    classes: dict[int, np.ndarray],
    power_factor: np.ndarray,
    margin: float = 1.5,
) -> LoadModel:
    """Fit a log-normal per class. `margin` widens the observed load range into
    the truncation box, so sampling is not pinned to the historical extremes.
    """
    mu, Sigma, p_min, p_max = {}, {}, {}, {}

    for label, members in classes.items():
        # Flatten bus and day into one axis: a pile of (T,) observations.
        data = archive[members]                       # (m, n_days, T)
        flat = data.reshape(-1, data.shape[-1])       # (m * n_days, T)

        logs = np.log(flat)
        mu[label] = logs.mean(axis=0)

        cov = np.cov(logs, rowvar=False)              # rows are observations

        # With T = 96 and a strongly correlated AR(1) the sample covariance is
        # often numerically semi-definite, and we must invert it later.
        cov = cov + 1e-9 * np.eye(cov.shape[0])
        Sigma[label] = cov

        lo, hi = float(flat.min()), float(flat.max())
        p_min[label] = lo / margin
        p_max[label] = hi * margin

    return LoadModel(
        mu=mu, Sigma=Sigma, members=classes,
        p_min=p_min, p_max=p_max,
        power_factor=power_factor, T=archive.shape[-1],
    )


# ---------------------------------------------------------------------------
# 4. Sampling
# ---------------------------------------------------------------------------

def sample_truncated_gaussian(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    size: int,
    rng: np.random.Generator,
    sweeps: int = 40,
) -> tuple[np.ndarray, float]:
    """Draw from a Gaussian truncated to the box [lo, hi].

    Two stages: take ordinary draws, then repair only those falling outside the
    box by Gibbs sampling. Rejection sampling is hopeless in T = 96 dimensions
    (essentially every draw violates some coordinate), while Gibbs conditions
    one coordinate at a time and stays inside by construction.

    Returns the draws and the fraction that needed repair.
    """
    T = len(mu)

    # ---- stage 1: ordinary draws ------------------------------------------
    x = rng.multivariate_normal(mu, Sigma, size=size)
    inside = np.all((x >= lo) & (x <= hi), axis=1)
    n_repair = int((~inside).sum())

    if n_repair == 0:
        return x, 0.0

    # ---- stage 2: repair the violators by Gibbs ---------------------------
    P = np.linalg.inv(Sigma)                # precision matrix
    cond_sd = np.sqrt(1.0 / np.diag(P))     # conditional standard deviations

    bad = np.flatnonzero(~inside)
    y = np.clip(x[bad], lo + 1e-12, hi - 1e-12)

    for _ in range(sweeps):
        for i in range(T):
            # Conditional mean of coordinate i given all the others.
            resid = y - mu
            adj = resid @ P[:, i] - resid[:, i] * P[i, i]
            m_i = mu[i] - adj / P[i, i]

            # Truncated 1-D draw by inverse transform: map the bounds into
            # probability space, draw uniformly between them, map back.
            a = _std_normal_cdf((lo[i] - m_i) / cond_sd[i])
            b = _std_normal_cdf((hi[i] - m_i) / cond_sd[i])
            u = a + (b - a) * rng.random(len(bad))
            u = np.clip(u, 1e-12, 1 - 1e-12)      # keep the inverse finite
            y[:, i] = m_i + cond_sd[i] * _std_normal_ppf(u)

    x[bad] = y
    return x, n_repair / size


def _std_normal_cdf(z):
    from scipy.special import ndtr
    return ndtr(z)


def _std_normal_ppf(u):
    from scipy.special import ndtri
    return ndtri(u)


def sample_loads(
    model: LoadModel,
    n_days: int,
    rng: np.random.Generator | None = None,
    sweeps: int = 40,
    report: bool = False,
) -> np.ndarray:
    """Sample synthetic loads, (n_buses, n_days, T) in per-unit.

    Draws in log space inside each class's truncation box, then exponentiates
    (Algorithm 1 line 10). With report=True also returns the per-class fraction
    of draws that needed Gibbs repair.
    """
    if rng is None:
        rng = np.random.default_rng(1)

    n = sum(len(v) for v in model.members.values())
    out = np.zeros((n, n_days, model.T))
    repairs = {}

    for label, members in model.members.items():
        lo = np.full(model.T, np.log(model.p_min[label]))
        hi = np.full(model.T, np.log(model.p_max[label]))

        draws, frac = sample_truncated_gaussian(
            model.mu[label], model.Sigma[label],
            lo, hi, size=len(members) * n_days, rng=rng, sweeps=sweeps,
        )
        repairs[label] = frac
        out[members] = np.exp(draws).reshape(len(members), n_days, model.T)

    if report:
        return out, repairs
    return out


def reactive_from_active(p: np.ndarray, power_factor: np.ndarray) -> np.ndarray:
    """q = p * tan(theta), at the bus's fixed power factor angle."""
    tan_theta = np.tan(power_factor)
    # Reshape so the per-bus factor broadcasts across days and time steps.
    return p * tan_theta.reshape(-1, *([1] * (p.ndim - 1)))
