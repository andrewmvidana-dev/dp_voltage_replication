# Figures 2 and 3, which answer different questions. Figure 2 (Wasserstein-1)
# asks how far the released voltage DISTRIBUTION is from the true one. Figure 3
# (masked recovery) asks whether a model trained on released data works on REAL
# data -- the question that matters for training grid foundation models. A
# method can do well on one and badly on the other.
#
# Three methods compared: proposed (private loads through true power flow, no
# noise on voltages), gaussian (output perturbation at matched epsilon), and
# noise-free (the ceiling, not a legal method).
#
# The "Gaussian noise on Y" baseline is omitted: the paper's Remark 3 reports
# OpenDSS failed to converge for it even at r = 1e-3, epsilon = 100.

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# 1. Figure 2: statistical fidelity
# ---------------------------------------------------------------------------

def voltage_wasserstein(V_true: np.ndarray, V_released: np.ndarray) -> float:
    """Mean per-node Wasserstein-1 distance between voltage MAGNITUDE
    distributions. Per-node then averaged, so a method cannot hide a badly
    wrong node behind well-modelled ones."""
    mag_true = np.abs(V_true)
    mag_rel = np.abs(V_released)

    if mag_true.shape[1] != mag_rel.shape[1]:
        raise ValueError("the two voltage sets must cover the same nodes")

    per_node = [
        wasserstein_distance(mag_true[:, j], mag_rel[:, j])
        for j in range(mag_true.shape[1])
    ]
    return float(np.mean(per_node))


def ansi_violation_rate(V: np.ndarray,
                        v_min: float = 0.95,
                        v_max: float = 1.05) -> float:
    """Fraction of released voltage magnitudes outside the ANSI C84.1 band.

    The physical-feasibility axis, which Wasserstein and masked recovery both
    miss: a method can match the true distribution well on average while still
    emitting voltages that cannot exist on a real feeder. Definition 2's "good
    voltage set" is this same band.
    """
    mag = np.abs(V)
    return float(np.mean((mag < v_min) | (mag > v_max)))


def mean_autocorrelation(V: np.ndarray, max_lag: int = 5) -> np.ndarray:
    """Mean lag-1..max_lag autocorrelation of voltage magnitude over time.

    V is (days, T, nodes) -- the time axis must be intact, so this takes the
    unflattened array. Averaged over every node and day.

    This is what separates the two output-perturbation baselines from the
    proposed method. Both baselines add noise independently per timestep, so
    both destroy autocorrelation regardless of whether that noise is bounded;
    bounding controls magnitude, not temporal structure. The proposed method
    never touches the voltages, so the correlation induced by the load model
    and the network survives intact.
    """
    mag = np.abs(V)
    n_days, T, n_nodes = mag.shape

    # (days * nodes, T): one row per series, time along the row.
    series = mag.transpose(0, 2, 1).reshape(-1, T)
    series = series - series.mean(axis=1, keepdims=True)

    var = (series ** 2).mean(axis=1)
    ok = var > 1e-24                      # drop dead-flat series
    series, var = series[ok], var[ok]

    out = []
    for lag in range(1, max_lag + 1):
        cov = (series[:, :-lag] * series[:, lag:]).mean(axis=1)
        out.append(float(np.mean(cov / var)))
    return np.array(out)


def empirical_voltage_sensitivity(
    runner,
    model,
    theta: np.ndarray,
    rng: np.random.Generator,
    n_trials: int = 12,
) -> float:
    """Largest observed ||V1 - V0|| when one bus's whole daily trajectory is
    replaced -- the L2 sensitivity calibrating the output-perturbation
    baseline, under the same bounded-record adjacency used for the loads.

    Empirical, so a lower bound on the true sensitivity; a closed form would
    require worst-casing the power flow. It is the same figure the baseline
    would be given in practice, so the comparison stays fair.
    """
    from dpvolt.loads import sample_loads, reactive_from_active

    n_loads = len(runner.load_names)
    worst = 0.0

    for _ in range(n_trials):
        base = sample_loads(model, 1, rng=rng, sweeps=10)
        q_base = reactive_from_active(base, theta)
        V0, ok0 = runner.solve_trajectory(base[:, 0, :], q_base[:, 0, :])
        if not ok0.all():
            continue

        # Replace exactly one bus's entire daily trajectory.
        alt = sample_loads(model, 1, rng=rng, sweeps=10)
        bus = int(rng.integers(n_loads))
        perturbed = base.copy()
        perturbed[bus] = alt[bus]

        q_pert = reactive_from_active(perturbed, theta)
        V1, ok1 = runner.solve_trajectory(perturbed[:, 0, :], q_pert[:, 0, :])
        if not ok1.all():
            continue

        worst = max(worst, float(np.linalg.norm(V1 - V0)))

    if worst == 0.0:
        raise RuntimeError("every sensitivity trial failed to converge")
    return worst


# ---------------------------------------------------------------------------
# 2. Figure 3: does the released data train a useful model?
# ---------------------------------------------------------------------------

def build_masked_dataset(
    V: np.ndarray,
    d_w: int = 48,
    mask_len: int = 24,
    stride: int = 6,
):
    """Masked-recovery task: from a d_w-step window of one node's voltage
    magnitude with the last mask_len steps hidden, predict the hidden tail.

    V is (days, T, nodes). Each x is the masked series concatenated with the
    mask itself, so the model can tell a hidden step from a genuine zero.
    Returns X (n, 2 * d_w) and Y (n, mask_len).
    """
    mag = np.abs(V)                                   # (days, T, nodes)
    n_days, T, n_nodes = mag.shape

    xs, ys = [], []
    for d in range(n_days):
        for start in range(0, T - d_w + 1, stride):
            window = mag[d, start:start + d_w, :]     # (d_w, nodes)
            for j in range(n_nodes):
                series = window[:, j]

                mask = np.ones(d_w)
                mask[-mask_len:] = 0.0                # hide the tail

                xs.append(np.concatenate([series * mask, mask]))
                ys.append(series[-mask_len:])

    return np.array(xs), np.array(ys)


class SmallMLP:
    """Two-hidden-layer ReLU MLP with Adam, in plain NumPy.

    Hand-written rather than pulled from a framework so the whole pipeline
    depends only on numpy/scipy, and so the training is fully seed-controlled
    for the multi-seed averaging Figure 3 needs.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        dims = [d_in, hidden, hidden, d_out]

        # He initialisation, sqrt(2 / fan_in): the standard choice for ReLU.
        self.W, self.B = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b)))
            self.B.append(np.zeros(b))

        # Adam's running averages, one pair per parameter array.
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mB = [np.zeros_like(b) for b in self.B]
        self.vB = [np.zeros_like(b) for b in self.B]
        self.t = 0

    def forward(self, X):
        acts = [X]
        h = X
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            z = h @ w + b
            # Output layer stays linear: we predict real-valued voltages.
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(h)
        return acts

    def train_epoch(self, X, Y, lr=1e-3, batch=128, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)

        order = rng.permutation(len(X))
        total = 0.0

        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], Y[idx]

            acts = self.forward(xb)
            pred = acts[-1]

            err = pred - yb                       # MSE and its gradient
            total += float(np.mean(err ** 2)) * len(idx)
            grad = 2.0 * err / len(idx)

            gW = [None] * len(self.W)
            gB = [None] * len(self.B)
            for i in reversed(range(len(self.W))):
                gW[i] = acts[i].T @ grad
                gB[i] = grad.sum(axis=0)
                if i > 0:
                    grad = grad @ self.W[i].T
                    grad = grad * (acts[i] > 0)      # derivative of ReLU

            self._adam_step(gW, gB, lr)

        return total / len(X)

    def _adam_step(self, gW, gB, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * gW[i] ** 2
            self.mB[i] = b1 * self.mB[i] + (1 - b1) * gB[i]
            self.vB[i] = b2 * self.vB[i] + (1 - b2) * gB[i] ** 2

            # Bias correction: the averages start at zero and are biased
            # towards it until rescaled.
            mw = self.mW[i] / (1 - b1 ** self.t)
            vw = self.vW[i] / (1 - b2 ** self.t)
            mb = self.mB[i] / (1 - b1 ** self.t)
            vb = self.vB[i] / (1 - b2 ** self.t)

            self.W[i] -= lr * mw / (np.sqrt(vw) + eps)
            self.B[i] -= lr * mb / (np.sqrt(vb) + eps)

    def mse(self, X, Y):
        return float(np.mean((self.forward(X)[-1] - Y) ** 2))


class Standardizer:
    """Per-column zero mean, unit variance, fitted on the TRAINING set only.

    Fitting on train alone matters here: each method trains on different
    released data, and standardising against test statistics would leak the
    real voltages into every method's inputs.
    """

    def __init__(self, A: np.ndarray):
        self.mean = A.mean(axis=0)
        self.std = A.std(axis=0)
        # Guard columns that never vary (the mask flags) against divide-by-zero.
        self.std = np.where(self.std < 1e-12, 1.0, self.std)

    def transform(self, A: np.ndarray) -> np.ndarray:
        return (A - self.mean) / self.std

    def inverse_scale_mse(self, mse_standardised: float) -> float:
        """Convert an MSE on standardised targets back to per-unit volts
        squared, so the reported number is comparable across methods."""
        return float(mse_standardised * np.mean(self.std ** 2))


def train_and_curve(
    X_train, Y_train, X_test, Y_test,
    epochs: int = 30,
    hidden: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
):
    """Train one seed, returning the test-MSE curve in original units, with
    entry 0 taken before any training."""
    rng = np.random.default_rng(seed)

    sx = Standardizer(X_train)
    sy = Standardizer(Y_train)

    Xtr, Ytr = sx.transform(X_train), sy.transform(Y_train)
    Xte, Yte = sx.transform(X_test), sy.transform(Y_test)

    net = SmallMLP(Xtr.shape[1], Ytr.shape[1], hidden=hidden, seed=seed)

    curve = [sy.inverse_scale_mse(net.mse(Xte, Yte))]
    for _ in range(epochs):
        net.train_epoch(Xtr, Ytr, lr=lr, rng=rng)
        curve.append(sy.inverse_scale_mse(net.mse(Xte, Yte)))

    return np.array(curve)


def run_seeds(
    X_train, Y_train, X_test, Y_test,
    n_seeds: int = 20,
    **kwargs,
):
    """Mean and standard deviation of the test-MSE curve over n_seeds runs.

    Averaging matters: single-seed differences between methods are within
    initialisation noise, so one run proves nothing.
    """
    curves = np.array([
        train_and_curve(X_train, Y_train, X_test, Y_test, seed=s, **kwargs)
        for s in range(n_seeds)
    ])
    return curves.mean(axis=0), curves.std(axis=0)
