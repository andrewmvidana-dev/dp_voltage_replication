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


# =============================================================================
# SECTION 1 -- Figure 2: statistical fidelity
# =============================================================================

def voltage_wasserstein(V_true: np.ndarray, V_released: np.ndarray) -> float:
   
    mag_true = np.abs(V_true)
    mag_rel = np.abs(V_released)

    if mag_true.shape[1] != mag_rel.shape[1]:
        raise ValueError("the two voltage sets must cover the same nodes")

    per_node = [
        wasserstein_distance(mag_true[:, j], mag_rel[:, j])
        for j in range(mag_true.shape[1])
    ]
    return float(np.mean(per_node))


def empirical_voltage_sensitivity(
    runner,
    model,
    theta: np.ndarray,
    rng: np.random.Generator,
    n_trials: int = 12,
) -> float:
   
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


# =============================================================================
# SECTION 2 -- Figure 3: does the released data train a useful model?
# =============================================================================

def build_masked_dataset(
    V: np.ndarray,
    d_w: int = 48,
    mask_len: int = 24,
    stride: int = 6,
):
   
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
   

    def __init__(self, d_in: int, d_out: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        dims = [d_in, hidden, hidden, d_out]

        # He initialisation: scale the starting weights by sqrt(2 / fan_in).
        # Too large and activations explode through the layers; too small and
        # the signal dies. This scaling is the standard choice for ReLU.
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
            # ReLU on the hidden layers; the output layer stays linear,
            # because we are predicting real-valued voltages, not classes.
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

            # Mean squared error, and its derivative with respect to pred.
            err = pred - yb
            total += float(np.mean(err ** 2)) * len(idx)
            grad = 2.0 * err / len(idx)

            # Backward pass: walk the layers in reverse.
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

            # Bias correction: the running averages start at zero, so early
            # steps are biased towards zero until this rescaling fixes them.
            mw = self.mW[i] / (1 - b1 ** self.t)
            vw = self.vW[i] / (1 - b2 ** self.t)
            mb = self.mB[i] / (1 - b1 ** self.t)
            vb = self.vB[i] / (1 - b2 ** self.t)

            self.W[i] -= lr * mw / (np.sqrt(vw) + eps)
            self.B[i] -= lr * mb / (np.sqrt(vb) + eps)

    def mse(self, X, Y):
        return float(np.mean((self.forward(X)[-1] - Y) ** 2))


class Standardizer:
    

    def __init__(self, A: np.ndarray):
        self.mean = A.mean(axis=0)
        self.std = A.std(axis=0)
        # Guard against columns that never vary, which would divide by zero.
        self.std = np.where(self.std < 1e-12, 1.0, self.std)

    def transform(self, A: np.ndarray) -> np.ndarray:
        return (A - self.mean) / self.std

    def inverse_scale_mse(self, mse_standardised: float) -> float:
        """Convert an MSE measured on standardised targets back to the
        original units, so the reported number is in per-unit volts squared
        and comparable across methods."""
        return float(mse_standardised * np.mean(self.std ** 2))


def train_and_curve(
    X_train, Y_train, X_test, Y_test,
    epochs: int = 30,
    hidden: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
):
    
    rng = np.random.default_rng(seed)

    sx = Standardizer(X_train)
    sy = Standardizer(Y_train)

    Xtr, Ytr = sx.transform(X_train), sy.transform(Y_train)
    Xte, Yte = sx.transform(X_test), sy.transform(Y_test)

    net = SmallMLP(Xtr.shape[1], Ytr.shape[1], hidden=hidden, seed=seed)

    curve = [sy.inverse_scale_mse(net.mse(Xte, Yte))]   # before any training
    for _ in range(epochs):
        net.train_epoch(Xtr, Ytr, lr=lr, rng=rng)
        curve.append(sy.inverse_scale_mse(net.mse(Xte, Yte)))

    return np.array(curve)


def run_seeds(
    X_train, Y_train, X_test, Y_test,
    n_seeds: int = 20,
    **kwargs,
):
    
    curves = np.array([
        train_and_curve(X_train, Y_train, X_test, Y_test, seed=s, **kwargs)
        for s in range(n_seeds)
    ])
    return curves.mean(axis=0), curves.std(axis=0)
