"""Shared setup for BNP experiment runners; experiment settings stay in callers."""

from __future__ import annotations

import numpy as np
import opendssdirect as dss

from dpvolt.loads import LoadModel, assign_classes, fit_load_model, make_historical


def print_banner(text: str, width: int) -> None:
    """Print the caller's section heading using its existing separator width."""
    print()
    print("=" * width)
    print(text)
    print("=" * width)


def feeder_load_ratings(master: str) -> tuple[np.ndarray, np.ndarray]:
    """Solve a feeder and return load kW and power-factor angles in DSS order."""
    dss.Text.Command("Clear")
    dss.Text.Command(f"Redirect {master}")
    dss.Text.Command("Solve")
    kw, pf = [], []
    i = dss.Loads.First()
    while i > 0:
        kw.append(dss.Loads.kW())
        pf.append(dss.Loads.PF())
        i = dss.Loads.Next()
    return np.array(kw), np.arccos(np.clip(np.array(pf), -1.0, 1.0))


def build_history(
    kw: np.ndarray,
    theta: np.ndarray,
    n_days: int,
    rng: np.random.Generator,
) -> tuple[dict[int, np.ndarray], np.ndarray, LoadModel]:
    """Build the existing three-class archive and fit, using the caller's RNG."""
    classes = assign_classes(kw, L=3)
    archive = make_historical(kw, classes, n_days=n_days, rng=rng)
    model = fit_load_model(archive, classes, theta)
    return classes, archive, model
