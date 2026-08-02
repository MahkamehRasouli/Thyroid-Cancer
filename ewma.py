"""EWMA statistic (Eq. 8) and time-varying control limits (Eqs. 9-10); Section 2.5."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ewma(x: np.ndarray, lam: float, z0: float | None = None) -> np.ndarray:
    """EWMA sequence z_t = lam*x_t + (1-lam)*z_{t-1} (Eq. 8). z0 defaults to mean(x)."""
    x = np.asarray(x, dtype=float)
    if z0 is None:
        z0 = x.mean()
    z = np.empty_like(x)
    prev = z0
    for t, xt in enumerate(x):
        prev = lam * xt + (1 - lam) * prev
        z[t] = prev
    return z


def compute_control_limits(
    n: int, mu: float, sigma: float, lam: float, L: float
) -> tuple[np.ndarray, np.ndarray]:
    """Time-varying UCL_t and LCL_t for t = 1..n (Eqs. 9-10)."""
    t = np.arange(1, n + 1)
    factor = L * sigma * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t)))
    return mu + factor, mu - factor


def build_ewma_chart(
    x: np.ndarray,
    mu: float,
    sigma: float,
    lam: float,
    L: float,
    two_sided: bool = True,
) -> pd.DataFrame:
    """EWMA statistic, time-varying limits, and out-of-control flag.

    two_sided=True  -> signal when z > UCL or z < LCL.
    two_sided=False -> one-sided (upper) chart; signal only when z > UCL.
    Returns columns: value, ewma, ucl, lcl, out_of_control.
    """
    x = np.asarray(x, dtype=float)
    z = compute_ewma(x, lam, z0=mu)
    ucl, lcl = compute_control_limits(len(x), mu, sigma, lam, L)
    out_of_control = (z > ucl) | (z < lcl) if two_sided else (z > ucl)
    return pd.DataFrame(
        {"value": x, "ewma": z, "ucl": ucl, "lcl": lcl, "out_of_control": out_of_control}
    )


def count_out_of_control(
    x: np.ndarray, mu: float, sigma: float, lam: float, L: float, two_sided: bool = True
) -> int:
    """Number of points outside the control limits (GA fitness quantity)."""
    chart = build_ewma_chart(x, mu, sigma, lam, L, two_sided=two_sided)
    return int(chart["out_of_control"].sum())
