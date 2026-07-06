"""
ewma.py
-------
Implements Section 2.5 "Design and implementation of the control chart":
the EWMA recursion (Eq. 8) and time-varying control limits (Eqs. 9-10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ewma(x: np.ndarray, lam: float, z0: float | None = None) -> np.ndarray:
    """Compute the EWMA statistic sequence z_t = lambda*x_t + (1-lambda)*z_{t-1}  (Eq. 8).

    Parameters
    ----------
    x : 1-D array of observed values (e.g. per-patient predicted risk scores
        in time/sequence order)
    lam : smoothing parameter (lambda)
    z0 : initial EWMA value; defaults to the mean of `x`, matching the
         manuscript's statement that "the initial value, z0, is typically
         set to the mean of the historical data."
    """
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
    """Compute UCL_t and LCL_t for t = 1..n  (Eqs. 9-10)."""
    t = np.arange(1, n + 1)
    factor = L * sigma * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t)))
    ucl = mu + factor
    lcl = mu - factor
    return ucl, lcl


def build_ewma_chart(
    x: np.ndarray, mu: float, sigma: float, lam: float, L: float
) -> pd.DataFrame:
    """Build a full EWMA chart: EWMA statistic + time-varying UCL/LCL + out-of-control flag.

    Returns a DataFrame with columns: value, ewma, ucl, lcl, out_of_control.
    """
    x = np.asarray(x, dtype=float)
    z = compute_ewma(x, lam, z0=mu)
    ucl, lcl = compute_control_limits(len(x), mu, sigma, lam, L)
    out_of_control = (z > ucl) | (z < lcl)

    return pd.DataFrame(
        {
            "value": x,
            "ewma": z,
            "ucl": ucl,
            "lcl": lcl,
            "out_of_control": out_of_control,
        }
    )


def count_out_of_control(x: np.ndarray, mu: float, sigma: float, lam: float, L: float) -> int:
    """Fitness-relevant quantity: number of points falling outside the control limits.

    Matches Section 2.6: "The fitness of each chromosome was evaluated using
    a fitness function that assesses control chart performance, defined as
    the number of observations falling outside the control limits."
    """
    chart = build_ewma_chart(x, mu, sigma, lam, L)
    return int(chart["out_of_control"].sum())
