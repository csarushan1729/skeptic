"""Data-snooping tests with the stationary bootstrap.

White (2000) Reality Check and Hansen (2005) Superior Predictive Ability test:
"is the best of these K strategies better than the benchmark, after accounting
for the fact that we looked at K of them?"  Inputs are performance
differentials d_{t,k} (strategy return minus benchmark return; use the raw
returns for a zero benchmark).  Serial dependence is preserved with the
Politis-Romano (1994) stationary bootstrap.

Bootstrap means are computed as (counts @ D) / T, where counts[b, t] is how
often observation t was drawn in replicate b, so memory is O(B T + T K) rather
than O(B T K).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def stationary_bootstrap_indices(T: int, n_boot: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, size=n_boot)
    jump = rng.random((n_boot, T)) < p
    fresh = rng.integers(0, T, size=(n_boot, T))
    for t in range(1, T):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def bootstrap_means(D: np.ndarray, n_boot: int, mean_block: float, rng: np.random.Generator,
                    chunk: int = 250) -> np.ndarray:
    T = D.shape[0]
    out = np.empty((n_boot, D.shape[1]))
    for a in range(0, n_boot, chunk):
        b = min(n_boot, a + chunk)
        idx = stationary_bootstrap_indices(T, b - a, mean_block, rng)
        counts = np.zeros((b - a, T))
        np.add.at(counts, (np.repeat(np.arange(b - a), T), idx.ravel()), 1.0)
        out[a:b] = counts @ D / T
    return out


@dataclass
class SnoopingTest:
    statistic: float
    p_value_rc: float          # White's Reality Check (non-studentised)
    p_value_spa: float         # Hansen SPA, consistent version
    p_value_spa_lower: float
    p_value_spa_upper: float
    best_index: int
    n_strategies: int
    n_boot: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def superior_predictive_ability(D, n_boot: int = 1000, mean_block: float = 10.0,
                                rng: np.random.Generator | None = None) -> SnoopingTest:
    """Reality Check and SPA p-values for H0: max_k E[d_k] <= 0."""
    rng = rng or np.random.default_rng(0)
    D = np.asarray(D, dtype=float)
    if D.ndim == 1:
        D = D[:, None]
    D = np.where(np.isfinite(D), D, 0.0)
    T, K = D.shape
    dbar = D.mean(axis=0)
    boot = bootstrap_means(D, n_boot, mean_block, rng)              # (B, K)
    omega = np.sqrt(np.maximum(T * boot.var(axis=0, ddof=1), 1e-18))  # long-run std
    sqT = math.sqrt(T)

    # Reality Check (White 2000): un-studentised max statistic, recentred at dbar
    rc_stat = np.max(sqT * dbar)
    rc_boot = np.max(sqT * (boot - dbar), axis=1)
    p_rc = float(np.mean(rc_boot >= rc_stat))

    # SPA (Hansen 2005): studentised, with three recentring choices
    t_obs = max(float(np.max(sqT * dbar / omega)), 0.0)
    thresh = -math.sqrt(2 * math.log(math.log(T))) if T > 15 else -np.inf
    mu = {
        "lower": np.minimum(dbar, 0.0),
        "consistent": dbar * (sqT * dbar / omega <= thresh),
        "upper": np.zeros(K),
    }
    p = {}
    for name, m in mu.items():
        z = sqT * (boot - dbar + m) / omega
        tb = np.maximum(np.max(z, axis=1), 0.0)
        p[name] = float(np.mean(tb >= t_obs)) if t_obs > 0 else 1.0
    return SnoopingTest(t_obs, p_rc, p["consistent"], p["lower"], p["upper"], int(np.argmax(dbar / omega)),
                        K, n_boot)
