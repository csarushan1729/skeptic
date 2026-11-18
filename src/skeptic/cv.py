"""Leakage-aware cross-validation splits for financial time series.

Labels in finance span time (a 5-day forward return computed at t uses prices
up to t+5).  Ordinary K-fold leaks those labels from the test fold into
training.  Lopez de Prado (2018, ch. 7 & 12) fixes this with *purging*
(drop training samples whose label window overlaps the test fold) and an
*embargo* (drop samples right after the test fold, where serial correlation
leaks information backwards).  Combinatorial Purged CV (CPCV) repeats this for
every choice of k test groups out of N, which yields many backtest paths
instead of one.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import numpy as np


def _groups(n_samples: int, n_groups: int) -> list[np.ndarray]:
    edges = np.linspace(0, n_samples, n_groups + 1).astype(int)
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _train_mask(n_samples: int, test_groups: list[np.ndarray], label_horizon: int,
                embargo: int) -> np.ndarray:
    mask = np.ones(n_samples, dtype=bool)
    for g in test_groups:
        a, b = int(g[0]), int(g[-1]) + 1
        # sample i uses labels on [i, i + label_horizon); purge if that hits [a, b)
        mask[max(0, a - label_horizon + 1):b] = False
        mask[b:min(n_samples, b + embargo)] = False
    return mask


def purged_kfold(n_samples: int, n_splits: int = 5, label_horizon: int = 1,
                 embargo: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for g in _groups(n_samples, n_splits):
        mask = _train_mask(n_samples, [g], label_horizon, embargo)
        yield np.nonzero(mask)[0], g


def combinatorial_purged_cv(n_samples: int, n_groups: int = 6, n_test_groups: int = 2,
                            label_horizon: int = 1, embargo: int = 0
                            ) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    groups = _groups(n_samples, n_groups)
    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test = [groups[i] for i in combo]
        mask = _train_mask(n_samples, test, label_horizon, embargo)
        yield np.nonzero(mask)[0], np.concatenate(test), combo


def cpcv_n_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of complete backtest paths CPCV produces: k/N * C(N, k)."""
    return math.comb(n_groups, n_test_groups) * n_test_groups // n_groups
