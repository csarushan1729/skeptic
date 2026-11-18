"""Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu, 2017).

Given the return series of every strategy that was tried (a T x N matrix), split
time into S blocks.  For every way of choosing S/2 blocks as "in-sample", pick
the strategy with the best in-sample Sharpe and record where it ranks
out-of-sample.  PBO is the fraction of splits in which the in-sample winner
lands in the bottom half out-of-sample.  PBO ~ 0.5 means selection is no
better than a coin flip; PBO near 0 means the in-sample ranking carries over.

Everything is computed from per-block sums, so all C(S, S/2) splits are
evaluated with two matrix products.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PBOResult:
    pbo: float                    # P(in-sample winner is below the OOS median)
    logits: np.ndarray            # one per split
    is_sharpe: np.ndarray         # in-sample Sharpe of the winner, per split
    oos_sharpe: np.ndarray        # out-of-sample Sharpe of the same strategy
    prob_oos_loss: float          # P(OOS Sharpe of the winner < 0)
    degradation_slope: float      # regression slope of OOS on IS Sharpe
    n_splits: int
    n_strategies: int

    def to_dict(self) -> dict:
        return {"pbo": self.pbo, "prob_oos_loss": self.prob_oos_loss,
                "degradation_slope": self.degradation_slope,
                "n_splits": self.n_splits, "n_strategies": self.n_strategies,
                "median_logit": float(np.median(self.logits))}


def _sharpe_from_sums(s, ss, n):
    mean = s / n
    var = np.maximum((ss - n * mean**2) / np.maximum(n - 1, 1), 1e-18)
    return mean / np.sqrt(var)


def pbo_cscv(returns_matrix, n_blocks: int = 16, max_splits: int | None = 20000,
             rng: np.random.Generator | None = None) -> PBOResult:
    R = np.asarray(returns_matrix, dtype=float)
    R = np.where(np.isfinite(R), R, 0.0)
    T, N = R.shape
    if N < 2:
        raise ValueError("PBO needs at least two strategies")
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be even and >= 2")
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    sums = np.stack([R[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    sqs = np.stack([(R[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges).astype(float)

    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if max_splits is not None and len(combos) > max_splits:
        rng = rng or np.random.default_rng(0)
        pick = rng.choice(len(combos), size=max_splits, replace=False)
        combos = [combos[i] for i in pick]
    C = np.zeros((len(combos), n_blocks))
    for i, c in enumerate(combos):
        C[i, list(c)] = 1.0
    O = 1.0 - C

    sr_is = _sharpe_from_sums(C @ sums, C @ sqs, (C @ cnt)[:, None])
    sr_oos = _sharpe_from_sums(O @ sums, O @ sqs, (O @ cnt)[:, None])
    best = np.argmax(sr_is, axis=1)
    rows = np.arange(len(combos))
    win_oos = sr_oos[rows, best]
    # relative rank in (0, 1): 1 = best out of sample
    rank = (sr_oos < win_oos[:, None]).sum(axis=1) + 1
    omega = rank / (N + 1.0)
    logits = np.log(omega / (1 - omega))
    x, y = sr_is[rows, best], win_oos
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")
    return PBOResult(float(np.mean(logits <= 0)), logits, x, y, float(np.mean(y < 0)), slope,
                     len(combos), N)


def n_cscv_splits(n_blocks: int) -> int:
    return math.comb(n_blocks, n_blocks // 2)
