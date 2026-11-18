"""Sharpe-ratio statistics that account for non-normality and multiple testing.

All Sharpe ratios inside this module are *per period* (daily if the returns are
daily) unless a function says otherwise; ``annualize`` converts.

References
----------
- Lo (2002), "The Statistics of Sharpe Ratios", FAJ.
- Mertens (2002); Opdyke (2007): standard error of the Sharpe ratio under
  skewness and kurtosis.
- Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier"
  (Probabilistic Sharpe Ratio).
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", JPM.
- Bailey, Borwein, Lopez de Prado & Zhu (2014), "Pseudo-Mathematics and
  Financial Charlatanism", Notices AMS (minimum backtest length).
- Harvey & Liu (2015), "Backtesting", JPM (Sharpe haircuts).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def _clean(r) -> np.ndarray:
    r = np.asarray(r, dtype=float).ravel()
    return r[np.isfinite(r)]


def sharpe(r, periods_per_year: float = 1.0) -> float:
    """Sample Sharpe ratio mean/std (ddof=1), optionally annualised."""
    r = _clean(r)
    if r.size < 2:
        return 0.0
    s = r.std(ddof=1)
    return 0.0 if s == 0 else float(r.mean() / s * math.sqrt(periods_per_year))


def annualize(sr_per_period: float, periods_per_year: float = 252) -> float:
    return sr_per_period * math.sqrt(periods_per_year)


def higher_moments(r) -> tuple[float, float]:
    """(skewness, kurtosis) with kurtosis *not* in excess form (normal = 3)."""
    r = _clean(r)
    if r.size < 4 or r.std() == 0:
        return 0.0, 3.0
    return float(st.skew(r)), float(st.kurtosis(r, fisher=False))


def sharpe_std(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a per-period Sharpe estimate (Mertens 2002).

    Var(SR_hat) = (1 - skew*SR + (kurt - 1)/4 * SR^2) / (T - 1)
    """
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    return math.sqrt(max(v, 1e-12) / max(n_obs - 1, 1))


def probabilistic_sharpe(r, sr_benchmark: float = 0.0) -> float:
    """PSR: probability that the true per-period Sharpe exceeds ``sr_benchmark``."""
    r = _clean(r)
    sr = sharpe(r)
    g3, g4 = higher_moments(r)
    return float(st.norm.cdf((sr - sr_benchmark) / sharpe_std(sr, r.size, g3, g4)))


def expected_max_sharpe(n_trials: float, var_sr: float) -> float:
    """E[max of n_trials Sharpe estimates] when every true Sharpe is zero.

    The "False Strategy Theorem" approximation (Bailey & Lopez de Prado):
        sqrt(V) * ((1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)))
    with g the Euler-Mascheroni constant and V the variance of the Sharpe
    estimates across trials.
    """
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    z1 = st.norm.ppf(1.0 - 1.0 / n)
    z2 = st.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


@dataclass
class DeflatedSharpe:
    sharpe: float              # observed per-period Sharpe of the selected strategy
    sharpe_annual: float
    n_trials: float            # number of (effective) independent trials
    var_trial_sharpe: float    # V used in the expected-maximum formula
    expected_max_null: float   # SR_0: the best Sharpe you'd expect from pure noise
    expected_max_null_annual: float
    dsr: float                 # P(true SR > SR_0)
    skew: float
    kurt: float
    n_obs: int

    def to_dict(self) -> dict:
        return asdict(self)


def deflated_sharpe(r, n_trials: float, var_trial_sharpe: float | None = None,
                    periods_per_year: float = 252) -> DeflatedSharpe:
    """Deflated Sharpe Ratio of the *selected* strategy ``r``.

    ``var_trial_sharpe`` is the cross-sectional variance of the per-period
    Sharpe ratios of all trials.  If omitted, the variance of a Sharpe estimate
    under the null, 1/(T-1), is used, which is the right value when every trial
    is noise.
    """
    r = _clean(r)
    T = r.size
    v = 1.0 / max(T - 1, 1) if var_trial_sharpe is None else float(var_trial_sharpe)
    sr0 = expected_max_sharpe(n_trials, v)
    sr = sharpe(r)
    g3, g4 = higher_moments(r)
    dsr = float(st.norm.cdf((sr - sr0) / sharpe_std(sr, T, g3, g4)))
    return DeflatedSharpe(sr, annualize(sr, periods_per_year), float(n_trials), v, sr0,
                          annualize(sr0, periods_per_year), dsr, g3, g4, T)


def adjust_pvalues(p, method: str = "holm") -> np.ndarray:
    """Family-wise / false-discovery adjustments: bonferroni, holm, bh, by."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    ps = p[order]
    if method == "bonferroni":
        adj = np.minimum(ps * m, 1.0)
    elif method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ps)
        adj = np.minimum(adj, 1.0)
    elif method in ("bh", "by"):
        c = 1.0 if method == "bh" else float(np.sum(1.0 / np.arange(1, m + 1)))
        adj = ps * m * c / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        adj = np.minimum(adj, 1.0)
    else:
        raise ValueError(f"unknown method {method!r}")
    out = np.empty(m)
    out[order] = adj
    return out


def haircut_sharpe(sr_per_period: float, n_obs: int, n_trials: int, method: str = "by",
                   periods_per_year: float = 252) -> dict:
    """Harvey-Liu haircut: deflate a Sharpe ratio for having been the best of N.

    The Sharpe is turned into a t-statistic (t = SR sqrt(T)), its two-sided
    p-value is adjusted as if it were the smallest of N p-values, and the
    adjusted p-value is turned back into a Sharpe ratio.
    """
    t = sr_per_period * math.sqrt(n_obs)
    p = 2 * st.norm.sf(abs(t))
    n = max(int(n_trials), 1)
    if method == "bonferroni" or method == "holm":   # identical for the smallest p-value
        p_adj = min(p * n, 1.0)
    elif method == "bh":
        p_adj = min(p * n, 1.0)
    elif method == "by":
        p_adj = min(p * n * float(np.sum(1.0 / np.arange(1, n + 1))), 1.0)
    else:
        raise ValueError(method)
    t_adj = st.norm.isf(p_adj / 2) if p_adj < 1 else 0.0
    sr_adj = math.copysign(t_adj / math.sqrt(n_obs), sr_per_period)
    return {
        "p_value": p, "p_adjusted": p_adj, "method": method,
        "sharpe_annual": annualize(sr_per_period, periods_per_year),
        "haircut_sharpe_annual": annualize(sr_adj, periods_per_year),
        "haircut_fraction": 1 - (sr_adj / sr_per_period) if sr_per_period else 0.0,
    }


def min_backtest_length(n_trials: float, target_sharpe_annual: float) -> float:
    """Years of data needed so that the best of N noise strategies is not
    *expected* to reach ``target_sharpe_annual`` in-sample (Bailey et al. 2014)."""
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    e = (1 - EULER_GAMMA) * st.norm.ppf(1 - 1 / n) + EULER_GAMMA * st.norm.ppf(1 - 1 / (n * math.e))
    return float((e / target_sharpe_annual) ** 2)


def effective_trials(returns_matrix) -> float:
    """Effective number of independent trials, by the participation ratio of
    the eigenvalues of the trials' correlation matrix: (sum l)^2 / sum l^2.

    N perfectly correlated strategies count as 1; N independent ones as ~N.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2 or R.shape[1] < 2:
        return 1.0
    R = np.where(np.isfinite(R), R, 0.0)
    keep = R.std(axis=0) > 0
    R = R[:, keep]
    if R.shape[1] < 2:
        return 1.0
    C = np.corrcoef(R, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(np.clip(lam.sum() ** 2 / np.sum(lam**2), 1.0, R.shape[1]))
