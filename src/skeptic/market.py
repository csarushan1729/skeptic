"""Market data containers, a synthetic market with *known* ground truth, and a
simple, honest backtester.

The synthetic market is what makes skeptic testable: you can generate a world
with no exploitable signal at all (every discovery is false by construction)
or a world with planted signals of known strength, and check whether a
research process tells them apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PLANTED = {
    # name: (description, DSL expression that approximately captures it)
    "reversal": ("5-day short-term reversal", "-rank(ts_sum(returns, 5))"),
    "momentum": ("60-day momentum", "rank(ts_sum(returns, 60))"),
    "volume": ("abnormal-volume continuation", "rank(slog(volume) - ts_mean(slog(volume), 20))"),
}


@dataclass
class Market:
    fields: dict[str, pd.DataFrame]          # each dates x assets
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.DataFrame:
        return self.fields["returns"]

    @property
    def n_days(self) -> int:
        return len(self.returns.index)

    def slice(self, start: int, end: int) -> "Market":
        return Market({k: v.iloc[start:end] for k, v in self.fields.items()}, dict(self.meta))

    @classmethod
    def from_long_csv(cls, path, date_col="date", asset_col="asset") -> "Market":
        """CSV in long format: one row per (date, asset), one column per field.

        Must contain ``close``; ``returns`` is derived from it if absent.
        """
        df = pd.read_csv(path, parse_dates=[date_col])
        fields = {}
        for col in df.columns:
            if col in (date_col, asset_col):
                continue
            fields[col] = df.pivot(index=date_col, columns=asset_col, values=col).sort_index()
        if "returns" not in fields:
            if "close" not in fields:
                raise ValueError("need a 'close' or 'returns' column")
            fields["returns"] = fields["close"].pct_change()
        return cls(fields, {"source": str(path)})


def synthetic_market(n_assets: int = 100, n_days: int = 2500, signals: dict | None = None,
                     seed: int = 0) -> Market:
    """Factor model with stochastic volatility and fat tails, plus optional
    planted signals.

    r_{t+1,i} = beta_i f_{t+1} + sigma_{t,i} eps_{t+1,i} + sum_k c_k sigma_i s_{k,t,i}

    where s_k are cross-sectional z-scores of the PLANTED signals computed from
    data up to day t and c_k = signals[k] is the strength (information
    coefficient ~ c_k).  ``signals=None`` gives a market with no edge.
    """
    rng = np.random.default_rng(seed)
    signals = signals or {}
    unknown = set(signals) - set(PLANTED)
    if unknown:
        raise ValueError(f"unknown signals {unknown}; choose from {list(PLANTED)}")
    N, T = n_assets, n_days
    beta = rng.uniform(0.5, 1.5, N)
    sig_i = rng.uniform(0.01, 0.03, N)
    log_mvol = np.zeros(T)
    for t in range(1, T):
        log_mvol[t] = 0.97 * log_mvol[t - 1] + 0.15 * rng.standard_normal()
    f = 0.01 * np.exp(log_mvol - log_mvol.var() / 2) * rng.standard_normal(T)
    lv = np.zeros((T, N))
    eps = rng.standard_t(4, size=(T, N)) / np.sqrt(2.0)
    hv = np.zeros(N)
    R = np.zeros((T, N))
    drift = np.zeros((T, N))  # the planted, exploitable part of each return
    logvol = np.zeros((T, N))
    base_lv = rng.uniform(12, 16, N)
    lv_state = np.zeros(N)

    def cs_z(x):
        x = x - x.mean()
        s = x.std()
        return x / s if s > 0 else x * 0

    for t in range(T):
        hv = 0.97 * hv + 0.12 * rng.standard_normal(N)
        sig_t = sig_i * np.exp(hv - 0.12**2 / (2 * (1 - 0.97**2)))
        r = beta * f[t] + sig_t * eps[t]
        if t >= 1 and signals:
            prev = t - 1  # signal observed at close of t-1, earns on day t
            for name, c in signals.items():
                if name == "reversal" and prev >= 4:
                    s = -cs_z(R[prev - 4:prev + 1].sum(axis=0))
                elif name == "momentum" and prev >= 59:
                    s = cs_z(R[prev - 59:prev + 1].sum(axis=0))
                elif name == "volume" and prev >= 19:
                    s = cs_z(logvol[prev] - logvol[prev - 19:prev + 1].mean(axis=0))
                else:
                    continue
                drift[t] += c * sig_i * s
        r = r + drift[t]
        R[t] = r
        lv_state = 0.9 * lv_state + 0.3 * rng.standard_normal(N) + 0.5 * np.abs(eps[t])
        logvol[t] = base_lv + lv_state
    lv[:] = logvol
    idx = pd.RangeIndex(T, name="day")
    cols = [f"A{i:03d}" for i in range(N)]
    returns = pd.DataFrame(R, index=idx, columns=cols)
    close = 100 * (1 + returns).cumprod()
    volume = pd.DataFrame(np.exp(lv), index=idx, columns=cols)
    # the drift is ground truth for scoring only; it is NOT a field, so no alpha can read it
    return Market({"returns": returns, "close": close, "volume": volume},
                  {"synthetic": True, "signals": dict(signals), "seed": seed,
                   "drift": pd.DataFrame(drift, index=idx, columns=cols)})


def alpha_to_weights(alpha: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral, unit-gross weights proportional to the de-meaned alpha."""
    a = alpha.sub(alpha.mean(axis=1), axis=0)
    gross = a.abs().sum(axis=1)
    w = a.div(gross.where(gross > 0), axis=0)
    return w.fillna(0.0)


def backtest(alpha: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    """Daily P&L of trading ``alpha`` with a one-day delay.

    pnl_t = sum_i w_{t-1,i} r_{t,i} - cost * turnover_t.  Returns a frame with
    columns pnl, gross (before costs), turnover.
    """
    w = alpha_to_weights(alpha)
    w_lag = w.shift(1).fillna(0.0)
    gross = (w_lag * returns.fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).shift(1).fillna(0.0)
    pnl = gross - cost_bps * 1e-4 * turnover
    return pd.DataFrame({"pnl": pnl, "gross": gross, "turnover": turnover})


def true_edge(alpha: pd.DataFrame, market: Market, cost_bps: float = 0.0, start: int = 0,
              end: int | None = None) -> float:
    """Exact annualised Sharpe of a strategy's *expected* P&L in a synthetic market.

    Conditional on its positions, a strategy's expected return on day t is
    sum_i w_{t-1,i} * drift_{t,i} minus costs; every other part of the return
    has mean zero.  Dividing by the realised P&L volatility gives the Sharpe
    ratio the strategy "really" has, free of sampling noise.  It is exactly
    zero-before-costs in a market without planted signals.
    """
    drift = market.meta.get("drift")
    if drift is None:
        raise ValueError("true_edge needs a synthetic market")
    w = alpha_to_weights(alpha).shift(1).fillna(0.0)
    bt = backtest(alpha, market.returns, cost_bps)
    exp_pnl = (w * drift).sum(axis=1) - cost_bps * 1e-4 * bt["turnover"]
    sl = slice(start, end)
    vol = bt["pnl"].iloc[sl].std()
    return float(exp_pnl.iloc[sl].mean() / vol * np.sqrt(252)) if vol > 0 else 0.0
