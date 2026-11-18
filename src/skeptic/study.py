"""The research loop with a statistical conscience.

    study = Study(market, research=(252, 1500), vault=(1520, 2000))
    study.run(EvolutionaryProposer(), n_trials=300)   # every trial -> ledger
    audit = study.audit()                              # deflate, PBO, SPA
    if audit.verdict == "CANDIDATE":
        study.open_vault(audit.best_expr)              # one shot, recorded

Rules the Study enforces:

1. Every evaluated idea is written to the ledger before its result is shown to
   anyone (human or agent).  The audit counts *all* of them.
2. Proposers only ever see in-sample results.  The vault period is never
   evaluated during research.
3. The vault can be opened more than once, but every opening is recorded and
   any opening after the first is reported as contaminated.
4. Knowledge-cutoff guard: if the ideas come from a language model trained on
   data up to ``model_cutoff``, a vault that starts before that date is not
   out-of-sample for the model.  The Study refuses it unless
   ``allow_contaminated_vault=True``.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import stats
from .bootstrap import superior_predictive_ability
from .dsl import DSLError, Evaluator, parse
from .ledger import Ledger, encode_returns
from .market import Market, backtest
from .overfit import pbo_cscv
from .proposers import HistoryItem

GATES = {"dsr": 0.95, "spa_p": 0.05, "pbo": 0.5}


@dataclass
class Audit:
    verdict: str                     # "CANDIDATE" or "REJECT"
    reasons: list[str]
    best_expr: str | None
    n_trials: int                    # evaluated trials, as recorded in the ledger
    n_effective: float               # participation-ratio estimate of independent trials
    n_obs: int
    sharpe_annual: float             # in-sample, the number everybody reports
    t_stat: float
    psr: float                       # P(SR > 0) ignoring selection
    dsr: float                       # P(SR > E[max of N noise SRs])
    expected_max_noise_sharpe: float # annualised
    haircut_sharpe_annual: float
    pbo: float | None
    prob_oos_loss: float | None
    spa_p: float
    rc_p: float
    min_backtest_years: float
    years_available: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VaultResult:
    expr: str
    sharpe_annual: float
    psr: float
    t_stat: float
    n_obs: int
    contaminated: bool
    opening_number: int
    confirmed: bool
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def audit_returns(R, names=None, n_trials: int | None = None, periods_per_year: int = 252,
                  n_blocks: int = 16, n_boot: int = 1000, mean_block: float = 10.0,
                  seed: int = 0, sharpe_variance: str = "null") -> Audit:
    """Audit the best of a set of strategies, given the returns of *all* of them.

    ``R`` is a (T x K) array of per-period returns, one column per strategy
    tried.  ``n_trials`` is the total number of ideas evaluated, if larger than
    K (e.g. duplicates or degenerate trials whose returns were not kept).
    Works on any source of strategies: a skeptic ledger, a CSV of backtests
    exported from another platform, a hyper-parameter sweep.

    ``sharpe_variance`` sets V in the expected-maximum formula of the deflated
    Sharpe ratio.  "null" (default) uses 1/(T-1), the variance of a Sharpe
    estimate when the true Sharpe is zero, which is what "how high would the
    best of N noise strategies go?" asks.  "empirical" uses the variance of the
    trials' Sharpe ratios, as in Bailey & Lopez de Prado (2014); it is more
    conservative when some trials carry real signal, because real dispersion
    is then counted as noise.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    R = np.where(np.isfinite(R), R, 0.0)
    T, K = R.shape
    if T < 30:
        raise ValueError("need at least 30 observations")
    names = list(names) if names is not None else [f"strategy_{k}" for k in range(K)]
    n_trials = max(int(n_trials or K), K)
    ppy = periods_per_year
    srs = np.array([stats.sharpe(R[:, k]) for k in range(K)])
    best = int(np.argmax(srs))
    r = R[:, best]
    var_null = 1.0 / (T - 1)
    var_emp = max(float(np.var(srs, ddof=1)), var_null) if K > 1 else var_null
    var_sr = var_null if sharpe_variance == "null" else var_emp
    if K > 1:
        n_eff = stats.effective_trials(R) * (n_trials / K)
    else:
        n_eff = float(n_trials)
    n_eff = float(min(max(n_eff, 1.0), n_trials))
    ds = stats.deflated_sharpe(r, n_eff, var_sr, ppy)
    hc = stats.haircut_sharpe(srs[best], T, max(int(round(n_eff)), 1), "by", ppy)
    rng = np.random.default_rng(seed)
    spa = superior_predictive_ability(R, n_boot=n_boot, mean_block=mean_block, rng=rng)
    pb = pbo_cscv(R, n_blocks=n_blocks, rng=rng) if K >= 2 else None
    sr_ann = stats.annualize(srs[best], ppy)
    minbtl = stats.min_backtest_length(n_eff, sr_ann) if sr_ann > 0 else float("inf")

    reasons = []
    if ds.dsr < GATES["dsr"]:
        reasons.append(f"deflated Sharpe {ds.dsr:.2f} < {GATES['dsr']}: the best of {n_eff:.0f} "
                       f"effective noise trials would be expected to reach a Sharpe of "
                       f"{ds.expected_max_null_annual:.2f}")
    if spa.p_value_spa > GATES["spa_p"]:
        reasons.append(f"SPA p-value {spa.p_value_spa:.3f} > {GATES['spa_p']}: the best strategy is "
                       f"not significantly better than zero once data snooping is accounted for")
    if pb is not None and pb.pbo > GATES["pbo"]:
        reasons.append(f"probability of backtest overfitting {pb.pbo:.2f} > {GATES['pbo']}")
    if sr_ann <= 0:
        reasons.append("best in-sample Sharpe is not positive")
    return Audit(
        "CANDIDATE" if not reasons else "REJECT", reasons, names[best], n_trials, n_eff, T, sr_ann,
        srs[best] * math.sqrt(T), stats.probabilistic_sharpe(r), ds.dsr, ds.expected_max_null_annual,
        hc["haircut_sharpe_annual"], pb.pbo if pb else None, pb.prob_oos_loss if pb else None,
        spa.p_value_spa, spa.p_value_rc, minbtl, T / ppy,
        details={"deflated": ds.to_dict(), "haircut": hc,
                 "dsr_conservative": {
                     "raw_trial_count": stats.deflated_sharpe(r, n_trials, var_sr, ppy).dsr,
                     "empirical_variance": stats.deflated_sharpe(r, n_eff, var_emp, ppy).dsr}, "spa": spa.to_dict(),
                 "pbo": pb.to_dict() if pb else None, "n_stored_trials": K,
                 "top": [{"expr": names[i], "sharpe_annual": stats.annualize(srs[i], ppy)}
                         for i in np.argsort(-srs)[:10]]},
    )


def _to_pos(index: pd.Index, x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    return int(index.searchsorted(pd.Timestamp(x)))


class Study:
    def __init__(self, market: Market, research: tuple, vault: tuple, cost_bps: float = 5.0,
                 ledger: Ledger | str | None = None, periods_per_year: int = 252,
                 model_cutoff=None, allow_contaminated_vault: bool = False, name: str = "study"):
        idx = market.returns.index
        self.market = market
        self.r0, self.r1 = _to_pos(idx, research[0]), _to_pos(idx, research[1])
        self.v0, self.v1 = _to_pos(idx, vault[0]), _to_pos(idx, vault[1])
        if not (0 <= self.r0 < self.r1 <= self.v0 < self.v1 <= len(idx)):
            raise ValueError("need research_start < research_end <= vault_start < vault_end")
        if model_cutoff is not None and _to_pos(idx, model_cutoff) > self.v0:
            msg = (f"vault starts at {idx[self.v0]} but the proposing model was trained on data up to "
                   f"{model_cutoff}: the vault is not out-of-sample for it")
            if not allow_contaminated_vault:
                raise ValueError(msg + " (pass allow_contaminated_vault=True to override)")
            warnings.warn(msg)
        self.cost_bps = cost_bps
        self.ppy = periods_per_year
        self.ledger = ledger if isinstance(ledger, Ledger) else Ledger(ledger)
        self.evaluator = Evaluator(market.fields)
        self.history: list[HistoryItem] = []
        self._seen: set[str] = set()
        self._fields = list(market.fields)
        self.ledger.append("study", name=name, research=[self.r0, self.r1], vault=[self.v0, self.v1],
                           cost_bps=cost_bps, model_cutoff=str(model_cutoff) if model_cutoff else None,
                           fields=self._fields, n_assets=int(market.returns.shape[1]))
        self._returns_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- research
    def _pnl(self, parsed) -> pd.Series:
        alpha = self.evaluator(parsed)
        return backtest(alpha, self.market.returns, self.cost_bps)["pnl"]

    def evaluate(self, expr: str, source: str = "human") -> HistoryItem:
        try:
            p = parse(expr, self._fields)
        except DSLError as e:
            self.ledger.append("rejected", expr=str(expr)[:500], source=source, reason=str(e))
            item = HistoryItem(str(expr), float("nan"), "rejected")
            self.history.append(item)
            return item
        duplicate = p.text in self._seen
        self._seen.add(p.text)
        try:
            pnl = self._pnl(p).iloc[self.r0:self.r1].to_numpy()
        except (DSLError, ValueError, FloatingPointError) as e:
            self.ledger.append("rejected", expr=p.text, source=source, reason=f"evaluation failed: {e}")
            item = HistoryItem(p.text, float("nan"), "rejected")
            self.history.append(item)
            return item
        active = float(np.mean(np.abs(pnl) > 0))
        sr = stats.sharpe(pnl)
        degenerate = active < 0.5 or not np.isfinite(sr) or np.std(pnl) == 0
        status = "degenerate" if degenerate else "ok"
        self.ledger.append(
            "trial", expr=p.text, source=source, status=status, duplicate=duplicate,
            sharpe_annual=stats.annualize(sr, self.ppy), t_stat=sr * math.sqrt(len(pnl)),
            active_fraction=active, n_nodes=p.n_nodes,
            returns=None if (degenerate or duplicate) else encode_returns(pnl),
        )
        item = HistoryItem(p.text, stats.annualize(sr, self.ppy) if not degenerate else float("nan"), status)
        self.history.append(item)
        return item

    def run(self, proposer, n_trials: int, batch_size: int = 10, verbose: bool = False) -> "Study":
        done = 0
        while done < n_trials:
            k = min(batch_size, n_trials - done)
            exprs = proposer.propose(list(self.history), k) or []
            if not exprs:
                break
            for e in exprs[:k]:
                self.evaluate(e, source=getattr(proposer, "name", "unknown"))
            done += len(exprs[:k])
            if verbose:
                best = max((h.sharpe for h in self.history if h.status == "ok"), default=float("nan"))
                print(f"  {done:5d} trials   best in-sample Sharpe {best:+.2f}")
        return self

    # ------------------------------------------------------------- audit
    def audit(self, n_blocks: int = 16, n_boot: int = 1000, mean_block: float = 10.0,
              seed: int = 0) -> Audit:
        stored = [t for t in self.ledger.trials() if t.get("returns")]
        R = self.ledger.returns_matrix()
        audit = audit_returns(R, [t["expr"] for t in stored], n_trials=len(self.ledger.trials()),
                              periods_per_year=self.ppy, n_blocks=n_blocks, n_boot=n_boot,
                              mean_block=mean_block, seed=seed)
        self.ledger.append("audit", **{k: v for k, v in audit.to_dict().items() if k != "details"})
        return audit

    # ------------------------------------------------------------- vault
    def open_vault(self, expr: str, min_psr: float = 0.95) -> VaultResult:
        prior = self.ledger.vault_openings()
        p = parse(expr, self._fields)
        pnl = self._pnl(p).iloc[self.v0:self.v1].to_numpy()
        sr = stats.sharpe(pnl)
        psr = stats.probabilistic_sharpe(pnl)
        contaminated = len(prior) > 0
        confirmed = psr >= min_psr and not contaminated
        if contaminated:
            verdict = (f"CONTAMINATED: the vault was already opened {len(prior)} time(s); this result "
                       f"is no longer out-of-sample evidence")
        elif confirmed:
            verdict = "CONFIRMED out-of-sample"
        else:
            verdict = "NOT CONFIRMED out-of-sample"
        res = VaultResult(p.text, stats.annualize(sr, self.ppy), psr, sr * math.sqrt(len(pnl)), len(pnl),
                          contaminated, len(prior) + 1, confirmed, verdict)
        self.ledger.append("vault_open", **res.to_dict())
        return res
