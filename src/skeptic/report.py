"""Human-readable verdicts."""
from __future__ import annotations

from .study import GATES, Audit, VaultResult


def _ok(flag: bool) -> str:
    return "pass" if flag else "FAIL"


def markdown(audit: Audit, vault: VaultResult | None = None, title: str = "skeptic audit") -> str:
    a = audit
    lines = [
        f"# {title}",
        "",
        f"**Verdict: {a.verdict}**" + (f" -> vault: **{vault.verdict}**" if vault else ""),
        "",
        f"Best in-sample alpha: `{a.best_expr}`",
        "",
        "| | value | gate |",
        "|---|---:|---|",
        f"| trials counted | {a.n_trials} | |",
        f"| effective independent trials | {a.n_effective:.0f} | |",
        f"| in-sample Sharpe (annualised) | {a.sharpe_annual:.2f} | what a naive report shows |",
        f"| t-statistic | {a.t_stat:.2f} | |",
        f"| best Sharpe expected from pure noise | {a.expected_max_noise_sharpe:.2f} | |",
        f"| Deflated Sharpe Ratio | {a.dsr:.3f} | {_ok(a.dsr >= GATES['dsr'])} (>= {GATES['dsr']}) |",
        f"| Hansen SPA p-value | {a.spa_p:.3f} | {_ok(a.spa_p <= GATES['spa_p'])} (<= {GATES['spa_p']}) |",
        f"| White Reality Check p-value | {a.rc_p:.3f} | |",
    ]
    if a.pbo is not None:
        lines.append(f"| Probability of backtest overfitting | {a.pbo:.2f} | {_ok(a.pbo <= GATES['pbo'])} "
                     f"(<= {GATES['pbo']}) |")
        lines.append(f"| P(in-sample winner loses out-of-sample) | {a.prob_oos_loss:.2f} | |")
    lines += [
        f"| Haircut Sharpe (BHY) | {a.haircut_sharpe_annual:.2f} | |",
        f"| Minimum backtest length for this Sharpe | {a.min_backtest_years:.1f} y | have {a.years_available:.1f} y |",
    ]
    if a.reasons:
        lines += ["", "Why:"] + [f"- {r}" for r in a.reasons]
    if vault:
        lines += ["", f"Vault (opening #{vault.opening_number}): Sharpe {vault.sharpe_annual:.2f}, "
                      f"P(SR>0) = {vault.psr:.3f} over {vault.n_obs} days."]
    return "\n".join(lines) + "\n"
