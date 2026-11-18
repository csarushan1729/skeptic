import math

import numpy as np
import pandas as pd
import pytest

from skeptic import stats
from skeptic.bootstrap import stationary_bootstrap_indices, superior_predictive_ability
from skeptic.cv import combinatorial_purged_cv, cpcv_n_paths, purged_kfold
from skeptic.dsl import DSLError, Evaluator, parse
from skeptic.ledger import Ledger, decode_returns, encode_returns
from skeptic.market import PLANTED, alpha_to_weights, backtest, synthetic_market, true_edge
from skeptic.overfit import pbo_cscv
from skeptic.proposers import (EvolutionaryProposer, LLMProposer, RandomProposer, TemplateProposer,
                               extract_expressions)
from skeptic.study import Study, audit_returns

RNG = np.random.default_rng(42)


# ------------------------------------------------------------------ stats
def test_sharpe_std_matches_monte_carlo():
    T, sims = 250, 4000
    srs = np.array([stats.sharpe(RNG.normal(0.05, 1, T)) for _ in range(sims)])
    assert srs.std() == pytest.approx(stats.sharpe_std(0.05, T), rel=0.06)


def test_sharpe_std_fat_tails_larger_than_normal():
    assert stats.sharpe_std(0.1, 500, skew=-1, kurt=8) > stats.sharpe_std(0.1, 500)


def test_expected_max_sharpe_matches_simulation():
    T, N, sims = 500, 100, 300
    maxes = [max(stats.sharpe(RNG.standard_normal(T)) for _ in range(N)) for _ in range(sims)]
    assert np.mean(maxes) == pytest.approx(stats.expected_max_sharpe(N, 1 / (T - 1)), rel=0.06)


def test_psr_is_half_at_benchmark_and_dsr_rarely_passes_on_noise():
    r = RNG.standard_normal(1000)
    assert stats.probabilistic_sharpe(r, stats.sharpe(r)) == pytest.approx(0.5)
    passes = 0
    for _ in range(60):
        R = RNG.standard_normal((750, 50))
        best = R[:, np.argmax([stats.sharpe(R[:, k]) for k in range(50)])]
        passes += stats.deflated_sharpe(best, 50).dsr > 0.95
    assert passes <= 8  # nominal 5%


def test_pvalue_adjustments():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(stats.adjust_pvalues(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])
    assert np.allclose(stats.adjust_pvalues(p, "holm"), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(stats.adjust_pvalues(p, "bh"), [0.02, 0.04, 0.04, 0.02])
    assert np.all(stats.adjust_pvalues(p, "by") >= stats.adjust_pvalues(p, "bh"))


def test_haircut_and_minbtl():
    h = stats.haircut_sharpe(0.1, 1000, 100)
    assert 0 < h["haircut_sharpe_annual"] < h["sharpe_annual"]
    assert stats.min_backtest_length(1000, 1.0) > stats.min_backtest_length(10, 1.0) > 0


def test_effective_trials():
    x = RNG.standard_normal((1000, 1))
    assert stats.effective_trials(np.repeat(x, 20, axis=1)) == pytest.approx(1.0)
    n = stats.effective_trials(RNG.standard_normal((5000, 20)))
    assert 18 < n <= 20


# ------------------------------------------------------------------ overfitting / snooping
def test_pbo_null_is_coin_flip_and_signal_is_low():
    R = RNG.standard_normal((1000, 40)) * 0.01
    assert 0.3 < pbo_cscv(R, 10).pbo < 0.7
    R[:, 7] += 0.004
    assert pbo_cscv(R, 10).pbo < 0.1


def test_stationary_bootstrap_block_length():
    idx = stationary_bootstrap_indices(5000, 4, 10.0, np.random.default_rng(0))
    breaks = np.mean(np.diff(idx, axis=1) != 1)
    assert breaks == pytest.approx(0.1, abs=0.01)


def test_spa_size_and_power():
    rej = sum(superior_predictive_ability(RNG.standard_normal((500, 30)), n_boot=300,
                                          rng=np.random.default_rng(i)).p_value_spa < 0.05 for i in range(20))
    assert rej <= 3
    D = RNG.standard_normal((1000, 30))
    D[:, 3] += 0.15
    assert superior_predictive_ability(D, n_boot=500).p_value_spa < 0.01


# ------------------------------------------------------------------ CV
def test_purged_kfold_has_no_leakage():
    for train, test in purged_kfold(100, 5, label_horizon=5, embargo=3):
        assert not set(train) & set(test)
        a, b = test.min(), test.max() + 1
        assert all(i + 5 <= a or i >= b + 3 for i in train)


def test_cpcv_counts():
    splits = list(combinatorial_purged_cv(120, 6, 2, label_horizon=2, embargo=1))
    assert len(splits) == 15 and cpcv_n_paths(6, 2) == 5
    for train, test, _ in splits:
        assert not set(train) & set(test)


# ------------------------------------------------------------------ ledger
def test_ledger_roundtrip_and_tamper_detection(tmp_path):
    path = tmp_path / "l.jsonl"
    led = Ledger(path)
    r = RNG.standard_normal(50)
    for i in range(5):
        led.append("trial", expr=f"x{i}", status="ok", returns=encode_returns(r + i))
    assert np.allclose(decode_returns(led.records[2]["returns"]), (r + 2).astype(np.float32))
    assert Ledger(path).verify().ok
    lines = path.read_text().splitlines()
    lines[2] = lines[2].replace('"x2"', '"x9"')
    path.write_text("\n".join(lines) + "\n")
    res = Ledger(path).verify()
    assert not res.ok and res.first_bad == 2
    path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")  # delete a record
    assert not Ledger(path).verify().ok


# ------------------------------------------------------------------ DSL
FIELDS = {"returns", "close", "volume"}


@pytest.mark.parametrize("bad", [
    "__import__('os').system('ls')", "close.__class__", "(lambda: 1)()", "foo(close)",
    "ts_mean(close, d)", "ts_mean(close, 1000)", "ts_mean(close, 5.5)", "rank(x=close)",
    "'close'", "close if 1 else volume", "1 + 2", "price", "close ** 2", "[close]",
    "+".join(["close"] * 50),
])
def test_dsl_rejects_unsafe_or_invalid(bad):
    with pytest.raises(DSLError):
        parse(bad, FIELDS)


def test_dsl_has_no_lookahead():
    """An alpha computed on data up to day t must not change when later data arrive."""
    m = synthetic_market(30, 400, seed=1)
    full, cut = Evaluator(m.fields), Evaluator(m.slice(0, 300).fields)
    rp = RandomProposer(seed=7)
    checked = 0
    for e in rp.propose([], 60) + [x for _, x in PLANTED.values()]:
        try:
            a, b = full(e).iloc[:300], cut(e)
        except DSLError:
            continue
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)
        checked += 1
    assert checked > 40


def test_extract_expressions():
    txt = "Here:\n```alpha\n1. -rank(ts_sum(returns, 5))\n- rank(volume)\n# comment\n```"
    assert extract_expressions(txt) == ["-rank(ts_sum(returns, 5))", "rank(volume)"]


# ------------------------------------------------------------------ market
def test_backtest_trades_with_one_day_delay():
    idx = pd.RangeIndex(4)
    ret = pd.DataFrame({"a": [0.0, 0.1, -0.1, 0.2], "b": [0.0, -0.1, 0.1, -0.2]}, index=idx)
    pnl = backtest(ret, ret)["pnl"]  # alpha = today's return, traded tomorrow
    w = alpha_to_weights(ret)
    expected = (w.shift(1).fillna(0) * ret).sum(axis=1)
    assert np.allclose(pnl, expected) and pnl.iloc[1] == 0


def test_true_edge_is_zero_in_null_world_and_positive_for_planted():
    m0 = synthetic_market(50, 600, seed=2)
    assert true_edge(Evaluator(m0.fields)("-rank(ts_sum(returns, 5))"), m0) == pytest.approx(0, abs=1e-12)
    m1 = synthetic_market(50, 600, {"reversal": 0.03}, seed=2)
    assert true_edge(Evaluator(m1.fields)(PLANTED["reversal"][1]), m1) > 1


# ------------------------------------------------------------------ study
def _study(signals=None, **kw):
    m = synthetic_market(60, 1400, signals, seed=5)
    return Study(m, research=(200, 1000), vault=(1010, 1400), cost_bps=0.0, **kw)


def test_study_null_world_rejects():
    s = _study()
    s.run(RandomProposer(seed=1), 80)
    a = s.audit(n_boot=200)
    assert a.verdict == "REJECT" and a.n_trials == len(s.ledger.trials())


def test_study_finds_strong_planted_signal_and_vault_rules():
    s = _study({"volume": 0.04})
    s.run(TemplateProposer(seed=0), 40)
    s.evaluate(PLANTED["volume"][1])
    a = s.audit(n_boot=300)
    assert a.verdict == "CANDIDATE"
    v1 = s.open_vault(a.best_expr)
    assert v1.confirmed and not v1.contaminated
    v2 = s.open_vault(a.best_expr)
    assert v2.contaminated and not v2.confirmed
    assert s.ledger.verify().ok


def test_knowledge_cutoff_guard():
    with pytest.raises(ValueError, match="not out-of-sample"):
        _study(model_cutoff=1200)
    with pytest.warns(UserWarning):
        _study(model_cutoff=1200, allow_contaminated_vault=True)


def test_llm_proposer_with_fake_model_survives_hostile_output():
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return "```alpha\n-rank(ts_sum(returns, 5))\n__import__('os').system('rm -rf /')\nrank(volume)\n```"

    s = _study()
    s.run(LLMProposer(fake), 6, batch_size=3)
    kinds = [r["kind"] for r in s.ledger.records]
    assert kinds.count("rejected") == 2 and kinds.count("trial") == 4
    assert "Results so far" in calls[-1] and "INVALID" in calls[-1]


def test_evolutionary_proposer_outputs_parse():
    s = _study()
    s.run(EvolutionaryProposer(seed=3), 60)
    ok = sum(r["kind"] == "trial" for r in s.ledger.records)
    assert ok >= 50


def test_audit_returns_on_plain_matrix():
    R = RNG.standard_normal((800, 20)) * 0.01
    a = audit_returns(R, n_boot=200)
    assert a.verdict == "REJECT" and a.n_trials == 20
