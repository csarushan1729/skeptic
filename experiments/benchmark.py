"""The benchmark: does skeptic tell real alphas from mirages?

Two kinds of synthetic market, many seeds each:

  null     no exploitable signal at all; every "discovery" is false
  planted  two real signals (short-term reversal, abnormal volume), each worth
           a net Sharpe of roughly 1 to 1.5 at 1 bp costs

In each market a proposer runs 200 alpha trials on 5 years of in-sample data
(2 years are sealed in the vault).  Two proposers:

  random   random expression trees
  agent    adaptive search seeded with classic factor templates, mutating its
           best in-sample performers (how a feedback-driven LLM agent behaves)

Then four decision rules judge the in-sample winner:

  naive          in-sample Sharpe >= 1 and t >= 2      (the screenshot test)
  holdout        naive, and the vault confirms (P(SR>0) >= 0.95, opened once)
  skeptic        Deflated Sharpe >= 0.95, SPA p <= 0.05, PBO <= 0.5 (in-sample only)
  skeptic+vault  skeptic, and the vault confirms

Truth comes from the market generator: the exact expected P&L of the chosen
strategy (true_edge), so "is it real?" has a noise-free answer.
A discovery is *real* if its true Sharpe after costs exceeds 0.3.
"""
import json
import pathlib
import sys
import time

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skeptic import EvolutionaryProposer, RandomProposer, Study, synthetic_market, true_edge  # noqa: E402

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
TRIALS = 200
COST = 1.0
WORLDS = {"null": None, "planted": {"reversal": 0.015, "volume": 0.015}}
PROPOSERS = {"random": RandomProposer, "agent": EvolutionaryProposer}
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

path = OUT / "benchmark_rows.json"
rows = json.loads(path.read_text()) if path.exists() else []  # resumable
done = {(r["seed"], r["world"], r["proposer"]) for r in rows}
t0 = time.time()
for seed in range(SEEDS):
    for world, sig in WORLDS.items():
        m = synthetic_market(100, 2100, sig, seed=1000 + seed)
        for pname, P in PROPOSERS.items():
            if (seed, world, pname) in done:
                continue
            s = Study(m, research=(250, 1500), vault=(1520, 2100), cost_bps=COST)
            s.run(P(seed=seed), TRIALS, batch_size=20)
            a = s.audit(n_boot=500, seed=seed)
            v = s.open_vault(a.best_expr)
            truth = true_edge(s.evaluator(a.best_expr), m, COST, 250, 2100)
            curve, best = [], -np.inf
            for h in s.history:
                if h.status == "ok" and np.isfinite(h.sharpe):
                    best = max(best, h.sharpe)
                curve.append(best if np.isfinite(best) else None)
            naive = a.sharpe_annual >= 1.0 and a.t_stat >= 2.0
            # "holdout peeking": the common practice of checking the top in-sample ideas against the
            # holdout one after another until one passes. Computed outside the Study on purpose.
            from skeptic import stats as _st
            peek = {"found": False, "true_sharpe": None, "looks": 0}
            for item in a.details["top"]:
                t_is = item["sharpe_annual"] / np.sqrt(252) * np.sqrt(1250)
                if item["sharpe_annual"] < 1.0 or t_is < 2.0:
                    continue
                peek["looks"] += 1
                pnl = s._pnl(item["expr"]).iloc[1520:2100].to_numpy()
                if _st.probabilistic_sharpe(pnl) >= 0.95:
                    peek.update(found=True, expr=item["expr"],
                                true_sharpe=true_edge(s.evaluator(item["expr"]), m, COST, 250, 2100))
                    break
            rows.append({
                "seed": seed, "world": world, "proposer": pname, "expr": a.best_expr,
                "is_sharpe": a.sharpe_annual, "t": a.t_stat, "n_trials": a.n_trials, "n_eff": a.n_effective,
                "dsr": a.dsr, "spa_p": a.spa_p, "pbo": a.pbo, "haircut": a.haircut_sharpe_annual,
                "vault_sharpe": v.sharpe_annual, "vault_psr": v.psr, "true_sharpe": truth,
                "dsr_raw_n": a.details["dsr_conservative"]["raw_trial_count"],
                "peek": peek,
                "rules": {"naive": naive, "holdout": naive and v.confirmed, "holdout-peeking": peek["found"],
                          "skeptic": a.verdict == "CANDIDATE",
                          "skeptic+vault": a.verdict == "CANDIDATE" and v.confirmed},
                "curve": curve,
            })
            print(f"[{time.time() - t0:6.0f}s] seed {seed:2d} {world:7s} {pname:6s} IS {a.sharpe_annual:5.2f} "
                  f"vault {v.sharpe_annual:5.2f} true {truth:5.2f} DSR {a.dsr:.2f} SPA {a.spa_p:.3f} "
                  f"PBO {a.pbo:.2f} -> {a.verdict}", flush=True)
            path.write_text(json.dumps(rows, default=float))
print("done")
