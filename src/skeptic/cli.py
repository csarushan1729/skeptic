"""Command line interface.

  skeptic demo [--world null|planted] [--trials 300] [--proposer agent|random|template]
  skeptic audit backtests.csv [--periods 252] [--trials N] [--date-col date]
  skeptic verify ledger.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .ledger import Ledger
from .market import synthetic_market, true_edge
from .proposers import EvolutionaryProposer, RandomProposer, TemplateProposer
from .report import markdown
from .study import Study, audit_returns


def _demo(args) -> int:
    signals = None if args.world == "null" else {"reversal": 0.015, "volume": 0.015}
    m = synthetic_market(args.assets, 2100, signals, seed=args.seed)
    study = Study(m, research=(250, 1500), vault=(1520, 2100), cost_bps=1.0, ledger=args.ledger,
                  name=f"demo-{args.world}")
    prop = {"agent": EvolutionaryProposer, "random": RandomProposer, "template": TemplateProposer}[args.proposer]
    print(f"world: {args.world}  ({'no exploitable signal' if signals is None else 'planted signals: ' + ', '.join(signals)})")
    print(f"searching {args.trials} alphas with the '{args.proposer}' proposer ...")
    study.run(prop(seed=args.seed), args.trials, verbose=True, batch_size=max(args.trials // 10, 1))
    audit = study.audit()
    vault = study.open_vault(audit.best_expr)
    print()
    print(markdown(audit, vault, title=f"skeptic demo ({args.world} world)"))
    truth = true_edge(study.evaluator(audit.best_expr), m, 1.0, 1520, 2100)
    print(f"Ground truth (known only because the market is synthetic): the selected alpha's true "
          f"Sharpe is {truth:.2f}.")
    return 0


def _audit(args) -> int:
    df = pd.read_csv(args.csv)
    if args.date_col and args.date_col in df.columns:
        df = df.set_index(args.date_col)
    df = df.select_dtypes(include=[np.number])
    a = audit_returns(df.to_numpy(), list(df.columns), n_trials=args.trials,
                      periods_per_year=args.periods, n_boot=args.boot)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2, default=float))
    else:
        print(markdown(a, title=f"skeptic audit of {args.csv}"))
    return 0 if a.verdict == "CANDIDATE" else 1


def _verify(args) -> int:
    res = Ledger(args.ledger).verify()
    print(("OK: " if res.ok else "TAMPERED: ") + f"{res.n_records} records, {res.message}")
    return 0 if res.ok else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="skeptic", description="A statistical conscience for alpha research.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run a search on a synthetic market and audit it")
    d.add_argument("--world", choices=["null", "planted"], default="null")
    d.add_argument("--trials", type=int, default=300)
    d.add_argument("--proposer", choices=["agent", "random", "template"], default="agent")
    d.add_argument("--assets", type=int, default=100)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--ledger", default=None, help="write the ledger to this JSONL file")
    d.set_defaults(func=_demo)
    a = sub.add_parser("audit", help="audit a CSV with one column of returns per strategy tried")
    a.add_argument("csv")
    a.add_argument("--periods", type=int, default=252, help="periods per year (252 daily, 52 weekly, 12 monthly)")
    a.add_argument("--trials", type=int, default=None, help="total ideas tried, if more than the columns")
    a.add_argument("--date-col", default="date")
    a.add_argument("--boot", type=int, default=1000)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=_audit)
    v = sub.add_parser("verify", help="check a ledger's hash chain")
    v.add_argument("ledger")
    v.set_defaults(func=_verify)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
