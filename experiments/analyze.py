"""Turn results/benchmark_rows.json into the summary table and figure in the README."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "results" / "benchmark_rows.json").read_text())
RULES = ["naive", "holdout-peeking", "holdout", "skeptic", "skeptic+vault"]
REAL = 0.3

summary = {}
for prop in ["random", "agent"]:
    for rule in RULES:
        null = [r for r in rows if r["world"] == "null" and r["proposer"] == prop]
        pl = [r for r in rows if r["world"] == "planted" and r["proposer"] == prop]
        disc_null = sum(r["rules"][rule] for r in null)
        disc_pl = [r for r in pl if r["rules"][rule]]
        tru = (lambda r: r["peek"]["true_sharpe"]) if rule == "holdout-peeking" else (lambda r: r["true_sharpe"])
        real_pl = [r for r in disc_pl if tru(r) > REAL]
        all_disc = disc_null + len(disc_pl)
        false_disc = disc_null + len(disc_pl) - len(real_pl)
        summary[f"{prop}/{rule}"] = {
            "null_false_positive_rate": disc_null / max(len(null), 1),
            "planted_power": len(real_pl) / max(len(pl), 1),
            "false_discovery_proportion": false_disc / all_disc if all_disc else 0.0,
            "n_null": len(null), "n_planted": len(pl), "discoveries": all_disc,
        }
extra = {
    "null_mean_best_is_sharpe": {p: float(np.mean([r["is_sharpe"] for r in rows if r["world"] == "null" and r["proposer"] == p])) for p in ["random", "agent"]},
    "null_mean_true_sharpe_of_winner": float(np.mean([r["true_sharpe"] for r in rows if r["world"] == "null"])),
    "null_mean_vault_sharpe_of_winner": float(np.mean([r["vault_sharpe"] for r in rows if r["world"] == "null"])),
    "planted_share_of_winners_really_real": float(np.mean([r["true_sharpe"] > REAL for r in rows if r["world"] == "planted"])),
    "n_seeds": len({r["seed"] for r in rows}),
}
(ROOT / "results" / "benchmark_summary.json").write_text(json.dumps({"rules": summary, "extra": extra}, indent=2))

print(f"{'proposer/rule':24s} {'FP rate (null)':>15s} {'power (planted)':>16s} {'false-discovery share':>22s}")
for k, v in summary.items():
    print(f"{k:24s} {v['null_false_positive_rate']:15.2f} {v['planted_power']:16.2f} {v['false_discovery_proportion']:22.2f}")
print(json.dumps(extra, indent=2))

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for world, mk, col in [("null", "o", "C3"), ("planted", "s", "C0")]:
    for prop, fill in [("random", "none"), ("agent", col)]:
        sel = [r for r in rows if r["world"] == world and r["proposer"] == prop]
        ax[0].scatter([r["is_sharpe"] for r in sel], [r["true_sharpe"] for r in sel], marker=mk,
                      facecolors=fill, edgecolors=col, s=36, label=f"{world} world, {prop}")
ax[0].axhline(0, color="k", lw=0.8)
ax[0].axvline(1.0, color="gray", ls="--", lw=0.8)
ax[0].text(1.02, ax[0].get_ylim()[1] * 0.9 if ax[0].get_ylim()[1] > 0 else 0.5, "naive bar: SR >= 1", fontsize=8)
ax[0].set_xlabel("in-sample Sharpe of the winner (what gets reported)")
ax[0].set_ylabel("true Sharpe of the winner (known from the generator)")
ax[0].set_title("the mirage: winners on pure noise look like real alphas")
ax[0].legend(fontsize=7)

x = np.arange(len(RULES))
for i, prop in enumerate(["random", "agent"]):
    fp = [summary[f"{prop}/{r}"]["null_false_positive_rate"] for r in RULES]
    pw = [summary[f"{prop}/{r}"]["planted_power"] for r in RULES]
    ax[1].bar(x + (i - 0.5) * 0.38, fp, 0.38, label=f"{prop}", color=f"C{3 + i}")
    ax[2].bar(x + (i - 0.5) * 0.38, pw, 0.38, label=f"{prop}", color=f"C{i}")
for a_, t in [(ax[1], "false positives in a market with NO signal"), (ax[2], "real alphas found in the planted market")]:
    a_.set_xticks(x)
    a_.set_xticklabels(RULES, fontsize=8, rotation=15)
    a_.set_ylim(0, 1)
    a_.set_title(t)
    a_.legend(title="proposer", fontsize=8)
ax[1].set_ylabel("share of markets")
ax[1].axhline(0.05, color="k", lw=0.8, ls=":")
fig.tight_layout()
(ROOT / "results").mkdir(exist_ok=True)
fig.savefig(ROOT / "results" / "benchmark.png", dpi=140, bbox_inches="tight")

fig, axc = plt.subplots(figsize=(7, 4.2))
for prop, col in [("random", "C3"), ("agent", "C4")]:
    curves = [r["curve"] for r in rows if r["world"] == "null" and r["proposer"] == prop]
    L = min(len(c) for c in curves)
    M = np.array([[np.nan if v is None else v for v in c[:L]] for c in curves])
    axc.plot(np.arange(1, L + 1), np.nanmedian(M, axis=0), color=col, label=f"{prop}: median best-so-far")
    axc.fill_between(np.arange(1, L + 1), np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0),
                     color=col, alpha=0.2)
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from skeptic.stats import expected_max_sharpe  # noqa: E402

n = np.arange(2, L + 1)
axc.plot(n, [expected_max_sharpe(k, 1 / 1249) * np.sqrt(252) for k in n], "k--", lw=1,
         label="E[max] of n independent noise Sharpes")
axc.set_xlabel("number of alphas tried")
axc.set_ylabel("best in-sample Sharpe so far")
axc.set_title("market with no signal: the best Sharpe climbs anyway")
axc.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "results" / "mirage_curve.png", dpi=140, bbox_inches="tight")
print("figures written")
