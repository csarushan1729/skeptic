"""Let a language model mine alphas, with skeptic keeping score.

    pip install "skeptic-alpha[llm]"
    export ANTHROPIC_API_KEY=...
    python examples/llm_agent.py

On real data, pass the model's training cutoff: skeptic refuses a vault that
starts before it, because the model may already "know" what worked there.
"""
from skeptic import LLMProposer, Study, markdown, synthetic_market
from skeptic.proposers import anthropic_complete

market = synthetic_market(100, 2100, {"reversal": 0.015, "volume": 0.015}, seed=7)
study = Study(market, research=(250, 1500), vault=(1520, 2100), cost_bps=1.0,
              ledger="llm_ledger.jsonl", name="llm-agent")
# With dated data:  Study(..., vault=("2025-01-01", "2026-06-30"), model_cutoff="2025-03-31")
# -> raises, because the vault starts before the model's training data ends.

proposer = LLMProposer(anthropic_complete("claude-sonnet-5-5"))
study.run(proposer, n_trials=100, batch_size=10, verbose=True)

audit = study.audit()
vault = study.open_vault(audit.best_expr) if audit.verdict == "CANDIDATE" else None
print(markdown(audit, vault, title="LLM alpha search"))
