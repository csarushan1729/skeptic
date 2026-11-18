"""skeptic: a statistical conscience for AI-driven alpha research."""
from .dsl import DOCS, DSLError, Evaluator, parse
from .ledger import Ledger
from .market import PLANTED, Market, backtest, synthetic_market, true_edge
from .proposers import EvolutionaryProposer, LLMProposer, RandomProposer, TemplateProposer
from .report import markdown
from .stats import deflated_sharpe, effective_trials, haircut_sharpe, probabilistic_sharpe, sharpe
from .study import Audit, Study, VaultResult, audit_returns

__version__ = "0.1.0"
__all__ = [
    "DOCS", "DSLError", "Evaluator", "parse", "Ledger", "PLANTED", "Market", "backtest",
    "synthetic_market", "true_edge", "TemplateProposer", "EvolutionaryProposer", "LLMProposer", "RandomProposer", "markdown",
    "deflated_sharpe", "effective_trials", "haircut_sharpe", "probabilistic_sharpe", "sharpe",
    "Audit", "Study", "VaultResult", "audit_returns",
]
