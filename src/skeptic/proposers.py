"""Where alpha ideas come from: random search, an adaptive (evolutionary) agent,
or a large language model.  All three speak the same tiny interface,

    proposer.propose(history, n) -> list[str]

where ``history`` holds only *in-sample* results.  Nothing a proposer sees can
leak the vault.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .dsl import DOCS, OPERATORS, DSLError, parse

WINDOWS = [2, 3, 5, 10, 20, 40, 60, 120]
TS_UNARY = [k for k, (kind, ar, w) in OPERATORS.items() if kind == "ts" and ar == 1]
CS = ["rank", "zscore", "demean"]
EL = ["abs", "sign", "slog"]


@dataclass
class HistoryItem:
    expr: str
    sharpe: float          # annualised in-sample Sharpe (nan if rejected)
    status: str            # "ok", "rejected", "degenerate"


class RandomProposer:
    """Uniformly random expression trees: the baseline 'monkey with a keyboard'."""

    name = "random"

    def __init__(self, fields=("returns", "close", "volume"), max_depth: int = 3, seed: int = 0):
        self.fields = list(fields)
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

    def _leaf(self) -> str:
        f = self.rng.choice(self.fields)
        return f"slog({f})" if f == "volume" and self.rng.random() < 0.7 else str(f)

    def _gen(self, depth: int) -> str:
        r = self.rng
        if depth <= 0 or r.random() < 0.25:
            return self._leaf()
        u = r.random()
        if u < 0.5:
            op = r.choice(TS_UNARY)
            return f"{op}({self._gen(depth - 1)}, {r.choice(WINDOWS)})"
        if u < 0.6:
            return f"ts_corr({self._gen(depth - 1)}, {self._gen(depth - 1)}, {r.choice(WINDOWS[2:])})"
        if u < 0.75:
            return f"{r.choice(CS)}({self._gen(depth - 1)})"
        if u < 0.85:
            return f"{r.choice(EL)}({self._gen(depth - 1)})"
        op = r.choice(["+", "-", "*", "/"])
        return f"({self._gen(depth - 1)} {op} {self._gen(depth - 1)})"

    def one(self) -> str:
        e = self._gen(self.max_depth)
        if self.rng.random() < 0.5:
            e = f"rank({e})"
        return f"-{e}" if self.rng.random() < 0.5 else e

    def propose(self, history: list[HistoryItem], n: int) -> list[str]:
        return [self.one() for _ in range(n)]


TEMPLATES = [
    "-rank(ts_sum(returns, {d}))",
    "rank(ts_sum(returns, {d}))",
    "rank(slog(volume) - ts_mean(slog(volume), {d}))",
    "-rank(ts_std(returns, {d}))",
    "rank(ts_corr(close, slog(volume), {d}))",
    "-rank(ts_corr(returns, delay(returns, 1), {d}))",
    "rank(ts_rank(close, {d}))",
    "-rank(ts_zscore(close, {d}))",
    "rank(ts_zscore(slog(volume), {d}))",
    "-rank(ts_max(returns, {d}))",
    "rank(ts_mean(returns, {d}) / ts_std(returns, {d}))",
    "-rank(delta(slog(close), {d}))",
]


class TemplateProposer:
    """Classic factor templates with random parameters and combinations.

    A stand-in for the *prior knowledge* an LLM brings: it has read the
    factor-zoo literature, so its first guesses are reversal, momentum,
    volume and volatility variants rather than random noise.
    """

    name = "template"

    def __init__(self, seed: int = 0, combine: float = 0.3):
        self.rng = np.random.default_rng(seed)
        self.combine = combine

    def one(self) -> str:
        r = self.rng
        t = str(r.choice(TEMPLATES)).replace("{d}", str(r.choice(WINDOWS)))
        if r.random() < self.combine:
            u = str(r.choice(TEMPLATES)).replace("{d}", str(r.choice(WINDOWS)))
            t = f"{t} {r.choice(['+', '-'])} {u}"
        return t

    def propose(self, history, n: int) -> list[str]:
        return [self.one() for _ in range(n)]


class _Mutator(ast.NodeTransformer):
    def __init__(self, rng, fields):
        self.rng, self.fields = rng, fields

    def mutate(self, tree: ast.AST) -> ast.AST:
        nodes = [n for n in ast.walk(tree)]
        target = nodes[self.rng.integers(len(nodes))]
        r = self.rng
        if isinstance(target, ast.Constant) and isinstance(target.value, int):
            target.value = int(r.choice(WINDOWS))
        elif isinstance(target, ast.Name):
            target.id = str(r.choice(self.fields))
        elif isinstance(target, ast.Call) and target.func.id in TS_UNARY:
            target.func.id = str(r.choice(TS_UNARY))
        elif isinstance(target, ast.Call) and target.func.id in CS:
            target.func.id = str(r.choice(CS))
        return tree


class EvolutionaryProposer:
    """An adaptive search agent: mutate the best in-sample performers.

    This is the behaviour of any feedback-driven miner (genetic programming,
    or an LLM told "here are your best alphas so far, improve them").  It
    climbs the in-sample Sharpe much faster than random search -- which is
    precisely why its winners need deflating.
    """

    name = "agent"

    def __init__(self, fields=("returns", "close", "volume"), explore: float = 0.25, elite: int = 10,
                 seed: int = 0):
        self.fields = list(fields)
        self.explore = explore
        self.elite = elite
        self.rng = np.random.default_rng(seed)
        self.random = RandomProposer(fields, seed=seed + 1)
        self.prior = TemplateProposer(seed=seed + 2)

    def propose(self, history: list[HistoryItem], n: int) -> list[str]:
        ok = [h for h in history if h.status == "ok" and np.isfinite(h.sharpe)]
        out = []
        for _ in range(n):
            if len(ok) < 5 or self.rng.random() < self.explore:
                out.append(self.prior.one() if self.rng.random() < 0.6 else self.random.one())
                continue
            top = sorted(ok, key=lambda h: h.sharpe, reverse=True)[: self.elite]
            parent = top[self.rng.integers(len(top))].expr
            tree = ast.parse(parent, mode="eval")
            u = self.rng.random()
            if u < 0.15:
                child = f"-({parent})" if not parent.startswith("-") else parent[1:]
            elif u < 0.3:
                op = self.rng.choice(["rank", f"ts_mean", "ts_ewm"])
                child = f"rank({parent})" if op == "rank" else f"{op}({parent}, {self.rng.choice(WINDOWS[:5])})"
            else:
                child = ast.unparse(_Mutator(self.rng, self.fields).mutate(tree))
            out.append(child)
        return out


PROMPT = """You are a quantitative researcher searching for daily equity alphas.
Write alphas in this language and nothing else:

{docs}

Available fields: {fields}

Results so far (annualised in-sample Sharpe, after costs) -- {n} trials:
{history}

Propose {k} NEW alphas that are likely to have a higher in-sample Sharpe.
Return them one per line inside a single ```alpha code block, no commentary."""


class LLMProposer:
    """Asks a language model for alphas.

    ``complete`` is any function ``prompt -> text`` (see ``anthropic_complete``
    and ``openai_complete``), which keeps this class testable offline.  Record
    the model's training cutoff in the Study (``model_cutoff``): a model that
    has read papers published after your vault period starts can 'propose'
    signals it already knows worked there.
    """

    name = "llm"

    def __init__(self, complete: Callable[[str], str], fields=("returns", "close", "volume"),
                 show_top: int = 15, show_bottom: int = 5):
        self.complete = complete
        self.fields = list(fields)
        self.show_top, self.show_bottom = show_top, show_bottom
        self.last_prompt = ""

    def _history_text(self, history: list[HistoryItem]) -> str:
        ok = sorted([h for h in history if h.status == "ok"], key=lambda h: h.sharpe, reverse=True)
        lines = [f"  {h.sharpe:+.2f}  {h.expr}" for h in ok[: self.show_top]]
        if len(ok) > self.show_top:
            lines.append("  ...")
            lines += [f"  {h.sharpe:+.2f}  {h.expr}" for h in ok[-self.show_bottom:]]
        bad = [h for h in history if h.status == "rejected"][-3:]
        lines += [f"  INVALID  {h.expr}" for h in bad]
        return "\n".join(lines) if lines else "  (none yet)"

    def propose(self, history: list[HistoryItem], n: int) -> list[str]:
        self.last_prompt = PROMPT.format(docs=DOCS, fields=", ".join(self.fields), n=len(history),
                                         history=self._history_text(history), k=n)
        return extract_expressions(self.complete(self.last_prompt))[:n]


def extract_expressions(text: str) -> list[str]:
    m = re.search(r"```(?:alpha)?\s*\n(.*?)```", text, flags=re.S)
    body = m.group(1) if m else text
    out = []
    for line in body.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s+", "", line).strip().rstrip(",;")
        if line and not line.startswith("#"):
            out.append(line)
    return out


def anthropic_complete(model: str = "claude-sonnet-5-5", max_tokens: int = 1500) -> Callable[[str], str]:
    import anthropic  # optional dependency

    client = anthropic.Anthropic()

    def complete(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    return complete


def openai_complete(model: str = "gpt-4o", max_tokens: int = 1500) -> Callable[[str], str]:
    import openai  # optional dependency

    client = openai.OpenAI()

    def complete(prompt: str) -> str:
        r = client.chat.completions.create(model=model, max_tokens=max_tokens,
                                           messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""

    return complete


def is_valid(expr: str, fields) -> bool:
    try:
        parse(expr, fields)
        return True
    except DSLError:
        return False
