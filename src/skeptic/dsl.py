"""A small, *safe* language for formulaic alphas.

Expressions look like the ones used on alpha-mining platforms::

    -rank(ts_sum(returns, 5))
    rank(ts_corr(close, volume, 20)) * sign(delta(close, 3))

They are parsed with Python's ``ast`` module and every node is checked against
a whitelist before anything is evaluated: only the operators below, the data
fields that exist, numeric constants, + - * / and unary minus.  There is no
``eval``, no attribute access, no imports; an LLM can propose anything and the
worst outcome is a ``DSLError``.

Every time-series operator looks only backwards, so an alpha computed at date
t uses information up to t; the backtester then trades it at t+1.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd


class DSLError(ValueError):
    pass


MAX_WINDOW = 252
MAX_NODES = 40

# name -> (kind, arity of data arguments, takes a window)
OPERATORS: dict[str, tuple[str, int, bool]] = {
    "ts_mean": ("ts", 1, True), "ts_std": ("ts", 1, True), "ts_sum": ("ts", 1, True),
    "ts_min": ("ts", 1, True), "ts_max": ("ts", 1, True), "ts_rank": ("ts", 1, True),
    "ts_zscore": ("ts", 1, True), "delta": ("ts", 1, True), "delay": ("ts", 1, True),
    "ts_ewm": ("ts", 1, True), "ts_corr": ("ts", 2, True),
    "rank": ("cs", 1, False), "zscore": ("cs", 1, False), "demean": ("cs", 1, False),
    "abs": ("el", 1, False), "sign": ("el", 1, False), "slog": ("el", 1, False),
    "max": ("el", 2, False), "min": ("el", 2, False),
}

DOCS = """\
Fields: one 2-D panel (dates x assets) per data field, e.g. returns, close, volume.
Time-series operators (window d is an integer 1..252, looks backwards only):
  ts_mean(x,d) ts_std(x,d) ts_sum(x,d) ts_min(x,d) ts_max(x,d) ts_rank(x,d)
  ts_zscore(x,d) delta(x,d)=x-delay(x,d) delay(x,d) ts_ewm(x,d) ts_corr(x,y,d)
Cross-sectional operators (per date, across assets): rank(x) zscore(x) demean(x)
Element-wise: abs(x) sign(x) slog(x)=sign(x)*log(1+|x|) max(x,y) min(x,y)
Arithmetic: + - * / and unary minus, numeric constants.
The alpha is turned into a dollar-neutral portfolio (weights proportional to the
de-meaned alpha) and traded the next day."""


@dataclass
class Parsed:
    text: str          # canonical form
    tree: ast.AST
    n_nodes: int
    fields: frozenset


def parse(expr: str, fields) -> Parsed:
    """Validate an expression; raise DSLError with a readable reason if unsafe or invalid."""
    if not isinstance(expr, str) or not expr.strip():
        raise DSLError("empty expression")
    if len(expr) > 500:
        raise DSLError("expression too long")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise DSLError(f"syntax error: {e.msg}") from None
    fields = set(fields)
    used: set[str] = set()
    count = 0

    def check(node, want_window=False):
        nonlocal count
        count += 1
        if count > MAX_NODES:
            raise DSLError(f"expression has more than {MAX_NODES} nodes")
        if want_window:
            if not (isinstance(node, ast.Constant) and isinstance(node.value, int)
                    and not isinstance(node.value, bool)):
                raise DSLError("window arguments must be integer literals")
            if not 1 <= node.value <= MAX_WINDOW:
                raise DSLError(f"window must be in 1..{MAX_WINDOW}")
            return
        if isinstance(node, ast.Expression):
            return check(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise DSLError("only numeric constants are allowed")
            return
        if isinstance(node, ast.Name):
            if node.id not in fields:
                raise DSLError(f"unknown field {node.id!r}; available: {sorted(fields)}")
            used.add(node.id)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise DSLError("only unary minus/plus are allowed")
            return check(node.operand)
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise DSLError("only + - * / are allowed")
            check(node.left)
            return check(node.right)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in OPERATORS:
                name = getattr(node.func, "id", "?")
                raise DSLError(f"unknown operator {name!r}")
            if node.keywords:
                raise DSLError("keyword arguments are not allowed")
            kind, arity, win = OPERATORS[node.func.id]
            if len(node.args) != arity + int(win):
                raise DSLError(f"{node.func.id} takes {arity + int(win)} arguments")
            for a in node.args[:arity]:
                check(a)
            if win:
                check(node.args[arity], want_window=True)
            return
        raise DSLError(f"construct not allowed: {type(node).__name__}")

    check(tree)
    if not used:
        raise DSLError("expression does not use any data field")
    return Parsed(ast.unparse(tree), tree, count, frozenset(used))


class Evaluator:
    """Evaluates parsed expressions on a dict of (dates x assets) DataFrames."""

    def __init__(self, fields: dict[str, pd.DataFrame]):
        self.fields = fields
        self._cache: dict[str, pd.DataFrame] = {}

    def __call__(self, expr: str | Parsed) -> pd.DataFrame:
        p = expr if isinstance(expr, Parsed) else parse(expr, self.fields)
        out = self._eval(p.tree.body)
        if not isinstance(out, pd.DataFrame):
            raise DSLError("expression evaluates to a constant")
        return out.replace([np.inf, -np.inf], np.nan)

    def _eval(self, node):
        key = ast.unparse(node)
        if key in self._cache:
            return self._cache[key]
        val = self._compute(node)
        if isinstance(val, pd.DataFrame) and len(self._cache) < 512:
            self._cache[key] = val
        return val

    def _compute(self, node):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return self.fields[node.id]
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(b, pd.DataFrame):
                b = b.where(b.abs() > 1e-12)
            elif abs(b) < 1e-12:
                raise DSLError("division by zero constant")
            return a / b
        name = node.func.id
        kind, arity, win = OPERATORS[name]
        args = [self._eval(a) for a in node.args[:arity]]
        d = node.args[arity].value if win else None
        x = args[0]
        if not isinstance(x, pd.DataFrame) and name not in ("max", "min"):
            raise DSLError(f"{name} needs a data argument")
        if name == "ts_mean":
            return x.rolling(d, min_periods=d).mean()
        if name == "ts_std":
            return x.rolling(d, min_periods=d).std()
        if name == "ts_sum":
            return x.rolling(d, min_periods=d).sum()
        if name == "ts_min":
            return x.rolling(d, min_periods=d).min()
        if name == "ts_max":
            return x.rolling(d, min_periods=d).max()
        if name == "ts_rank":
            return x.rolling(d, min_periods=d).rank(pct=True)
        if name == "ts_zscore":
            r = x.rolling(d, min_periods=d)
            return (x - r.mean()) / r.std().where(lambda s: s > 1e-12)
        if name == "delta":
            return x - x.shift(d)
        if name == "delay":
            return x.shift(d)
        if name == "ts_ewm":
            return x.ewm(halflife=d, min_periods=d).mean()
        if name == "ts_corr":
            y = args[1]
            if not isinstance(y, pd.DataFrame):
                raise DSLError("ts_corr needs two data arguments")
            return x.rolling(d, min_periods=d).corr(y)
        if name == "rank":
            return x.rank(axis=1, pct=True)
        if name == "zscore":
            return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).where(lambda s: s > 1e-12), axis=0)
        if name == "demean":
            return x.sub(x.mean(axis=1), axis=0)
        if name == "abs":
            return x.abs()
        if name == "sign":
            return np.sign(x)
        if name == "slog":
            return np.sign(x) * np.log1p(x.abs())
        if name in ("max", "min"):
            a, b = args
            if not isinstance(a, pd.DataFrame) and not isinstance(b, pd.DataFrame):
                raise DSLError(f"{name} needs a data argument")
            if not isinstance(a, pd.DataFrame):
                a, b = b, a
            return a.clip(lower=b) if name == "max" else a.clip(upper=b)
        raise DSLError(f"operator {name} not implemented")  # pragma: no cover
