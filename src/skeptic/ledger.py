"""A tamper-evident, append-only record of every research trial.

The single biggest source of false discoveries in systematic research is not a
wrong formula, it is *forgotten trials*: the 400 variants that did not work and
were never written down, so the survivor's Sharpe ratio is judged as if it were
the only attempt.  When an AI agent generates the variants, the count reaches
thousands in an afternoon.

Each line of the ledger is a JSON record carrying the SHA-256 hash of the
previous record, so deleting or editing a past trial breaks the chain and
``verify()`` reports where.  Trial return series are stored compactly
(float32, base64) so the audit can be recomputed from the ledger alone.
"""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import time
from dataclasses import dataclass

import numpy as np

GENESIS = "0" * 64


def _digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encode_returns(r: np.ndarray) -> str:
    return base64.b64encode(np.asarray(r, dtype=np.float32).tobytes()).decode()


def decode_returns(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).astype(float)


@dataclass
class VerifyResult:
    ok: bool
    n_records: int
    first_bad: int | None
    message: str


class Ledger:
    """Append-only JSONL ledger; pass ``path=None`` for an in-memory ledger."""

    def __init__(self, path: str | pathlib.Path | None = None):
        self.path = pathlib.Path(path) if path else None
        self.records: list[dict] = []
        if self.path and self.path.exists():
            with self.path.open() as fh:
                self.records = [json.loads(line) for line in fh if line.strip()]

    # ------------------------------------------------------------- writing
    def append(self, kind: str, **payload) -> dict:
        rec = {
            "seq": len(self.records),
            "time": time.time(),
            "kind": kind,
            "prev": self.records[-1]["hash"] if self.records else GENESIS,
            **payload,
        }
        rec["hash"] = _digest(rec)
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    # ------------------------------------------------------------- reading
    def trials(self) -> list[dict]:
        return [r for r in self.records if r["kind"] == "trial"]

    def vault_openings(self) -> list[dict]:
        return [r for r in self.records if r["kind"] == "vault_open"]

    def returns_matrix(self) -> np.ndarray:
        """T x N matrix of in-sample returns of all successfully evaluated trials."""
        cols = [decode_returns(r["returns"]) for r in self.trials() if r.get("returns")]
        if not cols:
            return np.empty((0, 0))
        return np.column_stack(cols)

    def verify(self) -> VerifyResult:
        prev = GENESIS
        for i, rec in enumerate(self.records):
            if rec.get("seq") != i:
                return VerifyResult(False, len(self.records), i, f"record {i}: sequence number {rec.get('seq')}")
            if rec.get("prev") != prev:
                return VerifyResult(False, len(self.records), i, f"record {i}: broken link to previous record")
            if _digest(rec) != rec.get("hash"):
                return VerifyResult(False, len(self.records), i, f"record {i}: contents were modified")
            prev = rec["hash"]
        return VerifyResult(True, len(self.records), None, "chain intact")

    def __len__(self) -> int:
        return len(self.records)
