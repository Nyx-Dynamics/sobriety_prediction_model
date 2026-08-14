"""
Immutable, tamper-evident audit log (P12).
===========================================
Sobriety Prediction Model | Nyx Dynamics LLC

Append-only JSONL with a SHA-256 hash chain: each entry commits to the previous
entry's hash, so any insertion, deletion, or edit anywhere in the history breaks
verification. Accountability without surveillance:

  * NO PHI. Entries carry a pseudonymous `subject_ref` (auth.pseudonymize), an
    action, an actor role, an outcome, and — for scoring — the escalation band
    and alert flag only. No feature values, no risk numbers, no free text.
  * Erasure (P5) removes the subject's session PHI but the audit trail persists
    (pseudonymously) — you can prove *that* actions happened, not re-derive who.

The local JSONL sink is a reference. In production, ship each line to WORM /
append-only storage (e.g. object-lock bucket, QLDB) so immutability is enforced
below the app, not just detectable above it.
"""

from __future__ import annotations
import hashlib
import json
import os
import threading
from pathlib import Path

GENESIS = "0" * 64


def _canonical(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def _hash(prev_hash: str, entry: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(entry)).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _last(self) -> tuple[int, str]:
        """(seq, hash) of the final record, or (-1, GENESIS) if empty."""
        if not self.path.exists():
            return -1, GENESIS
        last = None
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if last is None:
            return -1, GENESIS
        rec = json.loads(last)
        return rec["seq"], rec["hash"]

    def record(self, *, action: str, actor: str, subject_ref: str | None = None,
               outcome: str = "ok", ts: float = 0.0, **safe_fields) -> dict:
        """Append one entry. `safe_fields` must be non-PHI (e.g. escalation,
        clinician_alert). Callers pass a pseudonymous subject_ref, never user_id."""
        with self._lock:
            seq, prev_hash = self._last()
            body = {
                "seq": seq + 1,
                "ts": ts,
                "actor": actor,
                "action": action,
                "subject_ref": subject_ref,
                "outcome": outcome,
                **safe_fields,
                "prev_hash": prev_hash,
            }
            body["hash"] = _hash(prev_hash, {k: v for k, v in body.items() if k != "hash"})
            with self.path.open("a") as f:
                f.write(_canonical(body) + "\n")
            return body

    def verify(self) -> tuple[bool, int | None]:
        """Recompute the chain. Returns (ok, first_bad_seq_or_None)."""
        prev_hash = GENESIS
        expected_seq = 0
        if not self.path.exists():
            return True, None
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("seq") != expected_seq or rec.get("prev_hash") != prev_hash:
                    return False, rec.get("seq")
                recomputed = _hash(prev_hash, {k: v for k, v in rec.items() if k != "hash"})
                if recomputed != rec.get("hash"):
                    return False, rec.get("seq")
                prev_hash = rec["hash"]
                expected_seq += 1
        return True, None
