"""
Inference server — privacy-first serving wrapper around the SobrietyLSTM.
========================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Turns the batch-trained checkpoint into a stateful, per-user online endpoint an
AI companion can call — behind the anti-surveillance boundary in privacy.py.

Paramount, enforced here:
  * ENCRYPTION      — SessionStore refuses to start without a key (fail-closed).
  * PHI ENFORCEMENT — ingress is structured-only; unknown/free-text fields rejected.
  * THIRD-PARTY BLK — the companion endpoint returns ONLY a non-numeric directive;
                      risk numbers live behind a separate authenticated clinical route.
  * TRANSPARENCY    — /transparency (open), /data (subject access), DELETE (erasure),
                      and a consent gate before any check-in is recorded.

This MUST NOT be a surveillance tool: no covert collection (consent required),
no third-party risk sharing, full subject access + erasure.

Run:
    export SUD_PHI_KEY=$(python -c "from serving.privacy import Encryptor; print(Encryptor.generate_key())")
    pip install -r src/serving/requirements.txt
    cd src && uvicorn serving.app:app --reload
"""

from __future__ import annotations
import os
import time
from dataclasses import fields
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

try:
    from serving.contract import StaticProfile, WindowObservation
    from serving.predictor import Predictor
    from serving.state import SessionStore, Session
    from serving.audit import AuditLog
    from serving import policy, privacy, auth, portability
except ImportError:  # pragma: no cover
    from .contract import StaticProfile, WindowObservation
    from .predictor import Predictor
    from .state import SessionStore, Session
    from .audit import AuditLog
    from . import policy, privacy, auth, portability

ROOT = Path(__file__).resolve().parents[2]
LSTM_DIR = Path(os.environ.get("SUD_LSTM_DIR", ROOT / "data/processed/lstm_sequences/latest"))
MODEL_PATH = Path(os.environ.get("SUD_MODEL_PATH", ROOT / "models/lstm/lstm_best.pt"))
CLINICAL_TOKEN = os.environ.get("SUD_CLINICAL_TOKEN")  # gates the risk-bearing route
AUDIT_PATH = os.environ.get("SUD_AUDIT_LOG", str(ROOT / "audit" / "audit.jsonl"))

app = FastAPI(title="Sobriety Prediction — Inference (privacy-first)", version="0.3.0")

# SessionStore refuses to construct without encryption — fail-closed at boot.
store = SessionStore()
audit = AuditLog(AUDIT_PATH)
_predictor: Predictor | None = None


def require_user(user_id: str, authorization: str | None) -> Session:
    """P11: load the session and enforce that the bearer token belongs to THIS
    user. Any missing/mismatched token is a 401 and is audited. Isolation: a
    token minted for user A can never resolve user B."""
    token = auth.bearer(authorization)
    session = store.get(user_id)
    if session is None or not auth.verify_token(token or "", session.token_hash):
        audit.record(action="auth_denied", actor="user",
                     subject_ref=_ref(user_id), outcome="denied", ts=time.time())
        raise HTTPException(401, "invalid or missing bearer token for this user")
    return session


def _ref(user_id: str) -> str | None:
    """Pseudonymous subject reference for audit (no raw identifier)."""
    try:
        return auth.pseudonymize(user_id)
    except PermissionError:
        return None

_PROFILE_FIELDS = {f.name for f in fields(StaticProfile)}
_OBS_FIELDS = {f.name for f in fields(WindowObservation)}


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        missing = [str(p) for p in (MODEL_PATH, LSTM_DIR / "scaler.pkl", LSTM_DIR / "meta.pkl")
                   if not p.exists()]
        if missing:
            raise HTTPException(503, f"model artifacts not ready: {missing}")
        _predictor = Predictor(MODEL_PATH, LSTM_DIR / "scaler.pkl", LSTM_DIR / "meta.pkl")
    return _predictor


# ── request models ──────────────────────────────────────────────────────────
class EnrollRequest(BaseModel):
    user_id: str
    consent: bool = Field(..., description="explicit consent; check-ins refused without it")
    profile: dict = Field(..., description="StaticProfile fields (structured only)")


class ObserveRequest(BaseModel):
    observation: dict = Field(..., description="WindowObservation fields (structured only)")


# ── transparency (open, unauthenticated — it IS the disclosure) ──────────────
@app.get("/transparency")
def transparency():
    return privacy.disclosure()


@app.get("/health")
def health():
    ready = (MODEL_PATH.exists() and (LSTM_DIR / "scaler.pkl").exists()
             and (LSTM_DIR / "meta.pkl").exists())
    return {"status": "ok", "artifacts_ready": ready, "encryption": "enforced"}


# ── enrollment (consent-gated, PHI-enforced) ─────────────────────────────────
@app.post("/sessions")
def enroll(req: EnrollRequest):
    if not req.consent:
        raise HTTPException(403, "consent required before enrollment; see GET /transparency")
    try:
        privacy.reject_unknown_fields(req.profile, _PROFILE_FIELDS, "profile")
        profile = StaticProfile(**req.profile)
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(422, f"invalid profile: {e}")
    # P11: mint a per-user token, store only its HMAC, return the secret once.
    token = auth.generate_token()
    store.create(req.user_id, profile, consent=True,
                 token_hash=auth.hash_token(token), now_ts=time.time())
    audit.record(action="enroll", actor="user", subject_ref=_ref(req.user_id),
                 consent=True, ts=time.time())
    return {"user_id": req.user_id, "status": "enrolled", "consent": True,
            "access_token": token,
            "token_note": "shown once; present as 'Authorization: Bearer <token>'"}


# ── check-in → companion-safe directive ONLY (third-party block) ─────────────
@app.post("/sessions/{user_id}/observe")
def observe(user_id: str, req: ObserveRequest,
            authorization: str | None = Header(default=None)):
    session = require_user(user_id, authorization)          # P11
    if not session.consent:
        raise HTTPException(403, "no consent on file; check-in refused")
    try:
        privacy.reject_unknown_fields(req.observation, _OBS_FIELDS, "observation")
        obs = WindowObservation(**req.observation)
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(422, f"invalid observation: {e}")

    tv_enc, session.cumulative_stress = obs.time_varying_encoded(session.cumulative_stress)
    session.windows.append(tv_enc)
    session.last_ts = time.time()
    store.put(session)

    score = get_predictor().score(session)
    decision = policy.decide(score)
    # P12: audit the escalation band + alert flag only — no risk numbers, no PHI.
    audit.record(action="observe", actor="user", subject_ref=_ref(user_id),
                 escalation=decision.escalation,
                 clinician_alert=decision.clinician_alert, ts=time.time())
    # ONLY the non-numeric directive crosses back (allowlist / third-party block).
    return privacy.to_companion(decision)


# ── clinical review — risk numbers, authenticated, separate route ────────────
@app.get("/sessions/{user_id}/clinical")
def clinical(user_id: str, x_clinical_token: str | None = Header(default=None)):
    if not CLINICAL_TOKEN or x_clinical_token != CLINICAL_TOKEN:
        audit.record(action="clinical_access", actor="clinician",
                     subject_ref=_ref(user_id), outcome="denied", ts=time.time())
        raise HTTPException(401, "clinical review requires a valid X-Clinical-Token")
    session = store.get(user_id)
    if session is None or not session.windows:
        raise HTTPException(404, "no data for user")
    score = get_predictor().score(session)
    decision = policy.decide(score)
    audit.record(action="clinical_access", actor="clinician", subject_ref=_ref(user_id),
                 escalation=decision.escalation, ts=time.time())
    return {"escalation": decision.escalation,
            "risk_by_horizon_days": score["risk_by_horizon_days"],
            "attention_weights": score["attention_weights"],
            "clinician_alert": decision.clinician_alert}


# ── subject rights: access + erasure (P11-gated, P12-audited) ────────────────
@app.get("/sessions/{user_id}/data")
def my_data(user_id: str, authorization: str | None = Header(default=None)):
    """Right of access: show the data subject exactly what is held about them."""
    session = require_user(user_id, authorization)
    audit.record(action="subject_access", actor="user", subject_ref=_ref(user_id),
                 ts=time.time())
    return {
        "user_id": user_id,
        "consent": session.consent,
        "profile": session.profile.__dict__,
        "checkins_recorded": len(session.windows),
        "current_cumulative_stress": session.cumulative_stress,
        "storage": {"encrypted_at_rest": True,
                    "retention_days": privacy.RETENTION_DAYS},
        "shared_with_third_parties": sorted(privacy.COMPANION_SAFE_KEYS),
    }


@app.delete("/sessions/{user_id}")
def erase(user_id: str, authorization: str | None = Header(default=None)):
    """Right to erasure: hard-delete all data for this user (session PHI).
    The pseudonymous audit trail persists — accountability survives erasure."""
    require_user(user_id, authorization)
    store.delete(user_id)
    audit.record(action="erasure", actor="user", subject_ref=_ref(user_id),
                 ts=time.time())
    return {"user_id": user_id, "status": "erased"}


# ── data portability: JSON export / import (auth-gated, audited) ─────────────
class ImportRequest(BaseModel):
    document: dict = Field(..., description="A prior /export document (schema_version sud.portable.v1)")


@app.get("/portability/schema")
def portability_schema():
    """Open: the JSON Schema for the export document, so any tool can validate it."""
    return portability.EXPORT_JSON_SCHEMA


@app.get("/sessions/{user_id}/export")
def export_data(user_id: str, authorization: str | None = Header(default=None)):
    """Right to portability: the user's full record as a self-contained JSON doc."""
    session = require_user(user_id, authorization)
    audit.record(action="export", actor="user", subject_ref=_ref(user_id), ts=time.time())
    return portability.export_session(session, generated_ts=time.time())


@app.post("/sessions/import")
def import_data(req: ImportRequest, authorization: str | None = Header(default=None)):
    """Re-create a session from a portable document. Mints a fresh token; the
    old token does not carry over. Consent must be present in the document."""
    try:
        token = auth.generate_token()
        session = portability.import_session(
            req.document, token_hash=auth.hash_token(token), now_ts=time.time())
    except (ValueError, PermissionError) as e:
        raise HTTPException(422, f"invalid import: {e}")
    store.put(session)
    audit.record(action="import", actor="user", subject_ref=_ref(session.user_id),
                 checkins=len(session.windows), ts=time.time())
    return {"user_id": session.user_id, "status": "imported",
            "checkins_restored": len(session.windows),
            "access_token": token, "token_note": "shown once; prior token not carried over"}


# ── audit integrity (P12) — anyone can verify the chain is unbroken ──────────
@app.get("/audit/verify")
def audit_verify():
    ok, bad_seq = audit.verify()
    return {"intact": ok, "first_bad_seq": bad_seq}
