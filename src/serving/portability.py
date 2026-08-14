"""
Data portability — JSON export/import for the user (P: transportability).
=========================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Gives the data subject their data in a self-contained, versioned JSON document
they can download, back up, or carry to another provider (the GDPR Art. 20
right to portability), and re-import to reconstruct their session. Interoperable:
the document is plain JSON validated against `EXPORT_JSON_SCHEMA`, so any tool
can read it.

Privacy invariants preserved on the round-trip:
  * Export is the FULL first-party record (profile + check-in stream) for the
    subject — it is not the third-party egress path, so it legitimately contains
    PHI. Guard it with auth (P11) and audit it (P12); transmit over TLS.
  * Import stays structured-only: unknown/free-text fields are rejected (P1),
    consent must be present in the document, and encoding is re-validated.
"""

from __future__ import annotations
from dataclasses import fields

from .contract import StaticProfile, WindowObservation, FEATURE_COLS
from .state import Session
from . import privacy

SCHEMA_VERSION = "sud.portable.v1"

_PROFILE_FIELDS = {f.name for f in fields(StaticProfile)}


# ── export ─────────────────────────────────────────────────────────────────
def export_session(session: Session, *, generated_ts: float = 0.0) -> dict:
    """Serialize a session to a portable JSON-able document. Includes everything
    held about the user plus a manifest so the export is self-describing."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_ts": generated_ts,
        "manifest": {
            "feature_contract": FEATURE_COLS,
            "checkin_fields": [f.name for f in fields(WindowObservation)],
            "note": "Your data. Portable and re-importable. See /transparency.",
        },
        "subject": {
            "user_id": session.user_id,
            "consent": session.consent,
            "created_ts": session.created_ts,
            "last_ts": session.last_ts,
        },
        "profile": {k: getattr(session.profile, k) for k in _PROFILE_FIELDS},
        # encoded time-varying windows (already the numeric contract form)
        "checkins": session.windows,
        "derived_state": {"cumulative_stress": session.cumulative_stress},
    }


# ── import ─────────────────────────────────────────────────────────────────
def import_session(doc: dict, *, token_hash: str = "", now_ts: float = 0.0) -> Session:
    """Reconstruct a Session from a portable document, re-validating the contract.
    Raises ValueError on version mismatch, missing consent, or contract drift."""
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported document; expected schema_version={SCHEMA_VERSION!r}")

    subject = doc.get("subject") or {}
    if not subject.get("consent"):
        raise ValueError("refusing import without recorded consent")
    user_id = subject.get("user_id")
    if not user_id:
        raise ValueError("document missing subject.user_id")

    # Structured-only: reject any field not in the profile contract (P1).
    profile_doc = doc.get("profile") or {}
    privacy.reject_unknown_fields(profile_doc, _PROFILE_FIELDS, "imported profile")
    try:
        profile = StaticProfile(**profile_doc)
    except (TypeError, KeyError) as e:
        raise ValueError(f"invalid imported profile: {e}")

    # Validate each check-in window carries exactly the encoded contract keys.
    checkins = doc.get("checkins") or []
    expected = set(FEATURE_COLS[:13])  # the 13 time-varying encoded keys
    for i, w in enumerate(checkins):
        extra = set(w) - expected
        missing = expected - set(w)
        if extra or missing:
            raise ValueError(f"checkin[{i}] does not match the encoded contract "
                             f"(extra={sorted(extra)}, missing={sorted(missing)})")

    session = Session(
        user_id=user_id,
        profile=profile,
        consent=True,
        token_hash=token_hash,
        static_encoded=profile.encoded(),
        windows=list(checkins),
        cumulative_stress=float((doc.get("derived_state") or {}).get("cumulative_stress", 0.0)),
        created_ts=float(subject.get("created_ts", now_ts)),
        last_ts=now_ts,
    )
    return session


# ── interoperability: a JSON Schema other tools can validate against ────────
EXPORT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Sobriety Prediction — portable user export",
    "type": "object",
    "required": ["schema_version", "subject", "profile", "checkins"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "generated_ts": {"type": "number"},
        "manifest": {"type": "object"},
        "subject": {
            "type": "object",
            "required": ["user_id", "consent"],
            "properties": {
                "user_id": {"type": "string"},
                "consent": {"type": "boolean"},
                "created_ts": {"type": "number"},
                "last_ts": {"type": "number"},
            },
        },
        "profile": {"type": "object"},
        "checkins": {"type": "array", "items": {"type": "object"}},
        "derived_state": {"type": "object"},
    },
}
