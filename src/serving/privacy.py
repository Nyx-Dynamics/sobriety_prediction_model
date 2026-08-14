"""
Privacy core — the anti-surveillance boundary.
==============================================
Sobriety Prediction Model | Nyx Dynamics LLC

This module encodes the project's non-negotiable stance: **this MUST NOT be a
surveillance tool.** Four properties are treated as paramount and are enforced
here, fail-closed, rather than left to convention:

  1. ENCRYPTION       — PHI is never persisted in plaintext. No key ⇒ refuse to store.
  2. PHI ENFORCEMENT  — only the declared structured features may enter the model
                        layer. Free text / transcripts / unknown fields are rejected.
  3. THIRD-PARTY BLOCK— nothing but a non-numeric behavioral directive may cross
                        back toward the companion LLM or any external service.
                        Risk numbers and feature vectors physically cannot leave.
  4. TRANSPARENCY     — a machine- and human-readable disclosure of exactly what is
                        collected, why, how long, who it is shared with (no one),
                        and the user's rights — available without authentication.

Design rule: allowlists, not denylists. Egress is built from a fixed safe set;
you cannot "forget to redact" a field, because unlisted fields are never emitted.
"""

from __future__ import annotations
import os

# ── Anti-surveillance invariants (asserted by callers) ─────────────────────
# The ONLY keys allowed to cross back toward the companion / any 3rd party.
COMPANION_SAFE_KEYS = frozenset({
    "escalation",            # band name only: steady|monitor|outreach|urgent
    "companion_directive",   # non-numeric behavioral guidance
    "rationale",             # names a window, never a probability
})

# Fields that are PHI and therefore encrypt-at-rest + never egress.
PHI_CLASSES = {
    "identifiers": ["user_id"],
    "clinical_static": ["sud_type", "sud_severity", "prior_tx_episodes",
                        "mdd", "ptsd", "gad", "adhd", "bipolar_i", "bipolar_ii",
                        "psychosis", "bpd", "med_hcv", "med_hiv",
                        "med_chronic_pain", "med_tbi_history", "med_liver_disease",
                        "med_cardiovascular", "med_ms_autoimmune"],
    "demographic": ["age_at_enrollment", "age_first_use", "sex", "race_ethnicity",
                    "housing_stability", "insurance_status"],
    "behavioral_timeseries": ["phq9_score", "gad7_score", "uds_positive",
                              "sober_this_window", "adverse_event", "med_adherence",
                              "cumulative_stress", "active_depression"],
    "derived_risk": ["hazard_curve", "survival_curve", "risk_by_horizon_days",
                     "attention_weights"],
}


# ══════════════════════════════════════════════════════════════════════════
# 1. Encryption (fail-closed)
# ══════════════════════════════════════════════════════════════════════════
class Encryptor:
    """AES-128-CBC + HMAC via Fernet. Key from env `SUD_PHI_KEY` (urlsafe b64,
    32 bytes). If the library or key is absent, `available` is False and the
    store MUST refuse to persist PHI — we do not silently fall back to plaintext.
    """

    ENV_KEY = "SUD_PHI_KEY"

    def __init__(self, key: bytes | None = None):
        self._fernet = None
        self.reason = ""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            self.reason = "cryptography not installed (pip install -r serving/requirements.txt)"
            return
        raw = key or (os.environ.get(self.ENV_KEY, "").encode() or None)
        if not raw:
            self.reason = f"no key in ${self.ENV_KEY} (generate with Encryptor.generate_key())"
            return
        try:
            self._fernet = Fernet(raw)
        except Exception as e:  # malformed key
            self.reason = f"invalid {self.ENV_KEY}: {e}"

    @property
    def available(self) -> bool:
        return self._fernet is not None

    @staticmethod
    def generate_key() -> str:
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()

    def encrypt(self, plaintext: bytes) -> bytes:
        if not self.available:
            raise PermissionError(f"refusing to encrypt: {self.reason}")
        return self._fernet.encrypt(plaintext)

    def decrypt(self, token: bytes) -> bytes:
        if not self.available:
            raise PermissionError(f"refusing to decrypt: {self.reason}")
        return self._fernet.decrypt(token)


# ══════════════════════════════════════════════════════════════════════════
# 2. PHI enforcement at the ingress boundary
# ══════════════════════════════════════════════════════════════════════════
def reject_unknown_fields(payload: dict, allowed: set[str], where: str) -> None:
    """Structured-only gate. Any field not in the declared contract (e.g. a
    free-text `notes`, a raw `transcript`, a device id) is rejected — the model
    layer never ingests unstructured content it wasn't designed to hold."""
    extra = set(payload) - set(allowed)
    if extra:
        raise ValueError(
            f"{where}: unknown field(s) {sorted(extra)} rejected — only declared "
            f"contract features are accepted (no free text / transcripts / metadata)."
        )


# ══════════════════════════════════════════════════════════════════════════
# 3. Third-party egress block (allowlist)
# ══════════════════════════════════════════════════════════════════════════
def to_companion(decision) -> dict:
    """Build the ONLY payload permitted to reach the companion LLM / client.
    Constructed from the safe allowlist — risk numbers, hazards, attention, and
    the feature vector are structurally excluded because they are never added."""
    payload = {
        "escalation": decision.escalation,
        "companion_directive": decision.companion_directive,
        "rationale": decision.rationale,
    }
    assert_egress_clean(payload)
    return payload


def assert_egress_clean(payload: dict) -> None:
    """Defense in depth: prove nothing outside the safe set is leaving, and that
    no value is numeric (a probability could otherwise slip through a string)."""
    leaked = set(payload) - COMPANION_SAFE_KEYS
    if leaked:
        raise AssertionError(f"egress guard: disallowed keys would leave: {sorted(leaked)}")
    for k, v in payload.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            raise AssertionError(f"egress guard: numeric value for '{k}' — risk numbers must not egress")


# ══════════════════════════════════════════════════════════════════════════
# 4. Transparency (overreaching, always-on, unauthenticated)
# ══════════════════════════════════════════════════════════════════════════
RETENTION_DAYS = int(os.environ.get("SUD_RETENTION_DAYS", "365"))

DISCLOSURE = {
    "purpose": "Support recovery via check-ins and route to help when risk rises. "
               "This is a supportive tool, not a diagnostic or clinical decision system.",
    "not_a_surveillance_tool": True,
    "what_we_collect": {
        "structured_only": True,
        "no_free_text_or_transcripts": True,
        "static_profile_fields": PHI_CLASSES["clinical_static"] + PHI_CLASSES["demographic"],
        "per_checkin_fields": PHI_CLASSES["behavioral_timeseries"],
    },
    "what_we_derive": PHI_CLASSES["derived_risk"],
    "what_leaves_the_server": {
        "to_companion_or_third_parties": sorted(COMPANION_SAFE_KEYS),
        "risk_numbers_shared": False,
        "feature_vectors_shared": False,
        "sold_or_advertised": False,
    },
    "storage": {"encrypted_at_rest": True, "retention_days": RETENTION_DAYS},
    "your_rights": {
        "access": "GET /sessions/{user_id}/data — see everything held about you",
        "erasure": "DELETE /sessions/{user_id} — hard-delete on request",
        "consent": "required before any check-in is recorded; withdrawable",
    },
}


def disclosure() -> dict:
    return DISCLOSURE
