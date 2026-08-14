"""
Per-user session state — encrypted at rest, consent-gated, erasable.
====================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Holds the enrollment profile plus the append-only stream of encoded windows and
the running cumulative-stress value between check-ins.

Privacy posture (see privacy.py):
  * PHI is serialized and encrypted before it touches the store. If no encryption
    key is configured, the store REFUSES to persist (fail-closed) — it never
    writes plaintext PHI.
  * Every session carries an explicit consent flag; the API layer refuses to
    record check-ins without it.
  * `delete()` supports right-to-erasure. `created_ts`/`last_ts` support retention.

The in-memory dict here is a reference; back it with Redis/Postgres in
production. Because the stored value is already an opaque ciphertext blob, the
backing store never sees PHI in the clear.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict

from .contract import StaticProfile
from .privacy import Encryptor


@dataclass
class Session:
    user_id: str
    profile: StaticProfile
    consent: bool = False
    token_hash: str = ""                                 # HMAC of per-user bearer token (P11)
    static_encoded: dict = field(default_factory=dict)
    windows: list[dict] = field(default_factory=list)   # encoded time-varying dicts
    cumulative_stress: float = 0.0                       # running state
    created_ts: float = 0.0                              # caller stamps (Date.now unavailable in scripts)
    last_ts: float = 0.0

    def to_json(self) -> bytes:
        d = asdict(self)
        d["profile"] = asdict(self.profile)
        return json.dumps(d).encode()

    @staticmethod
    def from_json(raw: bytes) -> "Session":
        d = json.loads(raw)
        prof = StaticProfile(**d.pop("profile"))
        return Session(profile=prof, **d)


class SessionStore:
    """Encrypted, consent-aware, erasable reference store.

    Values are ciphertext blobs — the underlying map (or Redis/Postgres) never
    holds plaintext PHI. Construction fails loudly if encryption is unavailable,
    so a misconfigured deployment cannot silently start collecting in the clear.
    """

    def __init__(self, encryptor: Encryptor | None = None, require_encryption: bool = True):
        self._enc = encryptor or Encryptor()
        if require_encryption and not self._enc.available:
            raise PermissionError(
                f"SessionStore refuses to start without encryption: {self._enc.reason}. "
                "This system must not persist PHI in plaintext."
            )
        self._blobs: dict[str, bytes] = {}

    def create(self, user_id: str, profile: StaticProfile, consent: bool,
               token_hash: str = "", now_ts: float = 0.0) -> Session:
        s = Session(user_id=user_id, profile=profile, consent=consent,
                    token_hash=token_hash, static_encoded=profile.encoded(),
                    created_ts=now_ts, last_ts=now_ts)
        self.put(s)
        return s

    def get(self, user_id: str) -> Session | None:
        blob = self._blobs.get(user_id)
        if blob is None:
            return None
        return Session.from_json(self._enc.decrypt(blob))

    def put(self, session: Session) -> None:
        self._blobs[session.user_id] = self._enc.encrypt(session.to_json())

    def delete(self, user_id: str) -> bool:
        """Right-to-erasure: hard-delete the ciphertext. Returns True if existed."""
        return self._blobs.pop(user_id, None) is not None

    def purge_expired(self, now_ts: float, retention_seconds: float) -> int:
        """Retention enforcement: drop sessions older than the retention window."""
        expired = [uid for uid, blob in list(self._blobs.items())
                   if now_ts - Session.from_json(self._enc.decrypt(blob)).last_ts > retention_seconds]
        for uid in expired:
            self._blobs.pop(uid, None)
        return len(expired)
