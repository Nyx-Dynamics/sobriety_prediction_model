"""
Authentication & per-user isolation (P11).
===========================================
Sobriety Prediction Model | Nyx Dynamics LLC

Per-user bearer tokens with server-side pepper. Enrollment mints a token once;
only its HMAC is stored (never the token itself, mirroring password hygiene).
Every user-scoped route requires the caller to present the matching token, which
enforces isolation: user A's token cannot read or erase user B's data.

Fail-closed: no `SUD_AUTH_PEPPER` ⇒ token hashing refuses to operate, so a
misconfigured deployment cannot issue forgeable/unpeppered credentials.

`pseudonymize()` also derives a stable, non-reversible subject reference from a
user_id so the audit log (P12) can be correlated without storing the identifier.
"""

from __future__ import annotations
import hmac
import hashlib
import os
import secrets

ENV_PEPPER = "SUD_AUTH_PEPPER"


def _pepper() -> bytes:
    p = os.environ.get(ENV_PEPPER, "")
    if not p:
        raise PermissionError(
            f"no {ENV_PEPPER} configured — refusing to hash credentials "
            "(generate with `python -c \"import secrets;print(secrets.token_urlsafe(32))\"`)."
        )
    return p.encode()


def generate_token() -> str:
    """A fresh per-user bearer token, shown to the caller exactly once."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hmac.new(_pepper(), token.encode(), hashlib.sha256).hexdigest()


def verify_token(token: str, stored_hash: str) -> bool:
    if not token or not stored_hash:
        return False
    return hmac.compare_digest(hash_token(token), stored_hash)


def pseudonymize(user_id: str) -> str:
    """Stable, non-reversible subject reference for audit correlation (no PHI)."""
    return hmac.new(_pepper(), b"subject:" + user_id.encode(), hashlib.sha256).hexdigest()[:16]


def bearer(authorization: str | None) -> str | None:
    """Extract a token from an `Authorization: Bearer <token>` header."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None
