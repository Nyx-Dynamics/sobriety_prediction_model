"""
LinkCipher — optional encryption for the node<->brain audio hop.
================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

A room node ships raw voice to the brain over the LAN. Wired Ethernet keeps that
physically in the house, but raw voice on the wire still deserves in-transit
encryption. This wraps the audio (and the reply) with Fernet under a shared
COMPANION_LINK_KEY.

Fail-safe, not fail-closed BY DESIGN so a mixed fleet keeps working:
  * Key set + cryptography present  -> encrypt (Mac nodes: office Studio, kitchen Mini).
  * No key, or no cryptography lib   -> passthrough plaintext (e.g. a bare Pi that
    never got `pip install cryptography`). The hop is unencrypted but functional.

Generate a key once and put the SAME value in every node's + the brain's env:
    python -c "from companion.link import LinkCipher; print(LinkCipher.generate_key())"
    export COMPANION_LINK_KEY=...
"""

from __future__ import annotations
import os

try:
    from cryptography.fernet import Fernet
except Exception:   # cryptography not installed on this node
    Fernet = None


class LinkCipher:
    ENV = "COMPANION_LINK_KEY"
    ENV_PASS = "COMPANION_LINK_PASSPHRASE"

    def __init__(self, key: str | bytes | None = None):
        k = key if key is not None else os.environ.get(self.ENV)
        # No explicit 44-char key? Derive one from a shared PASSPHRASE. For machines
        # that can't copy-paste across a KVM/monitor switch, you just type the SAME
        # human phrase on each — deterministic, so identical phrase -> identical key
        # everywhere. Far less error-prone than transcribing base64 by hand.
        if not k:
            phrase = os.environ.get(self.ENV_PASS)
            if phrase:
                k = self.key_from_passphrase(phrase)
        if isinstance(k, str):
            k = k.encode()
        self._f = None
        if not Fernet:
            self.reason = "cryptography missing"
        elif not k:
            self.reason = "no key set"
        else:
            try:
                self._f = Fernet(k)
                self.reason = ""
            except Exception as e:
                # A mistyped/truncated key must NEVER crash the brain (its module-level
                # LinkCipher() would take the whole server down). Fall back to plaintext,
                # loudly — the state shows in the node's startup line, not in a traceback.
                self.reason = f"invalid key ({type(e).__name__}) -> plaintext"

    @property
    def on(self) -> bool:
        return self._f is not None

    def wrap(self, data: bytes) -> bytes:
        return self._f.encrypt(data) if self._f else data

    def unwrap(self, data: bytes) -> bytes:
        return self._f.decrypt(data) if self._f else data

    @staticmethod
    def key_from_passphrase(passphrase: str) -> bytes:
        """Derive a valid Fernet key from a human passphrase (PBKDF2, fixed app salt).
        Same passphrase -> same key on every machine, so a mistyped 44-char key stops
        being the failure mode. Stdlib only. Fixed salt is fine for this threat model
        (voice on your own wired LAN); use a non-trivial phrase, no spaces needed."""
        import hashlib
        import base64
        dk = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"),
                                 b"nyx-companion-link-v1", 200_000, dklen=32)
        return base64.urlsafe_b64encode(dk)

    @staticmethod
    def generate_key() -> str:
        if not Fernet:
            raise RuntimeError("cryptography not installed — cannot generate a link key")
        return Fernet.generate_key().decode()
