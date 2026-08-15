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

    def __init__(self, key: str | bytes | None = None):
        k = key if key is not None else os.environ.get(self.ENV)
        if isinstance(k, str):
            k = k.encode()
        self._f = Fernet(k) if (k and Fernet) else None
        self.reason = "" if self._f else ("cryptography missing" if not Fernet else "no key set")

    @property
    def on(self) -> bool:
        return self._f is not None

    def wrap(self, data: bytes) -> bytes:
        return self._f.encrypt(data) if self._f else data

    def unwrap(self, data: bytes) -> bytes:
        return self._f.decrypt(data) if self._f else data

    @staticmethod
    def generate_key() -> str:
        if not Fernet:
            raise RuntimeError("cryptography not installed — cannot generate a link key")
        return Fernet.generate_key().decode()
