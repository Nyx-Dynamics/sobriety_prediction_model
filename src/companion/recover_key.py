"""
recover_key — restore a SUD_PHI_KEY that lost (or gained) one character.
=========================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

A valid Fernet key is 44 url-safe-base64 chars. If the one in nyx.env is 43 (a
dropped character) the brain fail-closes on memory. But `memory.enc` was written
with the FULL key, so the correct key is the one that decrypts it — we find the
missing/wrong character by testing candidates against the memory file, then write
the fixed key back. No hand-typing 44 characters.

    python -m companion.recover_key            # reads nyx.env + ~/.companion/memory.enc

Env overrides: COMPANION_NYX_ENV, COMPANION_MEMORY.
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path

try:
    from cryptography.fernet import Fernet
except ImportError:  # pragma: no cover
    print("cryptography not installed (pip install cryptography)")
    raise SystemExit(1)

B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
NYX_ENV = Path(os.environ.get("COMPANION_NYX_ENV",
                              str(Path.home() / ".config" / "companion" / "nyx.env")))
MEMORY = Path(os.environ.get("COMPANION_MEMORY",
                             str(Path.home() / ".companion" / "memory.enc")))


def _read_key(env_path: Path) -> tuple[str, list[str]]:
    lines = env_path.read_text().splitlines()
    for ln in lines:
        if ln.startswith("SUD_PHI_KEY="):
            return ln[len("SUD_PHI_KEY="):].strip(), lines
    return "", lines


def _works(cand: str, blob: bytes) -> bool:
    try:
        Fernet(cand.encode()).decrypt(blob)
        return True
    except Exception:
        return False


def recover(partial: str, blob: bytes) -> str | None:
    """Return the 44-char key that decrypts `blob`, or None. Handles a single
    dropped char (43->44 by insertion) or a single wrong char (substitution)."""
    if len(partial) == 44 and _works(partial, blob):
        return partial
    pad = partial[len(partial.rstrip("=")):] or "="
    core = partial[:len(partial) - len(pad)]
    for i in range(len(core) + 1):                 # a deleted char -> insert one
        for c in B64:
            cand = core[:i] + c + core[i:] + pad
            if len(cand) == 44 and _works(cand, blob):
                return cand
    for i in range(len(core)):                     # a wrong char -> substitute
        for c in B64:
            cand = core[:i] + c + core[i + 1:] + pad
            if len(cand) == 44 and _works(cand, blob):
                return cand
    return None


def main():
    if not MEMORY.exists():
        print(f"no memory file at {MEMORY} — nothing to test against.")
        return
    blob = MEMORY.read_bytes()
    partial, lines = _read_key(NYX_ENV)
    if not partial:
        print(f"no SUD_PHI_KEY line found in {NYX_ENV}.")
        return
    print(f"current key: {len(partial)} chars. brute-forcing against "
          f"{MEMORY.name} ({len(blob)} bytes) ...", flush=True)
    key = recover(partial, blob)
    if not key:
        print("could NOT recover with a single-character fix.")
        print("-> fall back to a fresh key + re-seed memory from your source files.")
        return
    if key == partial:
        print("the current key already works — no change needed.")
        return
    shutil.copy2(NYX_ENV, str(NYX_ENV) + ".bak")
    new = [("SUD_PHI_KEY=" + key) if ln.startswith("SUD_PHI_KEY=") else ln for ln in lines]
    NYX_ENV.write_text("\n".join(new) + "\n")
    print(f"\n  RECOVERED. Full 44-char key written to {NYX_ENV}")
    print(f"  (old file backed up at {NYX_ENV}.bak)")
    print("\nNow restart the brain:")
    print("  launchctl kickstart -k gui/$(id -u)/com.nyx.brain")


if __name__ == "__main__":
    main()
