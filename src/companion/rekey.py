"""
rekey — rotate the memory encryption key (re-encrypt in place, keep all memory).
=================================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Use when the current key may have been exposed (pasted into a chat, sitting in shell
history, fetched over SSH, etc.). Decrypts memory.enc with the OLD key, re-encrypts it
under a brand-NEW key, and updates nyx.env. ALL memory is preserved — this is NOT a
re-seed. Afterward the old/exposed key no longer opens her memory.

    python -m companion.rekey OLDKEY     # OLDKEY = the current working key
    python -m companion.rekey            # reads the old key from nyx.env

Env: COMPANION_NYX_ENV, COMPANION_MEMORY.
"""

from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

try:
    from cryptography.fernet import Fernet
except ImportError:  # pragma: no cover
    print("cryptography not installed (pip install cryptography)")
    raise SystemExit(1)

NYX_ENV = Path(os.environ.get("COMPANION_NYX_ENV",
                              str(Path.home() / ".config" / "companion" / "nyx.env")))
MEMORY = Path(os.environ.get("COMPANION_MEMORY",
                             str(Path.home() / ".companion" / "memory.enc")))


def _read_key(p: Path) -> tuple[str, list[str]]:
    lines = p.read_text().splitlines() if p.exists() else []
    for ln in lines:
        if ln.startswith("SUD_PHI_KEY="):
            return ln[len("SUD_PHI_KEY="):].strip(), lines
    return "", lines


def main():
    if not MEMORY.exists():
        print(f"no memory file at {MEMORY} — nothing to rotate.")
        return
    old = sys.argv[1].strip() if len(sys.argv) > 1 else _read_key(NYX_ENV)[0]
    if not old:
        print("no old key — pass the current working key as an argument.")
        return

    blob = MEMORY.read_bytes()
    try:
        plain = Fernet(old.encode()).decrypt(blob)     # plaintext stays in RAM only
    except Exception as e:
        print(f"the old key does NOT decrypt {MEMORY.name} ({type(e).__name__}). "
              f"Aborting — memory untouched.")
        return

    new = Fernet.generate_key().decode()
    shutil.copy2(MEMORY, str(MEMORY) + ".prekey")       # backup old encrypted memory
    MEMORY.write_bytes(Fernet(new.encode()).encrypt(plain))
    # sanity: the new key must round-trip the new file before we commit to it
    assert Fernet(new.encode()).decrypt(MEMORY.read_bytes()) == plain, "re-encrypt check failed"

    _, lines = _read_key(NYX_ENV)
    if NYX_ENV.exists():
        shutil.copy2(NYX_ENV, str(NYX_ENV) + ".bak")
    if any(ln.startswith("SUD_PHI_KEY=") for ln in lines):
        out = [("SUD_PHI_KEY=" + new) if ln.startswith("SUD_PHI_KEY=") else ln for ln in lines]
    else:
        out = ["SUD_PHI_KEY=" + new] + lines
    NYX_ENV.parent.mkdir(parents=True, exist_ok=True)
    NYX_ENV.write_text("\n".join(out) + "\n")

    print(f"rotated: {MEMORY.name} re-encrypted under a FRESH key; nyx.env updated.")
    print(f"(old encrypted memory backed up at {MEMORY}.prekey; nyx.env at {NYX_ENV}.bak)")
    print("\n  >>> SAVE THIS NEW KEY DIGITALLY — password manager, COPY-PASTE:\n")
    print(f"      {new}\n")
    print("The old/exposed key no longer opens her memory. Now restart the brain:")
    print("  launchctl kickstart -k gui/$(id -u)/com.nyx.brain")


if __name__ == "__main__":
    main()
