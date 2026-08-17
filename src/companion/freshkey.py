"""
freshkey — generate a NEW SUD_PHI_KEY and write it into nyx.env, safely.
========================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

For when the old memory key is unrecoverable and memory will be re-seeded. The key
is generated and written into nyx.env IN PLACE (backing up the old file) — so there
is NO hand-typing and NO cross-machine transcription, which is exactly what corrupted
the last key. It also prints the key so you can save it DIGITALLY (password manager,
copy-paste) — never hand-write a 44-char key again.

    python -m companion.freshkey

Env override: COMPANION_NYX_ENV.
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

NYX_ENV = Path(os.environ.get("COMPANION_NYX_ENV",
                              str(Path.home() / ".config" / "companion" / "nyx.env")))


def write_fresh_key(env_path: Path) -> str:
    """Generate a fresh Fernet key and write it as SUD_PHI_KEY in env_path (backing
    up any existing file). Returns the new 44-char key."""
    key = Fernet.generate_key().decode()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    if env_path.exists():
        shutil.copy2(env_path, str(env_path) + ".bak")
    out, replaced = [], False
    for ln in lines:
        if ln.startswith("SUD_PHI_KEY="):
            out.append("SUD_PHI_KEY=" + key)
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.insert(0, "SUD_PHI_KEY=" + key)
    env_path.write_text("\n".join(out) + "\n")
    return key


def main():
    existed = NYX_ENV.exists()
    key = write_fresh_key(NYX_ENV)
    print(f"fresh SUD_PHI_KEY written to {NYX_ENV}")
    if existed:
        print(f"(previous file backed up at {NYX_ENV}.bak)")
    print("\n  >>> SAVE THIS KEY DIGITALLY — password manager, COPY-PASTE.")
    print("      Do NOT hand-write it; that is what broke the last one.\n")
    print(f"      {key}\n")
    print("Then move the old (undecryptable) memory aside and re-seed:")
    print("  mv ~/.companion/memory.enc ~/.companion/memory.enc.locked")
    print("  set -a; source ~/.config/companion/nyx.env; set +a")
    print("  ../.venv/bin/python -m companion.chat "
          "--persona companion/persona.example.json --seed ~/nyx_deep.json")
    print("  launchctl kickstart -k gui/$(id -u)/com.nyx.brain")


if __name__ == "__main__":
    main()
