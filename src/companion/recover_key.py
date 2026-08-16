"""
recover_key — restore a SUD_PHI_KEY that lost/changed one or two characters.
============================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

A valid Fernet key is 44 url-safe-base64 chars. A hand-transcribed key that's a
character short (and maybe one character wrong) makes the brain fail-closed on
memory. But `memory.enc` was written with the FULL key, so the correct key is the
one that decrypts it. We brute-force the missing/wrong character(s) and write the
fixed key back — no hand-typing.

Speed trick: naively testing a key means HMAC-ing the whole 2.3 MB token (slow ×
millions). Instead we AES-decrypt only the FIRST block and check it starts like the
known plaintext (`[{` — a JSON array of memories). That's ~microseconds, so even a
two-character search finishes in under a minute. Survivors are confirmed with a
full Fernet decrypt.

    python -m companion.recover_key            # nyx.env + ~/.companion/memory.enc

Env overrides: COMPANION_NYX_ENV, COMPANION_MEMORY.
"""

from __future__ import annotations
import base64
import os
import shutil
import sys
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:  # pragma: no cover
    print("cryptography not installed (pip install cryptography)")
    raise SystemExit(1)

# both alphabets: url-safe (-_) AND standard (+/) — keys may use either
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_+/"
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


def _prep(blob: bytes):
    """From a Fernet token: the IV and the first ciphertext block (for the fast test)."""
    raw = base64.urlsafe_b64decode(blob)
    return raw[9:25], raw[25:41]           # iv, first ciphertext block


def _fast_ok(cand: str, iv: bytes, c0: bytes) -> bool:
    """Cheap plausibility: AES-decrypt only block 0 and check it looks like the
    known plaintext start ('[{' then printable ASCII). ~microseconds."""
    try:
        kb = base64.urlsafe_b64decode(cand)
    except Exception:
        return False
    if len(kb) != 32:
        return False
    dec = Cipher(algorithms.AES(kb[16:]), modes.ECB()).decryptor()
    blk = dec.update(c0) + dec.finalize()
    p0 = bytes(x ^ y for x, y in zip(blk, iv))
    return p0[:2] == b"[{" and all(32 <= b < 127 for b in p0)


def recover(partial: str, blob: bytes) -> str | None:
    iv, c0 = _prep(blob)

    def confirm(cand: str) -> bool:
        try:
            Fernet(cand.encode()).decrypt(blob)
            return True
        except Exception:
            return False

    if len(partial) == 44 and confirm(partial):
        return partial

    pad = partial[len(partial.rstrip("=")):] or "="
    core = partial[:len(partial) - len(pad)]

    # 1 char dropped -> insert one (these are the 44-char single-insertion candidates)
    inserts = [core[:i] + c + core[i:] + pad for i in range(len(core) + 1) for c in B64]
    inserts = [k for k in inserts if len(k) == 44]
    for k in inserts:
        if _fast_ok(k, iv, c0) and confirm(k):
            return k

    # 1 char wrong -> substitute one
    for i in range(len(core)):
        for c in B64:
            k = core[:i] + c + core[i + 1:] + pad
            if len(k) == 44 and _fast_ok(k, iv, c0) and confirm(k):
                return k

    # 1 dropped AND 1 wrong -> substitute within each insertion candidate (the deep pass)
    total = len(inserts)
    print(f"  single-char fixes exhausted; deep 2-char search over "
          f"{total} bases ...", flush=True)
    for n, base in enumerate(inserts, 1):
        bcore = base[:len(base) - len(pad)]
        for j in range(len(bcore)):
            row = bcore[:j]
            tail = bcore[j + 1:]
            for c in B64:
                k = row + c + tail + pad
                if _fast_ok(k, iv, c0) and confirm(k):
                    return k
        if n % 200 == 0:
            print(f"    ... {n}/{total}", flush=True)
    return None


def main():
    if not MEMORY.exists():
        print(f"no memory file at {MEMORY} — nothing to test against.")
        return
    blob = MEMORY.read_bytes()
    _, lines = _read_key(NYX_ENV)
    # a candidate base key can be passed as an argument (e.g. a second key you have
    # written down) — we search around IT instead of the one in nyx.env
    if len(sys.argv) > 1:
        partial = sys.argv[1].strip()
        print(f"searching around the key you passed ({len(partial)} chars).", flush=True)
    else:
        partial, lines = _read_key(NYX_ENV)
    if not partial:
        print(f"no SUD_PHI_KEY line found in {NYX_ENV} and none passed as an argument.")
        return
    print(f"current key: {len(partial)} chars. searching against "
          f"{MEMORY.name} ({len(blob):,} bytes) ...", flush=True)
    key = recover(partial, blob)
    if not key:
        print("\ncould NOT recover with a one- or two-character fix.")
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
    print("  launchctl kickstart -k gui/$(id -u)/com.nyx.brain   (or bootstrap if needed)")


if __name__ == "__main__":
    main()
