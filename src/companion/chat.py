"""
Local companion REPL — talk to it from your terminal.
=====================================================
    export SUD_PHI_KEY=$(python -c "from serving.privacy import Encryptor; print(Encryptor.generate_key())")
    # Local (closed system — nothing leaves the machine):
    ollama serve &                 # or LM Studio / llama.cpp server
    python -m companion.chat --persona companion/persona.example.json

    # Or Claude backend (more capable; content leaves the machine):
    python -m companion.chat --backend claude --persona companion/persona.example.json

Keep SUD_PHI_KEY somewhere safe — it decrypts your memory. Lose it and the
memory file is unreadable (that's the point).
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

try:
    from companion.orchestrator import (Orchestrator, MemoryStore, Persona,
                                        LocalBackend, ClaudeBackend)
except ImportError:  # pragma: no cover
    from .orchestrator import (Orchestrator, MemoryStore, Persona,
                              LocalBackend, ClaudeBackend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "claude"], default="local")
    ap.add_argument("--persona", default=None, help="path to persona JSON")
    ap.add_argument("--memory", default=str(Path.home() / ".companion" / "memory.enc"))
    ap.add_argument("--seed", default=None, help="JSON list of {text,importance?} to seed memory")
    args = ap.parse_args()

    backend = ClaudeBackend() if args.backend == "claude" else LocalBackend()
    persona = Persona.from_json(args.persona) if args.persona else Persona()
    memory = MemoryStore(args.memory)

    if args.seed:
        import json, time
        memory.seed_from_export(json.loads(Path(args.seed).read_text()), now_ts=time.time())
        print(f"seeded memory from {args.seed}")

    orch = Orchestrator(backend, memory, persona)
    where = "local (closed)" if args.backend == "local" else "Claude API"
    print(f"[{persona.name} · {where}] — type your message. Ctrl-C to leave.\n")
    try:
        while True:
            msg = input("you › ").strip()
            if not msg:
                continue
            res = orch.turn(msg)
            print(f"\n{persona.name} › {res.reply}\n")
    except (KeyboardInterrupt, EOFError):
        print("\ntake care.")


if __name__ == "__main__":
    main()
