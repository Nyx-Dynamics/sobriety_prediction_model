"""
Brain server — the consolidated companion (runs on the Studio).
===============================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Her whole mind and memory live here: whisper (via the local whisper-server) +
orchestrator + encrypted memory + crisis safety + Ollama. Satellites POST a
window of audio; the brain transcribes, thinks, and returns the reply. The
satellite renders it to speech. This is the sovereign-central-brain of the
household design — one place all rooms point at.

Deps here are tiny: it talks to whisper-server (:8080) and Ollama (:11434) over
localhost HTTP, and the orchestrator is pure Python — only `cryptography` (for
the encrypted memory) is needed. No torch/numpy on the Studio.

Run (on the Studio, from src/):
    python -m companion.brain_server --persona companion/persona.example.json
Endpoints:
    GET  /health          liveness
    POST /turn            body = WAV bytes, header X-Node: <room> -> {reply, heard, crisis}
"""

from __future__ import annotations
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

try:
    from companion.orchestrator import Orchestrator, MemoryStore, Persona, LocalBackend
    from companion.voice import RemoteWhisperASR
except ImportError:  # pragma: no cover
    from .orchestrator import Orchestrator, MemoryStore, Persona, LocalBackend
    from .voice import RemoteWhisperASR


def handle_turn(asr, orch, wav: bytes, node: str = "default") -> dict:
    """Core turn: audio -> transcript -> orchestrated reply. Pure + testable.
    Safety runs inside orch.turn(), so it's centralized here on the brain."""
    text = asr.transcribe(wav)
    if not text:
        return {"reply": "", "heard": "", "crisis": False}
    res = orch.turn(text)
    return {"reply": res.reply, "heard": text, "crisis": res.crisis}


# module-level so the single-threaded server shares one orchestrator (turns
# serialize naturally — exactly what one shared memory/history wants).
ORCH: Orchestrator | None = None
ASR = None


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "who": ORCH.persona.name})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/turn":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        wav = self.rfile.read(n)
        node = self.headers.get("X-Node", "default")
        out = handle_turn(ASR, ORCH, wav, node)
        if out["heard"]:
            print(f"[{node}] you: {out['heard']}\n[{node}] {ORCH.persona.name}: {out['reply']}")
        self._json(out)

    def log_message(self, *a):  # silence default request logging
        pass


def main():
    global ORCH, ASR
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default=None)
    ap.add_argument("--memory", default=str(Path.home() / ".companion" / "memory.enc"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--whisper-url",
                    default=os.environ.get("WHISPER_REMOTE_URL", "http://localhost:8080"))
    args = ap.parse_args()

    persona = Persona.from_json(args.persona) if args.persona else Persona()
    ASR = RemoteWhisperASR(args.whisper_url)
    backend = LocalBackend()   # -> COMPANION_LOCAL_URL (localhost Ollama on the Studio)
    ORCH = Orchestrator(backend, MemoryStore(args.memory), persona)
    print(f"[brain] {persona.name} on {args.host}:{args.port}  "
          f"whisper={args.whisper_url}  model={backend.model}")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
