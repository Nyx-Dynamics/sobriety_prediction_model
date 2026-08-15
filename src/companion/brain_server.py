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
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

try:
    from companion.orchestrator import Orchestrator, MemoryStore, Persona, LocalBackend
    from companion.voice import RemoteWhisperASR
    from companion.presence import presence_sentence
    from companion.link import LinkCipher
except ImportError:  # pragma: no cover
    from .orchestrator import Orchestrator, MemoryStore, Persona, LocalBackend
    from .voice import RemoteWhisperASR
    from .presence import presence_sentence
    from .link import LinkCipher

# Shared-key encryption for the node<->brain hop (opt-in via COMPANION_LINK_KEY).
# Per-request: decrypt if the node marked it encrypted; a plaintext Pi still works.
LINK = LinkCipher()

# Optional presence daemon (companion.presence). Best-effort; None if not running.
PRESENCE_URL = os.environ.get("PRESENCE_URL", "http://localhost:9100/presence")


def _presence_note() -> str | None:
    try:
        with urllib.request.urlopen(PRESENCE_URL, timeout=1) as r:
            return presence_sentence(json.loads(r.read())) or None
    except Exception:
        return None   # daemon not up / unreachable -> no ambient context, fine


# Optional reference index (companion.reference). She looks in your DOCUMENTS only
# when you ask her to find something; excerpts go into the per-turn prompt, never
# into her memory. Best-effort; None if the daemon isn't running.
REFERENCE_URL = os.environ.get("REFERENCE_URL", "http://localhost:9200/search")
_LOOKUP_RE = re.compile(
    r"\b(look up|look for|find|search|pull up|dig up|what did|what does|what's in|"
    r"whats in|remind me what|in my (files?|docs?|documents?|records?|notes?|"
    r"filing|lease|results?|report|paperwork|emails?))\b", re.IGNORECASE)


def _reference_context(query: str) -> str | None:
    if not _LOOKUP_RE.search(query):
        return None
    try:
        from urllib.parse import quote
        with urllib.request.urlopen(f"{REFERENCE_URL}?q={quote(query)}", timeout=4) as r:
            hits = json.loads(r.read()).get("hits", [])
    except Exception:
        return None
    if not hits:
        return None
    lines = [f"- from \"{h['source']}\": {h['text'][:400]}" for h in hits[:4]]
    return ("The user is asking you to recall something from their own documents. "
            "Use these excerpts to answer and name the source file; if they don't "
            "actually answer it, say you couldn't find it in their files:\n"
            + "\n".join(lines))


def _context_provider(query: str) -> str | None:
    """Per-turn ambient context: presence (always) + document excerpts (on lookup).
    Goes into the system prompt only — never the message, history, or memory."""
    parts = []
    p = _presence_note()
    if p:
        parts.append(p)
    r = _reference_context(query)
    if r:
        parts.append(r)
    return "\n\n".join(parts) or None


# ── wake word: she stays dormant until addressed by name ─────────────────────
# Fuzzy set covers how whisper spells "Nyx" (Nicks/Nix/Knicks...). Override with
# WAKE_WORDS="nyx,nix,..."; WAKE_WINDOW_S is the follow-up window (0 disables the
# whole feature — always-listening, the old behavior).
WAKE_WORDS = set(w.strip() for w in
                 (os.environ.get("WAKE_WORDS") or "nyx,nix,nyxie,nicks,knicks,nix's").lower().split(","))
WAKE_WINDOW_S = float(os.environ.get("WAKE_WINDOW_S", "45"))
_engaged_until: dict[str, float] = {}   # node -> epoch it stays awake until


def _detect_wake(text: str) -> tuple[bool, str]:
    """(addressed?, message). True if a name-variant appears; strips a leading
    'hey/hi <name>' so the rest is the actual message."""
    tokens = re.findall(r"[a-z']+", text.lower())
    if not any(t in WAKE_WORDS for t in tokens):
        return False, text
    names = "|".join(re.escape(w) for w in WAKE_WORDS)
    cleaned = re.sub(rf"^\s*(hey|hi|ok|okay|yo|hello)?\s*(?:{names})\b[\s,.!?:;-]*",
                     "", text, count=1, flags=re.IGNORECASE).strip()
    return True, cleaned


def handle_turn(asr, orch, wav: bytes, node: str = "default", now: float | None = None,
                context_provider=None) -> dict:
    """Core turn: audio -> transcript -> (wake gate) -> orchestrated reply.
    Pure + testable. Safety runs inside orch.turn(). `context_provider(query)`
    supplies ambient info (presence + document lookup) for THIS turn only — folded
    into the system prompt, never the message/history/memory — and only when she
    actually responds."""
    if now is None:
        now = time.time()
    text = asr.transcribe(wav)
    if not text:
        return {"reply": "", "heard": "", "crisis": False, "addressed": False}

    engaged = WAKE_WINDOW_S <= 0 or _engaged_until.get(node, 0.0) > now
    is_wake, cleaned = _detect_wake(text)
    if not engaged and not is_wake:
        return {"reply": "", "heard": text, "crisis": False, "addressed": False}

    msg = (cleaned if is_wake else text).strip() or "(the user just said your name)"
    context = context_provider(msg) if context_provider else None
    res = orch.turn(msg, directive=context)
    if WAKE_WINDOW_S > 0:
        _engaged_until[node] = now + WAKE_WINDOW_S   # keep the conversation open
    return {"reply": res.reply, "heard": text, "crisis": res.crisis, "addressed": True}


# module-level so the single-threaded server shares one orchestrator (turns
# serialize naturally — exactly what one shared memory/history wants).
ORCH: Orchestrator | None = None
ASR = None


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200, headers=None):
        body = json.dumps(obj).encode()
        if headers and headers.get("X-Enc") == "1":
            body = LINK.wrap(body)          # encrypt the reply for the wire
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
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
        raw = self.rfile.read(n)
        enc = self.headers.get("X-Enc") == "1"
        if enc:
            if not LINK.on:
                self.send_error(400, "encrypted node but brain has no COMPANION_LINK_KEY")
                return
            try:
                wav = LINK.unwrap(raw)
            except Exception:
                self.send_error(400, "bad link ciphertext")
                return
        else:
            wav = raw                       # plaintext node (e.g. a bare Pi)
        node = self.headers.get("X-Node", "default")
        out = handle_turn(ASR, ORCH, wav, node, context_provider=_context_provider)
        resp_headers = {"X-Enc": "1"} if enc else None
        if out["heard"] and out.get("addressed"):
            print(f"[{node}] you: {out['heard']}\n[{node}] {ORCH.persona.name}: {out['reply']}")
        elif out["heard"]:
            print(f"[{node}] (dormant, not addressed) heard: {out['heard']}")
        self._json(out, headers=resp_headers)

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
