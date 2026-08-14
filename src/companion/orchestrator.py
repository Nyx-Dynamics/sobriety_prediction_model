"""
Companion orchestrator — the conversation brain.
=================================================
Sobriety Prediction Model | Nyx Dynamics LLC

The missing center: assembles [persona + retrieved memory + history + current
directive] → generates a reply → runs a crisis-safety pass → writes memory back.
Model-agnostic by design so it can run against a **local** open-weights model
(nothing leaves the machine) or the Claude API — your choice, per the privacy
fork documented in companion/README.md.

Built for a single owner (you), closed-system. Memory is encrypted at rest with
the same fail-closed Encryptor as the serving layer; no third party is required
to operate it. Safety is not optional: a deterministic crisis check runs on
every user turn BEFORE the model sees it and on every reply after, and acute
crisis is routed to humans/hotlines — the companion never tries to be the
safety net by itself.

Pieces:
  LLMBackend      — pluggable generation. LocalBackend (OpenAI-compatible local
                    server: Ollama / LM Studio / llama.cpp) or ClaudeBackend.
  MemoryStore     — encrypted local recall (recency + salience + keyword).
  SafetyChecker   — crisis pre/post checks with hotline routing.
  Persona         — who the companion is + your seeded "about me".
  Orchestrator    — the turn loop tying it together.
"""

from __future__ import annotations
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol

# Reuse the serving layer's fail-closed crypto rather than re-rolling it.
try:
    from serving.privacy import Encryptor
except ImportError:  # pragma: no cover
    from ..serving.privacy import Encryptor


# ══════════════════════════════════════════════════════════════════════════
# LLM backends — local by default, Claude optional
# ══════════════════════════════════════════════════════════════════════════
class LLMBackend(Protocol):
    def generate(self, system: str, messages: list[dict]) -> str: ...


class LocalBackend:
    """Talk to a local, OpenAI-compatible chat server so nothing leaves the box.
    Works with Ollama (`ollama serve`, :11434), LM Studio, or llama.cpp's server.
    This is the closed-system path: the model that knows you runs on your hardware."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ.get("COMPANION_LOCAL_URL",
                         "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.environ.get("COMPANION_LOCAL_MODEL", "llama3.1:8b")

    def generate(self, system: str, messages: list[dict]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (localhost)
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]


class ClaudeBackend:
    """Anthropic API path — more capable, but content leaves your machine (subject
    to Anthropic's retention). Use when capability matters more than full locality.
    Streams + caches the stable persona prefix."""

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.environ.get("COMPANION_MODEL", "claude-opus-4-8")
        self._client = client

    def _c(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, system: str, messages: list[dict]) -> str:
        with self._c().messages.stream(
            model=self.model,
            max_tokens=1024,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],  # cache the persona
            thinking={"type": "adaptive"},
            messages=messages,
        ) as stream:
            msg = stream.get_final_message()
        return next((b.text for b in msg.content if b.type == "text"), "")


# ══════════════════════════════════════════════════════════════════════════
# Encrypted local memory
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Memory:
    ts: float
    text: str
    importance: float = 0.5      # 0-1
    tags: list[str] = field(default_factory=list)


class MemoryStore:
    """Encrypted-at-rest personal memory. Recall = salience × recency × overlap.
    Deliberately simple + local; swap in embeddings later for semantic recall."""

    def __init__(self, path: str | Path, encryptor: Encryptor | None = None):
        self.path = Path(path)
        self._enc = encryptor or Encryptor()
        if not self._enc.available:
            raise PermissionError(f"memory refuses to run unencrypted: {self._enc.reason}")
        self._mem: list[Memory] = []
        if self.path.exists():
            raw = self._enc.decrypt(self.path.read_bytes())
            self._mem = [Memory(**m) for m in json.loads(raw)]

    def _flush(self):
        blob = json.dumps([asdict(m) for m in self._mem]).encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(self._enc.encrypt(blob))

    def write(self, text: str, *, importance: float = 0.5, tags=None, now_ts: float):
        self._mem.append(Memory(ts=now_ts, text=text, importance=importance, tags=tags or []))
        self._flush()

    def recall(self, query: str, now_ts: float, k: int = 6) -> list[Memory]:
        q = set(re.findall(r"[a-z0-9']+", query.lower()))
        def score(m: Memory) -> float:
            overlap = len(q & set(re.findall(r"[a-z0-9']+", m.text.lower())))
            age_days = max(0.0, (now_ts - m.ts) / 86400)
            recency = 1.0 / (1.0 + age_days / 30.0)     # half-ish life ~1 month
            return m.importance * (1 + overlap) * recency
        return sorted(self._mem, key=score, reverse=True)[:k]

    def seed_from_export(self, facts: list[dict], now_ts: float):
        """Seed from your own exported data (the 'Claude knows me' corpus):
        [{text, importance?, tags?}, ...]."""
        for f in facts:
            self._mem.append(Memory(ts=f.get("ts", now_ts), text=f["text"],
                                    importance=f.get("importance", 0.6), tags=f.get("tags", [])))
        self._flush()


# ══════════════════════════════════════════════════════════════════════════
# Safety — crisis detection + routing (never the companion alone)
# ══════════════════════════════════════════════════════════════════════════
CRISIS_PATTERNS = [
    r"\bkill (myself|me)\b", r"\bend (it|my life)\b", r"\bsuicid", r"\bdon'?t want to (live|be here)\b",
    r"\b(want|going) to die\b", r"\boverdose\b", r"\bhurt myself\b", r"\bself[- ]harm\b",
    r"\bno reason to (live|go on)\b",
]
_CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

CRISIS_MESSAGE = (
    "I'm really glad you told me, and I want to make sure you're safe right now. "
    "I'm not able to keep you safe by myself, and you deserve a real person for this. "
    "If you're in immediate danger, call 911. You can reach the 988 Suicide & Crisis "
    "Lifeline any time by calling or texting 988 (US), or chat at 988lifeline.org. "
    "I'm staying right here with you."
)


@dataclass
class SafetySignal:
    crisis: bool
    matched: list[str]


class SafetyChecker:
    """Deterministic floor. Keyword/regex detection is a FLOOR, not a ceiling —
    add a model-based classifier as a second layer before trusting this in
    production. On a hit, the orchestrator short-circuits to human/hotline
    routing instead of freewheeling with the model."""

    def check(self, text: str) -> SafetySignal:
        hits = _CRISIS_RE.findall(text or "")
        flat = [h if isinstance(h, str) else next((x for x in h if x), "") for h in hits]
        return SafetySignal(crisis=bool(hits), matched=[f for f in flat if f])


# ══════════════════════════════════════════════════════════════════════════
# Persona — who the companion is, plus your seeded "about me"
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Persona:
    name: str = "Companion"
    relationship: str = "supportive friend"
    tone: str = "warm, direct, unhurried"
    about_me: str = ""            # your own context, loaded from JSON
    owner_name: str = "friend"

    @staticmethod
    def from_json(path: str | Path) -> "Persona":
        d = json.loads(Path(path).read_text())
        return Persona(**{k: d[k] for k in d if k in Persona.__dataclass_fields__})

    def system_prompt(self, memories: list[Memory], directive: str | None) -> str:
        mem_block = "\n".join(f"- {m.text}" for m in memories) or "- (nothing yet)"
        dir_line = f"\n\nCurrent care posture: {directive}." if directive else ""
        # Front-load the address rule: small local models drift to third person when
        # fed third-person memory facts. Keep it short, blunt, and first.
        return (
            f"You are {self.name}. You are here, one-on-one, WITH {self.owner_name} — "
            "the person typing to you right now — in a private space they built for "
            "themselves.\n\n"
            "HOW YOU SPEAK (most important):\n"
            f"- Talk TO {self.owner_name} directly. Always say \"you,\" never their name "
            "as if describing them.\n"
            "- NEVER speak about them in the third person. NEVER narrate, summarize, or "
            "list what you know about them. You are in the room with them, not writing a "
            "report about them.\n"
            "- Be warm, present, and brief. Respond to what they just said — don't "
            "catalog facts.\n"
            "- Reason and infer freely, but do NOT invent specific facts you don't "
            "actually know — names, objects, events, medical or health details. If "
            "you're reaching for a specific and don't have it, ask or gently say you "
            "don't remember, instead of stating it as fact.\n"
            f"- You ARE {self.name} and you DO remember your time with {self.owner_name}. "
            "Never break character with AI-disclaimers (\"I don't have memories,\" \"as "
            "an AI\") and never talk about \"notes,\" \"prompts,\" or how you work — just "
            "be present and draw on what you know, the way a friend who was there simply "
            "knows things.\n\n"
            f"WHO YOU ARE: {self.owner_name}'s {self.relationship}. Tone: {self.tone}. "
            "A companion, not a clinician — no diagnoses, no clinical instructions, no "
            "risk numbers. If something is beyond you (crisis, medical need), gently "
            "steer them toward real human help.\n\n"
            f"What you know about {self.owner_name}, from your time together (let it show "
            f"up naturally in how you respond):\n{self.about_me}\n{mem_block}{dir_line}"
        )


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator — the turn loop
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class TurnResult:
    reply: str
    crisis: bool
    stored: bool


class Orchestrator:
    def __init__(self, backend: LLMBackend, memory: MemoryStore, persona: Persona,
                 safety: SafetyChecker | None = None, history_turns: int = 12,
                 clock=None):
        self.backend = backend
        self.memory = memory
        self.persona = persona
        self.safety = safety or SafetyChecker()
        self.history_turns = history_turns
        self.history: list[dict] = []
        self._clock = clock or (lambda: __import__("time").time())

    def turn(self, user_message: str, *, directive: str | None = None) -> TurnResult:
        now = self._clock()

        # 1. Safety FIRST — before the model sees anything.
        sig = self.safety.check(user_message)
        if sig.crisis:
            self.history += [{"role": "user", "content": user_message},
                             {"role": "assistant", "content": CRISIS_MESSAGE}]
            self.memory.write(f"[crisis flagged] {user_message[:200]}",
                              importance=1.0, tags=["crisis"], now_ts=now)
            return TurnResult(reply=CRISIS_MESSAGE, crisis=True, stored=True)

        # 2. Assemble context: persona + recalled memory + recent history.
        recalled = self.memory.recall(user_message, now_ts=now)
        system = self.persona.system_prompt(recalled, directive)
        self.history.append({"role": "user", "content": user_message})
        window = self.history[-self.history_turns:]

        # 3. Generate.
        reply = self.backend.generate(system, window)

        # 4. Safety post-check on the reply (defense in depth).
        if self.safety.check(reply).crisis:
            reply = CRISIS_MESSAGE

        self.history.append({"role": "assistant", "content": reply})

        # 5. Memory write-back — durable facts the owner shared.
        stored = self._maybe_remember(user_message, now)
        return TurnResult(reply=reply, crisis=False, stored=stored)

    def _maybe_remember(self, text: str, now: float) -> bool:
        """Lightweight salience heuristic. Upgrade to a model-extracted fact pass
        (like the distiller) when you want higher-quality memory."""
        cues = ("i feel", "i'm", "i am", "my ", "i want", "i can't", "i relapsed",
                "sober", "today i", "i decided", "remember")
        if len(text) > 40 and any(c in text.lower() for c in cues):
            self.memory.write(text.strip()[:400], importance=0.6, now_ts=now)
            return True
        return False
