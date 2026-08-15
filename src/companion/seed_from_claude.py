"""
Seed Nyx's memory from your Claude data export.
================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Turns your Claude conversation history — the richest "about me" corpus you have —
into memory facts for the companion. Produces a JSON file in the seed format;
NOTHING enters memory until you review it and run with --seed, so you stay in
control of what Nyx carries.

Get your export:  claude.ai → Settings → Privacy → "Export data" → you receive a
ZIP by email containing conversations.json. Unzip it, point this at that file.

Modes:
  raw      Your own (human) messages become low-weight memories. Fast, noisy —
           a quick way to give Nyx broad context. Good first pass.
  distill  An LLM reads each conversation and extracts a few DURABLE facts worth
           a companion remembering. Cleaner, slower.
             --backend local   → the Pi's model (fully closed; nothing leaves)
             --backend claude  → higher quality (data is already Anthropic-side)

Usage (produce facts, then review, then seed):
  python -m companion.seed_from_claude conversations.json -o facts.json --mode distill --backend local
  #   ... open facts.json, delete anything you don't want Nyx to hold ...
  python -m companion.chat --persona companion/persona.example.json --seed facts.json

Timestamps from the original conversations are preserved, so old history is aged
correctly in recall (recent facts surface first, but importance still carries).
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# ── tolerant export parsing ──────────────────────────────────────────────
def _blocks_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _epoch(iso) -> float:
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def iter_messages(export_path: str):
    """Yield (conv_name, sender, text, ts) tolerating schema vari/blanks."""
    data = json.loads(Path(export_path).read_text())
    if isinstance(data, dict):
        data = data.get("conversations", [data])
    for conv in data:
        name = conv.get("name") or conv.get("title") or ""
        created = _epoch(conv.get("created_at"))
        msgs = conv.get("chat_messages") or conv.get("messages") or []
        for m in msgs:
            sender = (m.get("sender") or m.get("role") or "").lower()
            text = (m.get("text") or _blocks_text(m.get("content")) or "").strip()
            if text:
                yield name, sender, text, created


# ── raw mode ─────────────────────────────────────────────────────────────
def build_raw(export_path: str, min_chars: int, self_only: bool) -> list[dict]:
    facts, seen = [], set()
    for name, sender, text, ts in iter_messages(export_path):
        if self_only and sender not in ("human", "user"):
            continue
        if len(text) < min_chars:
            continue
        key = text[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append({"text": text[:500], "importance": 0.4,
                      "tags": ["claude-history"], "ts": ts})
    return facts


# ── distill mode ─────────────────────────────────────────────────────────
_DISTILL_SYSTEM = (
    "You read one conversation between a user and an AI assistant. Extract up to 5 "
    "DURABLE facts about the USER that a personal companion should remember long-term "
    "— identity, people in their life, history, health/recovery, preferences, values, "
    "ongoing situations. Skip anything transient or about the assistant. Output one "
    "fact per line, concise, third person ('A.C. ...'). If nothing durable, output nothing."
)


def _conversation_transcripts(export_path: str, limit: int | None):
    """Group messages back into per-conversation transcripts."""
    convos: dict[str, dict] = {}
    order: list[str] = []
    for name, sender, text, ts in iter_messages(export_path):
        key = name or f"conv-{len(order)}"
        if key not in convos:
            convos[key] = {"name": name, "ts": ts, "lines": []}
            order.append(key)
        who = "User" if sender in ("human", "user") else "Assistant"
        convos[key]["lines"].append(f"{who}: {text}")
    items = [convos[k] for k in order]
    items.sort(key=lambda c: c["ts"], reverse=True)   # most recent first
    return items[:limit] if limit else items


def build_distill(export_path: str, backend, limit: int | None,
                  max_chars: int = 12000, out_path: str | None = None,
                  checkpoint_every: int = 25) -> list[dict]:
    facts = []
    convos = _conversation_transcripts(export_path, limit)
    for i, conv in enumerate(convos, 1):
        transcript = "\n".join(conv["lines"])[:max_chars]
        if len(transcript) < 80:
            continue
        try:
            out = backend.generate(_DISTILL_SYSTEM, [{"role": "user", "content": transcript}])
        except Exception as e:
            print(f"  [skip {i}/{len(convos)}] {conv['name'][:40]}: {e}", flush=True)
            continue
        for line in out.splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if len(line) > 8:
                facts.append({"text": line[:400], "importance": 0.65,
                              "tags": ["claude-history", "distilled"], "ts": conv["ts"]})
        print(f"  [{i}/{len(convos)}] {conv['name'][:40]} → {len(facts)} facts so far", flush=True)
        # checkpoint so a long/overnight run that gets interrupted still saves
        if out_path and i % checkpoint_every == 0:
            Path(out_path).write_text(json.dumps(facts, indent=2))
    if out_path:
        Path(out_path).write_text(json.dumps(facts, indent=2))
    return facts


# ── memories mode (best: ingest Claude's own distilled memory of you) ────────
import re as _re


def _split_facts(prose: str) -> list[str]:
    """Break Claude's memory prose into standalone fact-sized chunks."""
    out = []
    for chunk in _re.split(r"\n+", prose):
        chunk = chunk.strip().lstrip("-•*0123456789. ").strip()
        if not chunk:
            continue
        # further split long paragraphs on sentence boundaries
        parts = _re.split(r"(?<=[.!?])\s+(?=[A-Z])", chunk) if len(chunk) > 300 else [chunk]
        for p in parts:
            p = p.strip()
            if 15 <= len(p) <= 400:
                out.append(p)
    return out


def _project_names(memories_path: str) -> dict:
    """Map project UUID -> name using the sibling projects/ dir (if present)."""
    pdir = Path(memories_path).parent / "projects"
    names = {}
    if pdir.is_dir():
        for fn in pdir.glob("*.json"):
            try:
                d = json.loads(fn.read_text())
                if isinstance(d, dict):
                    names[fn.stem] = d.get("name") or d.get("title") or ""
            except Exception:
                pass
    return names


def _slug(s: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:32]


def build_from_memories(memories_path: str, include_projects: bool = False,
                        project_filter: list[str] | None = None,
                        core: bool = True) -> list[dict]:
    """Ingest memories.json -> facts. No model needed; source is already curated.
    Core = conversations_memory (always). Projects are opt-in and, with
    project_filter (name substrings, case-insensitive), selectable by name — so a
    'depth layer' is exactly the personal projects you pick, not the work firehose.
    Omits ts so facts get the current time at seed (they recall well, not aged out)."""
    data = json.loads(Path(memories_path).read_text())
    if isinstance(data, list):
        data = data[0] if data else {}
    facts = []
    if core:
        for f in _split_facts(data.get("conversations_memory") or ""):
            facts.append({"text": f, "importance": 0.7, "tags": ["claude-memory"]})
    if not include_projects and not project_filter:
        return facts

    names = _project_names(memories_path)
    filt = [x.lower() for x in (project_filter or [])]
    pm = data.get("project_memories") or {}
    for uuid, txt in pm.items():
        if not isinstance(txt, str):
            continue
        name = names.get(uuid, "")
        if filt and not any(x in name.lower() for x in filt):
            continue      # selecting by name and this one doesn't match -> skip
        tag = _slug(name) or "project"
        for f in _split_facts(txt):
            facts.append({"text": f, "importance": 0.6, "tags": ["project-memory", tag]})
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="conversations.json (raw/distill) or memories.json (memories)")
    ap.add_argument("-o", "--out", default="facts.json", help="output seed file")
    ap.add_argument("--mode", choices=["raw", "distill", "memories"], default="distill")
    ap.add_argument("--backend", choices=["local", "claude"], default="local",
                    help="distill mode only")
    ap.add_argument("--limit", type=int, default=None, help="distill: most recent N conversations")
    ap.add_argument("--min-chars", type=int, default=60, help="raw: min message length")
    ap.add_argument("--self-only", action="store_true", help="raw: only your own messages")
    ap.add_argument("--projects", action="store_true",
                    help="memories: include ALL project_memories (large, work-specific)")
    ap.add_argument("--project", action="append", default=None, metavar="NAME",
                    help="memories: include only projects whose name matches (repeatable)")
    ap.add_argument("--no-core", action="store_true",
                    help="memories: skip conversations_memory (for a projects-only depth add-on)")
    args = ap.parse_args()

    if args.mode == "memories":
        facts = build_from_memories(args.export, include_projects=args.projects,
                                    project_filter=args.project, core=not args.no_core)
    elif args.mode == "raw":
        facts = build_raw(args.export, args.min_chars, args.self_only)
    else:
        try:
            from companion.orchestrator import LocalBackend, ClaudeBackend
        except ImportError:
            from .orchestrator import LocalBackend, ClaudeBackend
        backend = ClaudeBackend() if args.backend == "claude" else LocalBackend()
        print(f"Distilling with {args.backend} backend (checkpointing to {args.out})...")
        facts = build_distill(args.export, backend, args.limit, out_path=args.out)

    Path(args.out).write_text(json.dumps(facts, indent=2))
    print(f"\nWrote {len(facts)} facts → {args.out}")
    print(f"REVIEW it, delete anything you don't want Nyx to hold, then:")
    print(f"  python -m companion.chat --persona companion/persona.example.json --seed {args.out}")


if __name__ == "__main__":
    main()
