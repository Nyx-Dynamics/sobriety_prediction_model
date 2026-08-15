"""
Reference — a private, local search index over your documents (Studio-side).
============================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Phase 2 of the household build: your ~8,200 PDFs/docs become a searchable
reference she *consults on request* — NOT memory she recalls, NOT personality
she confabulates from. "What did that April filing say?" -> she looks it up,
quotes it, cites the file. If it's not there, she says so.

Closed by construction:
  * Text extraction + chunking + embeddings all run on the Studio. Embeddings via
    the local Ollama (`nomic-embed-text`) — nothing leaves the machine.
  * The chunk text (the sensitive content — litigation, medical, etc.) is stored
    ENCRYPTED (Fernet / SUD_PHI_KEY). Embeddings are vectors (`.npy`); keep
    ~/.companion on a FileVault volume for at-rest cover of those too.
  * Documents are reference-only: retrieved excerpts go into the per-turn system
    prompt (ephemeral), never into her conversation history or memory.

Two modes:
    python -m companion.reference build <root-dir> [<root-dir> ...]   # index (background)
    python -m companion.reference serve                              # search daemon :9200

Env: REFERENCE_DIR(=~/.companion/reference), REFERENCE_EMBED_MODEL(=nomic-embed-text),
     OLLAMA_BASE(=http://localhost:11434), REFERENCE_PORT(=9200).
"""

from __future__ import annotations
import json
import os
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

try:
    from serving.privacy import Encryptor
except ImportError:  # pragma: no cover
    from ..serving.privacy import Encryptor

REFERENCE_DIR = Path(os.environ.get("REFERENCE_DIR", str(Path.home() / ".companion" / "reference")))
EMBED_MODEL = os.environ.get("REFERENCE_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
PORT = int(os.environ.get("REFERENCE_PORT", "9200"))

TEXT_EXT = {"txt", "md", "tex", "csv", "pdf", "docx", "rtf", "eml", "html", "htm"}
SKIP_DIRS = {"site-packages", "node_modules", "__pycache__", ".git", "venv", ".venv"}


# ── extraction / chunking ────────────────────────────────────────────────
def extract_text(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    try:
        if ext in ("txt", "md", "tex", "csv", "eml", "html", "htm", "rtf"):
            return Path(path).read_text(errors="ignore")
        if ext == "pdf":
            from pypdf import PdfReader
            return "\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages)
        if ext == "docx":
            import docx
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
    except Exception:
        return ""
    return ""


def chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if len(piece) > 40:
            out.append(piece)
        i += size - overlap
    return out


# ── embeddings via local Ollama ──────────────────────────────────────────
def embed(texts: list[str]) -> np.ndarray:
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(f"{OLLAMA_BASE}/api/embed", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    embs = data.get("embeddings") or ([data["embedding"]] if "embedding" in data else [])
    a = np.array(embs, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.clip(n, 1e-8, None)   # L2-normalized for cosine via dot


# ── the index (embeddings.npy + encrypted chunks) ────────────────────────
class ReferenceIndex:
    def __init__(self, directory: Path = REFERENCE_DIR, encryptor: Encryptor | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._enc = encryptor or Encryptor()
        if not self._enc.available:
            raise PermissionError(f"reference index refuses to run unencrypted: {self._enc.reason}")
        self.emb: np.ndarray | None = None
        self.chunks: list[dict] = []      # [{"source": rel_path, "text": chunk}]

    # -- build --
    def build(self, roots: list[str], batch: int = 64, checkpoint_every: int = 500):
        chunks, count = [], 0
        for root in roots:
            root = Path(root)
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if any(part in SKIP_DIRS for part in path.parts):
                    continue
                if path.suffix.lower().lstrip(".") not in TEXT_EXT:
                    continue
                text = extract_text(str(path))
                if len(text) < 40:
                    continue
                src = str(path.relative_to(root)) if root in path.parents else path.name
                for ch in chunk(text):
                    chunks.append({"source": src, "text": ch})
                count += 1
                if count % 100 == 0:
                    print(f"  extracted {count} docs, {len(chunks)} chunks so far", flush=True)
        print(f"embedding {len(chunks)} chunks ...", flush=True)
        vecs = []
        for i in range(0, len(chunks), batch):
            vecs.append(embed([c["text"] for c in chunks[i:i + batch]]))
            if (i // batch) % 20 == 0:
                self._save(np.vstack(vecs), chunks[:i + batch])   # checkpoint
                print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}", flush=True)
        self.emb = np.vstack(vecs) if vecs else np.zeros((0, 1), np.float32)
        self.chunks = chunks
        self._save(self.emb, self.chunks)
        print(f"done: {len(chunks)} chunks from {count} docs -> {self.dir}", flush=True)

    def _save(self, emb: np.ndarray, chunks: list[dict]):
        np.save(self.dir / "embeddings.npy", emb)
        (self.dir / "chunks.enc").write_bytes(self._enc.encrypt(json.dumps(chunks).encode()))

    # -- load / search --
    def load(self):
        self.emb = np.load(self.dir / "embeddings.npy")
        self.chunks = json.loads(self._enc.decrypt((self.dir / "chunks.enc").read_bytes()))
        return self

    def search(self, query: str, k: int = 4, min_score: float = 0.35) -> list[dict]:
        if self.emb is None or not len(self.emb):
            return []
        q = embed([query])[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:k]
        return [{"source": self.chunks[i]["source"], "text": self.chunks[i]["text"],
                 "score": float(scores[i])} for i in idx if scores[i] >= min_score]


# ── search daemon (brain queries GET /search?q=...) ──────────────────────
def _handler(index: ReferenceIndex):
    from urllib.parse import urlparse, parse_qs
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            if u.path != "/search":
                self.send_error(404); return
            q = (parse_qs(u.query).get("q") or [""])[0]
            hits = index.search(q) if q else []
            body = json.dumps({"hits": hits}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a): pass
    return H


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        roots = sys.argv[2:] or [str(Path.home())]
        ReferenceIndex().build(roots)
    elif len(sys.argv) >= 2 and sys.argv[1] == "serve":
        idx = ReferenceIndex().load()
        print(f"[reference] {len(idx.chunks)} chunks; serving /search on 127.0.0.1:{PORT}")
        HTTPServer(("127.0.0.1", PORT), _handler(idx)).serve_forever()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
