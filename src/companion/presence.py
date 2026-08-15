"""
Presence — she knows who's home and which room (UniFi Protect, Studio-side).
============================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Turns UniFi Protect person/identity detections into an EPHEMERAL presence signal
the brain can use for warmth ("you're in the kitchen") — never a dossier.

Anti-surveillance by construction (see docs/household_agent.md §privacy):
  * Presence is RAM-only and TTL'd. No movement history, no who-was-where log,
    nothing written to disk. Signals inform behavior, then evaporate.
  * OWNER-vs-OTHER collapse. Protect may recognize family/guests who never opted
    in. Identities are reduced to "owner" or "someone" BEFORE anything leaves this
    module — Nyx can never reason about or remember another identified person.
  * Local-only. Talks to the UDM Pro on your LAN; nothing leaves your network.

Runs as its own small daemon next to the brain and serves the current snapshot at
http://localhost:9100/presence. The brain queries it per turn (best-effort).

Config (env):
  UNIFI_HOST, UNIFI_PORT(=443), UNIFI_USER, UNIFI_PASS   — Protect login
  OWNER_NAME               — the Protect-recognized name to treat as the owner
  PRESENCE_ZONES_FILE      — JSON: {"<camera name or id>": "<room>", ...}
  PRESENCE_TTL_S(=90)      — how long a detection counts as "present"
  PRESENCE_PORT(=9100)

The UniFi source is a guarded import: without `uiprotect` installed / configured,
the daemon still runs and just reports "nobody detected" (fail-open to voice-only).
"""

from __future__ import annotations
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


# ══════════════════════════════════════════════════════════════════════════
# Ephemeral presence store (thread-safe, TTL'd, owner/other only)
# ══════════════════════════════════════════════════════════════════════════
class PresenceStore:
    def __init__(self, ttl_s: float = 90.0):
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        # zone -> {"owner": bool, "ts": epoch}. "owner" True if the owner was the
        # detected identity; False means "someone (not identified as owner)".
        self._zones: dict[str, dict] = {}

    def update(self, zone: str, is_owner: bool, now: float) -> None:
        with self._lock:
            cur = self._zones.get(zone)
            # owner presence "wins" a zone over an anonymous person
            if cur and cur["owner"] and not is_owner and now - cur["ts"] < self.ttl_s:
                cur["ts"] = now
            else:
                self._zones[zone] = {"owner": is_owner, "ts": now}

    def snapshot(self, now: float) -> dict:
        """Owner-vs-other only. No identities, no history, expired entries dropped."""
        with self._lock:
            live = {z: v for z, v in self._zones.items() if now - v["ts"] < self.ttl_s}
            self._zones = live  # prune as we read
            owner_zone = next((z for z, v in live.items() if v["owner"]), None)
            others = any(not v["owner"] for v in live.values())
            zones = {z: {"who": "owner" if v["owner"] else "someone",
                         "age_s": round(now - v["ts"], 1)} for z, v in live.items()}
        return {"owner_zone": owner_zone, "others_present": others,
                "zones": zones, "as_of": now}


def presence_sentence(snap: dict) -> str:
    """Turn a snapshot into a short natural line for the companion's context —
    or "" when there's nothing to say. Owner-only; others are just 'someone else'."""
    parts = []
    if snap.get("owner_zone"):
        parts.append(f"They're in the {snap['owner_zone']} right now")
    if snap.get("others_present"):
        parts.append("someone else is home too" if parts else "Someone else is home")
    return ("; ".join(parts) + ".") if parts else ""


# ══════════════════════════════════════════════════════════════════════════
# UniFi Protect source (guarded — needs `uiprotect` + your config to do anything)
# ══════════════════════════════════════════════════════════════════════════
class UniFiProtectSource:
    """Streams Protect person/identity events and feeds the store. Untested here
    against live hardware — configure and we debug against your UDM Pro. If the
    SDK/config isn't present, `available` is False and it stays silent."""

    def __init__(self, store: PresenceStore, zones: dict, owner_name: str):
        self.store = store
        self.zones = {k.lower(): v for k, v in zones.items()}  # camera name/id -> room
        self.owner = (owner_name or "").strip().lower()
        self.available = False
        self._client = None

    def _zone_for(self, camera_name: str, camera_id: str) -> str | None:
        return (self.zones.get((camera_name or "").lower())
                or self.zones.get((camera_id or "").lower()))

    async def run(self):
        try:
            from uiprotect import ProtectApiClient
        except ImportError:
            print("[presence] uiprotect not installed — presence disabled (voice-only).")
            return
        host = os.environ.get("UNIFI_HOST")
        if not host:
            print("[presence] UNIFI_HOST unset — presence disabled.")
            return
        self._client = ProtectApiClient(
            host, int(os.environ.get("UNIFI_PORT", "443")),
            os.environ["UNIFI_USER"], os.environ["UNIFI_PASS"],
            verify_ssl=False,
        )
        await self._client.update()   # bootstrap
        self.available = True
        print(f"[presence] connected to Protect at {host}; watching "
              f"{len(self.zones)} mapped cameras.")
        self._client.subscribe_websocket(self._on_ws)
        # keep the connection alive
        import asyncio
        while True:
            await asyncio.sleep(3600)

    def _on_ws(self, msg):
        """Best-effort event handler. Protect's exact event shape varies by version;
        we look for a person/smart detection carrying a camera + optional identity."""
        try:
            obj = getattr(msg, "new_obj", None) or getattr(msg, "changed_data", None)
            if obj is None:
                return
            cam_id = getattr(obj, "camera_id", "") or ""
            cam = getattr(getattr(obj, "camera", None), "name", "") or ""
            zone = self._zone_for(cam, cam_id)
            if not zone:
                return
            smart = getattr(obj, "smart_detect_types", None) or []
            if not any(str(s).lower() in ("person", "face") for s in smart):
                return
            # identity, if Protect recognized a face — collapse to owner/other HERE
            ident = ""
            for attr in ("smart_detect_face_name", "detected_thumbnails", "metadata"):
                v = getattr(obj, attr, None)
                if isinstance(v, str) and v:
                    ident = v
                    break
            is_owner = bool(self.owner) and ident.strip().lower() == self.owner
            self.store.update(zone, is_owner, time.time())
        except Exception:
            pass  # never let a malformed event break the daemon


# ══════════════════════════════════════════════════════════════════════════
# HTTP endpoint the brain queries
# ══════════════════════════════════════════════════════════════════════════
def _make_handler(store: PresenceStore):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/presence":
                self.send_error(404); return
            body = json.dumps(store.snapshot(time.time())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a): pass
    return H


def main():
    ttl = float(os.environ.get("PRESENCE_TTL_S", "90"))
    port = int(os.environ.get("PRESENCE_PORT", "9100"))
    zones = {}
    zf = os.environ.get("PRESENCE_ZONES_FILE")
    if zf and os.path.exists(zf):
        zones = json.loads(open(zf).read())
    store = PresenceStore(ttl_s=ttl)

    # HTTP endpoint in a thread
    srv = HTTPServer(("127.0.0.1", port), _make_handler(store))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[presence] serving /presence on 127.0.0.1:{port} (ttl={ttl}s, "
          f"{len(zones)} zones)")

    # Protect source in the asyncio loop (or idle if unconfigured)
    import asyncio
    src = UniFiProtectSource(store, zones, os.environ.get("OWNER_NAME", ""))
    try:
        asyncio.run(src.run())
    except KeyboardInterrupt:
        pass
    # if the source returns/short-circuits (no SDK/config), keep serving empty presence
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
