"""
Satellite — a dumb ear-and-mouth for the companion (runs on the Pi / a room node).
==================================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Captures a window of audio, ships it to the brain (Studio), and speaks the reply.
No ASR, no model, no memory, no safety logic here — all of that lives on the
brain. One of these runs per room; adding a room = running another one.

Reuses the existing Mic (ALSA capture + VAD endpointing) and PiperTTS (local
speech). Only cognition moved to the Studio; each room still renders speech
locally, so no TV/mic config changes.

Run (on the Pi, from src/):
    export COMPANION_BRAIN_URL=http://STUDIO_IP:9000
    python -m companion.satellite --node bedside \
        --mic-device plughw:2,0 --speaker plughw:1,0 --silence-ms 2000
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

try:
    from companion.voice import Mic, make_tts
    from companion.picar_body import make_body
    from companion.link import LinkCipher
except ImportError:  # pragma: no cover
    from .voice import Mic, make_tts
    from .picar_body import make_body
    from .link import LinkCipher


def post_turn(brain_url: str, wav: bytes, node: str,
              link: "LinkCipher | None" = None, timeout: int = 120) -> dict:
    link = link or LinkCipher()
    body, headers = wav, {"Content-Type": "application/octet-stream", "X-Node": node}
    if link.on:
        body = link.wrap(wav)          # encrypt the voice for the wire
        headers["X-Enc"] = "1"
    req = urllib.request.Request(f"{brain_url.rstrip('/')}/turn", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("X-Enc") == "1":
            raw = link.unwrap(raw)     # decrypt her reply
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=os.environ.get("COMPANION_BRAIN_URL",
                    "http://localhost:9000"), help="brain server URL (the Studio)")
    ap.add_argument("--node", default=os.environ.get("COMPANION_NODE", "default"),
                    help="room/zone name for this satellite")
    ap.add_argument("--mic-device", default=None)
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--threshold", type=float, default=500.0)
    ap.add_argument("--silence-ms", type=int, default=1500)
    # macOS speakers (Thunderbolt/USB) don't have the HDMI wake-up clip, so no lead needed there
    ap.add_argument("--lead-silence-ms", type=int,
                    default=(0 if sys.platform == "darwin" else 700))
    ap.add_argument("--tts", default="auto", choices=["auto", "piper", "say"],
                    help="speech backend (auto: piper if set up here, else macOS say)")
    ap.add_argument("--body", default=os.environ.get("COMPANION_BODY", "none"),
                    choices=["none", "picar"],
                    help="physical expression layer (picar = SunFounder PiCar-X)")
    args = ap.parse_args()

    mic = Mic(device=args.mic_device, threshold=args.threshold, silence_ms=args.silence_ms)
    tts = make_tts(args.tts, aplay_device=args.speaker, lead_silence_ms=args.lead_silence_ms)
    body = make_body(args.body, debug=True)   # NullBody unless --body picar
    link = LinkCipher()
    link_note = "encrypted" if link.on else f"PLAINTEXT ({link.reason})"
    print(f"[satellite:{args.node}] -> {args.brain}   tts={type(tts).__name__}   "
          f"body={args.body}   link={link_note}   Ctrl-C to stop.")
    try:
        while True:
            body.on_listen()                       # freeze + attend while the mic is open
            wav = mic.record_until_silence()
            body.on_think()                        # captured -> ask the brain
            try:
                res = post_turn(args.brain, wav, args.node, link=link)
            except Exception as e:
                print(f"  [brain unreachable: {e}]")
                body.on_idle()
                continue
            reply = (res.get("reply") or "").strip()
            if not reply:
                body.on_idle()                     # not addressed / nothing to say -> settle
                continue
            heard = res.get("heard", "")
            print(f"you › {heard}")
            print(f"› {reply}")
            body.on_speak()                        # perk + nod, then speak
            tts.say(reply)
            body.on_idle()
    except KeyboardInterrupt:
        print("\ntake care.")
    finally:
        body.close()


if __name__ == "__main__":
    main()
