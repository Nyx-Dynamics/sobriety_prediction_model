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
import urllib.request
from pathlib import Path

try:
    from companion.voice import Mic, PiperTTS
except ImportError:  # pragma: no cover
    from .voice import Mic, PiperTTS


def post_turn(brain_url: str, wav: bytes, node: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{brain_url.rstrip('/')}/turn", data=wav,
        headers={"Content-Type": "audio/wav", "X-Node": node})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


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
    ap.add_argument("--lead-silence-ms", type=int, default=700)
    args = ap.parse_args()

    mic = Mic(device=args.mic_device, threshold=args.threshold, silence_ms=args.silence_ms)
    tts = PiperTTS(aplay_device=args.speaker, lead_silence_ms=args.lead_silence_ms)
    print(f"[satellite:{args.node}] -> {args.brain}   Ctrl-C to stop.")
    try:
        while True:
            wav = mic.record_until_silence()
            try:
                res = post_turn(args.brain, wav, args.node)
            except Exception as e:
                print(f"  [brain unreachable: {e}]")
                continue
            reply = (res.get("reply") or "").strip()
            if not reply:
                continue
            heard = res.get("heard", "")
            print(f"you › {heard}")
            print(f"› {reply}")
            tts.say(reply)
    except KeyboardInterrupt:
        print("\ntake care.")


if __name__ == "__main__":
    main()
