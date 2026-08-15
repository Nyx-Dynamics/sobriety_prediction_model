"""
PiCarBody — a physical expression layer for the companion (SunFounder PiCar-X).
===============================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Gives her a *body that looks at you*. Implements the voice.py `Body` protocol
(on_listen / on_think / on_speak / on_idle) by driving the PiCar-X's pan/tilt
camera head — the expressive "face" — plus the two things that actually make an
embodiment feel alive:

  * FREEZE-TO-LISTEN — the instant the mic opens she STOPS all motion, so servo
    whine never bleeds into what you're saying. Attention, rendered physically.
  * A perk + nod when she's about to speak; a settle when the conversation rests.

Deliberately conservative by design:
  * It expresses with the HEAD (pan/tilt) and a subtle steering "lean" — it does
    NOT drive the wheels around during conversation. Autonomous approach toward a
    person needs localization + obstacle safety we don't have yet; a servo face is
    80% of "alive" with none of the drive-off-the-landing risk. `approach()` /
    `retreat()` exist but are unwired, gated on future presence/vision.
  * SunFounder's stock cloud-AI app (ChatGPT/Gemini/…) is ignored entirely — we
    use ONLY the low-level `picarx` hardware library. Her mind stays on the Studio.

Hardware-optional: if `picarx` isn't importable (i.e. you're not on the Pi/HAT),
it runs as a STUB that logs the gestures it *would* make — so the whole thing is
testable on a laptop before the kit ever ships.

Note (honest limits of this chassis):
  * No sound-direction sensor (the PiDog has one; the car doesn't) — so she faces
    forward attentively rather than turning toward where your voice came from.
    `orient(pan)` is here for the day a mic array or presence gives us a bearing.
  * The Body hooks carry conversational PHASE, not emotional tone — so tone-reactive
    gestures (a warmer nod on tenderness) are a future extension that needs the
    reply's sentiment threaded through. Marked below.
"""

from __future__ import annotations
import os
import time

try:
    from companion.voice import NullBody
except ImportError:  # pragma: no cover
    from .voice import NullBody

# Camera-head geometry (picarx conventions; degrees):
#   cam_pan  -90..90   0 = facing forward
#   cam_tilt -35..65   0 = level, + = looking up
#   steering -30..30   0 = wheels straight  (used only for a subtle "lean")
_ATTENTIVE_TILT = 14     # ears-up: "I'm listening"
_LEVEL_TILT = 5          # engaged, facing you
_REST_TILT = -8          # relaxed, settled
_NOD_TILT = 20           # top of a nod


class PiCarBody:
    """voice.py Body for the PiCar-X. Every hook swallows its own errors — a
    servo fault must never interrupt the conversation."""

    def __init__(self, debug: bool = False):
        self.debug = debug or os.environ.get("PICAR_DEBUG") == "1"
        self._px = None
        self._reason = ""
        try:
            from picarx import Picarx
            self._px = Picarx()
        except Exception as e:      # not on the Pi/HAT, or picarx missing -> stub
            self._reason = f"{type(e).__name__}: {e}"
        self._log(f"init ({'hardware' if self._px else 'STUB — ' + self._reason})")
        self._center()

    # ── low-level, all guarded ───────────────────────────────────────────
    def _log(self, msg: str):
        if self.debug:
            print(f"[picar] {msg}", flush=True)

    def _pan(self, a: float):
        if self._px:
            self._px.set_cam_pan_angle(int(max(-90, min(90, a))))

    def _tilt(self, a: float):
        if self._px:
            self._px.set_cam_tilt_angle(int(max(-35, min(65, a))))

    def _steer(self, a: float):
        if self._px:
            self._px.set_dir_servo_angle(int(max(-30, min(30, a))))

    def _stop(self):
        if self._px:
            self._px.stop()      # kill drive motors

    def _center(self):
        try:
            self._stop(); self._pan(0); self._tilt(_LEVEL_TILT); self._steer(0)
        except Exception:
            pass

    # ── Body protocol: conversational-state hooks ────────────────────────
    def on_listen(self):
        """Mic just opened. FREEZE — no motion into the mic — and hold an
        attentive, head-up pose facing forward."""
        try:
            self._stop()             # critical: silence the servos to hear
            self._pan(0)
            self._tilt(_ATTENTIVE_TILT)
            self._steer(0)
            self._log("listen: freeze + attentive")
        except Exception:
            pass

    def on_think(self):
        """Reply is generating. A small 'hmm' head-cock — brief, then hold."""
        try:
            self._pan(10); time.sleep(0.12); self._pan(0)   # quick cock-and-return
            self._log("think: head-cock")
        except Exception:
            pass

    def on_speak(self):
        """About to talk. Face forward and give a single warm nod, then settle
        level and engaged for the reply.
        # FUTURE: accept the reply's tone here for a bigger nod on tenderness /
        # a quicker one on lightness — needs sentiment threaded from the brain."""
        try:
            self._pan(0)
            self._tilt(_NOD_TILT); time.sleep(0.13)
            self._tilt(_LEVEL_TILT)
            self._log("speak: perk + nod")
        except Exception:
            pass

    def on_idle(self):
        """Conversation at rest. Settle into a relaxed, head-down pose, motors off."""
        try:
            self._stop()
            self._pan(0)
            self._tilt(_REST_TILT)
            self._log("idle: settle")
        except Exception:
            pass

    def close(self):
        try:
            self._center()
            self._log("close: centered")
        except Exception:
            pass

    # ── unwired, for later (need localization + obstacle safety) ─────────
    def orient(self, pan: float):
        """Turn the head toward a bearing (e.g. from a future mic array/presence)."""
        try:
            self._pan(pan)
        except Exception:
            pass

    def approach(self, speed: int = 20, max_s: float = 1.0):
        """DELIBERATELY UNWIRED. Drive forward briefly. Do NOT call until there's
        obstacle-aware presence/vision — a floor robot driving blind toward a voice
        is how it ends up down the stairs."""
        raise NotImplementedError("approach() is gated on presence/vision safety")


def make_body(kind: str = "none", debug: bool = False):
    """Factory: 'picar' -> PiCarBody, anything else -> voice.NullBody (no-op)."""
    if (kind or "none").lower() == "picar":
        return PiCarBody(debug=debug)
    return NullBody()
