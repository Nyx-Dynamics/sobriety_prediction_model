"""
Reachy Mini body — expression adapter for the companion.
========================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Maps conversational state to Reachy Mini's head/antenna motion so the companion
has a body that reacts. Pure presentation: every hook is best-effort and
swallows its own errors — a motor hiccup can never interrupt the conversation
or the safety path.

Reachy Mini (Pollen Robotics / Hugging Face) is a strong fit for this project:
  * Wireless model has a Raspberry Pi CM4 ONBOARD → the whole local companion
    (orchestrator + whisper.cpp + Piper + encrypted memory) runs on the robot,
    offline. Closed system, embodied.
  * Apache-2.0 + Python SDK → no reverse-engineering (unlike locked consumer bots).
  * Camera + 4 mics + speaker → everything voice.py needs.

Compute reality: the CM4 comfortably runs ASR/TTS/orchestrator/motion, but NOT a
large LLM. Two closed-system options:
  (a) a small local model on the CM4 (e.g. llama3.2:1b/3b) — lower quality; or
  (b) keep the LLM on a mini-PC/desktop on YOUR LAN and point LocalBackend at it
      (COMPANION_LOCAL_URL=http://<lan-box>:11434/v1). Still closed — never leaves
      your network — with full model quality. Recommended.

Requires the `reachy_mini` SDK on the robot. Import is guarded: if the SDK isn't
present, ReachyBody is inert (all hooks no-op), so code imports fine off-robot.
"""

from __future__ import annotations


class ReachyBody:
    """Expression hooks for Reachy Mini. Neutral/attentive/thinking/speaking
    head poses via the SDK. Poses are intentionally small and calm — this is a
    recovery companion, not a toy."""

    def __init__(self):
        self._mini = None
        try:
            from reachy_mini import ReachyMini  # guarded — only present on-robot
            self._mini = ReachyMini().__enter__()
            self._ReachyMini = ReachyMini
        except Exception:
            self._mini = None  # inert off-robot; hooks become no-ops

    @property
    def available(self) -> bool:
        return self._mini is not None

    def _pose(self, **kw):
        """Best-effort head move. Never raises into the conversation loop."""
        if self._mini is None:
            return
        try:
            self._mini.goto_target(head=kw)   # SDK: mini.goto_target(head=pose)
        except Exception:
            pass

    # ── expression hooks (subtle, calm) ──────────────────────────────────
    def on_listen(self):
        self._pose(pitch=8, yaw=0)     # lean in slightly, attentive

    def on_think(self):
        self._pose(pitch=-4, yaw=10)   # small tilt away — considering

    def on_speak(self):
        self._pose(pitch=0, yaw=0)     # face forward to talk

    def on_idle(self):
        self._pose(pitch=2, yaw=0)     # relaxed neutral

    def close(self):
        if self._mini is not None:
            try:
                self._mini.__exit__(None, None, None)
            except Exception:
                pass
