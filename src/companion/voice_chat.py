"""
Voice companion launcher for the Raspberry Pi.
==============================================
Fully local by default: local model + local ASR + local TTS + on-device presence.

Setup (one-time, on the Pi):
    # ASR: build whisper.cpp, fetch a model
    #   git clone https://github.com/ggerganov/whisper.cpp && make -C whisper.cpp
    #   ./whisper.cpp/models/download-ggml-model.sh base.en
    # TTS: install piper, fetch a voice (.onnx + .onnx.json)
    #   pipx install piper-tts   # or download the release binary
    # Model: ollama pull llama3.1:8b   (or a smaller model for the Pi)

Run:
    export SUD_PHI_KEY=$(python -c "from serving.privacy import Encryptor; print(Encryptor.generate_key())")
    export WHISPER_BIN=./whisper.cpp/main WHISPER_MODEL=./whisper.cpp/models/ggml-base.en.bin
    export PIPER_BIN=piper PIPER_VOICE=./voices/en_US-amy-medium.onnx
    python -m companion.voice_chat --persona companion/persona.example.json
    #   add --camera to enable presence-gated wake (frames never stored)
"""

from __future__ import annotations
import argparse
from pathlib import Path

try:
    from companion.orchestrator import Orchestrator, MemoryStore, Persona, LocalBackend, ClaudeBackend
    from companion.voice import VoiceLoop, Mic, WhisperCppASR, PiperTTS, PresenceCamera
except ImportError:  # pragma: no cover
    from .orchestrator import Orchestrator, MemoryStore, Persona, LocalBackend, ClaudeBackend
    from .voice import VoiceLoop, Mic, WhisperCppASR, PiperTTS, PresenceCamera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "claude"], default="local")
    ap.add_argument("--persona", default=None)
    ap.add_argument("--memory", default=str(Path.home() / ".companion" / "memory.enc"))
    ap.add_argument("--camera", action="store_true", help="presence-gated wake (no frames stored)")
    ap.add_argument("--reachy", action="store_true", help="drive Reachy Mini head/antenna expression")
    ap.add_argument("--mic-device", default=None, help="ALSA capture device, e.g. plughw:1,0")
    ap.add_argument("--speaker", default=None, help="ALSA playback device, e.g. plughw:2,0")
    ap.add_argument("--threshold", type=float, default=500.0, help="VAD energy threshold")
    ap.add_argument("--silence-ms", type=int, default=1500,
                    help="trailing silence (ms) that ends your turn — raise for more pause room")
    ap.add_argument("--lead-silence-ms", type=int, default=700,
                    help="silence prepended to speech to absorb HDMI/TV audio wake-up clipping")
    args = ap.parse_args()

    backend = ClaudeBackend() if args.backend == "claude" else LocalBackend()
    persona = Persona.from_json(args.persona) if args.persona else Persona()
    orch = Orchestrator(backend, MemoryStore(args.memory), persona)

    body = None
    if args.reachy:
        from companion.reachy import ReachyBody
        body = ReachyBody()

    loop = VoiceLoop(
        orchestrator=orch,
        asr=WhisperCppASR(),
        tts=PiperTTS(aplay_device=args.speaker, lead_silence_ms=args.lead_silence_ms),
        mic=Mic(device=args.mic_device, threshold=args.threshold, silence_ms=args.silence_ms),
        camera=PresenceCamera(enabled=args.camera),
        body=body,
    )
    loop.run()


if __name__ == "__main__":
    main()
