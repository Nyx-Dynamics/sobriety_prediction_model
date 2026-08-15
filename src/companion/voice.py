"""
Voice embodiment — local speech loop for a Raspberry Pi (or any Linux box).
===========================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Wraps the text-in/text-out Orchestrator in a fully local voice loop:

    mic ─► ASR (whisper.cpp) ─► orchestrator.turn() ─► TTS (Piper) ─► speaker

Everything runs on the device. No audio, transcript, or frame leaves the Pi
(assuming the Orchestrator's LocalBackend). Audio capture/playback use ALSA
(`arecord`/`aplay`) so there are no Python audio deps to fight on a Pi.

Camera is OPTIONAL, OFF by default, and — when on — computes only a presence
boolean on-device. Frames are never saved, never transmitted, never shown. It
exists to let the companion notice you sat down / stepped away, not to watch you.
This is consistent with the project's anti-surveillance stance.

Binaries you provide (all local, all free):
  * whisper.cpp   → transcription   (env WHISPER_BIN, WHISPER_MODEL)
  * piper         → speech          (env PIPER_BIN, PIPER_VOICE)
  * arecord/aplay → ALSA (preinstalled on Raspberry Pi OS)
"""

from __future__ import annotations
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SAMPLE_RATE = 16000          # whisper wants 16 kHz mono
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2   # int16


# ══════════════════════════════════════════════════════════════════════════
# Audio helpers (pure Python — no numpy/audioop)
# ══════════════════════════════════════════════════════════════════════════
def rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    a = array("h")
    a.frombytes(frame[: len(frame) // 2 * 2])
    if not a:
        return 0.0
    return (sum(s * s for s in a) / len(a)) ** 0.5


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════
# Microphone — energy-gated capture (endpoints on trailing silence)
# ══════════════════════════════════════════════════════════════════════════
class Mic:
    def __init__(self, device: str | None = None, threshold: float = 500.0,
                 silence_ms: int = 1500, max_ms: int = 30000, min_ms: int = 300):
        self.device = device
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.max_ms = max_ms
        self.min_ms = min_ms

    def _capture(self):
        """OS-aware raw-PCM capture to stdout (S16_LE, 16 kHz, mono) — the Python
        VAD below is identical on every platform; only the source binary differs.
          * Linux  → arecord (ALSA), honoring --mic-device (e.g. plughw:2,0)
          * macOS  → sox from the SYSTEM DEFAULT input device (set it in
            System Settings → Sound → Input). `brew install sox`. On macOS the
            --mic-device flag is ignored; picking the mic = choosing the default."""
        if sys.platform == "darwin":
            cmd = ["sox", "-q", "-d", "-t", "raw", "-r", str(SAMPLE_RATE),
                   "-c", "1", "-b", "16", "-e", "signed-integer", "-L", "-"]
        else:
            cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                   "-c", "1", "-t", "raw"]
            if self.device:
                cmd += ["-D", self.device]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def record_until_silence(self) -> bytes:
        """Stream from the mic; start on speech, stop after trailing silence.
        Returns 16 kHz mono WAV bytes ready for whisper."""
        proc = self._capture()
        frames, started, silence = [], False, 0
        speech_ms = 0
        try:
            while True:
                frame = proc.stdout.read(FRAME_BYTES)
                if len(frame) < FRAME_BYTES:
                    break
                loud = rms(frame) > self.threshold
                if loud:
                    started, silence = True, 0
                    speech_ms += FRAME_MS
                    frames.append(frame)
                elif started:
                    silence += FRAME_MS
                    frames.append(frame)
                    if silence >= self.silence_ms and speech_ms >= self.min_ms:
                        break
                if started and len(frames) * FRAME_MS >= self.max_ms:
                    break
        finally:
            proc.terminate()
            proc.wait()
        return pcm_to_wav(b"".join(frames))


# ══════════════════════════════════════════════════════════════════════════
# ASR / TTS — shell out to local binaries
# ══════════════════════════════════════════════════════════════════════════
class ASR(Protocol):
    def transcribe(self, wav_bytes: bytes) -> str: ...


class TTS(Protocol):
    def say(self, text: str) -> None: ...


class Body(Protocol):
    """Optional expression layer (e.g. Reachy Mini's head/antennas). Pure
    presentation — it reacts to conversational state and NEVER gates safety or
    logic. All hooks must be best-effort and swallow their own errors so a motor
    glitch can never break the conversation."""
    def on_listen(self) -> None: ...   # user is speaking / mic open
    def on_think(self) -> None: ...     # generating a reply
    def on_speak(self) -> None: ...     # about to speak
    def on_idle(self) -> None: ...      # waiting
    def close(self) -> None: ...


class NullBody:
    """Default no-op body — voice works with no robot attached."""
    def on_listen(self): pass
    def on_think(self): pass
    def on_speak(self): pass
    def on_idle(self): pass
    def close(self): pass


def _strip_nonspeech(text: str) -> str:
    """whisper labels NON-SPEECH as bracketed annotations — "(wind blowing)",
    "[BLANK_AUDIO]", "*sighs*". Strip them so the companion ignores ambient noise;
    if nothing lexical remains, the caller gets "" and skips the turn."""
    for pat in (r"\([^)]*\)", r"\[[^\]]*\]", r"\*[^*]*\*"):
        text = re.sub(pat, "", text)
    return re.sub(r"\s+", " ", text).strip()


class WhisperCppASR:
    """Local ASR — runs whisper.cpp on this machine (the Pi)."""

    def __init__(self, binary: str | None = None, model: str | None = None):
        self.binary = binary or os.environ.get("WHISPER_BIN", "whisper-cli")
        self.model = model or os.environ.get("WHISPER_MODEL", "models/ggml-base.en.bin")

    def transcribe(self, wav_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(wav_bytes)
            f.flush()
            out = subprocess.run(
                [self.binary, "-m", self.model, "-f", f.name, "-nt", "-l", "en"],
                capture_output=True, text=True, check=True,
            ).stdout
        text = " ".join(line.strip() for line in out.splitlines() if line.strip())
        return _strip_nonspeech(text)


class RemoteWhisperASR:
    """Networked ASR — POST audio to a whisper.cpp `whisper-server` on another box
    (e.g. the Studio). Moves *hearing* off the Pi to a fast machine, just like the
    LLM, so a bigger/better model handles atypical registers (excited dog-voice!)
    fast. Stdlib multipart POST — no extra deps on the Pi.

    Server: whisper-server -m ggml-large-v3-turbo.bin --host 0.0.0.0 --port 8080
    """

    _BOUNDARY = "----companionAudio7f3a2b1c"

    def __init__(self, url: str | None = None):
        self.url = (url or os.environ.get("WHISPER_REMOTE_URL",
                    "http://localhost:8080")).rstrip("/")

    def transcribe(self, wav_bytes: bytes) -> str:
        b = self._BOUNDARY
        parts = [
            f'--{b}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
            wav_bytes, b"\r\n",
        ]
        for k, v in (("response_format", "json"), ("temperature", "0.0")):
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"'
                         f'\r\n\r\n{v}\r\n'.encode())
        parts.append(f'--{b}--\r\n'.encode())
        req = urllib.request.Request(
            f"{self.url}/inference", data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode(errors="replace")
        try:
            text = json.loads(raw).get("text", "")
        except (ValueError, AttributeError):
            text = raw          # server returned plain text
        return _strip_nonspeech(text)


class PiperTTS:
    def __init__(self, binary: str | None = None, voice: str | None = None,
                 rate: int = 22050, aplay_device: str | None = None,
                 lead_silence_ms: int = 700):
        self.binary = binary or os.environ.get("PIPER_BIN", "piper")
        self.voice = voice or os.environ.get("PIPER_VOICE", "voices/en_US-amy-medium.onnx")
        self.rate = rate
        self.aplay_device = aplay_device
        # HDMI/TV audio inputs sleep during silence and take ~1s to re-sync, which
        # clips the first words. A lead of silence absorbs that wake-up instead.
        self.lead_silence_ms = lead_silence_ms

    @staticmethod
    def _prepend_silence(path: str, ms: int) -> None:
        with wave.open(path, "rb") as w:
            p = w.getparams()
            frames = w.readframes(w.getnframes())
        n = int(p.framerate * ms / 1000)
        silence = b"\x00" * (n * p.sampwidth * p.nchannels)
        with wave.open(path, "wb") as w:
            w.setparams(p)
            w.writeframes(silence + frames)

    def say(self, text: str) -> None:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run([self.binary, "-m", self.voice, "-f", path],
                           input=text, text=True, check=True)
            if self.lead_silence_ms:
                self._prepend_silence(path, self.lead_silence_ms)
            if sys.platform == "darwin":
                cmd = ["afplay", path]        # built-in; routes to default output
            else:
                cmd = ["aplay", "-q"] + (["-D", self.aplay_device] if self.aplay_device else [])
                cmd += [path]
            subprocess.run(cmd, check=True)
        finally:
            os.unlink(path)


class MacSayTTS:
    """Zero-setup macOS TTS via the built-in `say`. No piper binary, no voice
    model — routes to the system DEFAULT output device (set it to the room
    speaker). NOTE: a different voice from Piper, so a room on `say` won't match
    the bedroom PiCar's voice — install piper on the Macs too if you want ONE
    consistent voice everywhere (then this fallback isn't used)."""

    def __init__(self, voice: str | None = None, rate: int | None = None):
        self.voice = voice or os.environ.get("SAY_VOICE")     # e.g. "Samantha"
        self.rate = rate or (int(os.environ["SAY_RATE"]) if os.environ.get("SAY_RATE") else None)

    def say(self, text: str) -> None:
        cmd = ["say"]
        if self.voice:
            cmd += ["-v", self.voice]
        if self.rate:
            cmd += ["-r", str(self.rate)]
        subprocess.run(cmd + [text], check=True)


def _piper_ready(voice: str | None = None, binary: str | None = None) -> bool:
    b = binary or os.environ.get("PIPER_BIN", "piper")
    v = voice or os.environ.get("PIPER_VOICE", "voices/en_US-amy-medium.onnx")
    return bool(shutil.which(b)) and Path(v).exists()


def make_tts(kind: str = "auto", aplay_device: str | None = None,
             lead_silence_ms: int = 700, voice: str | None = None):
    """Pick a TTS: 'piper', 'say', or 'auto' (piper if it's set up here, else
    macOS `say`, else piper — which errors loudly if missing on Linux, as before)."""
    kind = (kind or "auto").lower()
    if kind == "say":
        return MacSayTTS()
    if kind == "piper" or _piper_ready(voice):
        return PiperTTS(voice=voice, aplay_device=aplay_device, lead_silence_ms=lead_silence_ms)
    if sys.platform == "darwin":
        return MacSayTTS()
    return PiperTTS(voice=voice, aplay_device=aplay_device, lead_silence_ms=lead_silence_ms)


# ══════════════════════════════════════════════════════════════════════════
# Camera — presence ONLY, on-device, no frames retained, OFF by default
# ══════════════════════════════════════════════════════════════════════════
class PresenceCamera:
    """Returns a presence boolean computed on-device. It NEVER writes, shows, or
    transmits a frame — a single frame is grabbed, reduced to yes/no, discarded.
    Purpose: let the companion notice arrival/departure, not surveil. Requires
    OpenCV + a Haar cascade; if unavailable, presence is always False (fail-open
    to voice-only)."""

    def __init__(self, enabled: bool = False, index: int = 0):
        self.enabled = enabled
        self._cap = None
        self._cascade = None
        if enabled:
            try:
                import cv2
                self._cv2 = cv2
                self._cap = cv2.VideoCapture(index)
                self._cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            except Exception:
                self.enabled = False

    def present(self) -> bool:
        if not self.enabled or self._cap is None:
            return False
        ok, frame = self._cap.read()
        if not ok:
            return False
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5)
        # frame + gray go out of scope here; nothing is stored.
        return len(faces) > 0

    def close(self):
        if self._cap is not None:
            self._cap.release()


# ══════════════════════════════════════════════════════════════════════════
# Voice loop
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class VoiceLoop:
    orchestrator: object          # companion.orchestrator.Orchestrator
    asr: ASR
    tts: TTS
    mic: Mic
    camera: "PresenceCamera | None" = None
    body: "Body | None" = None    # optional expression layer (e.g. Reachy Mini)

    def run(self):
        say = self.tts.say
        body = self.body or NullBody()
        name = getattr(self.orchestrator.persona, "name", "Companion")
        print(f"[{name}] voice loop — local. Ctrl-C to stop.")
        try:
            while True:
                # optional presence gate: wait until someone's actually there
                if self.camera and self.camera.enabled and not self.camera.present():
                    continue
                body.on_listen()
                wav = self.mic.record_until_silence()
                text = self.asr.transcribe(wav).strip()
                if not text:
                    body.on_idle()
                    continue
                print(f"you › {text}")
                body.on_think()
                res = self.orchestrator.turn(text)
                print(f"{name} › {res.reply}")
                body.on_speak()
                say(res.reply)
                body.on_idle()
        except KeyboardInterrupt:
            print("\ntake care.")
        finally:
            if self.camera:
                self.camera.close()
            body.close()
