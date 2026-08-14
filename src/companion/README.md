# Companion — a private, closed-system AI companion

The conversation brain that turns the prediction/privacy stack into an actual
companion. Built for a single owner, encrypted, safety-gated, and
**model-agnostic** so it can run with nothing leaving your machine.

```
you ─► SafetyChecker (crisis pre-check) ─► Orchestrator
                                              │  assembles: Persona + recalled Memory + history + directive
                                              ▼
                                          LLMBackend  ── Local (closed)  or  Claude (capable)
                                              │
                                          SafetyChecker (reply post-check)
                                              │
                                          MemoryStore (encrypted write-back)
                                              ▼
                                            reply
```

## Files
| file | role |
|---|---|
| `orchestrator.py` | the turn loop: safety → assemble → generate → safety → remember |
| `chat.py` | local terminal REPL (`python -m companion.chat`) |
| `distiller.py` | conversation → the 40-feature contract (feeds the risk model) |
| `persona.example.json` | who the companion is + your seeded "about me" |

## The privacy fork — read this before choosing a backend

This is the one real decision, and it's yours:

| | **LocalBackend** (default) | **ClaudeBackend** |
|---|---|---|
| Where the model runs | Your hardware (Ollama / LM Studio / llama.cpp) | Anthropic's API |
| Does your text leave the machine? | **No** | Yes (subject to Anthropic retention; ZDR possible) |
| Capability | Bounded by your local model | State-of-the-art |
| Right choice when | "thoughts I don't want beyond the LLM that knows me best" | capability > full locality |

For a truly closed system — the reason you're building this yourself — use
`LocalBackend`. Memory is always encrypted at rest regardless of backend, and
the store **refuses to run without a key** (`SUD_PHI_KEY`). Lose the key and the
memory is unreadable; that is the design.

## Run

```bash
export SUD_PHI_KEY=$(python -c "from serving.privacy import Encryptor; print(Encryptor.generate_key())")
ollama serve &                                   # local model, closed system
python -m companion.chat --persona companion/persona.example.json
# seed it from your own exported data:
python -m companion.chat --persona ... --seed my_export.json   # [{"text": "...", "importance": 0.7}, ...]
```

## Safety is not optional
A deterministic crisis check runs on every user turn **before** the model sees
it and on every reply after. On a hit it short-circuits to 988 / human routing —
the companion never tries to be the safety net alone. The regex layer is a
**floor**; add a model-based classifier before relying on it. If you are ever in
danger: 911, or call/text **988** (US).

## Embodiment — Raspberry Pi voice loop (`voice.py`, `voice_chat.py`)
The orchestrator is embodiment-agnostic: `turn()` takes text in, returns text
out. The Pi voice layer wraps it, fully local:

```
mic ─(arecord/VAD)─► whisper.cpp ─► orchestrator.turn() ─► Piper ─► aplay ─► speaker
```

- **ASR/TTS are local binaries** (whisper.cpp + Piper) — no audio leaves the Pi.
- **Mic** uses ALSA `arecord` with energy-based endpointing (starts on speech,
  stops on trailing silence) — no Python audio deps to fight on a Pi.
- Run: `python -m companion.voice_chat --persona companion/persona.example.json`
  (see `voice_chat.py` header for the one-time whisper.cpp / Piper / model setup).

### Reachy Mini body (`reachy.py`)
Reachy Mini Wireless (Pollen Robotics / Hugging Face) has a **Raspberry Pi CM4
onboard** — so the whole local companion runs on the robot, offline. Apache-2.0
+ Python SDK, camera + 4 mics + speaker, and an expressive head/antennas.
`ReachyBody` maps conversational state → calm head poses (lean in to listen,
tilt to think, face forward to speak); it's a pure presentation layer that
never gates safety or logic, and is **inert off-robot** (guarded SDK import).
Enable with `--reachy`. Compute note: the CM4 runs ASR/TTS/orchestrator/motion
fine but not a large LLM — either use a tiny local model or point
`COMPANION_LOCAL_URL` at an LLM box on your LAN (still closed — never leaves
your network).

### Run on boot — Pi 5 (`deploy/companion.service`)
A Pi 5 runs the whole companion standalone (it's faster than the CM4 Reachy
ships with). Install the systemd unit + a 0600 EnvironmentFile so it comes up on
boot and is just *there*:
```bash
sudo install -Dm600 deploy/companion.env.example /etc/companion/companion.env
sudoedit /etc/companion/companion.env     # SUD_PHI_KEY + whisper/piper/model paths
sudo cp deploy/companion.service /etc/systemd/system/
sudo systemctl enable --now companion.service
```
**Model pick for a Pi 5 (CPU):** `llama3.2:3b` / `qwen2.5:3b` is the sweet spot —
responsive and conversational. `llama3.1:8b` runs but is slow (~1–2 tok/s); use
it only if you keep the LLM on a stronger LAN box (set `COMPANION_LOCAL_URL`).
whisper.cpp `base.en` and Piper both run comfortably on a Pi 5.

**Later, when Reachy arrives:** keep the Pi 5 as the LAN model box and point
Reachy's CM4 at it (`COMPANION_LOCAL_URL=http://<pi5>:11434/v1`). The CM4 becomes
the body (audio + motion); the Pi 5 stays the brain. No code changes.

### Camera stance (`PresenceCamera`)
The camera is **OFF by default**. When enabled (`--camera`), it computes only a
**presence boolean on-device** — a frame is grabbed, reduced to yes/no, and
discarded. Frames are **never saved, shown, or transmitted**. Its only job is to
let the companion notice you arrived or stepped away — not to watch you. This is
the same anti-surveillance line the rest of the system holds.

## Not yet wired (honest gaps)
- Model-extracted memory (currently a keyword salience heuristic — upgrade with
  the `distiller` pattern for higher-quality facts).
- Semantic recall (currently recency×salience×overlap — add embeddings later).
- The risk/escalation hook: pass the `directive` from `/observe` into `turn()`
  to let the risk model shape the companion's care posture.
- A second-layer crisis classifier beyond the regex floor.
