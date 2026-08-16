# Whole-house — multi-room voice + presence

Two capabilities. **Multi-room already works** (no new code); **presence** is the
new piece (UniFi Protect). Everything stays on your LAN.

## 1. Multi-room voice (works today)

The brain is one central service; the satellite already takes `--node <room>` and
the wake-word engagement is per-room. So **adding a room = one more satellite**
pointed at the same brain. On a second Pi/node:

```bash
python -m companion.satellite --brain http://192.168.1.59:9000 --node livingroom \
  --mic-device plughw:2,0 --speaker plughw:1,0 --threshold 1500 --silence-ms 2000
```
Or make it boot-managed: copy `companion-satellite.service`, change `--node` and the
audio devices, `systemctl enable --now`. Each room shares her one memory; replies
go back to the room that called her; "Hey Nyx" wakes only that room.

## 2. Presence (UniFi Protect) — she knows who's home / which room

Optional. When on, she gets an ambient line ("They're in the kitchen right now")
folded into her context *only when you talk to her* — so she's situated, not
surveilling. **The brain works fine without this** — if the daemon isn't running,
she just has no location awareness.

**Privacy, enforced in code (`presence.py`):** RAM-only + TTL'd (no history, no
who-was-where log), **owner-vs-other collapse** (family/guests are reduced to
"someone" *before* anything reaches Nyx — she can't reason about or remember other
identified people), local-only.

### Setup (Studio)
```bash
# 1. dep
~/sobriety_prediction_model/.venv/bin/pip install uiprotect

# 2. map your cameras -> rooms
cp ~/sobriety_prediction_model/src/companion/deploy/zones.example.json ~/.config/companion/zones.json
nano ~/.config/companion/zones.json     # use YOUR Protect camera names

# 3. add UniFi creds to the shared env (already sourced by the daemons)
nano ~/.config/companion/nyx.env        # fill UNIFI_HOST/USER/PASS, OWNER_NAME
#    (use a dedicated read-only Protect local account, not your main admin login)

# 4. install the launchd job
chmod +x ~/sobriety_prediction_model/src/companion/deploy/nyx-presence.sh
cp ~/sobriety_prediction_model/src/companion/deploy/com.nyx.presence.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nyx.presence.plist

# 5. check it
curl http://localhost:9100/presence     # {"owner_zone": ..., "others_present": ..., "zones": {...}}
tail -f ~/Library/Logs/nyx-presence.log
```

Then walk past a mapped camera and `curl :9100/presence` should show your room.
Talk to her and she can reference where you are.

**Untested against live Protect from the build side** — the WebSocket event shape
varies by Protect version, so `_on_ws()` in `presence.py` may need a tweak for how
your version names the camera/identity fields. If `/presence` stays empty while you
move around, grab a raw event and we'll map the fields to your firmware.

### The next depth (not built yet)
- **Proactive greetings** — she speaks when you *arrive* (presence event → push a
  line to that room's satellite), not just when addressed. Needs a small
  push-to-satellite channel.
- **Close the risk loop** — presence/routine signals → the risk model → a *care
  posture* directive (same `directive` channel presence uses now), behind the
  governance wall so it shapes warmth without becoming monitoring. See
  `docs/household_agent.md`.

---

# Phase 2 — Reference index over your documents (`reference.py`)

She *looks things up* in your files on request — never bakes them into memory.
Retrieved excerpts go into the per-turn prompt only; she cites the source file
and says so if she can't find it. Chunk text is **encrypted at rest** (SUD_PHI_KEY).

### Build the index (Studio, one-time; background — like the distill)
```bash
~/sobriety_prediction_model/.venv/bin/pip install pypdf python-docx numpy
ollama pull nomic-embed-text                      # local embedder, nothing leaves the box
cd ~/sobriety_prediction_model/src
set -a; source ~/.config/companion/nyx.env; set +a   # for SUD_PHI_KEY
# point it at your extracted document roots (skip media/code automatically):
nohup ../.venv/bin/python -m companion.reference build \
  ~/Documents "/path/to/OrganizedDocuments" "/path/to/personal 2" \
  > ~/nyx_reference.log 2>&1 &
tail -f ~/nyx_reference.log     # extract -> embed, checkpointed
```

### Serve it (so the brain can query it)
```bash
chmod +x ~/sobriety_prediction_model/src/companion/deploy/nyx-reference.sh
cp ~/sobriety_prediction_model/src/companion/deploy/com.nyx.reference.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nyx.reference.plist
curl "http://localhost:9200/search?q=lease%20rent"     # sanity check
```

Then ask her out loud: *"Nyx, look up what my April filing said about X"* — the
brain detects the lookup, pulls the top excerpts from the encrypted index, and she
answers citing the file. Ambient/normal conversation never triggers a search.

Notes:
- Embedding runs on the local Ollama (shares the GPU) — build when you're not
  actively chatting. ~8,200 PDFs is a long job; it checkpoints.
- Scanned/image-only PDFs won't extract text without OCR (not built yet — a later
  add: run them through a local OCR pass before indexing).

---

# Embodiment — PiCar-X body (`picar_body.py`)

Her mobile body. The PiCar-X's Pi runs the **satellite** (ears+mouth to the Studio
brain) *plus* the `PiCarBody` expression layer — she looks at you, freezes to
listen, and nods when she speaks. All thinking still happens on the Studio; the car
is a body we puppet, never a SunFounder cloud login.

### One-time (on the car's Pi)
```bash
# 1. Assemble per SunFounder's tutorials. CRITICAL: zero the servos (center at 0)
#    BEFORE attaching the arms/head, or the angles will be off.
# 2. Install ONLY the hardware library (ignore their AI app entirely):
cd ~ && git clone https://github.com/sunfounder/picar-x.git -b v2.0
cd picar-x && sudo python3 setup.py install       # pulls robot-hat
#    (follow SunFounder's i2c/audio enable steps in their docs)
# 3. Sanity-test the body with no brain, watch the gestures:
cd ~/sobriety_prediction_model/src
PICAR_DEBUG=1 python3 -c "from companion.picar_body import PiCarBody; \
import time; b=PiCarBody(debug=True); \
[ (h(), time.sleep(0.6)) for h in (b.on_listen,b.on_think,b.on_speak,b.on_idle) ]; b.close()"
```

### Run her with the body
```bash
export COMPANION_BRAIN_URL=http://192.168.1.59:9000
python3 -m companion.satellite --node rover --body picar \
  --mic-device plughw:2,0 --speaker plughw:1,0 --silence-ms 2000
```
Same satellite as any room — `--body picar` just adds the head expression. Wake her
with "Hey Nyx"; she perks and nods, freezes while you talk, settles when you're done.

**By design she does NOT drive around** — the head/camera is the expressive face;
autonomous approach is gated on obstacle-safe presence/vision we haven't built.
`approach()`/`orient()` are stubbed for that day. The onboard mic is mono (no
sound-direction), so she faces forward attentively rather than turning toward your
voice — a mic array or presence bearing is what would unlock true orient-to-voice.

---

# macOS room nodes (office = Studio, kitchen = Mini) + encrypted hop

A Mac can *be* a room node — captures and speaks locally, ships audio to the brain
over an encrypted LAN hop. Same `satellite`, macOS backends auto-selected (`sox`
capture, `afplay`/`say` playback).

### Per-Mac setup
```bash
brew install sox                     # mic capture (playback is built-in afplay/say)
# System Settings → Sound: set Input = room mic, Output = room speaker
```

### Encrypted hop (do once, shared value everywhere)
```bash
python3 -c "from companion.link import LinkCipher; print(LinkCipher.generate_key())"
#  put the SAME value in the brain's env AND every node's env, then restart the brain:
export COMPANION_LINK_KEY=<that-value>
```
Node→brain voice is then Fernet-encrypted on the wire; the reply comes back
encrypted. A bare Pi with no key still works (plaintext) against the same brain.

### Run
```bash
# office — on the Studio itself, brain is localhost (no network, her fastest room)
COMPANION_BRAIN_URL=http://localhost:9000 python3 -m companion.satellite --node office

# kitchen — on the Mini, brain over ethernet
COMPANION_BRAIN_URL=http://192.168.1.59:9000 python3 -m companion.satellite --node kitchen
```

Voice defaults to macOS `say` (zero-setup, but a *different* voice than the PiCar's
Piper). For ONE voice in every room, install piper + the same `.onnx` on the Macs —
`--tts auto` picks it up automatically. `--lead-silence-ms` defaults to 0 on macOS
(no HDMI wake-up clip over Thunderbolt/USB).

---

# Kitchen Mini — from a bare machine (quickstart)

The Mini is a *dumb node*: it captures audio, ships it to the Studio brain over an
encrypted Ethernet hop, and speaks the reply. The only pip dep a node needs is
`cryptography` (for the hop) — no numpy/torch/model.

### One-time on the Mini
```bash
# Homebrew (if not already there), then sox for mic capture:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install sox

# the code + a tiny venv:
git clone https://github.com/Nyx-Dynamics/sobriety_prediction_model.git ~/sobriety_prediction_model
cd ~/sobriety_prediction_model
python3 -m venv .venv
.venv/bin/pip install cryptography

# System Settings → Sound:  Input = kitchen USB mic,  Output = kitchen speaker
```

### Encrypted hop (do the key on the STUDIO, share the value)
```bash
# on the Studio — generate once:
~/sobriety_prediction_model/.venv/bin/python -c \
  "from companion.link import LinkCipher; print(LinkCipher.generate_key())"
# add COMPANION_LINK_KEY=<that value> to the Studio's ~/.config/companion/nyx.env,
# then restart the brain so it can decrypt:
launchctl kickstart -k gui/$(id -u)/com.nyx.brain
```

### Run the kitchen node (on the Mini, brain over Ethernet)
```bash
cd ~/sobriety_prediction_model/src
export COMPANION_LINK_KEY=<same value as the brain>
COMPANION_BRAIN_URL=http://192.168.1.59:9000 \
  ../.venv/bin/python -m companion.satellite --node kitchen
```
Startup line should read `link=encrypted`. Say "Hey Nyx" and she answers through the
kitchen speaker — voice over the wire is now ciphertext. (macOS `say` voice until you
put piper on the Mini too.) Once it works, we make it boot-managed with a launchd job.
