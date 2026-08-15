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
