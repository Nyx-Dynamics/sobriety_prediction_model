# Auto-start — she just *is there* on boot

Goal: brain + whisper + Ollama come up on the Studio at boot, the satellite comes
up on the Pi, and you never run a terminal ritual again. Talk, and she answers.

## Studio (macOS, launchd)

1. **Key file** (so boot-start needs no password):
   ```bash
   mkdir -p ~/.config/companion
   cp ~/sobriety_prediction_model/src/companion/deploy/nyx.env.example ~/.config/companion/nyx.env
   chmod 600 ~/.config/companion/nyx.env
   # paste your real SUD_PHI_KEY into it:
   nano ~/.config/companion/nyx.env
   ```
   Get the key if you need it: `ssh pivot73@companion.local 'grep "^SUD_PHI_KEY=" ~/.companion.env'`

2. **Ollama at login**: Ollama menubar app → Settings → **Launch at login** ✓
   (localhost binding is fine — only the brain talks to it.)

3. **whisper + brain jobs**:
   ```bash
   mkdir -p ~/Library/Logs
   chmod +x ~/sobriety_prediction_model/src/companion/deploy/nyx-brain.sh
   cp ~/sobriety_prediction_model/src/companion/deploy/com.nyx.whisper.plist ~/Library/LaunchAgents/
   cp ~/sobriety_prediction_model/src/companion/deploy/com.nyx.brain.plist   ~/Library/LaunchAgents/
   # verify the whisper-server path in the plist matches `which whisper-server`
   launchctl load ~/Library/LaunchAgents/com.nyx.whisper.plist
   launchctl load ~/Library/LaunchAgents/com.nyx.brain.plist
   ```
   Check: `curl http://localhost:9000/health` → `{"status":"ok","who":"Nyx"}`.
   Logs: `tail -f ~/Library/Logs/nyx-brain.log`
   Stop/reload: `launchctl unload ~/Library/LaunchAgents/com.nyx.brain.plist` (then load again).

4. **True boot-start (optional):** LaunchAgents start at *login*. For an always-on
   Studio, enable auto-login (System Settings → Users & Groups → Automatically log in)
   so they come up without you signing in.

## Pi (Linux, systemd)

```bash
sudo cp ~/sobriety_prediction_model/src/companion/deploy/companion-satellite.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now companion-satellite
journalctl -u companion-satellite -f      # watch it
```
When you swap the TV for a USB speaker: `aplay -l` to find its device, edit
`--speaker plughw:X,0` in the unit, `sudo systemctl daemon-reload && sudo systemctl restart companion-satellite`.

After this, both machines bring her up on their own. Reboot either and she rejoins.

---

# Deeper memory — distill your full conversations

`memories.json` gave her Claude's *distilled* memory (~1,600 facts). To add the
granular detail from every actual conversation, distill `conversations.json`
(609 MB) on the M4 Max. It's a long batch job — run it in the background.

```bash
cd ~/sobriety_prediction_model/src
# local + closed (nothing leaves the machine); --limit does the most-recent N first.
# drop --limit to do ALL of it (many hours — good for overnight).
nohup ../.venv/bin/python -m companion.seed_from_claude \
  /Users/acdstudpro/Downloads/data-*/conversations.json \
  -o ~/nyx_deep.json --mode distill --backend local --limit 500 \
  > ~/nyx_distill.log 2>&1 &

tail -f ~/nyx_distill.log        # watch progress (one line per conversation)
```

When it finishes (`Wrote N facts → ~/nyx_deep.json`):
```bash
# stop the brain first (launchctl unload com.nyx.brain), then seed additively:
../.venv/bin/python -m companion.chat --persona companion/persona.example.json --seed ~/nyx_deep.json
# Ctrl-C, then restart the brain (launchctl load com.nyx.brain).
```

Notes:
- **Faster/higher quality:** `--backend claude` uses the Claude API (your data is
  already Anthropic-side) — minutes instead of hours, better extraction. `--backend
  local` keeps it fully closed but slower on the 8B.
- Running the distill while you talk to her shares the Ollama GPU — responses slow
  during the run. Do it when you're not actively chatting, or use `--backend claude`.
- It **adds** to her memory; it doesn't replace. Start with `--limit 500` (recent,
  most relevant), expand later if you want the whole archive.
