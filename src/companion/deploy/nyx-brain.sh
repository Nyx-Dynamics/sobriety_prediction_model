#!/bin/bash
# Nyx brain launcher (Studio / macOS) — sets env, then runs the brain server.
# Invoked by the launchd job com.nyx.brain so the key never lives in the plist.
set -a
source "$HOME/.config/companion/nyx.env"    # SUD_PHI_KEY + the three URLs, chmod 600
set +a
export PYTHONUNBUFFERED=1        # flush logs live so `tail` isn't misleadingly empty
cd "$HOME/sobriety_prediction_model/src" || exit 1
exec "$HOME/sobriety_prediction_model/.venv/bin/python" \
  -m companion.brain_server --persona companion/persona.example.json
