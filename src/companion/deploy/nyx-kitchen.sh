#!/bin/bash
# Nyx kitchen node launcher (Mini / macOS) — runs the satellite as the kitchen room,
# talking to the Studio brain over the LAN. Invoked by launchd (com.nyx.kitchen).
# Audio devices come from System Settings → Sound (Input = Blue Snowball, Output = Sony).
# To encrypt the hop later, add: export COMPANION_LINK_PASSPHRASE="<same phrase as the brain>"
export PYTHONUNBUFFERED=1
export PATH="/opt/homebrew/bin:$PATH"     # launchd's minimal PATH can't find sox otherwise
export COMPANION_BRAIN_URL="${COMPANION_BRAIN_URL:-http://192.168.1.59:9000}"
cd "$HOME/sobriety_prediction_model/src" || exit 1
exec "$HOME/sobriety_prediction_model/.venv/bin/python" -m companion.satellite --node kitchen
