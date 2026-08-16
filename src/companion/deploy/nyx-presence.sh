#!/bin/bash
# Nyx presence daemon launcher (Studio) — UniFi Protect -> ephemeral presence.
# Reads UNIFI_* / OWNER_NAME / PRESENCE_* from the shared env file.
set -a
source "$HOME/.config/companion/nyx.env"
set +a
export PYTHONUNBUFFERED=1        # flush logs live so `tail` isn't misleadingly empty
cd "$HOME/sobriety_prediction_model/src" || exit 1
exec "$HOME/sobriety_prediction_model/.venv/bin/python" -m companion.presence
