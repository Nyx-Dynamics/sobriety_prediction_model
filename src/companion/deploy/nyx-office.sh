#!/bin/bash
# Nyx office node launcher (Studio / macOS) — runs the satellite as the office room,
# talking to the local brain. Invoked by launchd (com.nyx.office). Audio devices come
# from System Settings → Sound (Input = office mic, Output = S9).
export PYTHONUNBUFFERED=1
export COMPANION_BRAIN_URL="${COMPANION_BRAIN_URL:-http://localhost:9000}"
cd "$HOME/sobriety_prediction_model/src" || exit 1
exec "$HOME/sobriety_prediction_model/.venv/bin/python" -m companion.satellite --node office
