#!/bin/bash
# Nyx reference search daemon launcher (Studio) — serves /search over the encrypted
# document index. Reads SUD_PHI_KEY (to decrypt chunks) + REFERENCE_* from the env.
set -a
source "$HOME/.config/companion/nyx.env"
set +a
cd "$HOME/sobriety_prediction_model/src" || exit 1
exec "$HOME/sobriety_prediction_model/.venv/bin/python" -m companion.reference serve
