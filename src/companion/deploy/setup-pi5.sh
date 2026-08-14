#!/usr/bin/env bash
# ============================================================================
# setup-pi5.sh — bare Raspberry Pi OS -> local voice companion, one run.
# ----------------------------------------------------------------------------
# Installs (all local, closed system):
#   * Ollama + a local LLM            (the brain)
#   * whisper.cpp + base.en           (local speech-to-text)
#   * Piper + a voice                 (local text-to-speech)
#   * Python venv + cryptography      (fail-closed encrypted memory)
#   * a chmod-600 ~/.companion.env with a freshly generated encryption key
#
# Run AFTER you've rsync'd the repo to the Pi (see the chat instructions).
# Idempotent-ish: safe to re-run; it will NOT regenerate your memory key.
# Expect 15-30 min the first time (downloads + a whisper.cpp build).
# ============================================================================
set -euo pipefail

REPO="${REPO:-$HOME/sobriety_prediction_model}"
SRC="$REPO/src"
VENV="$REPO/.venv"
ENVFILE="$HOME/.companion.env"
MODEL="${COMPANION_LOCAL_MODEL:-llama3.2:3b}"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

[ -d "$SRC/companion" ] || { echo "Repo not found at $REPO — rsync it first."; exit 1; }

log "1/7  System packages"
sudo apt-get update
sudo apt-get -y install git build-essential cmake wget curl \
    python3-venv python3-pip alsa-utils libopenblas-dev

log "2/7  Python venv + encrypted-memory dep"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install cryptography

log "3/7  Ollama + local model ($MODEL)  [big download, be patient]"
command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh
sleep 2
ollama pull "$MODEL"

log "4/7  whisper.cpp (local ASR)"
if [ ! -d "$HOME/whisper.cpp" ]; then
  git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$HOME/whisper.cpp"
fi
cd "$HOME/whisper.cpp"
cmake -B build >/dev/null
cmake --build build -j"$(nproc)" --config Release
[ -f models/ggml-base.en.bin ] || bash ./models/download-ggml-model.sh base.en
WHISPER_BIN="$HOME/whisper.cpp/build/bin/whisper-cli"
[ -x "$WHISPER_BIN" ] || WHISPER_BIN="$HOME/whisper.cpp/main"   # older layout fallback

log "5/7  Piper (local TTS) + a voice"
if [ ! -x "$HOME/piper/piper" ]; then
  wget -qO /tmp/piper.tgz \
    https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
  tar -xzf /tmp/piper.tgz -C "$HOME"      # extracts $HOME/piper/
fi
mkdir -p "$HOME/voices"
VBASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
[ -f "$HOME/voices/en_US-amy-medium.onnx" ] || \
  wget -qO "$HOME/voices/en_US-amy-medium.onnx" "$VBASE/en_US-amy-medium.onnx?download=true"
[ -f "$HOME/voices/en_US-amy-medium.onnx.json" ] || \
  wget -qO "$HOME/voices/en_US-amy-medium.onnx.json" "$VBASE/en_US-amy-medium.onnx.json?download=true"

log "6/7  Environment file (encrypted-memory key lives here)"
if [ -f "$ENVFILE" ] && grep -q SUD_PHI_KEY "$ENVFILE"; then
  echo "Keeping existing $ENVFILE (memory key preserved)."
else
  KEY="$("$VENV/bin/python" -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
  umask 077
  cat > "$ENVFILE" <<EOF
# Companion environment — 0600. BACK UP SUD_PHI_KEY; it decrypts your memory.
SUD_PHI_KEY=$KEY
WHISPER_BIN=$WHISPER_BIN
WHISPER_MODEL=$HOME/whisper.cpp/models/ggml-base.en.bin
PIPER_BIN=$HOME/piper/piper
PIPER_VOICE=$HOME/voices/en_US-amy-medium.onnx
COMPANION_LOCAL_URL=http://localhost:11434/v1
COMPANION_LOCAL_MODEL=$MODEL
EOF
  chmod 600 "$ENVFILE"
  echo "Wrote $ENVFILE (new memory key generated — back it up!)."
fi

log "7/7  Audio devices detected on this Pi"
echo "-- capture (mic) --";   arecord -l 2>/dev/null || echo "  (none)"
echo "-- playback (out) --";  aplay -l  2>/dev/null || echo "  (none)"

cat <<EOF

\033[1;32mDone.\033[0m Next:

  source $ENVFILE            # load key + paths into this shell
  cd $SRC

  # 1) Test the BRAIN first (text only, no speaker needed):
  $VENV/bin/python -m companion.chat --persona companion/persona.example.json

  # 2) Full VOICE loop (needs a mic AND an audio-OUT device — see chat):
  $VENV/bin/python -m companion.voice_chat --persona companion/persona.example.json

If the mic/speaker aren't the default ALSA device, note the card numbers from
the lists above and pass e.g.  --mic-device plughw:1,0
EOF
