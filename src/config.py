"""
Shared configuration for the sobriety prediction pipeline.
All output paths are routed through a timestamped run directory.

When run via run_pipeline.py, SUD_RUN_TIMESTAMP is set in the environment
so all scripts share the same RUN_ID. When run individually, each script
gets its own timestamp (and "latest" symlinks provide cross-script fallback).
"""

import os
from datetime import datetime
from pathlib import Path

# Project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Timestamped run ID — reuse env var if set (pipeline mode)
RUN_TIMESTAMP = os.environ.get("SUD_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"run_{RUN_TIMESTAMP}"

# Base directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / RUN_ID

# Data subdirectories (inside timestamped run)
RAW_DIR = DATA_DIR / "raw" / RUN_ID
STATIC_DIR = DATA_DIR / "processed" / "static" / RUN_ID
PANEL_DIR = DATA_DIR / "processed" / "panel" / RUN_ID
OUTCOMES_DIR = DATA_DIR / "processed" / "outcomes" / RUN_ID
LSTM_DIR = DATA_DIR / "processed" / "lstm_sequences" / RUN_ID
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Create all directories
for d in [RAW_DIR, STATIC_DIR, PANEL_DIR, OUTCOMES_DIR, LSTM_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Symlink "latest" to current run for convenience
def _symlink_latest(parent, run_id):
    latest = parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_id)

_symlink_latest(DATA_DIR / "raw", RUN_ID)
_symlink_latest(DATA_DIR / "processed" / "static", RUN_ID)
_symlink_latest(DATA_DIR / "processed" / "panel", RUN_ID)
_symlink_latest(DATA_DIR / "processed" / "outcomes", RUN_ID)
_symlink_latest(DATA_DIR / "processed" / "lstm_sequences", RUN_ID)
_symlink_latest(PROJECT_ROOT / "outputs", RUN_ID)

# Source directories
SRC_DIR = PROJECT_ROOT / "src"
SYNTHETIC_DIR = PROJECT_ROOT / "src" / "synthetic"
PIPELINE_DIR = PROJECT_ROOT / "src" / "pipeline"

# Data mode: "synthetic" runs generator first; "real" skips to pipeline
DATA_MODE = os.environ.get("SUD_DATA_MODE", "synthetic")

# Reproducibility
SEED = 42

print(f"[config] Run ID: {RUN_ID}")
print(f"[config] Data mode: {DATA_MODE}")
print(f"[config] Outputs: {OUTPUTS_DIR}")
