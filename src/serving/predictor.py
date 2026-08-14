"""
Predictor — stateful inference over the trained SobrietyLSTM.
=============================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Loads the artifacts the pipeline produces:
  * models/lstm/lstm_best.pt              (state_dict for SobrietyLSTM)
  * data/processed/lstm_sequences/<run>/scaler.pkl   (train-only StandardScaler)
  * data/processed/lstm_sequences/<run>/meta.pkl      (feature order, T, mask)

and turns a Session (enrollment profile + stream of check-ins) into a risk
assessment. The hazard->survival->risk math mirrors the C-index/Brier block in
`pipeline/train_lstm.py` exactly, so scores are comparable to the offline eval.

The Bayesian latent class (Remission vs Cycling) is optional and only wired if
its trace is present AND the training script has been patched to persist its
standardization constants (see `_maybe_latent_class`).
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import torch

from .contract import (FEATURE_COLS, CONTINUOUS_COLS, MASK_TOKEN, MAX_WINDOWS,
                       WINDOW_DAYS)
from .model_def import SobrietyLSTM
from .state import Session


class Predictor:
    def __init__(self, model_path: Path, scaler_path: Path, meta_path: Path,
                 device: str | None = None, horizons=(90, 180, 360)):
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.horizons = horizons

        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
        # Fail fast if the served contract drifts from what the model was trained on.
        if self.meta["feature_cols"] != FEATURE_COLS:
            raise ValueError("contract.FEATURE_COLS != meta['feature_cols']; "
                             "serving would feed the model mis-ordered features.")
        self.T = int(self.meta["max_windows"])

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        # Pre-extract scaler stats in CONTINUOUS_COLS order for cheap per-window scaling.
        self._mean = self.scaler.mean_.astype(np.float32)
        self._scale = self.scaler.scale_.astype(np.float32)

        self.model = SobrietyLSTM(input_dim=len(FEATURE_COLS), max_time=self.T)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self._feat_index = {c: i for i, c in enumerate(FEATURE_COLS)}
        self._cont_index = [self._feat_index[c] for c in CONTINUOUS_COLS]

    # ── feature assembly ──────────────────────────────────────────────────
    def _row(self, static_enc: dict, tv_enc: dict) -> np.ndarray:
        """Build one scaled 40-vector in FEATURE_COLS order."""
        row = np.empty(len(FEATURE_COLS), dtype=np.float32)
        merged = {**tv_enc, **static_enc}
        for i, c in enumerate(FEATURE_COLS):
            row[i] = merged[c]
        # scale continuous subset in place, mirroring StandardScaler.transform
        for j, fi in enumerate(self._cont_index):
            row[fi] = (row[fi] - self._mean[j]) / self._scale[j]
        return row

    def _tensor(self, session: Session):
        """(1, T, F) padded input + (1, T) mask from the session's window stream.
        Windows beyond T are truncated to the most recent T (model horizon)."""
        rows = [self._row(session.static_encoded, tv) for tv in session.windows]
        rows = rows[-self.T:]                      # keep most recent T check-ins
        n = len(rows)
        X = np.zeros((1, self.T, len(FEATURE_COLS)), dtype=np.float32)  # padded=0 (as in training)
        mask = np.ones((1, self.T), dtype=bool)    # True = padded
        for t, r in enumerate(rows):
            X[0, t] = r
            mask[0, t] = False
        return (torch.from_numpy(X).to(self.device),
                torch.from_numpy(mask).to(self.device), n)

    # ── scoring ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def score(self, session: Session) -> dict:
        if not session.windows:
            raise ValueError("no observations yet for this session")
        X, mask, n_obs = self._tensor(session)
        hazards, _context, attn = self.model(X, mask)
        hazards = hazards.cpu().numpy()[0]         # (T,)
        attn = attn.cpu().numpy()[0][:n_obs]       # attention over observed windows

        # S(t) = prod(1-h_k); survival curve + risk = 1 - S at each horizon.
        log_surv = np.log(1 - np.clip(hazards, 1e-8, 1 - 1e-8))
        surv_curve = np.exp(np.cumsum(log_surv))   # (T,)
        risk_at = {}
        for h in self.horizons:
            w = min(int(h / WINDOW_DAYS), self.T - 1)
            risk_at[h] = float(1 - surv_curve[w])

        return {
            "n_observed_windows": n_obs,
            "hazard_curve": hazards.tolist(),
            "survival_curve": surv_curve.tolist(),
            "risk_by_horizon_days": risk_at,       # e.g. {90: .12, 180: .3, 360: .55}
            "attention_weights": attn.tolist(),    # which recent windows drove the read
            "peak_attention_window": int(np.argmax(attn)) if n_obs else None,
        }

    # ── optional latent class ─────────────────────────────────────────────
    def _maybe_latent_class(self, session: Session):
        """Remission-vs-Cycling membership from the Bayesian posterior.

        GAP: train_bayesian.py fits alpha but persists only the trace and
        posterior_class_probs.npy for the *test* set — it does NOT persist the
        continuous standardization (train_mean/train_std over
        [sud_severity, social_support, prior_tx_episodes]) needed to score a NEW
        patient. Wire this up only after patching that script to dump those
        constants (e.g. models/bayesian/bayesian_scaler.json) alongside alpha.
        Until then, return None so the LSTM read stands alone.
        """
        return None
