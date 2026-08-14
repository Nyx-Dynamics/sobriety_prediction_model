"""
Canonical model architecture (import target for both training and serving).
=============================================================================
Sobriety Prediction Model | Nyx Dynamics LLC

This is a verbatim extraction of the DeepHit-style attention LSTM defined in
`pipeline/train_lstm.py`. Serving MUST use the identical class so that
`torch.load(lstm_best.pt)` state_dict keys line up.

REFACTOR NOTE: `pipeline/train_lstm.py` should be changed to
    from serving.model_def import SobrietyLSTM, TemporalAttention
instead of redefining these classes, so the two never drift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Learned attention weights over LSTM hidden states."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_seq, mask):
        scores = self.attn(h_seq).squeeze(-1)              # (B, T)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=1)                 # (B, T)
        nan_mask = torch.isnan(weights)
        if nan_mask.any():
            weights = weights.masked_fill(nan_mask, 0.0)
        context = torch.bmm(weights.unsqueeze(1), h_seq).squeeze(1)  # (B, H)
        return context, weights


class SobrietyLSTM(nn.Module):
    """2-layer LSTM + temporal attention + per-window hazard head."""

    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.3, max_time=34):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_time = max_time

        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_time),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        lengths = (~mask).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        h_packed, _ = self.lstm(packed)
        h_seq, _ = nn.utils.rnn.pad_packed_sequence(
            h_packed, batch_first=True, total_length=self.max_time
        )
        context, attn_weights = self.attention(h_seq, mask)
        context = self.dropout(context)
        hazards = self.hazard_head(context)                # (B, T)
        return hazards, context, attn_weights
