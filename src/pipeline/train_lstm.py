"""
Survival LSTM with Temporal Attention (Phase 3)
================================================
Sobriety Prediction Model | Nyx Dynamics LLC | MIT MicroMasters Capstone

DeepHit-style discrete-time survival LSTM with learned attention over
time windows and per-window hazard prediction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

# ── Device ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"[train_lstm] Device: {DEVICE}")

# ── Load data ─────────────────────────────────────────────────────────
X = np.load(config.LSTM_DIR / "X_sequences.npy")        # (N, T, F)
y_event = np.load(config.LSTM_DIR / "y_event.npy")      # (N,)
y_days = np.load(config.LSTM_DIR / "y_days.npy")         # (N,)
pad_mask = np.load(config.LSTM_DIR / "padding_mask.npy")  # (N, T) True=padded
train_idx = np.load(config.LSTM_DIR / "train_idx.npy")
val_idx = np.load(config.LSTM_DIR / "val_idx.npy")
test_idx = np.load(config.LSTM_DIR / "test_idx.npy")

with open(config.LSTM_DIR / "meta.pkl", "rb") as f:
    meta = pickle.load(f)

N, T, F_dim = X.shape
print(f"[train_lstm] X shape: ({N}, {T}, {F_dim})")
print(f"[train_lstm] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Replace mask token (-999) with 0 for LSTM input
X_clean = X.copy()
X_clean[pad_mask] = 0.0

# Convert days to discrete windows (30-day) for hazard targets
y_windows = np.clip((y_days / 30).astype(int), 0, T - 1)

# ── Tensors ───────────────────────────────────────────────────────────
def make_loader(idx, batch_size, shuffle=False):
    return DataLoader(
        TensorDataset(
            torch.FloatTensor(X_clean[idx]),
            torch.BoolTensor(pad_mask[idx]),
            torch.LongTensor(y_event[idx]),
            torch.LongTensor(y_windows[idx]),
            torch.FloatTensor(y_days[idx]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )

BATCH_SIZE = 64
train_loader = make_loader(train_idx, BATCH_SIZE, shuffle=True)
val_loader = make_loader(val_idx, BATCH_SIZE)
test_loader = make_loader(test_idx, BATCH_SIZE)

# ══════════════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════════════

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
        """
        h_seq: (B, T, H)
        mask:  (B, T) True = padded
        Returns: context (B, H), weights (B, T)
        """
        scores = self.attn(h_seq).squeeze(-1)  # (B, T)
        # Mask padded positions with -inf
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=1)  # (B, T)
        # Handle all-masked sequences (replace NaN with uniform)
        nan_mask = torch.isnan(weights)
        if nan_mask.any():
            weights = weights.masked_fill(nan_mask, 0.0)
        context = torch.bmm(weights.unsqueeze(1), h_seq).squeeze(1)  # (B, H)
        return context, weights


class SobrietyLSTM(nn.Module):
    """
    2-layer LSTM with temporal attention and per-window hazard head.

    Architecture:
        Input (B, T, F) → LSTM 2-layer (H=128, dropout=0.3)
        → Temporal attention over hidden states → context vector (B, H)
        → Survival head: Linear(H, T) → Sigmoid → per-window P(relapse)

    The attention-weighted context is also used for the ranking loss.
    """

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

        # Per-window hazard prediction
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_time),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        """
        x:    (B, T, F)
        mask: (B, T) True=padded
        Returns:
            hazards: (B, T) per-window hazard probabilities
            context: (B, H) attention-weighted representation
            attn_weights: (B, T) attention weights
        """
        # Pack padded sequences for LSTM efficiency
        lengths = (~mask).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        h_packed, _ = self.lstm(packed)
        h_seq, _ = nn.utils.rnn.pad_packed_sequence(
            h_packed, batch_first=True, total_length=self.max_time
        )  # (B, T, H)

        context, attn_weights = self.attention(h_seq, mask)
        context = self.dropout(context)
        hazards = self.hazard_head(context)  # (B, T)
        return hazards, context, attn_weights


# ══════════════════════════════════════════════════════════════════════
# DeepHit Loss
# ══════════════════════════════════════════════════════════════════════

class DeepHitLoss(nn.Module):
    """Combined NLL + ranking loss for discrete-time survival."""

    def __init__(self, alpha=0.5, sigma=0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, hazards, events, event_windows, days):
        """
        hazards:       (B, T) per-window hazard probs
        events:        (B,) 0/1
        event_windows: (B,) discrete window index of event/censor
        days:          (B,) continuous days to event
        """
        B, T_max = hazards.shape
        eps = 1e-8

        # Clamp hazards
        h = hazards.clamp(eps, 1 - eps)

        # Survival function: S(t) = prod_{k=0}^{t-1} (1 - h_k)
        log_one_minus_h = torch.log(1 - h)  # (B, T)

        # NLL component
        nll = torch.zeros(B, device=hazards.device)
        for i in range(B):
            t_i = event_windows[i].item()
            t_i = min(t_i, T_max - 1)
            if events[i] == 1:
                # log f(t) = log h(t) + sum_{k<t} log(1-h(k))
                nll[i] = -(torch.log(h[i, t_i]) + log_one_minus_h[i, :t_i].sum())
            else:
                # log S(t) = sum_{k<=t} log(1-h(k))
                nll[i] = -log_one_minus_h[i, :t_i + 1].sum()

        nll_loss = nll.mean()

        # Ranking component: concordance-like
        # For each event pair (i has event, j either later event or censored),
        # penalize if risk_i < risk_j (risk = cumulative hazard at event time)
        if self.alpha > 0 and events.sum() > 0:
            # Cumulative hazard as risk score
            cum_h = -log_one_minus_h.cumsum(dim=1)  # (B, T)
            risk = torch.zeros(B, device=hazards.device)
            for i in range(B):
                t_i = min(event_windows[i].item(), T_max - 1)
                risk[i] = cum_h[i, t_i]

            event_mask = events == 1
            rank_loss = torch.tensor(0.0, device=hazards.device)
            n_pairs = 0

            event_idx = torch.where(event_mask)[0]
            for i in event_idx:
                # Comparable pairs: j has longer survival time
                j_mask = days > days[i]
                if j_mask.sum() == 0:
                    continue
                diff = risk[j_mask] - risk[i]
                rank_loss = rank_loss + torch.exp(-diff / self.sigma).sum()
                n_pairs += j_mask.sum().item()

            if n_pairs > 0:
                rank_loss = rank_loss / n_pairs
        else:
            rank_loss = torch.tensor(0.0, device=hazards.device)

        # Clamp ranking loss to prevent explosions (NLL is naturally bounded)
        rank_loss = torch.clamp(rank_loss, max=10.0)

        total = (1 - self.alpha) * nll_loss + self.alpha * rank_loss
        return total, nll_loss.item(), rank_loss.item()


# ══════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════

model = SobrietyLSTM(
    input_dim=F_dim, hidden_dim=128, n_layers=2, dropout=0.3, max_time=T
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = DeepHitLoss(alpha=0.5, sigma=0.5)

MAX_EPOCHS = 60
PATIENCE = 10

model_dir = config.PROJECT_ROOT / "models" / "lstm"
model_dir.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "train_nll": [], "train_rank": []}

print("\n" + "=" * 70)
print("LSTM SURVIVAL MODEL — TRAINING")
print("=" * 70)
print(f"Architecture: LSTM(F={F_dim}, H=128, layers=2, dropout=0.3)")
print(f"Loss: DeepHit (alpha={criterion.alpha}, sigma={criterion.sigma})")
print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}, LR: 1e-3, MaxEpochs: {MAX_EPOCHS}")

for epoch in range(1, MAX_EPOCHS + 1):
    # ── Train ──
    model.train()
    train_losses, train_nlls, train_ranks = [], [], []

    for x_b, mask_b, ev_b, ew_b, days_b in train_loader:
        x_b = x_b.to(DEVICE)
        mask_b = mask_b.to(DEVICE)
        ev_b = ev_b.to(DEVICE)
        ew_b = ew_b.to(DEVICE)
        days_b = days_b.to(DEVICE)

        hazards, _, _ = model(x_b, mask_b)
        loss, nll_v, rank_v = criterion(hazards, ev_b, ew_b, days_b)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_losses.append(loss.item())
        train_nlls.append(nll_v)
        train_ranks.append(rank_v)

    avg_train = np.mean(train_losses)
    history["train_loss"].append(avg_train)
    history["train_nll"].append(np.mean(train_nlls))
    history["train_rank"].append(np.mean(train_ranks))

    # ── Validate ──
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_b, mask_b, ev_b, ew_b, days_b in val_loader:
            x_b = x_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            ev_b = ev_b.to(DEVICE)
            ew_b = ew_b.to(DEVICE)
            days_b = days_b.to(DEVICE)

            hazards, _, _ = model(x_b, mask_b)
            loss, _, _ = criterion(hazards, ev_b, ew_b, days_b)
            val_losses.append(loss.item())

    avg_val = np.mean(val_losses)
    history["val_loss"].append(avg_val)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}: train={avg_train:.4f}  val={avg_val:.4f}")

    # Early stopping
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), model_dir / "lstm_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

# Save final checkpoint
torch.save(model.state_dict(), model_dir / "lstm_final.pt")
print(f"\n✓ Best checkpoint: {model_dir / 'lstm_best.pt'}")
print(f"✓ Final checkpoint: {model_dir / 'lstm_final.pt'}")

# ══════════════════════════════════════════════════════════════════════
# Evaluation on test set
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

# Load best checkpoint
model.load_state_dict(torch.load(model_dir / "lstm_best.pt", map_location=DEVICE, weights_only=True))
model.eval()

all_hazards = []
all_contexts = []
all_attn = []
all_events = []
all_days = []

with torch.no_grad():
    for x_b, mask_b, ev_b, ew_b, days_b in test_loader:
        x_b = x_b.to(DEVICE)
        mask_b = mask_b.to(DEVICE)
        hazards, ctx, attn = model(x_b, mask_b)
        all_hazards.append(hazards.cpu().numpy())
        all_contexts.append(ctx.cpu().numpy())
        all_attn.append(attn.cpu().numpy())
        all_events.append(ev_b.numpy())
        all_days.append(days_b.numpy())

test_hazards = np.concatenate(all_hazards, axis=0)  # (N_test, T)
test_contexts = np.concatenate(all_contexts, axis=0)
test_attn = np.concatenate(all_attn, axis=0)
test_events = np.concatenate(all_events, axis=0)
test_days_arr = np.concatenate(all_days, axis=0)

# ── C-index ───────────────────────────────────────────────────────────
# Use fixed-horizon risk: 1 - S(t=360 days) as the risk score
# This avoids contamination by follow-up length
from lifelines.utils import concordance_index

log_surv = np.log(1 - np.clip(test_hazards, 1e-8, 1 - 1e-8))
t_horizon_win = min(int(360 / 30), T - 1)  # 12 windows = 360 days
surv_360 = np.exp(log_surv[:, :t_horizon_win + 1].sum(axis=1))
risk_scores = 1 - surv_360  # higher = more likely to relapse by 360 days

c_index = concordance_index(
    test_days_arr,
    -risk_scores,  # lifelines convention: higher predicted = longer survival
    test_events.astype(bool),
)
print(f"  C-index: {c_index:.4f}")

# ── Time-dependent AUC ────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score

def td_auc(hazards_arr, events, days, t_horizon):
    """AUC: can model distinguish events before t from survivors past t?"""
    # Survival prob to horizon
    t_win = min(int(t_horizon / 30), T - 1)
    surv_prob = np.exp(np.log(1 - np.clip(hazards_arr[:, :t_win + 1], 1e-8, 1 - 1e-8)).sum(axis=1))
    risk = 1 - surv_prob

    # Labels: event before t_horizon
    labels = ((events == 1) & (days <= t_horizon)).astype(int)

    # Exclude patients censored before t_horizon without event
    valid = (events == 1) | (days > t_horizon)
    if valid.sum() < 10 or labels[valid].sum() < 2:
        return float("nan")
    return roc_auc_score(labels[valid], risk[valid])

for t in [90, 180, 360]:
    auc = td_auc(test_hazards, test_events, test_days_arr, t)
    print(f"  AUC at t={t} days: {auc:.4f}" if not np.isnan(auc) else f"  AUC at t={t} days: insufficient data")

# ── Integrated Brier Score ────────────────────────────────────────────
def brier_score_at_t(hazards_arr, events, days, t_horizon):
    t_win = min(int(t_horizon / 30), T - 1)
    surv_prob = np.exp(np.log(1 - np.clip(hazards_arr[:, :t_win + 1], 1e-8, 1 - 1e-8)).sum(axis=1))
    labels = ((events == 1) & (days <= t_horizon)).astype(float)
    valid = (events == 1) | (days > t_horizon)
    if valid.sum() < 10:
        return float("nan")
    return np.mean((surv_prob[valid] - (1 - labels[valid])) ** 2)

time_points = np.arange(30, 361, 30)
brier_scores = [brier_score_at_t(test_hazards, test_events, test_days_arr, t)
                for t in time_points]
valid_bs = [(t, bs) for t, bs in zip(time_points, brier_scores) if not np.isnan(bs)]
if valid_bs:
    from scipy.integrate import trapezoid
    ibs = trapezoid([bs for _, bs in valid_bs], [t for t, _ in valid_bs]) / (valid_bs[-1][0] - valid_bs[0][0])
    print(f"  Integrated Brier Score (30-360d): {ibs:.4f}")
else:
    print("  Integrated Brier Score: insufficient data")

# ══════════════════════════════════════════════════════════════════════
# Training curve figure
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(history["train_loss"], label="Train total", color="#2c7bb6")
ax.plot(history["val_loss"], label="Val total", color="#d7191c")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training & Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(history["train_nll"], label="NLL", color="#2c7bb6")
ax.plot(history["train_rank"], label="Ranking", color="#fdae61")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss component")
ax.set_title("Loss Components (Train)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "lstm_training_curve.png", dpi=150)
print(f"\n✓ Training curve: {config.FIGURES_DIR / 'lstm_training_curve.png'}")

# ══════════════════════════════════════════════════════════════════════
# Attention weight visualization (6 sample test patients)
# ══════════════════════════════════════════════════════════════════════

# Find cumulative_stress feature index
feature_names = meta["feature_cols"]
stress_idx = feature_names.index("cumulative_stress") if "cumulative_stress" in feature_names else None

test_X_raw = X_clean[test_idx]
test_pad = pad_mask[test_idx]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Pick 6 patients with different profiles (3 events, 3 censored)
event_patients = np.where(test_events == 1)[0][:3]
censor_patients = np.where(test_events == 0)[0][:3]
sample_idx = np.concatenate([event_patients, censor_patients])

for plot_i, si in enumerate(sample_idx):
    ax = axes[plot_i]
    attn_w = test_attn[si]  # (T,)
    valid_len = (~test_pad[si]).sum()

    attn_valid = attn_w[:valid_len]
    windows = np.arange(valid_len)

    # Color by cumulative stress
    if stress_idx is not None:
        stress_vals = test_X_raw[si, :valid_len, stress_idx]
        # Normalize for coloring
        s_norm = (stress_vals - stress_vals.min()) / (stress_vals.max() - stress_vals.min() + 1e-8)
        colors = plt.cm.YlOrRd(s_norm)
    else:
        colors = "#2c7bb6"

    ax.bar(windows, attn_valid, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Window")
    ax.set_ylabel("Attention weight")
    status = "EVENT" if test_events[si] == 1 else "CENSORED"
    ax.set_title(f"Patient {test_idx[si]} ({status})\nDays={test_days_arr[si]:.0f}", fontsize=10)
    ax.set_xlim(-0.5, valid_len - 0.5)

fig.suptitle("LSTM Attention Weights (colored by cumulative stress)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "lstm_attention_sample.png", dpi=150)
print(f"✓ Attention plot: {config.FIGURES_DIR / 'lstm_attention_sample.png'}")
