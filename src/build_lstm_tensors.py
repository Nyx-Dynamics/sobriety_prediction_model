"""
Build LSTM Input Tensors (Steps 28-32)
=======================================
Sobriety Prediction Model | Nyx Dynamics LLC | MIT MicroMasters Capstone

Produces (N, T, F) tensor with padding mask, train-only StandardScaler,
and aligned outcome vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
MASK_TOKEN = -999.0

# ── Load data ─────────────────────────────────────────────────────────
panel = pd.read_csv(config.PANEL_DIR / "sud_panel.csv")
static = pd.read_csv(config.STATIC_DIR / "sud_static.csv")
outcomes = pd.read_csv(config.OUTCOMES_DIR / "sud_outcomes.csv")

# ══════════════════════════════════════════════════════════════════════
# Step 28: Merge panel + static (broadcast static to every window)
# ══════════════════════════════════════════════════════════════════════

# Select static features to broadcast (drop identifiers, free text)
static_features = [
    "patient_id",
    # Numeric
    "sud_type", "sud_severity", "prior_tx_episodes",
    "housing_stability", "social_support",
    "age_at_enrollment", "age_first_use",
    # MH binary flags
    "mdd", "ptsd", "gad", "adhd", "bipolar_i", "bipolar_ii", "psychosis", "bpd",
    # Medical comorbidities
    "med_hcv", "med_hiv", "med_chronic_pain", "med_tbi_history",
    "med_liver_disease", "med_cardiovascular", "med_ms_autoimmune",
]

# Encode categorical static features before merge
static_enc = static.copy()

# sex → numeric
sex_map = {"Male": 0, "Female": 1, "Non-binary": 2}
static_enc["sex"] = static_enc["sex"].map(sex_map)

# race_ethnicity → numeric
race_map = {"White": 0, "Black": 1, "Hispanic": 2, "Asian": 3, "Other/Multiracial": 4}
static_enc["race_ethnicity"] = static_enc["race_ethnicity"].map(race_map)

# route_of_admin → numeric
route_map = {"IV": 0, "Smoking": 1, "Intranasal": 2, "Oral": 3, "Mixed": 4}
static_enc["route_of_admin"] = static_enc["route_of_admin"].map(route_map)

# treatment_modality → numeric
tx_map = {"IOP": 0, "Residential": 1, "MAT": 2, "Outpatient": 3, "Detox_only": 4, "None": 5}
static_enc["treatment_modality"] = static_enc["treatment_modality"].map(tx_map).fillna(5)

# insurance → numeric
ins_map = {"Medicaid": 0, "Private": 1, "Medicare": 2, "Uninsured": 3, "VA": 4}
static_enc["insurance_status"] = static_enc["insurance_status"].map(ins_map)

static_cols = static_features + ["sex", "race_ethnicity", "route_of_admin",
                                  "treatment_modality", "insurance_status"]
# De-duplicate patient_id
static_cols = list(dict.fromkeys(static_cols))
static_enc = static_enc[static_cols]

# Encode panel categoricals
panel_enc = panel.copy()
panel_enc["active_depression"] = panel_enc["active_depression"].astype(int)
panel_enc["employed"] = panel_enc["employed"].astype(int)
panel_enc["positive_event"] = panel_enc["positive_event"].astype(int)
panel_enc["sparse_window"] = panel_enc["sparse_window"].astype(int)

# adverse_event → numeric
ae_map = {"none": 0, "mild": 1, "severe": 2}
panel_enc["adverse_event"] = panel_enc["adverse_event"].map(ae_map).fillna(0).astype(int)

# Select panel time-varying features
panel_tv_cols = [
    "patient_id", "window",
    "active_depression", "med_adherence", "employed",
    "adverse_event", "positive_event",
    "phq9_score", "gad7_score",
    "sober_this_window", "uds_positive",
    "meetings_per_month", "social_support_score",
    "sparse_window", "cumulative_stress",
]
panel_enc = panel_enc[panel_tv_cols]

# Merge: broadcast static to every window
merged = panel_enc.merge(static_enc, on="patient_id", how="left")

# Final feature columns (everything except patient_id, window)
feature_cols = [c for c in merged.columns if c not in ("patient_id", "window")]
print(f"Feature columns ({len(feature_cols)}):")
for i, c in enumerate(feature_cols):
    print(f"  [{i:2d}] {c}")

# ══════════════════════════════════════════════════════════════════════
# Step 29: StandardScaler on continuous features, fit on TRAIN only
# ══════════════════════════════════════════════════════════════════════

# Identify continuous features to scale
continuous_cols = [
    "sud_severity", "social_support", "med_adherence", "cumulative_stress",
    "phq9_score", "gad7_score", "social_support_score",
    "meetings_per_month", "age_at_enrollment", "age_first_use",
    "prior_tx_episodes",
]

# Get ordered patient list aligned with outcomes
patient_order = outcomes["patient_id"].values
assert set(patient_order) == set(merged["patient_id"].unique()), "Patient mismatch!"

# Train/test split at patient level (80/20) for scaler fitting
train_pids, test_pids = train_test_split(
    patient_order, test_size=0.2, random_state=42,
    stratify=outcomes["event"].values
)
print(f"\nTrain patients: {len(train_pids)}, Test patients: {len(test_pids)}")

# Fit scaler on training data only
train_mask = merged["patient_id"].isin(train_pids)
scaler = StandardScaler()
scaler.fit(merged.loc[train_mask, continuous_cols])

# Transform ALL data with train-fitted scaler
merged[continuous_cols] = scaler.transform(merged[continuous_cols])

# Save scaler
with open(config.LSTM_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✓ Scaler fitted on training data and saved")

# Print scaler stats
scaler_stats = pd.DataFrame({
    "feature": continuous_cols,
    "mean": scaler.mean_.round(4),
    "std": scaler.scale_.round(4),
})
print(f"\nScaler parameters (from training set):\n{scaler_stats.to_string(index=False)}")

# ══════════════════════════════════════════════════════════════════════
# Step 30: Pad sequences to max_windows with mask token
# ══════════════════════════════════════════════════════════════════════

max_windows = merged["window"].max() + 1  # 0-indexed
N = len(patient_order)
F = len(feature_cols)
T = max_windows

print(f"\nTarget tensor shape: ({N}, {T}, {F}) = (patients, max_windows, features)")

# Build 3D array
X = np.full((N, T, F), MASK_TOKEN, dtype=np.float32)
padding_mask = np.ones((N, T), dtype=bool)  # True = padded (masked)

for i, pid in enumerate(patient_order):
    pt = merged[merged["patient_id"] == pid].sort_values("window")
    windows = pt["window"].values
    values = pt[feature_cols].values.astype(np.float32)

    for j, w in enumerate(windows):
        if w < T:
            X[i, w, :] = values[j]
            padding_mask[i, w] = False  # False = valid (not padded)

# Replace any remaining NaN in valid positions with 0 (rare edge cases)
valid_positions = ~padding_mask
nan_in_valid = np.isnan(X[valid_positions[:, :, np.newaxis].repeat(F, axis=2)])
n_nan_valid = nan_in_valid.sum()
if n_nan_valid > 0:
    print(f"⚠ Found {n_nan_valid} NaN in valid positions — replacing with 0")
    # Only zero out NaN in valid timesteps
    for i in range(N):
        for t in range(T):
            if not padding_mask[i, t]:
                nan_mask = np.isnan(X[i, t, :])
                X[i, t, nan_mask] = 0.0

# ══════════════════════════════════════════════════════════════════════
# Step 31: Build outcome vectors and save everything
# ══════════════════════════════════════════════════════════════════════

y_event = outcomes.set_index("patient_id").loc[patient_order, "event"].values.astype(np.int64)
y_days = outcomes.set_index("patient_id").loc[patient_order, "days_to_event"].values.astype(np.float32)

# Save arrays
np.save(config.LSTM_DIR / "X_sequences.npy", X)
np.save(config.LSTM_DIR / "y_event.npy", y_event)
np.save(config.LSTM_DIR / "y_days.npy", y_days)
np.save(config.LSTM_DIR / "padding_mask.npy", padding_mask)

# Also save the train/test split indices and feature metadata
meta = {
    "patient_order": patient_order.tolist(),
    "train_pids": train_pids.tolist(),
    "test_pids": test_pids.tolist(),
    "feature_cols": feature_cols,
    "continuous_cols": continuous_cols,
    "max_windows": T,
    "mask_token": MASK_TOKEN,
}
with open(config.LSTM_DIR / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)

# ══════════════════════════════════════════════════════════════════════
# Step 32: Verification
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("LSTM TENSOR VERIFICATION")
print("=" * 70)

print(f"\n── Tensor shapes ──")
print(f"  X_sequences : {X.shape}  (N={N}, T={T}, F={F})")
print(f"  y_event     : {y_event.shape}  (N={N},)")
print(f"  y_days      : {y_days.shape}  (N={N},)")
print(f"  padding_mask: {padding_mask.shape}  (N={N}, T={T})")

# NaN check — only in valid (unmasked) positions
nan_check_total = 0
for i in range(N):
    valid_t = np.where(~padding_mask[i])[0]
    if len(valid_t) > 0:
        nan_check_total += np.isnan(X[i, valid_t, :]).sum()
print(f"\n── NaN check ──")
print(f"  NaN in valid (unmasked) positions: {nan_check_total}")
if nan_check_total == 0:
    print("  ✓ PASS — no NaN in X_sequences after masking")
else:
    print("  ✗ FAIL — NaN detected in valid positions!")

# Verify mask token in padded positions
padded_vals = X[padding_mask]
all_mask = (padded_vals == MASK_TOKEN).all()
print(f"\n── Padding check ──")
print(f"  All padded positions = {MASK_TOKEN}? {all_mask}")
print(f"  Padded timesteps: {padding_mask.sum()} / {N * T} ({padding_mask.mean()*100:.1f}%)")
print(f"  Valid timesteps:  {(~padding_mask).sum()} / {N * T} ({(~padding_mask).mean()*100:.1f}%)")

# Event rate match
tensor_event_rate = y_event.mean()
csv_event_rate = outcomes["event"].mean()
print(f"\n── Event rate check ──")
print(f"  Tensor event rate:  {tensor_event_rate:.4f}")
print(f"  CSV event rate:     {csv_event_rate:.4f}")
print(f"  Match: {np.isclose(tensor_event_rate, csv_event_rate)}")

# Days distribution match
print(f"\n── Days to event check ──")
print(f"  Tensor: mean={y_days.mean():.1f}, std={y_days.std():.1f}, "
      f"min={y_days.min():.0f}, max={y_days.max():.0f}")
csv_days = outcomes["days_to_event"].values
print(f"  CSV:    mean={csv_days.mean():.1f}, std={csv_days.std():.1f}, "
      f"min={csv_days.min():.0f}, max={csv_days.max():.0f}")

# Sequence lengths
seq_lengths = (~padding_mask).sum(axis=1)
print(f"\n── Sequence lengths ──")
print(f"  Mean: {seq_lengths.mean():.1f}")
print(f"  Min:  {seq_lengths.min()}")
print(f"  Max:  {seq_lengths.max()}")
print(f"  Median: {np.median(seq_lengths):.0f}")

# File sizes
files = {
    "X_sequences.npy": config.LSTM_DIR / "X_sequences.npy",
    "y_event.npy": config.LSTM_DIR / "y_event.npy",
    "y_days.npy": config.LSTM_DIR / "y_days.npy",
    "padding_mask.npy": config.LSTM_DIR / "padding_mask.npy",
    "scaler.pkl": config.LSTM_DIR / "scaler.pkl",
    "meta.pkl": config.LSTM_DIR / "meta.pkl",
}
print(f"\n── Saved files ──")
for name, path in files.items():
    size = os.path.getsize(path)
    if size > 1_000_000:
        print(f"  {name}: {size / 1_000_000:.1f} MB")
    else:
        print(f"  {name}: {size / 1_000:.1f} KB")

print(f"\n── Train/test split ──")
print(f"  Train: {len(train_pids)} patients ({len(train_pids)/N*100:.0f}%)")
print(f"  Test:  {len(test_pids)} patients ({len(test_pids)/N*100:.0f}%)")
train_event_rate = y_event[np.isin(patient_order, train_pids)].mean()
test_event_rate = y_event[np.isin(patient_order, test_pids)].mean()
print(f"  Train event rate: {train_event_rate:.4f}")
print(f"  Test event rate:  {test_event_rate:.4f}")
