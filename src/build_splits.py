"""
Stratified Train/Val/Test Splits (Steps 33-36)
================================================
Sobriety Prediction Model | Nyx Dynamics LLC | MIT MicroMasters Capstone

Three-way stratified split on event × sud_type × mh_burden,
with balance verification and scaler refit on final train split.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────
outcomes = pd.read_csv("/Users/acdstudpro/data/processed/outcomes/sud_outcomes.csv")
static = pd.read_csv("/Users/acdstudpro/data/processed/static/sud_static.csv")
panel = pd.read_csv("/Users/acdstudpro/data/processed/panel/sud_panel.csv")

# Load LSTM tensors and metadata
X = np.load("/Users/acdstudpro/data/processed/lstm_sequences/X_sequences.npy")
y_event = np.load("/Users/acdstudpro/data/processed/lstm_sequences/y_event.npy")
y_days = np.load("/Users/acdstudpro/data/processed/lstm_sequences/y_days.npy")
padding_mask = np.load("/Users/acdstudpro/data/processed/lstm_sequences/padding_mask.npy")

with open("/Users/acdstudpro/data/processed/lstm_sequences/meta.pkl", "rb") as f:
    meta = pickle.load(f)

patient_order = np.array(meta["patient_order"])
feature_cols = meta["feature_cols"]
continuous_cols = meta["continuous_cols"]
N = len(patient_order)

# ══════════════════════════════════════════════════════════════════════
# Step 33: Build stratification key and split 70/15/15
# ══════════════════════════════════════════════════════════════════════

# Build stratification variables aligned to patient_order
outcomes_idx = outcomes.set_index("patient_id").loc[patient_order]
static_idx = static.set_index("patient_id").loc[patient_order]

event = outcomes_idx["event"].values
sud_type = outcomes_idx["sud_type"].values

# MH burden: count of MH diagnoses per patient
mh_cols = ["mdd", "ptsd", "gad", "adhd", "bipolar_i", "bipolar_ii", "psychosis", "bpd"]
n_mh = static_idx[mh_cols].sum(axis=1).values
mh_high = (n_mh >= 2).astype(int)

# Composite stratification key: event × sud_type × mh_high → 8 strata
strat_key = event * 4 + sud_type * 2 + mh_high

# Check minimum stratum size
strat_counts = pd.Series(strat_key).value_counts().sort_index()
print("Stratification strata (event×sud_type×mh_high):")
labels = []
for e in [0, 1]:
    for s in [0, 1]:
        for m in [0, 1]:
            k = e * 4 + s * 2 + m
            label = f"event={e}, sud={'stim' if s==0 else 'poly'}, mh_high={m}"
            labels.append((k, label))
            count = strat_counts.get(k, 0)
            print(f"  [{k}] {label}: n={count}")

min_stratum = strat_counts.min()
print(f"\nSmallest stratum: n={min_stratum}")
if min_stratum < 6:
    print("⚠ Very small stratum — merging may be needed")

# Two-stage split: first 70/30, then 30 → 15/15
indices = np.arange(N)

train_idx, temp_idx = train_test_split(
    indices, test_size=0.30, random_state=42, stratify=strat_key
)

# Split temp into val/test (50/50 of the 30% = 15%/15%)
strat_key_temp = strat_key[temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=42, stratify=strat_key_temp
)

print(f"\nSplit sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
print(f"Proportions: {len(train_idx)/N:.2%} / {len(val_idx)/N:.2%} / {len(test_idx)/N:.2%}")

# ══════════════════════════════════════════════════════════════════════
# Step 34: Verify balance across splits
# ══════════════════════════════════════════════════════════════════════

def split_stats(idx, label):
    """Compute key metrics for a split."""
    return {
        "split": label,
        "n": len(idx),
        "event_rate": y_event[idx].mean(),
        "sud_type_1_rate": sud_type[idx].mean(),
        "mh_high_rate": mh_high[idx].mean(),
        "mean_n_mh": n_mh[idx].mean(),
        "mean_days": y_days[idx].mean(),
        "mean_seq_len": (~padding_mask[idx]).sum(axis=1).mean(),
    }

stats = pd.DataFrame([
    split_stats(train_idx, "train"),
    split_stats(val_idx, "val"),
    split_stats(test_idx, "test"),
])

print("\n" + "=" * 70)
print("SPLIT BALANCE VERIFICATION")
print("=" * 70)
print(stats.to_string(index=False, float_format="%.4f"))

# Check that no stratum rate differs by >5% (absolute) across splits
check_cols = ["event_rate", "sud_type_1_rate", "mh_high_rate"]
errors = []
for col in check_cols:
    vals = stats[col].values
    max_diff = vals.max() - vals.min()
    status = "✓ PASS" if max_diff <= 0.05 else "✗ FAIL"
    print(f"\n  {col}: range={vals.min():.4f}–{vals.max():.4f}, "
          f"max_diff={max_diff:.4f} {status}")
    if max_diff > 0.05:
        errors.append(f"{col} differs by {max_diff:.4f} (>5%)")

if errors:
    error_msg = "STRATIFICATION BALANCE CHECK FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
    raise ValueError(error_msg)
else:
    print("\n✓ All strata balanced within 5% tolerance across splits")

# Detailed stratum-level breakdown
print("\n── Per-stratum counts by split ──")
stratum_table = pd.DataFrame({
    "stratum": strat_key,
    "split": "unassigned",
})
stratum_table.loc[train_idx, "split"] = "train"
stratum_table.loc[val_idx, "split"] = "val"
stratum_table.loc[test_idx, "split"] = "test"

ct = pd.crosstab(stratum_table["stratum"], stratum_table["split"], margins=True)
ct = ct[["train", "val", "test", "All"]]
print(ct.to_string())

# ══════════════════════════════════════════════════════════════════════
# Step 35: Save split indices (NOT pre-split data)
# ══════════════════════════════════════════════════════════════════════

np.save("/Users/acdstudpro/data/processed/lstm_sequences/train_idx.npy", train_idx)
np.save("/Users/acdstudpro/data/processed/lstm_sequences/val_idx.npy", val_idx)
np.save("/Users/acdstudpro/data/processed/lstm_sequences/test_idx.npy", test_idx)

print("\n✓ Split indices saved:")
print("  train_idx.npy, val_idx.npy, test_idx.npy")

# ══════════════════════════════════════════════════════════════════════
# Refit scaler on final train split (overwrite previous 80/20 scaler)
# ══════════════════════════════════════════════════════════════════════

# Reload unscaled panel to refit scaler on the 70% train split
panel_raw = pd.read_csv("/Users/acdstudpro/data/processed/panel/sud_panel.csv")
static_raw = pd.read_csv("/Users/acdstudpro/data/processed/static/sud_static.csv")

# Encode categoricals (same as build_lstm_tensors.py)
panel_enc = panel_raw.copy()
panel_enc["active_depression"] = panel_enc["active_depression"].astype(int)
panel_enc["employed"] = panel_enc["employed"].astype(int)
panel_enc["positive_event"] = panel_enc["positive_event"].astype(int)
panel_enc["sparse_window"] = panel_enc["sparse_window"].astype(int)
ae_map = {"none": 0, "mild": 1, "severe": 2}
panel_enc["adverse_event"] = panel_enc["adverse_event"].map(ae_map).fillna(0).astype(int)

static_enc = static_raw.copy()
static_enc["sex"] = static_enc["sex"].map({"Male": 0, "Female": 1, "Non-binary": 2})
static_enc["race_ethnicity"] = static_enc["race_ethnicity"].map(
    {"White": 0, "Black": 1, "Hispanic": 2, "Asian": 3, "Other/Multiracial": 4})
static_enc["route_of_admin"] = static_enc["route_of_admin"].map(
    {"IV": 0, "Smoking": 1, "Intranasal": 2, "Oral": 3, "Mixed": 4})
static_enc["treatment_modality"] = static_enc["treatment_modality"].map(
    {"IOP": 0, "Residential": 1, "MAT": 2, "Outpatient": 3, "Detox_only": 4, "None": 5}).fillna(5)
static_enc["insurance_status"] = static_enc["insurance_status"].map(
    {"Medicaid": 0, "Private": 1, "Medicare": 2, "Uninsured": 3, "VA": 4})

# Merge
panel_tv_cols = [
    "patient_id", "window",
    "active_depression", "med_adherence", "employed",
    "adverse_event", "positive_event",
    "phq9_score", "gad7_score",
    "sober_this_window", "uds_positive",
    "meetings_per_month", "social_support_score",
    "sparse_window", "cumulative_stress",
]
static_merge_cols = [
    "patient_id", "sud_type", "sud_severity", "prior_tx_episodes",
    "housing_stability", "social_support", "age_at_enrollment", "age_first_use",
    "mdd", "ptsd", "gad", "adhd", "bipolar_i", "bipolar_ii", "psychosis", "bpd",
    "med_hcv", "med_hiv", "med_chronic_pain", "med_tbi_history",
    "med_liver_disease", "med_cardiovascular", "med_ms_autoimmune",
    "sex", "race_ethnicity", "route_of_admin", "treatment_modality", "insurance_status",
]

merged_raw = panel_enc[panel_tv_cols].merge(
    static_enc[static_merge_cols], on="patient_id", how="left"
)

# Fit scaler on TRAIN PATIENTS ONLY (70% split)
train_pids = patient_order[train_idx]
train_mask = merged_raw["patient_id"].isin(train_pids)

scaler_final = StandardScaler()
scaler_final.fit(merged_raw.loc[train_mask, continuous_cols])

# Transform all data
merged_scaled = merged_raw.copy()
merged_scaled[continuous_cols] = scaler_final.transform(merged_raw[continuous_cols])

# Rebuild X tensor with final scaler
MASK_TOKEN = -999.0
T = meta["max_windows"]
F = len(feature_cols)

X_final = np.full((N, T, F), MASK_TOKEN, dtype=np.float32)
padding_mask_final = np.ones((N, T), dtype=bool)

for i, pid in enumerate(patient_order):
    pt = merged_scaled[merged_scaled["patient_id"] == pid].sort_values("window")
    windows = pt["window"].values
    values = pt[feature_cols].values.astype(np.float32)
    for j, w in enumerate(windows):
        if w < T:
            X_final[i, w, :] = values[j]
            padding_mask_final[i, w] = False

# Zero out any NaN in valid positions
for i in range(N):
    for t in range(T):
        if not padding_mask_final[i, t]:
            nan_mask = np.isnan(X_final[i, t, :])
            X_final[i, t, nan_mask] = 0.0

# Overwrite saved arrays with correctly-scaled versions
np.save("/Users/acdstudpro/data/processed/lstm_sequences/X_sequences.npy", X_final)
np.save("/Users/acdstudpro/data/processed/lstm_sequences/padding_mask.npy", padding_mask_final)

with open("/Users/acdstudpro/data/processed/lstm_sequences/scaler.pkl", "wb") as f:
    pickle.dump(scaler_final, f)

# Update meta with final split info
meta["train_idx"] = train_idx.tolist()
meta["val_idx"] = val_idx.tolist()
meta["test_idx"] = test_idx.tolist()
meta["train_pids"] = train_pids.tolist()
meta["val_pids"] = patient_order[val_idx].tolist()
meta["test_pids"] = patient_order[test_idx].tolist()
with open("/Users/acdstudpro/data/processed/lstm_sequences/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

# ══════════════════════════════════════════════════════════════════════
# Step 36: Final data summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL DATA SUMMARY")
print("=" * 70)

# N per split
print(f"\n── Samples per split ──")
print(f"  Train:      {len(train_idx):>5}  ({len(train_idx)/N:>6.1%})")
print(f"  Validation: {len(val_idx):>5}  ({len(val_idx)/N:>6.1%})")
print(f"  Test:       {len(test_idx):>5}  ({len(test_idx)/N:>6.1%})")
print(f"  Total:      {N:>5}")

# Event rate per split
print(f"\n── Event rate per split ──")
for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
    er = y_event[idx].mean()
    n_events = y_event[idx].sum()
    print(f"  {name:>5}: {er:.4f}  ({n_events}/{len(idx)} events)")

# Feature completeness
print(f"\n── Feature completeness ──")
print(f"  Total features: {F}")
print(f"  Tensor shape: {X_final.shape}")
valid_mask = ~padding_mask_final
total_valid_cells = valid_mask.sum() * F
nan_count = 0
for i in range(N):
    valid_t = np.where(~padding_mask_final[i])[0]
    if len(valid_t) > 0:
        nan_count += np.isnan(X_final[i, valid_t, :]).sum()
completeness = (total_valid_cells - nan_count) / total_valid_cells * 100
print(f"  Valid timesteps: {valid_mask.sum()} / {N * T}")
print(f"  Feature completeness in valid positions: {completeness:.2f}%")
print(f"  NaN in valid positions: {nan_count}")

# Scaler verification
print(f"\n── Scaler verification ──")
print(f"  Fitted on: {len(train_idx)} train patients ({len(train_idx)/N:.0%} of data)")
print(f"  Scaled features ({len(continuous_cols)}):")
scaler_df = pd.DataFrame({
    "feature": continuous_cols,
    "train_mean": scaler_final.mean_.round(4),
    "train_std": scaler_final.scale_.round(4),
})
print(scaler_df.to_string(index=False))

# Verify scaler was NOT fitted on val/test
# Check that train continuous means are ~0 and stds ~1 in the tensor
train_continuous_indices = [feature_cols.index(c) for c in continuous_cols]
train_valid = []
for i in train_idx:
    valid_t = np.where(~padding_mask_final[i])[0]
    if len(valid_t) > 0:
        train_valid.append(X_final[i, valid_t][:, train_continuous_indices])
train_block = np.vstack(train_valid)
train_means = train_block.mean(axis=0)
train_stds = train_block.std(axis=0)

print(f"\n  Post-scaling train statistics (should be ~0 mean, ~1 std):")
for j, col in enumerate(continuous_cols):
    print(f"    {col}: mean={train_means[j]:.4f}, std={train_stds[j]:.4f}")

all_close = np.allclose(train_means, 0, atol=0.05) and np.allclose(train_stds, 1, atol=0.1)
print(f"\n  Scaler train-only fit confirmed: {'✓ PASS' if all_close else '✗ CHECK MANUALLY'}")
