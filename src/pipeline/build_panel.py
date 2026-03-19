"""
Build Longitudinal Panel — 30-day windows (Steps 19-23)
========================================================
Sobriety Prediction Model | Nyx Dynamics LLC | MIT MicroMasters Capstone

Converts quarterly observations into 30-day windowed panel,
derives time-varying features, computes cumulative stress,
and flags sparse windows.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Load source data ──────────────────────────────────────────────────
panel_q = pd.read_csv(config.PANEL_DIR / "patient_panel.csv")
static = pd.read_csv(config.STATIC_DIR / "sud_static.csv")

panel_q["visit_date"] = pd.to_datetime(panel_q["visit_date"])

# Get enrollment (entry) dates from the original static file
static_full = pd.read_csv(config.STATIC_DIR / "patient_static.csv")
static_full["enrollment_date"] = pd.to_datetime(static_full["enrollment_date"])
entry_dates = static_full.set_index("patient_id")["enrollment_date"]

# Merge MDD flag for active_depression derivation
mdd_flag = static.set_index("patient_id")["mdd"]

# ── Step 19: Expand quarterly → 30-day windows ───────────────────────
# For each patient, create 30-day windows from entry_date through their
# last observation. Quarterly data is interpolated into the windows.

rows = []
patients = panel_q["patient_id"].unique()

for pid in patients:
    pt = panel_q[panel_q["patient_id"] == pid].sort_values("visit_date").reset_index(drop=True)
    entry = entry_dates[pid]
    last_visit = pt["visit_date"].max()
    has_mdd = mdd_flag.get(pid, 0)

    # Total 30-day windows from entry to last visit
    total_days = (last_visit - entry).days
    n_windows = max(1, total_days // 30 + 1)

    for w in range(n_windows):
        window_start = entry + pd.Timedelta(days=30 * w)
        window_end = window_start + pd.Timedelta(days=29)

        # Find the nearest quarterly observation (within ±45 days of window midpoint)
        window_mid = window_start + pd.Timedelta(days=15)
        time_diffs = (pt["visit_date"] - window_mid).abs()
        nearest_idx = time_diffs.idxmin()
        nearest_dist = time_diffs[nearest_idx].days

        # If nearest quarterly obs is >60 days away, this window has limited data
        has_source = nearest_dist <= 60
        src = pt.loc[nearest_idx] if has_source else None

        # ── Step 20: Derive time-varying features ─────────────────

        # -- active_depression (bool) --
        # Based on PHQ-9 ≥ 10 (moderate+) AND baseline MDD flag
        if has_source:
            phq9 = src["phq9_score"]
            active_depression = bool((phq9 >= 10) and has_mdd) if pd.notna(phq9) else None
        else:
            active_depression = None

        # -- med_adherence (0-1 float) --
        # Derive from tx_adherence_pct, rescale, add within-quarter noise
        if has_source and pd.notna(src["tx_adherence_pct"]):
            base_adh = src["tx_adherence_pct"] / 100.0
            # Add 30-day level noise (adherence fluctuates within quarter)
            noise = np.random.normal(0, 0.08)
            med_adherence = np.clip(base_adh + noise, 0, 1).round(3)
        else:
            med_adherence = None

        # -- employed (bool) --
        # Derive from life events: got_employment sets employed=True,
        # job_loss resets. Use cumulative state tracking.
        if has_source:
            # Base employment from social_support > 5 as proxy, modified by events
            emp_base = src["social_support_score"] > 5.5 if pd.notna(src["social_support_score"]) else None
            got_job = bool(src["le_got_employment"]) if pd.notna(src.get("le_got_employment")) else False
            lost_job = bool(src["le_job_loss"]) if pd.notna(src.get("le_job_loss")) else False
            if got_job:
                employed = True
            elif lost_job:
                employed = False
            else:
                # Stochastic employment based on proxy + window jitter
                employed = bool(emp_base) if emp_base is not None else None
        else:
            employed = None

        # -- adverse_event (none/mild/severe) --
        if has_source:
            # Count negative life events this source quarter
            neg_events = (
                int(src.get("le_job_loss", 0) or 0)
                + int(src.get("le_housing_instability", 0) or 0)
                + int(src.get("le_incarceration", 0) or 0)
                + int(src.get("le_relationship_breakup", 0) or 0)
                + int(src.get("le_bereavement", 0) or 0)
                + int(src.get("le_legal_problem", 0) or 0)
                + int(src.get("le_peer_relapse", 0) or 0)
            )
            # Distribute across ~3 windows per quarter with some randomness
            # Each window gets a probabilistic share of the quarter's events
            p_event = min(neg_events / 3.0, 1.0)  # prob of event in this window
            if np.random.random() < p_event:
                # Severity: incarceration/bereavement → severe; others → mild
                severe_events = (
                    int(src.get("le_incarceration", 0) or 0)
                    + int(src.get("le_bereavement", 0) or 0)
                )
                if severe_events > 0 and np.random.random() < 0.6:
                    adverse_event = "severe"
                else:
                    adverse_event = "mild"
            else:
                adverse_event = "none"
        else:
            adverse_event = None

        # -- positive_event (bool) --
        if has_source:
            pos_events = (
                int(src.get("le_new_relationship", 0) or 0)
                + int(src.get("le_got_employment", 0) or 0)
                + int(src.get("le_stable_housing", 0) or 0)
            )
            p_pos = min(pos_events / 3.0, 1.0)
            positive_event = bool(np.random.random() < p_pos)
        else:
            positive_event = None

        # -- Carry forward source lab/score data for reference --
        if has_source:
            phq9_val = int(src["phq9_score"]) if pd.notna(src["phq9_score"]) else None
            gad7_val = int(src["gad7_score"]) if pd.notna(src["gad7_score"]) else None
            sober = int(src["sober_this_quarter"]) if pd.notna(src["sober_this_quarter"]) else None
            uds = int(src["uds_positive"]) if pd.notna(src["uds_positive"]) else None
            meetings = int(src["meetings_per_month"]) if pd.notna(src["meetings_per_month"]) else None
            social = float(src["social_support_score"]) if pd.notna(src["social_support_score"]) else None
        else:
            phq9_val = gad7_val = sober = uds = meetings = social = None

        # ── Step 22: Compute missingness for sparse_window flag ───
        tv_features = [
            active_depression, med_adherence, employed,
            adverse_event, positive_event,
            phq9_val, gad7_val, sober, uds, social
        ]
        n_missing = sum(1 for v in tv_features if v is None)
        pct_missing = n_missing / len(tv_features)
        sparse_window = pct_missing > 0.50

        rows.append({
            "patient_id": pid,
            "window": w,
            "window_start": window_start.strftime("%Y-%m-%d"),
            "window_end": window_end.strftime("%Y-%m-%d"),
            "source_quarter": int(src["quarter"]) if has_source else None,
            "source_distance_days": nearest_dist if has_source else None,
            # Step 20 features
            "active_depression": active_depression,
            "med_adherence": med_adherence,
            "employed": employed,
            "adverse_event": adverse_event,
            "positive_event": positive_event,
            # Carried-forward outcome/scores
            "phq9_score": phq9_val,
            "gad7_score": gad7_val,
            "sober_this_window": sober,
            "uds_positive": uds,
            "meetings_per_month": meetings,
            "social_support_score": social,
            # Step 22
            "sparse_window": sparse_window,
        })

panel_30 = pd.DataFrame(rows)

# ── Step 21: Cumulative stress (rolling weighted sum) ─────────────────
# stress_t = 0.8 * stress_{t-1} + severity_weight(adverse_event_t) - 0.3 * positive_event_t
# severity weights: none=0, mild=0.5, severe=1.5
# Clip to [0, 10]

severity_weight_map = {"none": 0.0, "mild": 0.5, "severe": 1.5}

stress_col = []
for pid in patients:
    mask = panel_30["patient_id"] == pid
    pt_idx = panel_30.index[mask]
    stress = 0.0
    for idx in pt_idx:
        ae = panel_30.loc[idx, "adverse_event"]
        pe = panel_30.loc[idx, "positive_event"]
        ae_weight = severity_weight_map.get(ae, 0.0) if pd.notna(ae) else 0.0
        pe_val = 0.3 if (pe is True or pe == 1) else 0.0
        stress = 0.8 * stress + ae_weight - pe_val
        stress = np.clip(stress, 0, 10)
        stress_col.append(round(stress, 3))

panel_30["cumulative_stress"] = stress_col

# ── Save ──────────────────────────────────────────────────────────────
panel_30.to_csv(config.PANEL_DIR / "sud_panel.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# Step 23: Diagnostics
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("LONGITUDINAL PANEL — sud_panel.csv (30-day windows)")
print("=" * 70)
print(f"Shape: {panel_30.shape}")
print(f"Patients: {panel_30['patient_id'].nunique()}")
print(f"Windows range: {panel_30['window'].min()} – {panel_30['window'].max()}")
print(f"\nDtypes:\n{panel_30.dtypes}")
print(f"\nFirst 5 rows:\n{panel_30.head().to_string()}")

# ── Distribution of windows per patient ──────────────────────────────
windows_per_pt = panel_30.groupby("patient_id")["window"].count()
print(f"\n── Windows per patient ──")
print(windows_per_pt.describe().round(1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram of windows per patient
axes[0].hist(windows_per_pt, bins=30, color="#2c7bb6", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Number of 30-day windows")
axes[0].set_ylabel("Number of patients")
axes[0].set_title("Distribution of Windows per Patient")
axes[0].axvline(windows_per_pt.median(), color="red", linestyle="--",
                label=f"Median = {windows_per_pt.median():.0f}")
axes[0].legend()

# ── % sparse windows ─────────────────────────────────────────────────
n_sparse = panel_30["sparse_window"].sum()
pct_sparse = n_sparse / len(panel_30) * 100
print(f"\n── Sparse windows (>50% missing time-varying features) ──")
print(f"Total windows: {len(panel_30)}")
print(f"Sparse windows: {n_sparse} ({pct_sparse:.2f}%)")

# Sparse rate by window number
sparse_by_window = panel_30.groupby("window")["sparse_window"].mean() * 100
axes[1].bar(sparse_by_window.index, sparse_by_window.values,
            color="#d7191c", alpha=0.7)
axes[1].set_xlabel("Window number")
axes[1].set_ylabel("% sparse")
axes[1].set_title("Sparse Window Rate by Window Number")

# ── Adverse event rate by window (temporal confounding check) ────────
ae_by_window = panel_30[panel_30["adverse_event"].notna()].copy()
ae_by_window["ae_any"] = (ae_by_window["adverse_event"] != "none").astype(int)
ae_by_window["ae_severe"] = (ae_by_window["adverse_event"] == "severe").astype(int)

ae_rate = ae_by_window.groupby("window").agg(
    n_obs=("ae_any", "count"),
    any_ae_rate=("ae_any", "mean"),
    severe_ae_rate=("ae_severe", "mean"),
)

# Only plot windows with enough data
ae_plot = ae_rate[ae_rate["n_obs"] >= 20]
print(f"\n── Adverse event rate by window (first 15 + last 5) ──")
display_rows = pd.concat([ae_rate.head(15), ae_rate.tail(5)]).drop_duplicates()
print(display_rows.round(4).to_string())

axes[2].plot(ae_plot.index, ae_plot["any_ae_rate"], "o-",
             color="#2c7bb6", label="Any AE", markersize=3)
axes[2].plot(ae_plot.index, ae_plot["severe_ae_rate"], "s-",
             color="#d7191c", label="Severe AE", markersize=3)
axes[2].set_xlabel("Window number")
axes[2].set_ylabel("Event rate")
axes[2].set_title("Adverse Event Rate by Window\n(Temporal Confounding Check)")
axes[2].legend()
axes[2].axhline(ae_by_window["ae_any"].mean(), color="gray",
                linestyle=":", alpha=0.5, label="Overall mean")

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "panel_diagnostics.png", dpi=150)
print(f"\n✓ Diagnostics figure saved: {config.FIGURES_DIR / 'panel_diagnostics.png'}")

# ── Cumulative stress distribution ────────────────────────────────────
print(f"\n── Cumulative stress distribution ──")
print(panel_30["cumulative_stress"].describe().round(3))

# ── Overall null counts ───────────────────────────────────────────────
print(f"\n── Null counts ──")
nulls = panel_30.isnull().sum()
print(nulls[nulls > 0])
