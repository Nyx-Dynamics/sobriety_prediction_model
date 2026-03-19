"""
Build Survival Outcome File (Steps 24-27)
==========================================
Sobriety Prediction Model | Nyx Dynamics LLC | MIT MicroMasters Capstone
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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import diptest

np.random.seed(42)

# ── Load source data ──────────────────────────────────────────────────
panel = pd.read_csv(config.PANEL_DIR / "sud_panel.csv")
static = pd.read_csv(config.STATIC_DIR / "sud_static.csv")
panel_q = pd.read_csv(config.RAW_DIR / "patient_panel.csv")
panel_q["visit_date"] = pd.to_datetime(panel_q["visit_date"])

static_full = pd.read_csv(config.RAW_DIR / "patient_static.csv")
static_full["enrollment_date"] = pd.to_datetime(static_full["enrollment_date"])
entry_dates = static_full.set_index("patient_id")["enrollment_date"]

# Merge sud_type for stratification
sud_type_map = static.set_index("patient_id")["sud_type"]
severity_map = static.set_index("patient_id")["sud_severity"]

# ── Step 24-25: Determine event and time-to-event ────────────────────
# Logic:
#   - Walk through each patient's quarterly observations chronologically
#   - A confirmed relapse = UDS positive AND not sober (or UDS positive after
#     a period of sobriety). This is event=1.
#   - Censoring reasons: treatment completion (sober at end, no relapse),
#     loss to follow-up (early dropout), transfer, death
#   - days_to_event = event_date - enrollment_date
#   - For censored: days_to_event = last_contact_date - enrollment_date

patients = panel_q["patient_id"].unique()
outcome_rows = []

for pid in patients:
    pt = panel_q[panel_q["patient_id"] == pid].sort_values("visit_date").reset_index(drop=True)
    entry = entry_dates[pid]
    sud = sud_type_map.get(pid, 0)
    sev = severity_map.get(pid, 5.0)
    n_obs = len(pt)
    last_contact = pt["visit_date"].max()

    # Track sobriety trajectory to detect relapse
    # Relapse = first UDS+ after achieving sobriety (at least 1 sober quarter)
    # OR first UDS+ if patient never achieved sobriety (initial treatment failure)
    achieved_sobriety = False
    relapse_detected = False
    relapse_date = None
    relapse_quarter = None

    for idx, row in pt.iterrows():
        if row["sober_this_quarter"] == 1:
            achieved_sobriety = True
        elif achieved_sobriety and row["uds_positive"] == 1:
            # Relapse after sobriety
            relapse_detected = True
            relapse_date = row["visit_date"]
            relapse_quarter = row["quarter"]
            break

    # For patients who never achieved sobriety: check if they had a
    # confirmed positive screen (treatment failure = event)
    if not relapse_detected and not achieved_sobriety:
        # Use a probabilistic model: ~65% of never-sober patients are
        # coded as confirmed relapse (ongoing use), rest are censored
        # (lost to follow-up before confirmation)
        if np.random.random() < 0.65:
            # Event at their last positive UDS
            pos_visits = pt[pt["uds_positive"] == 1]
            if len(pos_visits) > 0:
                relapse_detected = True
                relapse_date = pos_visits.iloc[-1]["visit_date"]
                relapse_quarter = pos_visits.iloc[-1]["quarter"]

    # Determine censoring reason for non-events
    if not relapse_detected:
        last_sober = pt.iloc[-1]["sober_this_quarter"]
        # Assign censoring reason
        if last_sober == 1 and n_obs >= 8:
            censor_reason = "treatment_completion"
        elif n_obs <= 3:
            censor_reason = "loss_to_followup"
        elif np.random.random() < 0.03:
            censor_reason = "death"
        elif np.random.random() < 0.05:
            censor_reason = "transfer"
        else:
            censor_reason = "loss_to_followup" if not last_sober else "treatment_completion"

    # Compute days_to_event
    if relapse_detected:
        event = 1
        days_to_event = (relapse_date - entry).days
        censor_reason = None
        event_date = relapse_date
    else:
        event = 0
        days_to_event = (last_contact - entry).days
        event_date = pd.NaT

    # Ensure positive time
    days_to_event = max(1, days_to_event)

    outcome_rows.append({
        "patient_id": pid,
        "sud_type": sud,
        "sud_severity": sev,
        "event": event,
        "days_to_event": days_to_event,
        "event_date": event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else None,
        "entry_date": entry.strftime("%Y-%m-%d"),
        "last_contact_date": last_contact.strftime("%Y-%m-%d"),
        "censor_reason": censor_reason,
        "n_observations": n_obs,
        "achieved_sobriety": int(achieved_sobriety),
    })

outcomes = pd.DataFrame(outcome_rows)

# ── Save ──────────────────────────────────────────────────────────────
outcomes.to_csv(config.OUTCOMES_DIR / "sud_outcomes.csv", index=False)

# ══════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SURVIVAL OUTCOME FILE — sud_outcomes.csv")
print("=" * 70)
print(f"Shape: {outcomes.shape}")
print(f"\nEvent distribution:")
print(outcomes["event"].value_counts().rename({0: "Censored", 1: "Relapse"}))
print(f"\nEvent rate: {outcomes['event'].mean():.3f}")
print(f"\nCensoring reasons (event=0 only):")
print(outcomes[outcomes["event"] == 0]["censor_reason"].value_counts())
print(f"\nDays to event summary:")
print(outcomes["days_to_event"].describe().round(1))
print(f"\nFirst 5 rows:\n{outcomes.head().to_string()}")

# ── Step 26: Kaplan-Meier by sud_type + log-rank test ─────────────────
stim = outcomes[outcomes["sud_type"] == 0]
poly = outcomes[outcomes["sud_type"] == 1]

kmf_stim = KaplanMeierFitter()
kmf_poly = KaplanMeierFitter()

kmf_stim.fit(stim["days_to_event"], event_observed=stim["event"],
             label="Stimulant (n={})".format(len(stim)))
kmf_poly.fit(poly["days_to_event"], event_observed=poly["event"],
             label="Polysubstance (n={})".format(len(poly)))

# Log-rank test
lr = logrank_test(
    stim["days_to_event"], poly["days_to_event"],
    event_observed_A=stim["event"], event_observed_B=poly["event"]
)

print(f"\n── Log-rank test ──")
print(f"Test statistic: {lr.test_statistic:.3f}")
print(f"p-value: {lr.p_value:.2e}")
if lr.p_value < 0.001:
    print("→ Highly significant difference in survival curves (p < 0.001)")
elif lr.p_value < 0.05:
    print(f"→ Significant difference in survival curves (p = {lr.p_value:.4f})")
else:
    print(f"→ No significant difference (p = {lr.p_value:.4f})")

# Median survival times
print(f"\nMedian time to relapse:")
print(f"  Stimulant:     {kmf_stim.median_survival_time_:.0f} days")
print(f"  Polysubstance: {kmf_poly.median_survival_time_:.0f} days")

# Plot KM curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
kmf_stim.plot_survival_function(ax=ax, color="#2c7bb6", linewidth=2)
kmf_poly.plot_survival_function(ax=ax, color="#d7191c", linewidth=2)
ax.set_xlabel("Days from treatment entry", fontsize=12)
ax.set_ylabel("Sobriety probability", fontsize=12)
ax.set_title("Kaplan-Meier: Time to Relapse by SUD Type", fontsize=13)

# Add log-rank p-value annotation
if lr.p_value < 0.001:
    p_text = "p < 0.001"
else:
    p_text = f"p = {lr.p_value:.4f}"
ax.annotate(
    f"Log-rank test\n{p_text}\nχ² = {lr.test_statistic:.2f}",
    xy=(0.65, 0.85), xycoords="axes fraction",
    fontsize=11, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc="lower left")

# ── Step 27: Hartigan's dip test for bimodality ──────────────────────
event_times = outcomes[outcomes["event"] == 1]["days_to_event"].values

dip_stat, dip_p = diptest.diptest(event_times)

print(f"\n── Hartigan's dip test for bimodality ──")
print(f"Dip statistic: {dip_stat:.4f}")
print(f"p-value: {dip_p:.4f}")
if dip_p < 0.05:
    print("→ REJECT unimodality (p < 0.05): evidence of bimodality/multimodality")
    print("  Interpretation: relapse times cluster into distinct subgroups,")
    print("  suggesting early relapsers vs. late relapsers — consider mixture models")
else:
    print("→ FAIL TO REJECT unimodality (p ≥ 0.05): no strong evidence of bimodality")
    print("  Interpretation: relapse times follow a roughly unimodal distribution;")
    print("  standard survival models (Cox PH, AFT) are appropriate without mixture components")

# Plot event time distribution with dip test result
ax2 = axes[1]
ax2.hist(event_times, bins=40, color="#fdae61", edgecolor="white",
         alpha=0.85, density=True, label="Relapse events")
ax2.set_xlabel("Days to relapse", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("Distribution of Time to Relapse (Events Only)", fontsize=13)

# KDE overlay
from scipy.stats import gaussian_kde
kde = gaussian_kde(event_times, bw_method=0.2)
x_grid = np.linspace(event_times.min(), event_times.max(), 300)
ax2.plot(x_grid, kde(x_grid), color="#d7191c", linewidth=2, label="KDE")

# Annotate dip test
dip_text = (f"Hartigan's dip test\nD = {dip_stat:.4f}, p = {dip_p:.4f}\n"
            + ("→ Bimodal" if dip_p < 0.05 else "→ Unimodal"))
ax2.annotate(
    dip_text,
    xy=(0.60, 0.85), xycoords="axes fraction",
    fontsize=11, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8)
)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "survival_diagnostics.png", dpi=150)
print(f"\n✓ Figures saved: {config.FIGURES_DIR / 'survival_diagnostics.png'}")

# ── Summary table by SUD type ─────────────────────────────────────────
print(f"\n── Summary by SUD type ──")
summary = outcomes.groupby("sud_type").agg(
    n=("event", "count"),
    n_events=("event", "sum"),
    event_rate=("event", "mean"),
    median_days=("days_to_event", "median"),
    mean_days=("days_to_event", "mean"),
).round(3)
summary.index = ["Stimulant", "Polysubstance"]
print(summary.to_string())
