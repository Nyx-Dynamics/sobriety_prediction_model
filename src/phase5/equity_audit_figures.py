"""
Phase 5: Equity Audit Figures (E1-E4)
=======================================
Publication-quality figures for the structural equity analysis.

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

np.random.seed(config.SEED)

# ── Load data (NTIES primary, synthetic fallback) ─────────────────────
proc_dir = config.PROJECT_ROOT / "data" / "phase5" / "processed"
synth_dir = config.PROJECT_ROOT / "data" / "phase5" / "stratified"

nties_path = proc_dir / "nties_phase5.csv"
synth_path = synth_dir / "patient_static_stratified.csv"

if nties_path.exists():
    static = pd.read_csv(nties_path)
    data_source = "NTIES"
    print(f"[figures] Using NTIES data (N={len(static)})")
elif synth_path.exists():
    static = pd.read_csv(synth_path)
    data_source = "Synthetic"
    print(f"[figures] Using synthetic data (N={len(static)})")
else:
    raise FileNotFoundError("No Phase 5 data found")

# For NTIES: event and days_to_event are already in static
# For synthetic: would need panel-based computation
panel = None

model_dir = config.PROJECT_ROOT / "models" / "phase5"
with open(model_dir / "pathway_results.json") as f:
    results = json.load(f)

gamma_posteriors = np.load(model_dir / "gamma_posteriors.npy")
gamma_mean = np.array(results["gamma_mean"])
gamma_lo = np.array(results["gamma_hdi_lo"])
gamma_hi = np.array(results["gamma_hdi_hi"])
pathway_names = results["pathway_names"]
resource_cols = results["resource_cols"]
pafs = results["pafs"]
R_std = np.array(results["R_std"])
resource_ranges_raw = np.array(results["resource_ranges_raw"])

fig_dir = config.PROJECT_ROOT / "outputs" / "phase5" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ── Build outcomes dataframe ──────────────────────────────────────────
# NTIES: event and days_to_event are columns in static
# Stratify by treatment_intensity (maps to resource level)
if "event" in static.columns and "days_to_event" in static.columns:
    oc = static[["caseid", "event", "days_to_event", "treatment_intensity"]].copy()
    oc.rename(columns={"days_to_event": "days"}, inplace=True)
else:
    raise ValueError("Expected event and days_to_event in data")

# Map treatment_intensity to resource strata for visualization
# 5=long residential → A, 4=short residential → B, 3=MAT → C, 2=OP → D
tx_to_stratum = {5: "A", 4: "B", 3: "C", 2: "D", 1: "D", 0: "D"}
oc["stratum"] = oc["treatment_intensity"].map(tx_to_stratum).fillna("D")

STRATUM_COLORS = {"A": "#2ca02c", "B": "#1f77b4", "C": "#ff7f0e", "D": "#d62728"}
STRATUM_LABELS = {
    "A": "Long-term residential",
    "B": "Short-term residential",
    "C": "MAT/Methadone",
    "D": "Outpatient/None",
}

# ══════════════════════════════════════════════════════════════════════
# Figure E1: KM by resource stratum
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 7))
medians = {}
strata_present = [s for s in ["A", "B", "C", "D"] if (oc["stratum"] == s).sum() > 0]
for sk in strata_present:
    sub = oc[oc["stratum"] == sk]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["days"], event_observed=sub["event"],
            label=f"{STRATUM_LABELS[sk]} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, color=STRATUM_COLORS[sk], linewidth=2, ci_alpha=0.12)
    medians[sk] = kmf.median_survival_time_

lr = multivariate_logrank_test(oc["days"], oc["stratum"], oc["event"])
ax.set_xlabel("Days from treatment entry", fontsize=12)
ax.set_ylabel("Sobriety probability", fontsize=12)
ax.set_title("Kaplan-Meier by Treatment Intensity (NTIES)", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="lower left")

median_text = "\n".join(f"  {sk}: {medians[sk]:.0f}d" for sk in strata_present)
ax.annotate(
    f"Log-rank p={lr.p_value:.2e}\n\nMedian survival:\n{median_text}",
    xy=(0.62, 0.65), xycoords="axes fraction", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
)
plt.tight_layout()
plt.savefig(fig_dir / "equity_figure1_km_by_stratum.png", dpi=200)
print(f"✓ E1: {fig_dir / 'equity_figure1_km_by_stratum.png'}")

# ══════════════════════════════════════════════════════════════════════
# Figure E2: Pathway attribution posteriors (4 panels)
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
pathway_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
resource_ranges_z = resource_ranges_raw / R_std

for k in range(4):
    ax = axes[k]
    samples = gamma_posteriors[:, k]
    # Convert to HR across full resource range
    hr_samples = np.exp(samples * resource_ranges_z[k])
    hr_mean = hr_samples.mean()
    hr_lo, hr_hi = np.percentile(hr_samples, [3, 97])

    ax.hist(hr_samples, bins=60, color=pathway_colors[k], alpha=0.7, density=True, edgecolor="white")

    # 94% HDI shading
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(hr_samples)
        x_fill = np.linspace(hr_lo, hr_hi, 200)
        ax.fill_between(x_fill, kde(x_fill), alpha=0.25, color=pathway_colors[k])
    except Exception:
        pass

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.5, label="HR=1")
    ax.axvline(hr_mean, color="black", linestyle="-", linewidth=2, label=f"Mean={hr_mean:.2f}")
    ax.set_title(f"{pathway_names[k]}\nHR = {hr_mean:.2f} [{hr_lo:.2f}, {hr_hi:.2f}]",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Hazard Ratio (Stratum A → D)")
    if k == 0:
        ax.legend(fontsize=9)

fig.suptitle("Structural Resource Pathway Hazard Ratios (94% HDI)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(fig_dir / "equity_figure2_pathway_posteriors.png", dpi=200)
print(f"✓ E2: {fig_dir / 'equity_figure2_pathway_posteriors.png'}")

# ══════════════════════════════════════════════════════════════════════
# Figure E3: Equity gap decomposition waterfall
# ══════════════════════════════════════════════════════════════════════

# Compute stratum-level sobriety rates (1 - event_rate)
stratum_sobriety = {}
for sk in strata_present:
    sub = oc[oc["stratum"] == sk]
    stratum_sobriety[sk] = 1 - sub["event"].mean()

p_best = stratum_sobriety.get("A", max(stratum_sobriety.values()))
p_worst = stratum_sobriety.get("D", min(stratum_sobriety.values()))
# Use best/worst strata present
best_stratum = max(stratum_sobriety, key=stratum_sobriety.get)
worst_stratum = min(stratum_sobriety, key=stratum_sobriety.get)
p_a = stratum_sobriety[best_stratum]
p_d = stratum_sobriety[worst_stratum]

# ── Load seed-trajectory PAF values ───────────────────────────────────
pafs_seed = results.get("paf_vs_seed", {})
nties_sobriety = 1 - static["event"].mean()  # ~0.091
seed_sobriety = 0.926
total_gap = seed_sobriety - nties_sobriety

# Monitoring PAF is the only meaningful positive contributor
monitoring_paf = pafs_seed.get("monitoring_intensity", 0.141)
monitoring_contribution = total_gap * monitoring_paf
unexplained = total_gap - monitoring_contribution

fig, ax = plt.subplots(figsize=(14, 6))

# Waterfall: NTIES → +monitoring → +unexplained → seed
bar_data = [
    ("NTIES population\n(observed)", 0, nties_sobriety, "#888888"),
    ("Monitoring continuity\nequalized to seed", nties_sobriety, monitoring_contribution, "#1f77b4"),
    ("Unexplained gap\n(confounding by indication)", nties_sobriety + monitoring_contribution, unexplained, "#cccccc"),
]

for label, start, width, color in bar_data:
    ax.barh(label, width, left=start, color=color, edgecolor="white", height=0.55)
    pct = width / total_gap * 100 if total_gap > 0 else 0
    if width > 0.02:
        ax.text(start + width / 2, label, f"{width:.3f}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if color != "#cccccc" else "black")

# Reference lines
ax.axvline(0.35, color="darkblue", linestyle=":", linewidth=1.5, alpha=0.7, label="Community midpoint (0.35)")
ax.axvline(seed_sobriety, color="darkgreen", linestyle=":", linewidth=1.5, alpha=0.7, label="Seed trajectory (0.926)")
ax.axvline(nties_sobriety, color="darkred", linestyle=":", linewidth=1.5, alpha=0.7, label=f"NTIES mean ({nties_sobriety:.1%})")

ax.set_xlabel("Sobriety rate", fontsize=12)
ax.set_title("Equity Gap Decomposition: NTIES → Seed Trajectory", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 1.05)
ax.grid(axis="x", alpha=0.3)

ax.annotate(
    f"85.9% of equity gap not explained by observable\n"
    f"resource pathways — confounding by indication\n"
    f"prevents full attribution in NTIES",
    xy=(0.55, 0.15), xycoords="axes fraction", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
)

plt.tight_layout()
plt.savefig(fig_dir / "equity_figure3_gap_decomposition.png", dpi=200)
print(f"✓ E3: {fig_dir / 'equity_figure3_gap_decomposition.png'}")

# ══════════════════════════════════════════════════════════════════════
# Figure E4: PAF vs. Seed Trajectory Reference
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 5))

# Use seed-trajectory PAF values
paf_seed_vals = [pafs_seed.get(col, 0) * 100 for col in resource_cols]
# Sort by absolute value
sorted_pairs = sorted(zip(pathway_names, paf_seed_vals, resource_cols, pathway_colors),
                       key=lambda x: abs(x[1]), reverse=True)
sorted_names = [p[0] for p in sorted_pairs]
sorted_vals = [p[1] for p in sorted_pairs]
sorted_colors = [p[3] for p in sorted_pairs]

y_pos = np.arange(len(sorted_names))
bars = ax.barh(y_pos, sorted_vals, color=sorted_colors, edgecolor="white", height=0.55)

# Labels — outside the bar on the appropriate side
for bar, val in zip(bars, sorted_vals):
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    if width < 0:
        ax.text(width - 0.3, y, f"{val:.1f}%",
                ha="right", va="center",
                fontsize=11, fontweight="bold")
    else:
        ax.text(width + 0.3, y, f"+{val:.1f}%",
                ha="left", va="center",
                fontsize=11, fontweight="bold",
                color="#1565C0")

ax.axvline(0, color="black", linewidth=1.0)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=11)
ax.set_xlabel("Population-Attributable Fraction vs. Seed Trajectory (%)", fontsize=12)
ax.set_title("PAF vs. Seed Trajectory Reference\n"
             "(physician monitoring program = reference level)",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

ax.annotate(
    "Negative PAF = confounding by indication\n"
    "(higher-intensity treatment given to\n"
    " more severe patients in NTIES)",
    xy=(0.02, 0.65), xycoords="axes fraction", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
)

plt.tight_layout()
plt.savefig(fig_dir / "equity_figure4_paf.png", dpi=200)
print(f"✓ E4: {fig_dir / 'equity_figure4_paf.png'}")
