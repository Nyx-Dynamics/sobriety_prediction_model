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

# ── Load data ─────────────────────────────────────────────────────────
data_dir = config.PROJECT_ROOT / "data" / "phase5" / "stratified"
static = pd.read_csv(data_dir / "patient_static_stratified.csv")
panel = pd.read_csv(data_dir / "patient_panel_stratified.csv")

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

# ── Build per-patient outcomes ────────────────────────────────────────
outcome_rows = []
for pid in static["patient_id"]:
    pt = panel[panel["patient_id"] == pid].sort_values("quarter")
    n_obs = len(pt)
    ever_sober = False
    event = 0
    days = n_obs * 90

    for _, row in pt.iterrows():
        if row["sober_this_quarter"] == 1:
            ever_sober = True
        elif ever_sober and row["uds_positive"] == 1:
            event = 1
            days = int(row["quarter"]) * 90
            break

    if event == 0 and not ever_sober and (pt["uds_positive"] == 1).any():
        if np.random.random() < 0.65:
            event = 1
            days = int(pt[pt["uds_positive"] == 1].iloc[-1]["quarter"]) * 90

    outcome_rows.append({"patient_id": pid, "event": event, "days": max(days, 1)})

oc = pd.DataFrame(outcome_rows).merge(static[["patient_id", "stratum"]], on="patient_id")

STRATUM_COLORS = {"A": "#2ca02c", "B": "#1f77b4", "C": "#ff7f0e", "D": "#d62728"}
STRATUM_LABELS = {"A": "High resource", "B": "Moderate", "C": "Low", "D": "Minimal"}

# ══════════════════════════════════════════════════════════════════════
# Figure E1: KM by resource stratum
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 7))
medians = {}
for sk in ["A", "B", "C", "D"]:
    sub = oc[oc["stratum"] == sk]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["days"], event_observed=sub["event"],
            label=f"Stratum {sk}: {STRATUM_LABELS[sk]} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, color=STRATUM_COLORS[sk], linewidth=2, ci_alpha=0.12)
    medians[sk] = kmf.median_survival_time_

# Log-rank test
lr = multivariate_logrank_test(oc["days"], oc["stratum"], oc["event"])
ax.set_xlabel("Days from treatment entry", fontsize=12)
ax.set_ylabel("Sobriety probability", fontsize=12)
ax.set_title("Kaplan-Meier by Resource Stratum", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="lower left")

median_text = "\n".join(f"  {sk}: {medians[sk]:.0f}d" for sk in ["A", "B", "C", "D"])
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

# Compute stratum-level sobriety rates
stratum_sobriety = {}
for sk in ["A", "B", "C", "D"]:
    pids = static[static["stratum"] == sk]["patient_id"]
    sub_panel = panel[panel["patient_id"].isin(pids)]
    stratum_sobriety[sk] = sub_panel.groupby("patient_id")["sober_this_quarter"].mean().mean()

p_d = stratum_sobriety["D"]
p_a = stratum_sobriety["A"]

# Pathway contributions (proportional to PAF)
total_paf_sum = sum(pafs.values())
gap = p_a - p_d
contributions = {}
for col, pname in zip(resource_cols, pathway_names):
    contributions[pname] = gap * (pafs[col] / total_paf_sum) if total_paf_sum > 0 else gap / 4

fig, ax = plt.subplots(figsize=(12, 7))

# Waterfall construction
running = p_d
bar_data = []
for pname in pathway_names:
    c = contributions[pname]
    bar_data.append((pname, running, c))
    running += c

# Draw bars
for i, (pname, start, delta) in enumerate(bar_data):
    color = pathway_colors[i]
    ax.barh(i, delta, left=start, color=color, edgecolor="white", height=0.6)
    ax.text(start + delta + 0.005, i, f"+{delta:.3f}", va="center", fontsize=10, fontweight="bold")

# Start and end markers
ax.barh(len(bar_data), 0.001, left=p_d, color="gray", height=0.4)
ax.text(p_d - 0.02, len(bar_data), f"Stratum D\n{p_d:.1%}", ha="right", va="center", fontsize=10)

ax.barh(-1, 0.001, left=p_a, color="gray", height=0.4)
ax.text(p_a + 0.01, -1, f"Stratum A\n{p_a:.1%}", ha="left", va="center", fontsize=10)

# Reference lines
ax.axvline(0.35, color="darkblue", linestyle=":", linewidth=1.5, alpha=0.7, label="Community midpoint (0.35)")
ax.axvline(0.926, color="darkgreen", linestyle=":", linewidth=1.5, alpha=0.7, label="Seed trajectory (0.926)")

ax.set_yticks(range(len(bar_data)))
ax.set_yticklabels([b[0] for b in bar_data], fontsize=11)
ax.set_xlabel("Sobriety rate", fontsize=12)
ax.set_title("Equity Gap Decomposition by Resource Pathway", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 1.05)
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(fig_dir / "equity_figure3_gap_decomposition.png", dpi=200)
print(f"✓ E3: {fig_dir / 'equity_figure3_gap_decomposition.png'}")

# ══════════════════════════════════════════════════════════════════════
# Figure E4: Population-attributable fraction
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
paf_vals = [pafs[col] * 100 for col in resource_cols]
y_pos = np.arange(len(pathway_names))

bars = ax.barh(y_pos, paf_vals, color=pathway_colors, edgecolor="white", height=0.6)
for i, (bar, paf) in enumerate(zip(bars, paf_vals)):
    label = f"{paf:.1f}%"
    action = {
        0: "If treatment access equalized",
        1: "If monitoring equalized",
        2: "If social support equalized",
        3: "If self-advocacy equalized",
    }[i]
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{label} — {action}", va="center", fontsize=10)

ax.set_yticks(y_pos)
ax.set_yticklabels(pathway_names, fontsize=11)
ax.set_xlabel("Population-Attributable Fraction (%)", fontsize=12)
ax.set_title("Policy Levers: Share of Equity Gap by Pathway", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "equity_figure4_paf.png", dpi=200)
print(f"✓ E4: {fig_dir / 'equity_figure4_paf.png'}")
