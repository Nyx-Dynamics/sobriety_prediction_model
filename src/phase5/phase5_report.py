"""
Phase 5: Structural Equity Audit Report
=========================================
Prints structured summary and saves equity_audit_results.json.

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import json
import numpy as np
import pandas as pd

# ── Load all Phase 5 outputs ──────────────────────────────────────────
data_dir = config.PROJECT_ROOT / "data" / "phase5" / "stratified"
model_dir = config.PROJECT_ROOT / "models" / "phase5"
out_dir = config.PROJECT_ROOT / "outputs" / "phase5"
out_dir.mkdir(parents=True, exist_ok=True)

static = pd.read_csv(data_dir / "patient_static_stratified.csv")
panel = pd.read_csv(data_dir / "patient_panel_stratified.csv")

with open(model_dir / "pathway_results.json") as f:
    results = json.load(f)

gamma_mean = np.array(results["gamma_mean"])
gamma_lo = np.array(results["gamma_hdi_lo"])
gamma_hi = np.array(results["gamma_hdi_hi"])
pathway_names = results["pathway_names"]
resource_cols = results["resource_cols"]
pafs = results["pafs"]
R_std = np.array(results["R_std"])
resource_ranges_raw = np.array(results["resource_ranges_raw"])
resource_ranges_z = resource_ranges_raw / R_std

# ── Compute stratum-level stats ───────────────────────────────────────
stratum_stats = {}
for sk in ["A", "B", "C", "D"]:
    pids = static[static["stratum"] == sk]["patient_id"]
    sub = panel[panel["patient_id"].isin(pids)]
    sobriety = sub.groupby("patient_id")["sober_this_quarter"].mean().mean()

    # Relapse rate
    relapse_n = 0
    for pid in pids:
        pt = sub[sub["patient_id"] == pid].sort_values("quarter")
        ever_sober = False
        for _, row in pt.iterrows():
            if row["sober_this_quarter"] == 1:
                ever_sober = True
            elif ever_sober and row["uds_positive"] == 1:
                relapse_n += 1
                break
        if not ever_sober and (pt["uds_positive"] == 1).any():
            relapse_n += 1

    stratum_stats[sk] = {
        "n": len(pids),
        "relapse_rate": relapse_n / len(pids),
        "sobriety_rate": sobriety,
    }

# ── HRs ───────────────────────────────────────────────────────────────
hrs = np.exp(gamma_mean * resource_ranges_z)
hr_los = np.exp(gamma_lo * resource_ranges_z)
hr_his = np.exp(gamma_hi * resource_ranges_z)

# ── Equity gap ────────────────────────────────────────────────────────
p_a = stratum_stats["A"]["sobriety_rate"]
p_d = stratum_stats["D"]["sobriety_rate"]
equity_ratio = p_a / max(p_d, 0.01)
total_paf = sum(pafs.values())

# Find top pathway
top_pathway_idx = max(range(len(resource_cols)), key=lambda k: pafs[resource_cols[k]])
top_pathway = pathway_names[top_pathway_idx]
top_paf = pafs[resource_cols[top_pathway_idx]]

# ── Print report ──────────────────────────────────────────────────────
print()
print("═" * 60)
print("PHASE 5: STRUCTURAL EQUITY AUDIT")
print("Sobriety Prediction Model | Nyx Dynamics LLC")
print("═" * 60)

print(f"\n  Cohort: N={len(static):,} stratified synthetic patients "
      f"(4 strata × {len(static)//4:,})")

print(f"\n  Relapse rates by stratum:")
labels = {"A": "High resource", "B": "Moderate", "C": "Low", "D": "Minimal resource"}
for sk in ["A", "B", "C", "D"]:
    s = stratum_stats[sk]
    print(f"    Stratum {sk} ({labels[sk]:>20}): {s['relapse_rate']:>6.1%}  "
          f"(sobriety: {s['sobriety_rate']:.1%})")

print(f"\n  Pathway attribution (posterior means, 94% HDI):")
for k in range(len(resource_cols)):
    print(f"    {pathway_names[k]:>25}: gamma = {gamma_mean[k]:>6.3f} "
          f"[{gamma_lo[k]:.3f}, {gamma_hi[k]:.3f}]  HR = {hrs[k]:.2f}")

print(f"\n  Population-attributable fractions:")
for k in range(len(resource_cols)):
    paf_pct = pafs[resource_cols[k]] * 100
    print(f"    {pathway_names[k]:>25}: {paf_pct:>5.1f}% of equity gap")
print(f"    {'Total explained':>25}: {total_paf*100:>5.1f}%")

print(f"\n  Equity gap:")
print(f"    Seed trajectory (p* = 0.926) / Stratum D (p* = {p_d:.2f})")
print(f"    Observed ratio: {equity_ratio:.1f}x")
print(f"    Explained by pathways: {total_paf*100:.1f}%")

print(f"\n  Policy implication:")
print(f"    The single largest lever for closing the equity gap is")
print(f"    {top_pathway} (PAF = {top_paf:.1%}).")

print("═" * 60)

# ── Save JSON ─────────────────────────────────────────────────────────
audit_results = {
    "cohort_n": len(static),
    "strata": {sk: stratum_stats[sk] for sk in ["A", "B", "C", "D"]},
    "pathway_attribution": {
        pathway_names[k]: {
            "gamma_mean": round(gamma_mean[k], 4),
            "gamma_hdi_94": [round(gamma_lo[k], 4), round(gamma_hi[k], 4)],
            "hazard_ratio": round(float(hrs[k]), 3),
            "hr_hdi_94": [round(float(hr_los[k]), 3), round(float(hr_his[k]), 3)],
            "paf": round(pafs[resource_cols[k]], 4),
        }
        for k in range(len(resource_cols))
    },
    "equity_gap": {
        "stratum_a_sobriety": round(p_a, 3),
        "stratum_d_sobriety": round(p_d, 3),
        "seed_trajectory": 0.926,
        "observed_ratio": round(equity_ratio, 2),
        "total_paf": round(total_paf, 4),
    },
    "policy_top_lever": top_pathway,
    "policy_top_paf": round(top_paf, 4),
}

with open(out_dir / "equity_audit_results.json", "w") as f:
    json.dump(audit_results, f, indent=2)
print(f"\n✓ Saved: {out_dir / 'equity_audit_results.json'}")

print("""
── Dataset Citations ──────────────────────────────────────
PRIMARY DATASET:
  United States DHHS. SAMHSA. CSAT. National Treatment
  Improvement Evaluation Study (NTIES), 1992-1997.
  ICPSR 2884. https://doi.org/10.3886/ICPSR02884.v4

VALIDATION DATASET:
  United States DHHS. NIH. NIDA. Drug Abuse Treatment
  Outcome Study (DATOS), 1991-1994. ICPSR 2258.
  https://doi.org/10.3886/ICPSR02258.v5

SECONDARY VALIDATION:
  California Department of Alcohol and Drug Programs.
  CALDATA, 1991-1993. ICPSR 2295.
  https://doi.org/10.3886/ICPSR02295.v1

NOTE: Per ICPSR Terms of Use (accepted 2026-03-19),
these data are used solely for statistical analysis
and aggregated reporting. No individual re-identification.
Publications using these data must send citations to:
bibliography@icpsr.umich.edu
""")
