"""
Feature contract — the companion <-> model interface.
======================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Everything a caller needs to turn a natural-language check-in + an enrollment
profile into the exact 40-feature vector the LSTM was trained on. The encoding
maps and the cumulative-stress recursion mirror `pipeline/build_lstm_tensors.py`
and the README so that live inputs are distributed identically to training.

Serving invariants (do not reorder / re-encode without retraining):
  * FEATURE_COLS order is the tensor's feature axis (F=40).
  * CONTINUOUS_COLS is the subset the saved StandardScaler was fit on.
  * MASK_TOKEN / MAX_WINDOWS match the training tensors.
"""

from __future__ import annotations
from dataclasses import dataclass, field

MASK_TOKEN = -999.0
MAX_WINDOWS = 34            # T — model.max_time; horizons are clipped to this
WINDOW_DAYS = 30

# ── Feature axis (must match meta['feature_cols'] exactly) ─────────────────
FEATURE_COLS = [
    # time-varying (0..12) — supplied per check-in
    "active_depression", "med_adherence", "employed", "adverse_event",
    "positive_event", "phq9_score", "gad7_score", "sober_this_window",
    "uds_positive", "meetings_per_month", "social_support_score",
    "sparse_window", "cumulative_stress",
    # static (13..39) — supplied once at enrollment, broadcast to every window
    "sud_type", "sud_severity", "prior_tx_episodes", "housing_stability",
    "social_support", "age_at_enrollment", "age_first_use",
    "mdd", "ptsd", "gad", "adhd", "bipolar_i", "bipolar_ii", "psychosis", "bpd",
    "med_hcv", "med_hiv", "med_chronic_pain", "med_tbi_history",
    "med_liver_disease", "med_cardiovascular", "med_ms_autoimmune",
    "sex", "race_ethnicity", "route_of_admin", "treatment_modality",
    "insurance_status",
]
assert len(FEATURE_COLS) == 40

# subset (and order) the StandardScaler in scaler.pkl was fit on
CONTINUOUS_COLS = [
    "sud_severity", "social_support", "med_adherence", "cumulative_stress",
    "phq9_score", "gad7_score", "social_support_score",
    "meetings_per_month", "age_at_enrollment", "age_first_use",
    "prior_tx_episodes",
]

# ── Categorical encoders (verbatim from build_lstm_tensors.py) ─────────────
SEX_MAP = {"Male": 0, "Female": 1, "Non-binary": 2}
RACE_MAP = {"White": 0, "Black": 1, "Hispanic": 2, "Asian": 3, "Other/Multiracial": 4}
ROUTE_MAP = {"IV": 0, "Smoking": 1, "Intranasal": 2, "Oral": 3, "Mixed": 4}
TX_MAP = {"IOP": 0, "Residential": 1, "MAT": 2, "Outpatient": 3, "Detox_only": 4, "None": 5}
INS_MAP = {"Medicaid": 0, "Private": 1, "Medicare": 2, "Uninsured": 3, "VA": 4}
AE_MAP = {"none": 0, "mild": 1, "severe": 2}
# cumulative-stress severity weights (README): none=0, mild=0.5, severe=1.5
AE_STRESS_WEIGHT = {"none": 0.0, "mild": 0.5, "severe": 1.5}


@dataclass
class StaticProfile:
    """Set once per user at enrollment. Categoricals are human-readable and
    encoded here so the caller never has to know the integer codes."""
    sud_type: int                  # 0=stimulant, 1=polysubstance
    sud_severity: float            # 0-10
    prior_tx_episodes: int
    housing_stability: int
    social_support: float
    age_at_enrollment: float
    age_first_use: float
    sex: str
    race_ethnicity: str
    route_of_admin: str
    treatment_modality: str
    insurance_status: str
    # dx / comorbidity flags (0/1)
    mdd: int = 0; ptsd: int = 0; gad: int = 0; adhd: int = 0
    bipolar_i: int = 0; bipolar_ii: int = 0; psychosis: int = 0; bpd: int = 0
    med_hcv: int = 0; med_hiv: int = 0; med_chronic_pain: int = 0
    med_tbi_history: int = 0; med_liver_disease: int = 0
    med_cardiovascular: int = 0; med_ms_autoimmune: int = 0

    def encoded(self) -> dict:
        return {
            "sud_type": float(self.sud_type),
            "sud_severity": float(self.sud_severity),
            "prior_tx_episodes": float(self.prior_tx_episodes),
            "housing_stability": float(self.housing_stability),
            "social_support": float(self.social_support),
            "age_at_enrollment": float(self.age_at_enrollment),
            "age_first_use": float(self.age_first_use),
            "sex": float(SEX_MAP[self.sex]),
            "race_ethnicity": float(RACE_MAP[self.race_ethnicity]),
            "route_of_admin": float(ROUTE_MAP[self.route_of_admin]),
            "treatment_modality": float(TX_MAP.get(self.treatment_modality, 5)),
            "insurance_status": float(INS_MAP[self.insurance_status]),
            "mdd": float(self.mdd), "ptsd": float(self.ptsd), "gad": float(self.gad),
            "adhd": float(self.adhd), "bipolar_i": float(self.bipolar_i),
            "bipolar_ii": float(self.bipolar_ii), "psychosis": float(self.psychosis),
            "bpd": float(self.bpd),
            "med_hcv": float(self.med_hcv), "med_hiv": float(self.med_hiv),
            "med_chronic_pain": float(self.med_chronic_pain),
            "med_tbi_history": float(self.med_tbi_history),
            "med_liver_disease": float(self.med_liver_disease),
            "med_cardiovascular": float(self.med_cardiovascular),
            "med_ms_autoimmune": float(self.med_ms_autoimmune),
        }


@dataclass
class WindowObservation:
    """One 30-day check-in, distilled from the companion conversation.
    `cumulative_stress` is intentionally NOT accepted here — the server derives
    it from the adverse/positive-event stream so it matches the training-time
    recursion and cannot be spoofed or drift out of range."""
    active_depression: int          # 0/1
    med_adherence: float            # 0-1
    employed: int                   # 0/1
    adverse_event: str              # "none" | "mild" | "severe"
    positive_event: int             # 0/1
    phq9_score: float               # 0-27
    gad7_score: float               # 0-21
    sober_this_window: int          # 0/1
    uds_positive: int               # 0/1
    meetings_per_month: float
    social_support_score: float
    sparse_window: int = 0          # 1 if the check-in was missed / imputed

    def time_varying_encoded(self, prev_stress: float) -> dict:
        """Encode this window and advance the cumulative-stress state."""
        stress = update_cumulative_stress(prev_stress, self.adverse_event, self.positive_event)
        return {
            "active_depression": float(self.active_depression),
            "med_adherence": float(self.med_adherence),
            "employed": float(self.employed),
            "adverse_event": float(AE_MAP[self.adverse_event]),
            "positive_event": float(self.positive_event),
            "phq9_score": float(self.phq9_score),
            "gad7_score": float(self.gad7_score),
            "sober_this_window": float(self.sober_this_window),
            "uds_positive": float(self.uds_positive),
            "meetings_per_month": float(self.meetings_per_month),
            "social_support_score": float(self.social_support_score),
            "sparse_window": float(self.sparse_window),
            "cumulative_stress": stress,
        }, stress


def update_cumulative_stress(prev: float, adverse_event: str, positive_event: int) -> float:
    """stress_t = 0.8*stress_{t-1} + severity_weight(ae) - 0.3*positive, clipped [0,10]."""
    s = 0.8 * prev + AE_STRESS_WEIGHT[adverse_event] - 0.3 * float(positive_event)
    return float(min(10.0, max(0.0, s)))
