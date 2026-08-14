"""
Policy layer — risk -> action, with a hard governance boundary.
================================================================
This is deliberately NOT the LLM. It is a small, auditable, deterministic map
from model output to a companion behavior decision. Rationale (see project
meshability notes): an emotionally-bonded agent must never surface raw relapse
probabilities to the user, and clinical escalation must be a reviewable rule,
not a generated sentence.

Outputs an `escalation` level the companion's behavior policy consumes, plus an
optional `clinician_alert`. Numbers stay server-side; only the *level* and
non-numeric `companion_directive` cross back toward the user experience.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# 360-day relapse risk bands. Tune on held-out calibration, not vibes.
BANDS = [
    (0.00, 0.25, "steady"),
    (0.25, 0.50, "monitor"),
    (0.50, 0.75, "outreach"),
    (0.75, 1.01, "urgent"),
]

# Non-numeric guidance the companion may act on. Never includes a probability.
DIRECTIVES = {
    "steady":   "affirm progress; normal check-in cadence",
    "monitor":  "increase warmth; gently probe recent stressors",
    "outreach": "proactive supportive outreach; surface a coping resource",
    "urgent":   "supportive outreach + route to human/clinical channel",
}


@dataclass
class Decision:
    escalation: str
    companion_directive: str
    clinician_alert: bool
    rationale: str
    # numbers retained for logging/audit only — caller must not echo to user
    _risk_360: float = field(repr=False, default=0.0)


def decide(score: dict) -> Decision:
    risk = score["risk_by_horizon_days"].get(360, 0.0)
    level = next(name for lo, hi, name in BANDS if lo <= risk < hi)

    # Attention-informed rationale: name the driving window without leaking risk.
    peak = score.get("peak_attention_window")
    n = score.get("n_observed_windows", 0)
    if peak is not None and n:
        recency = "most recent" if peak >= n - 1 else f"~{(n - 1 - peak)} check-ins ago"
        rationale = f"risk read driven by the {recency} window"
    else:
        rationale = "insufficient history for a driver read"

    return Decision(
        escalation=level,
        companion_directive=DIRECTIVES[level],
        clinician_alert=(level == "urgent"),
        rationale=rationale,
        _risk_360=risk,
    )
