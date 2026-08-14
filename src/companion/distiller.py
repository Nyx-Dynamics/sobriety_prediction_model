"""
Conversation → features distiller (companion-side adapter).
============================================================
Sobriety Prediction Model | Nyx Dynamics LLC

Turns one 30-day span of companion conversation into the structured
`WindowObservation` the model contract expects — then DISCARDS the transcript.
This is the safe way to borrow a Replika-style rich input channel: the raw
chat is read once by the extraction model in-memory and never persisted; only
the ~13 declared numeric features cross toward the serving layer (contract P1).

Runs COMPANION-SIDE, outside the serving trust boundary. It does not import
serving.* and never calls the model server. Wiring: companion collects a
window of chat → `distill_window()` → POST the returned dict to
`/sessions/{id}/observe`. The transcript stays on the companion side and is
dropped here.

Extraction uses the Anthropic SDK with structured outputs (a Pydantic schema),
so the model is forced to return the exact field set — no free-text parsing.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Default to Opus 4.8 (the resolved capable default); override per deployment.
DISTILLER_MODEL = os.environ.get("SUD_DISTILLER_MODEL", "claude-opus-4-8")

# Clamp ranges mirror the training feature space (README).
_RANGES = {
    "med_adherence": (0.0, 1.0),
    "phq9_score": (0.0, 27.0),
    "gad7_score": (0.0, 21.0),
    "meetings_per_month": (0.0, 60.0),
    "social_support_score": (0.0, 10.0),
}


class AdverseEvent(str, Enum):
    none = "none"
    mild = "mild"
    severe = "severe"


class ExtractedWindow(BaseModel):
    """The strict schema the extraction model must fill. `null` = not evidenced
    in the conversation; the caller decides whether to impute or flag sparse."""
    active_depression: Optional[bool] = Field(None, description="Depressive symptoms present this period")
    med_adherence: Optional[float] = Field(None, description="Fraction of prescribed meds taken, 0-1")
    employed: Optional[bool] = Field(None, description="Currently employed")
    adverse_event: Optional[AdverseEvent] = Field(None, description="Worst adverse life event this period")
    positive_event: Optional[bool] = Field(None, description="A meaningful positive event occurred")
    phq9_score: Optional[float] = Field(None, description="PHQ-9 depression score 0-27 if elicited")
    gad7_score: Optional[float] = Field(None, description="GAD-7 anxiety score 0-21 if elicited")
    sober_this_window: Optional[bool] = Field(None, description="Reported sober for this period")
    uds_positive: Optional[bool] = Field(None, description="A urine drug screen came back positive")
    meetings_per_month: Optional[float] = Field(None, description="Recovery meetings attended this period")
    social_support_score: Optional[float] = Field(None, description="Perceived social support 0-10")


# Fields the serving contract requires; used to compute what stayed unresolved.
_CONTRACT_FIELDS = list(ExtractedWindow.model_fields.keys())

_SYSTEM = (
    "You extract structured recovery check-in signals from a support-companion "
    "conversation. Only report a field when the conversation gives evidence for "
    "it; otherwise leave it null. Do not guess clinical scores (PHQ-9/GAD-7) "
    "unless the user actually reported symptoms you can map to them. Return "
    "signals only — never restate or store the conversation."
)


@dataclass
class DistilledWindow:
    """Result of distillation. `observation` is a WindowObservation-shaped dict
    ready to POST to /observe. `unresolved` lists fields with no evidence, so
    the caller can impute or set sparse_window=1. The transcript is NOT here."""
    observation: dict
    unresolved: list[str]
    sparse: bool          # True if too little was resolved to trust this window


def _clamp(name: str, value):
    lo, hi = _RANGES.get(name, (None, None))
    if lo is None or value is None:
        return value
    return float(min(hi, max(lo, value)))


def distill_window(conversation: str, *, client=None,
                   sparse_threshold: int = 6) -> DistilledWindow:
    """Extract a WindowObservation from a window of conversation text.

    `conversation` is transcript text (companion-side). It is sent to the
    extraction model in-memory and then dropped — this function returns only
    the structured observation, never the input. `client` is an optional
    `anthropic.Anthropic` (injected for tests); constructed on demand otherwise.
    """
    if client is None:
        import anthropic  # local import so the module loads without the SDK
        client = anthropic.Anthropic()

    # Structured output: the model is constrained to the ExtractedWindow schema.
    resp = client.messages.parse(
        model=DISTILLER_MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": conversation}],
        output_format=ExtractedWindow,
    )
    extracted: ExtractedWindow = resp.parsed_output
    # `conversation` and `resp` are intentionally not retained past this point.

    return finalize(extracted, sparse_threshold=sparse_threshold)


def finalize(extracted: ExtractedWindow, *, sparse_threshold: int = 6) -> DistilledWindow:
    """Pure post-processing (no I/O) — encode, clamp, and flag sparsity.
    Split out so tests can exercise it without an API call."""
    data = extracted.model_dump()
    unresolved = [f for f in _CONTRACT_FIELDS if data.get(f) is None]

    obs = {
        "active_depression": int(bool(data["active_depression"])) if data["active_depression"] is not None else 0,
        "med_adherence": _clamp("med_adherence", data["med_adherence"]) if data["med_adherence"] is not None else 0.0,
        "employed": int(bool(data["employed"])) if data["employed"] is not None else 0,
        "adverse_event": (data["adverse_event"].value if isinstance(data["adverse_event"], AdverseEvent)
                          else (data["adverse_event"] or "none")),
        "positive_event": int(bool(data["positive_event"])) if data["positive_event"] is not None else 0,
        "phq9_score": _clamp("phq9_score", data["phq9_score"]) if data["phq9_score"] is not None else 0.0,
        "gad7_score": _clamp("gad7_score", data["gad7_score"]) if data["gad7_score"] is not None else 0.0,
        "sober_this_window": int(bool(data["sober_this_window"])) if data["sober_this_window"] is not None else 0,
        "uds_positive": int(bool(data["uds_positive"])) if data["uds_positive"] is not None else 0,
        "meetings_per_month": _clamp("meetings_per_month", data["meetings_per_month"]) if data["meetings_per_month"] is not None else 0.0,
        "social_support_score": _clamp("social_support_score", data["social_support_score"]) if data["social_support_score"] is not None else 0.0,
    }
    # Too little evidence → mark the window sparse so the model down-weights it.
    sparse = len(unresolved) >= sparse_threshold
    obs["sparse_window"] = 1 if sparse else 0
    return DistilledWindow(observation=obs, unresolved=unresolved, sparse=sparse)
