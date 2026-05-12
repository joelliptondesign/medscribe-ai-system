"""Deterministic intake parsing for synthetic denial-management cases."""

from __future__ import annotations

from typing import Any


DOCUMENTATION_KEYWORDS = {
    "prior_authorization": ("prior authorization", "authorization", "preauth"),
    "medical_necessity": ("medical necessity", "necessity", "criteria", "guideline"),
    "clinical_evidence": ("exam", "assessment", "plan", "imaging", "lab", "trend", "therapy", "rationale"),
    "modifier": ("modifier", "distinct service", "same-day", "separate service"),
    "coding": ("icd", "cpt", "code", "coding", "diagnosis code"),
    "missing": ("missing", "not included", "omitted", "absent", "not documented", "insufficient"),
    "conflict": ("conflict", "conflicting", "inconsistent", "contradict", "does not match"),
    "partial_support": ("partial", "weak but non-zero", "some support", "limited support", "borderline"),
    "timeline_issue": ("timeline", "out of sequence", "before authorization", "after denial", "inconsistent dates"),
    "specialist": ("specialist", "orthopedics", "cardiology", "neurology", "radiology"),
    "conservative_treatment": ("conservative treatment", "physical therapy", "trial", "home exercise", "medication trial"),
    "high_risk": ("high risk", "urgent", "safety-sensitive", "specialist-level"),
}
EVIDENCE_FIELDS = (
    "imaging_summary",
    "lab_summary",
    "treatment_history",
    "medication_context",
    "prior_authorization_context",
    "utilization_review_notes",
    "specialist_notes",
    "timeline_flags",
    "conflicting_documentation",
    "evidence_strength",
    "missing_required_evidence",
    "conservative_treatment_history",
    "medical_necessity_rationale",
    "coding_specificity_flags",
)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _source_text(payload: dict[str, Any]) -> str:
    parts = [
        _text(payload.get("payer_reason")),
        _text(payload.get("clinical_summary")),
        _text(payload.get("documentation_summary")),
        _text(payload.get("input_text")),
    ]
    parts.extend(_text(payload.get(field)) for field in EVIDENCE_FIELDS)
    return " ".join(part for part in parts if part)


def _signals(payload: dict[str, Any]) -> dict[str, bool]:
    combined = _source_text(payload).lower()
    signals = {
        signal: any(keyword in combined for keyword in keywords)
        for signal, keywords in DOCUMENTATION_KEYWORDS.items()
    }
    signals["structured_evidence_present"] = any(bool(_text(payload.get(field))) for field in EVIDENCE_FIELDS)
    signals["missing_required_evidence"] = bool(_text(payload.get("missing_required_evidence")))
    signals["conflicting_documentation"] = bool(_text(payload.get("conflicting_documentation"))) or signals["conflict"]
    return signals


def _evidence_profile(payload: dict[str, Any]) -> dict[str, Any]:
    strength = _text(payload.get("evidence_strength")).lower() or "unspecified"
    if strength not in {"strong", "moderate", "partial", "weak", "low", "conflicting", "unspecified"}:
        strength = "unspecified"
    present_fields = [field for field in EVIDENCE_FIELDS if _text(payload.get(field))]
    combined = _source_text(payload).lower()
    return {
        "present_fields": present_fields,
        "evidence_strength": strength,
        "has_imaging": bool(_text(payload.get("imaging_summary"))),
        "has_labs": bool(_text(payload.get("lab_summary"))),
        "has_treatment_history": bool(_text(payload.get("treatment_history")) or _text(payload.get("conservative_treatment_history"))),
        "has_prior_authorization_context": bool(_text(payload.get("prior_authorization_context"))),
        "has_utilization_review": bool(_text(payload.get("utilization_review_notes"))),
        "has_specialist_notes": bool(_text(payload.get("specialist_notes"))),
        "has_timeline_flags": bool(_text(payload.get("timeline_flags"))),
        "has_conflicting_documentation": bool(_text(payload.get("conflicting_documentation"))) or "conflict" in combined,
        "has_missing_required_evidence": bool(_text(payload.get("missing_required_evidence"))),
        "has_partial_support": any(term in combined for term in ("partial", "some support", "limited support", "weak but non-zero", "borderline")),
        "has_conservative_treatment": bool(_text(payload.get("conservative_treatment_history"))) or "conservative treatment" in combined,
    }


def run(state: dict[str, Any]) -> dict[str, Any]:
    updated = dict(state)
    input_text = _text(updated.get("input_text"))
    updated["denial_reason"] = _text(updated.get("payer_reason")) or input_text
    updated["service_context"] = _text(updated.get("clinical_summary")) or input_text
    updated["documentation_context"] = _text(updated.get("documentation_summary")) or input_text
    updated["documentation_signals"] = _signals(updated)
    updated["evidence_profile"] = _evidence_profile(updated)
    return updated
