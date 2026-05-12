from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

from evaluation.langsmith_experiment_runner import _hosted_denial_evaluator, denial_final_route_present
from graph.denial_graph import run_denial_graph
from graph.llm_client import HybridLLMError, _extract_provider_metadata
from graph.nodes.denials import documentation_gap_analyzer, governance_policy, recoverability_analyzer, routing_engine
from graph.operational_alerts import build_layer1_payload


ROUTING_ACTIONS = {"APPEAL", "RESUBMIT", "WRITE_OFF", "ESCALATE"}
GOVERNANCE_POSTURES = {"SUPPORTED", "LOW_CONFIDENCE", "LOW_EVIDENCE", "AMBIGUOUS", "HIGH_RISK"}
BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "denial_benchmark_cases.json"


def test_run_denial_graph_imports_successfully() -> None:
    assert callable(run_denial_graph)


def test_representative_appeal_case_routes_to_appeal() -> None:
    result = run_denial_graph(
        {
            "payer_reason": "Denied for medical necessity criteria not met.",
            "clinical_summary": "Synthetic service is supported by exam, imaging, assessment, plan, and rationale.",
            "documentation_summary": "Documentation is consistent and available.",
        }
    )
    assert result["routing_action"] == "APPEAL"


def test_recoverable_documentation_gap_routes_to_resubmit() -> None:
    result = run_denial_graph(
        {
            "payer_reason": "Denied for missing documentation.",
            "clinical_summary": "Synthetic service has clinical support.",
            "documentation_summary": "The note was omitted but can provide and attach with resubmission.",
        }
    )
    assert result["routing_action"] == "RESUBMIT"


def test_ambiguous_case_routes_to_escalate() -> None:
    result = run_denial_graph(
        {
            "payer_reason": "Denied for unclear and conflicting rationale.",
            "clinical_summary": "Synthetic service context is inconsistent.",
            "documentation_summary": "Ambiguous documentation requires review.",
        }
    )
    assert result["routing_action"] == "ESCALATE"


def test_low_recoverability_low_evidence_can_route_to_write_off() -> None:
    result = run_denial_graph(
        {
            "payer_reason": "Denied for no evidence and not documented support.",
            "clinical_summary": "Synthetic service has no supporting rationale.",
            "documentation_summary": "Low evidence packet with absent documentation.",
        }
    )
    assert result["routing_action"] == "WRITE_OFF"


def test_governance_posture_present_and_distinct_from_routing_action() -> None:
    result = run_denial_graph(
        {
            "payer_reason": "Denied for medical necessity.",
            "clinical_summary": "Synthetic service is supported by exam and plan.",
            "documentation_summary": "Documentation is consistent and available.",
        }
    )
    assert result["governance_posture"] in GOVERNANCE_POSTURES
    assert result["governance_posture"] != result["routing_action"]


def test_node_diagnostics_present() -> None:
    result = run_denial_graph({"payer_reason": "Denied for missing documentation."})
    assert len(result["node_diagnostics"]) == 6
    assert all("node_name" in diagnostic for diagnostic in result["node_diagnostics"])


def test_input_payload_is_not_mutated_in_place() -> None:
    payload = {
        "payer_reason": "Denied for missing documentation.",
        "clinical_summary": "Synthetic service has support.",
        "documentation_summary": "Missing attachment can provide.",
    }
    original = deepcopy(payload)
    run_denial_graph(payload)
    assert payload == original


def test_benchmark_cases_are_synthetic_and_non_phi() -> None:
    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert 24 <= len(cases) <= 36
    for case in cases:
        assert "Synthetic" in case["clinical_summary"] or "synthetic" in case["clinical_summary"]
        assert case["metadata"]["phi"] is False


def test_benchmark_expected_vocabularies_are_bounded() -> None:
    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert {case["expected_routing_action"] for case in cases} <= ROUTING_ACTIONS
    assert {case["expected_governance_posture"] for case in cases} <= GOVERNANCE_POSTURES


def test_routing_and_governance_have_no_llm_dependency() -> None:
    assert "invoke_json" not in inspect.getsource(routing_engine)
    assert "invoke_json" not in inspect.getsource(governance_policy)


def test_hybrid_denial_nodes_fallback_when_provider_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("MEDSCRIBE_EXECUTION_MODE", "hybrid")

    def fail_provider(*args, **kwargs):
        raise HybridLLMError("client_call_failure", "forced_for_test")

    monkeypatch.setattr(documentation_gap_analyzer, "invoke_json", fail_provider)
    monkeypatch.setattr(recoverability_analyzer, "invoke_json", fail_provider)

    result = run_denial_graph(
        {
            "payer_reason": "Denied for missing documentation.",
            "clinical_summary": "Synthetic service has clinical support.",
            "documentation_summary": "The note was omitted but can provide and attach with resubmission.",
            "metadata": {"phi": False},
        }
    )

    assert result["routing_action"] in ROUTING_ACTIONS
    assert result["governance_posture"] in GOVERNANCE_POSTURES
    assert result["fallback_used"] is True
    assert result["degraded_mode"] is True
    assert result["error"] is None
    assert any(alert["class"] == "fallback_used" for alert in result["alerts"])
    hybrid_nodes = {
        diagnostic["node_name"]: diagnostic
        for diagnostic in result["node_diagnostics"]
        if diagnostic["node_name"] in {"documentation_gap_analyzer", "recoverability_analyzer"}
    }
    assert hybrid_nodes["documentation_gap_analyzer"]["fallback_triggered"] is True
    assert hybrid_nodes["recoverability_analyzer"]["fallback_triggered"] is True


def test_invalid_llm_schema_triggers_documentation_gap_fallback(monkeypatch) -> None:
    monkeypatch.setenv("MEDSCRIBE_EXECUTION_MODE", "hybrid")

    def invalid_documentation_response(*args, **kwargs):
        return {"missing_evidence": "yes", "ambiguity_level": "extreme"}

    def valid_recoverability_response(*args, **kwargs):
        return {
            "recoverability": "partially recoverable",
            "recoverability_factors": ["synthetic missing attachment"],
            "uncertainty": "medium",
            "rationale": "Recoverability depends on missing documentation.",
        }

    monkeypatch.setattr(documentation_gap_analyzer, "invoke_json", invalid_documentation_response)
    monkeypatch.setattr(recoverability_analyzer, "invoke_json", valid_recoverability_response)

    result = run_denial_graph(
        {
            "payer_reason": "Denied for missing documentation.",
            "clinical_summary": "Synthetic service has clinical support.",
            "documentation_summary": "The note was omitted but can provide and attach with resubmission.",
            "metadata": {"phi": False},
        }
    )

    docs_diag = next(
        diagnostic
        for diagnostic in result["node_diagnostics"]
        if diagnostic["node_name"] == "documentation_gap_analyzer"
    )
    recoverability_diag = next(
        diagnostic
        for diagnostic in result["node_diagnostics"]
        if diagnostic["node_name"] == "recoverability_analyzer"
    )
    assert docs_diag["fallback_triggered"] is True
    assert recoverability_diag["hybrid_interpretation_used"] is True
    assert result["routing_action"] in ROUTING_ACTIONS
    assert result["governance_posture"] in GOVERNANCE_POSTURES


def test_latency_warning_does_not_populate_runtime_error() -> None:
    layer1 = build_layer1_payload(
        workflow="denial",
        status="completed",
        output="route=APPEAL posture=SUPPORTED status=completed",
        metadata={
            "workflow": "denial",
            "contains_phi": False,
            "latency_ms": 6000,
            "token_cost_available": False,
        },
        record={"routing_action": "APPEAL", "governance_posture": "SUPPORTED"},
    )

    assert layer1["status"] == "completed"
    assert layer1["error"] is None
    assert any(alert["class"] == "latency_spike" and alert["severity"] == "warning" for alert in layer1["alerts"])


def test_degraded_warning_can_complete_without_runtime_error() -> None:
    layer1 = build_layer1_payload(
        workflow="denial",
        status="degraded",
        output="route=RESUBMIT posture=LOW_CONFIDENCE status=degraded",
        metadata={
            "workflow": "denial",
            "contains_phi": False,
            "latency_ms": 25,
            "token_cost_available": False,
            "fallback_used": True,
            "degraded_mode": True,
        },
        record={
            "routing_action": "RESUBMIT",
            "governance_posture": "LOW_CONFIDENCE",
            "fallback_used": True,
            "degraded_mode": True,
        },
    )

    assert layer1["status"] == "degraded"
    assert layer1["error"] is None
    assert {alert["class"] for alert in layer1["alerts"]} >= {"fallback_used", "degraded_mode"}


def test_hosted_denial_evaluator_does_not_emit_warning_alert_as_comment() -> None:
    case = {
        "case_id": "synthetic-evaluator-case",
        "title": "Synthetic evaluator case",
        "denial_type": "medical necessity",
        "payer_reason": "Synthetic denial reason.",
        "clinical_summary": "Synthetic clinical support.",
        "documentation_summary": "Synthetic documentation support.",
        "operational_theme": "alert evaluator separation",
        "expected_routing_action": "APPEAL",
        "expected_governance_posture": "SUPPORTED",
        "metadata": {"phi": False},
    }

    class FakeRun:
        outputs = {
            "output": {
                "record": {
                    "status": "completed",
                    "routing_action": "APPEAL",
                    "governance_posture": "SUPPORTED",
                    "node_diagnostics": [{"node_name": "routing_engine"}],
                },
                "status": "completed",
                "error": "latency_spike",
                "alerts": [{"class": "latency_spike", "severity": "warning"}],
            }
        }

    class FakeExample:
        metadata = {"case_id": case["case_id"]}
        inputs = {}
        outputs = {}

    evaluator = _hosted_denial_evaluator(
        "denial_final_route_present",
        {case["case_id"]: case},
        denial_final_route_present,
    )
    result = evaluator(FakeRun(), FakeExample())

    assert result["score"] is True
    assert result["comment"] == "route=APPEAL posture=SUPPORTED"
    assert result["comment"] != "latency_spike"


def test_provider_metadata_extraction_handles_absent_and_present_usage() -> None:
    class EmptyResponse:
        usage_metadata = None
        response_metadata = {}

    empty = _extract_provider_metadata(EmptyResponse())
    assert empty["usage_metadata_available"] is False
    assert "token_usage" not in empty

    class UsageResponse:
        usage_metadata = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
        response_metadata = {"model_name": "synthetic-model"}

    usage = _extract_provider_metadata(UsageResponse())
    assert usage["model_name"] == "synthetic-model"
    assert usage["token_usage"]["prompt_tokens"] == 12
    assert usage["token_usage"]["completion_tokens"] == 8
    assert usage["token_usage"]["total_tokens"] == 20
