"""LangGraph builder for the deterministic MED-SCRIBE demo pipeline."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.nodes import critic, diagnosis_engine, governance_policy, icd_mapper, intake_parser, triage_engine
from graph.state import MedScribeState


def final_formatter(state: dict[str, Any]) -> dict[str, Any]:
    governance_result = state.get("governance_result", {})
    final_status = governance_result.get("final_status", "FAIL")
    return {
        "final_output": {
            "status": final_status,
            "escalation_required": governance_result.get("escalation_required", state.get("escalation_required", False)),
            "policy_version": governance_result.get("policy_version", ""),
            "governance_version": governance_result.get("governance_version", ""),
            "reason_codes": governance_result.get("reason_codes", []),
            "summary": (
                state.get("critic_review", {}).get("summary", "")
                if final_status == "PASS"
                else "Escalation required due to policy thresholds."
            ),
        }
    }


def build_graph():
    graph = StateGraph(MedScribeState)
    graph.add_node("intake_parser", intake_parser.run)
    graph.add_node("triage_engine", triage_engine.run)
    graph.add_node("diagnosis_engine", diagnosis_engine.run)
    graph.add_node("icd_mapper", icd_mapper.run)
    graph.add_node("critic", critic.run)
    graph.add_node("governance_policy", governance_policy.run)
    graph.add_node("final_formatter", final_formatter)

    graph.add_edge(START, "intake_parser")
    graph.add_edge("intake_parser", "triage_engine")
    graph.add_edge("triage_engine", "diagnosis_engine")
    graph.add_edge("diagnosis_engine", "icd_mapper")
    graph.add_edge("icd_mapper", "critic")
    graph.add_edge("critic", "governance_policy")
    graph.add_edge("governance_policy", "final_formatter")
    graph.add_edge("final_formatter", END)
    return graph.compile()
