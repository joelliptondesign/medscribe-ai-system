"""Static service tool dispatcher over existing pipeline nodes."""

from __future__ import annotations

from typing import Any

from graph.nodes import critic, diagnosis_engine, icd_mapper, intake_parser


tools = {
    "parse_input": intake_parser.run,
    "generate_diagnosis": diagnosis_engine.run,
    "map_icd": icd_mapper.run,
    "score_case": critic.run,
}


def call_tool(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name not in tools:
        raise ValueError("Invalid tool")
    return tools[name](payload)
