"""Small LLM client helper for MED-SCRIBE hybrid mode."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.config import get_model_max_tokens, get_model_name
from graph.tracing import trace_span


class HybridLLMError(RuntimeError):
    """Structured hybrid execution error without secret-bearing context."""

    def __init__(self, failure_mode: str, detail: str = "") -> None:
        super().__init__(detail or failure_mode)
        self.failure_mode = failure_mode
        self.detail = detail or failure_mode


def get_chat_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for MED-SCRIBE hybrid mode")

    return ChatOpenAI(
        api_key=api_key,
        model=get_model_name(),
        temperature=0,
        max_tokens=get_model_max_tokens(),
        max_retries=1,
        timeout=20,
    )


def _dict_like(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    return {}


def _extract_provider_metadata(response: Any) -> dict[str, Any]:
    usage_metadata = _dict_like(getattr(response, "usage_metadata", None))
    response_metadata = _dict_like(getattr(response, "response_metadata", None))
    token_usage = _dict_like(response_metadata.get("token_usage"))

    token_data: dict[str, Any] = {}
    for source in (token_usage, usage_metadata):
        for source_key, target_key in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_token_count", "total_tokens"),
        ):
            if source_key in source and source.get(source_key) is not None:
                token_data[target_key] = source.get(source_key)

    model_name = response_metadata.get("model_name") or response_metadata.get("model") or get_model_name()
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "response_metadata_available": bool(response_metadata),
        "usage_metadata_available": bool(usage_metadata),
    }
    if token_data:
        metadata["token_usage"] = token_data
    if response_metadata:
        metadata["response_metadata"] = response_metadata
    for key in ("first_token_latency_ms", "time_to_first_token_ms", "ttft_ms", "estimated_cost_usd", "cost_usd"):
        if key in response_metadata:
            metadata[key] = response_metadata[key]
    return metadata


def invoke_json(node_name: str, prompt_text: str, payload: dict[str, Any], diagnostic: dict[str, Any] | None = None) -> dict[str, Any]:
    if diagnostic is not None:
        diagnostic["live_call_attempted"] = True

    try:
        model = get_chat_model()
        messages = [
            SystemMessage(content=prompt_text),
            HumanMessage(
                content=(
                    f"Node name: {node_name}\n"
                    "Return valid JSON only with no markdown fences or commentary.\n"
                    f"Input payload: {json.dumps(payload, ensure_ascii=True)}"
                )
            ),
        ]
        config = {
            "tags": ["medscribe", f"stage:{node_name}", "hybrid-llm"],
            "metadata": {
                "node_name": node_name,
                "model": get_model_name(),
                "temperature": 0,
                "max_tokens": get_model_max_tokens(),
                "stabilization_profile": "latency_token_concise_v2",
            },
        }
        with trace_span(
            f"medscribe.denial.{node_name}.llm",
            run_type="llm",
            inputs={
                "node_name": node_name,
                "payload_keys": sorted(str(key) for key in payload),
                "payload_field_count": len(payload),
            },
            metadata={
                "workflow": "denial",
                "node_name": node_name,
                "trace_type": "llm_provider",
                "model": get_model_name(),
                "provider": "openai",
            },
            tags=["medscribe", "workflow:denial", f"stage:{node_name}", "hybrid-llm"],
        ) as llm_span:
            response = model.invoke(messages, config=config)
            provider_metadata = _extract_provider_metadata(response)
            llm_span.set_outputs(
                {
                    "node_name": node_name,
                    "response_metadata_available": provider_metadata.get("response_metadata_available"),
                    "usage_metadata_available": provider_metadata.get("usage_metadata_available"),
                    "token_usage_available": bool(provider_metadata.get("token_usage")),
                }
            )
    except Exception as exc:
        if diagnostic is not None:
            diagnostic["fallback_reason"] = f"client_call_failure:{exc.__class__.__name__}"
        raise HybridLLMError("client_call_failure", exc.__class__.__name__) from exc

    if diagnostic is not None:
        diagnostic["live_call_returned"] = True
        diagnostic["provider_metadata"] = provider_metadata
        diagnostic["token_cost_available"] = bool(provider_metadata.get("token_usage") or provider_metadata.get("estimated_cost_usd") or provider_metadata.get("cost_usd"))
        if not diagnostic["token_cost_available"]:
            diagnostic["token_cost_unavailable_reason"] = "provider_metadata_unavailable"

    content = response.content
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = str(content)

    cleaned = text.strip()
    if not cleaned:
        if diagnostic is not None:
            diagnostic["fallback_reason"] = "empty_response"
        raise HybridLLMError("empty_response")

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        failure_mode = "json_parse_failure" if cleaned.startswith("{") or cleaned.startswith("[") else "non_json_response"
        if diagnostic is not None:
            diagnostic["fallback_reason"] = failure_mode
        raise HybridLLMError(failure_mode, exc.__class__.__name__) from exc

    if not isinstance(data, dict):
        if diagnostic is not None:
            diagnostic["fallback_reason"] = "contract_rejection"
        raise HybridLLMError("contract_rejection", f"{node_name} returned a non-object JSON payload")

    if diagnostic is not None:
        diagnostic["parse_succeeded"] = True
    return data
