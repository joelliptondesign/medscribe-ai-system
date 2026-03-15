"""Small LLM client helper for MED-SCRIBE hybrid mode."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.config import get_model_name


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
        max_retries=1,
        timeout=20,
    )


def invoke_json(node_name: str, prompt_text: str, payload: dict[str, Any], diagnostic: dict[str, Any] | None = None) -> dict[str, Any]:
    if diagnostic is not None:
        diagnostic["live_call_attempted"] = True

    try:
        model = get_chat_model()
        response = model.invoke(
            [
                SystemMessage(content=prompt_text),
                HumanMessage(
                    content=(
                        f"Node name: {node_name}\n"
                        "Return valid JSON only with no markdown fences or commentary.\n"
                        f"Input payload: {json.dumps(payload, ensure_ascii=True)}"
                    )
                ),
            ]
        )
    except Exception as exc:
        if diagnostic is not None:
            diagnostic["fallback_reason"] = f"client_call_failure:{exc.__class__.__name__}"
        raise HybridLLMError("client_call_failure", exc.__class__.__name__) from exc

    if diagnostic is not None:
        diagnostic["live_call_returned"] = True

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
