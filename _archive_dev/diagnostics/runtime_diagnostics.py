"""Startup diagnostics for comparing entrypoint runtime environments."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any

from graph.config import get_execution_mode, get_model_name
from graph.llm_client import get_chat_model


def _mask_key(value: str) -> str | None:
    key = value.strip()
    if not key:
        return None
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _sanitize_message(exc: Exception) -> str:
    text = str(exc).strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and api_key in text:
        return exc.__class__.__name__
    return text or exc.__class__.__name__


def collect_startup_diagnostics(repo_root: Path | None = None) -> dict[str, Any]:
    host = "api.openai.com"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    diagnostics: dict[str, Any] = {
        "interpreter_path": sys.executable,
        "cwd": os.getcwd(),
        "repo_root": str(repo_root) if repo_root is not None else None,
        "execution_mode": get_execution_mode(),
        "model_name": get_model_name(),
        "openai_api_key_present": bool(api_key),
        "openai_api_key_masked": _mask_key(api_key),
        "dns_resolution_result": {
            "attempted": True,
            "succeeded": False,
            "error_type": None,
        },
        "direct_client_preflight_result": {
            "attempted": True,
            "succeeded": False,
        },
        "direct_client_preflight_error_type": None,
        "direct_client_preflight_error_message": None,
    }

    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except Exception as exc:
        diagnostics["dns_resolution_result"]["error_type"] = exc.__class__.__name__
    else:
        diagnostics["dns_resolution_result"]["succeeded"] = True

    try:
        model = get_chat_model()
        response = model.invoke('Return exactly this JSON object and nothing else: {"status":"ok"}')
        content = response.content
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        else:
            text = str(content).strip()
        if not text:
            raise ValueError("empty_content")
    except Exception as exc:
        diagnostics["direct_client_preflight_error_type"] = exc.__class__.__name__
        diagnostics["direct_client_preflight_error_message"] = _sanitize_message(exc)
    else:
        diagnostics["direct_client_preflight_result"]["succeeded"] = True

    return diagnostics
