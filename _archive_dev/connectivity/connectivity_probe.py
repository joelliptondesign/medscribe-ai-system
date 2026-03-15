"""Standalone connectivity probe for DNS, TLS, HTTPS, and OpenAI SDK layers."""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request

from dotenv import load_dotenv
from openai import OpenAI


def mask_key(value: str) -> str | None:
    key = value.strip()
    if not key:
        return None
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def sanitize_message(exc: Exception) -> str:
    text = str(exc).strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and api_key in text:
        return exc.__class__.__name__
    return text or exc.__class__.__name__


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("MEDSCRIBE_MODEL", "").strip() or None
    host = "api.openai.com"

    artifact: dict[str, Any] = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interpreter_path": sys.executable,
        "cwd": os.getcwd(),
        "openai_api_key_present": bool(api_key),
        "openai_api_key_masked": mask_key(api_key),
        "model_name": model_name,
        "dns_probe": {
            "succeeded": False,
            "resolved_addresses": [],
            "error_type": None,
            "error_message": None,
        },
        "tcp_tls_probe": {
            "succeeded": False,
            "error_type": None,
            "error_message": None,
            "peer_name": None,
        },
        "raw_https_probe": {
            "succeeded": False,
            "http_status": None,
            "error_type": None,
            "error_message": None,
        },
        "sdk_construction_probe": {
            "succeeded": False,
            "error_type": None,
            "error_message": None,
        },
        "sdk_call_probe": {
            "succeeded": False,
            "error_type": None,
            "error_message": None,
        },
    }

    try:
        results = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        artifact["dns_probe"]["succeeded"] = True
        artifact["dns_probe"]["resolved_addresses"] = sorted({item[4][0] for item in results})
    except Exception as exc:
        artifact["dns_probe"]["error_type"] = exc.__class__.__name__
        artifact["dns_probe"]["error_message"] = sanitize_message(exc)

    try:
        raw_sock = socket.create_connection((host, 443), timeout=10)
        try:
            with ssl.create_default_context().wrap_socket(raw_sock, server_hostname=host) as tls_sock:
                artifact["tcp_tls_probe"]["succeeded"] = True
                peer = tls_sock.getpeername()
                artifact["tcp_tls_probe"]["peer_name"] = f"{peer[0]}:{peer[1]}" if isinstance(peer, tuple) else str(peer)
        finally:
            try:
                raw_sock.close()
            except Exception:
                pass
    except Exception as exc:
        artifact["tcp_tls_probe"]["error_type"] = exc.__class__.__name__
        artifact["tcp_tls_probe"]["error_message"] = sanitize_message(exc)

    try:
        req = request.Request("https://api.openai.com/v1/models", method="GET")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with request.urlopen(req, timeout=10) as response:
            artifact["raw_https_probe"]["succeeded"] = True
            artifact["raw_https_probe"]["http_status"] = response.status
    except Exception as exc:
        artifact["raw_https_probe"]["error_type"] = exc.__class__.__name__
        artifact["raw_https_probe"]["error_message"] = sanitize_message(exc)
        status = getattr(exc, "code", None)
        if status is not None:
            artifact["raw_https_probe"]["http_status"] = status

    client: OpenAI | None = None
    try:
        client = OpenAI(api_key=api_key, timeout=10, max_retries=1)
        artifact["sdk_construction_probe"]["succeeded"] = True
    except Exception as exc:
        artifact["sdk_construction_probe"]["error_type"] = exc.__class__.__name__
        artifact["sdk_construction_probe"]["error_message"] = sanitize_message(exc)

    if client is not None:
        try:
            client.models.list()
            artifact["sdk_call_probe"]["succeeded"] = True
        except Exception as exc:
            artifact["sdk_call_probe"]["error_type"] = exc.__class__.__name__
            artifact["sdk_call_probe"]["error_message"] = sanitize_message(exc)

    evaluation_dir = repo_root / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    output_path = evaluation_dir / "connectivity_probe.json"
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
