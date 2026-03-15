"""Prompt scaffold loader for MED-SCRIBE nodes."""

from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(node_name: str) -> str:
    prompt_path = PROMPTS_DIR / f"{node_name}_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")
