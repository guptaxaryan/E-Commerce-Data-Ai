"""Gemini model wrapper functions."""
from __future__ import annotations
import json
from typing import Any, Dict, List

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore


def configure(api_key: str) -> None:
    if genai is None:
        raise RuntimeError("google-generativeai package not installed")
    genai.configure(api_key=api_key)


def get_model(name: str = "gemini-1.5-flash") -> Any:
    if genai is None:
        raise RuntimeError("Gemini SDK unavailable")
    return genai.GenerativeModel(name)


def generate_json(model: Any, prompt: str) -> Dict[str, Any]:
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 1024,
        },
    )
    text = response.text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON output: {text}") from exc
