"""Sandbox execution for model-emitted pandas code."""
from __future__ import annotations
import traceback
from typing import Any, Dict
import pandas as pd

__all__ = ["execute_code"]


def execute_code(code: str, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Execute pandas code within a controlled local namespace.

    The snippet must define `analysis_output`.
    Raises RuntimeError on failure.
    """
    local_env: Dict[str, Any] = {"dataframes": dataframes, "analysis_output": None}
    global_env = {"pd": pd}
    try:
        exec(compile(code, "<agent>", "exec"), global_env, local_env)
    except Exception as exc:
        raise RuntimeError(
            f"Agent code failed: {exc}\n{traceback.format_exc()}"
        ) from exc
    output = local_env.get("analysis_output")
    if output is None:
        raise RuntimeError("Agent code did not define `analysis_output`.")
    return output
