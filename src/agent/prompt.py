"""Prompt construction utilities for Gemini agent."""
from __future__ import annotations
import json
import textwrap
from typing import Dict, List

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an analytical agent that writes concise pandas code to answer questions about
    structured e-commerce data. The user provides a dictionary named `dataframes` with
    pandas.DataFrame objects keyed as <workbook>__<sheet>.

    Respond ONLY with valid JSON using the schema below:
    {
      "thoughts": "short bullet-style outline of your plan (max 3 bullets)",
      "python": "pandas code that defines `analysis_output`",
      "final_answer": "one-paragraph natural language answer"
    }

    Code requirements:
    - Use only the provided `dataframes` and `pd`.
    - Do not read files, call networks, or mutate global state.
    - Summarise results rather than dumping huge tables.
    
    CRITICAL - Your code MUST end with:
    analysis_output = {
        "answer_text": "Brief text answer",
        "metrics": {"key": value},  # Optional
        "result_table": [{"col": val}],  # Optional, max 10 rows
        "chart_data": {"type": "bar", "data": [{"label": "x", "value": y}]}  # Optional
    }
    
    The variable `analysis_output` MUST be defined or the code will fail.
    At minimum: analysis_output = {"answer_text": "Your answer"}
    
    - Keep code deterministic and free of random seeds.

    Be rigorous with calculations and always define analysis_output.
    """
).strip()


def build_schema_summary(dataframes: Dict[str, 'pd.DataFrame']) -> str:
    """Create a compact textual description of loaded tables."""
    if not dataframes:
        return "No data loaded"
    lines = []
    for name, df in list(dataframes.items())[:8]:
        num_rows, num_cols = df.shape
        cols = ", ".join([f"{c} ({df[c].dtype})" for c in df.columns[:12]])
        sample = df.head(2).to_dict(orient="records")
        lines.append(
            textwrap.dedent(
                f"""
                Table: {name}
                Rows: {num_rows}, Columns: {num_cols}
                Columns: {cols}
                Sample: {json.dumps(sample, default=str)}
                """
            ).strip()
        )
    if len(dataframes) > 8:
        lines.append("…additional tables omitted…")
    return "\n\n".join(lines)


def assemble_prompt(question: str, history: List[Dict[str, str]], schema_summary: str) -> str:
    """Combine system instructions, dataset schema and recent history."""
    snippets = []
    for msg in history[-6:]:
        snippets.append(f"{msg.get('role','user')}: {msg.get('content','')}")
    history_text = "\n".join(snippets) or "No prior conversation"
    return textwrap.dedent(
        f"""
        {SYSTEM_PROMPT}

        Dataset profile:
        {schema_summary}

        Conversation history:
        {history_text}

        Current question: {question}
        """
    ).strip()
