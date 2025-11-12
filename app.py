import json
import os
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
from src.retrieval.tfidf import build_retriever
from src.agent.intent import classify_intent
from src.utils.validation import DataValidator

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment."""
    return "AIzaSyDSEXYfoIGhQxlrX14LbJWRb3lXfH0ty9I"


@st.cache_resource(show_spinner=False)
def load_dataframes(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load tabular data (CSV + Excel) from the data directory into memory.

    Keys:
      CSV  -> <filename_stem>
      XLSX -> <workbook_stem>__<sheet>
    Fault-tolerant: parsing errors surface as toast warnings but do not stop loading.
    """
    tables: Dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        return tables

    # CSV files
    for csv_file in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:  # pragma: no cover
            st.toast(f"⚠️ Failed to load {csv_file.name}: {exc}", icon="⚠️")
            continue
        tables[csv_file.stem] = df

    # Excel workbooks
    for workbook in sorted(data_dir.glob("*.xls*")):
        try:
            xls = pd.ExcelFile(workbook)
        except Exception as exc:  # pragma: no cover
            st.toast(f"⚠️ Failed to load {workbook.name}: {exc}", icon="⚠️")
            continue
        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet)
            except Exception as exc:  # pragma: no cover
                st.toast(
                    f"⚠️ Could not parse sheet '{sheet}' in {workbook.name}: {exc}",
                    icon="⚠️",
                )
                continue
            key = f"{workbook.stem}__{sheet}"
            tables[key] = df
    return tables


def build_schema_summary(dataframes: Dict[str, pd.DataFrame]) -> str:
    """Create a compact textual description of all loaded tables."""
    if not dataframes:
        return "No data loaded"

    summary_lines: List[str] = []
    for name, df in list(dataframes.items())[:8]:
        num_rows, num_cols = df.shape
        columns = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns[:12]])
        sample = df.head(2).to_dict(orient="records")
        summary_lines.append(
            textwrap.dedent(
                f"""
                Table: {name}
                Rows: {num_rows}, Columns: {num_cols}
                Columns: {columns}
                Sample rows: {json.dumps(sample, default=str)}
                """
            ).strip()
        )
    if len(dataframes) > 8:
        summary_lines.append("…additional tables omitted from summary…")
    return "\n\n".join(summary_lines)


def ensure_model() -> Any:
    """Initialise the Gemini model if the SDK is available and configured."""
    api_key = get_api_key()
    if not api_key:
        return None
    if genai is None:
        return None
    genai.configure(api_key=api_key)
    model_name = os.environ.get("MODEL_NAME") or "gemini-2.5-flash"
    try:
        return genai.GenerativeModel(model_name)
    except Exception as exc:  # pragma: no cover
        # Fallback to flash-latest if provided name invalid
        st.toast(f"Model '{model_name}' failed: {exc}. Falling back to gemini-2.5-flash.", icon="⚠️")
        return genai.GenerativeModel("gemini-2.5-flash")


SYSTEM_PROMPT = textwrap.dedent(
        """
        You are an analytical agent that writes concise pandas code to answer questions about
        structured e-commerce data. The user provides a dictionary named `dataframes` with
        pandas.DataFrame objects keyed as <workbook>__<sheet>.

        Respond ONLY with valid JSON using the schema below (NO markdown fences, NO extra text):
        {
            "thoughts": "short bullet-style outline of your plan (max 3 bullets)",
            "python_lines": ["pandas code line 1", "line 2", "..."],
            "final_answer": "one-paragraph natural language answer"
        }

        Output rules:
        - Absolutely no ``` fences or leading identifiers.
        - Each line of code MUST be a separate string element in the python_lines array; do not embed raw newlines inside one string.
        - Keep python_lines concise - aim for 5-15 lines of code maximum.
        - Keep thoughts brief - maximum 3 bullet points.
        - Do not include blank trailing elements.

        Code requirements:
        - Use only the provided `dataframes` and `pd`.
        - Do not read files, call networks, or mutate global state.
        - Summarise results rather than dumping huge tables.
        - Write efficient, compact pandas code - use method chaining where possible.
        
        CRITICAL - Your code MUST end with this pattern:
        analysis_output = {
            "answer_text": "Brief text answer here",
            "metrics": {"key": value},  # Optional: numeric KPIs
            "result_table": [{"col": val}, ...],  # Optional: max 10 rows
            "chart_data": {"type": "bar", "data": [{"label": "x", "value": y}]}  # Optional
        }
        
        The variable `analysis_output` MUST be defined or the code will fail.
        At minimum, include: analysis_output = {"answer_text": "Your answer"}
        
        - Keep code deterministic and free of random seeds.

        Be rigorous with calculations and double-check aggregations.
        IMPORTANT: Raw JSON object only. Always define analysis_output in your code.
        """
).strip()


def call_model(
    model: Any,
    question: str,
    history: List[Dict[str, str]],
    schema_summary: str,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Send a well-structured prompt to Gemini and parse the JSON response.
    
    Includes retry logic for transient failures.
    """
    conversation_snippets = []
    for entry in history[-6:]:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        conversation_snippets.append(f"{role}: {content}")
    history_text = "\n".join(conversation_snippets) or "No prior conversation"

    prompt = textwrap.dedent(
        f"""
        {SYSTEM_PROMPT}

        Dataset profile:
        {schema_summary}

        Conversation history:
        {history_text}

        Current question: {question}
        """
    ).strip()

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                },
            )
            raw = response.text.strip()
            
            # Sanitize common fence patterns
            if raw.startswith("```"):
                # remove first fence
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                # remove closing fence
                if raw.endswith("```"):
                    raw = raw.rsplit("```", 1)[0].strip()
            # If content includes a leading json identifier like 'json {', trim to first '{'
            if "{" in raw and not raw.strip().startswith("{"):
                raw = raw[raw.index("{") :]
            # Heuristic: cut after last '}' to drop stray text
            if raw.count("{") >= 1 and raw.count("}") >= 1:
                last_brace = raw.rfind("}")
                raw = raw[: last_brace + 1]
            
            # If JSON seems truncated, try to close it
            if raw.count("{") > raw.count("}"):
                missing_braces = raw.count("{") - raw.count("}")
                raw += "}" * missing_braces
            
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                # If still failing, provide more context
                if attempt < max_retries:
                    # On retry, we'll try again
                    raise exc
                else:
                    # Last attempt - show what we got
                    raise ValueError(
                        f"Model returned non-JSON output after sanitation.\n"
                        f"Attempted to parse: {raw[:500]}...\n"
                        f"Parse error: {exc}"
                    ) from exc

            # Backward compatibility: if 'python' provided, split into lines
            if 'python_lines' not in payload and 'python' in payload:
                code_block = payload['python']
                # Split on newlines & strip
                payload['python_lines'] = [ln for ln in code_block.split('\n') if ln.strip()]
            
            return payload
            
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                continue  # Retry
            # All retries exhausted
            raise ValueError(f"Model call failed after {max_retries + 1} attempts: {last_error}") from last_error


def execute_agent_code(
    code: str,
    dataframes: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Run the pandas code emitted by the model in a constrained namespace."""
    local_env: Dict[str, Any] = {"dataframes": dataframes, "analysis_output": None}
    global_env = {"pd": pd}

    try:
        exec(compile(code, "<agent>", "exec"), global_env, local_env)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        raise RuntimeError(f"Agent code failed: {exc}\n{traceback_text}") from exc

    output = local_env.get("analysis_output")
    if output is None:
        # Provide helpful error message with the code that was attempted
        error_msg = (
            "Agent code did not define `analysis_output`.\n\n"
            "The code must end with:\n"
            "analysis_output = {'answer_text': 'your answer here'}\n\n"
            f"Generated code was:\n{code[:500]}"
        )
        raise RuntimeError(error_msg)
    return output


def render_result(block: Dict[str, Any]) -> None:
    """Display the dictionary returned by the executed agent code."""
    answer_text = block.get("answer_text")
    if answer_text:
        st.markdown(answer_text)

    metrics = block.get("metrics")
    if isinstance(metrics, dict) and metrics:
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value}")

    table = block.get("result_table")
    if isinstance(table, list) and table:
        st.dataframe(pd.DataFrame(table))

    chart = block.get("chart_data")
    if isinstance(chart, dict):
        chart_type = chart.get("type")
        rows = chart.get("data")
        if isinstance(rows, list) and rows:
            chart_df = pd.DataFrame(rows)
            if chart_type == "bar":
                st.bar_chart(chart_df.set_index("label"))
            elif chart_type == "line":
                st.line_chart(chart_df.set_index("label"))


def main() -> None:
    # Load environment variables from .env if present
    load_dotenv()
    st.set_page_config(page_title="E-commerce Copilot", page_icon="🛒", layout="wide")
    st.title("🛒 E-commerce Data Copilot")
    st.caption("Chat with your Excel-based operations data using Gemini")

    data_dir = Path("data")
    dataframes = load_dataframes(data_dir)
    # Merge in any previously uploaded tables stored in session state
    if "uploaded_dfs" in st.session_state and st.session_state.uploaded_dfs:
        dataframes = {**dataframes, **st.session_state.uploaded_dfs}

    with st.sidebar:
        st.header("🔄 Session")
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            if "uploaded_dfs" in st.session_state:
                del st.session_state["uploaded_dfs"]
            # Delete persistent memory file
            memory_file = Path("memory.json")
            if memory_file.exists():
                memory_file.unlink()
            st.rerun()
        
        st.header("Dataset")
        uploaded_files = st.file_uploader(
            "Upload CSV or Excel files",
            type=["csv", "xls", "xlsx"],
            accept_multiple_files=True,
            help="These will be held in memory only (not written to disk).",
        )
        if uploaded_files:
            new_tables = {}
            for f in uploaded_files:
                suffix = Path(f.name).suffix.lower()
                try:
                    if suffix == ".csv":
                        df = pd.read_csv(f)
                        key = Path(f.name).stem
                        new_tables[key] = df
                    else:
                        xls = pd.ExcelFile(f)
                        for sheet in xls.sheet_names:
                            df = xls.parse(sheet)
                            key = f"{Path(f.name).stem}__{sheet}"
                            new_tables[key] = df
                except Exception as exc:  # pragma: no cover
                    st.toast(f"Failed to parse {f.name}: {exc}", icon="⚠️")
            if new_tables:
                # Persist in session for subsequent questions
                existing = st.session_state.get("uploaded_dfs", {})
                existing.update(new_tables)
                st.session_state.uploaded_dfs = existing
                st.success(f"Uploaded {len(new_tables)} tables. Total in-memory: {len(st.session_state.uploaded_dfs)}")
                # Rebuild combined set for preview after upload
                dataframes = {**load_dataframes(data_dir), **st.session_state.uploaded_dfs}

        if dataframes:
            st.success(f"Loaded {len(dataframes)} tables from disk + uploads.")
            
            # Data quality validation
            issues, stats = DataValidator.validate_all(dataframes)
            if issues:
                errors = [i for i in issues if i.severity == "error"]
                warnings = [i for i in issues if i.severity == "warning"]
                if errors:
                    st.error(f"❌ {len(errors)} data quality errors detected")
                elif warnings:
                    st.warning(f"⚠️ {len(warnings)} data quality warnings")
                
                with st.expander("📊 Data Quality Report", expanded=False):
                    st.text(DataValidator.format_issues_report(issues, limit=15))
            
            selected_table = st.selectbox(
                "Preview table",
                options=sorted(dataframes.keys()),
            )
            st.dataframe(dataframes[selected_table].head())
        else:
            st.warning("Upload .csv / .xls / .xlsx files or place them inside the `data/` directory and rerun.")

        st.header("Model status")
        if get_api_key():
            st.success("API key detected in environment.")
            current_model = os.environ.get("MODEL_NAME") or "gemini-2.5-flash"
            st.caption(f"Using model: `{current_model}`")
        else:
            st.error("Set the GEMINI_API_KEY environment variable before chatting.")

    if "messages" not in st.session_state:
        # Start with empty messages for fresh session
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not dataframes:
        st.info("Upload data to begin chatting.")
        return
    if not get_api_key():
        st.info("Configure GEMINI_API_KEY and refresh the page.")
        return
    if genai is None:
        st.error("google-generativeai package is missing. Install requirements first.")
        return

    user_question = st.chat_input("Ask about your e-commerce data")
    if not user_question:
        return

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_question)
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        model = ensure_model()
        if model is None:
            placeholder.error("Gemini SDK not ready. Check installation and API key.")
            return

        # Intent classification: decide chat vs analysis
        intent = classify_intent(user_question, dataframes)
        if intent == "chat":
            # Use the model to generate a natural conversational response
            chat_prompt = textwrap.dedent(f"""
                You are a friendly data analyst assistant. The user said: "{user_question}"
                
                Respond naturally and conversationally. Keep your response brief (1-2 sentences).
                
                Context:
                - You help analyze e-commerce data from Excel/CSV files
                - You can answer questions about products, sales, customers, orders, etc.
                - If they greet you, greet back warmly
                - If they thank you, acknowledge graciously
                - If they ask what you can do, give 2-3 concrete examples
                
                Respond as a helpful assistant would in a natural conversation.
            """).strip()
            
            try:
                chat_response = model.generate_content(
                    chat_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 150,
                    },
                )
                greeting_reply = chat_response.text.strip()
                # Remove any markdown formatting
                greeting_reply = greeting_reply.replace("```", "").replace("**", "")
            except Exception:
                # Fallback to generic response if model fails
                if any(w in user_question.lower() for w in ["thank", "thanks"]):
                    greeting_reply = "You're welcome! Feel free to ask me anything about your data."
                elif any(w in user_question.lower() for w in ["hi", "hello", "hey"]):
                    greeting_reply = "Hello! I'm here to help you analyze your e-commerce data. What would you like to know?"
                else:
                    greeting_reply = "I'm your data analysis assistant. Ask me about sales trends, top products, customer insights, or any metrics you'd like to explore!"
            
            placeholder.empty()
            st.markdown(greeting_reply)
            st.session_state.messages.append({"role": "assistant", "content": greeting_reply})
            return

        schema_summary = build_schema_summary(dataframes)
        retriever = build_retriever(dataframes)
        top_docs = retriever.retrieve(user_question, k=3)
        retrieval_context = "\n\nRETRIEVED CONTEXT:\n" + "\n---\n".join([d.text[:1000] for d in top_docs]) if top_docs else ""
        try:
            augmented_question = user_question + retrieval_context
            model_reply = call_model(model, augmented_question, st.session_state.messages, schema_summary)
        except Exception as exc:
            placeholder.error(f"Model call failed: {exc}")
            return

        thoughts = model_reply.get("thoughts", "")
        python_lines = model_reply.get("python_lines", [])
        python_code = "\n".join(python_lines)
        final_answer = model_reply.get("final_answer", "")

        with st.expander("Model plan", expanded=False):
            st.code(thoughts)
        with st.expander("Generated code", expanded=False):
            st.code(python_code or "# (no code generated)", language="python")

        try:
            analysis_block = execute_agent_code(python_code, dataframes)
        except Exception as exc:
            placeholder.error(str(exc))
            return

        placeholder.empty()
        render_result(analysis_block)
        if final_answer:
            st.markdown("---")
            st.markdown(final_answer)

        st.session_state.messages.append({"role": "assistant", "content": final_answer or "I have shared the analysis above."})


if __name__ == "__main__":
    main()
