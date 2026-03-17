"""
Streamlit UI — interactive dashboard for the Intelligent Data Analysis Agent.

Run:  streamlit run ui/streamlit_app.py
      (requires API service: uvicorn api.main:app --port 8000)
"""
import time
import os
import requests
import streamlit as st

# ── config ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
POLL_INTERVAL = 2  # seconds

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ────────────────────────────────────────────────────────────────────

def api_health() -> dict:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except Exception:
        return {"status": "unreachable", "ollama_reachable": False, "available_models": []}


def start_analysis(file_bytes: bytes, filename: str, query: str, model: str) -> str | None:
    try:
        r = requests.post(
            f"{API_URL}/analyse",
            files={"file": (filename, file_bytes)},
            data={"query": query, "model": model},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["session_id"]
    except Exception as e:
        st.error(f"Failed to start analysis: {e}")
        return None


def poll_status(session_id: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/analyse/{session_id}/status", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e), "progress": []}


def get_result(session_id: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/analyse/{session_id}/result", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Could not fetch result: {e}")
        return None


def ask_question(session_id: str, question: str, model: str) -> str:
    try:
        r = requests.post(
            f"{API_URL}/sessions/{session_id}/query",
            json={"question": question, "model": model},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["answer"]
    except Exception as e:
        return f"Error: {e}"


def get_history(session_id: str) -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/sessions/{session_id}/history", timeout=5)
        r.raise_for_status()
        return r.json().get("messages", [])
    except Exception:
        return []


# ── session state init ─────────────────────────────────────────────────────────

for key, default in {
    "session_id": None,
    "status":     "idle",
    "result":     None,
    "chat":       [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Data Analysis Agent")
    st.divider()

    # Health indicator
    health = api_health()
    if health["status"] == "ok":
        st.success("API: Online")
        ollama_status = "Online" if health["ollama_reachable"] else "Offline"
        colour = "green" if health["ollama_reachable"] else "red"
        st.markdown(f"Ollama: :{colour}[{ollama_status}]")
    else:
        st.error("API: Offline — start with `uvicorn api.main:app`")

    st.divider()

    # Model selector
    available = health.get("available_models", [])
    model_options = available if available else ["qwen2.5:3b", "llama3.2:3b", "phi3:mini", "mistral:7b"]
    model = st.selectbox("Ollama model", model_options, index=0)

    st.divider()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="CSV, Excel, JSON or Parquet",
    )

    query = st.text_area(
        "Analysis goal (optional)",
        placeholder="e.g. What factors most influence sales?",
        height=80,
    )

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if st.session_state.session_id:
        st.divider()
        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        if st.button("New Analysis", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.status     = "idle"
            st.session_state.result     = None
            st.session_state.chat       = []
            st.rerun()

# ── trigger analysis ───────────────────────────────────────────────────────────

if run_btn and uploaded_file:
    st.session_state.result     = None
    st.session_state.chat       = []
    sid = start_analysis(
        uploaded_file.getvalue(),
        uploaded_file.name,
        query,
        model,
    )
    if sid:
        st.session_state.session_id = sid
        st.session_state.status     = "running"
        st.rerun()
elif run_btn and not uploaded_file:
    st.warning("Please upload a dataset first.")

# ── poll while running ─────────────────────────────────────────────────────────

if st.session_state.status == "running" and st.session_state.session_id:
    status_data = poll_status(st.session_state.session_id)
    current_status = status_data.get("status", "running")

    with st.container():
        st.subheader("Analysis in progress...")
        progress_log = status_data.get("progress", [])
        if progress_log:
            for entry in progress_log:
                sym = "✅" if entry.get("status") == "success" else "❌"
                st.write(f"{sym} `{entry['step']}` — {entry.get('duration_sec', '?')}s")
        else:
            st.spinner("Initialising...")

    if current_status == "done":
        st.session_state.status = "done"
        st.session_state.result = get_result(st.session_state.session_id)
        st.rerun()
    elif current_status == "failed":
        st.session_state.status = "failed"
        st.error(f"Analysis failed: {status_data.get('error', 'unknown error')}")
    else:
        time.sleep(POLL_INTERVAL)
        st.rerun()

# ── results view ───────────────────────────────────────────────────────────────

elif st.session_state.status == "done" and st.session_state.result:
    result = st.session_state.result
    session_id = st.session_state.session_id

    tab_overview, tab_plots, tab_insights, tab_report, tab_chat = st.tabs([
        "Overview", "Visualisations", "AI Insights", "Report", "Chat"
    ])

    # ── Overview tab ──────────────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Execution Summary")
        log = result.get("execution_log", [])
        cols = st.columns(3)
        cols[0].metric("Steps completed", sum(1 for e in log if e["status"] == "success"))
        cols[1].metric("Plots generated", len(result.get("plots", [])))
        cols[2].metric("Steps failed",    sum(1 for e in log if e["status"] == "failed"))

        st.divider()
        for entry in log:
            colour = "green" if entry["status"] == "success" else "red"
            st.markdown(
                f":{colour}[{'✓' if entry['status'] == 'success' else '✗'}] "
                f"`{entry['step']}` — {entry.get('duration_sec', '?')}s"
            )
            if entry.get("error"):
                st.caption(f"  Error: {entry['error']}")

    # ── Plots tab ─────────────────────────────────────────────────────────────
    with tab_plots:
        plots = result.get("plots", [])
        if not plots:
            st.info("No plots generated.")
        else:
            st.subheader(f"{len(plots)} visualisations")
            cols = st.columns(2)
            for i, path in enumerate(plots):
                if os.path.exists(path):
                    cols[i % 2].image(path, caption=os.path.basename(path), use_container_width=True)
                else:
                    cols[i % 2].warning(f"Plot not found: {os.path.basename(path)}")

    # ── Insights tab ──────────────────────────────────────────────────────────
    with tab_insights:
        insights = result.get("insights", "")
        if insights:
            st.markdown(insights)
        else:
            st.info("No insights generated.")

    # ── Report tab ────────────────────────────────────────────────────────────
    with tab_report:
        report_path = result.get("report_path")
        if report_path and os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
            st.divider()
            st.download_button(
                "Download Report (.md)",
                data=content,
                file_name=os.path.basename(report_path),
                mime="text/markdown",
            )
        else:
            st.info("Report not available.")

    # ── Chat tab ──────────────────────────────────────────────────────────────
    with tab_chat:
        st.subheader("Ask a follow-up question")
        st.caption("Powered by conversation memory — the agent remembers the full analysis.")

        # Display history
        history = get_history(session_id)
        for msg in history:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])

        # New message
        if user_input := st.chat_input("e.g. Which columns have the most outliers?"):
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(session_id, user_input, model)
                st.markdown(answer)

            st.session_state.chat.append({"role": "user",      "content": user_input})
            st.session_state.chat.append({"role": "assistant", "content": answer})

# ── idle state ─────────────────────────────────────────────────────────────────

else:
    st.markdown("""
# Intelligent Data Analysis Agent

Upload a dataset in the sidebar and click **Run Analysis** to get started.

**What it does:**
- Automatically plans the analysis using an LLM
- Runs EDA, statistics, and correlation in parallel
- Generates tailored visualisations (standard + query-driven)
- Produces AI insights with a self-critique reflection loop
- Generates a full Markdown report
- Lets you ask follow-up questions in the Chat tab

**Supported formats:** CSV, Excel, JSON, Parquet
**LLM backend:** Ollama (local, no API key needed)
""")
