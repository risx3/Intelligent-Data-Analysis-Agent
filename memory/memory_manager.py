"""
MemoryManager — builds LLM context from analysis results + conversation history,
enabling multi-turn natural-language queries over a completed analysis session.
"""
import asyncio
from memory.session_store import SessionStore
from utils.llm import call_llm_with_history
from config import DEFAULT_MODEL, MEMORY_HISTORY_WINDOW

SYSTEM_PROMPT = """You are a data analysis assistant with access to a completed EDA report.
Answer the user's question using ONLY the data from the analysis context provided.
Be specific: cite actual numbers, column names, and statistics.
If you can't answer from the data, say so clearly."""


class MemoryManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.store = SessionStore(session_id)

    async def answer_question(self, question: str, model: str = DEFAULT_MODEL) -> str:
        """
        Answer a follow-up NL question using:
          1. The analysis context (EDA results, insights, report)
          2. Prior conversation history in this session
        """
        data = self.store.load() or {}

        # Build context summary from stored results
        context = self._build_context(data)

        # Build message history for the LLM
        history = data.get("messages", [])
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}]

        # Add prior turns (limit to configured window to stay within context)
        for msg in history[-MEMORY_HISTORY_WINDOW:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current question
        messages.append({"role": "user", "content": question})

        # Call LLM
        answer = await asyncio.to_thread(call_llm_with_history, messages, model)

        # Persist both sides to memory
        self.store.add_message("user",      question)
        self.store.add_message("assistant", answer)

        return answer

    # ── private ───────────────────────────────────────────────────────────────

    def _build_context(self, data: dict) -> str:
        lines = ["=== ANALYSIS CONTEXT ==="]

        # Dataset meta
        lines.append(f"Dataset: {data.get('query', 'N/A')} | Model: {data.get('model')}")

        # Insights
        insights = data.get("insights", "")
        if insights:
            lines += ["", "--- AI Insights ---", insights[:3000]]

        # Execution log
        log = data.get("execution_log", [])
        if log:
            completed = [e["step"] for e in log if e.get("status") == "success"]
            lines.append(f"\nCompleted steps: {', '.join(completed)}")

        # Plot list
        plots = data.get("plots", [])
        if plots:
            import os
            lines.append(f"Generated plots: {', '.join(os.path.basename(p) for p in plots)}")

        return "\n".join(lines)
