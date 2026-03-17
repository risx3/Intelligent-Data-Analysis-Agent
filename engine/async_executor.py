"""
Async Execution Engine — thin wrapper around MultiAgentOrchestrator.

Provides a clean entry point for both the CLI (asyncio.run) and the
FastAPI background task system.
"""
import asyncio
from agents.orchestrator import MultiAgentOrchestrator
from config import DEFAULT_MODEL


class AsyncAnalysisExecutor:
    """
    Drop-in async replacement for the original AnalysisExecutor.

    Usage (CLI):
        executor = AsyncAnalysisExecutor("data.csv", query="...", model="qwen2.5:3b")
        result   = asyncio.run(executor.run())

    Usage (FastAPI background task):
        executor = AsyncAnalysisExecutor(...)
        result   = await executor.run()
    """

    def __init__(
        self,
        filepath: str,
        user_query: str = "",
        session_id: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.filepath   = filepath
        self.user_query = user_query
        self.session_id = session_id
        self.model      = model

    async def run(self) -> dict:
        orchestrator = MultiAgentOrchestrator(
            session_id=self.session_id,
            model=self.model,
        )
        return await orchestrator.run(self.filepath, self.user_query)


def run_sync(filepath: str, user_query: str = "", model: str = DEFAULT_MODEL) -> dict:
    """Convenience wrapper for synchronous contexts (e.g., scripts, tests)."""
    executor = AsyncAnalysisExecutor(filepath=filepath, user_query=user_query, model=model)
    return asyncio.run(executor.run())
