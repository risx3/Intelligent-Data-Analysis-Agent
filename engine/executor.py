"""
Sync executor — thin shim kept for backward compatibility.
Delegates to AsyncAnalysisExecutor via asyncio.run().
"""
from engine.async_executor import run_sync
from config import DEFAULT_MODEL


class AnalysisExecutor:
    """Synchronous wrapper — use AsyncAnalysisExecutor for new code."""

    def __init__(self, filepath: str, user_query: str = "", model: str = DEFAULT_MODEL):
        self.filepath   = filepath
        self.user_query = user_query
        self.model      = model

    def run(self) -> dict:
        return run_sync(self.filepath, self.user_query, self.model)
