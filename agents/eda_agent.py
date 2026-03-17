"""
EDA Agent — specialist agent for data quality and statistical analysis.
Runs all EDA tools in parallel using asyncio.
"""
import asyncio
import pandas as pd
from tools.eda_tools import describe_data, check_missing_values, correlation_matrix, run_basic_stats


class EDAAgent:
    """Runs all four EDA tools in parallel and returns a merged context dict."""

    def __init__(self, model: str | None = None):
        self.model = model  # reserved for future LLM-assisted EDA steps

    async def run(
        self,
        df: pd.DataFrame,
        focus_columns: list[str] | None = None,
    ) -> dict:
        """
        Execute describe, missing-values, stats, and correlation in parallel.

        Args:
            df:             The full DataFrame.
            focus_columns:  Optional subset — used to filter column-level outputs.

        Returns:
            Dict with keys: describe_data, check_missing_values,
                            run_basic_stats, correlation_matrix
        """
        describe, missing, stats, corr = await asyncio.gather(
            asyncio.to_thread(describe_data,          df),
            asyncio.to_thread(check_missing_values,   df),
            asyncio.to_thread(run_basic_stats,        df),
            asyncio.to_thread(correlation_matrix,     df),
        )

        # Narrow stats to focus_columns if provided
        if focus_columns:
            focus_set = set(focus_columns)
            col_stats = stats.get("column_stats", {})
            stats["column_stats"] = {
                k: v for k, v in col_stats.items() if k in focus_set
            } or col_stats  # fall back to all if no overlap

        return {
            "describe_data":       describe,
            "check_missing_values": missing,
            "run_basic_stats":     stats,
            "correlation_matrix":  corr,
        }
