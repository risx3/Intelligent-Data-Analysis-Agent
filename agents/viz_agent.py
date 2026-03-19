"""
Viz Agent — specialist agent for visualization.

Responsibilities:
  1. Parse the user query with the LLM to produce a list of targeted plot specs.
  2. Render those dynamic plots via viz_tools.generate_dynamic_plot().
  3. Also run the standard EDA plot suite (generate_plot).
"""
import asyncio
import pandas as pd
from utils.llm import call_llm
from utils.parser import extract_json
from utils.logger import get_logger
from tools.viz_tools import generate_plot, generate_dynamic_plot
from config import DEFAULT_MODEL

logger = get_logger(__name__)

_INTENT_SYSTEM = """You are a data visualisation expert.
Given a dataset description and a user query, decide which plots would best answer the query.

Return ONLY valid JSON — no markdown, no extra text:
{
  "plots": [
    {"type": "scatter",   "x": "col_a",   "y": "col_b",   "title": "A vs B"},
    {"type": "histogram", "column": "col_c",               "title": "Distribution of C"},
    {"type": "bar",       "column": "cat_col",             "title": "Category breakdown"},
    {"type": "line",      "x": "date_col","y": "value_col","title": "Trend over time"},
    {"type": "box",       "column": "col_d", "hue": "cat", "title": "D by category"}
  ]
}

Supported types: scatter | bar | line | histogram | box | heatmap | pie | count
Limit to 4 most impactful plots.
If no meaningful plots can be derived from the query, return {"plots": []}.
"""


class VizAgent:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    async def run(
        self,
        df: pd.DataFrame,
        user_query: str = "",
        plan: dict | None = None,
    ) -> dict:
        """
        Generate both standard + query-driven plots in parallel.

        Returns:
            {"saved_plots": [path1, path2, ...]}
        """
        focus_columns = (plan or {}).get("focus_columns", [])
        viz_intents   = (plan or {}).get("viz_intents",   [])

        # If query but no intents from planner, ask LLM now
        if user_query and not viz_intents:
            viz_intents = await asyncio.to_thread(
                self._parse_intent, user_query, df.columns.tolist(),
                {c: str(t) for c, t in df.dtypes.items()},
            )

        # Run standard + dynamic plots in parallel
        standard_task = asyncio.to_thread(generate_plot, df, user_query, focus_columns)
        dynamic_task  = asyncio.to_thread(self._render_dynamic, df, viz_intents)

        standard_plots, dynamic_plots = await asyncio.gather(standard_task, dynamic_task)

        return {"saved_plots": standard_plots + dynamic_plots}

    # ── private ────────────────────────────────────────────────────────────────

    def _parse_intent(
        self,
        query: str,
        columns: list[str],
        dtypes: dict,
    ) -> list[dict]:
        prompt = f"""Dataset columns: {columns}
Column types   : {dtypes}
User query     : {query}

Which 4 plots would best answer the query?"""
        try:
            response = call_llm(_INTENT_SYSTEM, prompt, model=self.model)
            data = extract_json(response)
            return data.get("plots", [])
        except ValueError as exc:
            logger.warning("VizAgent could not parse intent from LLM response: %s", exc)
            return []
        except Exception as exc:
            logger.error("VizAgent intent parsing failed unexpectedly: %s", exc, exc_info=True)
            return []

    def _render_dynamic(self, df: pd.DataFrame, intents: list[dict]) -> list[str]:
        paths = []
        for spec in intents:
            path = generate_dynamic_plot(df, spec)
            if path:
                paths.append(path)
        return paths
