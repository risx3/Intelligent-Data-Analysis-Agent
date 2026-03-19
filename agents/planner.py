"""
Planner Agent — creates a structured, query-aware analysis plan.

Level 2 upgrade:
  - Returns focus_columns derived from the user query
  - Returns viz_intents for NL-driven dynamic plotting
"""
from utils.llm import call_llm
from utils.parser import extract_json
from utils.logger import get_logger
from config import DEFAULT_MODEL

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a data analysis planning agent.

Given dataset metadata and an optional user query, produce a JSON analysis plan.

Return ONLY valid JSON — no markdown, no extra text:
{
  "steps": ["load_data", ...],
  "focus_columns": ["col1", "col2"],
  "analysis_goals": ["goal 1", "goal 2"],
  "viz_intents": [
    {"type": "scatter", "x": "col_a", "y": "col_b", "title": "A vs B"},
    {"type": "bar",     "column": "category",       "title": "Category breakdown"}
  ]
}

Available steps (use only these exact strings):
  load_data | describe_data | check_missing_values | run_basic_stats |
  correlation_matrix | generate_visualizations | build_insights | generate_report

Rules:
- Always start with load_data, end with generate_report.
- Always include build_insights and generate_visualizations.
- focus_columns: columns most relevant to the user query (empty list if no query).
- viz_intents: 2-4 specific plots that directly answer the query.
  Supported types: scatter, bar, line, histogram, box, heatmap, pie, count
- If no query, viz_intents may be empty.
"""

_DEFAULT_STEPS = [
    "load_data", "describe_data", "check_missing_values",
    "run_basic_stats", "correlation_matrix",
    "generate_visualizations", "build_insights", "generate_report",
]


def create_plan(
    dataset_meta: dict,
    user_query: str = "",
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Args:
        dataset_meta: Output from load_data (without the DataFrame).
        user_query:   Optional natural-language question/goal.
        model:        Ollama model to use.

    Returns dict with: steps, focus_columns, analysis_goals, viz_intents.
    """
    query_section = f"\nUser Query: {user_query}" if user_query else ""

    prompt = f"""Dataset:
- File    : {dataset_meta.get('filename', 'unknown')}
- Rows    : {dataset_meta.get('rows')}
- Columns : {dataset_meta.get('columns')}
- Names   : {dataset_meta.get('column_names')}
- Dtypes  : {dataset_meta.get('dtypes')}
{query_section}

Create the analysis plan."""

    response = call_llm(SYSTEM_PROMPT, prompt, model=model)

    try:
        plan = extract_json(response)
    except ValueError as exc:
        logger.warning(
            "Planner could not parse LLM response as JSON — using default plan. Error: %s", exc
        )
        plan = {"_auto_generated": True}

    plan.setdefault("steps", _DEFAULT_STEPS)
    plan.setdefault("focus_columns", [])
    plan.setdefault("analysis_goals", [])
    plan.setdefault("viz_intents", [])

    if plan.get("_auto_generated"):
        logger.warning("Using auto-generated default plan (LLM did not return valid JSON)")

    return plan
