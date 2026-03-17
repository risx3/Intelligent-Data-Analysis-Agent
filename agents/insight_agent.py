"""
Insight Agent — converts raw EDA outputs into structured business insights.

Level 3 upgrade:
  - Supports reflection: regenerate_with_critique() accepts prior critique
    and produces improved insights.
"""
from utils.llm import call_llm
from config import DEFAULT_MODEL

_SYSTEM = """You are a senior data analyst generating business insights from EDA results.

Output MUST be structured markdown with these exact section headings:
## Key Findings
## Statistical Highlights
## Data Quality Notes
## Recommendations

Rules:
- Reference specific numbers (means, percentages, counts) from the data.
- Keep each bullet point to one clear observation or action.
- Do not pad with generic statements like "data looks good".
"""

_REVISE_SYSTEM = """You are a senior data analyst improving a previous set of insights.

A reviewer found issues — address every point in the critique.
Output MUST use the same section structure:
## Key Findings
## Statistical Highlights
## Data Quality Notes
## Recommendations

Be MORE specific: cite exact numbers, name columns explicitly.
"""


def generate_insights(
    dataset_meta: dict,
    describe_result: dict,
    missing_result: dict,
    stats_result: dict,
    corr_result: dict,
    user_query: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    prompt = _build_prompt(
        dataset_meta, describe_result, missing_result,
        stats_result, corr_result, user_query,
    )
    return call_llm(_SYSTEM, prompt, model=model)


def regenerate_with_critique(
    previous_insights: str,
    critique: dict,
    dataset_meta: dict,
    describe_result: dict,
    missing_result: dict,
    stats_result: dict,
    corr_result: dict,
    user_query: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    context = _build_prompt(
        dataset_meta, describe_result, missing_result,
        stats_result, corr_result, user_query,
    )
    prompt = f"""Previous insights (score {critique.get('score')}/10):
{previous_insights}

Reviewer issues:
{chr(10).join(f'  - {i}' for i in critique.get('issues', []))}

Reviewer suggestions:
{chr(10).join(f'  - {s}' for s in critique.get('suggestions', []))}

=== Raw Data Context ===
{context}

Generate improved insights addressing every critique point."""
    return call_llm(_REVISE_SYSTEM, prompt, model=model)


# ── private ────────────────────────────────────────────────────────────────────

def _build_prompt(dataset_meta, describe_result, missing_result,
                  stats_result, corr_result, user_query) -> str:
    query_line = f"User question: {user_query}\n" if user_query else ""
    return f"""{query_line}Dataset: {dataset_meta.get('filename')} | {dataset_meta.get('rows')} rows

Numeric columns  : {describe_result.get('numeric_columns')}
Categorical cols : {describe_result.get('categorical_columns')}
High-cardinality : {describe_result.get('high_cardinality_columns')}

Missing cells  : {missing_result.get('total_missing_cells')}
Complete rows  : {missing_result.get('complete_rows')} ({missing_result.get('complete_rows_pct')}%)
Columns w/ missing: {list(missing_result.get('missing_by_column', {}).keys())}

{_fmt_stats(stats_result)}

High correlations:
{corr_result.get('high_correlation_pairs', 'none')}"""


def _fmt_stats(stats_result: dict) -> str:
    col_stats = stats_result.get("column_stats", {})
    if not col_stats:
        return "No numeric columns."
    lines = ["Per-column stats:"]
    for col, s in col_stats.items():
        lines.append(
            f"  {col}: mean={s['mean']}, median={s['median']}, std={s['std']}, "
            f"skew={s['skewness']}, outliers={s['outlier_count']} ({s['outlier_pct']}%)"
        )
    return "\n".join(lines)
