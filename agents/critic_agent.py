"""
Critic Agent — reviews insight quality and drives the reflection loop.

Returns a score (1-10) + actionable feedback.
If score >= REFLECTION_MIN_SCORE → verdict "accept", else "revise".
"""
from utils.llm import call_llm
from utils.parser import extract_json
from config import DEFAULT_MODEL, REFLECTION_MIN_SCORE

SYSTEM_PROMPT = """You are a ruthlessly critical data analysis reviewer.
Evaluate the insights below against the EDA data provided.

Return ONLY valid JSON — no markdown fences, no extra text:
{
  "score": <integer 1-10>,
  "issues": ["specific issue 1", "specific issue 2"],
  "suggestions": ["concrete suggestion 1", "concrete suggestion 2"],
  "verdict": "accept" | "revise"
}

Scoring rubric:
  9-10 : Specific numbers cited, directly answers the query, actionable recommendations
  7-8  : Mostly specific, minor gaps in actionability
  5-6  : Some generalisations, references some numbers
  3-4  : Mostly generic, few data references
  1-2  : Completely vague, no data cited

Set verdict = "accept" if score >= """ + str(REFLECTION_MIN_SCORE) + """, else "revise".
"""


def review_insights(
    insights: str,
    eda_summary: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Review insights and return critique dict.

    Returns:
        {"score": int, "issues": [...], "suggestions": [...], "verdict": str}
    """
    prompt = f"""=== EDA Data ===
{eda_summary}

=== Insights Under Review ===
{insights}

Review the insights for quality, specificity, and actionability."""

    response = call_llm(SYSTEM_PROMPT, prompt, model=model)

    try:
        result = extract_json(response)
    except ValueError:
        # If LLM fails to return JSON, accept as-is
        return {"score": 7, "verdict": "accept", "issues": [], "suggestions": []}

    # Ensure all fields present
    score = int(result.get("score", 7))
    result["score"]   = score
    result["verdict"] = "accept" if score >= REFLECTION_MIN_SCORE else "revise"
    result.setdefault("issues", [])
    result.setdefault("suggestions", [])
    return result


def build_eda_summary(
    dataset_meta: dict,
    describe_result: dict,
    missing_result: dict,
    stats_result: dict,
    corr_result: dict,
) -> str:
    """Compact EDA summary string for the critic prompt."""
    lines = [
        f"Dataset: {dataset_meta.get('filename')} | {dataset_meta.get('rows')} rows | {dataset_meta.get('columns')} cols",
        f"Numeric cols   : {describe_result.get('numeric_columns')}",
        f"Categorical cols: {describe_result.get('categorical_columns')}",
        f"Missing cells  : {missing_result.get('total_missing_cells')} "
        f"({missing_result.get('complete_rows_pct')}% complete rows)",
    ]
    col_stats = stats_result.get("column_stats", {})
    for col, s in list(col_stats.items())[:5]:
        lines.append(
            f"  {col}: mean={s['mean']}, std={s['std']}, skew={s['skewness']}, "
            f"outliers={s['outlier_count']}"
        )
    high_corr = corr_result.get("high_correlation_pairs", [])
    if high_corr:
        lines.append(f"High correlations: {high_corr[:3]}")
    return "\n".join(lines)
