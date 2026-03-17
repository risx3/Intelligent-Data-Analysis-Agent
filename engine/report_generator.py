import os
import json
from datetime import datetime
from typing import Any
from config import REPORTS_DIR


def build_report(context: dict[str, Any], user_query: str = "") -> dict:
    """
    Assemble a Markdown report from all analysis context and save it to disk.
    Returns a dict with the report path and content.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = context.get("load_data", {}).get("filename", "dataset").replace(".", "_")
    report_filename = f"report_{filename_base}_{timestamp}.md"
    report_path = os.path.join(REPORTS_DIR, report_filename)

    meta = {k: v for k, v in context.get("load_data", {}).items() if k != "dataframe"}
    plan = context.get("plan", {})
    missing = context.get("check_missing_values", {})
    stats = context.get("run_basic_stats", {})
    corr = context.get("correlation_matrix", {})
    describe = context.get("describe_data", {})
    insights = context.get("build_insights", "")
    plots = context.get("generate_visualizations", {}).get("saved_plots", [])

    lines = []

    # Header
    lines += [
        f"# Data Analysis Report: {meta.get('filename', 'Unknown Dataset')}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if user_query:
        lines += [f"**User Query:** _{user_query}_", ""]

    # Dataset Overview
    lines += [
        "---",
        "## Dataset Summary",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| File | `{meta.get('filename')}` |",
        f"| Rows | {meta.get('rows'):,} |",
        f"| Columns | {meta.get('columns')} |",
        f"| Numeric Columns | {len(describe.get('numeric_columns', []))} |",
        f"| Categorical Columns | {len(describe.get('categorical_columns', []))} |",
        f"| Complete Rows | {missing.get('complete_rows', 'N/A')} ({missing.get('complete_rows_pct', 'N/A')}%) |",
        f"| Total Missing Cells | {missing.get('total_missing_cells', 0)} |",
        "",
    ]

    # Column Types
    if describe.get("numeric_columns"):
        lines += [
            "### Numeric Columns",
            ", ".join(f"`{c}`" for c in describe["numeric_columns"]),
            "",
        ]
    if describe.get("categorical_columns"):
        lines += [
            "### Categorical Columns",
            ", ".join(f"`{c}`" for c in describe["categorical_columns"]),
            "",
        ]

    # Missing Values
    lines += ["---", "## Data Quality", ""]
    missing_by_col = missing.get("missing_by_column", {})
    if missing_by_col:
        lines += ["### Missing Values", ""]
        lines += ["| Column | Missing Count | Percentage | Severity |"]
        lines += ["|--------|--------------|------------|----------|"]
        for col, info in missing_by_col.items():
            lines.append(
                f"| `{col}` | {info['count']} | {info['percentage']}% | {info['severity']} |"
            )
        lines.append("")
    else:
        lines += ["No missing values detected.", ""]

    # Statistical Summary
    col_stats = stats.get("column_stats", {})
    if col_stats:
        lines += ["---", "## Statistical Summary", ""]
        lines += ["| Column | Mean | Median | Std Dev | Skewness | Outliers |"]
        lines += ["|--------|------|--------|---------|----------|----------|"]
        for col, s in col_stats.items():
            lines.append(
                f"| `{col}` | {s['mean']} | {s['median']} | {s['std']} "
                f"| {s['skewness']} | {s['outlier_count']} ({s['outlier_pct']}%) |"
            )
        lines.append("")

    # Correlations
    high_corr = corr.get("high_correlation_pairs", [])
    if high_corr:
        lines += ["---", "## High Correlations", ""]
        lines += ["| Column 1 | Column 2 | Correlation |"]
        lines += ["|----------|----------|-------------|"]
        for pair in high_corr:
            lines.append(
                f"| `{pair['col1']}` | `{pair['col2']}` | {pair['correlation']:.4f} |"
            )
        lines.append("")

    # Insights
    if insights:
        lines += ["---", "## AI-Generated Insights", ""]
        lines.append(insights if isinstance(insights, str) else str(insights))
        lines.append("")

    # Visualizations
    if plots:
        lines += ["---", "## Visualizations", ""]
        for plot_path in plots:
            plot_name = os.path.basename(plot_path)
            lines.append(f"![{plot_name}]({plot_path})")
        lines.append("")

    # Analysis Plan
    lines += [
        "---",
        "## Analysis Plan Executed",
        "",
        f"**Steps:** {' → '.join(plan.get('steps', []))}",
        "",
    ]
    if plan.get("analysis_goals"):
        lines += ["**Goals:**"]
        for goal in plan["analysis_goals"]:
            lines.append(f"- {goal}")
        lines.append("")

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n  Report saved → {report_path}")
    return {"report_path": report_path, "content": report_content}
