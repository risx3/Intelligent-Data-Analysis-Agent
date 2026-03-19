"""
Visualization tools — static EDA plots + NL query-driven dynamic plots.

Level 2: generate_plot() now accepts an optional `query` and `focus_columns`
         list so the VizAgent can steer which charts are produced.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    PLOTS_DIR,
    VIZ_MAX_HIST_COLS,
    VIZ_MAX_BOX_COLS,
    VIZ_MAX_BAR_ROWS,
    VIZ_MAX_PIE_SLICES,
    VIZ_TOP_CAT_COLS,
    VIZ_TOP_VALUES,
)
from utils.logger import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="muted")

# ── helpers ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# ── standard EDA plots ─────────────────────────────────────────────────────────

def generate_plot(
    df: pd.DataFrame,
    query: str | None = None,
    focus_columns: list[str] | None = None,
) -> list[str]:
    """
    Generate standard EDA plot suite.
    If focus_columns is provided, only those columns are used for numeric plots.
    """
    saved: list[str] = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [
        c for c in df.select_dtypes(exclude=[np.number]).columns
        if df[c].nunique() <= 20
    ]

    # narrow to focus columns if given
    if focus_columns:
        focus_set = set(focus_columns)
        numeric_cols = [c for c in numeric_cols if c in focus_set] or numeric_cols
        cat_cols     = [c for c in cat_cols     if c in focus_set] or cat_cols

    # 1. Distribution histograms
    if numeric_cols:
        n = min(len(numeric_cols), VIZ_MAX_HIST_COLS)
        sub = numeric_cols[:n]
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(sub):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
            axes[i].set_title(f"Distribution: {col}")
        for j in range(len(sub), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Numeric Distributions", fontsize=14, y=1.01)
        saved.append(_save(fig, "distributions.png"))

    # 2. Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        size = max(8, len(numeric_cols))
        fig, ax = plt.subplots(figsize=(size, size - 1))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, square=True, linewidths=0.5)
        ax.set_title("Correlation Heatmap", fontsize=14)
        saved.append(_save(fig, "correlation_heatmap.png"))

    # 3. Box plots
    if numeric_cols:
        n = min(len(numeric_cols), VIZ_MAX_BOX_COLS)
        fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 6))
        df[numeric_cols[:n]].boxplot(ax=ax, patch_artist=True)
        ax.set_title("Box Plots (Outlier Detection)", fontsize=14)
        ax.set_xticklabels(numeric_cols[:n], rotation=30, ha="right")
        saved.append(_save(fig, "boxplots.png"))

    # 4. Top-value bar charts for categorical columns
    for col in cat_cols[:VIZ_TOP_CAT_COLS]:
        top = df[col].value_counts().head(VIZ_TOP_VALUES)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top.values, y=top.index.astype(str), ax=ax, palette="Blues_d")
        ax.set_title(f"Top Values: {col}", fontsize=14)
        ax.set_xlabel("Count")
        saved.append(_save(fig, f"barplot_{col}.png"))

    # 5. Pairplot (≤4 numeric columns)
    if 2 <= len(numeric_cols) <= 4:
        pg = sns.pairplot(df[numeric_cols].dropna(), diag_kind="kde",
                          plot_kws={"alpha": 0.5})
        pg.fig.suptitle("Pairplot", y=1.02, fontsize=14)
        path = os.path.join(PLOTS_DIR, "pairplot.png")
        pg.savefig(path, bbox_inches="tight", dpi=150)
        plt.close("all")
        saved.append(path)

    logger.info("Standard EDA plots saved: %d charts", len(saved))
    return saved


# ── dynamic / NL-driven plots ──────────────────────────────────────────────────

def generate_dynamic_plot(df: pd.DataFrame, spec: dict) -> str | None:
    """
    Render a single plot described by `spec` (from VizAgent.parse_intent).

    spec schema:
      {
        "type":    "scatter"|"bar"|"line"|"histogram"|"box"|"heatmap"|"pie"|"count",
        "x":       "<col>",          # optional
        "y":       "<col>",          # optional
        "column":  "<col>",          # for single-column plots
        "columns": ["c1","c2",...],  # for heatmap
        "hue":     "<col>",          # optional colour grouping
        "title":   "<string>"
      }
    """
    plot_type = spec.get("type", "").lower()
    title     = spec.get("title", plot_type)
    hue_col   = spec.get("hue") if spec.get("hue") in df.columns else None

    try:
        match plot_type:
            case "scatter":
                x, y = spec.get("x"), spec.get("y")
                missing = [c for c in (x, y) if c not in df.columns]
                if missing:
                    logger.warning(
                        "scatter plot skipped — columns not found: %s. Available: %s",
                        missing, df.columns.tolist(),
                    )
                    return None
                fig, ax = plt.subplots(figsize=(9, 6))
                sns.scatterplot(data=df, x=x, y=y, hue=hue_col, ax=ax, alpha=0.7)
                ax.set_title(title)
                fname = f"dynamic_scatter_{x}_vs_{y}.png"

            case "line":
                x, y = spec.get("x"), spec.get("y")
                missing = [c for c in (x, y) if c not in df.columns]
                if missing:
                    logger.warning(
                        "line plot skipped — columns not found: %s. Available: %s",
                        missing, df.columns.tolist(),
                    )
                    return None
                sorted_df = df[[x, y]].dropna().sort_values(x)
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.plot(sorted_df[x], sorted_df[y], linewidth=1.8)
                ax.set_title(title); ax.set_xlabel(x); ax.set_ylabel(y)
                fname = f"dynamic_line_{x}_vs_{y}.png"

            case "histogram":
                col = spec.get("column") or spec.get("x")
                if col not in df.columns:
                    logger.warning(
                        "histogram skipped — column '%s' not found. Available: %s",
                        col, df.columns.tolist(),
                    )
                    return None
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
                ax.set_title(title)
                fname = f"dynamic_hist_{col}.png"

            case "box":
                col = spec.get("column") or spec.get("y")
                if col not in df.columns:
                    logger.warning(
                        "box plot skipped — column '%s' not found. Available: %s",
                        col, df.columns.tolist(),
                    )
                    return None
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df, y=col, x=hue_col, ax=ax, palette="Set2")
                ax.set_title(title)
                fname = f"dynamic_box_{col}.png"

            case "bar":
                col = spec.get("column") or spec.get("x")
                if col not in df.columns:
                    logger.warning(
                        "bar chart skipped — column '%s' not found. Available: %s",
                        col, df.columns.tolist(),
                    )
                    return None
                top = df[col].value_counts().head(VIZ_MAX_BAR_ROWS)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=top.values, y=top.index.astype(str),
                            ax=ax, palette="Blues_d")
                ax.set_title(title)
                fname = f"dynamic_bar_{col}.png"

            case "count":
                col = spec.get("column") or spec.get("x")
                if col not in df.columns:
                    logger.warning(
                        "count plot skipped — column '%s' not found. Available: %s",
                        col, df.columns.tolist(),
                    )
                    return None
                fig, ax = plt.subplots(figsize=(10, 5))
                order = df[col].value_counts().index.tolist()
                sns.countplot(data=df, x=col, hue=hue_col, order=order[:VIZ_MAX_BAR_ROWS], ax=ax)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=35)
                fname = f"dynamic_count_{col}.png"

            case "pie":
                col = spec.get("column")
                if col not in df.columns:
                    logger.warning(
                        "pie chart skipped — column '%s' not found. Available: %s",
                        col, df.columns.tolist(),
                    )
                    return None
                counts = df[col].value_counts().head(VIZ_MAX_PIE_SLICES)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(counts.values, labels=counts.index.astype(str),
                       autopct="%1.1f%%", startangle=140)
                ax.set_title(title)
                fname = f"dynamic_pie_{col}.png"

            case "heatmap":
                cols = _safe_cols(df, spec.get("columns", []))
                if len(cols) < 2:
                    logger.warning(
                        "heatmap skipped — fewer than 2 valid columns found in spec %s. Available: %s",
                        spec.get("columns"), df.columns.tolist(),
                    )
                    return None
                corr = df[cols].corr()
                fig, ax = plt.subplots(figsize=(max(6, len(cols)), max(5, len(cols)-1)))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                            center=0, ax=ax, linewidths=0.5)
                ax.set_title(title)
                fname = f"dynamic_heatmap_{'_'.join(cols[:4])}.png"

            case _:
                logger.warning("Unknown plot type '%s' — skipping", plot_type)
                return None

        return _save(fig, fname)

    except Exception as exc:
        logger.error("Could not render %s plot: %s", plot_type, exc, exc_info=True)
        plt.close("all")
        return None
