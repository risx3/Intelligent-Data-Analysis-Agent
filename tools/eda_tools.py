import pandas as pd
import numpy as np
from config import MAX_CATEGORICAL_UNIQUE, MISSING_VALUE_THRESHOLD, CORRELATION_THRESHOLD


def describe_data(df: pd.DataFrame) -> dict:
    """Return a structured description of the dataset."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and df[col].nunique() <= MAX_CATEGORICAL_UNIQUE
    ]
    high_cardinality_cols = [
        col for col in df.columns
        if col not in numeric_cols and df[col].nunique() > MAX_CATEGORICAL_UNIQUE
    ]

    description = {}
    if numeric_cols:
        stats = df[numeric_cols].describe().round(4)
        description["numeric_summary"] = stats.to_dict()

    cat_summary = {}
    for col in categorical_cols:
        counts = df[col].value_counts()
        cat_summary[col] = {
            "unique_values": int(df[col].nunique()),
            "top_values": counts.head(5).to_dict(),
        }
    description["categorical_summary"] = cat_summary

    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "high_cardinality_columns": high_cardinality_cols,
        "description": description,
    }


def check_missing_values(df: pd.DataFrame) -> dict:
    """Analyse missing values per column."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)).round(4)

    missing_info = {}
    for col in df.columns:
        count = int(missing_counts[col])
        pct = float(missing_pct[col])
        if count > 0:
            missing_info[col] = {
                "count": count,
                "percentage": round(pct * 100, 2),
                "severity": (
                    "high" if pct > MISSING_VALUE_THRESHOLD else
                    "medium" if pct > 0.05 else "low"
                ),
            }

    return {
        "total_missing_cells": int(missing_counts.sum()),
        "columns_with_missing": len(missing_info),
        "missing_by_column": missing_info,
        "complete_rows": int(df.dropna().shape[0]),
        "complete_rows_pct": round(df.dropna().shape[0] / len(df) * 100, 2),
    }


def correlation_matrix(df: pd.DataFrame) -> dict:
    """Compute correlations and flag high-correlation pairs."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {"message": "Not enough numeric columns for correlation analysis."}

    corr = numeric_df.corr().round(4)
    high_corr_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= CORRELATION_THRESHOLD:
                high_corr_pairs.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": float(val),
                })

    return {
        "correlation_matrix": corr.to_dict(),
        "high_correlation_pairs": sorted(
            high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True
        ),
    }


def run_basic_stats(df: pd.DataFrame) -> dict:
    """Compute distribution and outlier statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"message": "No numeric columns found."}

    stats = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]

        stats[col] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "skewness": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurtosis()), 4),
            "outlier_count": int(len(outliers)),
            "outlier_pct": round(len(outliers) / len(series) * 100, 2),
            "iqr": round(float(iqr), 4),
        }

    return {"column_stats": stats}
