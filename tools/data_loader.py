import pandas as pd
from pathlib import Path


def load_data(filepath: str) -> dict:
    """
    Load a CSV file and return a summary dict with the DataFrame and metadata.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix in (".xls", ".xlsx"):
        df = pd.read_excel(filepath)
    elif suffix == ".json":
        df = pd.read_json(filepath)
    elif suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return {
        "dataframe": df,
        "filename": path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
