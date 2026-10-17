"""Download and validate the processed Cleveland heart disease dataset."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]
FEATURES = COLUMNS[:-1]


def load_dataset(source: str = UCI_URL) -> pd.DataFrame:
    """Read the Cleveland CSV, preserve '?' as missing, and normalize target."""
    df = pd.read_csv(source, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)
    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = (df["target"] > 0).astype(int)
    if df.empty:
        raise ValueError("Dataset contains no usable rows")
    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary after normalization")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/heart.csv")
    parser.add_argument("--source", default=UCI_URL, help="CSV URL or local file path")
    args = parser.parse_args()
    df = load_dataset(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows and {len(df.columns) - 1} features to {output}")
    print("Missing values by column:\n", df.isna().sum().to_string())
    print("Target balance:\n", df.target.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
