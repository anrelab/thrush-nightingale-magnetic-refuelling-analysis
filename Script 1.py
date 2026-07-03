"""
Script 1 — Data preparation: convert the raw CSV into a tidy (long) dataset.

Input:
  - Birds_2024-2025_for_supplementary.csv (semicolon-separated; 2-row header)
  - Body mass recorded on selected calendar dates
  - Daily food intake (feed consumption)

Output (tidy/long CSV):
  One row per bird per experimental day (Day 0 is the baseline day, 14 Aug by default).
  Missing values are kept as NaN when a variable was not measured on a given day.

Columns in the output CSV:
  ring, year, conditions, day, calendar_day_of_month,
  body_mass_g, baseline_mass_g, body_mass_gain_g, food_intake_g

Example:
  python "Script S1.py" -i Birds_2024-2025_for_supplementary.csv -o data_tidy_long.csv

"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def fix_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill unnamed first-level headers and clean unnamed second-level headers."""
    lvl0, lvl1, prev = [], [], None
    for a, b in df.columns:
        a_str = str(a)
        b_str = str(b)
        if a_str.startswith("Unnamed"):
            a = prev
        else:
            prev = a
        lvl0.append(a)
        lvl1.append("" if b_str.startswith("Unnamed") else b)
    df.columns = pd.MultiIndex.from_arrays([lvl0, lvl1])
    return df


def to_float(x) -> float:
    """Robust float parsing (handles comma decimals)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() == "none":
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def day_from_label(label: str, day0_dom: int) -> int | float:
    """
    Convert a date label like '16.авг' to experimental day index (Day 0 is day0_dom).
    Only the numeric day-of-month is used.
    """
    m = re.match(r"(\d+)", str(label))
    return (int(m.group(1)) - day0_dom) if m else np.nan


def melt_group(
    df: pd.DataFrame,
    group_name: str,
    id_cols: Iterable[Tuple[str, str]],
    value_name: str,
) -> pd.DataFrame:
    """
    Melt a MultiIndex-column table for one group (first-level header).

    Returns long format with:
      ring, conditions, year, date_label, <value_name>
    """
    group_cols = [c for c in df.columns if c[0] == group_name]
    if not group_cols:
        raise KeyError(f"No columns found for group '{group_name}'")

    sub = df[list(id_cols) + group_cols].copy()

    # Flatten columns for melt
    flat = []
    for a, b in sub.columns:
        if (a in {"Ring", "Conditions", "Year"}) and b == "":
            flat.append(a.lower())
        else:
            flat.append(f"v_{b}")
    sub.columns = flat

    value_cols = [c for c in sub.columns if c.startswith("v_")]
    long = sub.melt(
        id_vars=["ring", "conditions", "year"],
        value_vars=value_cols,
        var_name="tmp",
        value_name=value_name,
    )
    long[value_name] = long[value_name].map(to_float)
    long["date_label"] = long["tmp"].str.replace("v_", "", regex=False)
    long = long.drop(columns=["tmp"]).dropna(subset=[value_name])

    # Keep year consistently as a string
    long["year"] = long["year"].astype(str).str.strip()

    return long


def load_table_s3(csv_path: Path, sep: str = ";", encoding: str = "utf-8-sig") -> pd.DataFrame:
    """Read source dataset and return a cleaned MultiIndex-column DataFrame."""
    df = pd.read_csv(csv_path, sep=sep, header=[0, 1], encoding=encoding)
    return fix_multiindex_columns(df)


def make_tidy(
    df: pd.DataFrame,
    day0_dom: int = 14,
    mass_baseline_col: Tuple[str, str] = ("Initial weight (14.08)", ""),
    mass_group: str = "Bird weight",
    food_group: str = "Feed consumption",
) -> pd.DataFrame:
    """
    Create a tidy bird-by-day dataset.

    Body mass:
      - Day 0 is taken from `mass_baseline_col` and treated as `mass_group` with label '14.авг'
      - Subsequent masses are taken from `mass_group` columns

    Food intake:
      - Taken from `food_group` columns

    Returns a DataFrame with one row per bird per experimental day.
    """
    id_cols = [("Ring", ""), ("Conditions", ""), ("Year", "")]

    # Map the baseline mass into the same group as other mass columns
    if mass_baseline_col not in df.columns:
        raise KeyError(f"Baseline mass column not found: {mass_baseline_col}")

    work = df.copy()
    renamed = []
    for col in work.columns:
        if col == mass_baseline_col:
            renamed.append((mass_group, "14.авг"))
        else:
            renamed.append(col)
    work.columns = pd.MultiIndex.from_tuples(renamed)

    # Long tables
    mass_long = melt_group(work, mass_group, id_cols=id_cols, value_name="body_mass_g")
    food_long = melt_group(work, food_group, id_cols=id_cols, value_name="food_intake_g")

    # Experimental days
    mass_long["day"] = mass_long["date_label"].map(lambda s: day_from_label(s, day0_dom))
    food_long["day"] = food_long["date_label"].map(lambda s: day_from_label(s, day0_dom))

    mass_long = mass_long.dropna(subset=["day"]).copy()
    food_long = food_long.dropna(subset=["day"]).copy()
    mass_long["day"] = mass_long["day"].astype(int)
    food_long["day"] = food_long["day"].astype(int)

    # Merge (outer join keeps days with only food or only mass)
    tidy = pd.merge(
        mass_long.drop(columns=["date_label"]),
        food_long.drop(columns=["date_label"]),
        on=["ring", "conditions", "year", "day"],
        how="outer",
    )

    # Calendar day-of-month (August) for convenience
    tidy["calendar_day_of_month"] = tidy["day"] + day0_dom

    # Baseline mass and gain (relative to Day 0)
    baseline = (
        tidy[tidy["day"] == 0]
        .groupby(["year", "ring"], as_index=False)["body_mass_g"]
        .mean()
        .rename(columns={"body_mass_g": "baseline_mass_g"})
    )
    tidy = tidy.merge(baseline, on=["year", "ring"], how="left")
    tidy["body_mass_gain_g"] = tidy["body_mass_g"] - tidy["baseline_mass_g"]

    # Order columns for readability
    tidy = tidy[
        [
            "ring",
            "year",
            "conditions",
            "day",
            "calendar_day_of_month",
            "body_mass_g",
            "baseline_mass_g",
            "body_mass_gain_g",
            "food_intake_g",
        ]
    ].sort_values(["year", "conditions", "ring", "day"]).reset_index(drop=True)

    return tidy


def main() -> int:
    # Silence a common pandas warning that may appear when exporting mixed dtypes.
    warnings.filterwarnings("ignore", message="invalid value encountered in cast")

    parser = argparse.ArgumentParser(
        description="Convert source dataset to a tidy bird-by-day dataset (Script S1)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the source dataset CSV file (e.g., Birds_2024-2025_for_supplementary.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data_tidy_long.csv"),
        help="Path to output tidy CSV (default: source dataset_tidy.csv).",
    )
    parser.add_argument(
        "--day0-dom",
        type=int,
        default=14,
        help="Day-of-month that corresponds to Day 0 (default: 14 for 14 Aug).",
    )
    parser.add_argument("--sep", type=str, default=";", help="CSV separator (default: ';').")
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding (default: utf-8-sig).",
    )

    args = parser.parse_args()

    df = load_table_s3(args.input, sep=args.sep, encoding=args.encoding)
    tidy = make_tidy(df, day0_dom=args.day0_dom)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
