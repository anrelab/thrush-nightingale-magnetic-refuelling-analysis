"""
Script 6 — Mann–Whitney U tests on per-bird body-mass-gain slopes.

Purpose
- Split the mass-gain time series into two periods in the weighing-day numbering used in the manuscript.
- For each bird and each period, estimate an individual slope from a linear regression:
      mass_gain ~ day
- Compare slope distributions between groups using a two-sided Mann–Whitney U test.
- Run separately for each year.

Input
- Birds_2024-2025_for_supplementary.csv (semicolon-separated; 2-row header)

Outputs (saved to --outdir)
- mass_gain_slopes_pre_post_mannwhitney.csv
- mass_gain_slopes_pre_post_mannwhitney.xlsx

"""

from __future__ import annotations
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import mannwhitneyu


PERIODS = [
    ("Before (days 3–5)", 3, 5),
    ("After (days 7–13)", 7, 13),
]


def to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def day_from_date(d):
    """Map '14.авг' -> 0, '15.авг' -> 1, ... (Day 0 = 14 Aug)."""
    m = re.match(r"(\d+)", str(d))
    return int(m.group(1)) - 14 if m else np.nan


def load_multilevel_csv(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path, sep=";", header=[0, 1], encoding="utf-8-sig")

    # Forward-fill level 0 where "Unnamed"; clean level 1
    lvl0, lvl1, prev = [], [], None
    for a, b in df_raw.columns:
        if str(a).startswith("Unnamed"):
            a = prev
        else:
            prev = a
        lvl0.append(a)
        lvl1.append("" if str(b).startswith("Unnamed") else b)
    df_raw.columns = pd.MultiIndex.from_arrays([lvl0, lvl1])
    return df_raw


def build_mass_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Mass gain (g) relative to Day 0 baseline, at weighing days."""
    tmp = df_raw[[("Ring", ""), ("Conditions", ""), ("Year", ""),
                  ("Initial weight (14.08)", ""),
                  ("Bird weight", "16.авг"), ("Bird weight", "18.авг"), ("Bird weight", "20.авг"),
                  ("Bird weight", "22.авг"), ("Bird weight", "24.авг"), ("Bird weight", "26.авг")]].copy()

    # Rename baseline into Bird weight group: '14.aug'
    cols = list(tmp.columns)
    cols[3] = ("Bird weight", "14.авг")
    tmp.columns = pd.MultiIndex.from_tuples(cols)

    # Flatten
    flat = []
    for a, b in tmp.columns:
        if (a, b) == ("Ring", ""):
            flat.append("bird_id")
        elif (a, b) == ("Conditions", ""):
            flat.append("condition")
        elif (a, b) == ("Year", ""):
            flat.append("year")
        else:
            flat.append(f"mass_{b}")
    tmp.columns = flat

    mass_cols = [c for c in tmp.columns if c.startswith("mass_")]
    long = tmp.melt(id_vars=["bird_id", "condition", "year"], value_vars=mass_cols,
                    var_name="date", value_name="mass_g")

    long["mass_g"] = long["mass_g"].map(to_float)
    long["date"] = long["date"].str.replace("mass_", "", regex=False)
    long["day_calendar"] = long["date"].map(day_from_date)

    long = long.dropna(subset=["mass_g", "day_calendar"]).copy()
    long["day_calendar"] = long["day_calendar"].astype(int)
    long["year"] = long["year"].astype(str).str.strip()

    # Baseline at Day 0
    baseline = (long[long["day_calendar"] == 0]
                .groupby(["year", "bird_id"])["mass_g"]
                .mean()
                .rename("baseline_mass_g"))
    long = long.merge(baseline.reset_index(), on=["year", "bird_id"], how="left")
    long = long.dropna(subset=["baseline_mass_g"]).copy()

    long["mass_gain_g"] = long["mass_g"] - long["baseline_mass_g"]

    # Paper day numbering for weighings (shift by +1)
    long["day_paper"] = long["day_calendar"] + 1

    return long


def per_bird_slope(mass_long: pd.DataFrame, year: str, day_lo: int, day_hi: int) -> pd.DataFrame:
    """Slope of mass_gain_g ~ day_paper per bird within [day_lo, day_hi]."""
    s = mass_long[(mass_long["year"] == year) &
                  (mass_long["day_paper"] >= day_lo) &
                  (mass_long["day_paper"] <= day_hi)].copy()

    rows = []
    for (cond, bid), sub in s.groupby(["condition", "bird_id"]):
        sub = sub.dropna(subset=["mass_gain_g"])
        if sub["day_paper"].nunique() < 2:
            continue
        X = sm.add_constant(sub["day_paper"].astype(float).values)
        slope = sm.OLS(sub["mass_gain_g"].astype(float).values, X).fit().params[1]
        rows.append({"condition": cond, "bird_id": bid, "slope_g_per_day": float(slope)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Mann–Whitney tests on per-bird mass-gain slopes")
    ap.add_argument("-i", "--input", type=str, required=True, help="Input CSV file")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("."), help="Output directory")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_raw = load_multilevel_csv(args.input)
    mass_long = build_mass_long(df_raw)

    out = []
    for yr in sorted(mass_long["year"].unique(), key=lambda x: int(x)):
        for label, lo, hi in PERIODS:
            sl = per_bird_slope(mass_long, yr, lo, hi)
            g1 = sl[sl["condition"] == "Control"]["slope_g_per_day"]
            g2 = sl[sl["condition"] == "Experiment"]["slope_g_per_day"]

            if len(g1) > 0 and len(g2) > 0:
                test = mannwhitneyu(g1, g2, alternative="two-sided")
                p, U = float(test.pvalue), float(test.statistic)
            else:
                p, U = np.nan, np.nan

            out.append({
                "Year": int(yr),
                "Period": label,
                "n_control": int(len(g1)),
                "n_experiment": int(len(g2)),
                "median_slope_control (g/day)": float(np.median(g1)) if len(g1) else np.nan,
                "median_slope_experiment (g/day)": float(np.median(g2)) if len(g2) else np.nan,
                "U": U,
                "p": p,
            })

    table2 = pd.DataFrame(out).sort_values(["Year", "Period"]).reset_index(drop=True)

    csv_out = str((args.outdir / "mass_gain_slopes_pre_post_mannwhitney.csv").resolve())
    xlsx_out = str((args.outdir / "mass_gain_slopes_pre_post_mannwhitney.xlsx").resolve())

    table2.to_csv(csv_out, index=False)
    table2.to_excel(xlsx_out, index=False)

    print("Saved:")
    print(f" - {csv_out}")
    print(f" - {xlsx_out}")


if __name__ == "__main__":
    main()
