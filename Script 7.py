"""
Script 7 — Per-bird correlations between intake and mass change (interval-based).

Purpose
- Build paired observations at the interval level between consecutive weighings:
    * mass change rate over the interval (g/day)
    * mean daily food intake over the same interval (g/day)
- For each bird with at least 3 intervals, compute Pearson and Spearman correlations.

Input
- Birds_2024-2025_for_supplementary.csv (semicolon-separated; 2-row header)

Outputs (saved to --outdir)
- mass_intake_interval_correlations_per_bird.csv
- mass_intake_interval_correlations_per_bird.xlsx

"""

from __future__ import annotations
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr



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

    # Rename baseline into Bird weight group: '14.авг'
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
    return long


def build_food_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Daily food intake (g/day) by calendar day."""
    feed_cols = [c for c in df_raw.columns if c[0] == "Feed consumption"]
    if not feed_cols:
        raise ValueError("No 'Feed consumption' columns found.")

    tmp = df_raw[[("Ring", ""), ("Conditions", ""), ("Year", "")] + feed_cols].copy()

    tmp.columns = [
        ("bird_id" if c == ("Ring", "") else
         "condition" if c == ("Conditions", "") else
         "year" if c == ("Year", "") else
         f"feed_{c[1]}")
        for c in tmp.columns
    ]

    day_cols = [c for c in tmp.columns if c.startswith("feed_")]
    long = tmp.melt(id_vars=["bird_id", "condition", "year"], value_vars=day_cols,
                    var_name="date", value_name="food_g_per_day")

    long["food_g_per_day"] = long["food_g_per_day"].map(to_float)
    long["date"] = long["date"].str.replace("feed_", "", regex=False)
    long["day_calendar"] = long["date"].map(day_from_date)

    long = long.dropna(subset=["food_g_per_day", "day_calendar"]).copy()
    long["day_calendar"] = long["day_calendar"].astype(int)
    long["year"] = long["year"].astype(str).str.strip()

    return long


def main():
    ap = argparse.ArgumentParser(description="Interval-based correlations between intake and mass change")
    ap.add_argument("-i", "--input", type=str, required=True, help="Input CSV file")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("."), help="Output directory")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_raw = load_multilevel_csv(args.input)
    mass_long = build_mass_long(df_raw)
    food_long = build_food_long(df_raw)

    # Build interval-level pairs between consecutive weighings for each bird
    pairs = []
    for (yr, cond, bid), w in mass_long.sort_values("day_calendar").groupby(["year", "condition", "bird_id"]):
        w = w.sort_values("day_calendar")
        days = w["day_calendar"].to_numpy(dtype=int)
        gains = w["mass_gain_g"].to_numpy(dtype=float)

        for i in range(1, len(w)):
            d0, d1 = int(days[i-1]), int(days[i])
            if d1 <= d0:
                continue

            rate = (gains[i] - gains[i-1]) / (d1 - d0)

            # Mean daily intake over interval days (d0+1 ... d1)
            fsub = food_long[(food_long["year"] == yr) &
                             (food_long["bird_id"] == bid) &
                             (food_long["day_calendar"] >= d0 + 1) &
                             (food_long["day_calendar"] <= d1)]
            if fsub.empty:
                continue
            mean_intake = float(fsub["food_g_per_day"].mean())

            pairs.append({
                "Year": int(yr),
                "Condition": cond,
                "Bird ID": bid,
                "interval_from_day": d0,
                "interval_to_day": d1,
                "daily_mass_gain_rate_g_per_day": float(rate),
                "mean_daily_food_intake_g_per_day": mean_intake,
            })

    pairs_df = pd.DataFrame(pairs)

    # Correlations per bird (>= 3 paired intervals)
    out = []
    for (yr, cond, bid), sub in pairs_df.groupby(["Year", "Condition", "Bird ID"]):
        x = sub["mean_daily_food_intake_g_per_day"].to_numpy(dtype=float)
        y = sub["daily_mass_gain_rate_g_per_day"].to_numpy(dtype=float)
        n = len(sub)
        if n < 3:
            continue

        pr = pearsonr(x, y)
        sr = spearmanr(x, y)

        pearson_r = pr.statistic if hasattr(pr, "statistic") else pr[0]
        pearson_p = pr.pvalue if hasattr(pr, "pvalue") else pr[1]
        spearman_rho = sr.statistic if hasattr(sr, "statistic") else sr[0]
        spearman_p = sr.pvalue if hasattr(sr, "pvalue") else sr[1]

        out.append({
            "Year": int(yr),
            "Condition": cond,
            "Bird ID": bid,
            "n_pairs": int(n),
            "Pearson r": float(pearson_r),
            "Pearson p": float(pearson_p),
            "Spearman rho": float(spearman_rho),
            "Spearman p": float(spearman_p),
        })

    tableS4 = pd.DataFrame(out).sort_values(["Year", "Condition", "Bird ID"]).reset_index(drop=True)

    csv_out = str((args.outdir / "mass_intake_interval_correlations_per_bird.csv").resolve())
    xlsx_out = str((args.outdir / "mass_intake_interval_correlations_per_bird.xlsx").resolve())

    tableS4.to_csv(csv_out, index=False)
    tableS4.to_excel(xlsx_out, index=False)

    print("Saved:")
    print(f" - {csv_out}")
    print(f" - {xlsx_out}")


if __name__ == "__main__":
    main()
