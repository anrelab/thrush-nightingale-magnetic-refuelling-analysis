"""
Script 4 — Cumulative food intake plots.

What it does
- Reads the raw CSV.
- Converts calendar dates to experimental days (Day 0 = 14 Aug by default).
- Uses daily intake values for Days 1–11 and computes cumulative intake (g) per bird
  as a running sum starting from Day 1.
- Plots trajectories by group (Control/NMF vs Experiment/SMF):
    * thin semi-transparent lines: individual birds
    * thick line with markers: group mean
    * shaded band: ±SEM
    * vertical dashed line: Day 6 (key experimental day)

Outputs (saved to --outdir)
- Pooled plot (default): food_intake_cumulative_pooled_2024_2025.png/.pdf
- Pooled day-by-day summary: food_intake_cumulative_summary_pooled_2024_2025.csv
- Optional per-year plots: food_intake_cumulative_2024.png/.pdf, food_intake_cumulative_2025.png/.pdf
- Optional per-year summary (stacked): food_intake_cumulative_summary_by_year.csv

Input
- Birds_2024-2025_for_supplementary.csv (semicolon-separated; 2-row header)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _fix_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def _to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan"}:
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _day_from_label(label: str, day0_dom: int) -> Optional[int]:
    m = re.match(r"(\d+)", str(label))
    if not m:
        return None
    return int(m.group(1)) - day0_dom


def _find_group(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    lvl0 = [str(x) for x in df.columns.get_level_values(0).unique()]
    for cand in candidates:
        cl = cand.lower()
        for v in lvl0:
            if cl in v.lower():
                return v
    raise KeyError(f"Could not find a column group matching {candidates}. Available groups: {sorted(set(lvl0))}")


def _get_id_cols(df: pd.DataFrame):
    def pick(name: str):
        for col in df.columns:
            if str(col[0]).strip().lower() == name.lower() and str(col[1]).strip() == "":
                return col
        for col in df.columns:
            if name.lower() in str(col[0]).lower():
                return col
        raise KeyError(f"Cannot find identifier column for '{name}'.")
    return pick("Ring"), pick("Conditions"), pick("Year")


def _map_treatment(cond: str) -> int:
    s = str(cond).strip().lower()
    if ("control" in s) or ("nmf" in s) or ("контрол" in s):
        return 0
    if ("experiment" in s) or ("smf" in s) or ("экспер" in s):
        return 1
    raise ValueError(f"Cannot map Conditions value to treatment (0/1): '{cond}'")


def load_intake_long(csv_path: Path, day0_dom: int = 14, sep: str = ";", encoding: str = "utf-8-sig") -> pd.DataFrame:
    """Return long-format daily intake with columns: year, ring, conditions, day, intake_g."""
    df_raw = pd.read_csv(csv_path, sep=sep, header=[0, 1], encoding=encoding)
    df_raw = _fix_multiindex_columns(df_raw)

    ring_col, cond_col, year_col = _get_id_cols(df_raw)
    intake_group = _find_group(df_raw, ("feed consumption", "food intake", "feed intake"))

    intake_cols = [c for c in df_raw.columns if c[0] == intake_group]
    df = df_raw[[ring_col, cond_col, year_col] + intake_cols].copy()

    # Flatten for melt
    flat = []
    for a, b in df.columns:
        if (a, b) == ring_col:
            flat.append("ring")
        elif (a, b) == cond_col:
            flat.append("conditions")
        elif (a, b) == year_col:
            flat.append("year")
        else:
            flat.append(f"intake_{b}")
    df.columns = flat

    value_cols = [c for c in df.columns if c.startswith("intake_")]
    long = df.melt(id_vars=["ring", "conditions", "year"], value_vars=value_cols,
                   var_name="date_label", value_name="intake_g")
    long["intake_g"] = long["intake_g"].map(_to_float)
    long["date_label"] = long["date_label"].str.replace("intake_", "", regex=False)
    long = long.dropna(subset=["intake_g"])

    long["year"] = long["year"].astype(str).str.strip()
    long["day"] = long["date_label"].map(lambda s: _day_from_label(s, day0_dom=day0_dom))
    long = long.dropna(subset=["day"])
    long["day"] = long["day"].astype(int)

    long["bird_id"] = long["year"].astype(str) + "_" + long["ring"].astype(str)
    long["treatment"] = long["conditions"].map(_map_treatment).astype(int)

    return long[["year", "ring", "conditions", "bird_id", "day", "intake_g"]].copy()


def _condition_color_map(conditions: list[str]) -> Dict[str, str]:
    cmap: Dict[str, str] = {}
    for c in conditions:
        cl = c.lower()
        if "control" in cl or "nmf" in cl or "контрол" in cl:
            cmap[c] = "#1f77b4"
        elif "experiment" in cl or "smf" in cl or "экспер" in cl:
            cmap[c] = "#ff7f0e"
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e"])
    k = 0
    for c in conditions:
        if c not in cmap:
            cmap[c] = cycle[k % len(cycle)]
            k += 1
    return cmap


def summarize_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    s = (df.groupby(["conditions", "day"])["cum_intake_g"]
         .agg(n="size", mean="mean", sd="std")
         .reset_index()
         .sort_values(["conditions", "day"]))
    s["sem"] = s["sd"] / np.sqrt(s["n"])
    return s


def plot_cumulative(df: pd.DataFrame, out_png: Path, out_pdf: Path, vline_day: int = 6, dpi: int = 600,
                    figsize: Tuple[float, float] = (8, 5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    conditions = sorted(df["conditions"].unique())
    colors = _condition_color_map(conditions)
    xticks = sorted(df["day"].unique())

    for cond in conditions:
        dfg = df[df["conditions"] == cond].copy()

        for _, dfr in dfg.groupby("bird_id"):
            dfr = dfr.sort_values("day")
            ax.plot(dfr["day"].to_numpy(), dfr["cum_intake_g"].to_numpy(),
                    linewidth=1.0, alpha=0.30, color=colors[cond])

        stats = summarize_cumulative(dfg)
        x = stats["day"].to_numpy(dtype=float)
        y = stats["mean"].to_numpy(dtype=float)
        sem = stats["sem"].to_numpy(dtype=float)
        n_birds = dfg["bird_id"].nunique()

        ax.fill_between(x, y - sem, y + sem, alpha=0.18, color=colors[cond])
        ax.plot(x, y, linewidth=3.5, marker="o", color=colors[cond],
                label=f"{cond} mean (n={n_birds})")

    ax.axvline(vline_day, linestyle="--")
    ax.set_xlabel("Experimental day (Day 0 = 14 Aug)")
    ax.set_ylabel("Cumulative food intake (g; running sum from Day 1)")
    ax.set_xticks(xticks)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cumulative food intake plots (pooled and/or per year).")
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="Birds_2024-2025_for_supplementary.csv")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("."),
                    help="Output directory")
    ap.add_argument("--mode", choices=["pooled", "year", "both"], default="pooled",
                    help="pooled (two years together), year (separate), or both")
    ap.add_argument("--day0-dom", type=int, default=14,
                    help="Day-of-month for Day 0 (default: 14)")
    ap.add_argument("--vline-day", type=int, default=6,
                    help="Vertical dashed line day (default: 6)")
    ap.add_argument("--dpi", type=int, default=600, help="PNG resolution (default: 600)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_intake_long(args.input, day0_dom=args.day0_dom)
    df = df[(df["day"] >= 1) & (df["day"] <= 11)].copy()

    # compute cumulative per bird
    df = df.sort_values(["bird_id", "day"])
    df["cum_intake_g"] = df.groupby("bird_id")["intake_g"].cumsum()

    # pooled
    if args.mode in {"pooled", "both"}:
        plot_cumulative(
            df,
            args.outdir / "food_intake_cumulative_pooled_2024_2025.png",
            args.outdir / "food_intake_cumulative_pooled_2024_2025.pdf",
            vline_day=args.vline_day,
            dpi=args.dpi,
        )
        summarize_cumulative(df).to_csv(
            args.outdir / "food_intake_cumulative_summary_pooled_2024_2025.csv",
            index=False
        )

    # per year
    if args.mode in {"year", "both"}:
        yearly_rows = []
        for yr in sorted(df["year"].unique()):
            dfx = df[df["year"] == yr].copy()
            plot_cumulative(
                dfx,
                args.outdir / f"food_intake_cumulative_{yr}.png",
                args.outdir / f"food_intake_cumulative_{yr}.pdf",
                vline_day=args.vline_day,
                dpi=args.dpi,
            )
            s = summarize_cumulative(dfx)
            s.insert(0, "year", yr)
            yearly_rows.append(s)
        if yearly_rows:
            pd.concat(yearly_rows, ignore_index=True).to_csv(
                args.outdir / "food_intake_cumulative_summary_by_year.csv",
                index=False
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
