"""
Script 2 — Body mass gain plots.

What it does
- Reads the raw CSV.
- Converts calendar dates to experimental days (Day 0 = 14 Aug by default).
- Computes body mass gain (g) relative to each bird's Day 0 mass.
- Plots trajectories by group (Control/NMF vs Experiment/SMF):
    * thin semi-transparent lines: individual birds
    * thick line with markers: group mean
    * shaded band: ±SEM
    * vertical dashed line: Day 6 (key experimental day)

Outputs (saved to --outdir)
- Pooled plot (default): body_mass_gain_pooled_2024_2025.png/.pdf
- Optional per-year plots: body_mass_gain_2024.png/.pdf, body_mass_gain_2025.png/.pdf

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
    """Convert a label like '16.авг' to experimental day index using day-of-month only."""
    m = re.match(r"(\d+)", str(label))
    if not m:
        return None
    return int(m.group(1)) - day0_dom


def _find_group(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    """Find a first-level group name by substring match."""
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


def load_weights_long(csv_path: Path, sep: str = ";", encoding: str = "utf-8-sig") -> pd.DataFrame:
    """Return long-format body mass with columns: year, ring, conditions, day, mass_g."""
    df_raw = pd.read_csv(csv_path, sep=sep, header=[0, 1], encoding=encoding)
    df_raw = _fix_multiindex_columns(df_raw)

    ring_col, cond_col, year_col = _get_id_cols(df_raw)
    weight_group = _find_group(df_raw, ("bird weight",))
    weight_cols = [c for c in df_raw.columns if c[0] == weight_group]

    # optional separate baseline column
    baseline_cols = [c for c in df_raw.columns if "initial weight" in str(c[0]).lower()]
    cols = [ring_col, cond_col, year_col] + baseline_cols[:1] + weight_cols
    df = df_raw[cols].copy()

    # Normalize: if baseline exists, treat it as day label '14.авг' so Day 0 is included
    if baseline_cols:
        base_col = baseline_cols[0]
        new_cols = []
        for col in df.columns:
            if col == base_col:
                new_cols.append((weight_group, "14.авг"))
            else:
                new_cols.append(col)
        df.columns = pd.MultiIndex.from_tuples(new_cols)

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
            flat.append(f"mass_{b}")
    df.columns = flat

    mass_cols = [c for c in df.columns if c.startswith("mass_")]
    long = df.melt(id_vars=["ring", "conditions", "year"], value_vars=mass_cols,
                   var_name="date_label", value_name="mass_g")
    long["mass_g"] = long["mass_g"].map(_to_float)
    long["date_label"] = long["date_label"].str.replace("mass_", "", regex=False)
    long = long.dropna(subset=["mass_g"])

    long["year"] = long["year"].astype(str).str.strip()
    return long


def compute_gain(long: pd.DataFrame, day0_dom: int = 14) -> pd.DataFrame:
    out = long.copy()
    out["day"] = out["date_label"].map(lambda s: _day_from_label(s, day0_dom))
    out = out.dropna(subset=["day"])
    out["day"] = out["day"].astype(int)

    baseline = (
        out[out["day"] == 0]
        .groupby(["year", "ring"], as_index=False)["mass_g"]
        .mean()
        .rename(columns={"mass_g": "baseline_mass"})
    )
    out = out.merge(baseline, on=["year", "ring"], how="left")
    out = out.dropna(subset=["baseline_mass"])
    out["gain_g"] = out["mass_g"] - out["baseline_mass"]

    # make bird id unique across years
    out["bird_id"] = out["year"].astype(str) + "_" + out["ring"].astype(str)
    out["treatment"] = out["conditions"].map(_map_treatment).astype(int)
    return out


def _condition_color_map(conditions: list[str]) -> Dict[str, str]:
    """Stable colors: Control=blue, Experiment=orange (fallback to Matplotlib cycle)."""
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


def plot_gain(df: pd.DataFrame, out_png: Path, out_pdf: Path, vline_day: int = 6, dpi: int = 600,
              figsize: Tuple[float, float] = (8, 5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    conditions = sorted(df["conditions"].unique())
    colors = _condition_color_map(conditions)
    xticks = sorted(df["day"].unique())

    for cond in conditions:
        dfg = df[df["conditions"] == cond].copy()

        # individual trajectories
        for _, dfr in dfg.groupby("bird_id"):
            dfr = dfr.sort_values("day")
            ax.plot(dfr["day"].to_numpy(), dfr["gain_g"].to_numpy(),
                    linewidth=1.0, alpha=0.30, color=colors[cond])

        # mean + SEM
        stats = (dfg.groupby("day")["gain_g"]
                 .agg(mean="mean", sd="std", n="size")
                 .reset_index().sort_values("day"))
        stats["sem"] = stats["sd"] / np.sqrt(stats["n"])

        x = stats["day"].to_numpy(dtype=float)
        y = stats["mean"].to_numpy(dtype=float)
        sem = stats["sem"].to_numpy(dtype=float)

        n_birds = dfg["bird_id"].nunique()
        ax.fill_between(x, y - sem, y + sem, alpha=0.18, color=colors[cond])
        ax.plot(x, y, linewidth=3.5, marker="o", color=colors[cond],
                label=f"{cond} mean (n={n_birds})")

    ax.axvline(vline_day, linestyle="--")
    ax.axhline(0, linewidth=1, alpha=0.5)

    ax.set_xlabel("Experimental day (Day 0 = 14 Aug)")
    ax.set_ylabel("Body mass gain (g) relative to Day 0")
    ax.set_xticks(xticks)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Body mass gain plots (pooled and/or per year).")
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

    long = load_weights_long(args.input)
    df = compute_gain(long, day0_dom=args.day0_dom)

    if args.mode in {"pooled", "both"}:
        plot_gain(
            df,
            args.outdir / "body_mass_gain_pooled_2024_2025.png",
            args.outdir / "body_mass_gain_pooled_2024_2025.pdf",
            vline_day=args.vline_day,
            dpi=args.dpi,
        )

    if args.mode in {"year", "both"}:
        for yr in sorted(df["year"].unique()):
            dfx = df[df["year"] == yr].copy()
            plot_gain(
                dfx,
                args.outdir / f"body_mass_gain_{yr}.png",
                args.outdir / f"body_mass_gain_{yr}.pdf",
                vline_day=args.vline_day,
                dpi=args.dpi,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
