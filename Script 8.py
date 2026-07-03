"""
Script 8 — Bootstrap analysis of integrative endpoints (pooled across years).

Endpoints (per bird)
1) Total body-mass gain (g): last available body mass − body mass on Day 0
2) Mean daily food intake (g/day): average across available Days 1–11

Bootstrap
- Statistic: difference in group means (Experiment − Control)
- Resampling unit: birds (resampled with replacement within each group)
- Confidence interval: percentile 95% (2.5% and 97.5%)
- Two-sided bootstrap p-value: 2 * min(P(diff <= 0), P(diff >= 0))

Input
- Birds_2024-2025_for_supplementary.csv (semicolon-separated; 2-row header)

Outputs (saved to --outdir)
- bootstrap_integrative_metrics_2024_2025.csv
- bootstrap_integrative_metrics_2024_2025.png
- bootstrap_integrative_metrics_2024_2025.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, Dict

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


def load_long_tables(csv_path: Path, day0_dom: int = 14, sep: str = ";", encoding: str = "utf-8-sig"):
    """Return long-format weights and intake."""
    df_raw = pd.read_csv(csv_path, sep=sep, header=[0, 1], encoding=encoding)
    df_raw = _fix_multiindex_columns(df_raw)

    ring_col, cond_col, year_col = _get_id_cols(df_raw)

    weight_group = _find_group(df_raw, ("bird weight",))
    intake_group = _find_group(df_raw, ("feed consumption", "food intake", "feed intake"))

    # weights
    weight_cols = [c for c in df_raw.columns if c[0] == weight_group]
    baseline_cols = [c for c in df_raw.columns if "initial weight" in str(c[0]).lower()]
    w_cols = [ring_col, cond_col, year_col] + baseline_cols[:1] + weight_cols
    w = df_raw[w_cols].copy()

    if baseline_cols:
        base_col = baseline_cols[0]
        new_cols = []
        for col in w.columns:
            if col == base_col:
                new_cols.append((weight_group, "14.авг"))
            else:
                new_cols.append(col)
        w.columns = pd.MultiIndex.from_tuples(new_cols)

    flat = []
    for a, b in w.columns:
        if (a, b) == ring_col:
            flat.append("ring")
        elif (a, b) == cond_col:
            flat.append("conditions")
        elif (a, b) == year_col:
            flat.append("year")
        else:
            flat.append(f"mass_{b}")
    w.columns = flat

    w_long = w.melt(id_vars=["ring", "conditions", "year"],
                    value_vars=[c for c in w.columns if c.startswith("mass_")],
                    var_name="date_label", value_name="mass_g")
    w_long["mass_g"] = w_long["mass_g"].map(_to_float)
    w_long["date_label"] = w_long["date_label"].str.replace("mass_", "", regex=False)
    w_long = w_long.dropna(subset=["mass_g"])
    w_long["year"] = w_long["year"].astype(str).str.strip()
    w_long["day"] = w_long["date_label"].map(lambda s: _day_from_label(s, day0_dom))
    w_long = w_long.dropna(subset=["day"])
    w_long["day"] = w_long["day"].astype(int)

    # intake
    intake_cols = [c for c in df_raw.columns if c[0] == intake_group]
    f = df_raw[[ring_col, cond_col, year_col] + intake_cols].copy()

    flat = []
    for a, b in f.columns:
        if (a, b) == ring_col:
            flat.append("ring")
        elif (a, b) == cond_col:
            flat.append("conditions")
        elif (a, b) == year_col:
            flat.append("year")
        else:
            flat.append(f"intake_{b}")
    f.columns = flat

    f_long = f.melt(id_vars=["ring", "conditions", "year"],
                    value_vars=[c for c in f.columns if c.startswith("intake_")],
                    var_name="date_label", value_name="intake_g")
    f_long["intake_g"] = f_long["intake_g"].map(_to_float)
    f_long["date_label"] = f_long["date_label"].str.replace("intake_", "", regex=False)
    f_long = f_long.dropna(subset=["intake_g"])
    f_long["year"] = f_long["year"].astype(str).str.strip()
    f_long["day"] = f_long["date_label"].map(lambda s: _day_from_label(s, day0_dom))
    f_long = f_long.dropna(subset=["day"])
    f_long["day"] = f_long["day"].astype(int)

    # common ids
    for d in (w_long, f_long):
        d["bird_id"] = d["year"].astype(str) + "_" + d["ring"].astype(str)
        d["treatment"] = d["conditions"].map(_map_treatment).astype(int)

    return w_long, f_long


def compute_endpoints(w_long: pd.DataFrame, f_long: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bird endpoints and return one row per bird."""
    # baseline mass at day 0 (per bird-year)
    baseline = (w_long[w_long["day"] == 0]
                .groupby(["bird_id"], as_index=False)["mass_g"]
                .mean()
                .rename(columns={"mass_g": "baseline_mass_g"}))

    # last available mass (max day per bird)
    last = (w_long.sort_values(["bird_id", "day"])
            .groupby("bird_id", as_index=False)
            .tail(1)[["bird_id", "mass_g"]]
            .rename(columns={"mass_g": "last_mass_g"}))

    birds = (w_long[["bird_id", "year", "ring", "conditions", "treatment"]]
             .drop_duplicates("bird_id")
             .copy())

    ep = birds.merge(baseline, on="bird_id", how="left").merge(last, on="bird_id", how="left")
    ep = ep.dropna(subset=["baseline_mass_g", "last_mass_g"])
    ep["total_mass_gain_g"] = ep["last_mass_g"] - ep["baseline_mass_g"]

    # mean daily intake Days 1–11
    f = f_long[(f_long["day"] >= 1) & (f_long["day"] <= 11)].copy()
    mean_intake = (f.groupby("bird_id", as_index=False)["intake_g"]
                   .mean()
                   .rename(columns={"intake_g": "mean_daily_intake_g_per_day"}))
    ep = ep.merge(mean_intake, on="bird_id", how="left")
    ep = ep.dropna(subset=["mean_daily_intake_g_per_day"])

    return ep


def bootstrap_diff_means(values: np.ndarray, groups: np.ndarray, n: int, rng: np.random.Generator):
    """Bootstrap difference in means (exp - ctrl) resampling birds within each group."""
    v_ctrl = values[groups == 0]
    v_exp = values[groups == 1]
    if len(v_ctrl) < 2 or len(v_exp) < 2:
        raise ValueError("Not enough birds in one of the groups for bootstrap.")

    diffs = np.empty(n, dtype=float)
    for i in range(n):
        s_ctrl = rng.choice(v_ctrl, size=len(v_ctrl), replace=True)
        s_exp = rng.choice(v_exp, size=len(v_exp), replace=True)
        diffs[i] = s_exp.mean() - s_ctrl.mean()

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    p_le0 = np.mean(diffs <= 0)
    p_ge0 = np.mean(diffs >= 0)
    p_two = 2 * min(p_le0, p_ge0)

    return diffs, float(ci_low), float(ci_high), float(p_two)


def plot_bootstrap(diffs_a: np.ndarray, diffs_b: np.ndarray, out_png: Path, out_pdf: Path, dpi: int = 600) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    axes[0].hist(diffs_a, bins=40)
    axes[0].axvline(0, linestyle="--")
    axes[0].set_title("Total body-mass gain: mean difference")
    axes[0].set_xlabel("Experiment − Control (g)")
    axes[0].set_ylabel("Bootstrap count")

    axes[1].hist(diffs_b, bins=40)
    axes[1].axvline(0, linestyle="--")
    axes[1].set_title("Mean daily intake: mean difference")
    axes[1].set_xlabel("Experiment − Control (g/day)")
    axes[1].set_ylabel("Bootstrap count")

    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap analysis of integrative endpoints (pooled across years).")
    ap.add_argument("-i", "--input", type=Path, required=True, help="Input CSV file")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("."), help="Output directory")
    ap.add_argument("--n", type=int, default=10000, help="Number of bootstrap resamples (default: 10000)")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed (default: 12345)")
    ap.add_argument("--day0-dom", type=int, default=14, help="Day-of-month for Day 0 (default: 14)")
    ap.add_argument("--dpi", type=int, default=600, help="PNG resolution (default: 600)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    w_long, f_long = load_long_tables(args.input, day0_dom=args.day0_dom)
    endpoints = compute_endpoints(w_long, f_long)

    # Bootstrap for each endpoint
    diffs_mass, ciL_mass, ciH_mass, p_mass = bootstrap_diff_means(
        endpoints["total_mass_gain_g"].to_numpy(),
        endpoints["treatment"].to_numpy(),
        n=int(args.n),
        rng=rng,
    )
    diffs_int, ciL_int, ciH_int, p_int = bootstrap_diff_means(
        endpoints["mean_daily_intake_g_per_day"].to_numpy(),
        endpoints["treatment"].to_numpy(),
        n=int(args.n),
        rng=rng,
    )

    # Summary table (group means + bootstrap results)
    group_means = endpoints.groupby("treatment").agg(
        n_birds=("bird_id", "nunique"),
        mean_total_mass_gain_g=("total_mass_gain_g", "mean"),
        mean_daily_intake_g_per_day=("mean_daily_intake_g_per_day", "mean"),
    ).reset_index()
    group_means["group"] = group_means["treatment"].map({0: "Control", 1: "Experiment"})

    summary = pd.DataFrame([
        {
            "endpoint": "total_mass_gain_g",
            "difference_exp_minus_ctrl": float(diffs_mass.mean()),
            "ci95_low": ciL_mass,
            "ci95_high": ciH_mass,
            "p_boot_two_sided": p_mass,
        },
        {
            "endpoint": "mean_daily_intake_g_per_day",
            "difference_exp_minus_ctrl": float(diffs_int.mean()),
            "ci95_low": ciL_int,
            "ci95_high": ciH_int,
            "p_boot_two_sided": p_int,
        },
    ])

    out_csv = args.outdir / "bootstrap_integrative_metrics_2024_2025.csv"
    out_png = args.outdir / "bootstrap_integrative_metrics_2024_2025.png"
    out_pdf = args.outdir / "bootstrap_integrative_metrics_2024_2025.pdf"

    # Save: endpoints per bird + group means + summary
    with pd.ExcelWriter(args.outdir / "bootstrap_integrative_metrics_2024_2025.xlsx", engine="openpyxl") as writer:
        endpoints.to_excel(writer, sheet_name="per_bird", index=False)
        group_means.to_excel(writer, sheet_name="group_means", index=False)
        summary.to_excel(writer, sheet_name="bootstrap_summary", index=False)

    # Save CSV summary (bootstrap results only, compact)
    summary.to_csv(out_csv, index=False)

    # Save plot
    plot_bootstrap(diffs_mass, diffs_int, out_png, out_pdf, dpi=int(args.dpi))

    print("Saved:")
    print(f" - {out_csv.resolve()}")
    print(f" - {out_png.resolve()}")
    print(f" - {out_pdf.resolve()}")
    print(f" - {(args.outdir / 'bootstrap_integrative_metrics_2024_2025.xlsx').resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
