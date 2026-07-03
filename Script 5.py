"""
Script 5 — Primary pooled mixed-effects analysis (minimal model)

Goal
- Answer the main biological question: do trajectories differ between magnetic conditions?
- Pool both experimental years (2024 + 2025).
- Use a minimal linear mixed-effects model for repeated measures:
      y ~ day * treatment + (1 | bird_id)

Notes
- treatment is coded as 0 = control (NMF), 1 = experiment (SMF).
- Random intercept (1|bird_id) accounts for repeated measures within birds.
- Bird IDs are made unique across years (year + ring) to avoid accidental ID collisions.

Outcomes analysed (pooled across years)
1) Body mass gain (g): mass(day) − mass(Day 0), using available weighing days after baseline.
2) Daily food intake (g/day): Days 1–11 only; Day 0 excluded (non-standard transition interval).
3) Cumulative food intake (g): running sum of daily intake from Day 1 (Days 1–11).

Input
- Birds_2024-2025_for_supplementary.csv (semicolon-separated; 2-row header)

Outputs (saved to --outdir)
- primary_mixedlm_mass_gain_coefficients.csv
- primary_mixedlm_daily_intake_coefficients.csv
- primary_mixedlm_cumulative_intake_coefficients.csv
- primary_mixedlm_all_outcomes.xlsx
- primary_mixedlm_model_summaries.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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


def day_from_label(label: str, day0_dom: int) -> Optional[int]:
    """
    Convert labels like '16.авг' / '16.08' to experimental day index.
    Uses only the day-of-month number.
    """
    m = re.match(r"(\d+)", str(label))
    if not m:
        return None
    return int(m.group(1)) - day0_dom


def find_group_name(df: pd.DataFrame, candidates: List[str]) -> str:
    """Find a first-level column group name by substring match (case-insensitive)."""
    lvl0_values = [str(x) for x in df.columns.get_level_values(0).unique()]
    for cand in candidates:
        cand_l = cand.lower()
        for v in lvl0_values:
            if cand_l in v.lower():
                return v
    raise KeyError(
        f"Cannot find a column group matching any of: {candidates}. "
        f"Available groups: {sorted(set(lvl0_values))}"
    )


def get_id_cols(df: pd.DataFrame) -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    """Return MultiIndex keys for Ring, Conditions, Year."""
    def pick(name: str) -> Tuple[str, str]:
        for col in df.columns:
            if str(col[0]).strip().lower() == name.lower() and str(col[1]).strip() in {"", "nan"}:
                return col
        for col in df.columns:
            if name.lower() in str(col[0]).lower():
                return col
        raise KeyError(f"Cannot find identifier column for '{name}'.")
    return pick("Ring"), pick("Conditions"), pick("Year")


def map_treatment(cond: str) -> int:
    """Map condition label -> 0/1 (0 = control/NMF, 1 = experiment/SMF)."""
    s = str(cond).strip().lower()
    if ("control" in s) or ("nmf" in s) or ("контрол" in s):
        return 0
    if ("experiment" in s) or ("smf" in s) or ("экспер" in s):
        return 1
    raise ValueError(f"Cannot map Conditions value to treatment (0/1): '{cond}'")


def load_source_csv(path: Path, sep: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, header=[0, 1], encoding=encoding)
    return fix_multiindex_columns(df)


def melt_group(
    df_raw: pd.DataFrame,
    group_lvl0: str,
    value_name: str,
    day0_dom: int,
) -> pd.DataFrame:
    """Melt a group (first-level header = group_lvl0) into long format."""
    ring_col, cond_col, year_col = get_id_cols(df_raw)
    group_cols = [c for c in df_raw.columns if c[0] == group_lvl0]
    if not group_cols:
        raise KeyError(f"No columns found for group '{group_lvl0}'.")

    df = df_raw[[ring_col, cond_col, year_col] + group_cols].copy()

    # Flatten for melt
    flat_cols = []
    for a, b in df.columns:
        if (a, b) == ring_col:
            flat_cols.append("ring")
        elif (a, b) == cond_col:
            flat_cols.append("conditions")
        elif (a, b) == year_col:
            flat_cols.append("year")
        else:
            flat_cols.append(f"val_{b}")
    df.columns = flat_cols

    value_cols = [c for c in df.columns if c.startswith("val_")]
    long = df.melt(
        id_vars=["ring", "conditions", "year"],
        value_vars=value_cols,
        var_name="date_label",
        value_name=value_name,
    )

    long[value_name] = long[value_name].map(to_float)
    long["date_label"] = long["date_label"].str.replace("val_", "", regex=False)
    long = long.dropna(subset=[value_name])

    long["year"] = long["year"].astype(str).str.strip()
    long["day"] = long["date_label"].map(lambda s: day_from_label(s, day0_dom))
    long = long.dropna(subset=["day"])
    long["day"] = long["day"].astype(int)

    return long[["year", "ring", "conditions", "day", value_name]].copy()


def add_baseline_weight_if_separate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Some files store baseline weight in a separate column like 'Initial weight (14.08)'.
    If present, return a long table with day==0 baseline weights.
    """
    ring_col, cond_col, year_col = get_id_cols(df_raw)
    baseline_candidates = [c for c in df_raw.columns if "initial weight" in str(c[0]).lower()]
    if not baseline_candidates:
        return pd.DataFrame(columns=["year", "ring", "conditions", "day", "mass_g"])

    base_col = baseline_candidates[0]
    df = df_raw[[ring_col, cond_col, year_col, base_col]].copy()
    df.columns = ["ring", "conditions", "year", "mass_g"]
    df["mass_g"] = df["mass_g"].map(to_float)
    df = df.dropna(subset=["mass_g"])
    df["year"] = df["year"].astype(str).str.strip()
    df["day"] = 0
    return df[["year", "ring", "conditions", "day", "mass_g"]].copy()


def prepare_outcome_tables(
    csv_path: Path,
    day0_dom: int,
    sep: str,
    encoding: str,
) -> Dict[str, pd.DataFrame]:
    df_raw = load_source_csv(csv_path, sep=sep, encoding=encoding)

    weight_group = find_group_name(df_raw, ["bird weight"])
    feed_group = find_group_name(df_raw, ["feed consumption", "food intake", "feed intake"])

    # Weights
    w_long = melt_group(df_raw, weight_group, "mass_g", day0_dom=day0_dom)
    w_base = add_baseline_weight_if_separate(df_raw)
    if len(w_base) > 0:
        w_long = pd.concat([w_long, w_base], ignore_index=True)

    # Baseline per bird-year
    baseline = (
        w_long[w_long["day"] == 0]
        .groupby(["year", "ring"], as_index=False)["mass_g"]
        .mean()
        .rename(columns={"mass_g": "baseline_mass"})
    )
    w_long = w_long.merge(baseline, on=["year", "ring"], how="left")
    w_long = w_long.dropna(subset=["baseline_mass"])
    w_long["gain_g"] = w_long["mass_g"] - w_long["baseline_mass"]

    # Intake
    f_long = melt_group(df_raw, feed_group, "intake_g", day0_dom=day0_dom)

    # Make IDs unique across years
    for df in (w_long, f_long):
        df["bird_id"] = df["year"].astype(str) + "_" + df["ring"].astype(str)
        df["treatment"] = df["conditions"].map(map_treatment).astype(int)

    # Outcome 1: mass gain (exclude baseline day where gain=0 for all)
    mg = w_long[w_long["day"] > 0].copy()
    mg = mg.dropna(subset=["gain_g"])
    mg["day_c"] = mg["day"] - mg["day"].min()  # first analysed weighing day -> 0
    mass_gain = mg[["bird_id", "treatment", "day_c", "gain_g"]].rename(columns={"gain_g": "y"})

    # Outcome 2: daily intake (Days 1–11 only; Day 0 excluded)
    di = f_long[(f_long["day"] >= 1) & (f_long["day"] <= 11)].copy()
    di = di.dropna(subset=["intake_g"])
    di["day_c"] = di["day"] - 1  # Day 1 -> 0
    daily_intake = di[["bird_id", "treatment", "day_c", "intake_g"]].rename(columns={"intake_g": "y"})

    # Outcome 3: cumulative intake (running sum from Day 1)
    ci = di.sort_values(["bird_id", "day_c"]).copy()
    ci["cum_intake_g"] = ci.groupby("bird_id")["intake_g"].cumsum()
    cumulative_intake = ci[["bird_id", "treatment", "day_c", "cum_intake_g"]].rename(columns={"cum_intake_g": "y"})

    return {
        "mass_gain": mass_gain,
        "daily_intake": daily_intake,
        "cumulative_intake": cumulative_intake,
    }


def fit_mixedlm(df: pd.DataFrame):
    """Fit minimal mixed model: y ~ day_c * treatment + (1|bird_id)."""
    model = smf.mixedlm("y ~ day_c * treatment", data=df, groups=df["bird_id"])
    res = model.fit(reml=False, method="lbfgs", maxiter=2000, disp=False)
    return res


def tidy_coefficients(res) -> pd.DataFrame:
    """Return a tidy coefficient table with Wald CI."""
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    conf = res.conf_int()
    conf.columns = ["ci_low", "ci_high"]

    out = pd.DataFrame({
        "term": params.index,
        "estimate": params.values,
        "std_error": bse.values,
        "p_value": pvals.values,
    }).merge(conf.reset_index().rename(columns={"index": "term"}), on="term", how="left")

    if hasattr(res, "tvalues"):
        out["z_value"] = np.asarray(res.tvalues)
    else:
        out["z_value"] = np.nan

    cols = ["term", "estimate", "std_error", "z_value", "p_value", "ci_low", "ci_high"]
    return out[cols]


def add_derived_slopes(df_coef: pd.DataFrame) -> pd.DataFrame:
    """Add derived slopes for control and experiment (day effect and day+interaction)."""
    term_day = "day_c"
    term_int = "day_c:treatment"

    if term_day not in set(df_coef["term"]):
        return df_coef

    beta_day = float(df_coef.loc[df_coef["term"] == term_day, "estimate"].iloc[0])
    beta_int = float(df_coef.loc[df_coef["term"] == term_int, "estimate"].iloc[0]) if term_int in set(df_coef["term"]) else 0.0

    derived = pd.DataFrame([
        {"term": "derived_slope_control", "estimate": beta_day, "std_error": np.nan, "z_value": np.nan,
         "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan},
        {"term": "derived_slope_experiment", "estimate": beta_day + beta_int, "std_error": np.nan, "z_value": np.nan,
         "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan},
    ])
    return pd.concat([df_coef, derived], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Primary pooled mixed-effects analysis (minimal model)")
    ap.add_argument("-i", "--input", type=Path, required=True, help="Input CSV file")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("."), help="Output directory")
    ap.add_argument("--sep", type=str, default=";", help="CSV separator (default: ';')")
    ap.add_argument("--encoding", type=str, default="utf-8-sig", help="CSV encoding (default: utf-8-sig)")
    ap.add_argument("--day0-dom", type=int, default=14, help="Day-of-month for Day 0 (default: 14)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    tables = prepare_outcome_tables(
        csv_path=args.input,
        day0_dom=args.day0_dom,
        sep=args.sep,
        encoding=args.encoding,
    )

    coef_tables: Dict[str, pd.DataFrame] = {}
    summaries: List[str] = []

    for outcome_name, df in tables.items():
        res = fit_mixedlm(df)
        coef = tidy_coefficients(res)
        coef = add_derived_slopes(coef)
        coef_tables[outcome_name] = coef

        summaries.append(f"\n===== {outcome_name} =====\n")
        summaries.append(str(res.summary()))
        summaries.append("\n")

        out_csv = args.outdir / f"primary_mixedlm_{outcome_name}_coefficients.csv"
        coef.to_csv(out_csv, index=False)

    out_xlsx = args.outdir / "primary_mixedlm_all_outcomes.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for outcome_name, coef in coef_tables.items():
            coef.to_excel(writer, sheet_name=outcome_name[:31], index=False)

    out_txt = args.outdir / "primary_mixedlm_model_summaries.txt"
    out_txt.write_text("".join(summaries), encoding="utf-8")

    print(f"Saved outputs to: {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
