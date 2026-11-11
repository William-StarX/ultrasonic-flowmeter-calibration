from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "attachment1_window_data.csv"


def relative_error_pct(model_volume: pd.Series, standard_volume: pd.Series) -> pd.Series:
    return (model_volume.astype(float) - standard_volume.astype(float)) / standard_volume.astype(float) * 100.0


def group_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (date, flow_point), group in df.groupby(["date", "flow_point"], sort=True):
        if len(group) < 3:
            continue
        values = group["model_error_pct"].astype(float)
        mean_err = float(values.mean())
        sd = float(values.std(ddof=1))
        rows.append(
            {
                "date": str(date),
                "flow_point": int(flow_point),
                "n": int(len(group)),
                "mean_error_pct": mean_err,
                "repeatability_sd_pct": sd,
                "range_pct": float(values.max() - values.min()),
                "pass_group": abs(mean_err) <= 0.2 and sd <= 0.040,
            }
        )
    return pd.DataFrame(rows)


def linearity_u_l(df: pd.DataFrame) -> float:
    use = df[(df["condition_note"].eq("no_disturbance_reference")) & (df["flow_point"].between(40, 100))]
    by_flow = use.groupby("flow_point")["model_error_pct"].mean()
    if len(by_flow) <= 1:
        return float("nan")
    return math.sqrt(float((by_flow**2).sum()) / (len(by_flow) - 1))


def disturbance_u_d(df: pd.DataFrame) -> float:
    use = df[df["flow_point"].between(40, 100)].copy()
    base = (
        use[use["condition_note"].eq("no_disturbance_reference")]
        .groupby("flow_point")["model_error_pct"]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "baseline_mean", "count": "baseline_n", "std": "baseline_sd"})
        .reset_index()
    )
    disturb = (
        use[use["condition_note"].eq("disturbed_test")]
        .groupby(["disturbance_id", "flow_point"])["model_error_pct"]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "disturb_mean", "count": "disturb_n", "std": "disturb_sd"})
        .reset_index()
    )
    detail = disturb.merge(base, on="flow_point", how="left")
    detail["disturb_sd"] = detail["disturb_sd"].fillna(0.0)
    detail["abs_drift"] = (detail["baseline_mean"] - detail["disturb_mean"]).abs()
    available = detail[(detail["baseline_n"] >= 1) & (detail["disturb_n"] >= 1)].copy()
    if available.empty:
        return float("nan")
    u_d_c = float(available["abs_drift"].max()) / math.sqrt(3.0)
    u_d_r = float(available["disturb_sd"].max())
    return math.sqrt(u_d_c * u_d_c + u_d_r * u_d_r)


def evaluate(submission: Path, output_dir: Path) -> None:
    data = pd.read_csv(DATA, encoding="utf-8-sig")
    sub = pd.read_csv(submission, encoding="utf-8-sig")
    if not {"window_id", "model_volume_m3"}.issubset(sub.columns):
        raise ValueError("submission must contain columns: window_id, model_volume_m3")

    df = data.merge(sub[["window_id", "model_volume_m3"]], on="window_id", how="left")
    if df["model_volume_m3"].isna().any():
        missing = int(df["model_volume_m3"].isna().sum())
        raise ValueError(f"submission misses {missing} window_id rows")

    df["model_error_pct"] = relative_error_pct(df["model_volume_m3"], df["standard_volume_m3"])
    groups = group_table(df)
    u_l = linearity_u_l(df)
    u_r = float(groups["repeatability_sd_pct"].max()) if not groups.empty else float("nan")
    u_d = disturbance_u_d(df)

    summary = pd.DataFrame(
        [
            {
                "window_count": int(len(df)),
                "mae_pct": float(df["model_error_pct"].abs().mean()),
                "mean_error_pct": float(df["model_error_pct"].mean()),
                "group_pass": f"{int(groups['pass_group'].sum())}/{len(groups)}",
                "u_nor_L_pct": u_l,
                "u_nor_r_pct": u_r,
                "u_nor_d_pct": u_d,
                "pass_u_nor_L": bool(pd.notna(u_l) and u_l < 0.036),
                "pass_u_nor_r": bool(pd.notna(u_r) and u_r < 0.040),
                "pass_u_nor_d": bool(pd.notna(u_d) and u_d < 0.115),
                "pass_all_3": bool(pd.notna(u_l) and u_l < 0.036 and pd.notna(u_r) and u_r < 0.040 and pd.notna(u_d) and u_d < 0.115),
            }
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    groups.to_csv(output_dir / "evaluation_groups.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    df[["window_id", "date", "flow_point", "standard_volume_m3", "model_volume_m3", "model_error_pct"]].to_csv(
        output_dir / "evaluation_windows.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ultrasonic-flowmeter modeling submission.")
    parser.add_argument("submission", type=Path, help="CSV with columns window_id, model_volume_m3")
    parser.add_argument("--output-dir", type=Path, default=HERE / "evaluation_output")
    args = parser.parse_args()
    evaluate(args.submission, args.output_dir)


if __name__ == "__main__":
    main()
