from __future__ import annotations

# 论文附录复现代码。
# 题设复验：按官方分组和阈值计算全部评价指标。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "calibration" / "attachment1_window_data.csv"


def relativeErrorPct(moxingTiji: pd.Series, biaozhunTiji: pd.Series) -> pd.Series:
    return (moxingTiji.astype(float) - biaozhunTiji.astype(float)) / biaozhunTiji.astype(float) * 100.0


def groupTable(df: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame(
        rows,
        columns=["date", "flow_point", "n", "mean_error_pct", "repeatability_sd_pct", "range_pct", "pass_group"],
    )


def linearityUL(df: pd.DataFrame) -> float:
    use = df[(df["condition_note"].eq("no_disturbance_reference")) & (df["flow_point"].between(40, 100))]
    by_flow = use.groupby("flow_point")["model_error_pct"].mean()
    if len(by_flow) <= 1:
        return float("nan")
    return math.sqrt(float((by_flow**2).sum()) / (len(by_flow) - 1))


def disturbanceUD(df: pd.DataFrame) -> float:
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


def evaluate(submission: Path, output_dir: Path, reference: Path = DATA) -> None:
    data = pd.read_csv(reference, encoding="utf-8-sig")
    sub = pd.read_csv(submission, encoding="utf-8-sig")
    required_reference = {"window_id", "date", "flow_point", "condition_note", "disturbance_id", "standard_volume_m3"}
    if not required_reference.issubset(data.columns):
        raise ValueError(f"reference is missing columns: {', '.join(sorted(required_reference - set(data.columns)))}")
    if not {"window_id", "model_volume_m3"}.issubset(sub.columns):
        raise ValueError("estimates must contain columns: window_id, model_volume_m3")
    if any(frame.empty or frame["window_id"].isna().any() or not frame["window_id"].is_unique for frame in (data, sub)):
        raise ValueError("reference and estimates must have nonempty, unique window_id values")
    if set(data["window_id"]) != set(sub["window_id"]):
        raise ValueError("reference and estimates must cover the same window_id values")
    standard_volume = data["standard_volume_m3"].to_numpy(dtype=float)
    model_volume = sub["model_volume_m3"].to_numpy(dtype=float)
    if not np.isfinite(standard_volume).all() or (standard_volume <= 0).any():
        raise ValueError("standard_volume_m3 must contain finite, positive values")
    if not np.isfinite(model_volume).all():
        raise ValueError("model_volume_m3 must contain finite values")

    df = data.merge(sub[["window_id", "model_volume_m3"]], on="window_id", validate="one_to_one")
    df["model_error_pct"] = relativeErrorPct(df["model_volume_m3"], df["standard_volume_m3"])
    groups = groupTable(df)
    u_l = linearityUL(df)
    u_r = float(groups["repeatability_sd_pct"].max()) if not groups.empty else float("nan")
    u_d = disturbanceUD(df)

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
