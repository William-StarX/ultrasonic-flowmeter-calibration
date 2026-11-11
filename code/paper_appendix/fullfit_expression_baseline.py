from __future__ import annotations

# 论文附录复现代码。
# 问题四对照：估计高维全量拟合的样本内表达上限。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

import json
import math
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "A题 (1)"
OUTPUT_DIR = Path(__file__).resolve().parent / "verification_outputs"
FIG_DIR = Path(__file__).resolve().parent / "verification_figures"
REPORT_DIR = Path(__file__).resolve().parent / "verification_reports"

WINDOW_DATA = DATA_DIR / "attachment1_window_data.csv"
BASELINE_SUMMARY = DATA_DIR / "attachment4_baseline_summary.csv"
EVALUATE_SCRIPT = DATA_DIR / "evaluate_submission.py"

RANDOM_SEED = 20260709
warnings.filterwarnings("ignore", category=PerformanceWarning)


@dataclass
class FittedRidge:
    name: str
    columns: list[str]
    median: pd.Series
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    lam: float
    target_mode: str
    base_col: str


def ensureDirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


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
                "pass_group": bool(abs(mean_err) <= 0.2 and sd <= 0.040),
            }
        )
    return pd.DataFrame(rows)


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


def evaluatePredictions(data: pd.DataFrame, moxingTiji: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = data.copy()
    df["model_volume_m3"] = moxingTiji
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
                "max_abs_error_pct": float(df["model_error_pct"].abs().max()),
                "group_pass": f"{int(groups['pass_group'].sum())}/{len(groups)}",
                "group_pass_count": int(groups["pass_group"].sum()),
                "group_count": int(len(groups)),
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
    windows = df[["window_id", "date", "flow_point", "condition_note", "disturbance_id", "standard_volume_m3", "model_volume_m3", "model_error_pct"]]
    return summary, groups, windows


def loadData() -> pd.DataFrame:
    data = pd.read_csv(WINDOW_DATA, encoding="utf-8-sig")
    return data.sort_values("window_id").reset_index(drop=True)


def buildFeatures(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    allowed_prefixes = ("chord", "ab", "profile_", "dyn_", "zero_")
    base_cols = [
        "duration_s",
        "flow_point",
        "phys6_volume_m3",
        "owics_volume_m3",
        "lagrange_volume_m3",
        "equal_weight_volume_m3",
    ]
    online_cols = [c for c in data.columns if c.startswith(allowed_prefixes)]
    feats = data[base_cols + online_cols].copy()

    eps = 1e-12
    q = (data["flow_point"].astype(float) - 50.0) / 30.0
    feats["q"] = q
    feats["q2"] = q**2
    feats["q3"] = q**3
    feats["log_duration"] = np.log(np.maximum(data["duration_s"].astype(float), eps))
    feats["phys6_rate_proxy"] = data["phys6_volume_m3"].astype(float) / np.maximum(data["duration_s"].astype(float), eps)

    for col in ["owics_volume_m3", "lagrange_volume_m3", "equal_weight_volume_m3"]:
        feats[f"{col}_minus_phys6_frac"] = (data[col].astype(float) - data["phys6_volume_m3"].astype(float)) / np.maximum(data["phys6_volume_m3"].astype(float).abs(), eps)

    chord_cols = [f"chord{i}" for i in range(5)]
    ab_cols = [f"ab{i}" for i in range(5)]
    if all(c in data.columns for c in chord_cols):
        chord = data[chord_cols].astype(float)
        feats["chord_mean"] = chord.mean(axis=1)
        feats["chord_std"] = chord.std(axis=1)
        feats["chord_range"] = chord.max(axis=1) - chord.min(axis=1)
        feats["chord_center_minus_edges"] = chord["chord2"] - 0.5 * (chord["chord0"] + chord["chord4"])
        feats["chord_inner_skew_calc"] = chord["chord1"] - chord["chord3"]
        for col in chord_cols:
            feats[f"{col}_rel_mean"] = (chord[col] - feats["chord_mean"]) / np.maximum(feats["chord_mean"].abs(), eps)

    if all(c in data.columns for c in ab_cols):
        ab = data[ab_cols].astype(float)
        feats["ab_mean"] = ab.mean(axis=1)
        feats["ab_abs_mean"] = ab.abs().mean(axis=1)
        feats["ab_std"] = ab.std(axis=1)
        feats["ab_range"] = ab.max(axis=1) - ab.min(axis=1)

    interaction_roots = [
        "profile_top_bottom",
        "profile_center_all",
        "profile_edge_inner",
        "profile_inner_skew",
        "profile_ab_abs",
        "profile_swirl",
        "dyn_plateau_cv",
        "dyn_active_eq_s",
        "zero_rate_med",
        "zero_rate_mad",
        "phys6_rate_proxy",
        "lagrange_volume_m3_minus_phys6_frac",
        "owics_volume_m3_minus_phys6_frac",
    ]
    for col in interaction_roots:
        if col in feats.columns:
            feats[f"{col}_x_q"] = feats[col].astype(float) * q
            feats[f"{col}_x_q2"] = feats[col].astype(float) * q**2

    if mode == "compact":
        keep = [
            "duration_s",
            "flow_point",
            "phys6_volume_m3",
            "q",
            "q2",
            "q3",
            "phys6_rate_proxy",
            "lagrange_volume_m3_minus_phys6_frac",
            "owics_volume_m3_minus_phys6_frac",
            "profile_top_bottom",
            "profile_center_all",
            "profile_edge_inner",
            "profile_inner_skew",
            "profile_ab_abs",
            "profile_swirl",
            "dyn_plateau_cv",
            "dyn_active_eq_s",
            "zero_rate_med",
            "zero_rate_mad",
        ]
        return feats[[c for c in keep if c in feats.columns]].copy()

    if mode == "rich":
        return feats.copy()

    if mode == "poly":
        rich = feats.copy()
        roots = [c for c in rich.columns if c not in {"phys6_volume_m3", "owics_volume_m3", "lagrange_volume_m3", "equal_weight_volume_m3"}]
        numeric = rich[roots].astype(float)
        for col in roots:
            rich[f"{col}_sq"] = numeric[col] ** 2
        selected = [c for c in roots if c.startswith(("profile_", "dyn_", "zero_", "chord", "ab"))][:30]
        for i, a in enumerate(selected):
            for b in selected[i + 1 : min(i + 9, len(selected))]:
                rich[f"{a}_x_{b}"] = numeric[a] * numeric[b]
        return rich

    raise ValueError(f"Unknown feature mode: {mode}")


def targetFromMode(data: pd.DataFrame, mode: str, base_col: str) -> np.ndarray:
    base = np.maximum(data[base_col].astype(float).to_numpy(), 1e-12)
    std = data["standard_volume_m3"].astype(float).to_numpy()
    if mode == "relative_correction":
        return std / base - 1.0
    if mode == "log_ratio":
        return np.log(std / base)
    if mode == "direct_volume":
        return std
    raise ValueError(f"Unknown target mode: {mode}")


def volumeFromPrediction(data: pd.DataFrame, pred: np.ndarray, mode: str, base_col: str) -> np.ndarray:
    base = data[base_col].astype(float).to_numpy()
    if mode == "relative_correction":
        return base * (1.0 + pred)
    if mode == "log_ratio":
        return base * np.exp(pred)
    if mode == "direct_volume":
        return pred
    raise ValueError(f"Unknown target mode: {mode}")


def fitRidge(features: pd.DataFrame, y: np.ndarray, lam: float, name: str, target_mode: str, base_col: str) -> FittedRidge:
    clean = features.replace([np.inf, -np.inf], np.nan)
    median = clean.median(numeric_only=True).fillna(0.0)
    filled = clean.fillna(median)
    x = filled.to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    ident = np.eye(design.shape[1])
    ident[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + lam * ident, design.T @ y)
    return FittedRidge(name, list(features.columns), median, mean, scale, coef, lam, target_mode, base_col)


def predictRidge(model: FittedRidge, features: pd.DataFrame) -> np.ndarray:
    clean = features[model.columns].replace([np.inf, -np.inf], np.nan).fillna(model.median)
    x = clean.to_numpy(dtype=float)
    z = (x - model.mean) / model.scale
    design = np.column_stack([np.ones(len(z)), z])
    return design @ model.coef


def ridgeExperiment(data: pd.DataFrame, feature_mode: str, target_mode: str, base_col: str, lam: float) -> tuple[FittedRidge, np.ndarray]:
    features = buildFeatures(data, feature_mode)
    y = targetFromMode(data, target_mode, base_col)
    model = fitRidge(features, y, lam, f"{feature_mode}_{target_mode}_{base_col}_lam{lam:g}", target_mode, base_col)
    pred_target = predictRidge(model, features)
    volume = volumeFromPrediction(data, pred_target, target_mode, base_col)
    return model, volume


def scoreSummary(summary: pd.Series) -> float:
    penalty = 0.0
    penalty += max(0.0, float(summary["u_nor_L_pct"]) - 0.036) * 20.0
    penalty += max(0.0, float(summary["u_nor_r_pct"]) - 0.040) * 20.0
    penalty += max(0.0, float(summary["u_nor_d_pct"]) - 0.115) * 20.0
    penalty += (int(summary["group_count"]) - int(summary["group_pass_count"])) * 0.5
    return float(summary["mae_pct"]) + penalty


def runModelSearch(data: pd.DataFrame) -> tuple[pd.DataFrame, FittedRidge, np.ndarray]:
    rows = []
    best_model: FittedRidge | None = None
    best_volume: np.ndarray | None = None
    best_score = float("inf")

    baseline_cols = ["phys6_volume_m3", "lagrange_volume_m3", "owics_volume_m3", "equal_weight_volume_m3"]
    for col in baseline_cols:
        summary, _, _ = evaluatePredictions(data, data[col].astype(float).to_numpy())
        row = summary.iloc[0].to_dict()
        row.update({"model_name": col, "feature_count": 0, "lambda": None, "target_mode": "baseline", "base_col": col})
        row["selection_score"] = scoreSummary(summary.iloc[0])
        rows.append(row)

    feature_modes = ["compact", "rich", "poly"]
    target_modes = ["relative_correction", "log_ratio", "direct_volume"]
    base_cols = ["phys6_volume_m3", "lagrange_volume_m3"]
    lambdas = [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    for feature_mode in feature_modes:
        feature_count = buildFeatures(data, feature_mode).shape[1]
        for target_mode in target_modes:
            for base_col in base_cols:
                if target_mode == "direct_volume" and base_col != "phys6_volume_m3":
                    continue
                for lam in lambdas:
                    try:
                        model, volume = ridgeExperiment(data, feature_mode, target_mode, base_col, lam)
                        summary, _, _ = evaluatePredictions(data, volume)
                    except np.linalg.LinAlgError:
                        continue
                    row = summary.iloc[0].to_dict()
                    row.update(
                        {
                            "model_name": model.name,
                            "feature_count": feature_count,
                            "lambda": lam,
                            "target_mode": target_mode,
                            "base_col": base_col,
                        }
                    )
                    row["selection_score"] = scoreSummary(summary.iloc[0])
                    rows.append(row)
                    if row["selection_score"] < best_score:
                        best_score = row["selection_score"]
                        best_model = model
                        best_volume = volume

    if best_model is None or best_volume is None:
        raise RuntimeError("No ridge model fitted")

    metrics = pd.DataFrame(rows).sort_values(["pass_all_3", "selection_score"], ascending=[False, True])
    return metrics, best_model, best_volume


def leaveOneDateCv(data: pd.DataFrame, model_template: FittedRidge, feature_mode: str) -> pd.DataFrame:
    yuceJieguo = []
    features_all = buildFeatures(data, feature_mode)
    for date in sorted(data["date"].unique()):
        train_idx = data["date"] != date
        test_idx = data["date"] == date
        y = targetFromMode(data.loc[train_idx], model_template.target_mode, model_template.base_col)
        model = fitRidge(
            features_all.loc[train_idx],
            y,
            model_template.lam,
            f"cv_{date}",
            model_template.target_mode,
            model_template.base_col,
        )
        pred_target = predictRidge(model, features_all.loc[test_idx])
        volume = volumeFromPrediction(data.loc[test_idx], pred_target, model.target_mode, model.base_col)
        fold = data.loc[test_idx, ["window_id", "date", "flow_point", "standard_volume_m3"]].copy()
        fold["cv_model_volume_m3"] = volume
        fold["cv_error_pct"] = relativeErrorPct(fold["cv_model_volume_m3"], fold["standard_volume_m3"])
        yuceJieguo.append(fold)
    return pd.concat(yuceJieguo, ignore_index=True).sort_values("window_id")


def exportSubmission(data: pd.DataFrame, volume: np.ndarray, path: Path) -> None:
    sub = pd.DataFrame({"window_id": data["window_id"], "model_volume_m3": volume})
    sub.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def saveModelArtifacts(model: FittedRidge) -> None:
    coef_df = pd.DataFrame({"term": ["intercept"] + model.columns, "coefficient": model.coef})
    coef_df.to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    meta = {
        "artifact_role": "full-data interpolation submission; not a generalization result",
        "name": model.name,
        "lambda": model.lam,
        "target_mode": model.target_mode,
        "base_col": model.base_col,
        "feature_count": len(model.columns),
        "forbidden_online_inputs_excluded": [
            "standard_volume_m3",
            "standard_rate_m3h",
            "date",
            "start_time",
            "window_id",
            "condition_note",
            "is_disturbed",
            "disturbance_id",
            "disturbance_angle_deg",
        ],
    }
    (OUTPUT_DIR / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def runOfficialEvaluator(submission_path: Path) -> None:
    official_dir = OUTPUT_DIR / "official_evaluation"
    if official_dir.exists():
        shutil.rmtree(official_dir)
    cmd = [sys.executable, str(EVALUATE_SCRIPT), str(submission_path), "--output-dir", str(official_dir)]
    subprocess.run(cmd, cwd=str(DATA_DIR), check=True)


def registerFonts() -> str:
    font_candidates = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    for path in font_candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont("CNFont", str(path)))
                return "CNFont"
            except Exception:
                continue
    return "Helvetica"


def drawBarChart(path: Path, title: str, labels: list[str], values: list[float], y_label: str, font: str) -> None:
    width, height = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin - 30
    c.setFont(font, 14)
    c.drawCentredString(width / 2, height - 35, title)
    max_v = max(values) if values else 1.0
    max_v = max_v * 1.15 if max_v > 0 else 1.0
    bar_gap = 16
    bar_w = (plot_w - bar_gap * (len(labels) + 1)) / max(1, len(labels))
    x0 = margin + bar_gap
    y0 = margin
    c.setStrokeColor(colors.black)
    c.line(margin, y0, margin, y0 + plot_h)
    c.line(margin, y0, margin + plot_w, y0)
    c.setFont(font, 10)
    c.drawString(12, y0 + plot_h / 2, y_label)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = x0 + i * (bar_w + bar_gap)
        h = value / max_v * plot_h
        c.setFillColor(colors.HexColor("#4C78A8"))
        c.rect(x, y0, bar_w, h, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawCentredString(x + bar_w / 2, y0 - 18, label)
        c.drawCentredString(x + bar_w / 2, y0 + h + 8, f"{value:.4f}")
    c.save()


def drawErrorByFlow(path: Path, data: pd.DataFrame, font: str) -> None:
    width, height = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin - 30
    c.setFont(font, 14)
    c.drawCentredString(width / 2, height - 35, "各流量点模型误差分布")
    stats = data.groupby("flow_point")["model_error_pct"].agg(["mean", "std", "min", "max"]).reset_index()
    stats.to_csv(FIG_DIR / "fig_error_by_flow_data.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    y_abs = max(float(stats["max"].abs().max()), float(stats["min"].abs().max()), 0.01)
    y_abs *= 1.2
    x_vals = stats["flow_point"].astype(float).to_numpy()
    xs = margin + (x_vals - x_vals.min()) / max(1e-9, x_vals.max() - x_vals.min()) * plot_w
    y_mid = margin + plot_h / 2
    c.setStrokeColor(colors.black)
    c.line(margin, y_mid, margin + plot_w, y_mid)
    c.line(margin, margin, margin, margin + plot_h)
    c.setFont(font, 10)
    c.drawString(12, y_mid, "误差(%)")
    for _, row in stats.iterrows():
        x = margin + (float(row["flow_point"]) - x_vals.min()) / max(1e-9, x_vals.max() - x_vals.min()) * plot_w
        y_mean = y_mid + float(row["mean"]) / y_abs * plot_h / 2
        y_min = y_mid + float(row["min"]) / y_abs * plot_h / 2
        y_max = y_mid + float(row["max"]) / y_abs * plot_h / 2
        c.setStrokeColor(colors.HexColor("#E45756"))
        c.line(x, y_min, x, y_max)
        c.setFillColor(colors.HexColor("#E45756"))
        c.circle(x, y_mean, 4, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawCentredString(x, margin - 18, str(int(row["flow_point"])))
    c.save()


def drawHistogram(path: Path, errors: pd.Series, font: str) -> None:
    width, height = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin - 30
    c.setFont(font, 14)
    c.drawCentredString(width / 2, height - 35, "最终模型相对误差分布")
    counts, edges = np.histogram(errors.to_numpy(dtype=float), bins=16)
    hist_df = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})
    hist_df.to_csv(FIG_DIR / "fig_error_distribution_data.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    max_count = max(int(counts.max()), 1)
    bar_w = plot_w / len(counts)
    y0 = margin
    c.line(margin, y0, margin + plot_w, y0)
    c.line(margin, y0, margin, y0 + plot_h)
    c.setFont(font, 10)
    c.drawString(12, y0 + plot_h / 2, "窗口数")
    for i, count in enumerate(counts):
        x = margin + i * bar_w
        h = count / max_count * plot_h
        c.setFillColor(colors.HexColor("#72B7B2"))
        c.rect(x + 2, y0, bar_w - 4, h, stroke=0, fill=1)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2, margin - 22, "相对误差区间(%)")
    c.save()


def makeFigures(data: pd.DataFrame, metrics: pd.DataFrame, final_windows: pd.DataFrame) -> None:
    font = registerFonts()
    baseline = pd.read_csv(BASELINE_SUMMARY, encoding="utf-8-sig")
    labels = baseline["method"].tolist()
    values = baseline["mae_pct"].astype(float).tolist()
    pd.DataFrame({"method": labels, "mae_pct": values}).to_csv(
        FIG_DIR / "fig_model_mae_comparison_data.csv", index=False, encoding="utf-8-sig", float_format="%.12g"
    )
    drawBarChart(FIG_DIR / "fig_model_mae_comparison.pdf", "基础积分方法 MAE 对比", labels, values, "MAE(%)", font)
    drawErrorByFlow(FIG_DIR / "fig_error_by_flow.pdf", final_windows, font)
    drawHistogram(FIG_DIR / "fig_error_distribution.pdf", final_windows["model_error_pct"], font)


def writeResultsReport(
    metrics: pd.DataFrame,
    final_summary: pd.DataFrame,
    final_groups: pd.DataFrame,
    cv_predictions: pd.DataFrame,
    model: FittedRidge,
) -> None:
    best = final_summary.iloc[0]
    cv_mae = float(cv_predictions["cv_error_pct"].abs().mean())
    cv_max = float(cv_predictions["cv_error_pct"].abs().max())
    top = metrics.head(8)[["model_name", "mae_pct", "group_pass", "u_nor_L_pct", "u_nor_r_pct", "u_nor_d_pct", "pass_all_3"]]
    worst_groups = final_groups.sort_values(["pass_group", "repeatability_sd_pct", "mean_error_pct"], ascending=[True, False, False]).head(8)
    mae = float(best["mae_pct"])
    mean_err = float(best["mean_error_pct"])
    max_abs = float(best["max_abs_error_pct"])
    u_l = float(best["u_nor_L_pct"])
    u_r = float(best["u_nor_r_pct"])
    u_d = float(best["u_nor_d_pct"])

    report = f"""# 计算结果

## 运行环境

- Python: `{sys.version.split()[0]}`
- 主要依赖：`numpy`、`pandas`、`reportlab`
- 随机种子：`{RANDOM_SEED}`
- 主程序：`code/model_experiment.py`

## 数据读取与预处理

读取 `attachment1_window_data.csv` 的 159 个窗口样本。运行时输入仅使用持续时间、流量点、基础积分估计量、五声道观测量、A/B 差异、剖面特征、动态特征和零点特征。最终模型未使用 `standard_volume_m3`、`standard_rate_m3h`、`date`、`start_time`、`window_id`、`condition_note`、`is_disturbed`、`disturbance_id`、`disturbance_angle_deg` 作为在线输入。

## 问题一结果

基础积分方法中，Phys6 的整体 MAE 为 0.180653%，Lagrange 为 0.183613%，OWICS 和等权积分误差更大。因此实验以 Phys6 和 Lagrange 为主要物理基准，并训练相对修正项。

## 问题二与问题三模型

最终采用 `{model.name}`。模型形式为岭回归残差补偿，目标模式为 `{model.target_mode}`，物理基准列为 `{model.base_col}`，特征数为 `{len(model.columns)}`，正则化参数为 `{model.lam:g}`。该模型属于全量标定型模型：参数在全部已给定窗口标签上估计，再固化用于输出本题提交文件；运行时输入仍只包含在线可获得的流量与声道特征。

## 全量插值提交评价（非泛化结果）

- 窗口数：{int(best['window_count'])}
- MAE：{mae:.6e}%
- 平均误差：{mean_err:.6e}%
- 最大绝对误差：{max_abs:.6e}%
- day-flow 分组通过数：{best['group_pass']}
- $u_{{\\mathrm{{nor}},L}}$：{u_l:.6e}%
- $u_{{\\mathrm{{nor}},r}}$：{u_r:.6e}%
- $u_{{\\mathrm{{nor}},d}}$：{u_d:.6e}%
- 三项综合指标是否全部达标：{bool(best['pass_all_3'])}

## 留一日期验证

按日期留一验证的 MAE 为 {cv_mae:.6f}%，最大绝对误差为 {cv_max:.6f}%。这说明 159 个窗口的小样本数据难以支撑高维补偿项跨日期外推；若比赛存在未知测试窗口，应优先改用更强正则化或更低维的稳健模型。当前 `submission_fullfit_interpolation.csv` 仅用于记录全量插值基准，不是最终提交模型。

## 候选模型摘要

{markdownTable(top)}

## 最差分组检查

{markdownTable(worst_groups)}

## 输出文件

- `code/outputs/submission_fullfit_interpolation.csv`：题目给定 159 个窗口的全量插值基准文件
- `code/outputs/experiment_metrics.csv`：候选模型评价
- `code/outputs/model_coefficients.csv`：最终模型系数
- `code/outputs/final_evaluation_summary.csv`：最终模型总体指标
- `code/outputs/final_evaluation_groups.csv`：day-flow 分组指标
- `code/outputs/final_evaluation_windows.csv`：逐窗口误差
- `code/outputs/cv_by_date_predictions.csv`：留一日期验证预测
- `code/outputs/official_evaluation/`：题目官方评价脚本输出

## 图表

- `figures/fig_model_mae_comparison.pdf`
- `figures/fig_error_by_flow.pdf`
- `figures/fig_error_distribution.pdf`

## 可复现运行方式

```powershell
python code/model_experiment.py
```
"""
    (REPORT_DIR / "RESULTS_REPORT.md").write_text(report, encoding="utf-8")


def markdownTable(df: pd.DataFrame) -> str:
    if df.empty:
        return "无"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.astype(str).to_numpy()]
    return "\n".join([header, sep] + rows)


def main() -> None:
    ensureDirs()
    np.random.seed(RANDOM_SEED)
    data = loadData()

    metrics, best_model, best_volume = runModelSearch(data)
    final_summary, final_groups, final_windows = evaluatePredictions(data, best_volume)

    metrics.to_csv(OUTPUT_DIR / "experiment_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    final_summary.to_csv(OUTPUT_DIR / "fullfit_interpolation_evaluation_summary.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    final_groups.to_csv(OUTPUT_DIR / "fullfit_interpolation_evaluation_groups.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    final_windows.to_csv(OUTPUT_DIR / "fullfit_interpolation_evaluation_windows.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    # This model is a full-data interpolation benchmark, not the final
    # constrained D0-RBF postcalibrated dual-kernel KRR.
    # submission. Keep the filename distinct so rerunning this comparison
    # cannot overwrite the official-targeted result.
    submission = OUTPUT_DIR / "submission_fullfit_interpolation.csv"
    exportSubmission(data, best_volume, submission)
    saveModelArtifacts(best_model)

    feature_mode = best_model.name.split("_", 1)[0]
    cv_predictions = leaveOneDateCv(data, best_model, feature_mode)
    cv_predictions.to_csv(OUTPUT_DIR / "cv_by_date_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    makeFigures(data, metrics, final_windows)
    runOfficialEvaluator(submission)
    writeResultsReport(metrics, final_summary, final_groups, cv_predictions, best_model)

    print("Best model:", best_model.name)
    print(final_summary.to_string(index=False))
    print("Submission:", submission)
    print("Report:", REPORT_DIR / "RESULTS_REPORT.md")


if __name__ == "__main__":
    main()
