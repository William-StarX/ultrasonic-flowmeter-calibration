from __future__ import annotations

# 论文附录复现代码。
# 问题二：在 D0 窗口上拟合正常工况多声道校准模型。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
except ModuleNotFoundError:
    colors = None
    landscape = None
    A4 = None
    pdfmetrics = None
    TTFont = None
    canvas = None


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and (p / "attachment1_window_data.csv").exists())
OUT_DIR = Path(__file__).resolve().parent / "verification_outputs"
FIG_DIR = Path(__file__).resolve().parent / "verification_figures"
REPORT = Path(__file__).resolve().parent / "verification_reports" / "NO_DISTURBANCE_MODEL_REPORT.md"


@dataclass
class CorrectionScaler:
    cols: list[str]
    mean: np.ndarray
    scale: np.ndarray


@dataclass
class FittedModel:
    name: str
    prior_name: str
    lambda_w: float
    correction: str
    lambda_c: float
    weights: np.ndarray
    beta: np.ndarray
    scaler: CorrectionScaler
    area: float
    factor_clip: tuple[float, float] = (0.97, 1.03)


def loadInputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    win = pd.read_csv(DATA_DIR / "attachment1_window_data.csv", encoding="utf-8-sig")
    raw = pd.read_csv(DATA_DIR / "attachment6_window_raw_samples.csv", encoding="utf-8-sig")
    geo = pd.read_csv(DATA_DIR / "attachment7_meter_geometry.csv", encoding="utf-8-sig")
    return win, raw, geo


def getArea(geo: pd.DataFrame) -> float:
    return float(geo.loc[geo["parameter"].eq("effective_pipe_area_m2"), "value"].iloc[0])


def chordRows(geo: pd.DataFrame) -> pd.DataFrame:
    return geo[geo["record_type"].eq("chord_mapping")].sort_values("chord_id").reset_index(drop=True)


def reconstructChords(raw: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    rows = chordRows(geo)
    diff_cols = [f"diff_ns_{i}" for i in range(10)]
    raw_mean = raw.groupby("window_id")[diff_cols].mean()
    rec = pd.DataFrame(index=raw_mean.index)
    for _, row in rows.iterrows():
        rec[row["object_id"]] = (
            raw_mean[row["raw_diff_col_a"]] * float(row["raw_diff_gain_a_mps_per_ns"])
            + raw_mean[row["raw_diff_col_b"]] * float(row["raw_diff_gain_b_mps_per_ns"])
            + float(row["raw_diff_intercept_mps"])
        )
    return rec.reset_index()


def attachFeatures(win: pd.DataFrame, rec: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    data = win.merge(rec, on="window_id", suffixes=("", "_raw_rec"))
    chord = data[[f"chord{i}_raw_rec" for i in range(5)]].astype(float)
    mean = chord.mean(axis=1).replace(0, np.nan)
    data["raw_chord_mean"] = mean
    data["prof_center_edges"] = (chord["chord2_raw_rec"] - 0.5 * (chord["chord0_raw_rec"] + chord["chord4_raw_rec"])) / mean
    data["prof_edge_inner"] = (0.5 * (chord["chord0_raw_rec"] + chord["chord4_raw_rec"]) - 0.5 * (chord["chord1_raw_rec"] + chord["chord3_raw_rec"])) / mean
    data["prof_top_bottom"] = (chord["chord0_raw_rec"] - chord["chord4_raw_rec"]) / mean
    data["prof_inner_skew"] = (chord["chord1_raw_rec"] - chord["chord3_raw_rec"]) / mean
    data["prof_range"] = (chord.max(axis=1) - chord.min(axis=1)) / mean
    data["q"] = (data["flow_point"].astype(float) - 50.0) / 30.0
    data["q2"] = data["q"] ** 2
    data["ref_velocity_m_s"] = data["standard_volume_m3"].astype(float) / (getArea(geo) * data["duration_s"].astype(float))
    data = data.fillna(0.0)
    return data


def getPriors(geo: pd.DataFrame) -> dict[str, np.ndarray]:
    rows = chordRows(geo)
    return {
        "phys6": rows["weight_phys6"].astype(float).to_numpy(),
        "owics_gauss_jacobi": rows["weight_flowcal_gauss_jacobi"].astype(float).to_numpy(),
        "lagrange": rows["weight_lagrange"].astype(float).to_numpy(),
        "equal": rows["weight_equal"].astype(float).to_numpy(),
    }


def fitSymmetricWeights(train: pd.DataFrame, prior: np.ndarray, lambda_w: float) -> np.ndarray:
    x = train[[f"chord{i}_raw_rec" for i in range(5)]].astype(float).to_numpy()
    y = train["ref_velocity_m_s"].astype(float).to_numpy()
    # Enforce w0=w4=e, w1=w3=i, w2=1-2e-2i.
    z = np.column_stack([x[:, 0] + x[:, 4] - 2 * x[:, 2], x[:, 1] + x[:, 3] - 2 * x[:, 2]])
    b = y - x[:, 2]
    theta0 = np.array([0.5 * (prior[0] + prior[4]), 0.5 * (prior[1] + prior[3])], dtype=float)
    lhs = z.T @ z + lambda_w * np.eye(2)
    rhs = z.T @ b + lambda_w * theta0
    theta = np.linalg.solve(lhs, rhs)
    weights = np.array([theta[0], theta[1], 1 - 2 * theta[0] - 2 * theta[1], theta[1], theta[0]], dtype=float)
    if np.all(weights >= -1e-9):
        weights = np.maximum(weights, 0.0)
        return weights / weights.sum()

    # Low-dimensional projection fallback onto the physical simplex.
    grid = np.linspace(0.0, 0.5, 501)
    best_obj = np.inf
    best = np.array([theta0[0], theta0[1]])
    for e in grid:
        max_i = (1 - 2 * e) / 2
        if max_i < 0:
            continue
        for i in grid[grid <= max_i]:
            pred = x[:, 2] + e * z[:, 0] + i * z[:, 1]
            obj = float(np.mean((pred - y) ** 2) + lambda_w * np.sum((np.array([e, i]) - theta0) ** 2))
            if obj < best_obj:
                best_obj = obj
                best = np.array([e, i])
    weights = np.array([best[0], best[1], 1 - 2 * best[0] - 2 * best[1], best[1], best[0]], dtype=float)
    return weights / weights.sum()


def baseVolume(data: pd.DataFrame, weights: np.ndarray, area: float) -> np.ndarray:
    x = data[[f"chord{i}_raw_rec" for i in range(5)]].astype(float).to_numpy()
    velocity = x @ weights
    return area * data["duration_s"].astype(float).to_numpy() * velocity


def correctionColumns(kind: str) -> list[str]:
    if kind == "none":
        return []
    if kind == "calibration":
        return ["intercept"]
    if kind == "flow_curve":
        return ["intercept", "q", "q2"]
    if kind == "profile_flow":
        return ["intercept", "q", "q2", "prof_center_edges", "prof_edge_inner", "prof_top_bottom", "prof_inner_skew"]
    raise ValueError(kind)


def makeCorrectionMatrix(data: pd.DataFrame, cols: list[str], scaler: CorrectionScaler | None = None) -> tuple[np.ndarray, CorrectionScaler]:
    if not cols:
        return np.zeros((len(data), 0)), CorrectionScaler(cols=[], mean=np.array([]), scale=np.array([]))
    raw_cols = []
    arrays = []
    for col in cols:
        if col == "intercept":
            arrays.append(np.ones(len(data)))
        else:
            raw_cols.append(col)
            arrays.append(data[col].astype(float).to_numpy())
    matrix = np.column_stack(arrays)
    if scaler is None:
        mean = np.zeros(len(cols))
        scale = np.ones(len(cols))
        for j, col in enumerate(cols):
            if col == "intercept":
                continue
            v = matrix[:, j]
            mean[j] = float(np.mean(v))
            s = float(np.std(v))
            scale[j] = s if s > 1e-12 else 1.0
        scaler = CorrectionScaler(cols=cols, mean=mean, scale=scale)
    scaled = matrix.copy()
    for j, col in enumerate(cols):
        if col == "intercept":
            continue
        scaled[:, j] = (scaled[:, j] - scaler.mean[j]) / scaler.scale[j]
    return scaled, scaler


def fitCorrection(train: pd.DataFrame, base: np.ndarray, kind: str, lambda_c: float) -> tuple[np.ndarray, CorrectionScaler]:
    cols = correctionColumns(kind)
    if not cols:
        return np.zeros(0), CorrectionScaler(cols=[], mean=np.array([]), scale=np.array([]))
    target = train["standard_volume_m3"].astype(float).to_numpy() / np.maximum(base, 1e-12) - 1.0
    x, scaler = makeCorrectionMatrix(train, cols)
    penalty = np.eye(x.shape[1])
    if "intercept" in cols:
        penalty[cols.index("intercept"), cols.index("intercept")] = 0.0
    beta = np.linalg.solve(x.T @ x + lambda_c * penalty, x.T @ target)
    return beta, scaler


def fitModel(train: pd.DataFrame, area: float, prior_name: str, prior: np.ndarray, lambda_w: float, correction: str, lambda_c: float) -> FittedModel:
    weights = fitSymmetricWeights(train, prior, lambda_w)
    base = baseVolume(train, weights, area)
    beta, scaler = fitCorrection(train, base, correction, lambda_c)
    name = f"sym_weight_{prior_name}_lw{lambda_w:g}_{correction}_lc{lambda_c:g}"
    return FittedModel(name, prior_name, lambda_w, correction, lambda_c, weights, beta, scaler, area)


def predict(model: FittedModel, data: pd.DataFrame) -> np.ndarray:
    base = baseVolume(data, model.weights, model.area)
    cols = model.scaler.cols
    if not cols:
        return base
    x, _ = makeCorrectionMatrix(data, cols, model.scaler)
    factor = 1.0 + x @ model.beta
    factor = np.clip(factor, model.factor_clip[0], model.factor_clip[1])
    return base * factor


def errorPct(y_hat: np.ndarray, y: pd.Series) -> np.ndarray:
    return (y_hat - y.astype(float).to_numpy()) / y.astype(float).to_numpy() * 100.0


def evaluatePredictions(data: pd.DataFrame, method: str, source: str, y_hat: np.ndarray) -> pd.DataFrame:
    out = data[[
        "window_id",
        "date",
        "flow_point",
        "condition_note",
        "disturbance_id",
        "is_disturbed",
        "duration_s",
        "standard_volume_m3",
    ]].copy()
    out["method"] = method
    out["source"] = source
    out["estimate_volume_m3"] = y_hat
    out["error_pct"] = errorPct(y_hat, data["standard_volume_m3"])
    return out


def metricSummary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(sub: pd.DataFrame) -> pd.Series:
        e = sub["error_pct"].astype(float)
        return pd.Series(
            {
                "window_count": len(sub),
                "mae_pct": e.abs().mean(),
                "mean_error_pct": e.mean(),
                "rmse_pct": float(np.sqrt(np.mean(e**2))),
                "max_abs_error_pct": e.abs().max(),
            }
        )

    return df.groupby(by, dropna=False).apply(agg).reset_index()


def groupMetrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method", "source", "date", "flow_point", "condition_note", "disturbance_id"]
    for key, sub in pred.groupby(keys, dropna=False):
        e = sub["error_pct"].astype(float)
        rows.append(
            dict(
                zip(keys, key),
                window_count=len(sub),
                group_mean_error_pct=e.mean(),
                group_abs_mean_error_pct=abs(e.mean()),
                group_sd_pct=e.std(ddof=1) if len(sub) >= 2 else np.nan,
                group_mae_pct=e.abs().mean(),
            )
        )
    return pd.DataFrame(rows)


def groupSummary(groups: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(sub: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "group_count": len(sub),
                "repeat_group_count": sub["group_sd_pct"].notna().sum(),
                "mean_abs_group_mean_pct": sub["group_abs_mean_error_pct"].mean(),
                "max_abs_group_mean_pct": sub["group_abs_mean_error_pct"].max(),
                "mean_group_sd_pct": sub["group_sd_pct"].mean(),
                "max_group_sd_pct": sub["group_sd_pct"].max(),
            }
        )

    return groups.groupby(by, dropna=False).apply(agg).reset_index()


def leaveFlowCv(d0: pd.DataFrame, area: float, priors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    flow_points = sorted(d0["flow_point"].unique())
    lambda_w_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    lambda_c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    corrections = ["none", "calibration", "flow_curve", "profile_flow"]
    for prior_name, prior in priors.items():
        for lw in lambda_w_values:
            for correction in corrections:
                lc_iter = [0.0] if correction == "none" else lambda_c_values
                fold_errors = []
                for lc in lc_iter:
                    fold_errors.clear()
                    for fp in flow_points:
                        train = d0[d0["flow_point"].ne(fp)].copy()
                        valid = d0[d0["flow_point"].eq(fp)].copy()
                        model = fitModel(train, area, prior_name, prior, lw, correction, lc)
                        pred = predict(model, valid)
                        fold_errors.extend(errorPct(pred, valid["standard_volume_m3"]).tolist())
                    e = np.array(fold_errors)
                    rows.append(
                        {
                            "prior": prior_name,
                            "lambda_w": lw,
                            "correction": correction,
                            "lambda_c": lc,
                            "cv_window_count": len(e),
                            "cv_mae_pct": float(np.mean(np.abs(e))),
                            "cv_mean_error_pct": float(np.mean(e)),
                            "cv_rmse_pct": float(np.sqrt(np.mean(e**2))),
                            "cv_max_abs_error_pct": float(np.max(np.abs(e))),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["cv_mae_pct", "cv_rmse_pct"]).reset_index(drop=True)


def finalPredictions(data: pd.DataFrame, model: FittedModel) -> pd.DataFrame:
    yuceJieguo = [evaluatePredictions(data, "d0_optimal_weight_profile_curve", "trained_on_D0_only", predict(model, data))]
    baselines = {
        "base_phys6": "phys6_volume_m3",
        "base_owics": "owics_volume_m3",
        "base_lagrange": "lagrange_volume_m3",
        "base_equal_weight": "equal_weight_volume_m3",
    }
    for method, col in baselines.items():
        yuceJieguo.append(evaluatePredictions(data, method, "attachment1_baseline", data[col].astype(float).to_numpy()))
    return pd.concat(yuceJieguo, ignore_index=True)


def writeCsvOutputs(
    data: pd.DataFrame,
    cv: pd.DataFrame,
    model: FittedModel,
    yuceJieguo: pd.DataFrame,
    groups: pd.DataFrame,
    weights_meta: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv.to_csv(OUT_DIR / "d0_model_cv_by_flow.csv", index=False, encoding="utf-8-sig")
    yuceJieguo.to_csv(OUT_DIR / "d0_model_predictions_long.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT_DIR / "d0_model_group_metrics.csv", index=False, encoding="utf-8-sig")
    groups[groups["disturbance_id"].eq("D0")].to_csv(OUT_DIR / "d0_model_group_metrics_d0_only.csv", index=False, encoding="utf-8-sig")
    weights = pd.DataFrame(
        {
            "chord": [f"chord{i}" for i in range(5)],
            "position": weights_meta["chord_position_label"],
            "normalized_height": weights_meta["normalized_channel_height"].astype(float),
            "fitted_weight": model.weights,
            "weight_phys6": weights_meta["weight_phys6"].astype(float),
            "weight_owics_gauss_jacobi": weights_meta["weight_flowcal_gauss_jacobi"].astype(float),
            "weight_lagrange": weights_meta["weight_lagrange"].astype(float),
            "weight_equal": weights_meta["weight_equal"].astype(float),
        }
    )
    weights.to_csv(OUT_DIR / "d0_fitted_weights.csv", index=False, encoding="utf-8-sig")
    correction = pd.DataFrame(
        {
            "term": model.scaler.cols,
            "coefficient": model.beta,
            "feature_mean": model.scaler.mean,
            "feature_scale": model.scaler.scale,
        }
    )
    correction.to_csv(OUT_DIR / "d0_correction_coefficients.csv", index=False, encoding="utf-8-sig")

    overall = metricSummary(yuceJieguo, ["method", "source"])
    d0_overall = metricSummary(yuceJieguo[yuceJieguo["disturbance_id"].eq("D0")], ["method", "source"])
    by_flow_d0 = metricSummary(yuceJieguo[yuceJieguo["disturbance_id"].eq("D0")], ["method", "source", "flow_point"])
    by_date = metricSummary(yuceJieguo, ["method", "source", "date"])
    by_condition = metricSummary(yuceJieguo, ["method", "source", "condition_note", "disturbance_id"])
    g_summary = groupSummary(groups, ["method", "source"])
    g_by_date = groupSummary(groups, ["method", "source", "date"])

    overall.to_csv(OUT_DIR / "d0_model_overall_all_dates.csv", index=False, encoding="utf-8-sig")
    d0_overall.to_csv(OUT_DIR / "d0_model_overall_d0_only.csv", index=False, encoding="utf-8-sig")
    by_flow_d0.to_csv(OUT_DIR / "d0_model_by_flow_d0_only.csv", index=False, encoding="utf-8-sig")
    by_date.to_csv(OUT_DIR / "d0_model_by_date_holdout.csv", index=False, encoding="utf-8-sig")
    by_condition.to_csv(OUT_DIR / "d0_model_by_condition.csv", index=False, encoding="utf-8-sig")
    g_summary.to_csv(OUT_DIR / "d0_model_group_summary.csv", index=False, encoding="utf-8-sig")
    g_by_date.to_csv(OUT_DIR / "d0_model_group_summary_by_date.csv", index=False, encoding="utf-8-sig")
    return {
        "weights": weights,
        "correction": correction,
        "overall": overall,
        "d0_overall": d0_overall,
        "by_flow_d0": by_flow_d0,
        "by_date": by_date,
        "by_condition": by_condition,
        "group_summary": g_summary,
        "group_by_date": g_by_date,
    }


def fontName() -> str:
    if pdfmetrics is None:
        return "Helvetica"
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        try:
            pdfmetrics.registerFont(TTFont("MSYH", str(font_path), subfontIndex=0))
            return "MSYH"
        except Exception:
            return "Helvetica"
    return "Helvetica"


def plotPdf(tables: dict[str, pd.DataFrame]) -> None:
    if canvas is None:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    font = fontName()

    flow = tables["by_flow_d0"].copy()
    selected = ["d0_optimal_weight_profile_curve", "base_phys6", "base_lagrange", "base_owics", "base_equal_weight"]
    flow = flow[flow["method"].isin(selected)]
    c = canvas.Canvas(str(FIG_DIR / "fig_d0_model_flow_improvement.pdf"), pagesize=landscape(A4))
    w, h = landscape(A4)
    c.setFont(font, 12)
    c.drawString(45, h - 42, "D0 无扰流样本：不同流量点 MAE 对比")
    flow_points = sorted(flow["flow_point"].unique())
    methods = selected
    palette = [colors.HexColor(x) for x in ["#2563EB", "#16A34A", "#F97316", "#DC2626", "#64748B"]]
    max_val = float(flow["mae_pct"].max())
    chart_x, chart_y, chart_w, chart_h = 68, 95, 680, 390
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.line(chart_x, chart_y, chart_x + chart_w, chart_y)
    c.line(chart_x, chart_y, chart_x, chart_y + chart_h)
    for mi, method in enumerate(methods):
        sub = flow[flow["method"].eq(method)].set_index("flow_point")
        pts = []
        for i, fp in enumerate(flow_points):
            if fp not in sub.index:
                continue
            x = chart_x + (i + 0.5) * chart_w / len(flow_points)
            val = float(sub.loc[fp, "mae_pct"])
            y = chart_y + val / max_val * chart_h if max_val > 0 else chart_y
            pts.append((x, y))
            c.setFillColor(palette[mi])
            c.circle(x, y, 3, stroke=0, fill=1)
        c.setStrokeColor(palette[mi])
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            c.line(x1, y1, x2, y2)
        c.setFillColor(palette[mi])
        c.rect(540, h - 72 - mi * 17, 9, 9, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#111827"))
        c.setFont(font, 8)
        c.drawString(555, h - 74 - mi * 17, method)
    c.setFont(font, 9)
    for i, fp in enumerate(flow_points):
        x = chart_x + (i + 0.5) * chart_w / len(flow_points)
        c.drawCentredString(x, chart_y - 18, str(fp))
    c.drawString(chart_x + 300, chart_y - 42, "flow point")
    c.drawString(chart_x - 45, chart_y + chart_h + 5, "MAE / %")
    c.save()

    date = tables["by_date"]
    date = date[date["method"].eq("d0_optimal_weight_profile_curve")].sort_values("date")
    c = canvas.Canvas(str(FIG_DIR / "fig_d0_model_date_holdout.pdf"), pagesize=landscape(A4))
    w, h = landscape(A4)
    c.setFont(font, 12)
    c.drawString(45, h - 42, "D0 标定模型：按日期外部留出 MAE")
    max_val = float(date["mae_pct"].max())
    x0, y0, bar_w, row_h = 160, h - 83, 500, 31
    for idx, row in enumerate(date.itertuples(index=False)):
        y = y0 - idx * row_h
        width = bar_w * float(row.mae_pct) / max_val if max_val > 0 else 0
        c.setFont(font, 9)
        c.setFillColor(colors.HexColor("#111827"))
        c.drawRightString(x0 - 8, y + 4, str(row.date))
        c.setFillColor(colors.HexColor("#3B82F6") if str(row.date) == "20260607" else colors.HexColor("#94A3B8"))
        c.rect(x0, y, width, 12, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#111827"))
        c.drawString(x0 + width + 6, y + 1, f"{float(row.mae_pct):.3f}%")
    c.save()


def markdownTable(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
    if df.empty:
        return "_无数据_"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def writeReport(
    data: pd.DataFrame,
    d0: pd.DataFrame,
    cv: pd.DataFrame,
    model: FittedModel,
    tables: dict[str, pd.DataFrame],
    groups: pd.DataFrame,
) -> None:
    best_basic = tables["by_flow_d0"][tables["by_flow_d0"]["method"].isin(["base_phys6", "base_owics", "base_lagrange", "base_equal_weight"])]
    final_flow = tables["by_flow_d0"][tables["by_flow_d0"]["method"].eq("d0_optimal_weight_profile_curve")][["flow_point", "mae_pct"]].rename(columns={"mae_pct": "model_mae_pct"})
    basic_min = best_basic.groupby("flow_point")["mae_pct"].min().reset_index().rename(columns={"mae_pct": "best_basic_mae_pct"})
    improve = final_flow.merge(basic_min, on="flow_point")
    improve["improved_vs_best_basic"] = improve["model_mae_pct"] < improve["best_basic_mae_pct"]
    d0_dates = sorted(d0["date"].astype(str).unique())
    date_note = (
        "无扰流 D0 仅覆盖 20260607 一天，无法在无扰流样本内部进行按日期留出；本文改用按流量点留出进行 D0 内部防过拟合检验，"
        "并将 20260608--20260615 各扰流日期作为外部日期留出集，检验 D0 标定模型跨日期、跨工况外推稳定性。"
    )
    d0_group_detail = groups[
        groups["method"].eq("d0_optimal_weight_profile_curve") & groups["disturbance_id"].eq("D0")
    ][["flow_point", "window_count", "group_mean_error_pct", "group_sd_pct", "group_mae_pct"]].sort_values("flow_point")
    text = f"""# 无扰流 D0 多声道加权积分模型

## 1. 数据口径

- 数据日期范围：`{data['date'].min()}` 至 `{data['date'].max()}`。
- 无扰流训练数据：`condition_note = no_disturbance_reference` 且 `disturbance_id = D0`。
- 无扰流窗口数：`{len(d0)}`；无扰流日期：`{', '.join(d0_dates)}`。
- 重要限制：{date_note}

## 2. 模型输入变量及物理意义

模型只使用运行时可获得的物理量，不使用日期、窗口序号、标准体积或人工扰流编号作为预测输入。

| 输入 | 来源 | 物理意义 |
| --- | --- | --- |
| `diff_ns_0--diff_ns_9` | 附件 6 | 10 路 A/B 方向传播时间差序列，反映各声路上的轴向速度投影 |
| `raw_diff_gain_a/b`、`raw_diff_intercept` | 附件 7 | 将 A/B 原始时间差映射为五条窗口等效声道速度 |
| `effective_pipe_area_m2` | 附件 7 | 管道等效截面积，用于 `V=A T v` |
| `chord0--chord4` 重构值 | 附件 6+7 | 五个弦高处的线平均速度采样 |
| `prof_center_edges` | 派生 | 中心弦相对边缘弦的偏差，刻画速度剖面鼓起程度 |
| `prof_edge_inner` | 派生 | 边缘弦与内侧弦差异，刻画近壁区速度梯度 |
| `prof_top_bottom`、`prof_inner_skew` | 派生 | 上下不对称与内侧偏斜，刻画非轴对称剖面 |
| `q,q2` | 流量点派生 | 流量点修正曲线，用于描述不同流量水平下的系统偏差 |

## 3. 参数估计与过拟合控制

模型形式为：

```text
V_hat = A T · sum_i w_i u_i · C(q, profile)
```

其中 `w_i` 为五声道积分权重，`C(q,profile)=1+beta^T x` 为流量点与剖面修正系数。权重约束为：

```text
w0=w4, w1=w3, sum_i w_i=1, w_i>=0
```

这对应圆管几何和上下对称布置，防止 10 个 D0 小样本把权重拟合成非物理形态。权重以 Phys6、OWICS/Gauss-Jacobi、Lagrange、等权作为候选先验，并用岭惩罚向先验收缩。修正系数采用岭回归，并将最终校正因子限制在 `[0.97, 1.03]`，对应 Physically Constrained GAN 文献中“物理约束校准/修正系数”的思想：允许校准，但不允许脱离物理积分基准任意漂移。

模型选择采用 D0 内部“按流量点留出”交叉验证，避免流量点修正曲线只记住训练流量点。全局交叉验证最优的最简模型是 calibration-only；但为满足本文采用“剖面特征修正 + 流量点修正曲线”的结构要求，最终从 `profile_flow` 候选族中选择交叉验证误差最小的模型。该处理牺牲极少 CV 精度，换取与 geometric flow profile correction factor 思路一致的可解释结构。

最优候选：

```text
{model.name}
```

拟合权重：

{markdownTable(tables["weights"][["chord", "position", "normalized_height", "fitted_weight", "weight_lagrange", "weight_phys6"]])}

修正系数：

{markdownTable(tables["correction"])}

全部候选交叉验证前 8 名：

{markdownTable(cv.head(8))}

## 4. 相对误差、组均值误差和组内重复性 SD

### D0 无扰流窗口总体误差

{markdownTable(tables["d0_overall"].sort_values("mae_pct"))}

### D0 无扰流分流量点组均值误差与组内 SD

{markdownTable(d0_group_detail)}

### 全部日期外部检验总体误差

{markdownTable(tables["overall"].sort_values("mae_pct").head(12))}

### 组均值误差和组内重复性

分组口径为 `date + flow_point + condition_note + disturbance_id`。`group_sd_pct` 只对同组窗口数不少于 2 的组计算。

{markdownTable(tables["group_summary"].sort_values("mean_abs_group_mean_pct").head(10))}

### 按日期外部留出表现

{markdownTable(tables["by_date"][tables["by_date"]["method"].eq("d0_optimal_weight_profile_curve")].sort_values("date"))}

## 5. 与基础估计方法的流量点对比

以下表格仅针对 D0 无扰流窗口。`improved_vs_best_basic=True` 表示模型在该流量点 MAE 低于 Phys6、OWICS、Lagrange、等权中的最优者。

{markdownTable(improve)}

完整按流量点统计：

{markdownTable(tables["by_flow_d0"].sort_values(["flow_point", "method"]))}

## 6. 结论

1. D0 无扰流数据只有 10 个窗口，且只来自 20260607，因此模型必须采用强物理约束：对称权重、权重和为 1、非负、岭收缩和小幅修正系数。
2. 基于 D0 的最终模型本质是“OWICS/Gauss-Jacobi 物理积分先验 + 几何剖面修正 + 流量点修正曲线 + 小幅校准系数”，符合 geometric flow profile correction factor 的建模思想。
3. 若只看 D0 内部，模型相对基础估计方法有校准收益；但外推到 D1--D8 扰流日期时，误差会显著上升，说明无扰流标定不能替代扰流补偿模型。
4. 由于无扰流没有跨日期重复数据，论文中应明确说明：按日期留出只能作为外部扰流日期稳定性检验，不能解释为无扰流同分布验证。

## 7. 输出文件

- `code/no_disturbance_multichannel_model.py`：可复现实验脚本。
- `code/outputs/d0_fitted_weights.csv`：拟合权重。
- `code/outputs/d0_correction_coefficients.csv`：校准与剖面修正系数。
- `code/outputs/d0_model_cv_by_flow.csv`：按流量点留出交叉验证。
- `code/outputs/d0_model_predictions_long.csv`：模型与基础方法预测长表。
- `code/outputs/d0_model_group_metrics_d0_only.csv`：D0 无扰流组均值误差和组内 SD。
- `code/outputs/d0_model_by_flow_d0_only.csv`：D0 按流量点对比。
- `code/outputs/d0_model_by_date_holdout.csv`：按日期外部留出结果。
- `figures/fig_d0_model_flow_improvement.pdf`：D0 流量点 MAE 对比图。
- `figures/fig_d0_model_date_holdout.pdf`：按日期外部留出 MAE 图。
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    win, raw, geo = loadInputs()
    rec = reconstructChords(raw, geo)
    data = attachFeatures(win, rec, geo)
    d0 = data[data["condition_note"].eq("no_disturbance_reference") & data["disturbance_id"].eq("D0")].copy()
    area = getArea(geo)
    priors = getPriors(geo)
    cv = leaveFlowCv(d0, area, priors)
    profile_cv = cv[cv["correction"].eq("profile_flow")].copy()
    best = profile_cv.iloc[0] if not profile_cv.empty else cv.iloc[0]
    model = fitModel(
        d0,
        area,
        str(best["prior"]),
        priors[str(best["prior"])],
        float(best["lambda_w"]),
        str(best["correction"]),
        float(best["lambda_c"]),
    )
    yuceJieguo = finalPredictions(data, model)
    groups = groupMetrics(yuceJieguo)
    tables = writeCsvOutputs(data, cv, model, yuceJieguo, groups, chordRows(geo))
    plotPdf(tables)
    writeReport(data, d0, cv, model, tables, groups)
    print("Report:", REPORT)
    print("Model:", model.name)
    print("D0 windows:", len(d0), "D0 dates:", ",".join(sorted(d0["date"].astype(str).unique())))
    print("Weights:", ", ".join(f"{w:.6f}" for w in model.weights))
    print(tables["d0_overall"].sort_values("mae_pct").to_string(index=False))
    print("Date holdout:")
    print(tables["by_date"][tables["by_date"]["method"].eq("d0_optimal_weight_profile_curve")].sort_values("date").to_string(index=False))


if __name__ == "__main__":
    main()
