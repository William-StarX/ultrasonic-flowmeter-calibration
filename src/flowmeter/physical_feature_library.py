from __future__ import annotations

# 论文附录复现代码。
# 问题四公共模块：构造紧凑物理特征与稳定性代理。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model_evolution_ridge_ann import (
    FIG_DIR,
    OUT_DIR,
    RANDOM_SEED,
    FittedModel,
    ModelSpec,
    Standardizer,
    buildFeatureFrame,
    evaluatePredictions,
    featureColumns,
    loadData,
    makeRandomSplit,
    standardsGap,
)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
except ModuleNotFoundError:
    colors = None
    A4 = None
    landscape = None
    pdfmetrics = None
    TTFont = None
    canvas = None


ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent / "verification_reports" / "GENERALIZATION_IMPROVEMENT_EXPERIMENT_REPORT.md"
CORRECTION_CLIP_PCT = 2.0


@dataclass(frozen=True)
class ImproveSpec:
    model_id: str
    improvement_group: str
    improvement_items: str
    base_col: str = "lagrange_volume_m3"
    feature_set: str = "proxy_interaction"
    hidden_size: int = 12
    l2: float = 1e-3
    learning_rate: float = 0.003
    epochs: int = 8000
    seed_offset: int = 0
    shrinkage: float = 1.0
    w_mse: float = 1.0
    w_group_mean: float = 0.0
    w_group_sd: float = 0.0
    w_linearity: float = 0.0
    w_drift: float = 0.0
    w_ud_sd: float = 0.0
    w_smooth_flow: float = 0.0
    w_ridge_input: float = 0.0


def enhancedFeatureFrame(data: pd.DataFrame) -> pd.DataFrame:
    feats = buildFeatureFrame(data).copy()
    eps = 1e-12
    q = feats["q"].astype(float)

    profile_bulge = feats.get("chord_center_minus_edges_frac", 0.0)
    edge_gradient = feats.get("chord_edge_minus_inner_frac", 0.0)
    top_bottom = feats.get("chord_top_bottom_frac", 0.0)
    inner_skew = feats.get("chord_inner_skew_frac", 0.0)
    ab_abs = feats.get("ab_abs_mean", feats.get("profile_ab_abs", 0.0))
    ab_range = feats.get("ab_range", 0.0)
    dyn_cv = feats.get("dyn_plateau_cv", 0.0)
    dyn_active = feats.get("dyn_active_eq_s", 0.0) / np.maximum(data["duration_s"].astype(float), eps)
    zero_med = feats.get("zero_rate_med", 0.0)
    zero_mad = feats.get("zero_rate_mad", 0.0)

    feats["bias"] = 1.0
    feats["proxy_profile_bulge"] = profile_bulge
    feats["proxy_edge_gradient"] = edge_gradient
    feats["proxy_asymmetry"] = 0.5 * top_bottom + 0.5 * inner_skew
    feats["proxy_ab_direction"] = 0.5 * ab_abs + 0.5 * ab_range
    feats["proxy_dynamic"] = dyn_cv + 0.2 * (1.0 - dyn_active)
    feats["proxy_zero"] = zero_med + zero_mad
    feats["proxy_integral_disagreement"] = (
        feats.get("lagrange_volume_m3_minus_phys6_frac", 0.0).abs()
        + feats.get("owics_volume_m3_minus_phys6_frac", 0.0).abs()
    )
    proxy_cols = [
        "proxy_profile_bulge",
        "proxy_edge_gradient",
        "proxy_asymmetry",
        "proxy_ab_direction",
        "proxy_dynamic",
        "proxy_zero",
        "proxy_integral_disagreement",
    ]
    for col in proxy_cols:
        feats[f"{col}_x_q"] = feats[col].astype(float) * q
        feats[f"{col}_x_q2"] = feats[col].astype(float) * feats["q2"].astype(float)
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def selectColumns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "runtime_full":
        cols = featureColumns(frame, "runtime_full")
        return ["bias"] + [c for c in cols if c != "bias"]
    proxy = [
        "bias",
        "q",
        "q2",
        "proxy_profile_bulge",
        "proxy_edge_gradient",
        "proxy_asymmetry",
        "proxy_ab_direction",
        "proxy_dynamic",
        "proxy_zero",
        "proxy_integral_disagreement",
    ]
    if feature_set == "proxy_factor":
        return [c for c in proxy if c in frame.columns]
    if feature_set == "proxy_interaction":
        interactions = [c for c in frame.columns if c.startswith("proxy_") and (c.endswith("_x_q") or c.endswith("_x_q2"))]
        return [c for c in proxy + interactions if c in frame.columns]
    if feature_set == "smooth_flow":
        cols = proxy + ["q3", "profile_top_bottom_x_q", "profile_center_all_x_q", "profile_swirl_x_q"]
        return [c for c in cols if c in frame.columns]
    raise ValueError(feature_set)


def fitStandardizer(frame: pd.DataFrame, cols: list[str]) -> Standardizer:
    clean = frame[cols].replace([np.inf, -np.inf], np.nan).copy()
    median = clean.median(numeric_only=True).fillna(0.0)
    clean = clean.fillna(median)
    arr = clean.to_numpy(dtype=float)
    mean = arr.mean(axis=0)
    scale = arr.std(axis=0)
    scale[scale < 1e-12] = 1.0
    if "bias" in cols:
        idx = cols.index("bias")
        median.iloc[idx] = 1.0
        mean[idx] = 0.0
        scale[idx] = 1.0
    return Standardizer(cols, median, mean, scale)


def makeGroupIndices(data: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    groups = []
    for _, g in data.groupby(["date", "flow_point"], sort=False):
        if len(g) >= 3:
            groups.append(g.index.to_numpy())

    d0_flows = []
    for _, g in data[data["condition_note"].eq("no_disturbance_reference")].groupby("flow_point", sort=False):
        if len(g) >= 1:
            d0_flows.append(g.index.to_numpy())

    drift_pairs = []
    for flow, base in data[data["condition_note"].eq("no_disturbance_reference")].groupby("flow_point", sort=False):
        bidx = base.index.to_numpy()
        disturbed = data[(data["condition_note"].eq("disturbed_test")) & (data["flow_point"].eq(flow))]
        for _, dg in disturbed.groupby("disturbance_id", sort=False):
            if len(dg) >= 1 and len(bidx) >= 1:
                drift_pairs.append((bidx, dg.index.to_numpy()))

    disturbance_flow_groups = []
    disturbed_official = data[(data["condition_note"].eq("disturbed_test")) & (data["flow_point"].between(40, 100))]
    for _, g in disturbed_official.groupby(["disturbance_id", "flow_point"], sort=False):
        if len(g) >= 2:
            disturbance_flow_groups.append(g.index.to_numpy())

    flow_groups = []
    for _, g in data.groupby("flow_point", sort=True):
        flow_groups.append(g.index.to_numpy())

    return {
        "groups": groups,
        "d0_flows": d0_flows,
        "drift_pairs": drift_pairs,
        "disturbance_flow_groups": disturbance_flow_groups,
        "flow_groups": flow_groups,
    }


def errorAndGradP(data: pd.DataFrame, p: np.ndarray, spec: ImproveSpec, group_info: dict[str, list[np.ndarray]]) -> tuple[float, np.ndarray]:
    base = data[spec.base_col].to_numpy(dtype=float)
    ref = data["standard_volume_m3"].to_numpy(dtype=float)
    ratio = base / ref
    e = (base * (1.0 + p / 100.0) - ref) / ref * 100.0
    grad_e = np.zeros_like(e)
    loss = 0.0

    if spec.w_mse:
        loss += spec.w_mse * float(np.mean(e**2))
        grad_e += spec.w_mse * (2.0 / len(e)) * e

    groups = group_info["groups"]
    if groups and (spec.w_group_mean or spec.w_group_sd):
        g_count = len(groups)
        for idx in groups:
            values = e[idx]
            mean = float(values.mean())
            if spec.w_group_mean:
                loss += spec.w_group_mean * mean * mean / g_count
                grad_e[idx] += spec.w_group_mean * 2.0 * mean / (g_count * len(idx))
            if spec.w_group_sd and len(idx) > 1:
                centered = values - mean
                var = float(np.sum(centered**2) / (len(idx) - 1))
                loss += spec.w_group_sd * var / g_count
                grad_e[idx] += spec.w_group_sd * 2.0 * centered / (g_count * (len(idx) - 1))

    d0_flows = group_info["d0_flows"]
    if d0_flows and spec.w_linearity:
        for idx in d0_flows:
            mean = float(e[idx].mean())
            loss += spec.w_linearity * mean * mean / len(d0_flows)
            grad_e[idx] += spec.w_linearity * 2.0 * mean / (len(d0_flows) * len(idx))

    drift_pairs = group_info["drift_pairs"]
    if drift_pairs and spec.w_drift:
        for bidx, didx in drift_pairs:
            diff = float(e[bidx].mean() - e[didx].mean())
            loss += spec.w_drift * diff * diff / len(drift_pairs)
            grad_e[bidx] += spec.w_drift * 2.0 * diff / (len(drift_pairs) * len(bidx))
            grad_e[didx] -= spec.w_drift * 2.0 * diff / (len(drift_pairs) * len(didx))

    disturbance_flow_groups = group_info["disturbance_flow_groups"]
    if disturbance_flow_groups and spec.w_ud_sd:
        g_count = len(disturbance_flow_groups)
        for idx in disturbance_flow_groups:
            values = e[idx]
            centered = values - float(values.mean())
            var = float(np.sum(centered**2) / (len(idx) - 1))
            loss += spec.w_ud_sd * var / g_count
            grad_e[idx] += spec.w_ud_sd * 2.0 * centered / (g_count * (len(idx) - 1))

    grad_p = grad_e * ratio

    flow_groups = group_info["flow_groups"]
    if len(flow_groups) >= 2 and spec.w_smooth_flow:
        means = np.array([p[idx].mean() for idx in flow_groups], dtype=float)
        diffs = np.diff(means)
        loss += spec.w_smooth_flow * float(np.mean(diffs**2))
        grad_means = np.zeros_like(means)
        denom = max(1, len(diffs))
        for i, d in enumerate(diffs):
            grad_means[i] -= spec.w_smooth_flow * 2.0 * d / denom
            grad_means[i + 1] += spec.w_smooth_flow * 2.0 * d / denom
        for gidx, idx in enumerate(flow_groups):
            grad_p[idx] += grad_means[gidx] / len(idx)

    return loss, grad_p


def trainMetricMlp(spec: ImproveSpec, train: pd.DataFrame) -> FittedModel:
    feats = enhancedFeatureFrame(train)
    cols = selectColumns(feats, spec.feature_set)
    std = fitStandardizer(feats, cols)
    x = std.transform(feats[cols])
    group_info = makeGroupIndices(train)
    rng = np.random.default_rng(RANDOM_SEED + spec.seed_offset + spec.hidden_size * 37)
    n, p_dim = x.shape
    w1 = rng.normal(0.0, math.sqrt(2.0 / max(1, p_dim)), size=(p_dim, spec.hidden_size))
    b1 = np.zeros(spec.hidden_size)
    w2 = rng.normal(0.0, math.sqrt(2.0 / max(1, spec.hidden_size)), size=spec.hidden_size)
    b2 = 0.0

    mw1 = np.zeros_like(w1)
    vw1 = np.zeros_like(w1)
    mb1 = np.zeros_like(b1)
    vb1 = np.zeros_like(b1)
    mw2 = np.zeros_like(w2)
    vw2 = np.zeros_like(w2)
    mb2 = 0.0
    vb2 = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for epoch in range(1, spec.epochs + 1):
        hidden = np.tanh(x @ w1 + b1)
        pred = hidden @ w2 + b2
        _, grad_p = errorAndGradP(train, pred, spec, group_info)
        grad_w2 = hidden.T @ grad_p + 2.0 * spec.l2 * w2
        grad_b2 = float(grad_p.sum())
        dh = np.outer(grad_p, w2) * (1.0 - hidden**2)
        grad_w1 = x.T @ dh + 2.0 * spec.l2 * w1
        grad_b1 = dh.sum(axis=0)
        if spec.w_ridge_input:
            grad_w1 += 2.0 * spec.w_ridge_input * w1

        mw1 = beta1 * mw1 + (1.0 - beta1) * grad_w1
        vw1 = beta2 * vw1 + (1.0 - beta2) * (grad_w1**2)
        mb1 = beta1 * mb1 + (1.0 - beta1) * grad_b1
        vb1 = beta2 * vb1 + (1.0 - beta2) * (grad_b1**2)
        mw2 = beta1 * mw2 + (1.0 - beta1) * grad_w2
        vw2 = beta2 * vw2 + (1.0 - beta2) * (grad_w2**2)
        mb2 = beta1 * mb2 + (1.0 - beta1) * grad_b2
        vb2 = beta2 * vb2 + (1.0 - beta2) * (grad_b2**2)
        bc1 = 1.0 - beta1**epoch
        bc2 = 1.0 - beta2**epoch
        w1 -= spec.learning_rate * (mw1 / bc1) / (np.sqrt(vw1 / bc2) + eps)
        b1 -= spec.learning_rate * (mb1 / bc1) / (np.sqrt(vb1 / bc2) + eps)
        w2 -= spec.learning_rate * (mw2 / bc1) / (np.sqrt(vw2 / bc2) + eps)
        b2 -= spec.learning_rate * (mb2 / bc1) / (math.sqrt(vb2 / bc2) + eps)

    model_spec = ModelSpec(
        model_id=spec.model_id,
        family=spec.improvement_group,
        base_col=spec.base_col,
        feature_set=spec.feature_set,
        model_type="mlp",
        hidden_size=spec.hidden_size,
        l2=spec.l2,
        learning_rate=spec.learning_rate,
        seed_offset=spec.seed_offset,
        shrinkage=spec.shrinkage,
        epochs=spec.epochs,
    )
    return FittedModel(spec=model_spec, standardizer=std, w1=w1, b1=b1, w2=w2, b2=b2)


def evaluateModel(model: FittedModel, frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    feats = enhancedFeatureFrame(frame)
    pred = model.predictVolume(frame, feats)
    return evaluatePredictions(frame, pred)


def buildSpecs() -> list[ImproveSpec]:
    common = dict(base_col="lagrange_volume_m3", hidden_size=16, l2=1e-3, learning_rate=0.003, epochs=8000, seed_offset=17)
    return [
        ImproveSpec(
            model_id="single_metric_loss_runtime",
            improvement_group="single",
            improvement_items="metric_loss",
            feature_set="runtime_full",
            w_mse=1.0,
            w_group_mean=0.8,
            w_group_sd=1.5,
            w_linearity=0.8,
            w_drift=0.8,
            **common,
        ),
        ImproveSpec(
            model_id="single_proxy_factor_mse",
            improvement_group="single",
            improvement_items="proxy_factor",
            feature_set="proxy_factor",
            w_mse=1.0,
            hidden_size=10,
            epochs=8000,
            seed_offset=11,
        ),
        ImproveSpec(
            model_id="single_flow_smooth_proxy",
            improvement_group="single",
            improvement_items="smooth_flow",
            feature_set="smooth_flow",
            w_mse=1.0,
            w_smooth_flow=0.8,
            hidden_size=12,
            epochs=8000,
            seed_offset=13,
        ),
        ImproveSpec(
            model_id="single_drift_loss_proxy",
            improvement_group="single",
            improvement_items="drift_loss",
            feature_set="proxy_interaction",
            w_mse=1.0,
            w_drift=1.5,
            hidden_size=12,
            epochs=8000,
            seed_offset=15,
        ),
        ImproveSpec(
            model_id="single_ud_sd_proxy",
            improvement_group="single",
            improvement_items="disturbance_flow_sd",
            feature_set="proxy_interaction",
            w_mse=1.0,
            w_ud_sd=2.0,
            hidden_size=12,
            l2=2e-3,
            epochs=9000,
            seed_offset=31,
        ),
        ImproveSpec(
            model_id="combined_metric_proxy",
            improvement_group="combined",
            improvement_items="metric_loss+proxy_factor",
            feature_set="proxy_interaction",
            w_mse=1.0,
            w_group_mean=0.8,
            w_group_sd=1.2,
            w_linearity=0.8,
            w_drift=0.8,
            hidden_size=12,
            epochs=9000,
            seed_offset=17,
        ),
        ImproveSpec(
            model_id="combined_metric_udsd_proxy",
            improvement_group="combined",
            improvement_items="metric_loss+proxy_factor+disturbance_flow_sd",
            feature_set="proxy_interaction",
            w_mse=1.0,
            w_group_mean=0.8,
            w_group_sd=1.2,
            w_linearity=1.0,
            w_drift=0.8,
            w_ud_sd=2.0,
            hidden_size=12,
            l2=2e-3,
            epochs=9000,
            seed_offset=33,
        ),
        ImproveSpec(
            model_id="combined_metric_smooth_proxy",
            improvement_group="combined",
            improvement_items="metric_loss+proxy_factor+smooth_flow",
            feature_set="proxy_interaction",
            w_mse=1.0,
            w_group_mean=0.8,
            w_group_sd=1.2,
            w_linearity=0.8,
            w_drift=0.8,
            w_smooth_flow=0.6,
            hidden_size=12,
            epochs=9000,
            seed_offset=19,
        ),
        ImproveSpec(
            model_id="combined_metric_smooth_udsd_proxy",
            improvement_group="combined",
            improvement_items="metric_loss+proxy_factor+smooth_flow+disturbance_flow_sd",
            feature_set="proxy_interaction",
            w_mse=1.0,
            w_group_mean=0.8,
            w_group_sd=1.2,
            w_linearity=1.0,
            w_drift=1.0,
            w_ud_sd=2.5,
            w_smooth_flow=0.6,
            hidden_size=12,
            l2=2e-3,
            epochs=9000,
            seed_offset=35,
        ),
        ImproveSpec(
            model_id="combined_metric_runtime_smooth",
            improvement_group="combined",
            improvement_items="metric_loss+runtime_full+smooth_flow",
            feature_set="runtime_full",
            w_mse=1.0,
            w_group_mean=0.8,
            w_group_sd=1.2,
            w_linearity=0.8,
            w_drift=0.8,
            w_smooth_flow=0.5,
            w_ridge_input=1e-5,
            **common,
        ),
        ImproveSpec(
            model_id="combined_metric_runtime_udsd",
            improvement_group="combined",
            improvement_items="metric_loss+runtime_full+disturbance_flow_sd",
            feature_set="runtime_full",
            w_mse=0.9,
            w_group_mean=0.8,
            w_group_sd=1.5,
            w_linearity=1.2,
            w_drift=1.2,
            w_ud_sd=2.5,
            hidden_size=16,
            l2=2e-3,
            learning_rate=0.003,
            epochs=9000,
            seed_offset=37,
        ),
        ImproveSpec(
            model_id="combined_balanced_runtime_official",
            improvement_group="combined",
            improvement_items="runtime_full+official_metric_losses+smooth_flow",
            feature_set="runtime_full",
            w_mse=0.8,
            w_group_mean=0.8,
            w_group_sd=2.0,
            w_linearity=1.8,
            w_drift=2.5,
            w_ud_sd=3.0,
            w_smooth_flow=0.4,
            hidden_size=16,
            l2=3e-3,
            learning_rate=0.003,
            epochs=10000,
            seed_offset=39,
        ),
        ImproveSpec(
            model_id="combined_strong_repeatability",
            improvement_group="combined",
            improvement_items="proxy_factor+strong_group_sd+drift",
            feature_set="proxy_interaction",
            w_mse=0.8,
            w_group_mean=0.8,
            w_group_sd=3.0,
            w_linearity=0.8,
            w_drift=1.0,
            w_smooth_flow=0.4,
            hidden_size=12,
            epochs=9000,
            seed_offset=23,
        ),
        ImproveSpec(
            model_id="combined_strong_drift",
            improvement_group="combined",
            improvement_items="proxy_factor+strong_drift+smooth_flow",
            feature_set="proxy_interaction",
            w_mse=0.8,
            w_group_mean=0.6,
            w_group_sd=1.0,
            w_linearity=1.2,
            w_drift=3.0,
            w_smooth_flow=0.8,
            hidden_size=12,
            epochs=9000,
            seed_offset=29,
        ),
    ]


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    data = loadData()
    train, validation, split = makeRandomSplit(data)
    split.to_csv(OUT_DIR / "generalization_improvement_split.csv", index=False, encoding="utf-8-sig")
    specs = buildSpecs()
    rows = []
    groups_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    windows_by_key: dict[tuple[str, str], pd.DataFrame] = {}

    for train_name, train_frame, evals in [
        ("full", data, {"full_fit_all": data}),
        ("random", train, {"random_split_train": train, "random_split_validation": validation}),
    ]:
        for spec in specs:
            model = trainMetricMlp(spec, train_frame)
            for operation, frame in evals.items():
                summary, groups, windows = evaluateModel(model, frame)
                fail_count, gap = standardsGap(summary)
                row = {
                    "model_id": spec.model_id,
                    "improvement_group": spec.improvement_group,
                    "improvement_items": spec.improvement_items,
                    "operation": operation,
                    "base_col": spec.base_col,
                    "feature_set": spec.feature_set,
                    "hidden_size": spec.hidden_size,
                    "l2": spec.l2,
                    "learning_rate": spec.learning_rate,
                    "epochs": spec.epochs,
                    "shrinkage": spec.shrinkage,
                    "w_mse": spec.w_mse,
                    "w_group_mean": spec.w_group_mean,
                    "w_group_sd": spec.w_group_sd,
                    "w_linearity": spec.w_linearity,
                    "w_drift": spec.w_drift,
                    "w_ud_sd": spec.w_ud_sd,
                    "w_smooth_flow": spec.w_smooth_flow,
                    "standard_fail_count": fail_count,
                    "standard_gap_score": gap,
                }
                row.update(summary.to_dict())
                rows.append(row)
                groups_by_key[(operation, spec.model_id)] = groups
                windows_by_key[(operation, spec.model_id)] = windows

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "generalization_improvement_metrics_long.csv", index=False, encoding="utf-8-sig")
    comparison = buildComparison(metrics)
    comparison.to_csv(OUT_DIR / "generalization_improvement_comparison.csv", index=False, encoding="utf-8-sig")
    if not comparison.empty:
        best_id = str(comparison.iloc[0]["model_id"])
        for operation in ["full_fit_all", "random_split_train", "random_split_validation"]:
            key = (operation, best_id)
            if key in groups_by_key:
                groups_by_key[key].to_csv(OUT_DIR / f"generalization_improvement_best_{operation}_groups.csv", index=False, encoding="utf-8-sig")
                windows_by_key[key].to_csv(OUT_DIR / f"generalization_improvement_best_{operation}_windows.csv", index=False, encoding="utf-8-sig")
    writeReport(comparison)
    makeFigure(comparison)
    print(comparison.head(12).to_string(index=False))


def buildComparison(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics[metrics["operation"].eq("full_fit_all")].set_index("model_id")
    train = metrics[metrics["operation"].eq("random_split_train")].set_index("model_id")
    valid = metrics[metrics["operation"].eq("random_split_validation")].set_index("model_id")
    rows = []
    for model_id, v in valid.iterrows():
        if model_id not in full.index:
            continue
        f = full.loc[model_id]
        t = train.loc[model_id] if model_id in train.index else v
        overfit = max(0.0, float(v["mae_pct"]) - float(t["mae_pct"]))
        rows.append(
            {
                "model_id": model_id,
                "improvement_group": v["improvement_group"],
                "improvement_items": v["improvement_items"],
                "validation_fail_count": int(v["standard_fail_count"]),
                "full_fail_count": int(f["standard_fail_count"]),
                "validation_pass_all": bool(v["pass_all_standards"]),
                "full_pass_all": bool(f["pass_all_standards"]),
                "validation_mae_pct": float(v["mae_pct"]),
                "validation_group_pass": v["group_pass"],
                "validation_max_abs_group_mean_pct": float(v["max_abs_group_mean_pct"]),
                "validation_max_group_sd_pct": float(v["max_group_sd_pct"]),
                "validation_u_nor_L_pct": float(v["u_nor_L_pct"]),
                "validation_u_nor_r_pct": float(v["u_nor_r_pct"]),
                "validation_u_nor_d_pct": float(v["u_nor_d_pct"]),
                "train_mae_pct": float(t["mae_pct"]),
                "full_mae_pct": float(f["mae_pct"]),
                "full_group_pass": f["group_pass"],
                "full_u_nor_L_pct": float(f["u_nor_L_pct"]),
                "full_u_nor_r_pct": float(f["u_nor_r_pct"]),
                "full_u_nor_d_pct": float(f["u_nor_d_pct"]),
                "overfit_mae_gap": overfit,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_score"] = (
        1000.0 * out["validation_fail_count"]
        + 250.0 * out["full_fail_count"]
        + 20.0 * out["overfit_mae_gap"]
        + out["validation_mae_pct"]
    )
    return out.sort_values(["rank_score", "validation_mae_pct"]).reset_index(drop=True)


def mdTable(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    view = df[cols].head(max_rows).copy()
    if view.empty:
        return "_无数据_"
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in view.astype(str).to_numpy()]
    return "\n".join([header, sep] + body)


def writeReport(comparison: pd.DataFrame) -> None:
    previous = pd.read_csv(OUT_DIR / "final_model_comprehensive_ranking.csv", encoding="utf-8-sig")
    prev_cols = [
        "model_id",
        "validation_fail_count",
        "full_fail_count",
        "validation_mae_pct",
        "validation_u_nor_L_pct",
        "validation_u_nor_r_pct",
        "validation_u_nor_d_pct",
        "full_mae_pct",
        "full_group_pass",
    ]
    cols = [
        "model_id",
        "improvement_group",
        "improvement_items",
        "validation_fail_count",
        "full_fail_count",
        "validation_mae_pct",
        "validation_group_pass",
        "validation_u_nor_L_pct",
        "validation_u_nor_r_pct",
        "validation_u_nor_d_pct",
        "full_mae_pct",
        "full_group_pass",
    ]
    lines = [
        "# 泛化指标改进组实验",
        "",
        "## 实验设计",
        "",
        "本实验从论文库可迁移方法出发，设置单项改进和联合改进两类候选。",
        "单项改进包括：指标驱动损失、低维扰流代理因子、平滑流量点修正、扰流漂移损失、扰流-流量点组内标准差损失。",
        "联合改进包括：指标驱动损失 + 扰流代理因子、再叠加平滑流量点修正、扰流-流量点组内标准差或更强重复性/漂移权重。",
        "所有模型均沿用相同的 `(date, flow_point, disturbance_id)` 组内分层验证划分，目标约 70% 训练、30% 验证，评价指标与原实验完全一致。",
        "该划分保留 D0 流量锚点；对 `n >= 5` 的重复组取 2 个窗口训练、其余窗口验证，对 `n = 3` 或 `4` 的重复组整组放入训练或验证，从而保证验证集中可以计算组均值误差、组内 SD 与 `u_nor,r`。",
        "",
        "## 新增改进组排名",
        "",
        mdTable(comparison, cols),
        "",
        "## 原综合排名前五对照",
        "",
        mdTable(previous, prev_cols, max_rows=5),
        "",
        "## 与原综合最优的结论性对比",
        "",
        "新增改进组和原综合排名均采用同一组内分层验证划分，以及同一组达标阈值：",
        "`|eg_bar| <= 0.2%`、`SDg <= 0.040%`、`u_nor,L < 0.036%`、`u_nor,r < 0.040%`、`u_nor,d < 0.115%`。",
        "本次验证集显式保留重复组，因此 `group_pass` 与 `u_nor,r` 不再是 `0/0` 或缺失值。",
        "",
        "## 结论",
        "",
    ]
    if not comparison.empty:
        best = comparison.iloc[0]
        old_best = previous.iloc[0]
        lines.extend(
            [
                f"新增改进组中综合排序最好的是 `{best['model_id']}`，属于 `{best['improvement_items']}`。",
                f"其随机验证 MAE 为 `{best['validation_mae_pct']:.6g}%`，验证失败项数为 `{int(best['validation_fail_count'])}`，全量失败项数为 `{int(best['full_fail_count'])}`。",
                f"原综合最优 `{old_best['model_id']}` 的随机验证 MAE 为 `{float(old_best['validation_mae_pct']):.6g}%`，验证失败项数为 `{int(old_best['validation_fail_count'])}`，全量失败项数为 `{int(old_best['full_fail_count'])}`。",
                "因此新增组在验证 MAE 上有局部改进，但综合达标性没有超过原综合最优；新增组中没有模型在随机验证或全量评价下全部达标。",
                "这说明当前数据下仅靠损失函数和代理特征仍不能完全解决泛化不达标，后续需要更多真实跨日期/跨扰流强度重复组，或使用更贴近官方指标的可微近似目标继续调参。",
            ]
        )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def makeFigure(comparison: pd.DataFrame) -> None:
    if canvas is None or comparison.empty:
        return
    pdfmetrics.registerFont(TTFont("MSYH", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
    path = FIG_DIR / "fig_generalization_improvement_compare.pdf"
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    width, height = landscape(A4)
    c.setFont("MSYH", 14)
    c.setFillColor(colors.HexColor("#203040"))
    c.drawString(40, height - 45, "泛化指标改进组随机验证对比")
    view = comparison.head(9).copy()
    max_mae = max(0.1, float(view["validation_mae_pct"].max()))
    left = 185
    top = height - 85
    bar_w = width - 260
    row_h = 38
    palette = {
        "single": colors.HexColor("#8FB7E8"),
        "combined": colors.HexColor("#E3A76F"),
    }
    for i, row in view.iterrows():
        y = top - i * row_h
        label = str(row["model_id"]).replace("combined_", "cmb_").replace("single_", "sgl_")
        c.setFont("MSYH", 8)
        c.setFillColor(colors.HexColor("#506070"))
        c.drawRightString(left - 8, y, label[:36])
        c.setFillColor(palette.get(str(row["improvement_group"]), colors.HexColor("#AAB4C0")))
        w = bar_w * float(row["validation_mae_pct"]) / max_mae
        c.rect(left, y - 10, w, 12, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#203040"))
        c.drawString(left + w + 5, y - 9, f"{float(row['validation_mae_pct']):.4f}% / fail {int(row['validation_fail_count'])}")
    c.setFont("MSYH", 8)
    c.setFillColor(colors.HexColor("#667788"))
    c.drawCentredString(width / 2, 28, "validation MAE / %")
    c.save()


if __name__ == "__main__":
    run()
