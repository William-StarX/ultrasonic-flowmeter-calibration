from __future__ import annotations

# 论文附录复现代码。
# 问题四对照：统一实现 Ridge 与 ANN 残差学习路线。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
DATA_DIR = ROOT / "A题 (1)"
WINDOW_DATA = DATA_DIR / "attachment1_window_data.csv"
OUT_DIR = Path(__file__).resolve().parent / "verification_outputs"
FIG_DIR = Path(__file__).resolve().parent / "verification_figures"
REPORT = Path(__file__).resolve().parent / "verification_reports" / "FINAL_MODEL_THEORY_AND_RANKING.md"
RANDOM_SEED = 20260710
TRAIN_FRACTION = 0.70
SPLIT_MODE = "stratified_repeated_group_internal"
PRESSURE_SPLIT_MODE = "whole_group_holdout_pressure"
MIN_VALIDATION_REPEAT_GROUPS = 6
MLP_EPOCHS = 10000
CORRECTION_CLIP_PCT = 2.0

STANDARD_LIMITS = {
    "group_mean_abs_pct": 0.2,
    "group_sd_pct": 0.040,
    "u_nor_L_pct": 0.036,
    "u_nor_r_pct": 0.040,
    "u_nor_d_pct": 0.115,
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    base_col: str
    feature_set: str
    model_type: str
    ridge_lambda: float = 0.0
    hidden_size: int = 0
    l2: float = 0.0
    learning_rate: float = 0.0
    seed_offset: int = 0
    shrinkage: float = 1.0
    epochs: int = 0


@dataclass
class Standardizer:
    columns: list[str]
    median: pd.Series
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Standardizer":
        clean = frame.replace([np.inf, -np.inf], np.nan).copy()
        median = clean.median(numeric_only=True).fillna(0.0)
        clean = clean.fillna(median)
        arr = clean.to_numpy(dtype=float)
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(list(frame.columns), median, mean, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        clean = frame[self.columns].replace([np.inf, -np.inf], np.nan).copy()
        clean = clean.fillna(self.median)
        return (clean.to_numpy(dtype=float) - self.mean) / self.scale


@dataclass
class FittedModel:
    spec: ModelSpec
    standardizer: Standardizer | None = None
    beta: np.ndarray | None = None
    w1: np.ndarray | None = None
    b1: np.ndarray | None = None
    w2: np.ndarray | None = None
    b2: float = 0.0

    def correctionPct(self, features: pd.DataFrame) -> np.ndarray:
        if self.spec.model_type == "baseline":
            return np.zeros(len(features))
        if self.standardizer is None:
            raise ValueError("standardizer is required")
        x = self.standardizer.transform(features)
        if self.spec.model_type == "ridge":
            if self.beta is None:
                raise ValueError("ridge beta is required")
            raw = x @ self.beta
        elif self.spec.model_type == "mlp":
            if self.w1 is None or self.b1 is None or self.w2 is None:
                raise ValueError("MLP weights are required")
            hidden = np.tanh(x @ self.w1 + self.b1)
            raw = hidden @ self.w2 + self.b2
        else:
            raise ValueError(self.spec.model_type)
        return self.spec.shrinkage * np.clip(raw, -CORRECTION_CLIP_PCT, CORRECTION_CLIP_PCT)

    def predictVolume(self, data: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
        base = data[self.spec.base_col].to_numpy(dtype=float)
        return base * (1.0 + self.correctionPct(features) / 100.0)


def ensureDirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)


def loadData() -> pd.DataFrame:
    return pd.read_csv(WINDOW_DATA, encoding="utf-8-sig").sort_values("window_id").reset_index(drop=True)


GROUP_KEY = ["date", "flow_point", "disturbance_id"]
OFFICIAL_EVAL_GROUP_KEY = ["date", "flow_point"]


def _splitFromIndices(
    data: pd.DataFrame,
    train_idx: set[int],
    validation_idx: set[int],
    split_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = data[["window_id", "date", "flow_point", "condition_note", "disturbance_id"]].copy()
    split["split"] = ["train" if i in train_idx else "validation" for i in range(len(data))]
    split["split_mode"] = split_mode
    train = data.loc[split["split"].eq("train")].reset_index(drop=True)
    validation = data.loc[split["split"].eq("validation")].reset_index(drop=True)
    return train, validation, split


def makeRandomSplit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    target_validation = int(round(len(data) * (1.0 - TRAIN_FRACTION)))
    validation_idx: set[int] = set()
    train_idx: set[int] = set()

    d0 = data[data["condition_note"].eq("no_disturbance_reference")]
    for _, group in d0.groupby(GROUP_KEY, sort=True):
        validation_idx.update(int(i) for i in group.index.to_numpy())

    whole_group_candidates = []
    for key, group in data[data["condition_note"].eq("disturbed_test")].groupby(GROUP_KEY, sort=True):
        indices = group.index.to_numpy()
        n = len(indices)
        if n >= 5:
            train_pick = set(int(i) for i in rng.choice(indices, size=2, replace=False))
            train_idx.update(train_pick)
            validation_idx.update(int(i) for i in indices if int(i) not in train_pick)
            continue
        if n in {3, 4}:
            date, flow, disturbance_id = key
            priority = int(40 <= int(flow) <= 100)
            whole_group_candidates.append((priority, str(date), int(flow), str(disturbance_id), indices))
            continue
        train_idx.update(int(i) for i in indices)

    order = rng.permutation(len(whole_group_candidates))
    selected_repeat_groups = 0
    for idx in sorted(order, key=lambda i: (-whole_group_candidates[int(i)][0], rng.random())):
        _, _, _, _, indices = whole_group_candidates[int(idx)]
        if selected_repeat_groups >= MIN_VALIDATION_REPEAT_GROUPS and len(validation_idx) >= target_validation:
            break
        validation_idx.update(int(i) for i in indices)
        selected_repeat_groups += 1

    for _, _, _, _, indices in whole_group_candidates:
        idx_set = set(int(i) for i in indices)
        if not idx_set & validation_idx:
            train_idx.update(idx_set)

    remaining = [i for i in rng.permutation(len(data)).tolist() if i not in validation_idx and i not in train_idx]
    for i in remaining:
        if len(validation_idx) >= target_validation:
            break
        validation_idx.add(int(i))

    train_idx = set(range(len(data))) - validation_idx
    return _splitFromIndices(data, train_idx, validation_idx, SPLIT_MODE)


def makeGroupHoldoutPressureSplit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED + 911)
    target_validation = int(round(len(data) * (1.0 - TRAIN_FRACTION)))
    validation_idx: set[int] = set()

    d0 = data[data["condition_note"].eq("no_disturbance_reference")]
    for _, group in d0.groupby(GROUP_KEY, sort=True):
        validation_idx.update(int(i) for i in group.index.to_numpy())

    candidates = []
    for key, group in data[data["condition_note"].eq("disturbed_test")].groupby(GROUP_KEY, sort=True):
        if len(group) < 3:
            continue
        date, flow, disturbance_id = key
        priority = int(40 <= int(flow) <= 100)
        candidates.append((priority, str(date), int(flow), str(disturbance_id), group.index.to_numpy()))

    order = rng.permutation(len(candidates))
    selected_repeat_groups = 0
    for idx in sorted(order, key=lambda i: (-candidates[int(i)][0], rng.random())):
        _, _, _, _, indices = candidates[int(idx)]
        if selected_repeat_groups >= MIN_VALIDATION_REPEAT_GROUPS and len(validation_idx) >= target_validation:
            break
        validation_idx.update(int(i) for i in indices)
        selected_repeat_groups += 1

    train_idx = set(range(len(data))) - validation_idx
    return _splitFromIndices(data, train_idx, validation_idx, PRESSURE_SPLIT_MODE)


def buildFeatureFrame(data: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    feats = pd.DataFrame(index=data.index)
    allowed = [
        "duration_s",
        "flow_point",
        "phys6_volume_m3",
        "owics_volume_m3",
        "lagrange_volume_m3",
        "equal_weight_volume_m3",
        "profile_top_bottom",
        "profile_center_all",
        "profile_edge_inner",
        "profile_inner_skew",
        "profile_ab_abs",
        "profile_swirl",
        "dyn_first_0p1_s",
        "dyn_tail_0p1_s",
        "dyn_start_over_plateau",
        "dyn_end_over_plateau",
        "dyn_plateau_cv",
        "dyn_active_eq_s",
        "zero_rate_med",
        "zero_rate_mad",
        "zero_age_s",
    ]
    for col in allowed:
        if col in data.columns:
            feats[col] = data[col].astype(float)

    q = (data["flow_point"].astype(float) - 50.0) / 30.0
    feats["q"] = q
    feats["q2"] = q**2
    feats["q3"] = q**3
    feats["log_duration"] = np.log(np.maximum(data["duration_s"].astype(float), eps))

    base = data["phys6_volume_m3"].astype(float)
    for col in ["owics_volume_m3", "lagrange_volume_m3", "equal_weight_volume_m3"]:
        feats[f"{col}_minus_phys6_frac"] = (data[col].astype(float) - base) / np.maximum(base.abs(), eps)

    chord_cols = [f"chord{i}" for i in range(5)]
    ab_cols = [f"ab{i}" for i in range(5)]
    chord = data[chord_cols].astype(float)
    ab = data[ab_cols].astype(float)
    chord_mean = chord.mean(axis=1).replace(0.0, np.nan)
    feats["chord_mean"] = chord_mean
    feats["chord_std"] = chord.std(axis=1)
    feats["chord_range_frac"] = (chord.max(axis=1) - chord.min(axis=1)) / chord_mean
    feats["chord_center_minus_edges_frac"] = (
        chord["chord2"] - 0.5 * (chord["chord0"] + chord["chord4"])
    ) / chord_mean
    feats["chord_edge_minus_inner_frac"] = (
        0.5 * (chord["chord0"] + chord["chord4"]) - 0.5 * (chord["chord1"] + chord["chord3"])
    ) / chord_mean
    feats["chord_top_bottom_frac"] = (chord["chord0"] - chord["chord4"]) / chord_mean
    feats["chord_inner_skew_frac"] = (chord["chord1"] - chord["chord3"]) / chord_mean
    feats["ab_mean"] = ab.mean(axis=1)
    feats["ab_abs_mean"] = ab.abs().mean(axis=1)
    feats["ab_std"] = ab.std(axis=1)
    feats["ab_range"] = ab.max(axis=1) - ab.min(axis=1)

    for col in [
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
        "lagrange_volume_m3_minus_phys6_frac",
        "owics_volume_m3_minus_phys6_frac",
        "chord_center_minus_edges_frac",
        "chord_top_bottom_frac",
        "ab_abs_mean",
    ]:
        if col in feats.columns:
            feats[f"{col}_x_q"] = feats[col].astype(float) * q

    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def featureColumns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    profile = [
        "q",
        "q2",
        "profile_top_bottom",
        "profile_center_all",
        "profile_edge_inner",
        "profile_inner_skew",
        "profile_ab_abs",
        "profile_swirl",
        "owics_volume_m3_minus_phys6_frac",
        "lagrange_volume_m3_minus_phys6_frac",
        "equal_weight_volume_m3_minus_phys6_frac",
        "chord_range_frac",
        "chord_center_minus_edges_frac",
        "chord_edge_minus_inner_frac",
        "chord_top_bottom_frac",
        "chord_inner_skew_frac",
        "ab_abs_mean",
        "ab_range",
    ]
    dynamic_zero = [
        "duration_s",
        "log_duration",
        "dyn_start_over_plateau",
        "dyn_end_over_plateau",
        "dyn_plateau_cv",
        "dyn_active_eq_s",
        "zero_rate_med",
        "zero_rate_mad",
        "zero_age_s",
    ]
    interactions = [c for c in frame.columns if c.endswith("_x_q")]
    if feature_set == "profile_flow":
        cols = profile
    elif feature_set == "profile_dynamic_zero":
        cols = profile + dynamic_zero
    elif feature_set == "runtime_full":
        cols = profile + dynamic_zero + interactions + ["q3", "chord_mean", "chord_std", "ab_mean", "ab_std"]
    else:
        raise ValueError(feature_set)
    return [c for c in cols if c in frame.columns]


def residualTargetPct(data: pd.DataFrame, base_col: str) -> np.ndarray:
    base = np.maximum(data[base_col].to_numpy(dtype=float), 1e-12)
    return (data["standard_volume_m3"].to_numpy(dtype=float) / base - 1.0) * 100.0


def fitRidge(spec: ModelSpec, train: pd.DataFrame, xunlianTezheng: pd.DataFrame) -> FittedModel:
    cols = featureColumns(xunlianTezheng, spec.feature_set)
    std = Standardizer.fit(xunlianTezheng[cols])
    x = std.transform(xunlianTezheng[cols])
    y = residualTargetPct(train, spec.base_col)
    penalty = spec.ridge_lambda * np.eye(x.shape[1])
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return FittedModel(spec=spec, standardizer=std, beta=beta)


def fitMlp(spec: ModelSpec, train: pd.DataFrame, xunlianTezheng: pd.DataFrame) -> FittedModel:
    cols = featureColumns(xunlianTezheng, spec.feature_set)
    std = Standardizer.fit(xunlianTezheng[cols])
    x = std.transform(xunlianTezheng[cols])
    y = residualTargetPct(train, spec.base_col)
    rng = np.random.default_rng(RANDOM_SEED + spec.seed_offset + spec.hidden_size * 37)
    n, p = x.shape
    w1 = rng.normal(0.0, math.sqrt(2.0 / max(1, p)), size=(p, spec.hidden_size))
    b1 = np.zeros(spec.hidden_size)
    w2 = rng.normal(0.0, math.sqrt(2.0 / max(1, spec.hidden_size)), size=spec.hidden_size)
    b2 = float(np.mean(y))

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
    lr = spec.learning_rate
    l2 = spec.l2

    for epoch in range(1, spec.epochs + 1):
        hidden = np.tanh(x @ w1 + b1)
        pred = hidden @ w2 + b2
        err = pred - y
        dpred = (2.0 / n) * err
        grad_w2 = hidden.T @ dpred + 2.0 * l2 * w2
        grad_b2 = float(dpred.sum())
        dh = np.outer(dpred, w2) * (1.0 - hidden**2)
        grad_w1 = x.T @ dh + 2.0 * l2 * w1
        grad_b1 = dh.sum(axis=0)

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
        w1 -= lr * (mw1 / bc1) / (np.sqrt(vw1 / bc2) + eps)
        b1 -= lr * (mb1 / bc1) / (np.sqrt(vb1 / bc2) + eps)
        w2 -= lr * (mw2 / bc1) / (np.sqrt(vw2 / bc2) + eps)
        b2 -= lr * (mb2 / bc1) / (math.sqrt(vb2 / bc2) + eps)

    return FittedModel(spec=spec, standardizer=std, w1=w1, b1=b1, w2=w2, b2=b2)


def fitModel(spec: ModelSpec, train: pd.DataFrame, xunlianTezheng: pd.DataFrame) -> FittedModel:
    if spec.model_type == "baseline":
        return FittedModel(spec=spec)
    if spec.model_type == "ridge":
        return fitRidge(spec, train, xunlianTezheng)
    if spec.model_type == "mlp":
        return fitMlp(spec, train, xunlianTezheng)
    raise ValueError(spec.model_type)


def relativeErrorPct(moxingTiji: np.ndarray, biaozhunTiji: pd.Series | np.ndarray) -> np.ndarray:
    y = np.asarray(biaozhunTiji, dtype=float)
    return (np.asarray(moxingTiji, dtype=float) - y) / y * 100.0


def groupTable(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (date, flow_point), group in df.groupby(OFFICIAL_EVAL_GROUP_KEY, sort=True):
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
                "abs_mean_error_pct": abs(mean_err),
                "repeatability_sd_pct": sd,
                "range_pct": float(values.max() - values.min()),
                "pass_mean": bool(abs(mean_err) <= STANDARD_LIMITS["group_mean_abs_pct"]),
                "pass_sd": bool(sd <= STANDARD_LIMITS["group_sd_pct"]),
                "pass_group": bool(abs(mean_err) <= STANDARD_LIMITS["group_mean_abs_pct"] and sd <= STANDARD_LIMITS["group_sd_pct"]),
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
    if detail.empty:
        return float("nan")
    detail["disturb_sd"] = detail["disturb_sd"].fillna(0.0)
    detail["abs_drift"] = (detail["baseline_mean"] - detail["disturb_mean"]).abs()
    available = detail[(detail["baseline_n"] >= 1) & (detail["disturb_n"] >= 1)].copy()
    if available.empty:
        return float("nan")
    u_d_c = float(available["abs_drift"].max()) / math.sqrt(3.0)
    u_d_r = float(available["disturb_sd"].max())
    return math.sqrt(u_d_c * u_d_c + u_d_r * u_d_r)


def evaluatePredictions(data: pd.DataFrame, moxingTiji: np.ndarray) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    df = data.copy()
    df["model_volume_m3"] = moxingTiji
    df["model_error_pct"] = relativeErrorPct(moxingTiji, df["standard_volume_m3"])
    groups = groupTable(df)
    u_l = linearityUL(df)
    u_r = float(groups["repeatability_sd_pct"].max()) if not groups.empty else float("nan")
    u_d = disturbanceUD(df)
    max_mean = float(groups["abs_mean_error_pct"].max()) if not groups.empty else float("nan")
    max_sd = float(groups["repeatability_sd_pct"].max()) if not groups.empty else float("nan")
    pass_group_mean_all = bool(pd.notna(max_mean) and max_mean <= STANDARD_LIMITS["group_mean_abs_pct"])
    pass_group_sd_all = bool(pd.notna(max_sd) and max_sd <= STANDARD_LIMITS["group_sd_pct"])
    row = pd.Series(
        {
            "window_count": int(len(df)),
            "mae_pct": float(df["model_error_pct"].abs().mean()),
            "mean_error_pct": float(df["model_error_pct"].mean()),
            "max_abs_error_pct": float(df["model_error_pct"].abs().max()),
            "group_count": int(len(groups)),
            "group_pass_count": int(groups["pass_group"].sum()) if not groups.empty else 0,
            "group_pass": f"{int(groups['pass_group'].sum())}/{len(groups)}" if not groups.empty else "0/0",
            "max_abs_group_mean_pct": max_mean,
            "max_group_sd_pct": max_sd,
            "u_nor_L_pct": u_l,
            "u_nor_r_pct": u_r,
            "u_nor_d_pct": u_d,
            "pass_group_mean_all": pass_group_mean_all,
            "pass_group_sd_all": pass_group_sd_all,
            "pass_u_nor_L": bool(pd.notna(u_l) and u_l < STANDARD_LIMITS["u_nor_L_pct"]),
            "pass_u_nor_r": bool(pd.notna(u_r) and u_r < STANDARD_LIMITS["u_nor_r_pct"]),
            "pass_u_nor_d": bool(pd.notna(u_d) and u_d < STANDARD_LIMITS["u_nor_d_pct"]),
        }
    )
    row["pass_all_standards"] = bool(
        row["pass_group_mean_all"]
        and row["pass_group_sd_all"]
        and row["pass_u_nor_L"]
        and row["pass_u_nor_r"]
        and row["pass_u_nor_d"]
    )
    windows = df[
        [
            "window_id",
            "date",
            "flow_point",
            "condition_note",
            "disturbance_id",
            "standard_volume_m3",
            "model_volume_m3",
            "model_error_pct",
        ]
    ].copy()
    return row, groups, windows


def standardsGap(row: pd.Series) -> tuple[int, float]:
    checks = [
        ("max_abs_group_mean_pct", STANDARD_LIMITS["group_mean_abs_pct"], False),
        ("max_group_sd_pct", STANDARD_LIMITS["group_sd_pct"], False),
        ("u_nor_L_pct", STANDARD_LIMITS["u_nor_L_pct"], True),
        ("u_nor_r_pct", STANDARD_LIMITS["u_nor_r_pct"], True),
        ("u_nor_d_pct", STANDARD_LIMITS["u_nor_d_pct"], True),
    ]
    fails = 0
    gap = 0.0
    for col, limit, strict in checks:
        value = row.get(col, np.nan)
        if pd.isna(value):
            fails += 1
            gap += 10.0
            continue
        failed = value >= limit if strict else value > limit
        if failed:
            fails += 1
            gap += (float(value) - limit) / limit
    gap += float(row.get("mae_pct", 0.0))
    return fails, gap


def buildSpecs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for base_col in ["phys6_volume_m3", "lagrange_volume_m3", "owics_volume_m3"]:
        specs.append(
            ModelSpec(
                model_id=f"baseline_{base_col}",
                family="fixed_weight_integral",
                base_col=base_col,
                feature_set="none",
                model_type="baseline",
            )
        )

    for base_col in ["phys6_volume_m3", "lagrange_volume_m3"]:
        for feature_set in ["profile_flow", "profile_dynamic_zero", "runtime_full"]:
            for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
                specs.append(
                    ModelSpec(
                        model_id=f"ridge_{base_col}_{feature_set}_lam{lam:g}",
                        family=f"ridge_{feature_set}",
                        base_col=base_col,
                        feature_set=feature_set,
                        model_type="ridge",
                        ridge_lambda=lam,
                        shrinkage=1.0,
                    )
                )

    for base_col in ["phys6_volume_m3", "lagrange_volume_m3"]:
        for hidden in [8, 16, 24]:
            for l2 in [1e-4, 1e-3, 1e-2]:
                for lr in [0.003, 0.01]:
                    for seed_offset in [0, 17]:
                        for shrinkage in [0.50, 0.75, 1.00]:
                            specs.append(
                                ModelSpec(
                                    model_id=(
                                        f"ann_{base_col}_runtime_full_h{hidden}_"
                                        f"l2{l2:g}_lr{lr:g}_s{seed_offset}_shrink{shrinkage:g}"
                                    ),
                                    family="ann_runtime_full",
                                    base_col=base_col,
                                    feature_set="runtime_full",
                                    model_type="mlp",
                                    hidden_size=hidden,
                                    l2=l2,
                                    learning_rate=lr,
                                    seed_offset=seed_offset,
                                    shrinkage=shrinkage,
                                    epochs=MLP_EPOCHS,
                                )
                            )
    return specs


def runSpecs(
    specs: list[ModelSpec],
    train: pd.DataFrame,
    evaluations: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, FittedModel], dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    train_features_all = buildFeatureFrame(train)
    eval_features_all = {name: buildFeatureFrame(frame) for name, frame in evaluations.items()}
    rows = []
    fitted: dict[str, FittedModel] = {}
    groups_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    windows_by_key: dict[tuple[str, str], pd.DataFrame] = {}

    train_cache: dict[tuple, FittedModel] = {}
    for spec in specs:
        train_key = (
            spec.model_type,
            spec.base_col,
            spec.feature_set,
            spec.ridge_lambda,
            spec.hidden_size,
            spec.l2,
            spec.learning_rate,
            spec.seed_offset,
            spec.epochs,
        )
        if spec.model_type == "mlp":
            unshrunk_spec = ModelSpec(
                model_id=spec.model_id.rsplit("_shrink", 1)[0] + "_shrink1",
                family=spec.family,
                base_col=spec.base_col,
                feature_set=spec.feature_set,
                model_type=spec.model_type,
                hidden_size=spec.hidden_size,
                l2=spec.l2,
                learning_rate=spec.learning_rate,
                seed_offset=spec.seed_offset,
                shrinkage=1.0,
                epochs=spec.epochs,
            )
            train_key = (
                spec.model_type,
                spec.base_col,
                spec.feature_set,
                spec.hidden_size,
                spec.l2,
                spec.learning_rate,
                spec.seed_offset,
                spec.epochs,
            )
            if train_key not in train_cache:
                train_cache[train_key] = fitModel(unshrunk_spec, train, train_features_all)
            base_model = train_cache[train_key]
            model = FittedModel(
                spec=spec,
                standardizer=base_model.standardizer,
                w1=base_model.w1,
                b1=base_model.b1,
                w2=base_model.w2,
                b2=base_model.b2,
            )
        else:
            if train_key not in train_cache:
                train_cache[train_key] = fitModel(spec, train, train_features_all)
            model = train_cache[train_key]
            if model.spec.model_id != spec.model_id:
                model = FittedModel(spec=spec, standardizer=model.standardizer, beta=model.beta)
        fitted[spec.model_id] = model

        for operation, frame in evaluations.items():
            features = eval_features_all[operation]
            pred = model.predictVolume(frame, features)
            summary, groups, windows = evaluatePredictions(frame, pred)
            fail_count, gap = standardsGap(summary)
            row = {
                "model_id": spec.model_id,
                "family": spec.family,
                "operation": operation,
                "model_type": spec.model_type,
                "base_col": spec.base_col,
                "feature_set": spec.feature_set,
                "ridge_lambda": spec.ridge_lambda,
                "hidden_size": spec.hidden_size,
                "l2": spec.l2,
                "learning_rate": spec.learning_rate,
                "epochs": spec.epochs,
                "seed_offset": spec.seed_offset,
                "shrinkage": spec.shrinkage,
                "standard_fail_count": fail_count,
                "standard_gap_score": gap,
            }
            row.update(summary.to_dict())
            rows.append(row)
            groups_by_key[(operation, spec.model_id)] = groups
            windows_by_key[(operation, spec.model_id)] = windows

    metrics = pd.DataFrame(rows)
    return metrics, fitted, groups_by_key, windows_by_key


def buildComprehensiveRanking(metrics: pd.DataFrame) -> pd.DataFrame:
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
        rank_score = (
            1000.0 * float(v["standard_fail_count"])
            + 80.0 * float(v["standard_gap_score"])
            + 250.0 * float(f["standard_fail_count"])
            + 20.0 * float(f["standard_gap_score"])
            + 10.0 * overfit
            + float(v["mae_pct"])
        )
        rows.append(
            {
                "model_id": model_id,
                "family": v["family"],
                "model_type": v["model_type"],
                "base_col": v["base_col"],
                "feature_set": v["feature_set"],
                "rank_score": rank_score,
                "validation_fail_count": int(v["standard_fail_count"]),
                "full_fail_count": int(f["standard_fail_count"]),
                "validation_pass_all": bool(v["pass_all_standards"]),
                "full_pass_all": bool(f["pass_all_standards"]),
                "validation_mae_pct": float(v["mae_pct"]),
                "validation_mean_error_pct": float(v["mean_error_pct"]),
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
                "hidden_size": int(v["hidden_size"]),
                "ridge_lambda": float(v["ridge_lambda"]),
                "l2": float(v["l2"]),
                "learning_rate": float(v["learning_rate"]),
                "epochs": int(v["epochs"]),
                "shrinkage": float(v["shrinkage"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["rank_score", "validation_mae_pct"]).reset_index(drop=True)


def buildPressureRanking(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics[metrics["operation"].eq("full_fit_all")].set_index("model_id")
    train = metrics[metrics["operation"].eq("group_holdout_train")].set_index("model_id")
    valid = metrics[metrics["operation"].eq("group_holdout_validation")].set_index("model_id")
    rows = []
    for model_id, v in valid.iterrows():
        if model_id not in full.index:
            continue
        f = full.loc[model_id]
        t = train.loc[model_id] if model_id in train.index else v
        overfit = max(0.0, float(v["mae_pct"]) - float(t["mae_pct"]))
        rank_score = (
            1000.0 * float(v["standard_fail_count"])
            + 80.0 * float(v["standard_gap_score"])
            + 250.0 * float(f["standard_fail_count"])
            + 20.0 * float(f["standard_gap_score"])
            + 10.0 * overfit
            + float(v["mae_pct"])
        )
        rows.append(
            {
                "model_id": model_id,
                "family": v["family"],
                "model_type": v["model_type"],
                "base_col": v["base_col"],
                "feature_set": v["feature_set"],
                "rank_score": rank_score,
                "pressure_fail_count": int(v["standard_fail_count"]),
                "full_fail_count": int(f["standard_fail_count"]),
                "pressure_pass_all": bool(v["pass_all_standards"]),
                "full_pass_all": bool(f["pass_all_standards"]),
                "pressure_mae_pct": float(v["mae_pct"]),
                "pressure_mean_error_pct": float(v["mean_error_pct"]),
                "pressure_group_pass": v["group_pass"],
                "pressure_max_abs_group_mean_pct": float(v["max_abs_group_mean_pct"]),
                "pressure_max_group_sd_pct": float(v["max_group_sd_pct"]),
                "pressure_u_nor_L_pct": float(v["u_nor_L_pct"]),
                "pressure_u_nor_r_pct": float(v["u_nor_r_pct"]),
                "pressure_u_nor_d_pct": float(v["u_nor_d_pct"]),
                "pressure_train_mae_pct": float(t["mae_pct"]),
                "full_mae_pct": float(f["mae_pct"]),
                "full_group_pass": f["group_pass"],
                "full_u_nor_L_pct": float(f["u_nor_L_pct"]),
                "full_u_nor_r_pct": float(f["u_nor_r_pct"]),
                "full_u_nor_d_pct": float(f["u_nor_d_pct"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["rank_score", "pressure_mae_pct"]).reset_index(drop=True)


def bestByFamily(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, sub in ranking.groupby("family", sort=False):
        rows.append(sub.sort_values(["rank_score", "validation_mae_pct"]).iloc[0])
    return pd.DataFrame(rows).sort_values(["rank_score", "validation_mae_pct"]).reset_index(drop=True)


def buildFullFitRanking(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics[metrics["operation"].eq("full_fit_all")].copy()
    full = full.sort_values(["standard_fail_count", "standard_gap_score", "mae_pct"]).reset_index(drop=True)
    cols = [
        "model_id",
        "family",
        "model_type",
        "base_col",
        "feature_set",
        "standard_fail_count",
        "standard_gap_score",
        "pass_all_standards",
        "mae_pct",
        "mean_error_pct",
        "group_pass",
        "max_abs_group_mean_pct",
        "max_group_sd_pct",
        "u_nor_L_pct",
        "u_nor_r_pct",
        "u_nor_d_pct",
        "hidden_size",
        "ridge_lambda",
        "l2",
        "learning_rate",
        "epochs",
        "shrinkage",
    ]
    return full[cols]


def markdownTable(df: pd.DataFrame, max_rows: int | None = None, floatfmt: str = ".6g") -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_无数据_"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.astype(str).to_numpy()]
    return "\n".join([header, sep] + rows)


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


def drawRankingFigure(path: Path, ranking: pd.DataFrame) -> None:
    if canvas is None:
        return
    font = fontName()
    top = ranking.head(10).copy()
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    width, height = landscape(A4)
    c.setFont(font, 12)
    c.drawString(42, height - 40, "模型综合排名：随机验证 MAE 与达标差距")
    chart_x, chart_y, chart_w, chart_h = 70, 92, 650, 360
    max_mae = max(float(top["validation_mae_pct"].max()), 1e-9)
    row_h = chart_h / max(1, len(top))
    for i, row in enumerate(top.itertuples(index=False)):
        y = chart_y + chart_h - (i + 1) * row_h + 5
        bar_w = chart_w * float(row.validation_mae_pct) / max_mae
        color = "#22C55E" if bool(row.validation_pass_all) else "#3B82F6"
        c.setFillColor(colors.HexColor(color))
        c.rect(chart_x, y, bar_w, row_h * 0.55, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#111827"))
        c.setFont(font, 8)
        label = str(row.family)[:28] + f" ({row.base_col.replace('_volume_m3', '')})"
        c.drawRightString(chart_x - 6, y + 2, label)
        c.drawString(chart_x + bar_w + 5, y + 2, f"{float(row.validation_mae_pct):.4f}% / fail {int(row.validation_fail_count)}")
    c.setFont(font, 9)
    c.drawString(chart_x + chart_w / 2 - 40, 42, "validation MAE / %")
    c.save()


def drawStandardGapFigure(path: Path, best: pd.Series, metrics: pd.DataFrame) -> None:
    if canvas is None:
        return
    font = fontName()
    rows = metrics[(metrics["model_id"].eq(best["model_id"])) & (metrics["operation"].isin(["full_fit_all", "random_split_validation"]))]
    labels = [
        ("max_abs_group_mean_pct", STANDARD_LIMITS["group_mean_abs_pct"], "|eg_bar|"),
        ("max_group_sd_pct", STANDARD_LIMITS["group_sd_pct"], "SDg"),
        ("u_nor_L_pct", STANDARD_LIMITS["u_nor_L_pct"], "u_nor,L"),
        ("u_nor_r_pct", STANDARD_LIMITS["u_nor_r_pct"], "u_nor,r"),
        ("u_nor_d_pct", STANDARD_LIMITS["u_nor_d_pct"], "u_nor,d"),
    ]
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    width, height = landscape(A4)
    c.setFont(font, 12)
    c.drawString(42, height - 40, "最优模型与达标阈值对比")
    x0, y0, bar_w, row_h = 220, height - 95, 420, 42
    palette = {"full_fit_all": "#10B981", "random_split_validation": "#2563EB"}
    for i, (metric, limit, label) in enumerate(labels):
        y = y0 - i * row_h
        c.setFillColor(colors.HexColor("#111827"))
        c.setFont(font, 9)
        c.drawRightString(x0 - 10, y + 6, f"{label} limit={limit:g}%")
        c.setStrokeColor(colors.HexColor("#EF4444"))
        c.line(x0 + bar_w * 0.5, y - 3, x0 + bar_w * 0.5, y + 28)
        for j, r in enumerate(rows.itertuples(index=False)):
            value = float(getattr(r, metric)) if pd.notna(getattr(r, metric)) else 0.0
            width_v = min(bar_w, bar_w * value / max(2.0 * limit, 1e-9))
            yy = y + 15 - j * 14
            c.setFillColor(colors.HexColor(palette[getattr(r, "operation")]))
            c.rect(x0, yy, width_v, 8, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#111827"))
            c.drawString(x0 + width_v + 5, yy - 1, f"{getattr(r, 'operation')}: {value:.5f}%")
    c.save()


def theorySections() -> str:
    return """## 1. 论文库理论与公式准备

本节把论文库中可直接支撑最终建模的理论压缩为可落地公式。若多个理论重合，优先采用近期期刊和本题数据字段可以直接实现的版本。

### 1.1 多声道积分与圆截面平均速度

多声道 UFM 的基本目标是由各弦高处线平均速度 `u_i` 估计截面平均流速。圆截面平均速度可写成带几何权重的积分：

```text
v_bar = (2/pi) Integral[-1,1] u(eta) sqrt(1-eta^2) d eta
V_hat = A T v_bar
```

其中 `eta=y/R`，`A` 为等效截面积，`T` 为窗口时间。Gauss-Jacobi 和 OWICS 把圆截面几何权重或圆截面最优权重并入离散积分；Lagrange 型积分先用五个弦高节点重构速度剖面，再积分得到固定权重。这对应相诺林关于高精度多声道积分和影响因素的结论：物理积分是主尺度，速度剖面变化会导致固定权重失配。

### 1.2 剖面修正因子

Mousavi 等关于 geometric flow profile correction factor 的思想可写为：

```text
v_bar = K(profile) * v_path
```

在本题的五声道场景下，将其扩展为：

```text
V_hat = V_base * C(x),    C(x)=1+f(x)/100
```

其中 `V_base` 取 Phys6、Lagrange 或 OWICS 等物理积分估计，`x` 为中心-边缘、边缘-内侧、上下弦差、A/B 方向差、旋流代理等运行时剖面特征。Gryshanova 2023 和 Sakhavi 2022 的结论共同支持：非轴对称剖面会改变修正因子，模型不能只假设单一理想剖面。

### 1.3 扰流补偿与在线可观测特征

Papathanasiou 2022、金少杰 2025 和苏杰 2024 均说明上游扰流、弯管、非充分发展流会导致声道方向和剖面形态出现结构化偏移。因此本文不把 `disturbance_id` 当作输入，而是使用在线可观测代理：

```text
x_profile = (profile_top_bottom, profile_center_all, profile_edge_inner,
             profile_inner_skew, profile_ab_abs, profile_swirl)
```

这些变量分别刻画上下不对称、中心高速核、近壁梯度、内侧偏斜、A/B 方向差和旋流/偏流。

### 1.4 机器学习残差补偿

Zhao 2014 ANN、Li 2023 in-use measurement 和黄恩扬 2024 BP 神经网络补偿都支持用数据驱动方法学习多声道特征到测量误差的映射。为避免黑箱模型脱离物理主尺度，本文只学习相对残差：

```text
r = (V_ref / V_base - 1) * 100
V_hat = V_base * (1 + f_theta(x) / 100)
```

线性补偿采用岭回归：

```text
min_beta sum_j (r_j - beta^T z_j)^2 + lambda ||beta||_2^2
```

ANN 补偿采用一隐层 BP/MLP：

```text
f_theta(x)=a_2^T tanh(W_1 z+b_1)+b_2
```

结合 Cai 2025 物理约束数据增强/校准思想，最终将补偿幅度限制为 `|f_theta(x)| <= 2%`，即修正因子在 `[0.98,1.02]` 内，防止小样本下过拟合出非物理补偿。

### 1.5 评价公式

窗口相对误差、组均值误差和组内重复性为：

```text
e_j = (V_hat_j - V_ref_j) / V_ref_j * 100%
eg_bar = mean_{j in g}(e_j)
SD_g = sqrt( sum_{j in g}(e_j - eg_bar)^2 / (n_g-1) )
```

综合不确定度采用题面/评价脚本口径：

```text
u_nor,L = sqrt( sum_q eg_q^2 / (m-1) )
u_nor,r = max_g SD_g
u_nor,d = sqrt( (max |eg_D0,q - eg_Dk,q| / sqrt(3))^2 + (max SD_Dk,q)^2 )
```

达标参考为 `|eg_bar| <= 0.2%`、`SD_g <= 0.040%`、`u_nor,L < 0.036%`、`u_nor,r < 0.040%`、`u_nor,d < 0.115%`。

### 1.6 论文索引到模型组件的对应关系

| 文献 | 可吸收理论/公式 | 本实验落地方式 |
| --- | --- | --- |
| Mousavi2020FPCF | 几何速度剖面修正因子 `v_bar=K(profile)v_path` | 把 `K` 写成 `C(x)=1+f(x)/100` |
| Sakhavi2022GeneralizedProfile | 多声道广义速度剖面评价 | 使用中心-边缘、边缘-内侧、上下偏斜等剖面特征 |
| Gryshanova2023CorrectionFactor | 非轴对称剖面下修正因子随测量平面变化 | 加入 A/B 方向差、上下弦差和旋流代理 |
| Papathanasiou2022FlowDC | 基于流场仿真的扰流补偿因子 | 采用扰流代理特征学习残差补偿，不直接输入扰流编号 |
| JinUpstreamDisturbance | 上游扰流误差可由修正系数显著降低 | 将扰流漂移纳入 `u_nor,d` 排名约束 |
| SuUnderdevelopedFlow | 非充分发展流中 A/B 声道组合存在偏移 | 使用 `profile_ab_abs`、`profile_swirl`、A/B 统计量 |
| XiangMultipathIntegration | 速度剖面决定积分误差，AWICS-VP 自适应权重优于固定权重 | 保留 Phys6/Lagrange/OWICS 作为物理基准，再学习小幅修正 |
| Zhao2014ANN | ANN 可融合多路径声道估计截面平均速度 | 构造一隐层 MLP/BP 残差模型 |
| Zhao2014CFD | CFD 揭示声道方向、扰流强度和复杂流型对误差的影响 | 用方向性和剖面特征解释 ML 输入，不把 ML 当纯黑箱 |
| Li2023MLInUse | RF/ANN 可预测在用超声流量计误差 | 将监督目标设为相对残差 `r=(V_ref/V_base-1)100%` |
| HuangClampOnOil | Pearson 筛选 + BP 神经网络可补偿外夹式 UFM 误差 | 使用 BP/MLP 与动态、零点、声道特征联合补偿 |
| Cai2025PhysConstrainedGAN | 小样本下应加入物理约束和分布一致性 | 补偿幅度裁剪到 `[−2%,2%]`，并用随机验证检查泛化 |
| Li2025VMDDenoising | 回波噪声会影响瞬时流速和稳定性 | 保留 `dyn_*`、`zero_*` 特征作为噪声/稳定性代理 |
| Qin2016ProfileIdentification | 多声道特征可识别流型且对不确定性较稳健 | 构造在线剖面代理，避免依赖人工 `disturbance_id` |
"""


def writeReport(
    data: pd.DataFrame,
    split: pd.DataFrame,
    pressureSplit: pd.DataFrame,
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    pressure_ranking: pd.DataFrame,
    family_ranking: pd.DataFrame,
    full_fit_ranking: pd.DataFrame,
    best_groups: pd.DataFrame,
    best_full_groups: pd.DataFrame,
) -> None:
    best = ranking.iloc[0]
    best_full = full_fit_ranking.iloc[0]
    split_counts = split["split"].value_counts().rename_axis("split").reset_index(name="count")
    pressure_split_counts = pressureSplit["split"].value_counts().rename_axis("split").reset_index(name="count")
    group_sort_cols = ["pass_group", "repeatability_sd_pct", "abs_mean_error_pct"]

    def groupMarkdown(groups: pd.DataFrame) -> str:
        if groups.empty or not set(group_sort_cols).issubset(groups.columns):
            return "_无可计算重复组。当前随机验证窗口数较少，未形成同日期同流量点至少 3 个窗口的组。_"
        return markdownTable(groups.sort_values(group_sort_cols, ascending=[True, False, False]), max_rows=12)

    std_rows = metrics[
        (metrics["model_id"].eq(best["model_id"]))
        & (
            metrics["operation"].isin(
                ["full_fit_all", "random_split_train", "random_split_validation", "group_holdout_validation"]
            )
        )
    ][
        [
            "operation",
            "window_count",
            "mae_pct",
            "mean_error_pct",
            "group_pass",
            "max_abs_group_mean_pct",
            "max_group_sd_pct",
            "u_nor_L_pct",
            "u_nor_r_pct",
            "u_nor_d_pct",
            "pass_all_standards",
        ]
    ]
    top_cols = [
        "model_id",
        "family",
        "validation_fail_count",
        "full_fail_count",
        "validation_mae_pct",
        "validation_group_pass",
        "validation_max_abs_group_mean_pct",
        "validation_max_group_sd_pct",
        "validation_u_nor_L_pct",
        "validation_u_nor_r_pct",
        "validation_u_nor_d_pct",
        "full_mae_pct",
        "full_group_pass",
        "rank_score",
    ]
    full_cols = [
        "model_id",
        "family",
        "standard_fail_count",
        "pass_all_standards",
        "mae_pct",
        "group_pass",
        "max_abs_group_mean_pct",
        "max_group_sd_pct",
        "u_nor_L_pct",
        "u_nor_r_pct",
        "u_nor_d_pct",
    ]
    family_cols = [
        "family",
        "model_id",
        "validation_fail_count",
        "validation_mae_pct",
        "validation_group_pass",
        "validation_u_nor_L_pct",
        "validation_u_nor_r_pct",
        "validation_u_nor_d_pct",
        "full_mae_pct",
    ]
    pressure_cols = [
        "model_id",
        "family",
        "pressure_fail_count",
        "full_fail_count",
        "pressure_mae_pct",
        "pressure_group_pass",
        "pressure_max_abs_group_mean_pct",
        "pressure_max_group_sd_pct",
        "pressure_u_nor_L_pct",
        "pressure_u_nor_r_pct",
        "pressure_u_nor_d_pct",
        "full_mae_pct",
        "full_group_pass",
        "rank_score",
    ]
    report = f"""# 最终问题建模理论准备与组合模型排名

数据窗口数：`{len(data)}`；主验证采用 `(date, flow_point, disturbance_id)` 组内分层划分，目标比例为训练约 `70%`、验证约 `30%`；同时保留整组留出压力测试；划分种子：`{RANDOM_SEED}`；ANN 训练轮数：`{MLP_EPOCHS}`。

{theorySections()}

## 2. 候选组合模型

在问题二“物理积分 + 剖面修正 + 流量点修正曲线”的基础上，本实验增加三类补偿组合：

1. 固定物理积分基准：`Phys6`、`Lagrange`、`OWICS`，作为不训练对照。
2. `物理积分 + 岭回归修正因子`：分别使用 `profile_flow`、`profile_dynamic_zero`、`runtime_full` 三组特征，搜索 `lambda=0.01--100`。
3. `物理积分 + ANN/BP 残差补偿`：使用运行时全特征 `runtime_full`，隐藏层 `8/16/24`，L2 正则 `1e-4/1e-3/1e-2`，学习率 `0.003/0.01`，每个候选训练 `{MLP_EPOCHS}` 轮，并测试残差收缩系数 `0.50/0.75/1.00`。

所有模型输入均不包含 `date`、`window_id`、`standard_volume_m3`、`standard_rate_m3h`、`condition_note`、`disturbance_id`、`disturbance_angle_deg`。这些字段只用于离线划分和评价。

## 3. 训练操作

### 3.1 全量拟合

每个候选模型用全部 `{len(data)}` 个窗口估计参数，再在全部窗口上评价。该结果反映模型能否充分拟合已有标定数据，但不能单独代表泛化能力。

### 3.2 组内分层验证

主验证划分如下：

{markdownTable(split_counts)}

本次主验证按 `(date, flow_point, disturbance_id)` 分层：对 `n >= 5` 的重复组固定抽取 2 个窗口进入训练集，其余窗口进入验证集；对 `n = 3` 或 `4` 的重复组整组放入训练或验证，不再强拆；同时保留 D0 无扰流锚点以支持 `u_nor,L` 和 `u_nor,d` 计算。

每个候选模型只在训练集拟合参数，并分别输出训练集和验证集指标。最终排名以验证集达标情况为主，全量拟合结果为辅。

### 3.3 整组留出压力测试

压力测试划分如下：

{markdownTable(pressure_split_counts)}

压力测试把完整的 `(date, flow_point, disturbance_id)` 重复组直接留出，用来观察模型对未参与训练的重复工况是否仍然稳定；该结果比组内分层验证更严格，但也更容易暴露跨工况外推不足。

## 4. 综合排名结果

排名规则：优先减少验证集达标失败项，其次考虑验证集超过阈值的归一化差距、全量拟合达标情况、验证 MAE 和过拟合差距。

### 4.1 前 15 个候选模型

{markdownTable(ranking[top_cols], max_rows=15)}

### 4.2 全量拟合提交候选排名

全量拟合用于最终提交候选选择。若只追求训练集指标，ANN 容易非常低误差；因此本表只作为“全量标定后能达到什么水平”的证据，不能替代随机验证。

{markdownTable(full_fit_ranking[full_cols], max_rows=12)}

### 4.3 各模型族最佳方案

{markdownTable(family_ranking[family_cols], max_rows=12)}

### 4.4 整组留出压力测试排名

{markdownTable(pressure_ranking[pressure_cols], max_rows=12) if not pressure_ranking.empty else "_无压力测试结果_"}

## 5. 最优模型详细指标

综合最优模型为：

```text
{best['model_id']}
```

其两个训练操作和验证结果为：

{markdownTable(std_rows)}

最优模型验证集分组表现如下：

{groupMarkdown(best_groups)}

全量拟合提交候选的最佳模型为：

```text
{best_full['model_id']}
```

该模型在全部窗口上的分组表现如下：

{groupMarkdown(best_full_groups)}

## 6. 与达标标准的比较

- `|eg_bar| <= 0.2%`：看 `max_abs_group_mean_pct` 是否不超过 `0.2`。
- `SD_g <= 0.040%`：看 `max_group_sd_pct` 或 `u_nor,r` 是否不超过 `0.040`。
- `u_nor,L < 0.036%`、`u_nor,r < 0.040%`、`u_nor,d < 0.115%`：按评价脚本公式计算。

从当前随机验证结果看，若 `pass_all_standards=False`，说明模型虽然降低了 MAE，但仍有至少一个稳定性或扰流漂移指标没有达标。最终论文中应把这一点写成“模型改进方向”，而不能只引用全量拟合的训练误差。

## 7. 输出文件

- `code/final_model_ranking_experiment.py`：本实验脚本。
- `code/outputs/final_model_metrics_long.csv`：所有候选模型在全量、训练、验证三种评价集上的指标。
- `code/outputs/final_model_comprehensive_ranking.csv`：综合排名表。
- `code/outputs/final_model_full_fit_ranking.csv`：全量拟合提交候选排名。
- `code/outputs/final_model_family_ranking.csv`：按模型族汇总的最佳方案。
- `code/outputs/final_model_best_full_submission.csv`：综合最优方案对应的全量拟合预测，可作为最终提交候选。
- `code/outputs/final_model_best_full_pass_submission.csv`：全量拟合达标优先的提交候选。
- `figures/fig_final_model_validation_ranking.pdf`：验证集排名图。
- `figures/fig_final_model_standard_gap.pdf`：最优模型与达标阈值对比图。
"""
    REPORT.write_text(report, encoding="utf-8")


def saveModelMetadata(best_model: FittedModel, best_full_model: FittedModel, ranking: pd.DataFrame, full_fit_ranking: pd.DataFrame) -> None:
    meta = {
        "best_validation_primary_model_id": best_model.spec.model_id,
        "best_full_fit_pass_model_id": best_full_model.spec.model_id,
        "best_validation_primary": best_model.spec.__dict__,
        "best_full_fit_pass": best_full_model.spec.__dict__,
        "correction_clip_pct": CORRECTION_CLIP_PCT,
        "random_seed": RANDOM_SEED,
        "python": sys.version.split()[0],
        "top_rank_score": float(ranking.iloc[0]["rank_score"]),
        "top_full_fit_score": float(full_fit_ranking.iloc[0]["standard_gap_score"]),
    }
    (OUT_DIR / "final_model_best_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensureDirs()
    data = loadData()
    train, validation, split = makeRandomSplit(data)
    pressure_train, pressure_validation, pressureSplit = makeGroupHoldoutPressureSplit(data)
    specs = buildSpecs()

    full_metrics, full_models, full_groups, full_windows = runSpecs(specs, data, {"full_fit_all": data})
    split_metrics, split_models, split_groups, split_windows = runSpecs(
        specs,
        train,
        {"random_split_train": train, "random_split_validation": validation},
    )
    pressure_metrics, pressure_models, pressure_groups, pressure_windows = runSpecs(
        specs,
        pressure_train,
        {"group_holdout_train": pressure_train, "group_holdout_validation": pressure_validation},
    )
    metrics = pd.concat([full_metrics, split_metrics, pressure_metrics], ignore_index=True)
    ranking = buildComprehensiveRanking(metrics)
    pressure_ranking = buildPressureRanking(metrics)
    full_fit_ranking = buildFullFitRanking(metrics)
    family_ranking = bestByFamily(ranking)

    best_id = str(ranking.iloc[0]["model_id"])
    best_full_id = str(full_fit_ranking.iloc[0]["model_id"])
    pressure_best_id = str(pressure_ranking.iloc[0]["model_id"]) if not pressure_ranking.empty else ""
    best_full_model = full_models[best_id]
    best_full_pass_model = full_models[best_full_id]
    best_full_windows = full_windows[("full_fit_all", best_id)]
    best_full_pass_windows = full_windows[("full_fit_all", best_full_id)]
    best_full_pass_groups = full_groups[("full_fit_all", best_full_id)]
    best_validation_groups = split_groups[("random_split_validation", best_id)]
    best_validation_windows = split_windows[("random_split_validation", best_id)]
    pressure_best_groups = (
        pressure_groups[("group_holdout_validation", pressure_best_id)] if pressure_best_id else pd.DataFrame()
    )
    pressure_best_windows = (
        pressure_windows[("group_holdout_validation", pressure_best_id)] if pressure_best_id else pd.DataFrame()
    )

    metrics.to_csv(OUT_DIR / "final_model_metrics_long.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    ranking.to_csv(OUT_DIR / "final_model_comprehensive_ranking.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    pressure_ranking.to_csv(OUT_DIR / "final_model_group_holdout_pressure_ranking.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    full_fit_ranking.to_csv(OUT_DIR / "final_model_full_fit_ranking.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    family_ranking.to_csv(OUT_DIR / "final_model_family_ranking.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    split.to_csv(OUT_DIR / "final_model_random_split_assignments.csv", index=False, encoding="utf-8-sig")
    pressureSplit.to_csv(OUT_DIR / "final_model_group_holdout_pressure_split_assignments.csv", index=False, encoding="utf-8-sig")
    best_validation_groups.to_csv(OUT_DIR / "final_model_best_validation_groups.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    best_validation_windows.to_csv(OUT_DIR / "final_model_best_validation_windows.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    if not pressure_best_groups.empty:
        pressure_best_groups.to_csv(OUT_DIR / "final_model_best_pressure_groups.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
        pressure_best_windows.to_csv(OUT_DIR / "final_model_best_pressure_windows.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    best_full_windows.to_csv(OUT_DIR / "final_model_best_full_windows.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    best_full_pass_groups.to_csv(OUT_DIR / "final_model_best_full_pass_groups.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    best_full_pass_windows.to_csv(OUT_DIR / "final_model_best_full_pass_windows.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    best_full_windows[["window_id", "model_volume_m3"]].to_csv(
        OUT_DIR / "final_model_best_full_submission.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    best_full_pass_windows[["window_id", "model_volume_m3"]].to_csv(
        OUT_DIR / "final_model_best_full_pass_submission.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    drawRankingFigure(FIG_DIR / "fig_final_model_validation_ranking.pdf", ranking)
    drawStandardGapFigure(FIG_DIR / "fig_final_model_standard_gap.pdf", ranking.iloc[0], metrics)
    drawStandardGapFigure(FIG_DIR / "fig_final_model_full_pass_standard_gap.pdf", full_fit_ranking.iloc[0], metrics)
    writeReport(
        data,
        split,
        pressureSplit,
        metrics,
        ranking,
        pressure_ranking,
        family_ranking,
        full_fit_ranking,
        best_validation_groups,
        best_full_pass_groups,
    )
    saveModelMetadata(best_full_model, best_full_pass_model, ranking, full_fit_ranking)

    print("Best model:", best_id)
    print("Best full-fit pass candidate:", best_full_id)
    if pressure_best_id:
        print("Best group-holdout pressure model:", pressure_best_id)
    print("Family ranking:")
    print(
        family_ranking[
            [
                "family",
                "model_id",
                "validation_fail_count",
                "validation_mae_pct",
                "validation_group_pass",
                "validation_u_nor_L_pct",
                "validation_u_nor_r_pct",
                "validation_u_nor_d_pct",
                "full_mae_pct",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    print("Report:", REPORT)


if __name__ == "__main__":
    main()
