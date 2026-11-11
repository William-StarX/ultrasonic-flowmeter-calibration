from __future__ import annotations

# 论文附录复现代码。
# 问题四泛化模型：拟合紧凑物理特征 RBF-KRR。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model_evolution_ridge_ann import (
    GROUP_KEY,
    OUT_DIR,
    RANDOM_SEED,
    STANDARD_LIMITS,
    buildFeatureFrame,
    evaluatePredictions,
    featureColumns,
    loadData,
    standardsGap,
)
from physical_feature_library import enhancedFeatureFrame


ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent / "verification_reports" / "STRATIFIED_MODEL_SEARCH_REPORT.md"
TARGET_VALIDATION_FRACTION = 0.30
N_SPLITS = 40


@dataclass(frozen=True)
class SplitSpec:
    seed: int
    d0_n3_to_validation: bool
    n34_validation_bias: float
    small_validation_prob: float


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    family: str
    base_col: str
    feature_set: str
    model_type: str
    lam: float = 1.0
    gamma: float = 1.0
    poly_degree: int = 2
    anchor: str = "none"
    anchor_feature: str = ""
    anchor_strength: float = 1.0
    blend_d0_curve: bool = False


@dataclass
class StandardizerLite:
    columns: list[str]
    median: pd.Series
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "StandardizerLite":
        clean = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
        median = clean.median(numeric_only=True).fillna(0.0)
        clean = clean.fillna(median)
        arr = clean.to_numpy(dtype=float)
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(columns, median, mean, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        clean = frame[self.columns].replace([np.inf, -np.inf], np.nan).copy()
        clean = clean.fillna(self.median)
        return (clean.to_numpy(dtype=float) - self.mean) / self.scale


class FittedCandidate:
    def __init__(
        self,
        spec: CandidateSpec,
        standardizer: StandardizerLite | None = None,
        coef: np.ndarray | None = None,
        train_x: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
        train_y: np.ndarray | None = None,
        direct_coef: np.ndarray | None = None,
        anchor_params: dict[tuple, tuple[float, float, str]] | None = None,
        d0_curve_coef: np.ndarray | None = None,
    ) -> None:
        self.spec = spec
        self.standardizer = standardizer
        self.coef = coef
        self.train_x = train_x
        self.alpha = alpha
        self.train_y = train_y
        self.direct_coef = direct_coef
        self.anchor_params = anchor_params or {}
        self.d0_curve_coef = d0_curve_coef

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.spec.model_type == "krr_poly":
            return (1.0 + x @ y.T) ** self.spec.poly_degree
        x2 = np.sum(x * x, axis=1)[:, None]
        y2 = np.sum(y * y, axis=1)[None, :]
        dist2 = np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)
        return np.exp(-self.spec.gamma * dist2)

    def rawCorrectionPct(self, features: pd.DataFrame) -> np.ndarray:
        if self.spec.model_type == "baseline":
            return np.zeros(len(features))
        if self.spec.model_type == "direct_ridge_volume":
            return np.zeros(len(features))
        if self.standardizer is None:
            raise ValueError("missing standardizer")
        x = self.standardizer.transform(features)
        if self.spec.model_type == "ridge":
            if self.coef is None:
                raise ValueError("missing coefficients")
            return x @ self.coef
        if self.spec.model_type in {"krr_rbf", "krr_poly"}:
            if self.train_x is None or self.alpha is None:
                raise ValueError("missing kernel fit")
            return self._kernel(x, self.train_x) @ self.alpha
        if self.spec.model_type == "knn_rbf":
            if self.train_x is None or self.train_y is None:
                raise ValueError("missing knn fit")
            k = self._kernel(x, self.train_x)
            denom = np.maximum(k.sum(axis=1), 1e-12)
            return (k @ self.train_y) / denom
        raise ValueError(self.spec.model_type)

    def _baseVolume(self, data: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
        if self.spec.model_type == "direct_ridge_volume":
            if self.direct_coef is None:
                raise ValueError("missing direct coefficients")
            cols = [
                "phys6_volume_m3",
                "lagrange_volume_m3",
                "owics_volume_m3",
                "equal_weight_volume_m3",
                "duration_s",
                "q",
                "q2",
                "q3",
            ]
            x = features[[c for c in cols if c in features.columns]].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            return x @ self.direct_coef
        return data[self.spec.base_col].to_numpy(dtype=float)

    def predictVolume(self, data: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
        base = self._baseVolume(data, features)
        correction = np.clip(self.rawCorrectionPct(features), -3.0, 3.0)

        if self.d0_curve_coef is not None and self.spec.blend_d0_curve:
            q = features["q"].to_numpy(dtype=float)
            qx = np.column_stack([np.ones(len(q)), q, q**2])
            correction += qx @ self.d0_curve_coef

        if self.anchor_params and self.spec.anchor != "none":
            correction = correction.copy()
            for i, row in data.reset_index(drop=True).iterrows():
                key = tuple(row[k] for k in GROUP_KEY)
                if key not in self.anchor_params:
                    continue
                intercept, slope, feature = self.anchor_params[key]
                z = float(features.iloc[i][feature]) if feature and feature in features.columns else 0.0
                correction[i] += self.spec.anchor_strength * (intercept + slope * z)

        return base * (1.0 + correction / 100.0)


def residualTargetPct(data: pd.DataFrame, base_col: str) -> np.ndarray:
    base = np.maximum(data[base_col].to_numpy(dtype=float), 1e-12)
    return (data["standard_volume_m3"].to_numpy(dtype=float) / base - 1.0) * 100.0


def chooseN5TrainIndices(indices: np.ndarray, rng: np.random.Generator) -> set[int]:
    """Choose two training windows without consulting labels or baseline errors."""
    idx = np.asarray(indices, dtype=int)
    if len(idx) <= 2:
        return set(int(i) for i in idx)
    return set(int(i) for i in rng.choice(idx, size=2, replace=False))


def splitAssignments(data: pd.DataFrame, spec: SplitSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(spec.seed)
    target_validation = int(round(len(data) * TARGET_VALIDATION_FRACTION))
    train_idx: set[int] = set()
    validation_idx: set[int] = set()

    groups = list(data.groupby(GROUP_KEY, sort=True))

    # D0 is sparse and needed by u_nor,L/u_nor,d. Split n=2 D0 groups 1/1,
    # keep n=3 whole, and use the n=1 flow as a validation anchor in half the splits.
    for key, group in groups:
        disturbance_id = str(key[2])
        if disturbance_id != "D0":
            continue
        idx = group.index.to_numpy()
        n = len(idx)
        if n >= 5:
            pick = chooseN5TrainIndices(idx, rng)
            train_idx.update(pick)
            validation_idx.update(int(i) for i in idx if int(i) not in pick)
        elif n in {3, 4}:
            if spec.d0_n3_to_validation:
                validation_idx.update(int(i) for i in idx)
            else:
                train_idx.update(int(i) for i in idx)
        elif n == 2:
            pick = int(rng.choice(idx))
            train_idx.add(pick)
            validation_idx.update(int(i) for i in idx if int(i) != pick)
        else:
            if rng.random() < 0.7:
                validation_idx.update(int(i) for i in idx)
            else:
                train_idx.update(int(i) for i in idx)

    # Repeated disturbed groups.
    n34_candidates: list[tuple[float, np.ndarray]] = []
    small_candidates: list[int] = []
    for key, group in groups:
        if str(key[2]) == "D0":
            continue
        idx = group.index.to_numpy()
        n = len(idx)
        if n >= 5:
            pick = chooseN5TrainIndices(idx, rng)
            train_idx.update(pick)
            validation_idx.update(int(i) for i in idx if int(i) not in pick)
        elif n in {3, 4}:
            flow = int(key[1])
            official_flow = 1.0 if 40 <= flow <= 100 else 0.0
            jitter = rng.random()
            score = spec.n34_validation_bias * official_flow + jitter
            n34_candidates.append((score, idx))
        else:
            small_candidates.extend(int(i) for i in idx)

    rng.shuffle(n34_candidates)
    n34_candidates = sorted(n34_candidates, key=lambda x: x[0], reverse=True)
    selected_n34_validation_groups = 0
    max_n34_validation_groups = 3 + (spec.seed % 2)
    for _, idx in n34_candidates:
        idx_set = set(int(i) for i in idx)
        if len(validation_idx) < target_validation and selected_n34_validation_groups < max_n34_validation_groups:
            validation_idx.update(idx_set)
            selected_n34_validation_groups += 1
        else:
            train_idx.update(idx_set)

    rng.shuffle(small_candidates)
    for i in small_candidates:
        if len(validation_idx) < target_validation and rng.random() < spec.small_validation_prob:
            validation_idx.add(int(i))
        else:
            train_idx.add(int(i))

    unassigned = set(range(len(data))) - train_idx - validation_idx
    for i in sorted(unassigned):
        if len(validation_idx) < target_validation:
            validation_idx.add(i)
        else:
            train_idx.add(i)

    overlap = train_idx & validation_idx
    if overlap:
        train_idx -= overlap

    split = data[["window_id", "date", "flow_point", "condition_note", "disturbance_id"]].copy()
    split["split"] = ["validation" if i in validation_idx else "train" for i in range(len(data))]
    split["split_mode"] = "label_blind_stratified_group_internal"
    split["split_seed"] = spec.seed
    train = data.loc[split["split"].eq("train")].reset_index(drop=True)
    validation = data.loc[split["split"].eq("validation")].reset_index(drop=True)
    return train, validation, split


def pressureSplit(data: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 9001)
    target_validation = int(round(len(data) * TARGET_VALIDATION_FRACTION))
    validation_idx: set[int] = set()
    candidates = []
    for key, group in data.groupby(GROUP_KEY, sort=True):
        flow = int(key[1])
        n = len(group)
        priority = 2 if str(key[2]) == "D0" else int(40 <= flow <= 100)
        candidates.append((priority, n, rng.random(), group.index.to_numpy()))
    for _, _, _, idx in sorted(candidates, key=lambda x: (-x[0], -x[1], x[2])):
        if len(validation_idx) >= target_validation:
            break
        validation_idx.update(int(i) for i in idx)
    train_idx = set(range(len(data))) - validation_idx
    split = data[["window_id", "date", "flow_point", "condition_note", "disturbance_id"]].copy()
    split["split"] = ["validation" if i in validation_idx else "train" for i in range(len(data))]
    split["split_mode"] = "searched_whole_group_holdout_pressure"
    split["split_seed"] = seed
    train = data.loc[split["split"].eq("train")].reset_index(drop=True)
    validation = data.loc[split["split"].eq("validation")].reset_index(drop=True)
    return train, validation, split


def validateSplitRules(data: pd.DataFrame, split: pd.DataFrame) -> tuple[bool, str]:
    merged = data[GROUP_KEY].copy()
    merged["split"] = split["split"].to_numpy()
    for key, group in merged.groupby(GROUP_KEY, sort=True):
        n = len(group)
        train_n = int(group["split"].eq("train").sum())
        val_n = int(group["split"].eq("validation").sum())
        if n >= 5 and not (train_n == 2 and val_n >= 3):
            return False, f"n>=5 rule failed for {key}: train={train_n}, validation={val_n}"
        if n in {3, 4} and not ((train_n == n and val_n == 0) or (train_n == 0 and val_n == n)):
            return False, f"n=3/4 whole-group rule failed for {key}: train={train_n}, validation={val_n}"
    return True, "ok"


def columnsForFeatureSet(frame: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "runtime_full":
        return featureColumns(frame, "runtime_full")
    if feature_set == "enhanced_proxy":
        proxy = [
            "q",
            "q2",
            "q3",
            "proxy_profile_bulge",
            "proxy_edge_gradient",
            "proxy_asymmetry",
            "proxy_ab_direction",
            "proxy_dynamic",
            "proxy_zero",
            "proxy_integral_disagreement",
        ]
        interactions = [c for c in frame.columns if c.startswith("proxy_") and (c.endswith("_x_q") or c.endswith("_x_q2"))]
        return [c for c in proxy + interactions if c in frame.columns]
    if feature_set == "compact_physical":
        cols = [
            "q",
            "q2",
            "profile_top_bottom",
            "profile_center_all",
            "profile_edge_inner",
            "profile_ab_abs",
            "profile_swirl",
            "chord_center_minus_edges_frac",
            "chord_edge_minus_inner_frac",
            "chord_top_bottom_frac",
            "ab_abs_mean",
            "dyn_plateau_cv",
            "zero_rate_med",
            "lagrange_volume_m3_minus_phys6_frac",
            "owics_volume_m3_minus_phys6_frac",
        ]
        return [c for c in cols if c in frame.columns]
    if feature_set == "direct_volume":
        return []
    raise ValueError(feature_set)


def fitCandidate(spec: CandidateSpec, train: pd.DataFrame, xunlianTezheng: pd.DataFrame) -> FittedCandidate:
    if spec.model_type == "baseline":
        return FittedCandidate(spec)
    if spec.model_type == "direct_ridge_volume":
        cols = [
            "phys6_volume_m3",
            "lagrange_volume_m3",
            "owics_volume_m3",
            "equal_weight_volume_m3",
            "duration_s",
            "q",
            "q2",
            "q3",
        ]
        cols = [c for c in cols if c in xunlianTezheng.columns]
        x = xunlianTezheng[cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        y = train["standard_volume_m3"].to_numpy(dtype=float)
        penalty = spec.lam * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        model = FittedCandidate(spec, direct_coef=coef)
        addPostCalibration(model, train, xunlianTezheng)
        return model

    cols = columnsForFeatureSet(xunlianTezheng, spec.feature_set)
    std = StandardizerLite.fit(xunlianTezheng, cols)
    x = std.transform(xunlianTezheng)
    y = residualTargetPct(train, spec.base_col)

    if spec.model_type == "ridge":
        penalty = spec.lam * np.eye(x.shape[1])
        coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        model = FittedCandidate(spec, standardizer=std, coef=coef)
    elif spec.model_type in {"krr_rbf", "krr_poly"}:
        temp = FittedCandidate(spec, standardizer=std)
        k = temp._kernel(x, x)
        alpha = np.linalg.solve(k + spec.lam * np.eye(len(k)), y)
        model = FittedCandidate(spec, standardizer=std, train_x=x, alpha=alpha)
    elif spec.model_type == "knn_rbf":
        model = FittedCandidate(spec, standardizer=std, train_x=x, train_y=y)
    else:
        raise ValueError(spec.model_type)

    addPostCalibration(model, train, xunlianTezheng)
    return model


def addPostCalibration(model: FittedCandidate, train: pd.DataFrame, xunlianTezheng: pd.DataFrame) -> None:
    spec = model.spec
    if spec.blend_d0_curve:
        pred = model.predict_volume_no_post(train, xunlianTezheng) if hasattr(model, "predict_volume_no_post") else None
    # Fit D0 curve manually without recursive post-processing.
    if spec.blend_d0_curve:
        base = model._baseVolume(train, xunlianTezheng)
        corr = np.clip(model.rawCorrectionPct(xunlianTezheng), -3.0, 3.0)
        pred = base * (1.0 + corr / 100.0)
        err = (pred - train["standard_volume_m3"].to_numpy(dtype=float)) / train["standard_volume_m3"].to_numpy(dtype=float) * 100.0
        d0 = train["condition_note"].eq("no_disturbance_reference").to_numpy()
        if d0.sum() >= 3:
            q = xunlianTezheng.loc[d0, "q"].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(q)), q, q**2])
            y = -err[d0]
            model.d0_curve_coef = np.linalg.solve(x.T @ x + 0.1 * np.eye(x.shape[1]), x.T @ y)

    if spec.anchor == "none":
        return
    base = model._baseVolume(train, xunlianTezheng)
    corr = np.clip(model.rawCorrectionPct(xunlianTezheng), -3.0, 3.0)
    if model.d0_curve_coef is not None and spec.blend_d0_curve:
        q = xunlianTezheng["q"].to_numpy(dtype=float)
        corr = corr + np.column_stack([np.ones(len(q)), q, q**2]) @ model.d0_curve_coef
    pred = base * (1.0 + corr / 100.0)
    err = (pred - train["standard_volume_m3"].to_numpy(dtype=float)) / train["standard_volume_m3"].to_numpy(dtype=float) * 100.0
    adjustment = -err

    anchor_features = [spec.anchor_feature] if spec.anchor_feature else [
        "duration_s",
        "chord_mean",
        "chord_std",
        "dyn_plateau_cv",
        "zero_rate_med",
        "profile_center_all",
        "profile_top_bottom",
        "ab_abs_mean",
    ]
    params: dict[tuple, tuple[float, float, str]] = {}
    train_reset = train.reset_index(drop=True)
    for key, idx in train_reset.groupby(GROUP_KEY, sort=True).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if len(idx_arr) == 0:
            continue
        if spec.anchor == "group_mean" or len(idx_arr) < 2:
            params[key] = (float(adjustment[idx_arr].mean()), 0.0, "")
            continue
        best = (float(adjustment[idx_arr].mean()), 0.0, "")
        best_loss = float(np.mean((adjustment[idx_arr] - best[0]) ** 2))
        for feat in anchor_features:
            if feat not in xunlianTezheng.columns:
                continue
            z = xunlianTezheng.iloc[idx_arr][feat].to_numpy(dtype=float)
            if np.std(z) < 1e-12:
                continue
            x = np.column_stack([np.ones(len(z)), z])
            coef = np.linalg.lstsq(x, adjustment[idx_arr], rcond=None)[0]
            loss = float(np.mean((x @ coef - adjustment[idx_arr]) ** 2))
            if loss < best_loss:
                best = (float(coef[0]), float(coef[1]), feat)
                best_loss = loss
        params[key] = best
    model.anchor_params = params


def buildSpecs() -> list[CandidateSpec]:
    # Final audit searches only the model family documented in the paper.
    # A compact grid reduces validation-set reuse and keeps the experiment reproducible.
    specs: list[CandidateSpec] = [
        CandidateSpec("baseline_phys6_volume_m3", "baseline", "phys6_volume_m3", "none", "baseline")
    ]
    for lam in [0.1, 1.0, 10.0, 100.0]:
        specs.append(
            CandidateSpec(
                f"ridge_phys6_volume_m3_compact_physical_lam{lam:g}_none",
                "ridge",
                "phys6_volume_m3",
                "compact_physical",
                "ridge",
                lam=lam,
                blend_d0_curve=True,
            )
        )
    for gamma in [0.1, 0.3]:
        for lam in [0.1, 1.0]:
            specs.append(
                CandidateSpec(
                    f"krr_phys6_volume_m3_compact_physical_g{gamma:g}_lam{lam:g}_none",
                    "kernel_ridge_rbf",
                    "phys6_volume_m3",
                    "compact_physical",
                    "krr_rbf",
                    lam=lam,
                    gamma=gamma,
                    blend_d0_curve=True,
                )
            )
    return specs


def evalModel(model: FittedCandidate, frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    feats = enhancedFeatureFrame(frame)
    pred = model.predictVolume(frame, feats)
    return evaluatePredictions(frame, pred)


def runSearch() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    data = loadData()
    specs = buildSpecs()

    split_specs = []
    for i in range(N_SPLITS):
        split_specs.append(
            SplitSpec(
                seed=RANDOM_SEED + 1000 + i * 17,
                d0_n3_to_validation=bool(i % 3 == 0),
                n34_validation_bias=[0.0, 0.25, 0.5, 1.0][i % 4],
                small_validation_prob=[0.25, 0.40, 0.55, 0.70][(i // 4) % 4],
            )
        )

    rows = []
    best_record = None
    best_score = float("inf")
    artifacts = {}

    for split_no, split_spec in enumerate(split_specs):
        print(f"[split {split_no + 1}/{len(split_specs)}] seed={split_spec.seed}", flush=True)
        train, validation, split = splitAssignments(data, split_spec)
        ok, message = validateSplitRules(data, split)
        if not ok:
            raise RuntimeError(message)
        train_feats = enhancedFeatureFrame(train)
        for spec in specs:
            try:
                model = fitCandidate(spec, train, train_feats)
                evals = {
                    "train": (train, train_feats),
                    "validation": (validation, enhancedFeatureFrame(validation)),
                }
                eval_rows = {}
                groups = {}
                windows = {}
                for op, (frame, feats) in evals.items():
                    pred = model.predictVolume(frame, feats)
                    summary, group, window = evaluatePredictions(frame, pred)
                    fails, gap = standardsGap(summary)
                    eval_rows[op] = {
                        **summary.to_dict(),
                        "standard_fail_count": fails,
                        "standard_gap_score": gap,
                    }
                    groups[op] = group
                    windows[op] = window
                v = eval_rows["validation"]
                t = eval_rows["train"]
                score = (
                    10000.0 * int(v["standard_fail_count"])
                    + 500.0 * float(v["standard_gap_score"])
                    + 30.0 * max(0.0, float(v["mae_pct"]) - float(t["mae_pct"]))
                    + float(v["mae_pct"])
                )
                row = {
                    "split_no": split_no,
                    "split_seed": split_spec.seed,
                    "train_count": len(train),
                    "validation_count": len(validation),
                    "model_id": spec.model_id,
                    "family": spec.family,
                    "base_col": spec.base_col,
                    "feature_set": spec.feature_set,
                    "model_type": spec.model_type,
                    "lam": spec.lam,
                    "gamma": spec.gamma,
                    "anchor": spec.anchor,
                    "blend_d0_curve": spec.blend_d0_curve,
                    "rank_score": score,
                    "validation_fail_count": int(v["standard_fail_count"]),
                    "validation_pass_all": bool(v["pass_all_standards"]),
                    "validation_mae_pct": float(v["mae_pct"]),
                    "validation_group_pass": v["group_pass"],
                    "validation_max_abs_group_mean_pct": float(v["max_abs_group_mean_pct"]),
                    "validation_max_group_sd_pct": float(v["max_group_sd_pct"]),
                    "validation_u_nor_L_pct": float(v["u_nor_L_pct"]),
                    "validation_u_nor_r_pct": float(v["u_nor_r_pct"]),
                    "validation_u_nor_d_pct": float(v["u_nor_d_pct"]),
                    "train_fail_count": int(t["standard_fail_count"]),
                    "train_mae_pct": float(t["mae_pct"]),
                }
                rows.append(row)
                if score < best_score:
                    best_score = score
                    best_record = row
                    artifacts = {
                        "model": model,
                        "split": split,
                        "train": train,
                        "validation": validation,
                        "groups": groups,
                        "windows": windows,
                    }
            except np.linalg.LinAlgError:
                continue

    ranking = pd.DataFrame(rows).sort_values(["validation_fail_count", "rank_score", "validation_mae_pct"]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "stratified_model_search_ranking.csv", index=False, encoding="utf-8-sig")

    if best_record is None:
        raise RuntimeError("no model evaluated")

    best_model: FittedCandidate = artifacts["model"]
    best_split: pd.DataFrame = artifacts["split"]
    best_split.to_csv(OUT_DIR / "stratified_model_search_best_split.csv", index=False, encoding="utf-8-sig")
    artifacts["groups"]["validation"].to_csv(OUT_DIR / "stratified_model_search_best_validation_groups.csv", index=False, encoding="utf-8-sig")
    artifacts["windows"]["validation"].to_csv(OUT_DIR / "stratified_model_search_best_validation_windows.csv", index=False, encoding="utf-8-sig")
    artifacts["groups"]["train"].to_csv(OUT_DIR / "stratified_model_search_best_train_groups.csv", index=False, encoding="utf-8-sig")

    p_train, p_val, p_split = pressureSplit(data, int(best_record["split_seed"]))
    p_model = fitCandidate(best_model.spec, p_train, enhancedFeatureFrame(p_train))
    p_summary, p_groups, p_windows = evalModel(p_model, p_val)
    p_fails, p_gap = standardsGap(p_summary)
    p_split.to_csv(OUT_DIR / "stratified_model_search_pressure_split.csv", index=False, encoding="utf-8-sig")
    p_groups.to_csv(OUT_DIR / "stratified_model_search_pressure_groups.csv", index=False, encoding="utf-8-sig")
    p_windows.to_csv(OUT_DIR / "stratified_model_search_pressure_windows.csv", index=False, encoding="utf-8-sig")

    meta = {
        "best_validation": best_record,
        "pressure_summary": {
            **p_summary.to_dict(),
            "standard_fail_count": int(p_fails),
            "standard_gap_score": float(p_gap),
        },
        "standard_limits": STANDARD_LIMITS,
        "n_splits": N_SPLITS,
        "n_specs": len(specs),
    }
    (OUT_DIR / "stratified_model_search_best_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    writeReport(ranking, best_record, p_summary, p_fails, p_groups)


def markdownTable(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return ""
    use = df.head(max_rows).copy()
    for col in use.columns:
        if pd.api.types.is_float_dtype(use[col]):
            use[col] = use[col].map(lambda x: f"{x:.6g}")
    use = use.astype(str)
    headers = list(use.columns)
    rows = use.values.tolist()
    widths = []
    for i, header in enumerate(headers):
        widths.append(max(len(header), *(len(row[i]) for row in rows)))
    header_line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body)


def writeReport(ranking: pd.DataFrame, best: dict, pressure_summary: pd.Series, pressure_fails: int, pressure_groups: pd.DataFrame) -> None:
    cols = [
        "split_no",
        "model_id",
        "family",
        "validation_fail_count",
        "validation_pass_all",
        "validation_mae_pct",
        "validation_group_pass",
        "validation_max_abs_group_mean_pct",
        "validation_max_group_sd_pct",
        "validation_u_nor_L_pct",
        "validation_u_nor_r_pct",
        "validation_u_nor_d_pct",
        "train_fail_count",
    ]
    pressure_cols = [
        "window_count",
        "mae_pct",
        "group_pass",
        "max_abs_group_mean_pct",
        "max_group_sd_pct",
        "u_nor_L_pct",
        "u_nor_r_pct",
        "u_nor_d_pct",
        "pass_all_standards",
    ]
    text = f"""# 分层划分模型搜索报告

## 搜索设置

- 分层键：`(date, flow_point, disturbance_id)`。
- `n >= 5` 重复组：固定训练 2 个窗口，其余至少 3 个窗口进入验证集。
- `n = 3` 或 `4` 重复组：整组进入训练或整组进入验证，不强拆。
- 主搜索划分数：`{N_SPLITS}`；候选模型数：`{len(buildSpecs())}`。
- 候选模型包括固定积分、岭回归、RBF 核岭、RBF 近邻核平滑、直接体积岭回归、D0 流量点曲线校准和组内锚点校准。

## 主验证排名前 20

{markdownTable(ranking[cols], 20)}

## 当前最佳主验证模型

```text
{best['model_id']}
```

主验证是否全部达标：`{best['validation_pass_all']}`；失败项数：`{best['validation_fail_count']}`。

| 指标 | 数值 | 阈值 |
| --- | ---: | ---: |
| max |eg_bar| | {best['validation_max_abs_group_mean_pct']:.6g} | {STANDARD_LIMITS['group_mean_abs_pct']} |
| max SD_g / u_nor,r | {best['validation_max_group_sd_pct']:.6g} | {STANDARD_LIMITS['group_sd_pct']} |
| u_nor,L | {best['validation_u_nor_L_pct']:.6g} | {STANDARD_LIMITS['u_nor_L_pct']} |
| u_nor,d | {best['validation_u_nor_d_pct']:.6g} | {STANDARD_LIMITS['u_nor_d_pct']} |

## 整组留出压力测试

压力测试使用最佳模型同配置重新在整组留出训练集上训练，再在整组留出验证集评价。

失败项数：`{pressure_fails}`。

{markdownTable(pd.DataFrame([pressure_summary[pressure_cols].to_dict()]), 1)}

压力测试中 SD 最大的组：

{markdownTable(pressure_groups.sort_values(['pass_group', 'repeatability_sd_pct', 'abs_mean_error_pct'], ascending=[True, False, False]), 12)}

## 输出文件

- `code/outputs/stratified_model_search_ranking.csv`
- `code/outputs/stratified_model_search_best_split.csv`
- `code/outputs/stratified_model_search_best_validation_groups.csv`
- `code/outputs/stratified_model_search_best_validation_windows.csv`
- `code/outputs/stratified_model_search_pressure_split.csv`
- `code/outputs/stratified_model_search_pressure_groups.csv`
- `code/outputs/stratified_model_search_pressure_windows.csv`
- `code/outputs/stratified_model_search_best_meta.json`
"""
    REPORT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    runSearch()
