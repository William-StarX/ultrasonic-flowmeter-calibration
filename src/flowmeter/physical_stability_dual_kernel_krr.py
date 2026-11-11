from __future__ import annotations

# 论文附录复现代码。
# 问题四增强模型：融合物理核与逐秒稳定性核。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

"""
标签盲分层验证 + 附件6逐秒稳定性特征 + 物理--稳定性双核 RBF-KRR 残差补偿。

本脚本根据附件6的真实实验结果确定双核结构：
1. 保留物理核与稳定性核相加的双核结构，不再引入新的模型层级；
2. 由于冠军位于 raw_gamma 最小值与 raw_kernel_weight 最大值边界，
   将原始核搜索扩展到更平滑、更高权重区域；
3. 将附件6特征拆为“波动”“漂移同步”“Hampel修正”三组，执行消融实验；
4. 保留全部训练窗口，物理核和原始稳定性核分别标准化；
5. 只保留已被验证更稳健的线性 D0 后校准，同时加入少量无 D0 对照；
6. 增加 q90 重复性与 q75 综合超限分数，冠军选择更关注跨划分尾部风险；
7. 最终模型仍依据 40 次标签盲划分的聚合表现选择，并单独报告压力测试和全量重拟合。

依赖项目中已有文件：
    final_model_ranking_experiment.py
    generalization_improvement_experiment.py
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from model_evolution_ridge_ann import (
    DATA_DIR,
    GROUP_KEY,
    OUT_DIR,
    RANDOM_SEED,
    STANDARD_LIMITS,
    evaluatePredictions,
    featureColumns,
    loadData,
    standardsGap,
)
from physical_feature_library import enhancedFeatureFrame


REPORT = Path(__file__).resolve().parent / "verification_reports" / "STRATIFIED_MODEL_SEARCH_REPORT.md"
TARGET_VALIDATION_FRACTION = 0.30
N_SPLITS = 40
SCRIPT_VERSION = "physical-stability-dual-kernel-krr-20260712"

# 训练/验证划分仍按原题试验单元构造；官方重复性仍由 evaluate_predictions
# 按 (date, flow_point) 计算。
SPLIT_GROUP_KEY = list(GROUP_KEY)
OFFICIAL_GROUP_KEY = ["date", "flow_point"]

BASE_COLUMNS = [
    "phys6_volume_m3",
    "lagrange_volume_m3",
    "owics_volume_m3",
    "equal_weight_volume_m3",
]


RAW_FEATURE_COLUMNS = [
    "raw_diff_rel_mad_med",
    "raw_diff_rel_mad_max",
    "raw_diff_rel_iqr_med",
    "raw_diff_drift_med",
    "raw_diff_slope_abs_med",
    "raw_diff_sync_corr_med",
    "raw_hampel_replace_rate_mean",
    "raw_hampel_replace_rate_max",
    "raw_mean_us_rel_mad_med",
    "raw_mean_us_drift_med",
    "raw_temperature_mad",
    "raw_chord_rel_mad_med",
    "raw_chord_rel_mad_max",
    "raw_ab_mismatch_med",
    "raw_profile_center_edges",
    "raw_profile_edge_inner",
    "raw_profile_top_bottom",
    "raw_profile_range",
    "raw_phys6_hampel_ratio",
    "raw_phys6_hampel_delta_pct",
]

# 只把少量稳定、可解释的原始序列特征送入 KRR，避免小样本高维化。
RAW_MODEL_FEATURE_COLUMNS = [
    "raw_diff_rel_mad_med",
    "raw_diff_rel_mad_max",
    "raw_diff_drift_med",
    "raw_diff_sync_corr_med",
    "raw_hampel_replace_rate_max",
    "raw_mean_us_rel_mad_med",
    "raw_chord_rel_mad_med",
    "raw_chord_rel_mad_max",
    "raw_ab_mismatch_med",
    "raw_phys6_hampel_delta_pct",
]

# 附件6特征组消融。三组互不重叠，便于判断真正贡献来自哪里。
RAW_VARIABILITY_FEATURE_COLUMNS = [
    "raw_diff_rel_mad_med",
    "raw_diff_rel_mad_max",
    "raw_mean_us_rel_mad_med",
    "raw_chord_rel_mad_med",
    "raw_chord_rel_mad_max",
]

RAW_DRIFT_SYNC_FEATURE_COLUMNS = [
    "raw_diff_drift_med",
    "raw_diff_sync_corr_med",
    "raw_ab_mismatch_med",
]

RAW_HAMPEL_FEATURE_COLUMNS = [
    "raw_hampel_replace_rate_max",
    "raw_phys6_hampel_delta_pct",
]

RAW_NO_HAMPEL_FEATURE_COLUMNS = (
    RAW_VARIABILITY_FEATURE_COLUMNS + RAW_DRIFT_SYNC_FEATURE_COLUMNS
)


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

    # baseline / ridge / krr_rbf / multikernel_krr / two_stage_krr
    model_type: str = "krr_rbf"
    feature_set: str = "compact_physical"
    lam: float = 0.1
    gamma: float = 0.1

    # 多核KRR：紧凑物理核 + 附件6稳定性核。
    raw_feature_set: str = "raw_stability_only"
    raw_gamma: float = 0.03
    raw_kernel_weight: float = 0.25

    # 训练标签与样本权重控制。仅使用训练标签，不进入运行时输入。
    target_group_shrinkage: float = 0.0
    sample_group_balance: bool = False

    # 两阶段模型：系统组均值KRR + 强收缩窗口动态KRR（保留兼容，不再作为v5主候选）
    system_feature_set: str = "system_physical"
    system_gamma: float = 0.03
    system_lam: float = 0.3
    system_group_shrinkage: float = 1.0
    dynamic_feature_set: str = "dynamic_raw"
    dynamic_gamma: float = 0.01
    dynamic_lam: float = 1.0
    dynamic_rho: float = 0.25
    dynamic_group_balance: bool = True
    dynamic_center_target: bool = True

    # phys6 / raw_hampel_phys6 / lagrange / owics / equal / median4 / fixed_blend / fitted_blend
    base_mode: str = "phys6"
    fixed_weights: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # 保留兼容字段；v4候选默认不再直接用Hampel比值修改Phys6。
    raw_base_strength: float = 0.0
    raw_ratio_clip_pct: float = 0.5

    # 仅 fitted_blend 使用。
    base_prior: tuple[float, float, float, float] = (0.70, 0.20, 0.10, 0.0)
    base_reg: float = 0.5
    base_repeatability_eta: float = 0.0
    base_group_balance: bool = True

    blend_d0_curve: bool = True
    d0_curve_degree: int = 2
    d0_curve_lam: float = 0.1
    correction_clip_pct: float = 3.0

@dataclass
class StandardizerLite:
    columns: list[str]
    median: pd.Series
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "StandardizerLite":
        if not columns:
            raise ValueError("empty feature column list")
        clean = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
        median = clean.median(numeric_only=True).fillna(0.0)
        clean = clean.fillna(median)
        arr = clean.to_numpy(dtype=float)
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(columns=columns, median=median, mean=mean, scale=scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        clean = frame[self.columns].replace([np.inf, -np.inf], np.nan).copy()
        clean = clean.fillna(self.median)
        return (clean.to_numpy(dtype=float) - self.mean) / self.scale


class FittedCandidate:
    def __init__(
        self,
        spec: CandidateSpec,
        base_weights: np.ndarray | None = None,
        standardizer: StandardizerLite | None = None,
        coef: np.ndarray | None = None,
        train_x: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
        raw_standardizer: StandardizerLite | None = None,
        raw_train_x: np.ndarray | None = None,
        d0_curve_coef: np.ndarray | None = None,
        system_standardizer: StandardizerLite | None = None,
        system_train_x: np.ndarray | None = None,
        system_alpha: np.ndarray | None = None,
        dynamic_standardizer: StandardizerLite | None = None,
        dynamic_train_x: np.ndarray | None = None,
        dynamic_alpha: np.ndarray | None = None,
    ) -> None:
        self.spec = spec
        self.base_weights = None if base_weights is None else np.asarray(base_weights, dtype=float)

        # 单阶段模型参数
        self.standardizer = standardizer
        self.coef = coef
        self.train_x = train_x
        self.alpha = alpha

        # 多核KRR的原始稳定性核参数。
        self.raw_standardizer = raw_standardizer
        self.raw_train_x = raw_train_x

        # 两阶段模型参数
        self.system_standardizer = system_standardizer
        self.system_train_x = system_train_x
        self.system_alpha = system_alpha
        self.dynamic_standardizer = dynamic_standardizer
        self.dynamic_train_x = dynamic_train_x
        self.dynamic_alpha = dynamic_alpha

        self.d0_curve_coef = d0_curve_coef

    @staticmethod
    def _rbfKernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
        x2 = np.sum(x * x, axis=1)[:, None]
        y2 = np.sum(y * y, axis=1)[None, :]
        dist2 = np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)
        return np.exp(-float(gamma) * dist2)

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._rbfKernel(x, y, self.spec.gamma)

    def baseVolume(self, data: pd.DataFrame) -> np.ndarray:
        mode = self.spec.base_mode

        if mode == "raw_hampel_phys6":
            return rawHampelPhys6Volume(
                data,
                strength=self.spec.raw_base_strength,
                clip_pct=self.spec.raw_ratio_clip_pct,
            )

        matrix = baseMatrix(data)
        if mode == "median4":
            return np.median(matrix, axis=1)

        if self.base_weights is None:
            raise ValueError(f"missing base weights for mode={mode}")
        return matrix @ self.base_weights

    def correctionComponents(
        self,
        features: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回 (system_or_single, dynamic_raw, total_raw)。

        对单阶段KRR/Ridge，第一项就是单阶段修正，动态项为0。
        对两阶段KRR：
            total = system + dynamic_rho * dynamic_raw
        """
        n = len(features)
        if self.spec.model_type == "baseline":
            zeros = np.zeros(n, dtype=float)
            return zeros, zeros, zeros

        if self.spec.model_type == "multikernel_krr":
            if (
                self.standardizer is None
                or self.train_x is None
                or self.raw_standardizer is None
                or self.raw_train_x is None
                or self.alpha is None
            ):
                raise ValueError("missing multi-kernel KRR parameters")
            physical_x = self.standardizer.transform(features)
            raw_x = self.raw_standardizer.transform(features)
            beta = float(np.clip(self.spec.raw_kernel_weight, 0.0, 1.0))
            kernel_physical = self._rbfKernel(
                physical_x,
                self.train_x,
                self.spec.gamma,
            )
            kernel_raw = self._rbfKernel(
                raw_x,
                self.raw_train_x,
                self.spec.raw_gamma,
            )
            single = ((1.0 - beta) * kernel_physical + beta * kernel_raw) @ self.alpha
            zeros = np.zeros(n, dtype=float)
            return single, zeros, single

        if self.spec.model_type == "two_stage_krr":
            if (
                self.system_standardizer is None
                or self.system_train_x is None
                or self.system_alpha is None
            ):
                raise ValueError("missing two-stage system KRR parameters")
            system_x = self.system_standardizer.transform(features)
            system = (
                self._rbfKernel(system_x, self.system_train_x, self.spec.system_gamma)
                @ self.system_alpha
            )

            dynamic = np.zeros(n, dtype=float)
            if (
                self.spec.dynamic_rho > 0.0
                and self.dynamic_standardizer is not None
                and self.dynamic_train_x is not None
                and self.dynamic_alpha is not None
            ):
                dynamic_x = self.dynamic_standardizer.transform(features)
                dynamic = (
                    self._rbfKernel(dynamic_x, self.dynamic_train_x, self.spec.dynamic_gamma)
                    @ self.dynamic_alpha
                )
            total = system + float(self.spec.dynamic_rho) * dynamic
            return system, dynamic, total

        if self.standardizer is None:
            raise ValueError("missing standardizer")
        x = self.standardizer.transform(features)

        if self.spec.model_type == "ridge":
            if self.coef is None:
                raise ValueError("missing ridge coefficients")
            single = x @ self.coef
        elif self.spec.model_type == "krr_rbf":
            if self.train_x is None or self.alpha is None:
                raise ValueError("missing KRR parameters")
            single = self._kernel(x, self.train_x) @ self.alpha
        else:
            raise ValueError(f"unsupported model_type: {self.spec.model_type}")

        zeros = np.zeros(n, dtype=float)
        return single, zeros, single

    def rawCorrectionPct(self, features: pd.DataFrame) -> np.ndarray:
        _, _, total = self.correctionComponents(features)
        return total

    def d0PostCurve(self, features: pd.DataFrame) -> np.ndarray:
        if self.d0_curve_coef is None or not self.spec.blend_d0_curve:
            return np.zeros(len(features), dtype=float)
        q = features["q"].to_numpy(dtype=float)
        degree = max(int(self.spec.d0_curve_degree), 0)
        design = np.column_stack([q**power for power in range(degree + 1)])
        return design @ self.d0_curve_coef

    def predictWithDiagnostics(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        base = self.baseVolume(data)
        system_or_single, dynamic_raw, raw_total = self.correctionComponents(features)
        raw_total = np.clip(
            raw_total,
            -self.spec.correction_clip_pct,
            self.spec.correction_clip_pct,
        )
        d0_post = self.d0PostCurve(features)
        correction = np.clip(
            raw_total + d0_post,
            -self.spec.correction_clip_pct,
            self.spec.correction_clip_pct,
        )
        yuceJieguo = base * (1.0 + correction / 100.0)

        matrix = baseMatrix(data)
        diagnostics = pd.DataFrame(
            {
                "window_id": data["window_id"].to_numpy(),
                "phys6_volume_m3": matrix[:, 0],
                "lagrange_volume_m3": matrix[:, 1],
                "owics_volume_m3": matrix[:, 2],
                "equal_weight_volume_m3": matrix[:, 3],
                "selected_base_volume_m3": base,
                "system_or_single_correction_pct": system_or_single,
                "dynamic_raw_correction_pct": dynamic_raw,
                "dynamic_scaled_correction_pct": float(self.spec.dynamic_rho) * dynamic_raw,
                "raw_krr_correction_pct": raw_total,
                "d0_post_correction_pct": d0_post,
                "total_correction_pct": correction,
                "model_volume_m3": yuceJieguo,
            }
        )
        return yuceJieguo, diagnostics

    def predictVolume(self, data: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
        yuceJieguo, _ = self.predictWithDiagnostics(data, features)
        return yuceJieguo

# ---------------------------------------------------------------------------
# 附件6逐秒序列稳健特征
# ---------------------------------------------------------------------------


def hampelFilter(values: np.ndarray, window: int = 9, n_sigma: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Causal information is not used; this is an offline symmetric robust window statistic."""
    series = pd.Series(np.asarray(values, dtype=float))
    median = series.rolling(window, center=True, min_periods=3).median()
    deviation = (series - median).abs()
    mad = deviation.rolling(window, center=True, min_periods=3).median()
    threshold = n_sigma * 1.4826 * mad
    valid_threshold = threshold.where(threshold > 1e-12, np.inf)
    mask = (deviation > valid_threshold).fillna(False)
    filtered = series.copy()
    filtered.loc[mask] = median.loc[mask]
    return filtered.to_numpy(dtype=float), mask.to_numpy(dtype=bool)


def safeRelative(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return num / np.maximum(np.abs(den), 1e-12)


def normalisedSlope(time: np.ndarray, values: np.ndarray, location: float) -> float:
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return 0.0
    t = t[finite]
    y = y[finite]
    t = t - float(t.mean())
    denom = float(t @ t)
    if denom <= 1e-12:
        return 0.0
    slope = float(t @ (y - float(y.mean())) / denom)
    return slope / max(abs(float(location)), 1e-12)


def buildRawWindowFeatures(raw: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """
    从附件6提取完全标签无关的逐窗稳健特征。

    Hampel滤波只替换局部尖峰；整窗均值仍保留启停过程，因此不会把平台流速
    乘以整窗时长。若附件7映射可用，则用滤波前后五声道Phys6积分信号之比
    构造 raw_phys6_hampel_ratio。
    """
    diff_columns = [f"diff_ns_{i}" for i in range(10) if f"diff_ns_{i}" in raw.columns]
    mean_columns = [f"mean_us_{i}" for i in range(10) if f"mean_us_{i}" in raw.columns]
    if len(diff_columns) != 10:
        raise ValueError(f"attachment6 needs diff_ns_0..9, got {len(diff_columns)} columns")

    mapping = pd.DataFrame()
    if "record_type" in geo.columns:
        mapping = geo[geo["record_type"].eq("chord_mapping")].copy()
    required_mapping = {
        "raw_diff_col_a",
        "raw_diff_col_b",
        "raw_diff_gain_a_mps_per_ns",
        "raw_diff_gain_b_mps_per_ns",
        "raw_diff_intercept_mps",
        "weight_phys6",
    }
    mapping_ok = (not mapping.empty) and required_mapping.issubset(mapping.columns)
    if mapping_ok and "chord_id" in mapping.columns:
        mapping = mapping.sort_values("chord_id").reset_index(drop=True)

    rows: list[dict[str, float | str]] = []
    for window_id, group in raw.groupby("window_id", sort=False):
        group = group.sort_values("rel_time_s").reset_index(drop=True)
        time = group["rel_time_s"].to_numpy(dtype=float)
        original = group[diff_columns].to_numpy(dtype=float)
        n = len(group)

        filtered = np.empty_like(original)
        replaced = np.zeros_like(original, dtype=bool)
        for j in range(original.shape[1]):
            filtered[:, j], replaced[:, j] = hampelFilter(original[:, j])

        lo = int(math.floor(0.15 * n))
        hi = int(math.ceil(0.85 * n))
        if hi - lo < 20:
            lo, hi = 0, n
        centre = filtered[lo:hi]
        centre_time = time[lo:hi]

        centre_median = np.nanmedian(centre, axis=0)
        centre_mad = np.nanmedian(np.abs(centre - centre_median), axis=0)
        q25 = np.nanpercentile(centre, 25.0, axis=0)
        q75 = np.nanpercentile(centre, 75.0, axis=0)
        rel_mad = safeRelative(centre_mad, centre_median)
        rel_iqr = safeRelative(q75 - q25, centre_median)

        edge_n = max(5, int(round(0.15 * n)))
        first_median = np.nanmedian(filtered[:edge_n], axis=0)
        last_median = np.nanmedian(filtered[-edge_n:], axis=0)
        drift = safeRelative(np.abs(last_median - first_median), centre_median)
        slopes = np.asarray(
            [
                abs(normalisedSlope(centre_time, centre[:, j], centre_median[j]))
                for j in range(centre.shape[1])
            ],
            dtype=float,
        )

        centred = centre - np.nanmedian(centre, axis=0)
        valid_columns = np.nanstd(centred, axis=0) > 1e-12
        sync_corr = 0.0
        if int(valid_columns.sum()) >= 2 and len(centre) >= 3:
            corr = np.corrcoef(centred[:, valid_columns], rowvar=False)
            upper = corr[np.triu_indices_from(corr, k=1)]
            if np.isfinite(upper).any():
                sync_corr = float(np.nanmedian(upper))

        mean_rel_mad = 0.0
        mean_drift = 0.0
        if mean_columns:
            mean_values = group[mean_columns].to_numpy(dtype=float)
            mean_centre = mean_values[lo:hi]
            mean_location = np.nanmedian(mean_centre, axis=0)
            mean_mad = np.nanmedian(np.abs(mean_centre - mean_location), axis=0)
            mean_rel_mad = float(np.nanmedian(safeRelative(mean_mad, mean_location)))
            mean_first = np.nanmedian(mean_values[:edge_n], axis=0)
            mean_last = np.nanmedian(mean_values[-edge_n:], axis=0)
            mean_drift = float(np.nanmedian(safeRelative(np.abs(mean_last - mean_first), mean_location)))

        temperature_mad = 0.0
        if "water_temp_c" in group.columns:
            temperature = group["water_temp_c"].to_numpy(dtype=float)
            temp_med = float(np.nanmedian(temperature))
            temperature_mad = float(np.nanmedian(np.abs(temperature - temp_med)))

        feature: dict[str, float | str] = {
            "window_id": str(window_id),
            "raw_diff_rel_mad_med": float(np.nanmedian(rel_mad)),
            "raw_diff_rel_mad_max": float(np.nanmax(rel_mad)),
            "raw_diff_rel_iqr_med": float(np.nanmedian(rel_iqr)),
            "raw_diff_drift_med": float(np.nanmedian(drift)),
            "raw_diff_slope_abs_med": float(np.nanmedian(slopes)),
            "raw_diff_sync_corr_med": sync_corr,
            "raw_hampel_replace_rate_mean": float(replaced.mean()),
            "raw_hampel_replace_rate_max": float(replaced.mean(axis=0).max()),
            "raw_mean_us_rel_mad_med": mean_rel_mad,
            "raw_mean_us_drift_med": mean_drift,
            "raw_temperature_mad": temperature_mad,
            "raw_chord_rel_mad_med": 0.0,
            "raw_chord_rel_mad_max": 0.0,
            "raw_ab_mismatch_med": 0.0,
            "raw_profile_center_edges": 0.0,
            "raw_profile_edge_inner": 0.0,
            "raw_profile_top_bottom": 0.0,
            "raw_profile_range": 0.0,
            "raw_phys6_hampel_ratio": 1.0,
            "raw_phys6_hampel_delta_pct": 0.0,
        }

        if mapping_ok:
            full_chords: list[float] = []
            clean_chords: list[float] = []
            chord_rel_mad: list[float] = []
            ab_mismatch: list[float] = []
            weights: list[float] = []

            diff_index = {column: i for i, column in enumerate(diff_columns)}
            for _, map_row in mapping.iterrows():
                a_column = str(map_row["raw_diff_col_a"])
                b_column = str(map_row["raw_diff_col_b"])
                if a_column not in diff_index or b_column not in diff_index:
                    continue
                ia = diff_index[a_column]
                ib = diff_index[b_column]
                gain_a = float(map_row["raw_diff_gain_a_mps_per_ns"])
                gain_b = float(map_row["raw_diff_gain_b_mps_per_ns"])
                intercept = float(map_row["raw_diff_intercept_mps"])

                full_series = gain_a * original[:, ia] + gain_b * original[:, ib] + intercept
                clean_series = gain_a * filtered[:, ia] + gain_b * filtered[:, ib] + intercept
                clean_centre = clean_series[lo:hi]
                chord_location = float(np.nanmedian(clean_centre))
                chord_mad = float(np.nanmedian(np.abs(clean_centre - chord_location)))

                full_chords.append(float(np.nanmean(full_series)))
                clean_chords.append(float(np.nanmean(clean_series)))
                chord_rel_mad.append(float(chord_mad / max(abs(chord_location), 1e-12)))

                a_contribution = gain_a * filtered[lo:hi, ia]
                b_contribution = gain_b * filtered[lo:hi, ib]
                denominator = np.maximum(np.abs(clean_centre), 1e-12)
                ab_mismatch.append(float(np.nanmedian(np.abs(a_contribution - b_contribution) / denominator)))
                weights.append(float(map_row["weight_phys6"]))

            if len(clean_chords) >= 3:
                full_array = np.asarray(full_chords, dtype=float)
                clean_array = np.asarray(clean_chords, dtype=float)
                weight_array = np.asarray(weights, dtype=float)
                full_signal = float(weight_array @ full_array)
                clean_signal = float(weight_array @ clean_array)
                if abs(full_signal) > 1e-12 and np.isfinite(clean_signal):
                    ratio = clean_signal / full_signal
                    if np.isfinite(ratio):
                        feature["raw_phys6_hampel_ratio"] = float(ratio)
                        feature["raw_phys6_hampel_delta_pct"] = float((ratio - 1.0) * 100.0)

                chord_rel = np.asarray(chord_rel_mad, dtype=float)
                feature["raw_chord_rel_mad_med"] = float(np.nanmedian(chord_rel))
                feature["raw_chord_rel_mad_max"] = float(np.nanmax(chord_rel))
                feature["raw_ab_mismatch_med"] = float(np.nanmedian(np.asarray(ab_mismatch, dtype=float)))

                # 映射通常按 chord_id 0..4 排序。只有五声道齐全时才计算剖面量。
                if len(clean_array) >= 5:
                    c0, c1, c2, c3, c4 = clean_array[:5]
                    scale = max(abs(float(np.mean(clean_array[:5]))), 1e-12)
                    feature["raw_profile_center_edges"] = float((c2 - 0.5 * (c0 + c4)) / scale)
                    feature["raw_profile_edge_inner"] = float((0.5 * (c0 + c4) - 0.5 * (c1 + c3)) / scale)
                    feature["raw_profile_top_bottom"] = float((c0 - c4) / scale)
                    feature["raw_profile_range"] = float((np.max(clean_array[:5]) - np.min(clean_array[:5])) / scale)

        rows.append(feature)

    result = pd.DataFrame(rows)
    for column in RAW_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(result[column].median()) if result[column].notna().any() else 0.0
        result[column] = result[column].fillna(median)
    return result[["window_id", *RAW_FEATURE_COLUMNS]]


def loadDataWithRawFeatures(data_dir: Path = DATA_DIR, output_dir: Path = OUT_DIR) -> pd.DataFrame:
    data = loadData(data_dir / "attachment1_window_data.csv")
    raw_path = data_dir / "attachment6_window_raw_samples.csv"
    geo_path = data_dir / "attachment7_meter_geometry.csv"
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    geo = pd.read_csv(geo_path, encoding="utf-8-sig")
    raw_features = buildRawWindowFeatures(raw, geo)
    merged = data.merge(raw_features, on="window_id", how="left", validate="one_to_one")
    if merged[RAW_FEATURE_COLUMNS].isna().any().any():
        missing_windows = int(merged[RAW_FEATURE_COLUMNS].isna().any(axis=1).sum())
        raise ValueError(f"raw features missing for {missing_windows} windows")
    raw_features.to_csv(
        output_dir / "raw_window_robust_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return merged


def modelFeatureFrame(frame: pd.DataFrame) -> pd.DataFrame:
    # 先生成项目原有的窗口级增强特征，再附加附件6提取的稳健特征。
    # 这里必须调用 enhanced_feature_frame()，不能递归调用本函数。
    features = enhancedFeatureFrame(frame).copy()
    for column in RAW_FEATURE_COLUMNS:
        if column in frame.columns:
            features[column] = frame[column].to_numpy()
    return features


def rawHampelPhys6Volume(
    data: pd.DataFrame,
    strength: float,
    clip_pct: float,
) -> np.ndarray:
    phys6 = data["phys6_volume_m3"].to_numpy(dtype=float)
    if "raw_phys6_hampel_ratio" not in data.columns:
        return phys6.copy()
    ratio = pd.to_numeric(data["raw_phys6_hampel_ratio"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    half_width = max(float(clip_pct), 0.0) / 100.0
    ratio = np.clip(ratio, 1.0 - half_width, 1.0 + half_width)
    adjusted_ratio = 1.0 + float(strength) * (ratio - 1.0)
    return phys6 * adjusted_ratio


# ---------------------------------------------------------------------------
# 物理基线融合
# ---------------------------------------------------------------------------


def baseMatrix(data: pd.DataFrame) -> np.ndarray:
    missing = [column for column in BASE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"missing base volume columns: {missing}")
    frame = data[BASE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    for column in BASE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        frame[column] = values.fillna(median)
    return frame.to_numpy(dtype=float)


def normaliseWeights(values: Iterable[float]) -> np.ndarray:
    weights = np.asarray(list(values), dtype=float)
    if weights.shape != (len(BASE_COLUMNS),):
        raise ValueError(f"weights must have length {len(BASE_COLUMNS)}")
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        weights = np.zeros(len(BASE_COLUMNS), dtype=float)
        weights[0] = 1.0
        return weights
    return weights / total


def projectSimplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w)=1}."""
    v = np.asarray(vector, dtype=float)
    if v.ndim != 1:
        raise ValueError("simplex projection expects a vector")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    indices = np.arange(1, len(v) + 1)
    valid = u - cssv / indices > 0
    if not np.any(valid):
        result = np.zeros_like(v)
        result[int(np.argmax(v))] = 1.0
        return result
    rho = int(indices[valid][-1])
    theta = float(cssv[valid][-1] / rho)
    return np.maximum(v - theta, 0.0)


def officialGroupBalanceWeights(data: pd.DataFrame, enabled: bool) -> np.ndarray:
    n = len(data)
    if not enabled or n == 0:
        return np.ones(n, dtype=float)
    keys = [column for column in OFFICIAL_GROUP_KEY if column in data.columns]
    if not keys:
        return np.ones(n, dtype=float)
    reset = data[keys].reset_index(drop=True)
    sizes = reset.groupby(keys, sort=False)[keys[0]].transform("size").to_numpy(dtype=float)
    weights = 1.0 / np.maximum(sizes, 1.0)
    weights /= max(float(weights.mean()), 1e-12)
    return weights


def repeatabilityLaplacian(data: pd.DataFrame) -> np.ndarray:
    """Block-centering matrix for official (date, flow_point) repeatability groups."""
    n = len(data)
    lap = np.zeros((n, n), dtype=float)
    keys = [column for column in OFFICIAL_GROUP_KEY if column in data.columns]
    if not keys:
        return lap

    reset = data.reset_index(drop=True)
    for _, index in reset.groupby(keys, sort=False).groups.items():
        ids = np.asarray(list(index), dtype=int)
        m = len(ids)
        if m < 2:
            continue
        center = np.eye(m, dtype=float) - np.ones((m, m), dtype=float) / m
        center /= max(m - 1, 1)
        lap[np.ix_(ids, ids)] += center

    trace = float(np.trace(lap))
    if trace > 1e-12:
        lap *= n / trace
    return lap


def fitConvexBaseWeights(
    spec: CandidateSpec,
    train: pd.DataFrame,
) -> np.ndarray:
    """
    在训练集内拟合非负、和为1的全局物理积分权重。

    目标在百分比误差空间中构造：
        window fit + eta * within-day-flow spread + reg * distance-to-Phys6-prior

    日期和标准体积只参与离线参数估计，不进入运行时公式。
    """
    matrix = baseMatrix(train)
    standard = np.maximum(train["standard_volume_m3"].to_numpy(dtype=float), 1e-12)
    error_matrix_pct = (matrix / standard[:, None] - 1.0) * 100.0

    sample_weights = officialGroupBalanceWeights(train, spec.base_group_balance)
    metric = np.diag(sample_weights)
    if spec.base_repeatability_eta > 0.0:
        metric = metric + spec.base_repeatability_eta * repeatabilityLaplacian(train)

    prior = normaliseWeights(spec.base_prior)
    reg = max(float(spec.base_reg), 0.0)

    hessian = error_matrix_pct.T @ metric @ error_matrix_pct
    if reg > 0.0:
        hessian = hessian + reg * np.eye(len(BASE_COLUMNS), dtype=float)
    linear = reg * prior

    # Projected-gradient solution of a small convex quadratic on the simplex.
    weights = prior.copy()
    largest = float(np.linalg.eigvalsh(0.5 * (hessian + hessian.T)).max())
    step = 1.0 / max(2.0 * largest, 1e-8)
    for _ in range(5000):
        gradient = 2.0 * (hessian @ weights - linear)
        updated = projectSimplex(weights - step * gradient)
        if float(np.linalg.norm(updated - weights)) < 1e-12:
            weights = updated
            break
        weights = updated

    return normaliseWeights(weights)


def resolveBaseWeights(spec: CandidateSpec, train: pd.DataFrame) -> np.ndarray | None:
    mode = spec.base_mode
    if mode == "median4":
        return None
    if mode in {"phys6", "raw_hampel_phys6"}:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if mode == "lagrange":
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    if mode == "owics":
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    if mode == "equal":
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if mode == "fixed_blend":
        return normaliseWeights(spec.fixed_weights)
    if mode == "fitted_blend":
        return fitConvexBaseWeights(spec, train)
    raise ValueError(f"unknown base_mode: {mode}")


# ---------------------------------------------------------------------------
# 特征、KRR与后校准
# ---------------------------------------------------------------------------


def columnsForFeatureSet(frame: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "runtime_full":
        return featureColumns(frame, "runtime_full")

    compact = [
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

    # 第一阶段只学习跨工况系统偏差，避免把高频动态特征直接混入。
    system_physical = [
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
        "zero_rate_med",
        "lagrange_volume_m3_minus_phys6_frac",
        "owics_volume_m3_minus_phys6_frac",
    ]

    # 第二阶段只学习组内可预测的窗口变化，并用rho强收缩。
    dynamic_raw = [
        "q",
        "dyn_plateau_cv",
        "profile_top_bottom",
        "profile_center_all",
        "profile_ab_abs",
        "profile_swirl",
        "ab_abs_mean",
        "zero_rate_med",
        *RAW_MODEL_FEATURE_COLUMNS,
    ]

    if feature_set == "compact_physical":
        columns = compact
    elif feature_set == "compact_plus_raw":
        columns = compact + RAW_MODEL_FEATURE_COLUMNS
    elif feature_set == "system_physical":
        columns = system_physical
    elif feature_set == "dynamic_raw":
        columns = dynamic_raw
    elif feature_set == "raw_stability_only":
        columns = RAW_MODEL_FEATURE_COLUMNS
    elif feature_set == "raw_variability_only":
        columns = RAW_VARIABILITY_FEATURE_COLUMNS
    elif feature_set == "raw_drift_sync_only":
        columns = RAW_DRIFT_SYNC_FEATURE_COLUMNS
    elif feature_set == "raw_no_hampel":
        columns = RAW_NO_HAMPEL_FEATURE_COLUMNS
    elif feature_set == "raw_hampel_only":
        columns = RAW_HAMPEL_FEATURE_COLUMNS
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")

    # 去重并保持顺序。
    return list(dict.fromkeys(column for column in columns if column in frame.columns))


def residualTargetPct(
    model: FittedCandidate,
    data: pd.DataFrame,
) -> np.ndarray:
    base = np.maximum(model.baseVolume(data), 1e-12)
    standard = data["standard_volume_m3"].to_numpy(dtype=float)
    return (standard / base - 1.0) * 100.0


def stableLinearSolve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]


def rbfKernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    return FittedCandidate._rbfKernel(x, y, gamma)


def fitWeightedKrr(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    lam: float,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对角样本权重的对称KRR实现。

    令W=diag(w)，求解：
        (sqrt(W) K sqrt(W) + lambda I) beta = sqrt(W) y
    预测系数 alpha = sqrt(W) beta。
    """
    kernel = rbfKernel(x, x, gamma)
    if sample_weights is None:
        alpha = stableLinearSolve(
            kernel + float(lam) * np.eye(len(kernel), dtype=float),
            y,
        )
        return kernel, alpha

    weights = np.asarray(sample_weights, dtype=float)
    weights = np.maximum(weights, 1e-12)
    weights /= max(float(weights.mean()), 1e-12)
    sqrt_w = np.sqrt(weights)
    weighted_kernel = sqrt_w[:, None] * kernel * sqrt_w[None, :]
    beta = stableLinearSolve(
        weighted_kernel + float(lam) * np.eye(len(kernel), dtype=float),
        sqrt_w * y,
    )
    alpha = sqrt_w * beta
    return kernel, alpha


def fitWeightedKrrFromKernel(
    kernel: np.ndarray,
    y: np.ndarray,
    lam: float,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """对预计算的对称核矩阵执行带权KRR。"""
    kernel = np.asarray(kernel, dtype=float)
    y = np.asarray(y, dtype=float)
    if sample_weights is None:
        return stableLinearSolve(
            kernel + float(lam) * np.eye(len(kernel), dtype=float),
            y,
        )

    weights = np.maximum(np.asarray(sample_weights, dtype=float), 1e-12)
    weights /= max(float(weights.mean()), 1e-12)
    sqrt_w = np.sqrt(weights)
    weighted_kernel = sqrt_w[:, None] * kernel * sqrt_w[None, :]
    beta = stableLinearSolve(
        weighted_kernel + float(lam) * np.eye(len(kernel), dtype=float),
        sqrt_w * y,
    )
    return sqrt_w * beta


def shrinkTargetTowardGroupMean(
    data: pd.DataFrame,
    target: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    将窗口标签小幅收缩到训练试验组均值，但仍保留全部窗口样本：
        y*=(1-rho)y+rho*mean_group(y)
    """
    rho = float(np.clip(strength, 0.0, 1.0))
    y = np.asarray(target, dtype=float)
    if rho <= 0.0:
        return y.copy()
    keys = [key for key in SPLIT_GROUP_KEY if key in data.columns]
    temp = data[keys].reset_index(drop=True).copy()
    temp["_target"] = y
    group_mean = temp.groupby(keys, sort=False)["_target"].transform("mean").to_numpy(dtype=float)
    return (1.0 - rho) * y + rho * group_mean


def groupBalanceWeights(
    data: pd.DataFrame,
    keys: list[str],
) -> np.ndarray:
    if len(data) == 0:
        return np.ones(0, dtype=float)
    usable = [key for key in keys if key in data.columns]
    if not usable:
        return np.ones(len(data), dtype=float)
    reset = data[usable].reset_index(drop=True)
    sizes = reset.groupby(usable, sort=False)[usable[0]].transform("size").to_numpy(dtype=float)
    weights = 1.0 / np.maximum(sizes, 1.0)
    weights /= max(float(weights.mean()), 1e-12)
    return weights


def groupMeanSystemTraining(
    spec: CandidateSpec,
    train: pd.DataFrame,
    xunlianTezheng: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    将训练窗口压缩为 (date, flow_point, disturbance_id) 组均值样本。

    对小组目标向同流量点训练均值做经验收缩：
        y_shrunk = (n*y_group + k*y_flow_prior)/(n+k)
    其中k=system_group_shrinkage。
    """
    columns = columnsForFeatureSet(xunlianTezheng, spec.system_feature_set)
    keys = [key for key in SPLIT_GROUP_KEY if key in train.columns]
    if not keys:
        raise ValueError("missing split group keys for two-stage system model")

    temp = train[keys].reset_index(drop=True).copy()
    for column in columns:
        temp[column] = xunlianTezheng[column].to_numpy(dtype=float)
    temp["_target"] = np.asarray(y, dtype=float)

    aggregations = {column: "mean" for column in columns}
    aggregations["_target"] = "mean"
    grouped = temp.groupby(keys, sort=True).agg(aggregations)
    grouped["_n"] = temp.groupby(keys, sort=True).size().astype(float)
    grouped = grouped.reset_index()

    flow_prior = grouped.groupby("flow_point", sort=False)["_target"].transform("mean")
    shrink = max(float(spec.system_group_shrinkage), 0.0)
    if shrink > 0.0:
        target = (
            grouped["_n"].to_numpy(dtype=float) * grouped["_target"].to_numpy(dtype=float)
            + shrink * flow_prior.to_numpy(dtype=float)
        ) / (grouped["_n"].to_numpy(dtype=float) + shrink)
    else:
        target = grouped["_target"].to_numpy(dtype=float)

    return grouped[columns].copy(), target, grouped["_n"].to_numpy(dtype=float)


def centerTargetWithinGroups(
    data: pd.DataFrame,
    target: np.ndarray,
    keys: list[str],
) -> np.ndarray:
    reset = data[keys].reset_index(drop=True).copy()
    reset["_target"] = np.asarray(target, dtype=float)
    group_mean = reset.groupby(keys, sort=False)["_target"].transform("mean").to_numpy(dtype=float)
    return np.asarray(target, dtype=float) - group_mean


def fitCandidate(
    spec: CandidateSpec,
    train: pd.DataFrame,
    xunlianTezheng: pd.DataFrame,
) -> FittedCandidate:
    base_weights = resolveBaseWeights(spec, train)
    model = FittedCandidate(spec=spec, base_weights=base_weights)

    if spec.model_type == "baseline":
        return model

    y = residualTargetPct(model, train)

    if spec.model_type == "multikernel_krr":
        physical_columns = columnsForFeatureSet(
            xunlianTezheng,
            spec.feature_set,
        )
        raw_columns = columnsForFeatureSet(
            xunlianTezheng,
            spec.raw_feature_set,
        )
        physical_standardizer = StandardizerLite.fit(
            xunlianTezheng,
            physical_columns,
        )
        raw_standardizer = StandardizerLite.fit(
            xunlianTezheng,
            raw_columns,
        )
        physical_x = physical_standardizer.transform(xunlianTezheng)
        raw_x = raw_standardizer.transform(xunlianTezheng)

        beta = float(np.clip(spec.raw_kernel_weight, 0.0, 1.0))
        kernel = (
            (1.0 - beta) * rbfKernel(physical_x, physical_x, spec.gamma)
            + beta * rbfKernel(raw_x, raw_x, spec.raw_gamma)
        )
        fitted_target = shrinkTargetTowardGroupMean(
            train,
            y,
            spec.target_group_shrinkage,
        )
        sample_weights = (
            groupBalanceWeights(train, SPLIT_GROUP_KEY)
            if spec.sample_group_balance
            else None
        )
        alpha = fitWeightedKrrFromKernel(
            kernel,
            fitted_target,
            lam=spec.lam,
            sample_weights=sample_weights,
        )

        model.standardizer = physical_standardizer
        model.train_x = physical_x
        model.raw_standardizer = raw_standardizer
        model.raw_train_x = raw_x
        model.alpha = alpha
        fitD0PostCalibration(model, train, xunlianTezheng)
        return model

    if spec.model_type == "two_stage_krr":
        # 第一阶段：用试验组均值学习系统偏差。
        group_features, group_target, _ = groupMeanSystemTraining(
            spec,
            train,
            xunlianTezheng,
            y,
        )
        system_columns = columnsForFeatureSet(group_features, spec.system_feature_set)
        system_standardizer = StandardizerLite.fit(group_features, system_columns)
        system_x = system_standardizer.transform(group_features)
        _, system_alpha = fitWeightedKrr(
            system_x,
            group_target,
            gamma=spec.system_gamma,
            lam=spec.system_lam,
            sample_weights=None,  # 每个试验组等权
        )

        model.system_standardizer = system_standardizer
        model.system_train_x = system_x
        model.system_alpha = system_alpha

        # 第二阶段：只学习系统模型尚未解释的组内窗口偏差。
        if spec.dynamic_rho > 0.0:
            train_system_x = system_standardizer.transform(xunlianTezheng)
            system_prediction = (
                rbfKernel(train_system_x, system_x, spec.system_gamma)
                @ system_alpha
            )
            dynamic_target = y - system_prediction
            if spec.dynamic_center_target:
                dynamic_target = centerTargetWithinGroups(
                    train,
                    dynamic_target,
                    [key for key in SPLIT_GROUP_KEY if key in train.columns],
                )

            dynamic_columns = columnsForFeatureSet(
                xunlianTezheng,
                spec.dynamic_feature_set,
            )
            dynamic_standardizer = StandardizerLite.fit(
                xunlianTezheng,
                dynamic_columns,
            )
            dynamic_x = dynamic_standardizer.transform(xunlianTezheng)
            dynamic_weights = (
                groupBalanceWeights(train, SPLIT_GROUP_KEY)
                if spec.dynamic_group_balance
                else None
            )
            _, dynamic_alpha = fitWeightedKrr(
                dynamic_x,
                dynamic_target,
                gamma=spec.dynamic_gamma,
                lam=spec.dynamic_lam,
                sample_weights=dynamic_weights,
            )
            model.dynamic_standardizer = dynamic_standardizer
            model.dynamic_train_x = dynamic_x
            model.dynamic_alpha = dynamic_alpha

        fitD0PostCalibration(model, train, xunlianTezheng)
        return model

    # 单阶段基准模型。
    columns = columnsForFeatureSet(xunlianTezheng, spec.feature_set)
    standardizer = StandardizerLite.fit(xunlianTezheng, columns)
    x = standardizer.transform(xunlianTezheng)

    fitted_target = shrinkTargetTowardGroupMean(
        train,
        y,
        spec.target_group_shrinkage,
    )
    sample_weights = (
        groupBalanceWeights(train, SPLIT_GROUP_KEY)
        if spec.sample_group_balance
        else None
    )

    if spec.model_type == "ridge":
        penalty = spec.lam * np.eye(x.shape[1], dtype=float)
        coef = stableLinearSolve(x.T @ x + penalty, x.T @ fitted_target)
        model.standardizer = standardizer
        model.coef = coef
    elif spec.model_type == "krr_rbf":
        model.standardizer = standardizer
        _, alpha = fitWeightedKrr(
            x,
            fitted_target,
            gamma=spec.gamma,
            lam=spec.lam,
            sample_weights=sample_weights,
        )
        model.train_x = x
        model.alpha = alpha
    else:
        raise ValueError(f"unsupported model_type: {spec.model_type}")

    fitD0PostCalibration(model, train, xunlianTezheng)
    return model


def fitD0PostCalibration(
    model: FittedCandidate,
    train: pd.DataFrame,
    xunlianTezheng: pd.DataFrame,
) -> None:
    if not model.spec.blend_d0_curve or model.spec.model_type == "baseline":
        return

    base = model.baseVolume(train)
    raw = np.clip(
        model.rawCorrectionPct(xunlianTezheng),
        -model.spec.correction_clip_pct,
        model.spec.correction_clip_pct,
    )
    yuceJieguo = base * (1.0 + raw / 100.0)
    standard = train["standard_volume_m3"].to_numpy(dtype=float)
    errorPct = (yuceJieguo - standard) / np.maximum(standard, 1e-12) * 100.0

    d0_mask = train["condition_note"].eq("no_disturbance_reference").to_numpy()
    if int(d0_mask.sum()) < 3:
        return

    q = xunlianTezheng.loc[d0_mask, "q"].to_numpy(dtype=float)
    degree = max(int(model.spec.d0_curve_degree), 0)
    design = np.column_stack([q**power for power in range(degree + 1)])
    target = -errorPct[d0_mask]
    penalty = float(model.spec.d0_curve_lam) * np.eye(design.shape[1], dtype=float)
    model.d0_curve_coef = stableLinearSolve(
        design.T @ design + penalty,
        design.T @ target,
    )

# ---------------------------------------------------------------------------
# 标签盲划分
# ---------------------------------------------------------------------------


def chooseN5TrainIndices(indices: np.ndarray, rng: np.random.Generator) -> set[int]:
    """不读取标签和基础误差，随机选两个窗口进入训练。"""
    idx = np.asarray(indices, dtype=int)
    if len(idx) <= 2:
        return set(int(i) for i in idx)
    return set(int(i) for i in rng.choice(idx, size=2, replace=False))


def splitAssignments(
    data: pd.DataFrame,
    spec: SplitSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(spec.seed)
    target_validation = int(round(len(data) * TARGET_VALIDATION_FRACTION))
    train_idx: set[int] = set()
    validation_idx: set[int] = set()

    groups = list(data.groupby(SPLIT_GROUP_KEY, sort=True))

    # D0 很稀疏，延续原脚本的分配逻辑；选择过程不读取 standard_volume。
    for key, group in groups:
        disturbance_id = str(key[2])
        if disturbance_id != "D0":
            continue
        idx = group.index.to_numpy(dtype=int)
        n = len(idx)
        if n >= 5:
            chosen = chooseN5TrainIndices(idx, rng)
            train_idx.update(chosen)
            validation_idx.update(int(i) for i in idx if int(i) not in chosen)
        elif n in {3, 4}:
            if spec.d0_n3_to_validation:
                validation_idx.update(int(i) for i in idx)
            else:
                train_idx.update(int(i) for i in idx)
        elif n == 2:
            chosen = int(rng.choice(idx))
            train_idx.add(chosen)
            validation_idx.update(int(i) for i in idx if int(i) != chosen)
        else:
            if rng.random() < 0.7:
                validation_idx.update(int(i) for i in idx)
            else:
                train_idx.update(int(i) for i in idx)

    n34_candidates: list[tuple[float, np.ndarray]] = []
    small_candidates: list[int] = []
    for key, group in groups:
        if str(key[2]) == "D0":
            continue
        idx = group.index.to_numpy(dtype=int)
        n = len(idx)
        if n >= 5:
            chosen = chooseN5TrainIndices(idx, rng)
            train_idx.update(chosen)
            validation_idx.update(int(i) for i in idx if int(i) not in chosen)
        elif n in {3, 4}:
            flow = int(key[1])
            official_flow = 1.0 if 40 <= flow <= 100 else 0.0
            score = spec.n34_validation_bias * official_flow + rng.random()
            n34_candidates.append((score, idx))
        else:
            small_candidates.extend(int(i) for i in idx)

    n34_candidates = sorted(n34_candidates, key=lambda item: item[0], reverse=True)
    selected_whole_groups = 0
    max_whole_groups = 3 + (spec.seed % 2)
    for _, idx in n34_candidates:
        idx_set = set(int(i) for i in idx)
        if len(validation_idx) < target_validation and selected_whole_groups < max_whole_groups:
            validation_idx.update(idx_set)
            selected_whole_groups += 1
        else:
            train_idx.update(idx_set)

    rng.shuffle(small_candidates)
    for index in small_candidates:
        if len(validation_idx) < target_validation and rng.random() < spec.small_validation_prob:
            validation_idx.add(int(index))
        else:
            train_idx.add(int(index))

    unassigned = set(range(len(data))) - train_idx - validation_idx
    for index in sorted(unassigned):
        if len(validation_idx) < target_validation:
            validation_idx.add(index)
        else:
            train_idx.add(index)

    overlap = train_idx & validation_idx
    if overlap:
        raise RuntimeError(f"split overlap: {sorted(overlap)[:5]}")

    split = data[["window_id", "date", "flow_point", "condition_note", "disturbance_id"]].copy()
    split["split"] = ["validation" if i in validation_idx else "train" for i in range(len(data))]
    split["split_mode"] = "label_blind_stratified_group_internal"
    split["split_seed"] = spec.seed

    train = data.loc[split["split"].eq("train")].copy().reset_index(drop=True)
    validation = data.loc[split["split"].eq("validation")].copy().reset_index(drop=True)
    return train, validation, split


def validateSplitRules(data: pd.DataFrame, split: pd.DataFrame) -> tuple[bool, str]:
    merged = data[SPLIT_GROUP_KEY].copy()
    merged["split"] = split["split"].to_numpy()
    for key, group in merged.groupby(SPLIT_GROUP_KEY, sort=True):
        n = len(group)
        train_n = int(group["split"].eq("train").sum())
        validation_n = int(group["split"].eq("validation").sum())
        if n >= 5 and not (train_n == 2 and validation_n >= 3):
            return False, f"n>=5 rule failed for {key}: train={train_n}, validation={validation_n}"
        if n in {3, 4} and not (
            (train_n == n and validation_n == 0)
            or (train_n == 0 and validation_n == n)
        ):
            return False, f"n=3/4 whole-group rule failed for {key}: train={train_n}, validation={validation_n}"
    return True, "ok"


def pressureSplit(
    data: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    target_validation = int(round(len(data) * TARGET_VALIDATION_FRACTION))
    validation_idx: set[int] = set()
    candidates: list[tuple[int, int, float, np.ndarray]] = []

    for key, group in data.groupby(SPLIT_GROUP_KEY, sort=True):
        flow = int(key[1])
        priority = 2 if str(key[2]) == "D0" else int(40 <= flow <= 100)
        candidates.append((priority, len(group), rng.random(), group.index.to_numpy(dtype=int)))

    for _, _, _, idx in sorted(candidates, key=lambda item: (-item[0], -item[1], item[2])):
        if len(validation_idx) >= target_validation:
            break
        validation_idx.update(int(i) for i in idx)

    train_idx = set(range(len(data))) - validation_idx
    split = data[["window_id", "date", "flow_point", "condition_note", "disturbance_id"]].copy()
    split["split"] = ["validation" if i in validation_idx else "train" for i in range(len(data))]
    split["split_mode"] = "whole_stratified_group_holdout_pressure"
    split["split_seed"] = seed

    train = data.loc[split["split"].eq("train")].copy().reset_index(drop=True)
    validation = data.loc[split["split"].eq("validation")].copy().reset_index(drop=True)
    return train, validation, split


# ---------------------------------------------------------------------------
# 候选模型
# ---------------------------------------------------------------------------


def buildSpecs() -> list[CandidateSpec]:
    """
    物理--稳定性双核 KRR 预注册小网格。

    主搜索：
    - raw_gamma: 0.002 / 0.005 / 0.01 / 0.02
    - raw kernel weight beta: 0.35 / 0.50 / 0.65 / 0.80
    - 固定 physical gamma=0.1, lambda=0.3, 线性 D0

    消融：
    - 全部10个附件6特征；
    - 删除Hampel两特征；
    - 仅保留5个波动特征。

    局部稳健性：
    - 对最有希望的平滑原始核提高lambda；
    - 对少量候选比较 physical gamma；
    - 保留无D0对照，避免后校准造成虚假改善。
    """
    specs: list[CandidateSpec] = [
        CandidateSpec(
            model_id="baseline_phys6",
            family="baseline",
            model_type="baseline",
            base_mode="phys6",
            blend_d0_curve=False,
        ),
        CandidateSpec(
            model_id="krr_phys6_compact_g0.1_lam0.1_d0quad",
            family="kernel_ridge_rbf",
            model_type="krr_rbf",
            base_mode="phys6",
            feature_set="compact_physical",
            gamma=0.1,
            lam=0.1,
            blend_d0_curve=True,
            d0_curve_degree=2,
            d0_curve_lam=0.1,
        ),
        CandidateSpec(
            model_id="krr_phys6_raw_single_g0.03_lam0.3_d0lin",
            family="kernel_ridge_rbf",
            model_type="krr_rbf",
            base_mode="phys6",
            feature_set="compact_plus_raw",
            gamma=0.03,
            lam=0.3,
            blend_d0_curve=True,
            d0_curve_degree=1,
            d0_curve_lam=1.0,
        ),
        CandidateSpec(
            model_id="krr_mk_v51_pg0.1_rg0.01_b0.4_lam0.3_d0lin",
            family="kernel_ridge_rbf",
            model_type="multikernel_krr",
            base_mode="phys6",
            feature_set="compact_physical",
            raw_feature_set="raw_stability_only",
            gamma=0.1,
            raw_gamma=0.01,
            raw_kernel_weight=0.40,
            lam=0.3,
            blend_d0_curve=True,
            d0_curve_degree=1,
            d0_curve_lam=1.0,
        ),
    ]

    # A. 当前边界向外扩展：全部10个附件6特征。
    for raw_gamma in [0.002, 0.005, 0.01, 0.02]:
        for beta in [0.35, 0.50, 0.65, 0.80]:
            specs.append(
                CandidateSpec(
                    model_id=(
                        f"krr_mk_all_pg0.1_rg{raw_gamma:g}_b{beta:g}"
                        "_lam0.3_d0lin"
                    ),
                    family="kernel_ridge_rbf",
                    model_type="multikernel_krr",
                    base_mode="phys6",
                    feature_set="compact_physical",
                    raw_feature_set="raw_stability_only",
                    gamma=0.1,
                    raw_gamma=raw_gamma,
                    raw_kernel_weight=beta,
                    lam=0.3,
                    blend_d0_curve=True,
                    d0_curve_degree=1,
                    d0_curve_lam=1.0,
                )
            )

    # B. 特征组消融：删除Hampel特征，或仅保留纯波动特征。
    for raw_feature_set, tag in [
        ("raw_no_hampel", "nohp"),
        ("raw_variability_only", "var"),
    ]:
        for raw_gamma in [0.005, 0.01]:
            for beta in [0.50, 0.65]:
                specs.append(
                    CandidateSpec(
                        model_id=(
                            f"krr_mk_{tag}_pg0.1_rg{raw_gamma:g}_b{beta:g}"
                            "_lam0.3_d0lin"
                        ),
                        family="kernel_ridge_rbf",
                        model_type="multikernel_krr",
                        base_mode="phys6",
                        feature_set="compact_physical",
                        raw_feature_set=raw_feature_set,
                        gamma=0.1,
                        raw_gamma=raw_gamma,
                        raw_kernel_weight=beta,
                        lam=0.3,
                        blend_d0_curve=True,
                        d0_curve_degree=1,
                        d0_curve_lam=1.0,
                    )
                )

    # C. 更强正则：若高beta降低MAE但放大尾部，lambda=0.6可抑制。
    for raw_gamma in [0.005, 0.01]:
        for beta in [0.50, 0.65]:
            specs.append(
                CandidateSpec(
                    model_id=(
                        f"krr_mk_all_pg0.1_rg{raw_gamma:g}_b{beta:g}"
                        "_lam0.6_d0lin"
                    ),
                    family="kernel_ridge_rbf",
                    model_type="multikernel_krr",
                    base_mode="phys6",
                    feature_set="compact_physical",
                    raw_feature_set="raw_stability_only",
                    gamma=0.1,
                    raw_gamma=raw_gamma,
                    raw_kernel_weight=beta,
                    lam=0.6,
                    blend_d0_curve=True,
                    d0_curve_degree=1,
                    d0_curve_lam=1.0,
                )
            )

    # D. 物理核尺度局部消融，只围绕最平滑原始核区域。
    for physical_gamma in [0.05, 0.20]:
        for beta in [0.50, 0.65]:
            specs.append(
                CandidateSpec(
                    model_id=(
                        f"krr_mk_all_pg{physical_gamma:g}_rg0.005_b{beta:g}"
                        "_lam0.3_d0lin"
                    ),
                    family="kernel_ridge_rbf",
                    model_type="multikernel_krr",
                    base_mode="phys6",
                    feature_set="compact_physical",
                    raw_feature_set="raw_stability_only",
                    gamma=physical_gamma,
                    raw_gamma=0.005,
                    raw_kernel_weight=beta,
                    lam=0.3,
                    blend_d0_curve=True,
                    d0_curve_degree=1,
                    d0_curve_lam=1.0,
                )
            )

    # E. 少量无D0对照，判断线性D0是否真正改善跨划分稳健性。
    for raw_gamma, beta in [(0.005, 0.50), (0.01, 0.50)]:
        specs.append(
            CandidateSpec(
                model_id=(
                    f"krr_mk_all_pg0.1_rg{raw_gamma:g}_b{beta:g}"
                    "_lam0.3_nod0"
                ),
                family="kernel_ridge_rbf",
                model_type="multikernel_krr",
                base_mode="phys6",
                feature_set="compact_physical",
                raw_feature_set="raw_stability_only",
                gamma=0.1,
                raw_gamma=raw_gamma,
                raw_kernel_weight=beta,
                lam=0.3,
                blend_d0_curve=False,
            )
        )

    # 去重保护，避免同一规格被重复加入而无意增加选择权重。
    unique: dict[str, CandidateSpec] = {}
    for spec in specs:
        if spec.model_id in unique:
            raise RuntimeError(f"duplicate model_id: {spec.model_id}")
        unique[spec.model_id] = spec
    return list(unique.values())

# ---------------------------------------------------------------------------
# 评价、诊断与报告
# ---------------------------------------------------------------------------


def evaluateModel(
    model: FittedCandidate,
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = modelFeatureFrame(frame)
    yuceJieguo, diagnostics = model.predictWithDiagnostics(frame, features)
    summary, groups, windows = evaluatePredictions(frame, yuceJieguo)
    diagnostics = diagnostics.merge(
        windows[["window_id", "model_error_pct"]],
        on="window_id",
        how="left",
    )
    return summary, groups, windows, diagnostics


def evaluateNamedVolume(
    data: pd.DataFrame,
    name: str,
    volume: np.ndarray,
) -> dict[str, object]:
    summary, groups, _ = evaluatePredictions(data, np.asarray(volume, dtype=float))
    return {
        "method": name,
        "mae_pct": float(summary["mae_pct"]),
        "group_pass": summary["group_pass"],
        "max_abs_group_mean_pct": float(summary["max_abs_group_mean_pct"]),
        "max_group_sd_pct": float(summary["max_group_sd_pct"]),
        "u_nor_L_pct": float(summary["u_nor_L_pct"]),
        "u_nor_r_pct": float(summary["u_nor_r_pct"]),
        "u_nor_d_pct": float(summary["u_nor_d_pct"]),
        "pass_all_standards": bool(summary["pass_all_standards"]),
        "worst_group": (
            ""
            if groups.empty
            else f"{groups.sort_values('repeatability_sd_pct', ascending=False).iloc[0]['date']}-"
                 f"{groups.sort_values('repeatability_sd_pct', ascending=False).iloc[0]['flow_point']}"
        ),
    }


def physicalBaseDiagnostics(
    model: FittedCandidate,
    data: pd.DataFrame,
    model_prediction: np.ndarray,
) -> pd.DataFrame:
    matrix = baseMatrix(data)
    rows = [
        evaluateNamedVolume(data, "Phys6", matrix[:, 0]),
        evaluateNamedVolume(data, "RawHampelPhys6_s0.25", rawHampelPhys6Volume(data, 0.25, 0.5)),
        evaluateNamedVolume(data, "RawHampelPhys6_s0.5", rawHampelPhys6Volume(data, 0.5, 0.5)),
        evaluateNamedVolume(data, "RawHampelPhys6_s1.0", rawHampelPhys6Volume(data, 1.0, 0.5)),
        evaluateNamedVolume(data, "Lagrange", matrix[:, 1]),
        evaluateNamedVolume(data, "OWICS", matrix[:, 2]),
        evaluateNamedVolume(data, "EqualWeight", matrix[:, 3]),
        evaluateNamedVolume(data, "SelectedPhysicalBase", model.baseVolume(data)),
        evaluateNamedVolume(data, "FinalKRR", model_prediction),
    ]
    return pd.DataFrame(rows)


def parseGroupPassRatio(value: str) -> float:
    try:
        passed, total = str(value).split("/")
        total_i = int(total)
        return float(int(passed) / total_i) if total_i > 0 else float("nan")
    except Exception:
        return float("nan")


def aggregateMultisplit(ranking: pd.DataFrame) -> pd.DataFrame:
    frame = ranking.copy()
    frame["validation_group_pass_ratio"] = frame["validation_group_pass"].map(parseGroupPassRatio)

    rows: list[dict[str, object]] = []
    for model_id, group in frame.groupby("model_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "family": first["family"],
                "split_count": int(len(group)),
                "all_pass_count": int(group["validation_pass_all"].astype(bool).sum()),
                "all_pass_rate": float(group["validation_pass_all"].astype(bool).mean()),
                "median_fail_count": float(group["validation_fail_count"].median()),
                "median_gap_score": float(group["validation_gap_score"].median()),
                "q75_gap_score": float(group["validation_gap_score"].quantile(0.75)),
                "median_mae_pct": float(group["validation_mae_pct"].median()),
                "median_group_pass_ratio": float(group["validation_group_pass_ratio"].median()),
                "median_u_nor_L_pct": float(group["validation_u_nor_L_pct"].median()),
                "median_u_nor_r_pct": float(group["validation_u_nor_r_pct"].median()),
                "q75_u_nor_r_pct": float(group["validation_u_nor_r_pct"].quantile(0.75)),
                "q90_u_nor_r_pct": float(group["validation_u_nor_r_pct"].quantile(0.90)),
                "max_u_nor_r_pct": float(group["validation_u_nor_r_pct"].max()),
                "u_nor_r_pass_count": int(
                    (group["validation_u_nor_r_pct"] < STANDARD_LIMITS["u_nor_r_pct"]).sum()
                ),
                "median_u_nor_d_pct": float(group["validation_u_nor_d_pct"].median()),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        [
            "all_pass_count",
            "median_fail_count",
            "median_gap_score",
            "q75_gap_score",
            "q90_u_nor_r_pct",
            "median_u_nor_d_pct",
            "median_mae_pct",
        ],
        ascending=[False, True, True, True, True, True, True],
    ).reset_index(drop=True)


def chooseKrrChampion(multisplit: pd.DataFrame) -> pd.Series:
    candidates = multisplit[multisplit["family"].eq("kernel_ridge_rbf")].copy()
    if candidates.empty:
        raise RuntimeError("no KRR candidates in multisplit summary")
    return candidates.iloc[0]


def chooseRepresentativeRow(ranking: pd.DataFrame, champion_id: str) -> pd.Series:
    rows = ranking[ranking["model_id"].eq(champion_id)].copy()
    if rows.empty:
        raise RuntimeError(f"champion not found: {champion_id}")
    medians = {
        "mae": float(rows["validation_mae_pct"].median()),
        "ur": float(rows["validation_u_nor_r_pct"].median()),
        "ul": float(rows["validation_u_nor_L_pct"].median()),
        "ud": float(rows["validation_u_nor_d_pct"].median()),
    }
    rows["representative_distance"] = (
        np.abs(rows["validation_mae_pct"] - medians["mae"])
        + np.abs(rows["validation_u_nor_r_pct"] - medians["ur"])
        + np.abs(rows["validation_u_nor_L_pct"] - medians["ul"])
        + np.abs(rows["validation_u_nor_d_pct"] - medians["ud"])
    )
    return rows.sort_values(
        ["validation_fail_count", "representative_distance", "validation_gap_score"],
        ascending=[True, True, True],
    ).iloc[0]


def toJsonable(value):
    if isinstance(value, dict):
        return {str(key): toJsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [toJsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [toJsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def saveFittedModel(model: FittedCandidate, output_dir: Path) -> None:
    meta = {
        "spec": asdict(model.spec),
        "base_columns": BASE_COLUMNS,
        "base_weights": None if model.base_weights is None else model.base_weights.tolist(),
        "feature_columns": model.standardizer.columns if model.standardizer is not None else [],
        "system_feature_columns": (
            model.system_standardizer.columns if model.system_standardizer is not None else []
        ),
        "raw_kernel_feature_columns": (
            model.raw_standardizer.columns if model.raw_standardizer is not None else []
        ),
        "dynamic_feature_columns": (
            model.dynamic_standardizer.columns if model.dynamic_standardizer is not None else []
        ),
        "d0_curve_coef": None if model.d0_curve_coef is None else model.d0_curve_coef.tolist(),
        "note": (
            "Full-data fitted parameters. For multikernel_krr, the physical and raw-stability "
            "feature spaces use separate RBF kernels and one shared alpha vector. "
            "Generalisation evidence is reported separately from the full-data fit."
        ),
    }
    (output_dir / "krr_final_model_meta.json").write_text(
        json.dumps(toJsonable(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    arrays: dict[str, np.ndarray] = {}
    if model.base_weights is not None:
        arrays["base_weights"] = model.base_weights
    arrays["base_columns"] = np.asarray(BASE_COLUMNS, dtype="U")

    if model.standardizer is not None:
        arrays["feature_columns"] = np.asarray(model.standardizer.columns, dtype="U")
        arrays["median"] = model.standardizer.median.reindex(model.standardizer.columns).to_numpy(dtype=float)
        arrays["mean"] = model.standardizer.mean
        arrays["scale"] = model.standardizer.scale
    if model.train_x is not None:
        arrays["train_x"] = model.train_x
    if model.alpha is not None:
        arrays["alpha"] = model.alpha
    if model.raw_standardizer is not None:
        arrays["raw_kernel_feature_columns"] = np.asarray(
            model.raw_standardizer.columns,
            dtype="U",
        )
        arrays["raw_kernel_median"] = model.raw_standardizer.median.reindex(
            model.raw_standardizer.columns
        ).to_numpy(dtype=float)
        arrays["raw_kernel_mean"] = model.raw_standardizer.mean
        arrays["raw_kernel_scale"] = model.raw_standardizer.scale
    if model.raw_train_x is not None:
        arrays["raw_train_x"] = model.raw_train_x
    if model.coef is not None:
        arrays["coef"] = model.coef

    if model.system_standardizer is not None:
        arrays["system_feature_columns"] = np.asarray(model.system_standardizer.columns, dtype="U")
        arrays["system_median"] = model.system_standardizer.median.reindex(
            model.system_standardizer.columns
        ).to_numpy(dtype=float)
        arrays["system_mean"] = model.system_standardizer.mean
        arrays["system_scale"] = model.system_standardizer.scale
    if model.system_train_x is not None:
        arrays["system_train_x"] = model.system_train_x
    if model.system_alpha is not None:
        arrays["system_alpha"] = model.system_alpha

    if model.dynamic_standardizer is not None:
        arrays["dynamic_feature_columns"] = np.asarray(model.dynamic_standardizer.columns, dtype="U")
        arrays["dynamic_median"] = model.dynamic_standardizer.median.reindex(
            model.dynamic_standardizer.columns
        ).to_numpy(dtype=float)
        arrays["dynamic_mean"] = model.dynamic_standardizer.mean
        arrays["dynamic_scale"] = model.dynamic_standardizer.scale
    if model.dynamic_train_x is not None:
        arrays["dynamic_train_x"] = model.dynamic_train_x
    if model.dynamic_alpha is not None:
        arrays["dynamic_alpha"] = model.dynamic_alpha

    if model.d0_curve_coef is not None:
        arrays["d0_curve_coef"] = model.d0_curve_coef
    np.savez_compressed(output_dir / "krr_final_model.npz", **arrays)

def markdownTable(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return ""
    use = frame.head(max_rows).copy()
    for column in use.columns:
        if pd.api.types.is_float_dtype(use[column]):
            use[column] = use[column].map(lambda value: f"{value:.6g}")
    use = use.astype(str)
    headers = list(use.columns)
    rows = use.values.tolist()
    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    header_line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, separator] + body)


def writeReport(
    ranking: pd.DataFrame,
    multisplit: pd.DataFrame,
    champion: pd.Series,
    champion_spec: CandidateSpec,
    representative: pd.Series,
    representative_model: FittedCandidate,
    representative_groups: pd.DataFrame,
    representative_base_diag: pd.DataFrame,
    pressure_summary: pd.Series,
    pressure_groups: pd.DataFrame,
    pressure_base_diag: pd.DataFrame,
    fullfit_summary: pd.Series,
    fullfit_base_diag: pd.DataFrame,
    full_model: FittedCandidate,
) -> None:
    multi_columns = [
        "model_id",
        "all_pass_count",
        "median_fail_count",
        "median_gap_score",
        "median_mae_pct",
        "median_group_pass_ratio",
        "median_u_nor_L_pct",
        "median_u_nor_r_pct",
        "q75_u_nor_r_pct",
        "u_nor_r_pass_count",
        "median_u_nor_d_pct",
    ]
    ranking_columns = [
        "split_no",
        "model_id",
        "validation_fail_count",
        "validation_pass_all",
        "validation_mae_pct",
        "validation_group_pass",
        "validation_max_abs_group_mean_pct",
        "validation_u_nor_L_pct",
        "validation_u_nor_r_pct",
        "validation_u_nor_d_pct",
    ]

    rep_weights = (
        "median4（无固定线性权重）"
        if representative_model.base_weights is None
        else json.dumps(dict(zip(BASE_COLUMNS, representative_model.base_weights.tolist())), ensure_ascii=False)
    )
    full_weights = (
        "median4（无固定线性权重）"
        if full_model.base_weights is None
        else json.dumps(dict(zip(BASE_COLUMNS, full_model.base_weights.tolist())), ensure_ascii=False)
    )

    text = f"""# 分层划分多核 KRR 边界扩展与附件6特征消融报告

## 本轮修改

- 恢复按 `(date, flow_point, disturbance_id)` 进行标签盲分层划分。
- 官方指标仍按题目脚本计算，其中重复性组为 `(date, flow_point)`。
- 第一阶段按 `(date, flow_point, disturbance_id)` 组均值学习系统偏差；第二阶段只学习组内窗口偏差并用 `rho` 强收缩。
- 从附件6逐秒 `diff_ns_0–9` 与 `mean_us_0–9` 提取MAD、漂移、同步性、Hampel尖峰率和A/B不一致特征。
- Hampel滤波仅替换局部尖峰，并以滤波前后Phys6整窗积分信号之比微调附件1 Phys6，不使用平台均值替代整窗积分。
- 附件6逐秒序列先按window_id聚合为窗口级稳定性特征；本版比较全部特征、删除Hampel特征和仅波动特征，并扩展原始核尺度与核权重边界。
- 日期、标准体积和扰流编号只用于离线划分或评价，不进入最终运行时特征。

## 40 次标签盲划分聚合排名

{markdownTable(multisplit[multi_columns], 20)}

## 最终 KRR 配置

```text
{champion['model_id']}
```

- 规格：`{asdict(champion_spec)}`
- 全标准通过：`{int(champion['all_pass_count'])}/{int(champion['split_count'])}`
- 中位 MAE：`{champion['median_mae_pct']:.6g}%`
- 中位 $u_{{nor,r}}$：`{champion['median_u_nor_r_pct']:.6g}%`
- 75% 分位 $u_{{nor,r}}$：`{champion['q75_u_nor_r_pct']:.6g}%`
- 90% 分位 $u_{{nor,r}}$：`{champion['q90_u_nor_r_pct']:.6g}%`
- $u_{{nor,r}}$ 通过次数：`{int(champion['u_nor_r_pass_count'])}/{int(champion['split_count'])}`

## 代表性验证划分

- split：`{int(representative['split_no'])}`，seed：`{int(representative['split_seed'])}`
- MAE：`{representative['validation_mae_pct']:.6g}%`
- 分组通过：`{representative['validation_group_pass']}`
- $u_{{nor,L}}$：`{representative['validation_u_nor_L_pct']:.6g}%`
- $u_{{nor,r}}$：`{representative['validation_u_nor_r_pct']:.6g}%`
- $u_{{nor,d}}$：`{representative['validation_u_nor_d_pct']:.6g}%`
- 本折物理基线权重：`{rep_weights}`

### 代表性验证的物理基线方差分解

{markdownTable(representative_base_diag, 20)}

### 代表性验证中重复性最差的组

{markdownTable(representative_groups.sort_values('repeatability_sd_pct', ascending=False), 12)}

## 单次结果前 20（仅诊断）

{markdownTable(ranking[ranking_columns], 20)}

## 整组留出压力测试

{markdownTable(pd.DataFrame([pressure_summary.to_dict()]), 1)}

### 压力测试物理基线方差分解

{markdownTable(pressure_base_diag, 20)}

### 压力测试中重复性最差的组

{markdownTable(pressure_groups.sort_values('repeatability_sd_pct', ascending=False), 12)}

## 全量重拟合 KRR（样本内结果）

- MAE：`{fullfit_summary['mae_pct']:.6g}%`
- 分组通过：`{fullfit_summary['group_pass']}`
- $u_{{nor,L}}$：`{fullfit_summary['u_nor_L_pct']:.6g}%`
- $u_{{nor,r}}$：`{fullfit_summary['u_nor_r_pct']:.6g}%`
- $u_{{nor,d}}$：`{fullfit_summary['u_nor_d_pct']:.6g}%`
- 全量物理基线权重：`{full_weights}`

{markdownTable(fullfit_base_diag, 20)}

## 主要输出

- `code/outputs/stratified_model_search_ranking.csv`
- `code/outputs/stratified_model_search_multisplit_summary.csv`
- `code/outputs/stratified_model_search_champion_splits.csv`
- `code/outputs/stratified_model_search_representative_*.csv`
- `code/outputs/stratified_model_search_pressure_*.csv`
- `code/outputs/physical_base_diagnostics_*.csv`
- `code/outputs/submission_krr_final.csv`
- `code/outputs/krr_final_model.npz`
- `code/outputs/krr_final_model_meta.json`
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def runSearch() -> None:
    print(f"=== Running {SCRIPT_VERSION} ===", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # 必须读取附件1，并合并附件6构造的窗口级稳定性特征。
    # v5 中误用了 load_data()，导致所有多核候选的 raw_stability_only
    # 特征列表为空，同时 compact_plus_raw 候选悄悄退化为仅 compact 特征。
    data = loadDataWithRawFeatures().reset_index(drop=True)

    missing_raw = [
        column for column in RAW_MODEL_FEATURE_COLUMNS
        if column not in data.columns
    ]
    if missing_raw:
        raise RuntimeError(
            "raw stability features were not merged: "
            + ", ".join(missing_raw)
        )

    raw_numeric = (
        data[RAW_MODEL_FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    nonconstant_raw = [
        column for column in RAW_MODEL_FEATURE_COLUMNS
        if raw_numeric[column].notna().any()
        and float(raw_numeric[column].std(ddof=0)) > 1e-12
    ]
    if not nonconstant_raw:
        raise RuntimeError(
            "all raw stability features are missing or constant; "
            "check attachment6/attachment7 paths and chord mapping"
        )

    print(
        "Raw stability features loaded: "
        f"{len(RAW_MODEL_FEATURE_COLUMNS)} columns, "
        f"{len(nonconstant_raw)} non-constant, "
        f"{len(data)} windows",
        flush=True,
    )

    specs = buildSpecs()
    spec_lookup = {spec.model_id: spec for spec in specs}

    probe_features = modelFeatureFrame(data.head(min(len(data), 8)))
    raw_sets = sorted({
        spec.raw_feature_set
        for spec in specs
        if spec.model_type == "multikernel_krr"
    })
    for raw_set in raw_sets:
        raw_columns = columnsForFeatureSet(probe_features, raw_set)
        print(
            f"Raw kernel set {raw_set}: {len(raw_columns)} columns",
            flush=True,
        )

    for spec in specs:
        if spec.model_type == "multikernel_krr":
            physical_columns = columnsForFeatureSet(
                probe_features,
                spec.feature_set,
            )
            raw_columns = columnsForFeatureSet(
                probe_features,
                spec.raw_feature_set,
            )
            if not physical_columns:
                raise RuntimeError(
                    f"{spec.model_id}: physical kernel feature list is empty"
                )
            if not raw_columns:
                raise RuntimeError(
                    f"{spec.model_id}: raw kernel feature list is empty"
                )

    split_specs = [
        SplitSpec(
            seed=RANDOM_SEED + 1000 + i * 17,
            d0_n3_to_validation=bool(i % 3 == 0),
            n34_validation_bias=[0.0, 0.25, 0.5, 1.0][i % 4],
            small_validation_prob=[0.25, 0.40, 0.55, 0.70][(i // 4) % 4],
        )
        for i in range(N_SPLITS)
    ]

    rows: list[dict[str, object]] = []
    split_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for split_no, split_spec in enumerate(split_specs):
        print(f"[split {split_no + 1}/{N_SPLITS}] seed={split_spec.seed}", flush=True)
        train, validation, split = splitAssignments(data, split_spec)
        valid, message = validateSplitRules(data, split)
        if not valid:
            raise RuntimeError(message)
        split_cache[split_no] = (train, validation, split)

        xunlianTezheng = modelFeatureFrame(train)
        for spec in specs:
            try:
                model = fitCandidate(spec, train, xunlianTezheng)
                train_summary, _, _, _ = evaluateModel(model, train)
                validation_summary, _, _, _ = evaluateModel(model, validation)

                validation_fails, validation_gap = standardsGap(validation_summary)
                train_fails, _ = standardsGap(train_summary)
                score = (
                    10000.0 * int(validation_fails)
                    + 500.0 * float(validation_gap)
                    + 30.0 * max(
                        0.0,
                        float(validation_summary["mae_pct"]) - float(train_summary["mae_pct"]),
                    )
                    + float(validation_summary["mae_pct"])
                )

                if model.base_weights is None:
                    base_weight_values = [float("nan")] * len(BASE_COLUMNS)
                else:
                    base_weight_values = model.base_weights.tolist()

                row = {
                    "split_no": split_no,
                    "split_seed": split_spec.seed,
                    "train_count": len(train),
                    "validation_count": len(validation),
                    "model_id": spec.model_id,
                    "family": spec.family,
                    "rank_score": score,
                    "validation_fail_count": int(validation_fails),
                    "validation_gap_score": float(validation_gap),
                    "validation_pass_all": bool(validation_summary["pass_all_standards"]),
                    "validation_mae_pct": float(validation_summary["mae_pct"]),
                    "validation_group_pass": validation_summary["group_pass"],
                    "validation_max_abs_group_mean_pct": float(validation_summary["max_abs_group_mean_pct"]),
                    "validation_max_group_sd_pct": float(validation_summary["max_group_sd_pct"]),
                    "validation_u_nor_L_pct": float(validation_summary["u_nor_L_pct"]),
                    "validation_u_nor_r_pct": float(validation_summary["u_nor_r_pct"]),
                    "validation_u_nor_d_pct": float(validation_summary["u_nor_d_pct"]),
                    "train_fail_count": int(train_fails),
                    "train_mae_pct": float(train_summary["mae_pct"]),
                    **{f"base_weight_{BASE_COLUMNS[i]}": base_weight_values[i] for i in range(len(BASE_COLUMNS))},
                    **{f"spec_{key}": value for key, value in asdict(spec).items()},
                }
                rows.append(row)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                print(f"  [skip] {spec.model_id}: {exc}", flush=True)

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise RuntimeError("no candidate model completed")
    ranking = ranking.sort_values(
        ["validation_fail_count", "rank_score", "validation_mae_pct"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    ranking.to_csv(
        OUT_DIR / "stratified_model_search_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    multisplit = aggregateMultisplit(ranking)
    multisplit.to_csv(
        OUT_DIR / "stratified_model_search_multisplit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    champion = chooseKrrChampion(multisplit)
    champion_id = str(champion["model_id"])
    champion_spec = spec_lookup[champion_id]
    champion_rows = ranking[ranking["model_id"].eq(champion_id)].copy()
    champion_rows.to_csv(
        OUT_DIR / "stratified_model_search_champion_splits.csv",
        index=False,
        encoding="utf-8-sig",
    )

    representative = chooseRepresentativeRow(ranking, champion_id)
    representative_split_no = int(representative["split_no"])
    rep_train, rep_validation, rep_split = split_cache[representative_split_no]
    rep_model = fitCandidate(champion_spec, rep_train, modelFeatureFrame(rep_train))
    rep_summary, rep_groups, rep_windows, rep_diagnostics = evaluateModel(rep_model, rep_validation)
    rep_prediction = rep_diagnostics["model_volume_m3"].to_numpy(dtype=float)
    rep_base_diag = physicalBaseDiagnostics(rep_model, rep_validation, rep_prediction)

    rep_split.to_csv(OUT_DIR / "stratified_model_search_representative_split.csv", index=False, encoding="utf-8-sig")
    rep_groups.to_csv(OUT_DIR / "stratified_model_search_representative_validation_groups.csv", index=False, encoding="utf-8-sig")
    rep_windows.to_csv(OUT_DIR / "stratified_model_search_representative_validation_windows.csv", index=False, encoding="utf-8-sig")
    rep_diagnostics.to_csv(OUT_DIR / "stratified_model_search_representative_diagnostics.csv", index=False, encoding="utf-8-sig")
    rep_base_diag.to_csv(OUT_DIR / "physical_base_diagnostics_representative.csv", index=False, encoding="utf-8-sig")

    pressure_seed = RANDOM_SEED + 424242
    pressure_train, pressure_validation, pressure_assignment = pressureSplit(data, pressure_seed)
    pressure_model = fitCandidate(champion_spec, pressure_train, modelFeatureFrame(pressure_train))
    pressure_summary, pressure_groups, pressure_windows, pressure_diagnostics = evaluateModel(
        pressure_model,
        pressure_validation,
    )
    pressure_prediction = pressure_diagnostics["model_volume_m3"].to_numpy(dtype=float)
    pressure_base_diag = physicalBaseDiagnostics(
        pressure_model,
        pressure_validation,
        pressure_prediction,
    )
    pressure_fails, pressure_gap = standardsGap(pressure_summary)

    pressure_assignment.to_csv(OUT_DIR / "stratified_model_search_pressure_split.csv", index=False, encoding="utf-8-sig")
    pressure_groups.to_csv(OUT_DIR / "stratified_model_search_pressure_groups.csv", index=False, encoding="utf-8-sig")
    pressure_windows.to_csv(OUT_DIR / "stratified_model_search_pressure_windows.csv", index=False, encoding="utf-8-sig")
    pressure_diagnostics.to_csv(OUT_DIR / "stratified_model_search_pressure_diagnostics.csv", index=False, encoding="utf-8-sig")
    pressure_base_diag.to_csv(OUT_DIR / "physical_base_diagnostics_pressure.csv", index=False, encoding="utf-8-sig")

    # 全量重拟合仅用于生成题目给定 159 个窗口的 KRR 结果文件，不替代泛化验证。
    full_features = modelFeatureFrame(data)
    full_model = fitCandidate(champion_spec, data, full_features)
    full_prediction, full_diagnostics = full_model.predictWithDiagnostics(data, full_features)
    full_summary, full_groups, full_windows = evaluatePredictions(data, full_prediction)
    full_base_diag = physicalBaseDiagnostics(full_model, data, full_prediction)

    pd.DataFrame(
        {
            "window_id": data["window_id"],
            "model_volume_m3": full_prediction,
        }
    ).to_csv(
        OUT_DIR / "submission_krr_final.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    pd.DataFrame([full_summary.to_dict()]).to_csv(
        OUT_DIR / "krr_fullfit_evaluation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_groups.to_csv(OUT_DIR / "krr_fullfit_evaluation_groups.csv", index=False, encoding="utf-8-sig")
    full_windows.to_csv(OUT_DIR / "krr_fullfit_evaluation_windows.csv", index=False, encoding="utf-8-sig")
    full_diagnostics.to_csv(OUT_DIR / "krr_fullfit_diagnostics.csv", index=False, encoding="utf-8-sig")
    full_base_diag.to_csv(OUT_DIR / "physical_base_diagnostics_fullfit.csv", index=False, encoding="utf-8-sig")
    saveFittedModel(full_model, OUT_DIR)

    meta = {
        "selection_rule": "best aggregate KRR across 40 label-blind stratified splits",
        "split_group_key": SPLIT_GROUP_KEY,
        "official_repeatability_group_key": OFFICIAL_GROUP_KEY,
        "champion_multisplit": champion.to_dict(),
        "champion_spec": asdict(champion_spec),
        "representative_validation": representative.to_dict(),
        "representative_base_weights": (
            None if rep_model.base_weights is None else rep_model.base_weights.tolist()
        ),
        "pressure_summary": {
            **pressure_summary.to_dict(),
            "standard_fail_count": int(pressure_fails),
            "standard_gap_score": float(pressure_gap),
        },
        "pressure_base_weights": (
            None if pressure_model.base_weights is None else pressure_model.base_weights.tolist()
        ),
        "fullfit_sample_fit_summary": full_summary.to_dict(),
        "fullfit_base_weights": (
            None if full_model.base_weights is None else full_model.base_weights.tolist()
        ),
        "standard_limits": STANDARD_LIMITS,
        "n_splits": N_SPLITS,
        "n_specs": len(specs),
    }
    (OUT_DIR / "stratified_model_search_best_meta.json").write_text(
        json.dumps(toJsonable(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    writeReport(
        ranking=ranking,
        multisplit=multisplit,
        champion=champion,
        champion_spec=champion_spec,
        representative=representative,
        representative_model=rep_model,
        representative_groups=rep_groups,
        representative_base_diag=rep_base_diag,
        pressure_summary=pressure_summary,
        pressure_groups=pressure_groups,
        pressure_base_diag=pressure_base_diag,
        fullfit_summary=full_summary,
        fullfit_base_diag=full_base_diag,
        full_model=full_model,
    )

    print("\n=== KRR multisplit champion ===")
    print(champion.to_string())
    print("\n=== Representative validation ===")
    print(rep_summary.to_string())
    print("\nRepresentative physical-base weights:")
    if rep_model.base_weights is None:
        print("median4")
    else:
        print(dict(zip(BASE_COLUMNS, rep_model.base_weights.tolist())))
    print("\n=== Whole-group pressure test ===")
    print(pressure_summary.to_string())
    print("\n=== Full-fit KRR sample-fit result ===")
    print(full_summary.to_string())
    print("\nFull-fit physical-base weights:")
    if full_model.base_weights is None:
        print("median4")
    else:
        print(dict(zip(BASE_COLUMNS, full_model.base_weights.tolist())))
    print(f"\nSubmission: {OUT_DIR / 'submission_krr_final.csv'}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    runSearch()
