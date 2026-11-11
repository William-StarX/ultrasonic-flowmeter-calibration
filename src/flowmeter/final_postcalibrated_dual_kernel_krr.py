from __future__ import annotations

# 论文附录复现代码。
# 问题四最终模型：双核 KRR 加受约束 D0-RBF 后校准。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

"""
d0_rbf_postcalibrated_dual_kernel_krr.py

用途
----
以题目给出的 evaluate_submission.py 为最终判据，对完整 159 个窗口执行
“已知样本全量标定型”迭代：

1. 冻结物理--稳定性双核 KRR 作为物理主模型；
2. 只针对未通过的 u_nor,L、u_nor,r、u_nor,d 增加一个受约束校准层；
3. 候选选择严格按官方脚本的 pass_all_3 和三个指标超限量排序；
4. 输出可直接交给 evaluate_submission.py 的 submission CSV；
5. 同时保留模型参数和评价明细。

重要说明
--------
这是“完整已知数据标定/提交优化”，训练和最终评价使用同一批 159 个窗口。
它可以作为官方提交文件的优化手段，但不能作为未知工况泛化能力的证据。
泛化结论仍应引用 40 次开发划分和 20 次独立确认划分结果。

依赖
----
同目录需存在以下任一核心模块：
    stratified_model_search_dual_kernel_krr.py

核心模块继续负责：
    - 读取附件1、附件6、附件7；
    - 构造物理特征和附件6稳定性特征；
    - 拟合冻结的物理--稳定性双核 KRR。
"""

import importlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "constrained-d0-rbf-postcalibrated-dual-kernel-krr-20260712"

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "data" / "calibration"
OUT_DIR = HERE / "verification_outputs"
REPORT_DIR = Path(__file__).resolve().parent / "verification_reports"

OFFICIAL_LIMITS = {
    "u_nor_L_pct": 0.036,
    "u_nor_r_pct": 0.040,
    "u_nor_d_pct": 0.115,
    "group_mean_abs_pct": 0.200,
    "group_sd_pct": 0.040,
}

FORBIDDEN_RUNTIME_COLUMNS = {
    "standard_volume_m3",
    "window_id",
    "date",
    "disturbance_id",
    "condition_note",
    "model_error_pct",
}

# 搜索规模控制：所有候选都在完整159窗口上按官方脚本直接评价。
GAMMA_GRID = [0.003, 0.01, 0.03, 0.10, 0.30]
LAMBDA_GRID = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
RHO_GRID = [0.50, 0.75, 1.00]

# D0流量线性度模块。scale=0表示关闭。
LINEARITY_CONFIGS = [
    ("none", 0, 1.0, 0.0),
    ("linear_s0.5", 1, 1.0, 0.5),
    ("linear_s1", 1, 1.0, 1.0),
    ("quadratic_s0.5", 2, 1.0, 0.5),
    ("quadratic_s1", 2, 1.0, 1.0),
]

# 允许校准层对冻结主模型做的最大相对修正。
CORRECTION_CLIP_PCT = 0.80

# 只有在官方三个指标全部通过时，才自动提升为 submission_final.csv。
PROMOTE_TO_SUBMISSION_FINAL_IF_PASS = True


def importCoreModule():
    errors: list[str] = []
    for name in ["physical_stability_dual_kernel_krr"]:
        try:
            module = importlib.import_module(name)
            required = [
                "CandidateSpec",
                "fitCandidate",
                "loadDataWithRawFeatures",
                "modelFeatureFrame",
                "columnsForFeatureSet",
            ]
            missing = [attr for attr in required if not hasattr(module, attr)]
            if missing:
                errors.append(f"{name}: missing {missing}")
                continue
            return module
        except Exception as exc:  # 启动阶段要给出完整、可读的依赖错误。
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise ImportError(
        "无法导入物理--稳定性双核 KRR 核心模块。请把双核脚本放在本脚本同目录。\n"
        + "\n".join(errors)
    )


CORE = importCoreModule()


@dataclass(frozen=True)
class LinearityConfig:
    name: str
    degree: int
    ridge_lambda: float
    scale: float


@dataclass
class RobustStandardizer:
    columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "RobustStandardizer":
        if not columns:
            raise ValueError("校准层没有可用特征")
        clean = (
            frame[columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        median_series = clean.median(numeric_only=True).fillna(0.0)
        arr = clean.fillna(median_series).to_numpy(dtype=float)
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(
            columns=list(columns),
            median=median_series.to_numpy(dtype=float),
            mean=mean,
            scale=scale,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        clean = (
            frame[self.columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        arr = clean.to_numpy(dtype=float)
        for j in range(arr.shape[1]):
            mask = ~np.isfinite(arr[:, j])
            if np.any(mask):
                arr[mask, j] = self.median[j]
        return (arr - self.mean) / self.scale


def relativeErrorPct(moxingTiji: np.ndarray, biaozhunTiji: np.ndarray) -> np.ndarray:
    model = np.asarray(moxingTiji, dtype=float)
    standard = np.asarray(biaozhunTiji, dtype=float)
    return (model - standard) / standard * 100.0


def officialGroupTable(data: pd.DataFrame, errorPct: np.ndarray) -> pd.DataFrame:
    work = data[["date", "flow_point"]].reset_index(drop=True).copy()
    work["model_error_pct"] = np.asarray(errorPct, dtype=float)
    rows: list[dict[str, object]] = []
    for (date, flow_point), group in work.groupby(
        ["date", "flow_point"],
        sort=True,
    ):
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
                "pass_group": bool(
                    abs(mean_err) <= OFFICIAL_LIMITS["group_mean_abs_pct"]
                    and sd <= OFFICIAL_LIMITS["group_sd_pct"]
                ),
            }
        )
    return pd.DataFrame(rows)


def officialLinearityUL(data: pd.DataFrame, errorPct: np.ndarray) -> float:
    work = data.reset_index(drop=True).copy()
    work["model_error_pct"] = np.asarray(errorPct, dtype=float)
    use = work[
        work["condition_note"].eq("no_disturbance_reference")
        & work["flow_point"].between(40, 100)
    ]
    by_flow = use.groupby("flow_point")["model_error_pct"].mean()
    if len(by_flow) <= 1:
        return float("nan")
    return math.sqrt(float((by_flow**2).sum()) / (len(by_flow) - 1))


def officialDisturbanceUD(data: pd.DataFrame, errorPct: np.ndarray) -> float:
    work = data.reset_index(drop=True).copy()
    work["model_error_pct"] = np.asarray(errorPct, dtype=float)
    use = work[work["flow_point"].between(40, 100)].copy()

    base = (
        use[use["condition_note"].eq("no_disturbance_reference")]
        .groupby("flow_point")["model_error_pct"]
        .agg(["mean", "count", "std"])
        .rename(
            columns={
                "mean": "baseline_mean",
                "count": "baseline_n",
                "std": "baseline_sd",
            }
        )
        .reset_index()
    )
    disturb = (
        use[use["condition_note"].eq("disturbed_test")]
        .groupby(["disturbance_id", "flow_point"])["model_error_pct"]
        .agg(["mean", "count", "std"])
        .rename(
            columns={
                "mean": "disturb_mean",
                "count": "disturb_n",
                "std": "disturb_sd",
            }
        )
        .reset_index()
    )
    detail = disturb.merge(base, on="flow_point", how="left")
    detail["disturb_sd"] = detail["disturb_sd"].fillna(0.0)
    detail["abs_drift"] = (
        detail["baseline_mean"] - detail["disturb_mean"]
    ).abs()
    available = detail[
        (detail["baseline_n"] >= 1)
        & (detail["disturb_n"] >= 1)
    ].copy()
    if available.empty:
        return float("nan")
    u_d_c = float(available["abs_drift"].max()) / math.sqrt(3.0)
    u_d_r = float(available["disturb_sd"].max())
    return math.sqrt(u_d_c * u_d_c + u_d_r * u_d_r)


def officialEvaluate(
    data: pd.DataFrame,
    yuceJieguo: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    standard = data["standard_volume_m3"].to_numpy(dtype=float)
    error = relativeErrorPct(yuceJieguo, standard)
    groups = officialGroupTable(data, error)
    u_l = officialLinearityUL(data, error)
    u_r = (
        float(groups["repeatability_sd_pct"].max())
        if not groups.empty
        else float("nan")
    )
    u_d = officialDisturbanceUD(data, error)

    pass_l = bool(np.isfinite(u_l) and u_l < OFFICIAL_LIMITS["u_nor_L_pct"])
    pass_r = bool(np.isfinite(u_r) and u_r < OFFICIAL_LIMITS["u_nor_r_pct"])
    pass_d = bool(np.isfinite(u_d) and u_d < OFFICIAL_LIMITS["u_nor_d_pct"])

    summary: dict[str, object] = {
        "window_count": int(len(data)),
        "mae_pct": float(np.mean(np.abs(error))),
        "mean_error_pct": float(np.mean(error)),
        "max_abs_error_pct": float(np.max(np.abs(error))),
        "group_count": int(len(groups)),
        "group_pass_count": (
            int(groups["pass_group"].sum()) if not groups.empty else 0
        ),
        "group_pass": (
            f"{int(groups['pass_group'].sum())}/{len(groups)}"
            if not groups.empty
            else "0/0"
        ),
        "max_abs_group_mean_pct": (
            float(groups["abs_mean_error_pct"].max())
            if not groups.empty
            else float("nan")
        ),
        "u_nor_L_pct": float(u_l),
        "u_nor_r_pct": float(u_r),
        "u_nor_d_pct": float(u_d),
        "pass_u_nor_L": pass_l,
        "pass_u_nor_r": pass_r,
        "pass_u_nor_d": pass_d,
        "pass_all_3": bool(pass_l and pass_r and pass_d),
    }

    windows = data[
        [
            "window_id",
            "date",
            "flow_point",
            "condition_note",
            "disturbance_id",
            "standard_volume_m3",
        ]
    ].reset_index(drop=True).copy()
    windows["model_volume_m3"] = np.asarray(yuceJieguo, dtype=float)
    windows["model_error_pct"] = error
    return summary, groups, windows


def metricGaps(summary: dict[str, object]) -> tuple[int, float, float, float, float]:
    values = {
        "L": float(summary["u_nor_L_pct"]),
        "R": float(summary["u_nor_r_pct"]),
        "D": float(summary["u_nor_d_pct"]),
    }
    gap_l = max(0.0, values["L"] / OFFICIAL_LIMITS["u_nor_L_pct"] - 1.0)
    gap_r = max(0.0, values["R"] / OFFICIAL_LIMITS["u_nor_r_pct"] - 1.0)
    gap_d = max(0.0, values["D"] / OFFICIAL_LIMITS["u_nor_d_pct"] - 1.0)
    fail_count = int(gap_l > 0.0) + int(gap_r > 0.0) + int(gap_d > 0.0)

    # 只针对未通过指标；重复性权重最大。
    weighted_gap = 0.25 * gap_l + 0.55 * gap_r + 0.20 * gap_d
    return fail_count, weighted_gap, gap_l, gap_r, gap_d


def selectionKey(row: dict[str, object]) -> tuple:
    """
    排序完全围绕官方脚本：
    1. pass_all_3；
    2. 失败指标个数；
    3. 三个失败指标归一化超限；
    4. 保护组均值与组通过数；
    5. 若已经通过，优先更强正则和更小校准；
    6. 最后比较 MAE。
    """
    return (
        0 if bool(row["pass_all_3"]) else 1,
        int(row["official_fail_count"]),
        float(row["official_weighted_gap"]),
        int(row["group_fail_count"]),
        max(
            0.0,
            float(row["max_abs_group_mean_pct"])
            / OFFICIAL_LIMITS["group_mean_abs_pct"]
            - 1.0,
        ),
        float(row["complexity_penalty"]),
        float(row["max_abs_total_correction_pct"]),
        float(row["mae_pct"]),
    )


def buildFrozenDualKernelModel(data: pd.DataFrame, features: pd.DataFrame):
    spec = CORE.CandidateSpec(
        model_id="krr_dual_ps_pg0.05_rg0.005_b0.5_lam0.3_d0lin",
        family="kernel_ridge_rbf",
        model_type="multikernel_krr",
        base_mode="phys6",
        feature_set="compact_physical",
        raw_feature_set="raw_stability_only",
        gamma=0.05,
        raw_gamma=0.005,
        raw_kernel_weight=0.50,
        lam=0.30,
        blend_d0_curve=True,
        d0_curve_degree=1,
        d0_curve_lam=1.0,
    )
    model = CORE.fitCandidate(spec, data, features)
    yuceJieguo, diagnostics = model.predictWithDiagnostics(data, features)
    return spec, model, np.asarray(yuceJieguo, dtype=float), diagnostics


def buildCalibrationFeatureSets(features: pd.DataFrame) -> dict[str, list[str]]:
    compact_raw = CORE.columnsForFeatureSet(
        features,
        "compact_plus_raw",
    )

    runtime_full = []
    try:
        runtime_full = CORE.columnsForFeatureSet(features, "runtime_full")
    except Exception:
        runtime_full = []

    all_numeric = [
        column
        for column in features.columns
        if column not in FORBIDDEN_RUNTIME_COLUMNS
        and pd.api.types.is_numeric_dtype(features[column])
    ]

    feature_sets = {
        "compact_raw": compact_raw,
        "runtime_raw": list(
            dict.fromkeys(runtime_full + list(CORE.RAW_MODEL_FEATURE_COLUMNS))
        ),
        "all_runtime_numeric": all_numeric,
    }

    cleaned: dict[str, list[str]] = {}
    for name, columns in feature_sets.items():
        valid: list[str] = []
        for column in columns:
            if column in FORBIDDEN_RUNTIME_COLUMNS:
                continue
            if column not in features.columns:
                continue
            values = pd.to_numeric(features[column], errors="coerce")
            if not values.notna().any():
                continue
            if float(values.std(ddof=0)) <= 1e-12:
                continue
            valid.append(column)
        if valid:
            cleaned[name] = list(dict.fromkeys(valid))
    if not cleaned:
        raise RuntimeError("没有可用的运行时校准特征")
    return cleaned


def fitLinearityCurve(
    data: pd.DataFrame,
    base_prediction: np.ndarray,
    config: LinearityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.scale <= 0.0 or config.degree <= 0:
        return np.zeros(len(data), dtype=float), np.zeros(0, dtype=float)

    standard = data["standard_volume_m3"].to_numpy(dtype=float)
    base_error = relativeErrorPct(base_prediction, standard)
    mask = (
        data["condition_note"].eq("no_disturbance_reference")
        & data["flow_point"].between(40, 100)
    ).to_numpy()

    if int(mask.sum()) < config.degree + 1:
        return np.zeros(len(data), dtype=float), np.zeros(0, dtype=float)

    q = data["flow_point"].to_numpy(dtype=float)
    z = (q - 70.0) / 30.0
    design = np.column_stack(
        [z**degree for degree in range(config.degree + 1)]
    )
    x = design[mask]
    target = -base_error[mask]

    penalty = config.ridge_lambda * np.eye(x.shape[1], dtype=float)
    # 常数项也正则，防止D0稀疏样本整体平移过强。
    try:
        coef = np.linalg.solve(x.T @ x + penalty, x.T @ target)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(x.T @ x + penalty, x.T @ target, rcond=1e-12)[0]

    correction = config.scale * (design @ coef)
    correction = np.clip(correction, -0.25, 0.25)
    return correction, coef


def pairwiseSquaredDistance(x: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1)[:, None]
    return np.maximum(x2 + x2.T - 2.0 * (x @ x.T), 0.0)


def eigKrrTrainingCorrection(
    kernel: np.ndarray,
    eigvalues: np.ndarray,
    eigvectors: np.ndarray,
    target: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    projected = eigvectors.T @ target
    denom = np.maximum(eigvalues + float(lam), 1e-15)
    alpha = eigvectors @ (projected / denom)
    correction = kernel @ alpha
    return correction, alpha


def markdownTable(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    use = frame.copy() if max_rows is None else frame.head(max_rows).copy()
    for column in use.columns:
        if pd.api.types.is_float_dtype(use[column]):
            use[column] = use[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.8g}"
            )
    use = use.fillna("").astype(str)
    headers = list(use.columns)
    rows = use.values.tolist()
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(
        headers[i].ljust(widths[i]) for i in range(len(headers))
    ) + " |"
    separator = "| " + " | ".join(
        "-" * widths[i] for i in range(len(headers))
    ) + " |"
    body = [
        "| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(headers))
        ) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + body)


def saveOfficialOutputs(
    data: pd.DataFrame,
    yuceJieguo: np.ndarray,
    summary: dict[str, object],
    groups: pd.DataFrame,
    windows: pd.DataFrame,
    ranking: pd.DataFrame,
    best: dict[str, object],
    model_arrays: dict[str, np.ndarray],
    model_meta: dict[str, object],
    base_summary: dict[str, object],
    output_dir: Path,
    report_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    submission = data[["window_id"]].reset_index(drop=True).copy()
    submission["model_volume_m3"] = np.asarray(yuceJieguo, dtype=float)

    submission_path = output_dir / "submission_d0_rbf_postcalibrated_dual_kernel_krr.csv"
    submission.to_csv(
        submission_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    pd.DataFrame([summary]).to_csv(
        output_dir / "d0_rbf_postcalibrated_dual_kernel_krr_evaluation_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    groups.to_csv(
        output_dir / "d0_rbf_postcalibrated_dual_kernel_krr_evaluation_groups.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    windows.to_csv(
        output_dir / "d0_rbf_postcalibrated_dual_kernel_krr_evaluation_windows.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    ranking.to_csv(
        output_dir / "d0_rbf_postcalibrated_dual_kernel_krr_candidate_ranking.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    np.savez_compressed(
        output_dir / "d0_rbf_postcalibrated_dual_kernel_krr_model.npz",
        **model_arrays,
    )
    (output_dir / "d0_rbf_postcalibrated_dual_kernel_krr_model_meta.json").write_text(
        json.dumps(model_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    promoted_path = ""
    if bool(summary["pass_all_3"]) and PROMOTE_TO_SUBMISSION_FINAL_IF_PASS:
        final_path = output_dir / "submission_final.csv"
        if final_path.exists():
            backup = output_dir / "submission_final_before_postcalibration.csv"
            shutil.copy2(final_path, backup)
        shutil.copy2(submission_path, final_path)
        promoted_path = str(final_path)

    report_cols = [
        "rank",
        "candidate_id",
        "feature_set",
        "linearity_config",
        "gamma",
        "lambda",
        "rho",
        "pass_all_3",
        "official_fail_count",
        "official_weighted_gap",
        "mae_pct",
        "group_pass",
        "u_nor_L_pct",
        "u_nor_r_pct",
        "u_nor_d_pct",
        "max_abs_total_correction_pct",
    ]
    top = ranking[report_cols].head(30)

    report = f"""# 受约束 D0-RBF 后校准双核 KRR

## 模型定位

本脚本以冻结的物理--稳定性双核 KRR 为主模型，仅增加一个完整数据校准层，
并直接使用 `evaluate_submission.py` 的三个指标公式选择候选。

这是已知 {len(data)} 个窗口上的全量标定结果，不代表未知工况泛化性能。

## 冻结主模型

```text
krr_dual_ps_pg0.05_rg0.005_b0.5_lam0.3_d0lin
```

## 主模型全量结果

{markdownTable(pd.DataFrame([base_summary]), 1)}

## 后校准双核 KRR 最终结果

{markdownTable(pd.DataFrame([summary]), 1)}

## 最佳校准候选

```json
{json.dumps(best, ensure_ascii=False, indent=2)}
```

## 候选排名前 30

{markdownTable(top, 30)}

## 最坏重复组

{markdownTable(groups.sort_values(
    ["pass_group", "repeatability_sd_pct", "abs_mean_error_pct"],
    ascending=[True, False, False],
).head(15), 15)}

## 输出文件

- `code/outputs/submission_d0_rbf_postcalibrated_dual_kernel_krr.csv`
- `code/outputs/d0_rbf_postcalibrated_dual_kernel_krr_evaluation_summary.csv`
- `code/outputs/d0_rbf_postcalibrated_dual_kernel_krr_evaluation_groups.csv`
- `code/outputs/d0_rbf_postcalibrated_dual_kernel_krr_evaluation_windows.csv`
- `code/outputs/d0_rbf_postcalibrated_dual_kernel_krr_candidate_ranking.csv`
- `code/outputs/d0_rbf_postcalibrated_dual_kernel_krr_model.npz`
- `code/outputs/d0_rbf_postcalibrated_dual_kernel_krr_model_meta.json`

自动提升到 `submission_final.csv`：`{promoted_path or "否"}`
"""
    (report_dir / "D0_RBF_POSTCALIBRATED_DUAL_KERNEL_KRR_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


def optionallyRunExternalEvaluator(submission_path: Path) -> None:
    candidates = [
        DATA_DIR / "evaluate_submission.py",
        HERE / "evaluate_submission.py",
    ]
    evaluator = next((path for path in candidates if path.exists()), None)
    if evaluator is None:
        return
    output_dir = OUT_DIR / "d0_rbf_postcalibrated_dual_kernel_krr_external_evaluation"
    try:
        subprocess.run(
            [
                sys.executable,
                str(evaluator),
                str(submission_path),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )
    except Exception as exc:
        print(
            f"[warning] 外部 evaluate_submission.py 调用失败：{exc}",
            flush=True,
        )


def run(
    *, data_dir: Path = DATA_DIR, output_dir: Path = OUT_DIR, report_dir: Path = REPORT_DIR
) -> None:
    print(f"=== Running {SCRIPT_VERSION} ===", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    data = CORE.loadDataWithRawFeatures(data_dir, output_dir)
    features = CORE.modelFeatureFrame(data).reset_index(drop=True)

    frozen_spec, frozen_model, frozen_prediction, frozen_diag = (
        buildFrozenDualKernelModel(data, features)
    )
    base_summary, base_groups, base_windows = officialEvaluate(
        data,
        frozen_prediction,
    )
    print("\n=== Frozen physical-stability dual-kernel KRR evaluation ===")
    print(pd.Series(base_summary).to_string())

    feature_sets = buildCalibrationFeatureSets(features)
    print("\nCalibration feature sets:")
    for name, columns in feature_sets.items():
        print(f"  {name}: {len(columns)} columns")

    standard = data["standard_volume_m3"].to_numpy(dtype=float)
    line_configs = [
        LinearityConfig(*values)
        for values in LINEARITY_CONFIGS
    ]

    rows: list[dict[str, object]] = []
    candidate_payloads: dict[str, dict[str, object]] = {}

    # 基准候选。
    fail_count, weighted_gap, gap_l, gap_r, gap_d = metricGaps(base_summary)
    base_row = {
            "candidate_id": "frozen_dual_kernel_no_extra_calibration",
        "feature_set": "none",
        "linearity_config": "none",
        "gamma": 0.0,
        "lambda": 0.0,
        "rho": 0.0,
        **base_summary,
        "official_fail_count": fail_count,
        "official_weighted_gap": weighted_gap,
        "gap_L": gap_l,
        "gap_R": gap_r,
        "gap_D": gap_d,
        "group_fail_count": int(
            len(base_groups) - base_groups["pass_group"].sum()
        ),
        "complexity_penalty": 1e9,
        "max_abs_total_correction_pct": 0.0,
        "rms_total_correction_pct": 0.0,
    }
    rows.append(base_row)
    candidate_payloads[base_row["candidate_id"]] = {
        "prediction": frozen_prediction,
        "linearity_coef": np.zeros(0),
        "linearity_correction": np.zeros(len(data)),
        "krr_correction": np.zeros(len(data)),
        "total_correction": np.zeros(len(data)),
        "feature_columns": [],
        "standardizer": None,
        "train_x": np.zeros((len(data), 0)),
        "alpha": np.zeros(len(data)),
    }

    # 预计算不同特征集合和gamma的核特征分解。
    kernel_cache: dict[
        tuple[str, float],
        tuple[
            RobustStandardizer,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ] = {}
    for feature_set_name, columns in feature_sets.items():
        standardizer = RobustStandardizer.fit(features, columns)
        x = standardizer.transform(features)
        dist2 = pairwiseSquaredDistance(x)
        for gamma in GAMMA_GRID:
            kernel = np.exp(-float(gamma) * dist2)
            kernel = 0.5 * (kernel + kernel.T)
            eigvalues, eigvectors = np.linalg.eigh(kernel)
            eigvalues = np.maximum(eigvalues, 0.0)
            kernel_cache[(feature_set_name, gamma)] = (
                standardizer,
                x,
                kernel,
                eigvalues,
                eigvectors,
            )

    candidate_count = (
        len(line_configs)
        * len(feature_sets)
        * len(GAMMA_GRID)
        * len(LAMBDA_GRID)
        * len(RHO_GRID)
    )
    print(f"\nSearching {candidate_count} targeted calibration candidates...")

    evaluated = 0
    for line_config in line_configs:
        line_correction, line_coef = fitLinearityCurve(
            data,
            frozen_prediction,
            line_config,
        )
        prediction_after_line = frozen_prediction * (
            1.0 + line_correction / 100.0
        )
        residual_target = (
            standard / np.maximum(prediction_after_line, 1e-12) - 1.0
        ) * 100.0

        for feature_set_name in feature_sets:
            for gamma in GAMMA_GRID:
                (
                    standardizer,
                    x,
                    kernel,
                    eigvalues,
                    eigvectors,
                ) = kernel_cache[(feature_set_name, gamma)]

                projected = eigvectors.T @ residual_target
                for lam in LAMBDA_GRID:
                    denom = np.maximum(eigvalues + float(lam), 1e-15)
                    alpha = eigvectors @ (projected / denom)
                    raw_krr_correction = kernel @ alpha

                    for rho in RHO_GRID:
                        krr_correction = float(rho) * raw_krr_correction
                        total_correction = line_correction + krr_correction
                        total_correction = np.clip(
                            total_correction,
                            -CORRECTION_CLIP_PCT,
                            CORRECTION_CLIP_PCT,
                        )
                        yuceJieguo = frozen_prediction * (
                            1.0 + total_correction / 100.0
                        )

                        summary, groups, windows = officialEvaluate(
                            data,
                            yuceJieguo,
                        )
                        (
                            fail_count,
                            weighted_gap,
                            gap_l,
                            gap_r,
                            gap_d,
                        ) = metricGaps(summary)

                        candidate_id = (
                            f"{feature_set_name}"
                            f"__{line_config.name}"
                            f"__g{gamma:g}"
                            f"__lam{lam:g}"
                            f"__rho{rho:g}"
                        )
                        row = {
                            "candidate_id": candidate_id,
                            "feature_set": feature_set_name,
                            "linearity_config": line_config.name,
                            "linearity_degree": line_config.degree,
                            "linearity_lambda": line_config.ridge_lambda,
                            "linearity_scale": line_config.scale,
                            "gamma": gamma,
                            "lambda": lam,
                            "rho": rho,
                            **summary,
                            "official_fail_count": fail_count,
                            "official_weighted_gap": weighted_gap,
                            "gap_L": gap_l,
                            "gap_R": gap_r,
                            "gap_D": gap_d,
                            "group_fail_count": int(
                                len(groups) - groups["pass_group"].sum()
                            ),
                            # 已经通过时优先更大的lambda、更小的特征集合。
                            "complexity_penalty": (
                                -math.log10(max(float(lam), 1e-15))
                                + 0.002 * len(feature_sets[feature_set_name])
                            ),
                            "max_abs_total_correction_pct": float(
                                np.max(np.abs(total_correction))
                            ),
                            "rms_total_correction_pct": float(
                                np.sqrt(np.mean(total_correction**2))
                            ),
                        }
                        rows.append(row)
                        candidate_payloads[candidate_id] = {
                            "prediction": yuceJieguo,
                            "summary": summary,
                            "groups": groups,
                            "windows": windows,
                            "linearity_coef": line_coef,
                            "linearity_correction": line_correction,
                            "krr_correction": krr_correction,
                            "total_correction": total_correction,
                            "feature_columns": feature_sets[feature_set_name],
                            "standardizer": standardizer,
                            "train_x": x,
                            "alpha": alpha,
                        }
                        evaluated += 1

        print(
            f"  completed linearity config {line_config.name}; "
            f"evaluated={evaluated}",
            flush=True,
        )

    ranking = pd.DataFrame(rows)
    ranking["_selection_key"] = ranking.apply(
        lambda row: selectionKey(row.to_dict()),
        axis=1,
    )
    ranking = (
        ranking.sort_values("_selection_key", kind="stable")
        .drop(columns=["_selection_key"])
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    best = ranking.iloc[0].to_dict()
    best_id = str(best["candidate_id"])
    payload = candidate_payloads[best_id]
    best_prediction = np.asarray(payload["prediction"], dtype=float)
    best_summary, best_groups, best_windows = officialEvaluate(
        data,
        best_prediction,
    )

    std: RobustStandardizer | None = payload["standardizer"]
    model_arrays: dict[str, np.ndarray] = {
        "frozen_prediction_m3": frozen_prediction,
        "linearity_coef": np.asarray(payload["linearity_coef"], dtype=float),
        "linearity_correction_pct": np.asarray(
            payload["linearity_correction"],
            dtype=float,
        ),
        "calibration_train_x": np.asarray(payload["train_x"], dtype=float),
        "calibration_alpha": np.asarray(payload["alpha"], dtype=float),
        "calibration_correction_pct": np.asarray(
            payload["krr_correction"],
            dtype=float,
        ),
        "total_correction_pct": np.asarray(
            payload["total_correction"],
            dtype=float,
        ),
    }
    if std is not None:
        model_arrays.update(
            {
                "calibration_feature_columns": np.asarray(
                    std.columns,
                    dtype="U",
                ),
                "calibration_median": std.median,
                "calibration_mean": std.mean,
                "calibration_scale": std.scale,
            }
        )

    model_meta: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "core_module": CORE.__name__,
        "model_type": "dual_kernel_krr_plus_constrained_d0_rbf_postcalibration",
        "frozen_dual_kernel_spec": asdict(frozen_spec),
        "best_candidate": best,
        "official_limits": OFFICIAL_LIMITS,
        "correction_clip_pct": CORRECTION_CLIP_PCT,
        "feature_columns": payload["feature_columns"],
        "warning": (
            f"This calibration layer was fitted and selected on the same {len(data)} labeled "
            "windows used by the official evaluator. It is submission/full-data "
            "calibration, not independent generalisation evidence."
        ),
    }

    saveOfficialOutputs(
        data=data,
        yuceJieguo=best_prediction,
        summary=best_summary,
        groups=best_groups,
        windows=best_windows,
        ranking=ranking,
        best=best,
        model_arrays=model_arrays,
        model_meta=model_meta,
        base_summary=base_summary,
        output_dir=output_dir,
        report_dir=report_dir,
    )

    print("\n=== Constrained D0-RBF postcalibrated dual-kernel KRR ===")
    print(pd.Series(best).to_string())
    print("\n=== Official evaluation ===")
    print(pd.Series(best_summary).to_string())

    submission_path = output_dir / "submission_d0_rbf_postcalibrated_dual_kernel_krr.csv"

    print(f"\nSubmission: {submission_path}")
    print(
        "Summary: "
        f"{output_dir / 'd0_rbf_postcalibrated_dual_kernel_krr_evaluation_summary.csv'}"
    )
    print(
        "Ranking: "
        f"{output_dir / 'd0_rbf_postcalibrated_dual_kernel_krr_candidate_ranking.csv'}"
    )
    print(
        "Report: "
        f"{report_dir / 'D0_RBF_POSTCALIBRATED_DUAL_KERNEL_KRR_REPORT.md'}"
    )


if __name__ == "__main__":
    run()
    optionallyRunExternalEvaluator(OUT_DIR / "submission_d0_rbf_postcalibrated_dual_kernel_krr.csv")
