from __future__ import annotations

# 论文附录复现代码。
# 问题四对照：在同一标签盲划分上比较 ANN 与 KRR。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

from pathlib import Path

import numpy as np
import pandas as pd

from model_evolution_ridge_ann import (
    ModelSpec,
    buildFeatureFrame,
    evaluatePredictions,
    featureColumns,
    fitModel,
    loadData,
    standardsGap,
)
from physical_feature_library import enhancedFeatureFrame
from compact_rbf_krr import CandidateSpec, columnsForFeatureSet, fitCandidate


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "verification_outputs"
REPORT = Path(__file__).resolve().parent / "verification_reports" / "ANN_KRR_SAME_SPLIT_COMPARISON.md"
BEST_SPLIT = OUT_DIR / "stratified_model_search_best_split.csv"


ANN_SPEC = ModelSpec(
    model_id="ann_lagrange_volume_m3_runtime_full_h16_l20.001_lr0.01_s0_shrink1",
    family="ann_runtime_full",
    base_col="lagrange_volume_m3",
    feature_set="runtime_full",
    model_type="mlp",
    hidden_size=16,
    l2=0.001,
    learning_rate=0.01,
    seed_offset=0,
    shrinkage=1.0,
    epochs=10000,
)

KRR_SPEC = CandidateSpec(
    model_id="krr_phys6_volume_m3_compact_physical_g0.1_lam0.1_none",
    family="kernel_ridge_rbf",
    base_col="phys6_volume_m3",
    feature_set="compact_physical",
    model_type="krr_rbf",
    lam=0.1,
    gamma=0.1,
    anchor="none",
    blend_d0_curve=True,
)


def splitFromBestFile(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = pd.read_csv(BEST_SPLIT, encoding="utf-8-sig")
    assignment = split.set_index("window_id")["split"]
    tagged = data.copy()
    tagged["_split"] = tagged["window_id"].map(assignment)
    if tagged["_split"].isna().any():
        missing = tagged.loc[tagged["_split"].isna(), "window_id"].tolist()
        raise ValueError(f"missing split assignments for window_id={missing[:5]}")
    train = tagged[tagged["_split"].eq("train")].drop(columns="_split").reset_index(drop=True)
    validation = tagged[tagged["_split"].eq("validation")].drop(columns="_split").reset_index(drop=True)
    return train, validation, split


def addMetricsRow(
    rows: list[dict],
    model_label: str,
    model_id: str,
    operation: str,
    summary: pd.Series,
    feature_count: int,
    effective_parameter_count: int,
) -> None:
    fails, gap = standardsGap(summary)
    rows.append(
        {
            "model_label": model_label,
            "model_id": model_id,
            "operation": operation,
            "window_count": int(summary["window_count"]),
            "mae_pct": float(summary["mae_pct"]),
            "mean_error_pct": float(summary["mean_error_pct"]),
            "group_pass": str(summary["group_pass"]),
            "max_abs_group_mean_pct": float(summary["max_abs_group_mean_pct"]),
            "max_group_sd_pct": float(summary["max_group_sd_pct"]),
            "u_nor_L_pct": float(summary["u_nor_L_pct"]),
            "u_nor_r_pct": float(summary["u_nor_r_pct"]),
            "u_nor_d_pct": float(summary["u_nor_d_pct"]),
            "pass_all_standards": bool(summary["pass_all_standards"]),
            "standard_fail_count": int(fails),
            "standard_gap_score": float(gap),
            "feature_count": int(feature_count),
            "effective_parameter_count": int(effective_parameter_count),
        }
    )


def markdownTable(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    use = df.copy()
    for col in use.columns:
        if pd.api.types.is_float_dtype(use[col]):
            use[col] = use[col].map(lambda x: f"{x:.6f}")
    use = use.astype(str)
    headers = list(use.columns)
    rows = use.values.tolist()
    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def writeReport(metrics: pd.DataFrame, ann_feature_count: int, ann_parameter_count: int, krr_feature_count: int) -> None:
    validation = metrics[metrics["operation"].eq("validation")].copy()
    display_cols = [
        "model_label",
        "mae_pct",
        "group_pass",
        "max_abs_group_mean_pct",
        "u_nor_L_pct",
        "u_nor_r_pct",
        "u_nor_d_pct",
        "standard_fail_count",
        "pass_all_standards",
    ]
    text = f"""# ANN 与最终 KRR 同划分对比

本报告使用 `stratified_model_search_best_split.csv` 中的冠军 KRR 主分层划分，训练集 `118` 个窗口，验证集 `41` 个窗口。ANN 与 KRR 均只在训练集拟合，并用同一个评价函数计算验证集指标。

## 模型设置

- ANN 对照模型：`{ANN_SPEC.model_id}`，物理基线为 `Lagrange`，输入为 `runtime_full`，一隐层 `h=16`，`L2=0.001`，学习率 `0.01`，训练 `{ANN_SPEC.epochs}` 轮，学习相对残差。
- 最终 KRR 模型：`{KRR_SPEC.model_id}`，物理基线为 `Phys6 + D0 curve`，输入为 `compact_physical`，`gamma=0.1`，`lambda=0.1`，学习相对残差。

ANN 输入特征数为 `{ann_feature_count}`，可训练参数数约为

```text
P = (d + 1) h + (h + 1) = {ann_parameter_count}
```

KRR 输入特征数为 `{krr_feature_count}`，训练时保存 `118` 个支持样本系数；它仍是带正则化的凸优化问题，给定 `gamma` 和 `lambda` 后解确定。

## 主分层验证结果

{markdownTable(validation[display_cols])}

## 结论

ANN 的验证 MAE 低于固定积分和线性 Ridge，说明多声道运行时变量与残差之间确实存在可学习的非线性关系。但在同一主分层验证下，ANN 仍有多个工程指标未通过，尤其是组内重复性和扰流漂移相关指标。其原因主要是样本量只有 159 个窗口，而 `runtime_full` 与隐藏层参数带来较高自由度，窗口级损失不能直接约束最坏组 `SD_g`、`u_nor,r` 和 `u_nor,d`。

因此，ANN 在论文中适合作为复杂非线性补偿对照：它证明“需要非线性”，但也暴露“小样本高维黑箱模型不够稳”。最终模型转向 `Phys6 + D0 curve + compact_physical + RBF-KRR`，是在保留非线性能力的同时降低输入维度、增强物理解释和重复性稳定。
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    data = loadData()
    train, validation, split = splitFromBestFile(data)

    rows: list[dict] = []

    ann_train_features = buildFeatureFrame(train)
    ann_val_features = buildFeatureFrame(validation)
    ann_feature_cols = featureColumns(ann_train_features, ANN_SPEC.feature_set)
    ann_feature_count = len(ann_feature_cols)
    ann_parameter_count = (ann_feature_count + 1) * ANN_SPEC.hidden_size + (ANN_SPEC.hidden_size + 1)
    ann_model = fitModel(ANN_SPEC, train, ann_train_features)
    for operation, frame, features in [
        ("train", train, ann_train_features),
        ("validation", validation, ann_val_features),
    ]:
        pred = ann_model.predictVolume(frame, features)
        summary, groups, windows = evaluatePredictions(frame, pred)
        addMetricsRow(
            rows,
            "ANN: Lagrange + runtime_full",
            ANN_SPEC.model_id,
            operation,
            summary,
            ann_feature_count,
            ann_parameter_count,
        )
        if operation == "validation":
            groups.to_csv(OUT_DIR / "ann_same_split_validation_groups.csv", index=False, encoding="utf-8-sig")
            windows.to_csv(OUT_DIR / "ann_same_split_validation_windows.csv", index=False, encoding="utf-8-sig")

    krr_train_features = enhancedFeatureFrame(train)
    krr_val_features = enhancedFeatureFrame(validation)
    krr_feature_count = len(columnsForFeatureSet(krr_train_features, KRR_SPEC.feature_set))
    krr_model = fitCandidate(KRR_SPEC, train, krr_train_features)
    for operation, frame, features in [
        ("train", train, krr_train_features),
        ("validation", validation, krr_val_features),
    ]:
        pred = krr_model.predictVolume(frame, features)
        summary, groups, windows = evaluatePredictions(frame, pred)
        addMetricsRow(
            rows,
            "KRR: Phys6 + D0 + compact",
            KRR_SPEC.model_id,
            operation,
            summary,
            krr_feature_count,
            len(train),
        )
        if operation == "validation":
            groups.to_csv(OUT_DIR / "krr_same_split_validation_groups.csv", index=False, encoding="utf-8-sig")
            windows.to_csv(OUT_DIR / "krr_same_split_validation_windows.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "ann_krr_same_split_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    split.to_csv(OUT_DIR / "ann_krr_same_split_assignments.csv", index=False, encoding="utf-8-sig")
    writeReport(metrics, ann_feature_count, ann_parameter_count, krr_feature_count)

    print(metrics[metrics["operation"].eq("validation")].to_string(index=False))
    print("Report:", REPORT)


if __name__ == "__main__":
    main()
