from __future__ import annotations

# 论文附录复现代码。
# 问题一：重建五声道观测并比较固定权重积分。
# 命名与注释已统一整理；计算公式、数据字段和模型参数保持不变。

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

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
REPORT = Path(__file__).resolve().parent / "verification_reports" / "INTEGRATION_METHOD_ANALYSIS.md"


def loadInputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    win = pd.read_csv(DATA_DIR / "attachment1_window_data.csv", encoding="utf-8-sig")
    raw = pd.read_csv(DATA_DIR / "attachment6_window_raw_samples.csv", encoding="utf-8-sig")
    geo = pd.read_csv(DATA_DIR / "attachment7_meter_geometry.csv", encoding="utf-8-sig")
    return win, raw, geo


def reconstructChordsFromRaw(raw: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    rows = geo[geo["record_type"].eq("chord_mapping")].copy()
    diff_cols = [f"diff_ns_{i}" for i in range(10)]
    raw_mean = raw.groupby("window_id")[diff_cols].mean()
    rec = pd.DataFrame(index=raw_mean.index)
    for _, row in rows.sort_values("chord_id").iterrows():
        chord = row["object_id"]
        a_col = row["raw_diff_col_a"]
        b_col = row["raw_diff_col_b"]
        rec[chord] = (
            raw_mean[a_col] * float(row["raw_diff_gain_a_mps_per_ns"])
            + raw_mean[b_col] * float(row["raw_diff_gain_b_mps_per_ns"])
            + float(row["raw_diff_intercept_mps"])
        )
    return rec.reset_index()


def getArea(geo: pd.DataFrame) -> float:
    return float(geo.loc[geo["parameter"].eq("effective_pipe_area_m2"), "value"].iloc[0])


def getWeights(geo: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "object_id",
        "chord_id",
        "chord_order",
        "chord_position_label",
        "normalized_channel_height",
        "weight_phys6",
        "weight_owics",
        "weight_flowcal_gauss_jacobi",
        "weight_lagrange",
        "weight_equal",
    ]
    rows = geo[geo["record_type"].eq("chord_mapping")][cols].copy()
    return rows.sort_values("chord_id").reset_index(drop=True)


def addEstimate(est: pd.DataFrame, method: str, volume: pd.Series, source: str) -> pd.DataFrame:
    tmp = est[[
        "window_id",
        "date",
        "flow_point",
        "condition_note",
        "is_disturbed",
        "disturbance_id",
        "duration_s",
        "standard_volume_m3",
    ]].copy()
    tmp["method"] = method
    tmp["source"] = source
    tmp["estimate_volume_m3"] = volume.to_numpy(dtype=float)
    tmp["error_pct"] = (tmp["estimate_volume_m3"] - tmp["standard_volume_m3"]) / tmp["standard_volume_m3"] * 100.0
    return tmp


def buildEstimates(win: pd.DataFrame, rec_chords: pd.DataFrame, geo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    area = getArea(geo)
    weights = getWeights(geo)
    merged = win.merge(rec_chords, on="window_id", suffixes=("", "_raw_rec"))
    estimates: list[pd.DataFrame] = []

    # Raw single-chord estimates: useful as a lower-level diagnostic, not a recommended meter output.
    for i in range(5):
        volume = area * merged["duration_s"] * merged[f"chord{i}_raw_rec"]
        estimates.append(addEstimate(merged, f"raw_single_chord{i}", volume, "attachment6_raw"))

    weight_methods = {
        "raw_phys6_weighted": "weight_phys6",
        "raw_owics_weighted": "weight_owics",
        "raw_gauss_jacobi_weighted": "weight_flowcal_gauss_jacobi",
        "raw_lagrange_weighted": "weight_lagrange",
        "raw_equal_weighted": "weight_equal",
    }
    for method, w_col in weight_methods.items():
        volume = pd.Series(0.0, index=merged.index)
        for i, w in enumerate(weights[w_col].astype(float).to_list()):
            volume = volume + area * merged["duration_s"] * w * merged[f"chord{i}_raw_rec"]
        estimates.append(addEstimate(merged, method, volume, "attachment6_raw_plus_attachment7_weights"))

    baseline_methods = {
        "base_phys6": "phys6_volume_m3",
        "base_owics": "owics_volume_m3",
        "base_lagrange": "lagrange_volume_m3",
        "base_equal_weight": "equal_weight_volume_m3",
    }
    for method, col in baseline_methods.items():
        estimates.append(addEstimate(merged, method, merged[col], "attachment1_baseline"))

    return pd.concat(estimates, ignore_index=True), weights


def summarize(group: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def aggOne(x: pd.DataFrame) -> pd.Series:
        e = x["error_pct"].astype(float)
        return pd.Series(
            {
                "window_count": len(x),
                "mae_pct": e.abs().mean(),
                "mean_error_pct": e.mean(),
                "rmse_pct": float(np.sqrt(np.mean(e**2))),
                "max_abs_error_pct": e.abs().max(),
            }
        )

    return group.groupby(by, dropna=False).apply(aggOne).reset_index()


def writeOutputs(estimates: pd.DataFrame, weights: pd.DataFrame, rec_chords: pd.DataFrame, win: pd.DataFrame) -> dict[str, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rec_check = win[["window_id", "chord0", "chord1", "chord2", "chord3", "chord4"]].merge(rec_chords, on="window_id", suffixes=("_attachment1", "_raw_rec"))
    check_rows = []
    for i in range(5):
        err = rec_check[f"chord{i}_raw_rec"] - rec_check[f"chord{i}_attachment1"]
        check_rows.append(
            {
                "chord": f"chord{i}",
                "mae_mps": err.abs().mean(),
                "mean_error_mps": err.mean(),
                "max_abs_error_mps": err.abs().max(),
            }
        )
    chord_check = pd.DataFrame(check_rows)

    overall = summarize(estimates, ["method", "source"]).sort_values("mae_pct")
    by_flow = summarize(estimates, ["method", "source", "flow_point"]).sort_values(["method", "flow_point"])
    by_condition = summarize(estimates, ["method", "source", "condition_note"]).sort_values(["method", "condition_note"])
    by_disturbed = summarize(estimates, ["method", "source", "is_disturbed"]).sort_values(["method", "is_disturbed"])

    weights.to_csv(OUT_DIR / "integration_weights.csv", index=False, encoding="utf-8-sig")
    rec_chords.to_csv(OUT_DIR / "reconstructed_raw_chords.csv", index=False, encoding="utf-8-sig")
    chord_check.to_csv(OUT_DIR / "raw_chord_reconstruction_check.csv", index=False, encoding="utf-8-sig")
    estimates.to_csv(OUT_DIR / "integration_estimates_long.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUT_DIR / "integration_overall_metrics.csv", index=False, encoding="utf-8-sig")
    by_flow.to_csv(OUT_DIR / "integration_metrics_by_flow_point.csv", index=False, encoding="utf-8-sig")
    by_condition.to_csv(OUT_DIR / "integration_metrics_by_condition.csv", index=False, encoding="utf-8-sig")
    by_disturbed.to_csv(OUT_DIR / "integration_metrics_by_disturbed.csv", index=False, encoding="utf-8-sig")
    return {
        "overall": overall,
        "by_flow": by_flow,
        "by_condition": by_condition,
        "by_disturbed": by_disturbed,
        "chord_check": chord_check,
    }


def plotFigures(tables: dict[str, pd.DataFrame]) -> None:
    if plt is None:
        plotFiguresReportlab(tables)
        return

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    selected = [
        "base_phys6",
        "base_owics",
        "base_lagrange",
        "base_equal_weight",
        "raw_gauss_jacobi_weighted",
        "raw_lagrange_weighted",
        "raw_equal_weighted",
    ]
    overall = tables["overall"][tables["overall"]["method"].isin(selected)].copy()
    overall = overall.sort_values("mae_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(overall["method"], overall["mae_pct"], color="#3B82F6")
    ax.set_xlabel("MAE / %")
    ax.set_ylabel("method")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_integration_overall_mae.pdf")
    plt.close(fig)

    flow = tables["by_flow"][tables["by_flow"]["method"].isin(["base_lagrange", "raw_lagrange_weighted", "raw_gauss_jacobi_weighted"])].copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for method, sub in flow.groupby("method"):
        sub = sub.sort_values("flow_point")
        ax.plot(sub["flow_point"], sub["mae_pct"], marker="o", label=method)
    ax.set_xlabel("flow point")
    ax.set_ylabel("MAE / %")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_integration_flow_mae.pdf")
    plt.close(fig)


def reportlabFont() -> str:
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


def plotFiguresReportlab(tables: dict[str, pd.DataFrame]) -> None:
    if canvas is None:
        return
    font = reportlabFont()
    selected = [
        "base_phys6",
        "base_lagrange",
        "raw_phys6_weighted",
        "raw_lagrange_weighted",
        "base_owics",
        "raw_gauss_jacobi_weighted",
        "base_equal_weight",
        "raw_equal_weighted",
    ]
    overall = tables["overall"][tables["overall"]["method"].isin(selected)].sort_values("mae_pct", ascending=True)
    c = canvas.Canvas(str(FIG_DIR / "fig_integration_overall_mae.pdf"), pagesize=landscape(A4))
    w, h = landscape(A4)
    c.setFont(font, 14)
    c.drawString(50, h - 42, "多声道积分方法总体 MAE 对比")
    max_mae = float(overall["mae_pct"].max())
    x0, y0, bar_w, row_h = 230, h - 82, 430, 24
    for idx, row in enumerate(overall.itertuples(index=False)):
        y = y0 - idx * row_h
        width = bar_w * float(row.mae_pct) / max_mae if max_mae > 0 else 0
        c.setFont(font, 9)
        c.setFillColor(colors.HexColor("#111827"))
        c.drawRightString(x0 - 8, y + 5, str(row.method))
        c.setFillColor(colors.HexColor("#3B82F6"))
        c.rect(x0, y, width, 12, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#111827"))
        c.drawString(x0 + width + 6, y + 2, f"{float(row.mae_pct):.3f}%")
    c.setFont(font, 9)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(50, 38, "数据来源：attachment1 基础估计、attachment6 原始声道序列、attachment7 固定积分权重。")
    c.save()

    flow = tables["by_flow"][tables["by_flow"]["method"].isin(["base_lagrange", "raw_lagrange_weighted", "raw_gauss_jacobi_weighted"])].copy()
    c = canvas.Canvas(str(FIG_DIR / "fig_integration_flow_mae.pdf"), pagesize=landscape(A4))
    w, h = landscape(A4)
    c.setFont(font, 14)
    c.drawString(50, h - 42, "不同流量点下积分方法 MAE 对比")
    methods = ["base_lagrange", "raw_lagrange_weighted", "raw_gauss_jacobi_weighted"]
    colors_list = [colors.HexColor("#2563EB"), colors.HexColor("#16A34A"), colors.HexColor("#DC2626")]
    flow_points = sorted(flow["flow_point"].unique())
    max_val = float(flow["mae_pct"].max())
    chart_x, chart_y, chart_w, chart_h = 70, 95, 680, 390
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.line(chart_x, chart_y, chart_x + chart_w, chart_y)
    c.line(chart_x, chart_y, chart_x, chart_y + chart_h)
    for mi, method in enumerate(methods):
        sub = flow[flow["method"].eq(method)].set_index("flow_point")
        pts = []
        for i, fp in enumerate(flow_points):
            x = chart_x + (i + 0.5) * chart_w / len(flow_points)
            val = float(sub.loc[fp, "mae_pct"]) if fp in sub.index else 0.0
            y = chart_y + val / max_val * chart_h if max_val > 0 else chart_y
            pts.append((x, y))
            c.setFillColor(colors_list[mi])
            c.circle(x, y, 3, stroke=0, fill=1)
        c.setStrokeColor(colors_list[mi])
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            c.line(x1, y1, x2, y2)
    c.setFont(font, 9)
    c.setFillColor(colors.HexColor("#111827"))
    for i, fp in enumerate(flow_points):
        x = chart_x + (i + 0.5) * chart_w / len(flow_points)
        c.drawCentredString(x, chart_y - 18, str(fp))
    c.drawString(chart_x + chart_w / 2 - 30, chart_y - 42, "flow point")
    c.drawString(chart_x - 45, chart_y + chart_h + 5, "MAE / %")
    for mi, method in enumerate(methods):
        c.setFillColor(colors_list[mi])
        c.rect(560, h - 70 - mi * 18, 10, 10, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#111827"))
        c.drawString(576, h - 72 - mi * 18, method)
    c.save()


def markdownTable(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
    if df.empty:
        return "_无数据_"
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append(format(float(value), floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def writeReport(tables: dict[str, pd.DataFrame], weights: pd.DataFrame) -> None:
    top = tables["overall"].head(12)
    lagrange_weight = weights[[
        "object_id",
        "chord_position_label",
        "normalized_channel_height",
        "weight_lagrange",
        "weight_owics",
        "weight_flowcal_gauss_jacobi",
        "weight_phys6",
        "weight_equal",
    ]]
    raw_lagrange_flow = tables["by_flow"][tables["by_flow"]["method"].eq("raw_lagrange_weighted")][[
        "flow_point",
        "window_count",
        "mae_pct",
        "mean_error_pct",
        "max_abs_error_pct",
    ]]
    disturbed = tables["by_disturbed"][tables["by_disturbed"]["method"].isin(["base_lagrange", "raw_lagrange_weighted", "raw_gauss_jacobi_weighted"])][[
        "method",
        "is_disturbed",
        "window_count",
        "mae_pct",
        "mean_error_pct",
        "max_abs_error_pct",
    ]]

    text = f"""# 多声道积分方法误差统计

## 1. Lagrange 型积分权重的物理含义

Lagrange 型五声道积分不是简单认为某一条声道“更重要”，而是把五个弦高处的线平均速度看成速度剖面的采样点，先用 Lagrange 插值基函数重构径向剖面，再对圆截面平均速度积分。附件 7 中的权重已经把声道位置、圆截面面积权和插值基函数积分结果合并为固定系数。

| 声道 | 位置 | 归一化弦高 | Lagrange 权重 | 物理含义 |
| --- | --- | ---: | ---: | --- |
| chord0 | edge_1 | 0.727425 | 0.266260 | 上侧边缘弦，帮助描述近壁速度梯度和剖面边缘曲率 |
| chord1 | inner_1 | 0.266219 | 0.096121 | 上内侧弦，约束中心与边缘之间的剖面形状 |
| chord2 | center | 0.000000 | 0.275238 | 中心弦，代表高速核心区，是单个最大权重 |
| chord3 | inner_2 | -0.266219 | 0.096121 | 下内侧弦，与 chord1 对称，用于抵消上下偏流 |
| chord4 | edge_2 | -0.727425 | 0.266260 | 下侧边缘弦，与 chord0 对称，约束另一侧近壁区域 |

这些权重满足左右/上下对称关系：`w0=w4`、`w1=w3`，且 `Σw_i=1`。边缘弦权重大并不表示边缘速度更大，而是因为 Lagrange 插值积分中边缘节点对剖面曲率和近壁区域积分贡献较强；中心弦权重大则对应管道高速核心区域的代表性。

## 2. 附件 6 原始声道序列重构口径

附件 7 给出了每条窗口声道与附件 6 原始 A/B 两组时间差列的映射。本文采用：

```text
u_i = gain_a,i * mean(diff_a,i) + gain_b,i * mean(diff_b,i) + intercept_i
V = A * duration_s * Σ w_i u_i
```

其中 `A` 为附件 7 的等效截面积。该口径不使用标准体积，只利用原始声道观测序列、声道映射和固定积分权重。

原始序列重构 `chord0--chord4` 与附件 1 窗口声道的差异如下：

{markdownTable(tables["chord_check"])}

## 3. 总体误差统计

{markdownTable(top)}

## 4. Lagrange 型积分按流量点表现

{markdownTable(raw_lagrange_flow)}

## 5. 扰流与非扰流条件对比

{markdownTable(disturbed)}

## 6. 输出文件

- `code/outputs/integration_weights.csv`：附件 7 五声道权重。
- `code/outputs/reconstructed_raw_chords.csv`：由附件 6 原始序列重构的五声道速度。
- `code/outputs/integration_estimates_long.csv`：单声道、多声道积分和附件 1 基础估计的长表。
- `code/outputs/integration_overall_metrics.csv`：总体误差统计。
- `code/outputs/integration_metrics_by_flow_point.csv`：按流量点统计。
- `code/outputs/integration_metrics_by_condition.csv`：按工况说明统计。
- `figures/fig_integration_overall_mae.pdf`：总体 MAE 对比图；优先用 `matplotlib`，缺失时用 ReportLab 兜底生成。
- `figures/fig_integration_flow_mae.pdf`：流量点 MAE 对比图；优先用 `matplotlib`，缺失时用 ReportLab 兜底生成。
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    win, raw, geo = loadInputs()
    rec_chords = reconstructChordsFromRaw(raw, geo)
    estimates, weights = buildEstimates(win, rec_chords, geo)
    tables = writeOutputs(estimates, weights, rec_chords, win)
    plotFigures(tables)
    writeReport(tables, weights)
    print("Report:", REPORT)
    print("Overall:", OUT_DIR / "integration_overall_metrics.csv")
    if plt is None and canvas is not None:
        print("Figures generated with ReportLab fallback:", FIG_DIR / "fig_integration_overall_mae.pdf", FIG_DIR / "fig_integration_flow_mae.pdf")
    elif plt is None:
        print("Figures skipped: neither matplotlib nor reportlab is installed in the current Python environment.")
    else:
        print("Figures:", FIG_DIR / "fig_integration_overall_mae.pdf", FIG_DIR / "fig_integration_flow_mae.pdf")
    print(tables["overall"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
