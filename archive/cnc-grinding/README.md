# B题发布包说明

本目录为“高精度非圆曲线磨削的砂轮中心轨迹规划”数学建模题发布包。

## 文件说明

- `B题_高精度非圆曲线磨削.pdf`：赛题题面。
- `B题_高精度非圆曲线磨削.tex`：赛题题面 LaTeX 源文件。
- `attachment1_profile_points_raw.csv`：原始轮廓离散点。
- `attachment2_profile_points_clean.csv`：推荐建模轮廓点。
- `attachment3_machine_params.csv`：砂轮、机床和精度目标参数。
- `attachment4_baseline_summary.csv`：基础统计和样例提交评价结果。
- `attachment5_data_dictionary.csv`：字段和附件说明。
- `attachment6_problem_diagram.png`：问题几何示意图。
- `sample_submission_discrete_offset_2000.csv`：样例提交文件。
- `evaluate_submission.py`：公开评价脚本。

## 提交材料

参赛队应提交三类材料：

- 建模论文。
- 结果 CSV 文件。
- 可复现结果的支撑代码。

支撑代码要求参照全国大学生数学建模竞赛对支撑材料的要求：提交完整源程序、必要运行说明和依赖环境说明；代码应能从附件数据出发复现论文中的主要建模流程、最终提交 CSV 和主要评价指标；路径尽量使用相对路径，不依赖本机私有目录、手工中间操作或未随题提供的数据；不得只提交截图、伪代码、交互记录或不可运行片段。

## 结果文件格式

提交文件为 CSV，至少包含三列：

```text
tool_id,x_center_mm,y_center_mm
```

其中 `tool_id` 为刀位点编号，`x_center_mm` 与 `y_center_mm` 为砂轮中心坐标，单位 mm。

## 评价脚本示例

在本目录下运行：

```powershell
python evaluate_submission.py sample_submission_discrete_offset_2000.csv --output-dir output_test
```

运行后会生成：

- `evaluation_summary.csv`：总体指标。
- `evaluation_points.csv`：逐刀位点等距误差。
- `evaluation_segments.csv`：相邻刀位点弦长均匀性。

`sample_submission_discrete_offset_2000.csv` 仅用于说明提交格式，不代表推荐方法或最终结果。公开评价脚本采用折线距离和弦长均匀性作为粗略代理指标，主要用于格式检查和自测；正式评分应结合论文、支撑代码和正式高精度评价。
