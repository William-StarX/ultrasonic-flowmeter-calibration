# A题发布包说明

本目录为“扰流条件下的超声波流量计”数学建模题发布包。

## 文件说明

- `A题_扰流条件下的超声波流量计.pdf`：赛题题面。
- `A题_扰流条件下的超声波流量计.tex`：赛题题面 LaTeX 源文件。
- `attachment1_window_data.csv`：窗口级主数据。
- `attachment2_condition_schedule.csv`：测试日期与扰流状态汇总。
- `attachment3_data_dictionary.csv`：字段说明。
- `attachment4_baseline_summary.csv`：基础方法误差统计。
- `attachment5_inspection_targets.csv`：评价指标说明。
- `attachment6_window_raw_samples.csv`：已按窗口切分的原始声道观测序列，可通过 `window_id` 与主数据关联。
- `attachment7_meter_geometry.csv`：仪表等效几何、等效工程声道参数、五声道积分权重和原始链路映射。
- `sample_submission_phys6.csv`：样例提交文件。
- `evaluate_submission.py`：评价脚本。

`attachment1_window_data.csv` 是主建模表，适合直接建立窗口级流量估计和扰流补偿模型；`attachment6_window_raw_samples.csv` 包含 29322 条窗口内逐秒观测样本，适合用于验证单声道/多声道积分思想，或自行构造动态、剖面等特征；`attachment7_meter_geometry.csv` 给出使用附件6构造基础五声道积分时所需的等效工程声道参数、等效常数和声道映射。

## 提交材料

参赛队应提交三类材料：

- 建模论文。
- 结果 CSV 文件。
- 可复现结果的支撑代码。

支撑代码要求参照全国大学生数学建模竞赛对支撑材料的要求：提交完整源程序、必要运行说明和依赖环境说明；代码应能从附件数据出发复现论文中的主要建模流程、最终提交 CSV 和主要评价指标；路径尽量使用相对路径，不依赖本机私有目录、手工中间操作或未随题提供的数据；不得只提交截图、伪代码、交互记录或不可运行片段。

## 结果文件格式

提交文件为 CSV，至少包含两列：

```text
window_id,model_volume_m3
```

其中 `window_id` 必须与 `attachment1_window_data.csv` 中的窗口编号一致，`model_volume_m3` 为模型给出的窗口体积估计。

## 评价脚本示例

在本目录下运行：

```powershell
python evaluate_submission.py sample_submission_phys6.csv --output-dir output_test
```

运行后会生成：

- `evaluation_summary.csv`：总体指标。
- `evaluation_groups.csv`：按日期和流量点分组的重复性指标。
- `evaluation_windows.csv`：逐窗口误差。

`sample_submission_phys6.csv` 仅用于说明提交格式，不代表推荐方法或标准答案。
