# 流量计校准样例数据

数据包含 159 个试验窗口和 29,322 条逐秒声道观测。窗口覆盖 D0 无扰流参考和 D1–D8 扰流条件，仪表按 DN400 等效几何建模。

| 文件 | 用途 |
| --- | --- |
| [`attachment1_window_data.csv`](attachment1_window_data.csv) | 窗口级参考体积、工况、基础积分和剖面特征 |
| [`attachment2_condition_schedule.csv`](attachment2_condition_schedule.csv) | 试验日期与扰流工况记录 |
| [`attachment3_data_dictionary.csv`](attachment3_data_dictionary.csv) | 数据字段与单位 |
| [`attachment4_baseline_summary.csv`](attachment4_baseline_summary.csv) | 基础积分误差统计 |
| [`attachment5_inspection_targets.csv`](attachment5_inspection_targets.csv) | 基准评估阈值 |
| [`attachment6_window_raw_samples.csv`](attachment6_window_raw_samples.csv) | 按 `window_id` 对齐的逐秒声道观测 |
| [`attachment7_meter_geometry.csv`](attachment7_meter_geometry.csv) | 等效几何、声道映射和积分权重 |
| [`sample_submission_phys6.csv`](sample_submission_phys6.csv) | Phys6 体积估计样例，可直接用于 `evaluate` |

运行方式和自有数据要求见 [数据与验证](../../docs/数据与验证.md)。原始试验说明和评价脚本随数据保留，供核查试验出处和指标定义。
