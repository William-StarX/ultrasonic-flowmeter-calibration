# 多声道超声波流量计校准工具

用于离线分析管道扰流引起的计量偏差。输入带参考体积的试验窗口、逐秒声道观测和仪表几何参数，输出每个窗口的校准体积、分组误差，以及线性度、重复性和扰流影响指标。适用于台架数据复核和补偿方案比较。

## 快速开始

在仓库根目录使用 Python 3.11 或 3.12。下面两条命令分别复跑样例校准、独立评价生成的体积估计。

```powershell
python -m pip install -r requirements.txt
python src/flowmeter calibrate --data-dir data/calibration --output-dir runs/calibration
python src/flowmeter evaluate --estimates runs/calibration/submission_final.csv --output-dir runs/evaluation
```

- `runs/calibration/` 保存体积估计、逐窗口稳健特征、候选参数和校准报告。样例达到基准阈值时生成 `submission_final.csv`。
- `runs/evaluation/` 生成 `evaluation_summary.csv`、`evaluation_groups.csv` 和 `evaluation_windows.csv`。
- 可以单独运行 `evaluate`，用 `--reference` 指定参考窗口 CSV，用 `--estimates` 指定仪表或模型给出的体积估计。两份表以 `window_id` 一一对齐。

## 校准流程

1. 将窗口汇总数据与逐秒声道数据按 `window_id` 对齐，提取声道速度剖面和稳定性特征。
2. 以五声道物理积分为基准，使用双核岭回归补偿由剖面形态与声道稳定性变化引起的体积偏差。
3. 在带参考体积的试验批次上拟合受约束的 D0-RBF 后校准，输出每个窗口的体积估计。
4. 按试验日期与名义流量点评价误差均值和重复性，并计算线性度与扰流影响。

## 样例验证

样例包含 159 个试验窗口和 29,322 条逐秒观测，使用 DN400 仪表的等效几何参数。

| 评估口径 | 平均绝对相对误差 | 阈值通过情况 |
| --- | ---: | --- |
| 五声道物理积分基准，全部 159 个窗口 | 0.180653% | — |
| 后校准模型，同批次全量拟合 | 0.015891% | 30/30 个分组通过，三项指标均通过 |
| 双核补偿模型，40 次分层留出验证 | 中位数 0.061356% | 0/40 次同时达到全部阈值 |

全量拟合衡量当前批次的校准精度。现场部署前需用新仪表或新工况的独立参考数据验证误差与重复性。当前提供离线批处理入口。

## 项目目录

| 路径 | 内容 |
| --- | --- |
| [`src/flowmeter/`](src/flowmeter/) | 命令行入口、声道特征、补偿模型和评估 |
| [`data/calibration/`](data/calibration/) | 台架样例数据、几何参数和数据字典 |
| [`results/`](results/) | [评估摘要](results/benchmark-evaluation/evaluation_summary.csv)、[模型对比](results/model-validation/)和[整组留出诊断](results/robustness/) |
| [`docs/数据与验证.md`](docs/数据与验证.md) | 自有批次的输入约定、输出字段和验证方法 |
| [`paper/`](paper/) · [`figures/`](figures/) | 方法报告、论文源文件和技术图表 |
