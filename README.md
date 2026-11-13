# 多声道超声波流量计校准工具<br>Multipath ultrasonic flowmeter calibration

输入试验窗口、逐秒声道观测和仪表几何参数，输出校准体积及误差报告。可以用样例复跑校准，也可以单独评价已有的仪表或模型体积估计。

Provide test windows, per-second channel observations, and meter geometry to obtain calibrated volumes and error reports. Reproduce the sample calibration, or independently evaluate volume estimates from an instrument or another model.

## 快速开始：评价现成结果<br>Quick start: evaluate the supplied estimates

使用 **Python 3.11 或 3.12**。下面的命令读取仓库中已有的体积估计，生成一份评价报告：

Use **Python 3.11 or 3.12**. These commands read the supplied volume estimates and generate an evaluation report:

```sh
git clone https://github.com/William-StarX/ultrasonic-flowmeter-calibration.git
cd ultrasonic-flowmeter-calibration
python -m pip install -r requirements.txt
python src/flowmeter evaluate --estimates results/calibrated-model/submission_final.csv --output-dir runs/evaluation
```

完成后打开 `runs/evaluation/evaluation_summary.csv` 查看总体指标，逐窗口误差和分组结果保存在同一目录。

Open `runs/evaluation/evaluation_summary.csv` for the overall metrics. Per-window errors and grouped results are saved alongside it.

## 重新运行校准<br>Run the calibration

使用随仓库提供的台架样例重新拟合，再评价生成的体积估计：

Refit using the supplied bench-test data, then evaluate the generated estimates:

```sh
python src/flowmeter calibrate --data-dir data/calibration --output-dir runs/calibration
python src/flowmeter evaluate --estimates runs/calibration/submission_d0_rbf_postcalibrated_dual_kernel_krr.csv --output-dir runs/calibration-check
```

`runs/calibration/` 保存体积估计、候选参数、模型文件和校准报告；达到基准阈值时还会生成 `submission_final.csv`。详细报告位于其 `reports/` 子目录。

`runs/calibration/` contains volume estimates, candidate parameters, model files, and calibration reports. A `submission_final.csv` is also produced when the benchmark thresholds are met. Detailed reports are in its `reports/` subdirectory.

## 换成自己的数据<br>Use your own data

校准目录需要以下三份 CSV，字段与单位沿用 [数据字典](data/calibration/attachment3_data_dictionary.csv)。把 `--data-dir` 改成自己的目录即可。

The calibration directory needs these three CSV files, following the fields and units in the [data dictionary](data/calibration/attachment3_data_dictionary.csv). Point `--data-dir` to your directory.

| 文件<br>File | 内容<br>Contents |
|---|---|
| `attachment1_window_data.csv` | 窗口编号、参考体积、工况和窗口级特征<br>Window IDs, reference volumes, conditions, and window features |
| `attachment6_window_raw_samples.csv` | 按窗口编号对齐的逐秒声道观测<br>Per-second channel observations matched by window ID |
| `attachment7_meter_geometry.csv` | 管径、声道映射、弦位和积分权重<br>Pipe diameter, channel mapping, chord positions, and integration weights |

只想评价已有估计时，准备参考表和包含 `window_id`、`model_volume_m3` 的估计表，以相同窗口编号对齐，体积单位为 m³：

To evaluate existing estimates, prepare a reference table and an estimate table containing `window_id` and `model_volume_m3`. Match the same window IDs and use m³ for volume:

```sh
python src/flowmeter evaluate --reference path/to/reference.csv --estimates path/to/estimates.csv --output-dir runs/check
```

参考表的完整字段、工况标记和指标口径见 [数据与验证](docs/数据与验证.md)。

See [data and validation](docs/数据与验证.md) for reference-table fields, condition labels, and metric definitions.

## 读取评价输出<br>Read the evaluation output

| 文件<br>File | 用途<br>Purpose |
|---|---|
| `evaluation_summary.csv` | 总体误差、分组通过数、线性度、重复性和扰流影响<br>Overall error, passing groups, linearity, repeatability, and disturbance effects |
| `evaluation_groups.csv` | 按日期与流量点汇总的误差和重复性<br>Error and repeatability grouped by date and flow point |
| `evaluation_windows.csv` | 每个窗口的参考体积、估计体积与相对误差<br>Reference volume, estimated volume, and relative error for each window |

## 本项目的工作与样例结果<br>Project contributions and sample results

工具将声道原始观测、物理积分、双核误差补偿和独立评价接到同一套文件接口中。对带参考体积的少量窗口进行校准后，可以同时查看总体精度、重复性和不同扰流条件下的表现。

The tool connects raw channel observations, physical integration, dual-kernel error compensation, and independent evaluation through one file-based interface. Calibration on a small set of reference-labelled windows can be assessed through overall accuracy, repeatability, and performance under different disturbance conditions.

样例包含 **159 个参考窗口、29,322 条逐秒观测**，采用 DN400 等效几何。代表性对照使用 118 个窗口训练、41 个留出验证；40 次分层验证每次使用 111–128 个训练窗口。

The sample contains **159 reference windows and 29,322 per-second observations**, using equivalent DN400 geometry. A representative comparison uses 118 training windows and 41 held-out windows; the 40 stratified validations use 111–128 training windows each.

| 评估口径<br>Evaluation | 平均绝对相对误差<br>Mean absolute relative error | 阈值通过情况<br>Threshold results |
|---|---:|---|
| 五声道物理积分，全部窗口<br>Five-channel physical integration, all windows | 0.180653% | — |
| 后校准模型，同批次全量拟合<br>Post-calibrated model, full fit on the same batch | 0.015891% | 30/30 分组及三项指标通过<br>30/30 groups and all three metrics pass |
| 双核补偿，40 次分层留出<br>Dual-kernel compensation, 40 stratified holdouts | 中位数 0.061356%<br>Median 0.061356% | 0/40 次同时通过全部阈值<br>0/40 runs pass all thresholds together |

[评估摘要](results/benchmark-evaluation/evaluation_summary.csv) · [模型对比](results/model-validation/) · [整组留出诊断](results/robustness/)

[Evaluation summary](results/benchmark-evaluation/evaluation_summary.csv) · [Model comparisons](results/model-validation/) · [Group-holdout diagnostics](results/robustness/)

## 方法、文献与源码<br>Methods, references, and source

模型采用多声道物理积分与数据驱动误差补偿思路。公式和试验设计见 [方法报告](paper/main.pdf)，相关论文见 [文献依据](docs/LITERATURE_BASIS.md)，原始资料见 [资料索引](docs/资料索引.md)。命令行和模型实现在 [src/flowmeter/](src/flowmeter/)，NumPy、pandas 等依赖列于 [requirements.txt](requirements.txt)。

The model combines multipath physical integration with data-driven error compensation. See the [method report](paper/main.pdf) for formulas and experiment design, the [literature basis](docs/LITERATURE_BASIS.md) for related papers, and the [source-material index](docs/资料索引.md) for original materials. The CLI and models are in [src/flowmeter/](src/flowmeter/); dependencies such as NumPy and pandas are listed in [requirements.txt](requirements.txt).

## 注意事项<br>Attention

同批次全量拟合衡量当前批次的校准精度；新仪表或新工况需要独立参考数据验证。当前入口用于离线批处理。

A full fit measures calibration accuracy on the current batch. New instruments or operating conditions need independent reference data for validation. The current interface runs offline batches.
