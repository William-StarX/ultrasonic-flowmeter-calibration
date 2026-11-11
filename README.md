# 2026 年校内数学建模赛

本项目整理 2026 年校内数学建模赛的 A、B 题资料。A 题包含扰流条件下超声波流量计的完整建模与复验流程。B 题包含高精度非圆曲线磨削的赛题、附件、样例和公开评价程序。

| 目录 | 内容 |
| --- | --- |
| [A 题发布包](<A题 (1)/README.md>) | 题面、附件 1–7、数据字典、官方评分脚本和样例提交 |
| [B 题发布包](B题/README.md) | 砂轮中心轨迹规划题面、轮廓点、机床参数、几何图、评价脚本和样例提交 |
| [code/paper_appendix](code/paper_appendix/) | A 题 11 个建模、特征工程和复验程序 |
| [paper](paper/) | [最终论文 PDF](paper/main.pdf)、分节 LaTeX 源文件、参考文献和代码附录 |
| [figures](figures/) | 正文采用的 10 幅图和工程流程图的 DrawIO 源文件 |
| [results](results/) | A 题最终提交、官方评分、消融、标签盲验证和扰流诊断 |
| [docs](docs/资料索引.md) | 原文件夹盘点、取材来源和文献索引 |

## A 题建模与结果

1. 用顺逆流传播时间重建五声道观测，在同一批窗口上比较 Phys6、Lagrange、OWICS 和等权积分。
2. 对 D0 无扰流窗口拟合流量相关的校准曲线，通过留出窗口检查正常工况修正效果。
3. 用声道剖面、方向不对称和逐秒稳定性特征分析扰流，采用 PCA 和聚类检验工况可分性。
4. 比较 Ridge、ANN 和核岭回归，在物理与稳定性双核 KRR 上叠加受约束的 D0-RBF 后校准，生成体积估计。

| 评估口径 | MAE | 通过情况 |
| --- | ---: | --- |
| 最终模型在 159 个给定窗口上的全量标定 | 0.015891% | 30/30 个分组通过，三项官方不确定度指标均通过 |
| 双核 KRR 的 40 次标签盲分层验证 | 中位数 0.061356% | 0/40 次同时达到全部阈值 |

两行结果分别描述同样本标定精度和留出窗口表现。跨工况风险可结合 `results/灵敏度与失效/` 中的整组留出和 OOD 诊断阅读。

## 复验 A 题最终提交

在仓库根目录使用 Python 3.11 或 3.12，建议先激活独立虚拟环境。

```powershell
python -m pip install -r requirements.txt
python code/paper_appendix/final_postcalibrated_dual_kernel_krr.py
python "A题 (1)/evaluate_submission.py" code/paper_appendix/verification_outputs/submission_final.csv --output-dir code/paper_appendix/official_evaluation
```

重新生成的 CSV 可与 [随仓库保存的最终提交](results/问题四_最终模型/submission_final.csv) 对照。官方指标见 [evaluation_summary.csv](results/official-evaluation/evaluation_summary.csv)。

其余实验按原复验顺序执行 `code/paper_appendix/` 下的 `problem1_multichannel_integration.py`、`problem2_d0_calibration.py`、`model_evolution_ridge_ann.py`、`physical_feature_library.py`、`compact_rbf_krr.py`、`ann_krr_same_split.py`、`problem3_disturbance_features.py`、`fullfit_expression_baseline.py`、`physical_stability_dual_kernel_krr.py`。问题三引用 `code/outputs/` 中保存的整组留出诊断结果。各脚本的复跑产物保存在 `code/paper_appendix/verification_*`。

论文已提供成稿，附录 A 的文件清单对应原赛事支撑包。重新排版时，在 `paper/` 目录用支持中文字体的 XeLaTeX 连续执行两次 `xelatex main.tex`。

## 运行 B 题样例评价

```powershell
cd B题
python evaluate_submission.py sample_submission_discrete_offset_2000.csv --output-dir output_test
```

公开评价脚本用折线距离和弦长均匀性检查样例。该目录的 CSV 是赛题附带的样例提交。
