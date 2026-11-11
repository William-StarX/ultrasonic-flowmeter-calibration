# 项目论文库

## 原文目录

表中的路径相对于原工作区 `demo/`。原文保存在本地 `demo/literature/raw/`，论文引用见 [references.tex](../paper/references.tex)。

## 本地中文 CAJ 文献

| 编号 | 题名 | 作者 | 文件 | 当前读取状态 | 对本题的作用 |
| --- | --- | --- | --- | --- | --- |
| CN01 | 上游扰流条件对超声波流量计准确度的影响研究 | 金少杰等 | `literature/raw/上游扰流条件对超声波流量计准确度的影响研究_金少杰.pdf`；CAJ 备份同目录 | PDF 已读取，图片型 PDF，按页面图像识别摘要与结论 | 支撑上游缓冲段长度、90 度弯管等扰流会显著影响计量准确度，且可用修正系数降低误差 |
| CN02 | 多声道气体超声流量计非充分发展流条件下的仿真结果量化和实验研究 | 苏杰 | `literature/raw/多声道气体超声流量计非充分发展流条件下的仿真结果量化和实验研究_苏杰.pdf`；CAJ 备份同目录 | PDF 已读取，图片型 PDF，按页面图像识别摘要与结论 | 支撑非充分发展流下多声道仿真、声道组合和偏移量量化 |
| CN03 | 油品外夹式超声波流量计计量误差研究 | 黄恩扬 | `literature/raw/油品外夹式超声波流量计计量误差研究_黄恩扬.pdf`；CAJ 备份同目录 | PDF 已读取，图片型 PDF，按页面图像识别封面、摘要和附录 | 支撑弯管扰流、介质黏度、流速、温度和 BP 神经网络补偿对超声计量误差的影响 |
| CN04 | 高精度多声道超声波流量计积分方法及影响因素研究 | 相诺林 | `literature/raw/高精度多声道超声波流量计积分方法及影响因素研究_相诺林.pdf`；CAJ 备份同目录 | PDF 已读取，图片型 PDF，按页面图像识别摘要 | 支撑多声道积分权重、速度剖面、探头扰流和前直管段影响因素分析 |

## 本地英文 PDF 文献

| 编号 | 题名 | 作者 | 文件 | 当前读取状态 | 对本题的作用 |
| --- | --- | --- | --- | --- | --- |
| EN09 | VMD-based adaptive ultrasonic flowmeter echo signal denoising algorithm | Zhongbing Li 等 | `literature/raw/VMD-based adaptive ultrasonic flowmeter echo signal denoising algorithm_Li.pdf` | PDF 已读取，文本型 PDF，已抽取摘要和期刊信息 | 支撑超声回波信号降噪、VMD 分解、优化算法和小波阈值降噪，可作为数据预处理和测量稳定性分析依据 |
| EN10 | Physically Constrained Generative Adversarial Network Data Augmentation Method for Multichannel Ultrasonic Flowmeters of Natural Gas | Haohui Cai 等 | `literature/raw/Physically Constrained GAN Data Augmentation for Multichannel Ultrasonic Flowmeters_Cai.pdf` | PDF 已读取，文本型 PDF，已抽取摘要和 DOI | 支撑多声道超声流量计小样本数据增强、物理约束 CWGAN-GP 和数据驱动评价模型 |

## 读取说明

`zglw` 重新读取后发现已新增 PDF：`1.pdf` 对应苏杰论文，`2.pdf` 对应相诺林硕士论文，`3.pdf/4.pdf/5.pdf` 均对应金少杰论文且首页渲染内容一致，项目库只保留 `3.pdf` 的命名副本；`6.pdf` 对应黄恩扬硕士论文，已归档为命名 PDF。PDF 均为打印生成的图片型 PDF，`pypdf` 无法直接抽取正文，已通过 Poppler 渲染页面后进行视觉读取。

原资料目录 `a题论文/` 中的 `1.pdf` 对应 Li 等的超声流量计回波信号 VMD 自适应降噪论文，`2.pdf` 对应 Cai 等的多声道天然气超声流量计物理约束 GAN 数据增强论文。两篇论文在原工作区均有按题名保存的 PDF。
