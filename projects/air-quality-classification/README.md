# 空气质量四分类

### Air Quality Classification · Classical & Hybrid Quantum Models

本课题使用温度、湿度、PM2.5、PM10、NO2、SO2、CO、工业区距离和人口密度等特征，对空气质量进行四分类，并比较经典神经网络与量子机器学习模型。

## 目录

- `code-data/classical-baselines/`：三个来源版本的经典神经网络代码、预处理数据和结果
- `documents/final-report/`：司南杯最终报告及引用图片
- `documents/supporting/`：实现细节、测试文档和参数测试表
- `results/experiment-images/`：经典与量子混合模型实验图
- `archives/`：与本课题相关的原始压缩包

主要数据标识：`Temperature`、`PM2.5`、`Air Quality`；任务类型为多分类。

## 研究与阅读路线

| 内容 | 建议入口 |
| --- | --- |
| 任务、模型与性能分析 | [`documents/final-report/`](documents/final-report/) |
| 经典神经网络源码与 CSV | [`code-data/classical-baselines/complete/`](code-data/classical-baselines/complete/) |
| 其他来源版本 | [`code-data/classical-baselines/`](code-data/classical-baselines/) |
| 准确率、F1 与损失曲线 | [`results/experiment-images/`](results/experiment-images/) |
| 参数测试表与实现说明 | [`documents/supporting/`](documents/supporting/) |

`complete/template_classical.py` 使用 VQNet 构建全连接神经网络，包含标准化、标签编码、训练与测试集准确率 / macro-F1 计算。不同来源版本和实验文件应分别对照，不应直接混合比较。

## 运行准备

准备与代码兼容的 `pyvqnet`、NumPy、pandas 和 scikit-learn 环境。修改脚本末尾的 `data_directory`，使其指向包含原始与预处理 CSV 的本机目录，再运行：

```bash
cd projects/air-quality-classification/code-data/classical-baselines/complete
python template_classical.py
```

训练轮数、批量大小、学习率与隐藏层宽度在脚本末尾设置。仓库未锁定依赖版本；以上是源码入口说明，当前归档没有提供已验证的统一复现环境。

量子混合模型的记录见报告、测试表、实验图和 `archives/`；读取 ZIP 前请按[根目录说明](../../README.md)下载 Git LFS 内容。

[返回研究总览](../../README.md)
