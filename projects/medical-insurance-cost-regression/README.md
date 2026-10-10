# 医疗保险费用回归

### Medical Insurance Cost Regression · Hybrid Quantum–Classical Research

本课题使用年龄、性别、BMI、子女数量、吸烟情况和地区等特征预测医疗保险费用，并记录量子-经典混合模型的 RMSE 及超参数实验。

## 目录

- `code-data/WuKong/`：训练/测试数据、实验记录、图表和工作文档
- `documents/`：医疗保险费用预测分析报告
- `archives/`：与本课题相关的原始压缩包

主要数据标识：`age`、`bmi`、`smoker`、`charges`；任务类型为回归。

## 资料导览

| 内容 | 位置 |
| --- | --- |
| 训练与测试数据 | [`train.csv`](code-data/WuKong/train.csv)、[`test.csv`](code-data/WuKong/test.csv) |
| 费用预测分析报告 | [`documents/`](documents/) |
| 超参数与训练过程图表 | [`code-data/WuKong/`](code-data/WuKong/) |
| 原始实验归档 | [`archives/`](archives/) |

阅读时建议先确认报告中的数据划分、模型结构与评价方式，再对照 RMSE 和超参数曲线。历史实验图应与对应记录一起使用，不能据此推断不同设置在同一环境下的优劣。

## 复现准备

本课题当前公开目录主要保存 CSV、Markdown 报告、图表和原始 ZIP 归档，没有独立的 Python 训练入口或依赖锁定文件。要继续实验，应先查看原始归档是否包含所需代码，再按对应记录准备环境和参数。

ZIP 使用 Git LFS 管理，下载方式见[根目录 README](../../README.md)。当前文档描述的是已归档资料，不宣称这些实验已在当前环境重新验证。

[返回研究总览](../../README.md)
