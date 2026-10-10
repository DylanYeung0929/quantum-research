# Quantum Research Archive

本目录整理自 `D:\Study\Quantum`，用于版本管理与后续同步到 GitHub。

## 课题划分

仓库包含两个相互独立的研究课题：

1. [`air-quality-classification`](projects/air-quality-classification/README.md)：空气质量四分类，包含司南杯材料、经典神经网络基线、量子混合模型实验图片和测试表。
2. [`medical-insurance-cost-regression`](projects/medical-insurance-cost-regression/README.md)：医疗保险费用回归，包含保险数据、量子混合模型实验记录、RMSE 结果和报告。

两个课题均按以下逻辑组织：

- `code-data/`：代码与数据
- `documents/`：报告和辅助文档
- `results/`：实验图片和结果（存在时）
- `archives/`：原始压缩包，使用 Git LFS 管理

## 整理原则

1. 源目录仅作为只读来源，文件通过复制进入本仓库。
2. 保留各主题内部的原始目录结构，避免破坏文档与代码之间的相对引用。
3. 不纳入 Office 临时锁文件、Python 缓存及其他可再生成文件。
4. 两个研究课题相互隔离，避免数据、报告和实验结果混用。
5. 压缩包等二进制归档文件使用 Git LFS 管理。
6. 学习资料、课程文件及其配套数据集不纳入仓库。

详见 [`SOURCE-MANIFEST.md`](SOURCE-MANIFEST.md)。
