<div align="center">

# Quantum Research

### 经典神经网络与量子—经典混合机器学习研究归档

Air quality classification · Medical insurance cost regression

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![VQNet](https://img.shields.io/badge/Framework-VQNet-7C3AED?style=flat-square)
![Research](https://img.shields.io/badge/Focus-Hybrid_Machine_Learning-0F766E?style=flat-square)

[课题导览](#课题划分) · [阅读与复现](#阅读与复现) · [来源清单](SOURCE-MANIFEST.md)

</div>

---

本仓库整理自原始研究资料，保存经典神经网络基线、量子—经典混合模型实验记录、数据、图表与报告，便于按课题查阅和继续研究。

> A research archive comparing classical neural networks and hybrid quantum–classical approaches across classification and regression tasks.

## 课题划分

仓库包含两个相互独立的研究课题：

| 课题 | 任务与输入 | 资料入口 |
| --- | --- | --- |
| 空气质量四分类 | 以气象、污染物及区域特征预测四个空气质量类别；比较经典与量子模型 | [课题 README](projects/air-quality-classification/README.md) |
| 医疗保险费用回归 | 以年龄、BMI、吸烟情况等特征预测费用；记录混合模型的 RMSE 与超参数实验 | [课题 README](projects/medical-insurance-cost-regression/README.md) |

两个课题均按以下逻辑组织：

- `code-data/`：代码与数据
- `documents/`：报告和辅助文档
- `results/`：实验图片和结果（存在时）
- `archives/`：原始压缩包，使用 Git LFS 管理

## 阅读与复现

建议先阅读各课题 README 和报告，再查看数据、代码与实验图；报告中的历史结果应结合对应版本、超参数与数据划分理解。

克隆源码和普通文档：

```bash
git clone https://github.com/DylanYeung0929/quantum-research.git
cd quantum-research
```

需要原始 ZIP 归档时，在安装 Git LFS 后执行：

```bash
git lfs install
git lfs pull
```

`.gitattributes` 中的 `projects/**/archives/*.zip` 使用 LFS；没有下载 LFS 内容时，这些文件可能只是指针文件。

空气质量的可见 Python 入口包括 `classical-baselines/complete/template_classical.py` 和 `classical-baselines/zmz/template_classical.py`。它们使用 `pyvqnet`、NumPy、pandas 和 scikit-learn，并保留了原始 Windows 绝对路径。运行前需要配置兼容的依赖环境，将脚本中的 `data_directory` 改为本机对应数据目录。

仓库尚未提供统一的依赖锁定文件或覆盖全部课题的一键运行入口。量子混合实验应从对应报告与原始归档查找代码版本；已保存的图表和报告不代表当前环境已重新运行验证。

## 整理原则

1. 源目录仅作为只读来源，文件通过复制进入本仓库。
2. 保留各主题内部的原始目录结构，避免破坏文档与代码之间的相对引用。
3. 不纳入 Office 临时锁文件、Python 缓存及其他可再生成文件。
4. 两个研究课题相互隔离，避免数据、报告和实验结果混用。
5. 压缩包等二进制归档文件使用 Git LFS 管理。
6. 学习资料、课程文件及其配套数据集不纳入仓库。

详见 [`SOURCE-MANIFEST.md`](SOURCE-MANIFEST.md)。

## 许可证 / License

除另有注明外，本仓库中由 DylanYeung0929 有权授权的原创代码与文档采用 [MIT License](LICENSE)。第三方代码、依赖、合作成果、数据集、赛题、图片、字体及其他素材不因本许可证而改变授权，仍适用各自的许可与权利声明。

Unless otherwise noted, original code and documentation that DylanYeung0929 has the right to license are available under the [MIT License](LICENSE). This grant does not relicense third-party code, dependencies, jointly authored works, datasets, problem statements, images, fonts, or other assets; their respective licenses and rights remain applicable.
