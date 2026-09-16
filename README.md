# 基于多模态特征融合与域泛化的玉米南方锈病病情等级识别

本目录是面向论文复现与改进的研究骨架。目标是利用田间 RGB 图像、温湿度时间序列（可选加入光合指标），在“训练地点”和“未见测试地点”之间存在分布差异时，识别有序病情等级。

## 研究问题

给定叶片图像 $x_i$、采样日前一段时间的气象序列 $x_w$ 和来源域（地点/批次）$d$，预测等级 $y\in\{CK,1,3,5,7,9\}$。主评价采用留一地点外测（LOSO）：每轮仅以两个地点训练/验证，第三地点只用于最终测试，训练阶段禁止访问目标地点图像或统计量。

## 更新后的总体理解

本项目采用“基础视觉模型辅助任务模型”的分层路线，而不是把通用基础模型直接当作最终分级器：

1. **DINO 系列**提供可迁移的图像/稠密视觉表征；
2. **SAM 系列、PlantSeg 与 MMSegmentation**辅助提取叶片或病斑 ROI；
3. **温湿度时序编码器**提供病害发生环境信息；
4. **有序等级头与域泛化损失**负责 CK/1/3/5/7/9 的任务适配和未见地点泛化；
5. **多模态大模型**暂作为标注辅助、弱监督和解释工具，不直接替代可复现的监督分级器。

建议主模型：

`叶片/病斑分割 → DINOv2/ConvNeXt 图像编码器 + TST/InceptionTime 气象编码器 → 门控融合 → CORAL 有序分类头 + 源域 Deep CORAL 对齐`

注意：CORAL 有两个不同含义。本项目把有序回归写作 **CORAL ordinal regression**，把域对齐写作 **Deep CORAL alignment**。

基础模型不是自动解决病情分级的“黑盒捷径”。通用 SAM 不等于已经准确分割锈斑，DINO 特征也不等于已经学习本地农学等级标准；所有基础模型都必须通过同一 LOSO 划分、文字遮挡和天气打乱实验验证。DINOv3、SAM3 等较新的模型只有在官方论文、权重、代码和许可证核验后，才作为扩展对照加入，不将二手描述直接写成已证实结论。

## 目录

- `docs/literature_review.md`：近三年重点论文、开源项目与可复用点
- `docs/reproduction_plan.md`：8 周复现实验路线、验收标准和消融矩阵
- `references/references.bib`：已核验 DOI/arXiv 的最小 BibTeX 记录
- `configs/`：基线与多模态域泛化配置
- `src/`：数据、模型、损失、训练与评估骨架
- `tools/build_manifest.py`：只扫描解压后的图片目录，生成待人工审核清单
- `tests/test_splits.py`：验证留一地点外与组隔离逻辑

## 数据清单格式

训练前创建 UTF-8 CSV，至少包含：

```text
sample_id,image_path,site,date,plant_id,plot_id,grade,weather_path,text_masked,label_verified
```

`grade` 原始值可为 `CK,1,3,5,7,9`；模型内部映射为 `0..5`。同一植株/小区/同日连拍必须用统一 `group_id` 进入同一数据子集。任何带等级、地点或日期文字的图像必须先遮挡，并令 `text_masked=true`。

## 最小使用顺序

1. 只生成清单，不改原图：

   ```bash
   conda run -n tb python tools/build_manifest.py \
     --image-root ../2024玉米锈病数据图片 \
     --output data/manifest_draft.csv
   ```

2. 人工补齐并复核 `site/date/plant_id/plot_id/grade`，完成文字区域遮挡；将最终文件保存为 `data/manifest.csv`。
3. 安装依赖须另行确认；建议依赖见 `requirements-proposed.txt`。
4. 训练示例（依赖齐全后）：

   ```bash
   conda run -n tb python -m src.train --config configs/multimodal_dg.yaml --held-out-site Jiaozuo
   ```

5. 评估：

   ```bash
   conda run -n tb python -m src.evaluate --predictions outputs/predictions.csv
   ```

## 当前边界

- `build_manifest.py` 不从文件名自动猜病情等级，以避免伪标签污染。
- 当前骨架不读取压缩包，也不修改原始图像或表格。
- 气象表字段、时间窗和图片—气象匹配规则需在数据字典确认后实现。
- 没有独立年份时，“跨地点泛化”可以完成；严格跨年份泛化需新增年份数据。
