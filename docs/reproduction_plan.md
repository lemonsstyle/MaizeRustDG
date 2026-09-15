# 复现实验路线

## 0. 可证伪目标

主假设 H1：在完全未见地点测试中，多模态有序模型相对图像 ERM 基线显著提高 Macro-F1 和 QWK，并降低 MAE。
辅助假设 H2：天气模态对早期/视觉模糊样本的收益更大。
辅助假设 H3：病斑/叶片 ROI 与域泛化正则能降低模型对文字、土壤和拍摄设备的依赖。

如果三轮 LOSO 的平均改进不稳定，或置信区间覆盖零，应如实报告模型没有稳定收益，而不是只选最优地点。

## 1. 数据协议

- 来源域：郑州（南郊）、长葛、焦作；统一标准地点名称。
- 标签：`CK,1,3,5,7,9 → 0,1,2,3,4,5`；双人复核，冲突由第三人裁决。
- 分组：同一植株/小区/同日连拍为一个 `group_id`，不得跨 train/val/test。
- 天气窗口：主实验用采样结束前 72 h；24 h、168 h 为消融。仅用预测时刻之前的观测。
- 图像泄露：遮挡可见的地点、日期、等级文字；保留遮挡框审计记录。
- 缺失模态：记录 `weather_missing`，不使用全数据均值；标准化参数只从当前训练域拟合。
- 目标域隔离：每轮目标地点只在最终评估阶段打开，模型选择不能查看目标域指标。

## 2. 八周路线与验收

| 周次 | 工作 | 产出 | 验收标准 |
|---:|---|---|---|
| 1 | 数据审计、标签字典、文字遮挡、图像—天气配对 | `manifest.csv`、数据字典、审计报告、30 张双人标注样例 | 路径可读率 100%；必填字段无缺失；抽样一致率和冲突清单可追溯；无原图被覆盖 |
| 2 | 图像 ERM 基线 | ResNet/ConvNeXt-Tiny 六分类与训练日志 | 三轮 LOSO 均可重复；随机种子固定；输出 Macro-F1、QWK、MAE、混淆矩阵 |
| 3 | 叶片/病斑 ROI | SAM2 辅助标注 + SegFormer/PlantSeg 迁移；原图与 ROI 对照 | 独立掩膜集报告 Dice/IoU；遮挡文字后性能不反常暴跌；可视化覆盖病斑而非文字 |
| 4 | 自监督图像表征与有序学习 | DINOv2 冻结/微调；CE vs CORAL ordinal | QWK/MAE 至少一项稳定改善；阈值输出保持单调；记录参数量和训练时间 |
| 5 | 域泛化 | ERM、MixStyle、源域 Deep CORAL、组合方法 | 不访问目标域；三地点完整轮换；每法 3–5 个种子；报告均值、标准差和置信区间 |
| 6 | 气象编码与融合 | MLP 汇总、InceptionTime/TST；拼接、门控、交叉注意力 | 做 image-only/weather-only/multimodal；天气打乱后增益应消失或显著下降 |
| 7 | 消融、鲁棒性、解释 | 模块消融、缺失天气、文字遮挡、背景遮挡、Grad-CAM | 每个创新点都有独立对照；失败样本按地点/日期/等级归因 |
| 8 | 统计分析与论文图表 | 汇总表、置信区间、显著性检验、方法图、数据流程图 | 主表可由脚本重建；结论与表格一致；保留配置、种子、提交哈希和环境信息 |

## 3. 分阶段复现顺序

### 阶段 A：最小可信基线

1. 固定 3 个 LOSO 外层测试折。
2. 在两个源地点内部按 `group_id` 做 80/20 train/val。
3. 训练 ImageNet 预训练 ConvNeXt-Tiny + CE。
4. 同一划分换 CORAL ordinal head。
5. 每轮至少 3 个随机种子；不以目标地点挑超参数。

通过条件：流程可重复、无跨组泄露、所有指标能从逐样本预测重算。

### 阶段 B：复现开源部件

- PlantSeg/MMSeg：先在公开数据配置上复现作者给出的验证流程，再迁移到本地掩膜。
- DINOv2：先冻结编码器只训线性/有序头；再解冻最后 1–2 blocks。
- MixStyle：优先插入 CNN 中间层；Transformer 版本先验证张量维度与位置语义。
- coral-pytorch：用玩具有序数据核对标签 levels 和推理解码。
- tsai：以固定 72 h 时间窗复现 InceptionTime/TST 的输入输出。

通过条件：每个外部部件都有独立最小测试，且许可证和版本记录完整。

### 阶段 C：提出模型

主模型采用缺失感知门控融合：

```text
z_img = ImageEncoder(mask(image))
z_w   = WeatherEncoder(sequence, valid_mask)
g     = sigmoid(MLP([z_img, z_w, weather_missing]))
z     = LayerNorm(z_img + g * Project(z_w))
ordinal_logits = OrdinalHead(z)
```

总损失：

`L = L_ordinal + λ_domain L_deep_coral + λ_aux L_aux_classification`

Deep CORAL 只在同一训练 batch 的不同**源地点**特征之间计算；若 batch 只有一个域，该项返回 0。

## 4. 最小实验矩阵

| ID | 图像 | 天气 | 等级损失 | 域泛化 | 目的 |
|---|---|---|---|---|---|
| E0 | ConvNeXt | 无 | CE | 无 | 图像 ERM 基线 |
| E1 | ConvNeXt | 无 | CORAL ordinal | 无 | 有序性贡献 |
| E2 | DINOv2 frozen | 无 | CORAL ordinal | 无 | 自监督表征贡献 |
| E3 | DINOv2/ConvNeXt | 72h MLP | CORAL ordinal | 无 | 简单多模态贡献 |
| E4 | 同 E3 | 72h TST | CORAL ordinal | 无 | 时序编码贡献 |
| E5 | 同 E4 | 72h TST | CORAL ordinal | MixStyle | 风格泛化贡献 |
| E6 | 同 E4 | 72h TST | CORAL ordinal | Deep CORAL | 特征对齐贡献 |
| E7 | 同 E4 | 72h TST | CORAL ordinal | 最优 DG | 完整模型 |
| E8 | 同 E7 | 打乱天气 | 同 E7 | 同 E7 | 排除天气伪相关 |
| E9 | 同 E7 | 缺失 30% | 同 E7 | 同 E7 | 缺失模态鲁棒性 |

所有 E0–E9 使用完全相同的外层 LOSO 和源域验证组；预算不足时先跑 E0/E1/E3/E6/E8。

## 5. 指标与统计

- 主指标：Macro-F1、Quadratic Weighted Kappa（QWK）、等级 MAE。
- 次指标：balanced accuracy、每等级 recall、±1 等级准确率、ECE 校准误差。
- 分割：Dice、mIoU；同时报告分割失败对等级识别的影响。
- 不平衡：主表不以 accuracy 单指标下结论；训练可比较 class-balanced sampler/focal loss，但验证和测试保持自然分布。
- 统计：以“外层地点 × 随机种子”为重复单位，报告均值±标准差和 bootstrap 95% CI；配对比较使用同一划分、同一种子。

## 6. 关键失败模式和检查点

| 风险 | 诊断 | 处理 |
|---|---|---|
| 等级文字泄露 | 原图高分、遮字后崩溃；Grad-CAM 指向文字 | 所有模态统一遮字；发布遮挡审计 |
| 连拍泄露 | 近重复图出现在不同集合 | 感知哈希 + group split；人工检查近邻 |
| 目标域偷看 | 按测试地点结果选模型 | 超参只由源域 inner-validation 决定 |
| 天气伪相关 | 打乱天气仍同样增益 | 重查时间/地点编码泄露；显式去除 site 字段 |
| 小样本过拟合 | train 高、源域 val/目标域低 | 冻结骨干、减少融合参数、增强和正则化 |
| 标签主观性 | 相邻等级混淆、标注者差异大 | 双人复核，报告 weighted kappa；可改为软标签 |
| 病斑分割误差 | ROI 模型低于原图 | 原图+ROI 双视图或先只做叶片分割 |

## 7. 论文主表模板

| 方法 | RGB | 天气 | Ordinal | DG | 郑州→test | 长葛→test | 焦作→test | 平均 Macro-F1 | 平均 QWK | MAE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ERM-CE | ✓ |  |  |  |  |  |  |  |  |  |
| ERM-CORAL | ✓ |  | ✓ |  |  |  |  |  |  |  |
| Multimodal | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |
| Proposed | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |
