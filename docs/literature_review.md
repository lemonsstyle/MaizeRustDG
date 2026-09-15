# 文献综述表与研究定位

检索关注 2023—2025 年的植物病害域泛化、多模态融合、病情等级识别和可复用开源实现。引用量与 GitHub 星数会持续变化，下表只把它们作为项目成熟度线索，不作为学术质量的唯一依据。

## 一、优先阅读论文

| 优先级 | 年份 | 论文与来源 | 任务/模态 | 核心方法与域设定 | 代码或数据 | 对本课题的可复用点 | 局限与注意事项 |
|---|---:|---|---|---|---|---|---|
| A | 2025 | [Domain generalization plant leaf disease recognition: Toward from laboratory to field](https://doi.org/10.1016/j.engappai.2025.111168), EAAI | 叶病 RGB | 实验室到田间的域泛化 | 未核验到官方代码 | 直接支撑“未见田间域”论证与对照设计 | 需阅读全文确认数据划分细节；不能把域适配结果等同域泛化 |
| A | 2023 | [Toward Generalization of Deep Learning-Based Plant Disease Identification Under Controlled and Field Conditions](https://doi.org/10.1109/ACCESS.2023.3240100), IEEE Access | 叶病 RGB | 控制环境/田间跨域识别 | 未核验到官方代码 | 适合做跨场景基线、分析背景偏差 | 病种分类不等于病情等级；需改为有序评价 |
| A | 2024 | [Beyond supervision: Harnessing self-supervised learning in unseen plant disease recognition](https://doi.org/10.1016/j.neucom.2024.128608), Neurocomputing | 叶病 RGB | 自监督表征、未见病害识别 | 未核验到官方代码 | 支撑 DINOv2/自监督预训练，降低小样本过拟合 | 未见类别与未见地点是不同问题，实验叙述要分开 |
| A | 2023 | [Identifying plant disease and severity from leaves](https://doi.org/10.1016/j.compag.2023.107809), CEA | RGB，病种+严重度 | 三分支 Swin Transformer、多任务与深监督 | 未核验到官方代码 | 可复用病种/等级多任务、等级辅助监督设计 | 复杂模型在 338 张图上高风险过拟合 |
| A | 2024 | [玉米典型叶部病害高光谱识别及其烈度分类](https://doi.org/10.12133/j.smartag.SA202310016) | 玉米叶病高光谱 | 病种识别和烈度分类 | 本地有论文 PDF；未核验代码 | 最贴近玉米南方锈病等级任务；可比较等级指标 | 传感器模态与本项目 RGB/气象不同；文中南方锈病烈度 OA 约 71.39%、Macro-F1 约 0.69，须引用原文表格核对 |
| A | 2024 | [Hybrid Ensemble Learning With CNN and RNN for Multimodal Cotton Plant Disease Detection](https://doi.org/10.1109/ACCESS.2024.3515843), IEEE Access | 图像+环境/序列 | CNN、RNN 多模态集成 | 未核验到官方代码 | 可借鉴图像—环境双分支与融合基线 | 棉花病害、任务定义和数据规模不同 |
| B | 2025 | [Multimodal few-shot learning for plant disease detection with contrastive pre-training and query addressal](https://doi.org/10.1007/s00521-025-11438-5), NCA | 多模态、小样本 | 对比预训练、少样本查询 | 未核验到官方代码 | 小数据情境下的对比学习与原型基线 | 需确认可获得模态与本地数据是否一致 |
| B | 2025 | [PlantIF: Multimodal semantic interactive fusion via graph learning for plant disease diagnosis](https://doi.org/10.1016/j.plaphe.2025.100132), Plant Phenomics | 多模态 | 图学习、语义交互融合 | 未核验到官方代码 | 可作为高级融合方案的后续改进 | 图结构复杂，优先级低于可靠的门控融合基线 |
| B | 2024 | [AI Meets Climate: Advanced Transformer Multimodal Model for Early and Accurate Plant Disease Detection](https://doi.org/10.1109/ICCED64257.2024.10983899), ICCED | 图像+气候 | Transformer 多模态 | 未核验到官方代码 | 直接支撑气象信息参与识别的可行性 | 会议论文；需仔细核对数据与验证设计 |
| B | 2023 | [Improved domain adaptive rice disease image recognition based on a novel attention mechanism](https://doi.org/10.1016/j.compag.2023.107806), CEA | 水稻病害 RGB | 注意力、域适配 | 未核验到官方代码 | 可作为“允许访问目标域”的上界实验 | 属于域适配，不是主论文要求的域泛化，不能混为主结果 |
| B | 2024 | [PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation](https://arxiv.org/abs/2409.04038) | 野外病害分割 | 大规模病斑分割数据和 MMSeg 框架 | [GitHub](https://github.com/tqwei05/PlantSeg) / [Zenodo](https://zenodo.org/records/14935094) | 叶片/病斑 ROI、弱化田间背景和文字泄露 | 仓库未见明确许可证声明时，不应直接复制代码进论文项目 |

## 二、优先复用的开源代码

| 项目 | 用途 | 成熟度线索（2026-09-15 快照，约数） | 推荐用法 | 风险 |
|---|---|---:|---|---|
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | 自监督视觉表征 | 13.3k stars | 冻结/微调图像编码器；与监督 CNN 做对照 | 小数据全量微调易过拟合；注意权重许可 |
| [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | ConvNeXt、ViT 等统一实现 | 37.1k | 先做 ImageNet 预训练基线 | 不同模型预处理必须固定并记录 |
| [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | 叶片/病斑分割 | 10.0k，Apache-2.0 | 训练 SegFormer/SegNeXt，生成 ROI | 分割标注成本；分割误差会传递到等级识别 |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | 交互式/自动分割 | 19.9k，Apache-2.0 | 辅助生成叶片掩膜，再人工校正 | 不是植物病斑专用，不能把自动掩膜当真值 |
| [Raschka-research-group/coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch) | 有序等级学习 | 279 | CORAL ordinal head 与损失参考 | 与 Deep CORAL 域对齐同名，论文中必须消歧 |
| [KaiyangZhou/mixstyle-release](https://github.com/KaiyangZhou/mixstyle-release) | 单源/多源域泛化 | 335 | 在图像中间特征做风格混合消融 | 需要检查与 Transformer 特征形状的兼容性 |
| [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed) | 域泛化基准 | 1.6k，已归档 | 借鉴数据划分、算法接口和公平调参协议 | 已归档；不建议作为新项目主体依赖 |
| [KaiyangZhou/Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) | 域泛化/域适配研究框架 | 1.4k | 算法实现核对、实验配置参考 | 框架较重，本地小数据先用轻量骨架 |
| [timeseriesAI/tsai](https://github.com/timeseriesAI/tsai) | TST/InceptionTime 等时序模型 | 6.1k | 气象编码器原型 | 与 PyTorch 训练栈集成前需固定输入标准化 |
| [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | 可解释性 | 13.0k | Grad-CAM 检查模型是否看病斑而非文字/背景 | 热力图不是因果解释，需结合遮挡实验 |
| [shubham10divakar/Multimodal-Plant-Disease-Dataset](https://github.com/shubham10divakar/Multimodal-Plant-Disease-Dataset) | 多模态数据组织参考 | 约 2 stars | 只参考 CSV/模态配对方式 | 项目较弱且未见许可证，不作为核心技术依据 |

## 三、研究空缺与论文创新落点

现有工作往往只覆盖其中一项：病种识别、严重度分类、多模态融合或跨域泛化。本项目可以把创新集中在以下三个可检验问题上：

1. **等级有序性**：将 CK/1/3/5/7/9 作为有序变量，而非互不相关的六分类；比较交叉熵、CORAL ordinal regression、EMD/QWK 优化。
2. **图像—天气互补性**：使用采样日前 24/72/168 小时温湿度序列，用门控机制处理缺失天气；验证气象是否只在早期/视觉症状模糊样本上增益。
3. **未见地点泛化**：郑州（南郊）、长葛、焦作轮流作为完全隔离目标域；比较 ERM、MixStyle、源域 Deep CORAL、自监督表征及组合方法。

推荐主创新不是堆叠最多模块，而是“**有序学习 + 缺失感知多模态融合 + 不访问目标域的严格 LOSO 验证**”。如数据量不足，DINOv2 冻结特征与轻量门控融合通常比训练大型交叉注意力模型更可信。

## 四、综述写作结构

1. 玉米南方锈病监测与等级标准：发生规律、气象驱动、传统调查和高光谱研究。
2. 深度学习植物病害识别：从受控背景分类到自然田间识别。
3. 病情严重度估计：普通多分类、分割面积比、有序回归和多任务学习。
4. 多模态融合：图像与天气/生理指标的早期、晚期和交互融合。
5. 跨域问题：域适配与域泛化的区别、严格目标域隔离和常见数据泄露。
6. 研究空缺：针对真实田间、少样本、有序等级和跨地点的统一方法。

## 五、引用与核验说明

- DOI/arXiv 链接和项目主页已作为可追溯入口列出。
- 引用量与 stars 是动态近似值，正式论文中不要作为结论性证据。
- 对未核验到官方代码的论文明确标注，不用第三方同名仓库冒充官方实现。
- 本地中文资料可用于农学背景，但具体结论、数值和标准需逐篇回到原文页码核验。
