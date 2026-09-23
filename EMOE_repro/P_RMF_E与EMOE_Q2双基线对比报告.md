# E 题问题 2：EMOE 与 P-RMF-E 双基线对比报告

**状态：双基线对比已完成；本报告只决定问题 2 的后续主干，不加入新增强模块。** 两模型各用 3 个随机种子完成训练、验证和附件 3 无标签推理。正式结果位于 `results/dual_baseline_v2/`；此前 `results/dual_baseline/` 的 6 种模态组合预实验不与本次 7 组合协议混算。

## 1. 结论先行

**建议问题 2 以 P-RMF-E 为后续主干，EMOE 作为强对照；问题 3 可保留 EMOE 现有 Router 解释接口。** P-RMF-E 在同一数据、同一连续缺失 mask、同一训练预算和选模规则下，3 个种子的 Clean 与 25 个缺失场景平均四指标均优于 EMOE，参数量约为后者的 34%，峰值显存约为 49%。代价是训练约 2.1 倍、推理吞吐约低 35%，且本次验证集上的绝对收益约为 0.02–0.03，不能宣称巨大优势或替代原论文完整复现。两模型对中性类别和强情感强度都有明显不足。

本次所说的 **P-RMF-E** 是以 [P-RMF 原论文](https://aclanthology.org/2025.acl-long.1075/)和[作者代码](https://github.com/hawksilent/P-RMF/tree/769c31b50f173b3f7670404a0939a95388393610)为主体进行 E 题必要适配的版本；**EMOE** 以 [CVPR 2025 原论文](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html)及现有复现为主体。以下数值都是本地 E 题协议所得，不是两篇原论文公开表格的横向比较。

## 2. 赛题数据及“防 AI 坑”核查

| 核查点 | 文件证据与本次处理 |
|---|---|
| 附件 2 输入 | `aligned_50.pkl`：train 3395、valid 728、test 727；`text` 为 50×768、`audio` 为 50×74、`vision` 为 50×35；本次只用 train/valid 学习与选模。train/valid 的样本 ID 均唯一且互不重合。 |
| 附件 3 Text 接口 | 30 个独立 pkl **均无 `text`**，有 `text_bert`（3×50）。若直接按 `item["text"]` 读取确会 `KeyError`。本次用附件 3 自带 token ID/attention mask 经本机冻结 `bert-base-uncased` 生成 50×768 特征；没有重新训练 BERT。对附件 2 中可精确配对的一条未遮挡样本，重建与原 `text` 的全 50 步平均绝对差为 `3.83×10⁻⁷`，验证了特征接口匹配。 |
| 附件 3 缺失形态 | 30 条中 27 条在有效内容区有 Text `[UNK]`（token ID 100），同为 27 条有 Audio 与 Vision 全零槽；29 条的三路缺失槽完全一致，其中包含 3 条无缺失，余下一条多出视觉零值。因此“附件 3 文本完全没有遮挡、只缺 A/V”的说法与 pkl 不符。附件 2 全部 4850 条的有效 Text 内容区无 `[UNK]`；可与附件 2 精确配对的 3 条被遮挡样本，其 `[UNK]` 位置原本是其他 token。证据强烈支持附件 3 的 Text 同段遮挡，但能精确配对的仅 4/30 条，不能逐条证明所有 `[UNK]` 的生成过程。 |
| 缺失类型 | 关键是**局部连续时间段**；本次训练既模拟单模态、两模态，也模拟三模态同段缺失。没有把整条序列清零当作唯一缺失机制。附件 3 直接使用文件中已有 `[UNK]` 和 A/V 零槽，不再叠加随机 mask。 |
| 标签定义 | 连续 `regression_labels=0` 是中性；预编码 `classification_labels` 则是 `0=Negative、1=Neutral、2=Positive`。全部 train/valid 样本满足 `classification_labels = sign(regression_labels)+1`。训练集中负/中/正为 967/758/1670，验证集为 206/184/338；没有将中性并入正类。 |
| 数据边界 | 附件 3、附件 4 均无标签；本次没有伪标签、没有用附件 3 选模型，也没有导入 MOSI、IEMOCAP 等外部情感数据集。P-RMF 仓库里的 MOSEI 配置只提供网络超参数模板，数据路径被 E 题附件 2 替换。 |

审计原始记录：`results/dual_baseline_v2/attachment3_structure.json`、`attachment2_unk_baseline.json`、`attachment3_origin_check.json`。一个确实踩过的旧问题是：早期 `q2_aligned.py::infer_special` 虽能从 `text_bert` 重建 Text，**却只由 A/V 零值生成缺失 mask，没有把 `[UNK]` 内容槽标为 Text 缺失**。旧版附件 3 CSV 不应作为三模态缺失处理正确性的证据；本次双基线入口 `q2_dual_attachment3.py` 已独立按三路现有缺失构造 mask。两者的预测不宜直接解释为单纯模型差异。

## 3. 方法与公平实验协议

### 3.1 两个模型分别保留了什么

| 模型 | 保留的原论文主体 | 本次 E 题必要适配 |
|---|---|---|
| EMOE | 三路模态编码、Router 动态加权、单模态预测及其蒸馏/平衡目标。 | 输入已对齐连续特征；三分类头和强度回归头；显式缺失 mask 与现有 missing token。采用 `StagedEMOE(control)`，关闭此前试验性的 padding 池化和覆盖率修正，随机初始化。 |
| P-RMF-E | 三路 token Transformer、三个 VAE、均值/方差形成的 proxy 与不确定度权重、梯度反转、四次共享权重的跨模态注入、重建器及原回归 MLP。 | 跳过重复 BERT，直接收 `Text 50×768`；A/V 都改为 50 步；随机散点缺失改为连续窗口；增加三分类头。作者仓库训练入口遗漏的重建器参数纳入 optimizer，使重建损失实际更新该模块。这是明确的代码修正，不能称原仓库逐行无改动复现。 |

P-RMF-E 的预测只经**不完整输入**形成 proxy 和融合结果；训练时完整输入仅作为 VAE/重建辅助目标，不直接进入分类或回归路径。两模型共同任务项为 `CE(三类)+L1(强度)`；EMOE 保留自身的单模态/Router/蒸馏辅助项，P-RMF-E 保留 `0.1×重建 MSE+0.5×VAE/KL`。辅助目标不同是模型家族本身的差异，不能解释为仅更换一个融合算子的严格消融。

### 3.2 固定比较协议

- **数据与预处理**：同一附件 2 `aligned_50` train/valid。Text 原 float32；A/V 将 NaN/Inf 归零后转 float32；用 `text_bert` attention mask 确定有效内容区，并识别源数据的 A/V 全零内容槽。模型内部 mask 顺序是 Text/Vision/Audio，CSV 输出顺序明确写出。验证集真值只用于指标和固定选模式。
- **训练缺失增强**：每个 epoch、每条训练样本预生成同一份 mask；80% 概率注入一个连续窗口，窗口占有效内容 15%–50%；等概率抽取 Text、Audio、Vision、任意双模态或三模态同段这 7 类。两模型同 seed 的样本顺序、位置和时长完全相同。EMOE 在缺失槽使用既有 missing token；P-RMF-E 按原结构以零填充，这是模型机制差异。
- **优化预算**：种子 1111/2222/3333；每种子 8 epoch，batch 16，AdamW `lr=1e-4`、`weight_decay=1e-4`，AMP，梯度裁剪 1.0；硬件 RTX 5060 Ti 16GB。未针对任一模型单独搜参或额外训练。
- **同一 checkpoint 规则**：只在 valid 的 Clean 与 A+V 中段 30% 上选一次 checkpoint：`Clean F1 + AV30 F1 + 0.25×(两项 Pearson 之和) − 0.25×(两项 MAE 之和)`。每个模型每种子只保存这一个 best checkpoint；不按 Accuracy 单独选模，也不用附件 2 test/附件 3 的标签或预测分布选模。
- **最终网格**：Clean + Text/Audio/Vision/A+V/三模态同段五种类型，各做头/中/尾 30% 和中段 10/30/50%，去重后为 26 个场景，其中 25 个缺失场景。两模型的每个场景同 mask、同 728 条 valid；报告标称比例和实际新增遮挡比例。

完整事前协议见 `P_RMF_E双基线统一协议.md`。结果脚本为 `q2_dual_baseline.py`、`q2_dual_summarize.py`、`q2_dual_cases.py`、`q2_dual_bootstrap.py`。上游源码固定修订号为 `769c31b50f173b3f7670404a0939a95388393610`。

## 4. 验证结果

下表均为 **3 种子均值 ± 样本标准差**；Accuracy、Macro-F1、Pearson 越高越好，MAE 越低越好。缺失网格均值是 25 个场景先逐 seed 平均、再跨 seed 平均，不是新增独立验证样本。

| 场景 | 模型 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---|---:|---:|---:|---:|
| Clean | EMOE | .6140 ± .0139 | .5818 ± .0122 | .6259 ± .0112 | .6119 ± .0090 |
| Clean | P-RMF-E | **.6392 ± .0029** | **.6069 ± .0038** | **.6025 ± .0083** | **.6320 ± .0097** |
| A+V 中段 30% | EMOE | .6154 ± .0076 | .5849 ± .0079 | .6252 ± .0065 | .6108 ± .0089 |
| A+V 中段 30% | P-RMF-E | **.6369 ± .0048** | **.6068 ± .0035** | **.5980 ± .0064** | **.6339 ± .0089** |
| 三模态同段 30% | EMOE | .6117 ± .0177 | .5799 ± .0119 | .6255 ± .0063 | .6073 ± .0077 |
| 三模态同段 30% | P-RMF-E | **.6387 ± .0109** | **.6030 ± .0128** | **.5963 ± .0043** | **.6273 ± .0091** |
| 三模态同段 50% | EMOE | .6003 ± .0206 | .5719 ± .0178 | .6262 ± .0033 | .6045 ± .0064 |
| 三模态同段 50% | P-RMF-E | **.6360 ± .0131** | **.5994 ± .0139** | **.6039 ± .0093** | **.6225 ± .0106** |
| 25 缺失场景均值 | EMOE | .6123 ± .0134 | .5814 ± .0102 | .6266 ± .0086 | .6086 ± .0093 |
| 25 缺失场景均值 | P-RMF-E | **.6367 ± .0057** | **.6033 ± .0064** | **.6005 ± .0061** | **.6297 ± .0092** |

**注：**未四舍五入的数值见 `results/dual_baseline_v2/summary.json`。Clean 的配对平均差（P-RMF-E − EMOE）为 Accuracy `+.0252`、Macro-F1 `+.0252`、MAE `−.0234`、Pearson `+.0200`；三模态同段 50% 为 `+.0357 / +.0274 / −.0223 / +.0180`。全部 3 个配对种子在这四项主要汇总指标上均为相同有利方向。

为避免把有限 valid 的波动说成确定结论，对 728 个 ID 做 1000 次**成对重采样**，每次同时用于两模型和全部三个种子；它只衡量样本抽样不确定度，不代替更多随机种子的方差。Clean 差值的 95% 区间：Accuracy `[+.0050,+.0458]`、Macro-F1 `[+.0027,+.0478]`、MAE `[−.0410,−.0054]`、Pearson `[−.0038,+.0447]`。三模态同段 50%：`[+.0156,+.0559] / [+.0043,+.0493] / [−.0396,−.0044] / [−.0098,+.0461]`。**Pearson 的区间跨 0；A+V 中段 30% 的分类区间也接近或跨 0，因此不能宣称所有单项优势均稳健显著。** 详细数值见 `paired_bootstrap.json`。

### 4.1 缺失类型、比例、位置与时长

- **类型**：P-RMF-E 在 Text、Audio、Vision、A+V 和三模态同段场景的绝对四指标大体领先。音频或视觉单独缺 30% 时，两模型相对各自 Clean 的降幅都很小；A+V 同段 30% 也没有显著额外崩溃。不能据此认定 A/V 不重要，因为强 Text 信号和 E 题源特征性质会掩盖它们的边际影响。
- **比例与时长**：同一条有效序列的连续缺失率从 10% 到 50%，窗口时长同步增加；两者在本协议下不能被独立识别为两个因果因素。P-RMF-E 的 **Text 中段 50%** Macro-F1 `.5869`，比其 Clean `.6069` 下降 `.0201`；三模态中段 50% 降至 `.5994`，下降 `.0076`。EMOE 对 Text 中段 50% 的 F1 下降 `.0035`，对三模态中段 50% 下降 `.0099`。P-RMF-E 的绝对性能仍高，但**相对自身 Clean 对长 Text 窗口更敏感**。所谓“更鲁棒”在这里应指绝对缺失场景表现，不应笼统说每种相对降幅都更小。
- **位置**：30% 的 Text 缺失，两模型都是序列开头更差、末尾更好：P-RMF-E F1 `.5899/.5972/.6002`，EMOE `.5741/.5824/.5848`（头/中/尾）。三模态同段 30%，P-RMF-E `.5960/.6030/.6039`，EMOE `.5788/.5799/.5848`。这是分组观察，并未证明序列头部必然承载更多情感词。
- **实际新增遮挡**：标称 Vision 30% 实际约 28%，因为源数据已经有全零视觉槽；三模态同段中段 30% 的实际新增比例约 29.4%。因此分析图与 CSV 同时保留 `actual_new_missing_fraction`，不能只报标称值。

对应图：`results/dual_baseline_v2/missing_rate_curves.png`、`missing_position_curves.png`；全部 26 场景的数值在 `scenario_mean.csv`，各 seed 原始记录在 `emoe/seed_*/scenario_grid.csv`、`prmf_e/seed_*/scenario_grid.csv`。

### 4.2 错误与失败案例

- **中性仍难**：Clean 验证集 184 条中性样本的平均召回率，EMOE `.370`、P-RMF-E `.380`，均明显低于各自负/正类约 `.70–.73`。Text 中段缺 50% 时，P-RMF-E 中性召回跌至 `.315`，是其长 Text 窗口相对性能下降的主要可观察瓶颈之一。标签为中性即强度 0，从未并入正类。
- **极性对但强度回归弱**：验证集中 `|y|≥1.5` 的 Clean MAE，EMOE `1.198`、P-RMF-E `1.128`；三模态中段 50% 时分别为 `1.252`、`1.321`，P-RMF-E 在极端强度子集反而更差。例：`2S00zYaVrtc$_$10` 真值 `−2.667`，三模态中段 50% 下两者都判负，但 EMOE 强度 `−1.866`、P-RMF-E `−0.300`（seed 1111）。这说明 P-RMF-E 的总体 MAE 优势不能推广到每条强情感样本。
- **相反的 EMOE 失败**：`f_ZJ7L14oYQ$_$27` 真值正类、强度 `+1.667`，Clean 下 EMOE 判负且回归 `−1.351`，P-RMF-E 判正且回归 `+0.209`（seed 1111）。P-RMF-E 此例仍明显低估强度，但极性识别更合理。
- **配对错误量**：Clean 共有 728×3=2184 个样本种子配对；P-RMF-E 独对 229 个，EMOE 独对 174 个，二者都错 614 个。三模态中段 50% 时分别独对 249 与 171 个、都错 624 个。不能把这些当 2184 个独立视频，因同一 728 条 valid 在不同 seed 重复出现。

逐样本核对文件为 `sample_predictions.csv`、`failure_cases_seed1111.csv`、`slice_report.json`。失败案例的文字摘录只用于定位 ID 和辅助人工复核，不能仅凭一句文本断定音频/视觉证据。

### 4.3 训练状态与资源

| 指标 | EMOE | P-RMF-E |
|---|---:|---:|
| 参数总量 | 22,530,412 | 7,740,036 |
| 8 epoch 训练时长，三种子均值 | 65.2 秒 | 138.0 秒 |
| 峰值 CUDA reserved，三种子均值 | 560.7 MiB | 276.0 MiB |
| Clean valid 推理吞吐，728 条/次 | 约 1494 条/秒 | 约 969 条/秒 |
| 最优 epoch（3 seed） | 5 / 5 / 4 | 5 / 7 / 3 |

训练损失均持续下降，但最佳 valid 选模分数通常在第 3–7 epoch 出现，第 8 epoch 均低于各自峰值；存在继续训练后泛化回落的迹象，所以用 valid checkpoint 而非最后 epoch。P-RMF-E 的最佳 epoch 离散度更大，然而其 Clean Accuracy/F1 跨 seed 波动较小。显存数值是 PyTorch **reserved** 峰值，不代表包含 BERT 附件 3 预处理或整机显存；训练时间包含模型迭代、不包含全网格后评估。

## 5. 附件 3 无标签推理与结果边界

`results/dual_baseline_v2/attachment3_unlabeled_predictions.csv` 含 30 样本 × 2 模型 × 3 seed = **180 行**，记录样本文件名、极性及三类概率、强度、有效长度、三路已有缺失槽计数和 `additional_random_mask_applied=False`。已核验每行唯一、概率和为 1、概率与强度均有限且在合法区间。`attachment3_inference_report.json` 保存接口说明。EMOE 30 条中有 21 条得到三种子一致类别，P-RMF-E 有 25 条；这只是无标签预测稳定性观察，**不是正确率**。附件 3 及附件 4 都无真值，不做性能评估、伪标签或主干选择依据。

## 6. 主干裁决及局限

P-RMF-E 更契合问题 2 的理由来自**本地对照结果**，不只是“论文题目含 incomplete”：它在 Clean、A+V 同段、三模态同段和 25 场景平均均有较好绝对极性与回归性能；3 seed 的主要汇总指标方向一致；参数和峰值显存适合本机 16GB 显卡。其 proxy/不确定度加权也为后续缺失鲁棒研究提供自然接口，但本轮没有叠加新不确定度模块或蒸馏方法。

需要保留的限制：① 相对自身 Clean，P-RMF-E 在长 Text 缺失上的 F1 降幅较大；② 中性与极端强度仍弱；③ 训练更慢，跨模态注入结构也比 EMOE Router 更难直接给出问题 3 的可验证解释；④ 只有 3 seed、一个固定 train/valid 划分、8 epoch 预算，部分 bootstrap 区间跨 0；⑤ 两模型同用已有的 50 步预对齐特征，尚未检验原始视频端到端误差。因而本结论限定为**附件 2 aligned_50 的问题 2 双基线选择**。EMOE 的现有 Router 路线仍可用于问题 3 解释对照。当前不继续做 CMAD、额外蒸馏或架构大改。

## 7. 可复查文件索引

- 统一协议：`P_RMF_E双基线统一协议.md`
- 模型与训练：`prmf_e.py`、`q2_dual_baseline.py`、`missing_protocol.py`
- 汇总：`results/dual_baseline_v2/summary.json`、`scenario_mean.csv`、`paired_bootstrap.json`
- 逐样本与错误：`sample_predictions.csv`、`slice_report.json`、`failure_cases_seed1111.csv`
- 训练日志与 checkpoint：`results/dual_baseline_v2/{emoe,prmf_e}/seed_{1111,2222,3333}/report.json`、`best.pt`
- 附件 3：`attachment3_structure.json`、`attachment3_origin_check.json`、`attachment3_unlabeled_predictions.csv`、`attachment3_inference_report.json`
- 图：`missing_rate_curves.png`、`missing_position_curves.png`
