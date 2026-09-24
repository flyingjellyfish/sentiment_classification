# Q2：`[UNK]` 重编码与 padding 清零增量实验

日期：2026-09-24。状态：**完成协议差异诊断、四种轻量候选的 3 seed 训练与 26 场景评估，并对冻结的中性加权候选做了一次附件 2 独立 test 检查；未晋升新 Q2 主干。**全部学习和选模只用附件 2 `aligned_50` 的 train/valid；附件 2 test 仅作最后评价；附件 3/4 未用于训练或指标计算。正式双基线 checkpoint、旧结果和专项 CSV 均未覆盖。

## 1. 出发点和要检验的假设

原 Q2 协议先对完整词元 `u` 提取 BERT 特征 `B(u)`，再将缺失槽置零，即 `M⊙B(u)`。附件 3 给的是含 `[UNK]` 的 `text_bert`，推理实际为 `M⊙B(C(u))`，其中 `C` 先替换缺失词元。BERT 的上下文编码使两者不等；原模型还会接收非零的文本 padding 表示。[此前独立审查](Q2_Q3_pipeline独立评审.md)已经在 8 条样本上检查了表示差异，但没有量出下游指标差异。

本轮预先做两步：①冻结原 3 seed P-RMF-E，在**同 valid、同缺失位置**比较两种输入协议；②固定模型结构、mask、训练预算、任务/辅助损失与 checkpoint 选择规则，仅把训练中的 Text 缺失改为先置 `[UNK]` 再冻结 BERT 重编码，并测试将有效长度之后的原特征置零。这里的 padding **清零不是 Transformer attention mask**；位置编码和线性层偏置仍在，不能写成“完全屏蔽 padding”。

## 2. 实现和公平口径

- 输入仍为 T `B×50×768`、A `B×50×74`、V `B×50×35`，内部观测 mask 为 L/V/A 顺序 `B×3×50`。连续窗口位置、模态组合、epoch、随机种子、batch 顺序与原双基线相同。
- 对 Text 真正缺失的内容词元，把附件 2 `text_bert` 的 token ID 置为 `[UNK]=100`，attention mask 与 segment ID 保持原样，送入本地**冻结** `bert-base-uncased`。BERT 重新计算整段 50×768 表示，之后 P-RMF-E 仍在缺失 Text 槽乘观测 mask；非缺失样本直接保留附件 2 原 `text`。
- 训练辅助重建中的完整 Text target 仍用附件 2 原 `text`，不会错误地把已损坏表示当作“完整目标”。为此 [P-RMF-E 适配](../EMOE_repro/prmf_e.py)新增可选 `complete_text`；默认参数不变，历史 checkpoint 可按原路径严格加载。可选 `zero_padding` 同样默认关闭。
- 候选从头训练 3 seed `1111/2222/3333`，8 epoch、batch 16、AdamW `1e-4/1e-4`、AMP，仍按 Clean + A/V 中段 30% 的既定复合 valid 分数选 checkpoint。**模型参数量不变**。与原基线比较时，两个模型都用同一份重编码验证特征与同一 mask。
- 评估 Clean + 25 缺失场景，包含 Text/Audio/Vision/A+V/三模态的头中尾与 10/30/50% 条件。新的“真实重编码模拟”仍是对附件 2 valid 的人工单段缺失，并非附件 3 有标签性能，更不能完全代表附件 3 的多段/源缺失分布。

实现入口：[实验脚本](../EMOE_repro/q2_unk_reencode_experiment.py)；原始产物目录 `EMOE_repro/results/unk_reencode/`。主要文件为 `protocol_gap.json`、`comparison_pad_full.json`、`comparison_gated_pad_full.json`、`comparison_gated_split_full.json`、`comparison_gated_neutral_1.5_full.json`、`class_slices_gated_neutral_1.5.json`、`frozen_test_baseline_vs_gated_neutral.json`，以及单因素、集成检查文件和各候选 `seed_*/history.json`/`best.pt`。结果目录是本地实验产物，应在需要复现/归档时保留；本文保留关键数值口径。

## 3. 先量输入协议差距：冻结原 P-RMF-E，不重训

以下均为原 3 个 checkpoint 在**同 728 valid、同窗口**下的均值。“后置”即旧协议 `M⊙B(u)`；“重编码”即本轮 `M⊙B(C(u))`。

| 场景 | 后置 F1 | 重编码 F1 | 后置 MAE | 重编码 MAE |
|---|---:|---:|---:|---:|
| Clean | .6069 | .6069 | .6025 | .6025 |
| Text 头 30% | .5899 | .5404 | .6061 | .6469 |
| Text 中 30% | .5972 | .5570 | .5984 | .6407 |
| Text 尾 30% | .6002 | .5353 | .6049 | .6531 |
| Text 中 50% | .5869 | **.4914** | .6027 | **.6860** |
| 三模态同段中 30% | .6030 | .5570 | .5963 | .6394 |
| 三模态同段中 50% | .5994 | **.4996** | .6039 | **.6857** |

因此，旧协议下“Text 中段 50% 比 Clean 只降约 .020 F1”不能当作附件 3 机制的鲁棒性结论；在当前模拟的重编码协议下，冻结模型相对 Clean 下降约 **.1155 F1**。这些数值是输入链变换的**验证集响应**，不是附件 3 真值或重编码训练后的结果。[逐 seed 对比](../EMOE_repro/results/unk_reencode/protocol_gap.json)。

## 4. 主候选：重编码训练 + padding 清零，3 seed

下表均在同一**重编码验证协议**下评价，左列为既有 P-RMF-E，右列为新候选；各单项为 3 seed 均值，25 场景均值还先在每 seed 内平均 25 个场景。Accuracy/F1/Pearson 越高越好，MAE 越低越好。

| 场景 | 模型 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---|---:|---:|---:|---:|
| Clean | 原基线 | **.6392** | **.6069** | .6025 | .6320 |
| Clean | 重编码+清零 | .6268 | .5899 | **.5877** | **.6507** |
| Text 中 50% | 原基线 | .5440 | .4914 | .6860 | .5002 |
| Text 中 50% | 重编码+清零 | **.5733** | **.5246** | **.6510** | **.5333** |
| 三模态中 50% | 原基线 | .5444 | .4996 | .6857 | .4993 |
| 三模态中 50% | 重编码+清零 | **.5719** | **.5273** | **.6485** | **.5337** |
| **25 缺失场景均值** | 原基线 | **.6158** | **.5815** | .6179 | .6073 |
| **25 缺失场景均值** | 重编码+清零 | .6147 | .5739 | **.6013** | **.6276** |

Text 中 50% 的配对均值变化为 **ΔF1 +.0332、ΔMAE −.0350**，三模态中 50% 为 **+.0278/−.0372**；3 个 seed 在这两个重缺失场景的 F1/MAE 方向均有利。Clean 则 **ΔF1 −.0170**、ΔMAE **−.0148**；Clean F1 三种子变化为 `−.0263/−.0042/−.0207`。25 场景平均 **ΔAccuracy −.0011、ΔF1 −.0076、ΔMAE −.0166、ΔPearson +.0203**，因分类总指标退步，不能称整体全面涨点。

按模态类型分组，Text 5 场景平均 F1 原/新 `.5427/.5512`，三模态 `.5450/.5524`；但 Audio `.6066/.5878`、Vision `.6071/.5899`、A+V `.6061/.5881`。这是核心取舍：**受 Text 重编码影响的缺失类型更好，未缺 Text 的场景承受与 Clean 类似的分类损失**。各组平均 MAE 与 Pearson 则均朝新候选有利方向变化。[全部 156 条模型×seed×场景记录](../EMOE_repro/results/unk_reencode/comparison_pad_full.json)。

## 5. 单因素和额外训练尝试

为区分两个处理因素，本轮在 seed 1111 同预算做了 2×2 的部分对照；除原基线与主候选之外，另训“仅重编码”“仅 padding 清零”。再试一次从原 checkpoint 低学习率微调，以及对完整输入加权重 `0.5` 的额外 `CE+L1`。后两项仅 1 seed，是探索而非晋升依据。

| seed 1111 方案 | Clean F1 / MAE | 重编码 Text 中 50% F1 / MAE | 重编码三模态中 50% F1 / MAE | 解读 |
|---|---|---|---|---|
| 原后置训练、不清零 | .6027 / .5945 | .4869 / .6704 | .4958 / .6723 | 固定参照 |
| **仅重编码训练** | .5845 / .5989 | .4571 / .6677 | .4614 / .6658 | 单独提前 Text 缺失并未改善分类 |
| **仅 padding 清零、旧后置训练** | .5742 / .5942 | .5225 / .6427 | .5154 / .6445 | 重缺失提升，但 Clean 分类下降 |
| **重编码训练+padding 清零** | .5764 / .5926 | **.5423 / .6404** | .5392 / .6409 | 该 seed 重缺失更强，Clean 仍降 |
| 原 checkpoint 起步、清零+重编码，3 轮 `2e-5` | .5695 / .6167 | .4466 / .6483 | .4653 / .6493 | 低学习率续训未解取舍 |
| 清零+重编码，并额外完整输入监督 `0.5×(CE+L1)` | .5840 / **.5781** | .5233 / **.6387** | .5387 / **.6367** | 回归有利，Clean F1 未恢复到原基线 |

对“同一原 checkpoint 仅在推理时把 padding 清零”的 3 seed 完整网格实验为**明显负结果**：Clean F1 `.6069→.5184`，25 缺失场景均值 `.5815→.4990`。训练与推理的特征规则必须匹配，不能把清零当作无需重训的后处理。该实验也不证明从头做更严格 attention masking 必然无效。[推理侧对照](../EMOE_repro/results/unk_reencode/pad_inference_full.json)。

## 6. 同规模集成检查

为防止把原 3 seed 集成与新单模型不公平比较，本轮还对**原三模型**与**新三模型**各自作多数决分类、回归均值；分类平票用三模型平均概率作确定性破局，因此数值可能与旧 `Counter.most_common` 略有差异。

| 重编码验证场景 | 原 3 模型投票 F1 / MAE | 新 3 模型投票 F1 / MAE |
|---|---:|---:|
| Clean | **.6289 / .5873** | .5922 / **.5748** |
| Text 中 50% | .4903 / .6718 | **.5241 / .6383** |
| 三模态中 50% | .5012 / .6736 | **.5379 / .6375** |

两组三模型概率各取 50% 的六模型混合，在上述三场景 F1 分别是 `.6196/.5187/.5165`：能缓和取舍，但 Clean 仍不及原三模型投票，且六模型体积/推理代价更大。本轮只检查了 7 个代表性场景；不将六模型混合记作完整 25 场景结论，也不把集成写成新的网络模块。[集成对照](../EMOE_repro/results/unk_reencode/ensemble_3x3.json)。

## 7. 条件式 padding 与独立缺失分类头：Accuracy 取舍复核

由于“所有样本都清零 padding”损失 Clean 分类，我们继续做了两项**小改动**，仍固定三 seed、8 epoch、相同增强、同一选模规则，并在同一完整重编码验证网格评估：

1. **条件式 padding 清零（`gated_pad`）**：仅当某条样本的 Text 有内容槽缺失时，清零该条样本有效长度之后的原特征；Text 完整时保留原输入。**没有新增参数**。它处理的是 padding 数值，不是 attention mask。
2. **条件式清零 + 双分类头（`gated_split`）**：共享原 P-RMF-E 编码、proxy、融合和回归，仅为“Text 内容有缺失”增加一个 `128→3` 的分类头，约 **387 个参数**；根据观测 mask 选分类头。新头用共享分类头权重初始化。属于本轮自行设计，不是 P-RMF 原论文模块。

| 真实重编码协议的 valid 场景 | 方案 | Accuracy↑ | Macro-F1↑ | MAE↓ | Pearson↑ |
|---|---|---:|---:|---:|---:|
| Clean | 原基线 | .6392 | .6069 | .6025 | .6320 |
| Clean | 全部 padding 清零 + 重编码训练 | .6268 | .5899 | **.5877** | **.6507** |
| Clean | **条件式清零** | **.6429** | .6063 | .5997 | .6473 |
| Clean | 条件式清零 + 双分类头 | .6355 | **.6101** | .5919 | .6441 |
| 25 缺失场景均值 | 原基线 | .6158 | **.5815** | .6179 | .6073 |
| 25 缺失场景均值 | 全部 padding 清零 + 重编码训练 | .6147 | .5739 | **.6013** | **.6276** |
| 25 缺失场景均值 | **条件式清零** | **.6244** | .5758 | .6100 | .6231 |
| 25 缺失场景均值 | 条件式清零 + 双分类头 | .6115 | .5743 | .6068 | .6180 |

**直接回答 Accuracy：**条件式清零使 Clean Accuracy `+.0037`，25 场景平均 `+.0086`；后者在 seed 1111/2222/3333 的平均配对差分别为 `+.0119/+.0095/+.0045`，方向一致。但 25 场景 Macro-F1 `−.0057`，其中 Text 五场景均值 `.5427→.5192`、三模态五场景 `.5450→.5267`；Audio/Vision/A+V 场景的 F1 则略升。这表明 Accuracy 收益**没有均匀覆盖三类及全部缺失类型**。条件式清零的 Text 中段 50% F1 均值只 `.4914→.4926`，不像“全部清零”的 `.4914→.5246` 那样稳定改善重 Text 缺失。

双分类头在 seed 1111 的代表场景很亮眼：Clean Accuracy/F1 `.6401/.6027→.6538/.6199`，Text 中段 50% F1 `.4869→.5331`；但三 seed 完整网格反转：25 场景平均 Accuracy `−.0043`、F1 `−.0072`，Text 中段 30% 的 F1 配对差为 `−.0068/−.0092/−.0805`。这说明“一个 seed 同时全涨”不能作为推广证据。该候选**不晋升**。[条件式清零完整记录](../EMOE_repro/results/unk_reencode/comparison_gated_pad_full.json)、[双头完整记录](../EMOE_repro/results/unk_reencode/comparison_gated_split_full.json)。

## 8. 缺失时中性样本加权：3 seed valid 完整网格

针对条件式清零在 Text 缺失时中性召回下降的问题，额外训练一个**条件式清零 + 缺失中性样本分类损失权重 1.5**的候选 `gated_neutral_1.5`。只在训练样本的 Text 内容有缺失且真实类别为 Neutral 时，令该样本交叉熵权重为 1.5，按权重和归一；其余样本权重为 1。回归及 P-RMF-E 原辅助项不变。1.5 是本轮基于训练分布固定的单一取值，没有在 valid 上搜索多个系数。该加权和条件式 padding 都是**本轮适配尝试**，并非 P-RMF 原方法；不增加模型参数。

| 重编码 valid 场景，3 seed 单模型均值 | 方案 | Accuracy↑ | Macro-F1↑ | MAE↓ | Pearson↑ |
|---|---|---:|---:|---:|---:|
| Clean | 原基线 | **.6392** | **.6069** | .6025 | .6320 |
| Clean | 缺失中性加权 | .6387 | .6012 | **.5916** | **.6493** |
| 25 缺失场景均值 | 原基线 | .6158 | .5815 | .6179 | .6073 |
| 25 缺失场景均值 | 缺失中性加权 | **.6180** | **.5846** | **.6032** | **.6256** |
| Text 中段 50% | 原基线 | .5440 | .4914 | .6860 | .5002 |
| Text 中段 50% | 缺失中性加权 | **.5650** | **.5436** | **.6547** | **.5245** |
| 三模态中段 50% | 原基线 | .5444 | .4996 | .6857 | .4993 |
| 三模态中段 50% | 缺失中性加权 | **.5586** | **.5396** | **.6572** | **.5256** |

25 场景平均 Accuracy **+.0022**，Macro-F1 **+.0031**，MAE **−.0147**，Pearson **+.0183**；Text 与三模态各五场景 F1 分别 `.5427→.5602`、`.5450→.5613`，但 Audio、Vision、A+V 的 F1 均约下降 `.006`。Clean 分类也没有提高。三个 seed 的 25 场景平均 F1 配对差为 `+.0139/+.0041/−.0086`；因此总体均值的小幅增益仍有种子波动。Text 中段 50% 的 F1 则在三个 seed 均提高（`+.0595/+.0206/+.0763`），是更稳的局部收益。[完整网格](../EMOE_repro/results/unk_reencode/comparison_gated_neutral_1.5_full.json)。

按真实类别复核，Text 中段 50% 的 Neutral 召回 `.219→.442`、Positive `.606→.649`，同时 Negative `.733→.537`；三模态中段 50% 的 Neutral `.263→.466`、Negative `.681→.510`。分类收益伴随类别错误转移，不能只报宏 F1。[类别切片](../EMOE_repro/results/unk_reencode/class_slices_gated_neutral_1.5.json)。

## 9. 冻结后附件 2 test 独立检查

下述检查方案在读取 test 结果前确定：只比较**既有 P-RMF-E 三 seed**和**`gated_neutral_1.5` 三 seed**；同一附件 2 `aligned_50` test 727 条、Clean + 25 个连续缺失条件，Text 缺失均先把 token 换为 `[UNK]`、经冻结 BERT 重编码。结果是**三个单模型指标的均值**，25 场景行再对场景平均；不是多数决集成。test 不用于训练、调权重、选 epoch 或更换候选，附件 3/4 仍无标签且未参与该比较。

| 独立 test 场景 | 模型 | Accuracy↑ | Macro-F1↑ | MAE↓ | Pearson↑ |
|---|---|---:|---:|---:|---:|
| Clean | 原基线 | **.6717** | **.6133** | .6464 | .6631 |
| Clean | 缺失中性加权 | .6657 | .5977 | **.6442** | .6631 |
| 25 缺失场景均值 | 原基线 | .6458 | .5871 | .6779 | .6194 |
| 25 缺失场景均值 | 缺失中性加权 | **.6471** | **.5883** | **.6681** | **.6282** |
| Text 中段 50% | 原基线 | .5571 | .4845 | .7839 | .4771 |
| Text 中段 50% | 缺失中性加权 | **.5740** | **.5358** | **.7458** | **.5163** |
| 三模态中段 50% | 原基线 | .5589 | .4966 | .7846 | .4734 |
| 三模态中段 50% | 缺失中性加权 | **.5768** | **.5443** | **.7493** | **.5176** |

**Accuracy 结论：**独立 test 的 25 缺失场景平均仅 `.6458→.6471`（**+.0013**），Clean `.6717→.6657`（**−.0060**）；Text 中段 50% `.5571→.5740`（**+.0169**），三模态中段 50% `.5589→.5768`（**+.0179**）。后两者同时改善 F1/MAE/Pearson。按模态组，Text 五场景 Accuracy `.6082→.6215`、三模态 `.6087→.6215`；Audio、Vision、A+V 五场景的 Accuracy 分别 `.6704→.6642`、`.6702→.6644`、`.6714→.6638`。它更适合作为**文字缺失专项候选**，不构成整体替换原 Q2 主干的证据。

25 场景平均 Accuracy 逐 seed 配对变化为 `−.0055/+.0145/−.0052`，F1 为 `−.0032/−.0046/+.0114`。整体 `.0013/.0012` 的均值优势未跨 seed 保持同向；25 个场景是同一批 727 条样本的重复扰动，不能当作 25 份独立检验。相较之下，Text 中段 50% test 的 F1 `.4845→.5358`、MAE `.7839→.7458` 与 valid 的定向现象一致。[独立 test 156 条记录及元数据](../EMOE_repro/results/unk_reencode/frozen_test_baseline_vs_gated_neutral.json)。

## 10. 当前决定与下一轮真正值得做的事

1. **不替换现有正式 Q2 checkpoint 或附件 3 CSV。**全 padding 清零候选有 Text 重缺失与回归收益，但 Clean 和 25 场景平均 F1 退步；条件式清零候选有 Accuracy 收益但 F1 退步；双分类头多种子不稳定。缺失中性加权在 valid/test 的重 Text 缺失均改善四指标，但独立 test 的 Clean 分类退步、25 场景均值分类收益很小且逐 seed 不同向。尚无候选同时稳健改善完整输入、各缺失类型和四指标。
2. **论文方法口径必须更新。**旧 25 场景数值是“后置遮挡协议”；本文 25 场景是“Text 词元先置 `[UNK]` 重编码协议”，两张表不可混成同一评估条件，更不能把旧小幅掉点用来声称真实缺失几乎无损。
3. **下一步模型假设应围绕单模型的 Clean/缺失兼顾。**目标应同时达到 Clean F1 不低于原基线、Text/三模态重缺失 F1 上升且 MAE 不升；仍用同 3 seed、同完整 25 场景、同 8 epoch 与同选模规则。条件式清零与双分类头已经在本轮检验过，不能再将它们原样当作待验证新点。不要继续凭单一场景调系数。若最终需要两模型选择，应先实测压缩后包体积，因为赛题总附件 ≤50 MB，且 Q3 还有模型权重。
4. **附件 3 的多段缺失依然是外推风险。**本次训练/网格注入的是单段，附件 3 实查平均约 3.17 个有效音频零段；后续可在确定真实缺失生成链后测试多段，但必须记录**实际**新增缺失率。附件 3 没有真值，不能拿其预测分布选模型。

### 冻结后的独立 test 评价约束（运行前写定）

反复使用 valid 做上述探索后，只选定**既有 P-RMF-E 三 seed**与**`gated_neutral_1.5` 三 seed**作一次附件 2 `test` 划分的独立检查；不根据 test 重训、调权重、重选 checkpoint 或更换候选。比较 Clean + 完整 25 个人工连续缺失场景，Text 缺失均按“先 `[UNK]` 后冻结 BERT”的同一协议生成。固定输出四指标、Text/三模态中段 50% 和 25 场景均值及逐 seed 差。`test` 只用于这一轮最终评估，不与无标签附件 3/4 混淆。**已按该冻结方案完成一次，结果见第 9 节；不得据此继续调整模型并仍将同一 test 称为独立验证。**

复现命令（从 E 题目录，使用既有本地 `.venv`，离线 BERT 缓存）：

```powershell
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py diagnose
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py train --seeds 1111 2222 3333 --zero-padding
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py compare --seeds 1111 2222 3333 --zero-padding --full-grid
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py train --seeds 1111 2222 3333 --pad-when-text-missing
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py compare --seeds 1111 2222 3333 --pad-when-text-missing --full-grid
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py train --seeds 1111 2222 3333 --pad-when-text-missing --split-text-head
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py compare --seeds 1111 2222 3333 --pad-when-text-missing --split-text-head --full-grid
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py train --seeds 1111 2222 3333 --pad-when-text-missing --neutral-weight-missing 1.5
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py compare --seeds 1111 2222 3333 --pad-when-text-missing --neutral-weight-missing 1.5 --full-grid
& 'EMOE_repro/.venv/Scripts/python.exe' EMOE_repro/q2_unk_reencode_experiment.py test-final
```

这些命令会向独立的 `results/unk_reencode/` 写入检查点/JSON。原双基线 `results/dual_baseline_v2/` 不受影响。`test-final` 记录的是**已经执行的冻结检验**，复现命令并不授权再用同一 test 调参。当前完成的是探索性增量实验，未来方案的调参与选择须继续限于 train/valid，并另行说明独立评价边界。
