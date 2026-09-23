# E 题问题 2：aligned EMOE 局部缺失实验

> **历史实验与接口勘误（2026-09-23）：**本文下方保留早期单种子 EMOE 的实际运行记录，已不作为 Q2 最终主干选择依据。正式 3 seed 双基线比较见 `P_RMF_E与EMOE_Q2双基线对比报告.md`，Q2 后续主干为 P-RMF-E。本文旧推理只按 Audio/Vision 零槽构造附件 3 缺失 mask，漏掉 Text 的 `[UNK]`（ID 100）；“附件 3 以音频/视觉缺失为主”的旧判断应撤销。附件 3 30 条中 27 条有效 Text 区有 `[UNK]`，大多与 A/V 同段；正式输出使用 `results/dual_baseline_v2/attachment3_unlabeled_predictions.csv`。旧版 CSV 只作历史调试记录，不用作三模态缺失效果证据。分类编码 0/1/2 分别为负/中/正，连续强度 0 为中性。

## 数据核验与输入

赛题要求用附件 2 `train` 学习、`valid` 选模，对附件 3 无标签样本预测三类极性和 `[-3,3]` 强度，并分析局部模态缺失类型、位置、时长的影响。本实验只用 **aligned_50**，不混用 unaligned；附件 2 的 `test` 不参与训练或选模。

| 数据 | 样本数 | 文本 | 音频 | 视觉 | 标签 |
|---|---:|---|---|---|---|
| 附件 2 train | 3395 | `text: 50×768`，`text_bert: 3×50` | `50×74` | `50×35` | 分类 0/1/2，回归 `[-3,3]` |
| 附件 2 valid | 728 | 同上 | 同上 | 同上 | 同上 |
| 附件 2 test | 727 | 同上 | 同上 | 同上 | 文件中有标签，但未使用 |
| 附件 3 aligned | 30 个独立 pkl，各 1 条 | **仅 `text_bert: 1×3×50`，无 `text`** | `1×50×74` | `1×50×35` | 无标签 |

分类标签与回归符号严格一致：负/零/正分别为 0/1/2。`text_bert[:,1,:]` 是 token 有效位，长度在附件 2 为 3–50，在附件 3 为 8–50。音频、视觉的第 0 位、SEP 位和尾部 padding 常为全零；不能将其视为局部缺失。以 token 有效区间内部的整帧全零作为可观测缺失：附件 2 音频内部零值段为 0，视觉在原数据中已有内部零值；附件 3 有 27/30 条在音频、视觉有效区间内出现全零位置，文本 token 的有效区间内没有零 ID。附件 3 中音频、视觉的平均内部零值比例（以有效内容长度为分母）约为 20.7% 和 21.1%。具体逐文件区间见 [`results/q2_data_audit.json`](results/q2_data_audit.json)。

附件 3 不提供 `text` 是直接复用原可行性脚本的障碍。用冻结的 `bert-base-uncased` 从 `text_bert` 生成表示后，抽取附件 2 训练/验证各 8 条比较：有效位置的平均余弦相似度分别约为 0.9999998/0.9999995，平均绝对差约 0.00033/0.00037。因此附件 2 用已给的 `text`，附件 3 用**相同 BERT**重建 `text`；两者均为 aligned_50 的 768 维 BERT 特征。BERT 只负责特征重建，不在附件 3 上拟合参数。

## 模型和训练

作者 EMOE 的三路 Conv1D、4 层 Transformer、样本级 Router、融合回归与单模态预测沿用；Router 宽度设为 256，使用附件 2 预计算 BERT 表示，不在线微调 BERT。问题 2 的新增部分：

实际张量路径为：Text `B×50×768` 经 `Conv1D(k=5)` 到 `B×46×256`；Audio `B×50×74` 经 `Conv1D(k=1)` 到 `B×50×256`；Vision `B×50×35` 经 `Conv1D(k=3)` 到 `B×48×256`。各路再过共享的 `1×1` 卷积和独立的 4 层、8 头 Transformer，取末时位得到三个 `B×256` 表示。Router 输入按原源码顺序将 Text/Vision/Audio 原始特征拼为 `B×50×877`、展平为 `B×43850`，经 `Linear(43850,256)→Linear(256,3)→softmax` 得到三权重；融合加权和为 `B×256`。Router 中增加 `3→3` 缺失比例偏置。融合头分别输出极性 `B×3` 与强度 `B×1`。

1. 模型融合表示 `c_proj: B×256` 接三分类线性头 `B×3`；原回归头输出 `B×1`，推理时裁剪到 `[-3,3]`。
2. `observed_mask: B×3×50` 按 **Text、Vision、Audio** 顺序，1 表示保留，0 表示缺失；padding 不标记为人工缺失。缺失处使用可训练的模态专属 token 替代原特征；Router 除读取替代后的三路特征外，再读取三个模态的缺失比例，产生额外权重偏置。该模块是本实验对 EMOE 的适配，**不是原论文结构**。
3. 训练时对 80% 样本在有效内容区间随机选一个模态或模态组合，遮掉一个连续片段，长度为有效内容的 15%–50%。组合含文本、音频、视觉、音频+视觉、文本+音频、文本+视觉；实际附件 3 以音频/视觉缺失为主。
4. 损失为 `CE_cls + MAE_reg + 0.2·L_uni + 0.02·L_router_entropy + 0.002·L_router_similarity + 0.02·L_distill`。单模态损失与 Router 目标按可观测比例加权。蒸馏方向遵循作者源码实现。系数为本实验选择，**不是 EMOE 原论文超参数**。

训练使用 Windows、RTX 5060 Ti 16GB、`torch 2.10.0+cu128`、`transformers 4.44.2`、AMP、batch 16、8 轮、种子 1111、AdamW 学习率 `1e-4`、权重衰减 `0.001`、梯度范数裁剪 `1.0`；文本 dropout `0.3`、注意力 dropout `0.4`、输出 dropout `0.5` 沿用 MOSEI 配置。数据一次加载、DataLoader worker=0。按完整验证集的 clean 与音频+视觉中段缺失 30% 两场景联合选择最佳轮。另将显式 mask 模型从“仅缺失增强”最佳权重出发，以 `5e-5` 学习率微调 5 轮；最终采用该补充实验最佳的第 3 轮。主实验峰值 CUDA reserved memory 约 754 MiB。

## 验证与消融

下表均为附件 2 **728 条 valid**，F1 为三类 macro F1。缺失 30% 指相对有效内容区间额外遮掉的连续长度；“clean”表示不再人工遮挡，原始零值仍按上文规则处理。

| 模型 | clean Acc | clean F1 | clean MAE | clean Pearson | 音频+视觉中段 30% Acc | F1 | MAE | Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 双任务 EMOE，无增强 | 0.6071 | 0.5779 | **0.5992** | **0.6291** | 0.5948 | 0.5678 | **0.6057** | **0.6263** |
| +连续缺失增强，无显式 mask | **0.6250** | **0.5950** | 0.6182 | 0.6116 | 0.6003 | 0.5723 | 0.6155 | 0.6143 |
| +缺失 token 与 Router 缺失偏置，随机初始化 | 0.6140 | 0.5748 | 0.6200 | 0.6162 | **0.6154** | 0.5811 | 0.6176 | 0.6160 |
| +从增强模型微调显式 mask，最终采用 | 0.6085 | 0.5827 | 0.6287 | 0.6088 | 0.6113 | **0.5881** | 0.6299 | 0.6115 |

缺失增强和显式 mask **没有在所有指标上超过无增强模型**；本轮主要确认局部缺失方案可运行。最终模型在音频+视觉缺失时宏 F1 比无增强组高约 0.0203，但 MAE 高约 0.0241。不要将这组验证结果写成全面优势。权重初始化、损失系数和缺失模式仍可进一步调整。

最终模型的缺失影响（Acc / F1 / MAE / Pearson）：

| 人工缺失 | Acc | F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| 无额外遮挡 | 0.6085 | 0.5827 | 0.6287 | 0.6088 |
| 文本 30%，开头 / 中间 / 末尾 | 0.6085 / 0.6071 / 0.6085 | 0.5798 / 0.5826 / 0.5821 | 0.6403 / 0.6290 / 0.6383 | 0.5962 / 0.6035 / 0.5820 |
| 音频 30%，中间 | 0.6140 | 0.5897 | 0.6309 | 0.6108 |
| 视觉 30%，中间 | 0.6126 | 0.5877 | 0.6274 | 0.6091 |
| 音频+视觉 30%，开头 / 中间 / 末尾 | 0.6085 / 0.6113 / 0.6085 | 0.5853 / 0.5881 / 0.5895 | 0.6282 / 0.6299 / 0.6242 | 0.6094 / 0.6115 / 0.6097 |
| 音频+视觉中间 10% / 50% | 0.6113 / 0.6140 | 0.5865 / 0.5935 | 0.6285 / 0.6313 | 0.6098 / 0.6105 |

数据提示文本依赖更强：文本末尾缺失时 Pearson 降到 0.5820；音频/视觉的不同位置和时长影响较小，未呈稳定单调趋势。小幅指标差异还需多随机种子验证，不能解释为因果规律。所有类型、位置、10%/30%/50% 组合及三组消融结果见 [`results/q2/report.json`](results/q2/report.json)，微调模型结果见 [`results/q2/refined/report.json`](results/q2/refined/report.json)。

## 附件 3 最终推理文件

- [`results/q2/final/attachment3_aligned_predictions.csv`](results/q2/final/attachment3_aligned_predictions.csv)：30 条完整预测，含极性类别、`[-3,3]` 强度、三类概率、三模态 Router 权重、有效长度和检测到的缺失位置。无标签，不能计算附件 3 的 Accuracy/F1/MAE/Pearson。
- [`results/q2/refined/full.pt`](results/q2/refined/full.pt)：本地 float32 完整 checkpoint，约 90 MB。
- [`results/q2/final/final_model_fp16.pt`](results/q2/final/final_model_fp16.pt)：45.1 MB 的 float16 权重副本；重新加载后 30 条极性全部一致，最大强度差 0.00049。运行时仍需冻结的 `bert-base-uncased` 权重来补齐附件 3 文本特征；该权重不在轻量 checkpoint 内。

最终 CSV 经独立加载 checkpoint 后重新推理，文件哈希一致。30 条中 Negative 5、Neutral 13、Positive 12；强度范围约 `[-1.306, 1.263]`。这是模型输出分布，不是附件 3 真值分布。

附件 3 的 [预测汇总图](results/q2/final/attachment3_prediction_overview.png) 展示三类数量、强度分布，以及有效区间音频/视觉零值比例与预测强度、Router 文本权重的关系；[汇总 JSON](results/q2/final/attachment3_prediction_summary.json) 给出精确统计。30 条的平均 Router 权重按 Text/Vision/Audio 为 `0.321/0.372/0.307`，平均内部音频/视觉全零率约 `20.9%`。这些权重是模型输出，不能当作因果贡献。

## 验证集可视化与错误归因

[六联分析图](results/q2/final/q2_validation_analysis.png) 和 [逐样本验证预测 CSV](results/q2/final/validation_predictions.csv) 来自最终 checkpoint 的独立复核，clean 和音频+视觉中段 30% 缺失的 Accuracy/F1/MAE/Pearson 与上表、训练报告逐项一致。[错误分析 JSON](results/q2/final/validation_error_analysis.json) 包含混淆矩阵、分类别指标、标签强度/长度/原始视觉零值分组、最差样本 ID。

可核验的主要现象：

- clean 混淆矩阵（行是真值 Negative/Neutral/Positive，列是预测）为 `[[119,39,48],[31,82,71],[39,57,242]]`。Neutral 的召回仅 `0.446`，184 条中 71 条被判为 Positive；Positive 召回 `0.716`。分类错误共 285/728 条。
- 极性较弱（非零且 `|y|<0.5`）的 147 条分类准确率 `0.497`；极性较强（`|y|≥1.5`）的 131 条准确率 `0.794`。弱极性更容易混淆，但强极性的强度回归 MAE 高达 `1.126`，高于弱极性的 `0.471`。回归散点图中拟合关系约 `ŷ=0.451y+0.193`，提示输出强度向中间收缩。
- 有效内容长度最低四分位（192 条）MAE `0.512`，最高四分位（181 条）MAE `0.755`。这是分组相关性；未控制标签强度等因素，不能直接归因为“长句导致错误”。
- 对同 728 条样本加入音频+视觉中段 30% 遮挡后，有 49 条的分类结果改变；整体宏 F1 从 `0.5827` 到 `0.5881`，MAE 从 `0.6287` 到 `0.6299`。变化很小，不支持“缺失越多必然越差”的单调结论。
- 最大的回归错误出现在 `f_ZJ7L14oYQ$_$21`：真值 `+2.0`，预测 `-1.987`；其英文片段同时出现正向比较与负向词语。这个例子提示语义转折/比较可能是难点，但不能仅凭单例断言错误原因。

这些属于**错误关联与候选机制**，不是经过干预证明的因果归因。数据只有一个固定种子与单次训练，类别差异、缺失率曲线的小幅变化都需重复实验确认。

## 重现命令

在 E 题目录的 PowerShell 中，先确保 `.venv` 安装 `transformers==4.44.2`，并在 `EMOE_repro/hf_cache` 缓存 `bert-base-uncased`。随后：

```powershell
$env:HF_HOME=(Join-Path (Resolve-Path 'EMOE_repro').Path 'hf_cache')
$env:HF_HUB_OFFLINE='1'
.\EMOE_repro\.venv\Scripts\python.exe .\EMOE_repro\audit_q2_data.py
.\EMOE_repro\.venv\Scripts\python.exe .\EMOE_repro\q2_aligned.py --epochs 8 --batch-size 16
.\EMOE_repro\.venv\Scripts\python.exe .\EMOE_repro\q2_refine.py --epochs 5 --batch-size 16 --lr 0.00005
.\EMOE_repro\.venv\Scripts\python.exe .\EMOE_repro\q2_infer.py --checkpoint .\EMOE_repro\results\q2\refined\full.pt
.\EMOE_repro\.venv\Scripts\python.exe .\EMOE_repro\q2_validation_analysis.py
```

本实现是问题 2 的可运行版本，仍有两个边界：Transformer 内部没有逐位置 attention padding mask，缺失信息通过输入 token 与 Router 传递；训练主要模拟单段缺失，附件 3 可含多个不连续零值段。提交前应将预训练 BERT 的获取方式和当前模型参数大小写入环境说明。
