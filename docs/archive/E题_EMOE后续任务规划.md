# E 题 Q2/Q3 后续任务规划（P-RMF-E 主干修订版）

## 2026-09-23 主干裁决与执行路线

本节是**当前有效计划**。下方原 EMOE 方案保留为决策过程和文献思路，不再按其“EMOE 为 Q2 主干”“只借 P-RMF 的可靠性 Router”的旧顺序执行。正式双基线证据见 [`EMOE_repro/P_RMF_E与EMOE_Q2双基线对比报告.md`](EMOE_repro/P_RMF_E与EMOE_Q2双基线对比报告.md)：统一附件 2 aligned_50、连续缺失协议和 3 个种子后，P-RMF-E 的 Clean Acc/F1/MAE/Pearson 为 `.6392/.6069/.6025/.6320`，EMOE 为 `.6140/.5818/.6259/.6119`；25 个缺失场景的平均 F1 为 `.6033` 对 `.5814`。因此 **Q2 以 P-RMF-E 为创新主干、EMOE 为固定对照**。这只是当前一个划分、3 seed、8 epoch 的工程裁决，不是无条件证明。Q3 现有 EMOE Router 解释成果继续保留；P-RMF-E 的权重只能先叫内部 proxy 权重，不能直接叫真实贡献。

### 数据与评估边界已经修正

1. 附件 2 的 `classification_labels` 是 `0=负、1=中、2=正`；`regression_labels=0` 才对应中性，逐条满足 `classification_labels=sign(regression_labels)+1`。附件 3/4 无标签，不作伪标签或性能评估，不引入 MOSI/IEMOCAP 等外部情感数据。
2. 附件 3 有 `text_bert` 而没有 `text`，须用同一冻结 BERT 重建 50×768 特征。30 条中 27 条有效 Text 区有 `[UNK]`，且大多与 A/V 全零槽同段。旧 `q2_aligned.py::infer_special` 漏标 Text 缺失，旧附件 3 CSV 只作历史输出；正式双基线 `q2_dual_attachment3.py` 用三路现有缺失 mask 且不再二次遮挡。
3. 所有创新实验沿用 `results/dual_baseline_v2/` 的 train/valid、种子 `1111/2222/3333`、每样本每 epoch 同一连续段增强、26 场景评估、Clean+A/V30 复合选模。先单 seed 筛掉明显无效设计，入选方案再做 3 seed 并与**同 seed P-RMF-E**成对比较。报告 Accuracy、Macro-F1、MAE、Pearson、Neutral 召回、`|y|≥1.5` 的 MAE、实际新增缺失率和训练开销；不能仅挑单项最佳。

### 修订后的分阶段工作

| 阶段 | 模块与单因素实验 | 验收及回退 |
|---|---|---|
| **0. 固定锚点** | 冻结 P-RMF-E/EMOE 双基线 checkpoint 与同一 mask 生成器；重新审计旧 Q2 附件 3 Text mask 的影响。 | 任何新结果均与 `dual_baseline_v2` 同种子相比，不与此前单 seed refined 模型混用。 |
| **1. 最小改进：局部缺失显式信号** | P-RMF-E 原本在缺失槽直接零填充。为 Text/Vision/Audio 各加一个可学习的缺失位置 embedding，注入 token 投影后；其余架构和任务损失不变。做 `baseline vs embedding` 消融。 | Clean 不明显退化，缺失网格 F1/MAE 同向改善才保留；否则回退原零填充。该 embedding 是 E 题适配创新，不是 P-RMF 原方法。 |
| **2. 核心改进：缺失感知 proxy 权重** | 原 P-RMF 已由 VAE 方差产生不确定度权重。在此基础上，仅加有效内容覆盖率的轻量校正，并与阶段 1 分开测试 `coverage only / embedding+coverage`。先不叠新 VAE、复杂生成网络。 | 对比 Text、A/V、三模态同段 10/30/50% 的绝对与相对降幅，检查权重是否塌缩。若单因素不稳或强度 MAE 受损，关闭校正。 |
| **3. 可选完整→缺失一致性** | 仅在前两阶段有稳定信号后，给同一 train 样本的完整与局部缺失预测加轻量双头一致性；完整分支停梯度，主任务 CE+L1 始终保留。与 P-RMF 原有 VAE/重建辅助项单独消融，避免误写成原论文蒸馏。 | 重点查长 Text 缺失、三模态同段和极端强度；若错误复制、Neutral 召回或强度幅度变差，退回阶段 1/2 最佳 checkpoint。CMAD 全量结构仍不进入首轮。 |
| **4. Q3 解释衔接** | 分别输出 EMOE Router 和 P-RMF-E proxy 权重；两者都用独立扰动检查模态/时间证据。原视频仅准确声明 50 槽索引，秒数/帧号需人工或时间戳验证。 | 解释忠实性与人类语义可读性分别报告；不能把 proxy/Router 权重写成因果贡献百分比。 |

**预先规定的失败回退：**如果阶段 1、2、3 没有跨 3 seed 的综合增益，Q2 交付仍采用已完成的 P-RMF-E baseline，EMOE 保持对照。附件 3 的 30 条无标签预测可作接口/稳定性核查，不参与创新筛选。旧 EMOE 上的 padding、覆盖率、蒸馏、Router 宽度试验见 [`EMOE 创新阶段记录`](EMOE创新实验阶段记录_转P-RMF双基线前.md)，其负面/不稳定结果是设计警示，不能当成 P-RMF-E 上已验证的消融。

---

## 原 EMOE 主干计划（历史记录，已由上方路线取代）

本文是**后续开发计划**，不是已完成实验的结果。以当前双任务 EMOE 为主体，优先解决问题 2 的局部连续时间段缺失，再为问题 3 建立能回看原视频的证据定位。文中分别标注“原论文方法”“已有实现”和“拟议适配”，避免把新设计写成论文原方法。

## 1. 决策摘要

1. **先修数据接口和评价协议，再加模型结构。** 已有模型可在本机 RTX 5060 Ti 16GB 上训练，但一轮实验中缺失感知模型只提高了部分缺失场景的宏 F1，回归 MAE 反而变差；单一种子不足以宣称鲁棒性提升。
2. **最值得先借的是 P-RMF 的“估计模态可靠性再融合”，不是整套代理模态与 VAE。** 先在 EMOE 单模态分支增加低成本的可校准不确定度，再给现有 Router 一个可靠性修正项。P-RMF 原文采用随机的模态内部缺失，E 题要求局部连续缺失，训练缺失生成器必须另行适配。[P-RMF 原文](https://aclanthology.org/2025.acl-long.1075/)
3. **CMAD 的完整教师→缺失学生蒸馏适合第二个核心实验。** 先只蒸馏双任务输出与融合向量；其批内相关矩阵对齐和按模态组合重加权留作消融或可选增强。[CMAD 原文](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html)
4. **问题 3 从无训练的局部扰动证据开始。** Router 权重可报告为“模型内部融合权重”，时间证据要由原始 50 位输入上的窗口删除实验验证。注意力或 Router 权重本身不能等同于真实致因。[Attention is not Explanation](https://aclanthology.org/N19-1357/)

## 2. 先核对的数据接口与边界

2026-09-23 重新运行本地 `EMOE_repro/audit_q2_data.py` 并直接抽查 pkl。以下是**文件实况**，不是假设：

| 数据 | 结构 | 模型可用输入 | 需要处理的差异 |
|---|---|---|---|
| 附件 2 `aligned_50.pkl` | `train=3395`、`valid=728`、`test=727`；每个 split 有 `text(N,50,768)`、`text_bert(N,3,50)`、`audio(N,50,74)`、`vision(N,50,35)`、分类/回归标签及 id | 当前 EMOE 直接吃 `text/audio/vision`；`text_bert` 第 2 行给有效 token 长度 | 只在 train 学习、valid 选模；文件中的 test 不参与参数、阈值、超参数选择 |
| 附件 3 aligned | 30 个单独 pkl，每个为 `{'test': {...}}`，仅有 `text_bert(1,3,50)`、`audio(1,50,74)`、`vision(1,50,35)`，无标签、无预计算 `text` | 音频、视觉尺寸与附件 2 一致；文本需转换 | 已验证冻结 `bert-base-uncased(text_bert)` 能近似精确重建附件 2 的 `text`：抽样有效位余弦相似度约 0.9999995 以上。继续使用同一 BERT 和 aligned 版本 |
| 附件 4 aligned，供问题 3 | 20 个 pkl，各有 `raw_text/id/text(50,768)/text_bert(3,50)/audio(50,74)/vision(50,35)`，并有对应视频 | 三模态数值接口可用 | 当前 `.venv` 的 NumPy 1.26 读取其中样本会报 `numpy._core.numeric`；已用本机 NumPy 2.3.5 运行时读通 20 个。后续可一次性做**无损格式转换**，记录源文件哈希和转换日志；不要为了它直接升级训练环境的 NumPy |

`text_bert` 的有效长度：附件 2 为 3–50，附件 3 为 8–50。附件 2 的音频在有效区间内部没有整帧零值段，视觉在 train/valid 分别已有 210/68 个内部全零段；附件 3 有 27/30 条音频与视觉存在内部全零位置。**全零视觉帧不天然等于人为缺失真值。** 新接口至少区分：`P` 原本有效位置、`M` 训练中明确注入的缺失、`Z` 原文件观测到的全零。附件 3 无缺失标注，只能从 `P` 和 `Z` 推测疑似缺失，不能把推测当真值做监督。

模型代码目前使用 `(Text, Vision, Audio)` 即 **LVA** 顺序的 `observed_mask: B×3×50`、Router 权重和单模态头；文稿常写 `(Text, Audio, Vision)`。所有新增模块统一内部顺序 LVA，输出文件明确字段名并增加顺序断言。音频/视觉的零值填充与文本 BERT padding 均不能直接当作情感信息。问题 2、3 不使用题外有标签情感数据；公开预训练 BERT仅用于题目允许的文本特征重建，不在附件 3、4 上拟合。

**数据使用边界：**附件 3/4 可以检查字段、尺寸、可否解码，并作最终推理；不得利用无标签专项集的预测分布或缺失模式频率来选损失系数、缺失率、模型或阈值。模拟缺失分布和全部超参数只由赛题定义及附件 2 train/valid 确定。

## 3. 当前基线：保留什么，先核验什么

EMOE 原论文提出样本级 Mixture of Modality Experts、Router 动态融合、单模态预测能力及 Unimodal Distillation；**原论文不是局部连续缺失模型，也不产生时间级证据**。[EMOE 原文](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html)

本地已实现：附件 2 对齐三模态特征、三路卷积和 4 层 Transformer、256 宽 Router、双任务头、一个连续片段的缺失增强、缺失 token、缺失比例 Router 偏置、完整 valid 评价和附件 3 全量推理。主要文件是 `EMOE_source/trains/singleTask/model/emoe.py`、`router.py`、`EMOE_repro/q2_aligned.py`、`q2_refine.py`、`q2_validation_analysis.py`。主实验的 clean 基线验证为 Acc 0.6071 / 宏 F1 0.5779 / MAE 0.5992 / Pearson 0.6291；已选的显式 mask 微调模型是 0.6085 / 0.5827 / 0.6287 / 0.6088。后者在音频+视觉中段 30% 缺失时宏 F1 为 0.5881，高于无增强的 0.5678，但 MAE 0.6299，高于无增强的 0.6057。完整记录见 [`旧 Q2 说明`](Q2_README.md)。

近期需要先核验的两个实现细节：

- 当前 `base_mask` 将有效区间中的视觉全零位置标为缺失；附件 2 本身存在此类位置，因此应拆出“确定的训练注入缺失”与“原文件零值”两条信号，并单独消融“视觉零值是否标记”。
- 作者实现经 Text/Audio/Vision 宽 5/1/3 的卷积后分别变成 46/50/48 位，再取编码器最后一位。短样本的末位可能是 padding 对应位置；在不动主基线的前提下，单独比较“原最后位”与“按有效位置聚合”。若需改，卷积后有效掩码和时间位置都要同步映射。问题 3 的证据搜索宜在**卷积前统一 50 位输入**进行，避免三路卷积后的时间坐标错位。

## 4. 论文逐篇判断

“收益”是面向本题的**预期**，不是论文已在本题证实的分数。复杂度按在现有 EMOE 上实现与验证的工作量估计。

| 论文 | 它解决的问题及原文核心模块 | 在 EMOE 上能借什么 | 复杂度 / 预期收益 / 决策 |
|---|---|---|---|
| [EMOE，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html) | 模态重要性随样本变化与单模态能力丢失；模态专家、Router、单模态蒸馏 | **主体不替换**；保留三路专家、融合和单模态分支，在 Router 前加入明确缺失信息 | 已实现；作为所有阶段的比较锚点。论文权重只能先称模型融合权重 |
| [P-RMF，ACL 2025](https://aclanthology.org/2025.acl-long.1075/) | 不完整数据及噪声；每模态 VAE 的高斯均值/方差、由不确定度加权的 proxy modality、多层跨模态动态注入、特征重建；实验含模态内部**随机**缺失率 0–0.9 与整个模态缺失 | 优先借“可靠性修正 Router”，但先以低成本的单模态预测尺度/误差校准估计可靠度；只有其有效再试高斯潜变量，不直接搬入 proxy + 多层注入 | 简化版低到中；有望改善缺失时的权重分配，但必须同时看回归 MAE。**优先采用思想，不完整复现结构** |
| [CMAD，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html) | 完整教师到缺失学生的表示偏移和训练不稳定；CAFD 包含特征对齐与批内相关结构，MAR 按缺失模态组合难度重加权 | 冻结当前 clean EMOE 当教师，缺失增强的 EMOE 当学生；先蒸馏融合向量、分类分布和强度。批内相关矩阵后加 | 简化版中；可能兼顾 clean 性能与缺失鲁棒性。**推荐核心阶段**；原文主要是整个模态缺失，局部 span 是我们的适配 |
| [MPLMM，ACL 2024](https://aclanthology.org/2024.acl-long.94/) | 多种缺失模态组合；生成 prompt、missing-signal prompt、missing-type prompt，预训练骨干 + 缺失特征生成 | 当前缺失 token 已部分实现 missing-signal；可进一步给局部位置和缺失模态组合一个轻量 type embedding。不要把全零位置当普通输入；暂不复制生成 prompt 与完整 MulT | 轻量信号低；有望减少零值歧义。**最小改进可用**；原文跨数据集预训练设置不符合本题数据限制，不能照搬 |
| [FUSE-Net，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Factorize_Reconstruct_Enhance_A_Unified_Framework_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf) | 模态噪声与冗余；共享/特有/噪声子空间、对比与信息约束、变分重建、多因子动态融合 | 若视觉全零或噪声明显损害预测，可在单模态 256 维表示上试一个小型“信号/噪声”门控；不能直接把完整分解网络塞进 EMOE | 高；当前样本仅 3395 条，过拟合与论文归因复杂。**可选高级增强，首轮不采用** |
| [CmIR，ACL 2026](https://aclanthology.org/2026.acl-long.2119/) | 噪声、OOD 下的虚假相关；因果不变/环境特有表示分解，虚拟噪声环境，不变性、互信息与重建约束 | 可借“不同扰动下输出保持稳定”做轻量一致性损失；只对信息仍足够的局部缺失施加，避免要求严重缺失与完整预测完全一样 | 轻量借鉴中，完整框架高。**只考虑一致性**。原文主要是高斯噪声和分布变化，不是“完整→局部缺失”蒸馏；该具体配对是我们的设计 |

问题 3 的方法学参考：[Jain 与 Wallace](https://aclanthology.org/N19-1357/)提醒不能只凭注意力权重宣称解释；[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html)可作为时间候选窗口的低成本梯度筛选，但最终证据仍用窗口扰动与原视频回看核验。前者是诊断警示，后者是候选归因工具；两者都未替 E 题给出逐词时间戳。

### 4.1 值得做的轻量组合

**最小版本：**`有效位 P + 人工缺失 M + 原文件零值 Z → 显式缺失 token/type → 原 EMOE Router`。一开始不要新增恢复网络。训练在附件 2 `train` 上随机抽**连续窗口**；每条可有一个或少数多个窗口，不能把独立随机点缺失当 E 题主要设置。起点 10%/30%/50% 与头/中/尾由 train/valid 实验确定，不使用附件 3 的零值统计来调参。

**可靠性 Router：**保留作者 Router logits `r_m`，只加一个可撤销的修正：`w_m = softmax(r_m - β s_m + γ log(c_m+ε))`。其中 `c_m` 是有效区间内已知观测比例，`s_m` 是从该模态 `z_m` 的单模态强度预测分支学出的 log-scale（例如 Laplace NLL：`exp(-s_m)|ŷ_m-y|+s_m`），并与训练/验证误差做校准。`s_m` 是**我们为本题设计的预测不确定度近似**，不等于 P-RMF 原文 VAE 方差；若校准失败，退回仅使用 `c_m`，不要给噪声指标冠上“不确定度”名称。每步只改一个项：先覆盖率，再预测尺度，最后合并。

**完整→缺失蒸馏：**冻结由附件 2 train 训练的 clean EMOE 教师；同一 train 样本构造局部缺失给学生。建议先试：`L_KD = a·KL(p_T^cls || p_S^cls) + b·SmoothL1(r_T,r_S) + c·(1-cos(z_T,z_S))`，教师输出停梯度，分类 KL 加温度。主任务 CE/MAE 始终保留；严重缺失或教师在该 train 样本上明显错误时降低蒸馏权重，防止强迫学生复制错误或无法从剩余信息推断的内容。仅在简单蒸馏有收益时，才试 CMAD 的批内 `B×B` 相关结构对齐及模态组合难度权重。

## 5. 问题 3：模态级与时间级解释

先使用问题 2 选定模型，**不另训练一套解释模型**：

1. **模态级：**输出每条样本的 EMOE `w_text/w_audio/w_vision`，同时计算遮蔽该模态若干有效窗口后的分类原类别 logit/probability 下降和回归强度变化。权重与扰动敏感度并列报告，不一致时明确展示，而非只取较好看的一个。
2. **时间级：**在统一的原始 50 个 aligned 位置滑动短窗口（如 1、3、5 位，仅在附件 2 valid 上选窗长），排除 padding 和原本缺失位置。遮掉 `m,[t:t+k)` 后重新推理；分类正证据可用原预测类概率或 logit margin 的下降量，负值代表反向证据；回归报告强度变化的**方向与幅度**。每模态返回 Top-K 不重叠窗口。梯度/IG 只作候选加速，最终排序以同一缺失算子下的实际预测变化核验。
3. **可信度验证：**在 valid 上比较 Top-K 窗口与随机同长度窗口的 deletion 效果，做逐步删除曲线、重复种子稳定性，以及模态权重与遮挡敏感度的排序一致性。这里评价的是模型解释的忠实性；附件 4 无人工证据标注，不能报告“证据定位准确率”。缺失片段不可被重新选成关键证据。
4. **回原视频：**附件 4 的 `text`、`raw_text` 和 20 段视频可用于展示，但 pkl 的 50 位不带可靠的逐词秒数。先建立 BERT subword→原文词片段→原视频时段的映射，对至少若干样本人工复核；语音与视觉片段用对应时间戳/视频帧展示。映射未经核验时只报告“对齐位置索引及近似区间”，不能把 `t/50 × 总时长` 写成精确起止秒数。
5. **最终产物：**附件 4 全量预测与解释 CSV/JSON；每条含双任务结果、Router 权重、Text 片段、Audio 时间段、Vision 帧区间、扰动得分和定位精度标记；另给典型正确、错误、模态冲突与局部缺失样本的可回看图卡。

## 6. 分阶段开发与停损条件

每个阶段先固定 `train/valid` 划分、随机种子、评价网格、checkpoint 选择规则，随后只新增一种机制。分类报告三类 Accuracy、macro F1、分类别召回及混淆矩阵；回归报告 MAE、Pearson、强度分层 MAE。缺失场景按模态/组合 × 开头/中间/末尾 × 10%/30%/50% 展示，并给整个网格均值、最差场景及 clean→缺失的配对变化。至少对入选改进重复 3 个训练种子，在同一批 valid 样本上做配对区间估计；附件 3、4只做最终无标签推理。

| 阶段 | 要改的模块与交付 | 训练目标与消融 | 指标、通过条件、失败回退 |
|---|---|---|---|
| **Baseline：固化现状** | 固定 `q2_aligned.py` / `q2_validation_analysis.py` 的入口、配置、结果；新增只读的接口检查与缺失协议记录；核对附件 4 NumPy 2 转换路径 | 不新增损失；复现无增强、仅增强、显式 mask 三组。单独查原零视觉帧标记和末位池化问题 | 建立 clean/缺失全网格与分层误差；若复算不一致先修评估，不开始新结构。以无增强模型和当前最佳 mask 模型**共同**作锚点 |
| **最小改进：局部缺失语义** | 从 `q2_aligned.py` 抽出统一 `missing_protocol.py`：`P/M/Z`、单段/多段连续窗口、LVA 顺序断言；在 `emoe.py/router.py` 只保留轻量 mask token/type 与覆盖率项；新建 `q3_explain.py` 的无训练窗口遮蔽版 | 原双任务+EMOE 损失；消融 0 值直输、缺失 token、type、覆盖率；单段 vs 少数多段；原最后位 vs 有效聚合单独实验 | 重点看 clean 性能是否保持、缺失网格 F1/MAE 是否同时更稳；解释 Top-K 删除效果需超过随机窗口。若无收益，保留数据协议，回退到原 Router/原聚合 |
| **核心改进 A：可靠性 Router** | `router.py` 增加单模态 log-scale 与小标量门控；`loss_fn` 加观测位加权的 NLL/校准损失；`q2_validation_analysis.py` 加“预计不确定度 vs 实际单模态误差”曲线 | 先仅覆盖率、再仅 log-scale、最后两者合并；固定其他网络。监测 Router 权重塌缩和 Text 独占 | 若可靠性与绝对误差无正相关、clean MAE 变差或多种子鲁棒指标不稳，撤去 log-scale，仅留覆盖率/原 Router。不得把未经校准的门控称为置信度 |
| **核心改进 B：完整教师→缺失学生** | `q2_refine.py` 增加冻结教师、双输入成对前向；`loss_fn` 加双头与融合向量蒸馏；先用已有 clean checkpoint | 消融任务损失、分类 KL、回归一致性、融合特征对齐，以及教师可靠性门控；可再试一个简单 CmIR 式同样本扰动一致性，与教师蒸馏分开 | 必须与核心 A 分别比较后才合并；若教师错误传递、强度被进一步压缩或收益不稳定，关闭相应蒸馏项，保留最佳前阶段 checkpoint |
| **可选增强：仅按失败类型触发** | 若缺失后大量失真：CMAD `B×B` 相关对齐；若高缺失率确需补特征：极小 masked feature recovery；若视觉噪声主导：FUSE 风格信号/噪声门控；IG 加速 Q3 候选 | 每项只在清楚的失败切片上试；单项消融后才考虑组合，不复刻 P-RMF 全 VAE+proxy+多层注入或完整 CmIR 因果框架 | 参数量、训练时间、内存与论文可解释性成本必须与 valid 收益相称。无配对收益或 clean 明显受损即回退；问题 3 的视频时间映射质量不能靠更复杂模型替代 |

建议实际执行顺序：**接口/评价核验 → 掩码语义与缺失生成器 → 问题 3 无训练解释原型 → 可靠性 Router → 教师蒸馏 → 只有证据支持才试高级增强**。这样问题 3 的基础交付不依赖问题 2 全部研究完成。

## 7. 写作时必须说明的界限

- EMOE、P-RMF、CMAD、MPLMM、FUSE-Net、CmIR 的结构与本计划的**借鉴片段**要分别署名；“局部连续 span 的可靠性 Router”“本题双任务蒸馏”“窗口级证据”均为本方案适配，不能写成这些论文的原实验结果。
- P-RMF 的随机模态内部缺失与 E 题的连续段缺失不同；CMAD/MPLMM 主要按整个模态组合缺失；CmIR 主要是噪声与 OOD。它们为设计提供动机，是否适用于本题只能由附件 2 valid 上的专门实验判断。
- 无标签附件 3/4 的预测分布不能作为模型性能；不能据此选模、调参或编造人工证据真值。Router 权重、梯度和扰动得分均是模型行为证据，不证明视频中情绪的客观因果来源。
