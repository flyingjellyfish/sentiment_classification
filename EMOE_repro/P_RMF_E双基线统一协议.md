# E 题问题 2：EMOE 与 P-RMF-E 双基线统一实验协议

本协议在比较前固定。实验入口为 `q2_dual_baseline.py`，P-RMF-E 模型适配见 `prmf_e.py`；原仓库文件保持独立，固定修订号记在 `P_RMF_upstream/UPSTREAM_REVISION.txt`。此前 EMOE 优化实验见 [`EMOE 创新阶段记录`](../docs/archive/EMOE创新实验阶段记录_转P-RMF双基线前.md)，不参与这次选主干。

## 数据与可用信息

- 两模型只读取附件 2 `aligned_50.pkl` 的 train=3395、valid=728。Text `B×50×768`，Audio `B×50×74`，Vision `B×50×35`，类别 0/1/2 与强度 `[-3,3]`；不重新运行 BERT，不接触 test 的标签或附件 3/4 来选模。
- 三模态共享同一预处理：Text 原 float32；Audio/Vision 的 NaN/Inf 归零并转 float32。有效内容槽由 `text_bert` attention mask 长度确定；原 Audio/Vision 全零内容槽记为 source zero。内部 mask 顺序固定为 Text/Vision/Audio。
- 每个 epoch、每个样本的训练 mask 在模型训练前按 `seed + 1000003×epoch` 确定，和模型权重初始化、batch 内顺序无关。80% 样本尝试注入一个**连续**窗口，长度占有效内容的 15%–50%；随机选 Text、Audio、Vision、两模态或**三模态同段**共 7 种组合。已为零的原特征不算新增删除。两模型用相同样本排列与 mask；P-RMF-E 对缺失位填零，EMOE 用已有缺失 token，属于模型自身机制差异。
- 验证集固定同一组 mask：无人工缺失（Clean）、Text/Audio/Vision/Audio+Vision/三模态同段共 5 类 × 头/中/尾 30%，以及各类 × 中段 10/30/50%，去重后共 26 条场景。记录实际新增遮挡率。附件 3 直接传入其已有缺失特征，**不再注入随机 mask**。只使用附件 3 的**无标签缺失格式**核对接口，不用其预测分布选模。

## 模型边界

- **EMOE**：采用原三路编码器、Router、动态融合、单模态输出及现有双任务头；本次 `control` 关闭此前新增的 padding 池化和覆盖率修正。完整模型从头初始化。
- **P-RMF-E**：保留仓库的三路投影+Transformer token 编码器、三个 VAE、代理模态不确定度加权、梯度反转、四次共享权重的跨模态注入、重建器和原回归 MLP。输入改为附件 2 已提供的 Text 连续特征及三模态各 50 步；增加线性三分类头。原代码中重建器未列入 optimizer，本适配将其纳入，使论文描述的重建目标能训练。该修正需在报告中明确，不能称为原仓库逐行无改动复现。
- 两模型共享任务目标 `CE(三类)+L1(强度)`；各自保留其原有辅助目标。EMOE 为单模态预测、平衡和蒸馏项；P-RMF-E 为 `0.1×重建 MSE+0.5×VAE/KL`。这比较的是**对 E 题适配后的模型家族**，不等同于原论文公开表格复现。

## 训练、选模与裁决

- 种子 1111、2222、3333；每模型每种子 8 个 epoch，batch 16，AdamW `lr=1e-4`、`weight_decay=1e-4`；混合精度、梯度裁剪 1.0，硬件为 RTX 5060 Ti 16GB。
- 每 epoch 只在 valid 的 Clean 与 Audio+Vision 中段 30% 上计算四指标；单一 checkpoint 以 `clean_F1 + AV30_F1 + 0.25×(clean_Pearson+AV30_Pearson) − 0.25×(clean_MAE+AV30_MAE)` 最大选定。**不**按 Accuracy 单独选 epoch，也不使用原仓库的 test 选模或每指标不同 checkpoint。
- 最佳 checkpoint 再跑全部 26 场景，报告 Accuracy、macro-F1、MAE、Pearson，性能相对 Clean 的下降、缺失类型/位置/长度、实际遮挡率、三种子的均值和离散度；另记总参数、峰值显存、训练时间、728 条验证推理吞吐。
- 最终主干同时考虑 Clean 与缺失网格、回归 MAE/Pearson、种子稳定性、参数/显存/时间和后续问题 3 的解释接口。差距小则偏向实现和输出解释较简单、稳定者。不把无标签附件 3 的预测类别分布作为性能证据。

协议修订说明：首轮 6 组合预实验已保存在 `results/dual_baseline/`。随后核查附件 3 发现 27/30 条出现 Text `[UNK]` 内容槽，且 29/30 条的 Text/Audio/Vision 缺失位置一致（另 1 条有额外视觉零值），因此正式协议加入三模态同段组合与验证场景，写入 `results/dual_baseline_v2/`，不混用首轮结果。

进一步核验：附件 2 的 train/valid/test 共 4850 条在有效 Text 内容槽中均无 BERT `[UNK]`（ID 100）；附件 3 的 27 条非空 `[UNK]` 与 A/V 零值高度同步。因此“附件 3 文本完全未遮挡”的说法与文件证据不符。附件 3 也确实没有 `text` 字段，只能从其 `text_bert` 重建连续特征。审计数据见 `results/dual_baseline_v2/attachment2_unk_baseline.json` 和 `attachment3_structure.json`。**标签编码须区分**：连续 `regression_labels=0` 是中性；预编码 `classification_labels` 是 0=Negative、1=Neutral、2=Positive，逐条满足 `classification_labels=sign(regression_labels)+1`。绝不将中性并入正类。附件 3/4 无标签；不作伪标签训练。
