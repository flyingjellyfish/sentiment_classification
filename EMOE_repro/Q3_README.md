# E 题问题 3：EMOE 预测解释与附件 4 推理

## 任务与数据核对

问题 3 要在保留情感极性和连续情感强度预测的基础上，说明三模态的作用、主要参考模态和局部证据，并将证据对应回原文本及音视频。验证仍以附件 2 的 `valid` 为准；附件 4 无真值，只能做全量推理与解释展示，不能报告其 Accuracy/F1/MAE/Pearson。

本次**冻结**已有问题 2 checkpoint `results/q2/refined/full.pt`，没有重新训练或引入规划文档中的新模型结构。统一使用 `aligned_50`：Text `50×768`、Audio `50×74`、Vision `50×35`，Router/mask 的内部顺序为 **Text, Vision, Audio**；表格和 CSV 显示顺序通常为 Text, Audio, Vision。附件 2 的 train/valid/test 为 3395/728/727 条；附件 4 是 20 个独立 pkl，均含 `id`、`raw_text`、`text`、`text_bert`、`audio`、`vision`。20 个 pkl 的 ID 均与同名 mp4 对应；逐文件路径、SHA256、有效长度、零值行、视频时长及帧数见 [`input_audit.json`](results/q3/input_audit.json)。附件 4 ID 与附件 2 ID 无相同字符串，但这不能单独证明视频内容绝无重复。

附件 4 的 pkl 使用 NumPy 2 内部对象，现有 `.venv` 的 NumPy 1 无法直接反序列化。`q3_prepare.py` 用本机捆绑的 NumPy 2 Python 检查字段、维度、ID 和 MP4 元数据，导出无需 pickle 的 [`attachment4_aligned50.npz`](results/q3/attachment4_aligned50.npz)，模型推理仍在原 `.venv` 运行。导出仅将 Audio/Vision 转为与附件 2 模型接口一致的 float32。20/20 条附件 4 文本的 `raw_text` 经同版 BERT WordPiece 编码后与给定 `text_bert` ID 完全一致；验证集为 728/728，因此文本位置可准确映射到原文字符区间。

原始特征中存在零值视觉行：附件 2 验证集有 15 条视觉内容区间全零；附件 4 的 **13 号视觉 50 行全零**。该样本不能从视觉特征定位关键帧，CSV 将视觉关键帧留空；即使 Router 给它非零权重，也不能据此断言视觉提供了有效证据。

## 解释算法

输入经 EMOE 的 Conv1D、单模态 Transformer、Router 和动态融合后，原模型直接给出三类概率与 `[-3,3]` 强度。Router 输出 `w_Text,w_Vision,w_Audio`，作为**模型融合分配权重**；`argmax(w)` 是 `dominant_router_modality`。该权重不等于可验证的因果贡献。

为得到时间级证据，先用 `text_bert` 的有效长度去掉 `[CLS]`、`[SEP]` 和 padding，并排除 Audio/Vision 内容区间中的全零行。对剩余各模态位置按最多 4 个连续对齐槽位分窗。每次将一个窗的 `observed_mask` 置假，模型以训练时的缺失 token 处理该窗，其他输入不变。记录：

\[
\Delta p_{m,W}=p_{\hat y}(x)-p_{\hat y}(x\setminus W),\qquad
\Delta r_{m,W}=\hat r(x)-\hat r(x\setminus W).
\]

每模态选得分 `max(0, Δp) + 0.1 |Δr|/3` 最高的窗；CSV 同时保存原始的有符号 `Δp` 与 `Δr`，因此可识别**反向证据**，而不把所有窗口都称作支持证据。文本窗用 BERT offset 精确截取 `raw_text`。音频、视觉窗的**对齐槽位**是可核对的原始索引。附件 4 没有词级时间戳或视觉帧到槽位的映射，秒数和帧号使用 `视频时长 × 槽位在有效内容中的比例` 粗略换算，列名均带 `approx_`。特别是 07、18 号达到 50 token 上限，可能发生截断，时间估计更不可靠。CSV 提供 mp4 原路径，供人工校对。任何近似秒数或帧号都不应写成精确标注。

另外，对每个模态单独遮挡所有可观测内容槽位，计算 `A_m = p_ŷ(x) - p_ŷ(x\m)`。`ablated_*_prob_drop` 保留有符号值。另把 `max(A_m,0)` 归一化形成 `perturbation_contribution_*` 供结果表展示；总正向下降不足 0.005 时标为 `dominant_perturbation_modality=None`。这是针对**预测类别概率**的模型干预敏感度，不是语义上的真实贡献，也不能代替 Router 权重。回归解释可读每个关键窗的 `*_regression_signed_change`。

## 验证结果（附件 2 valid，728 条）

| 条件 | Accuracy | Macro F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| 原始预测 | 0.6085 | 0.5827 | 0.6287 | 0.6088 |
| 三个模态各遮挡其选中关键窗 | 0.5632 | 0.5435 | 0.6455 | 0.5609 |
| 同模态、同长度随机窗（3 次，指标范围） | 0.5962–0.6058 | 0.5716–0.5809 | 0.6272–0.6332 | 0.6037–0.6125 |

对原预测类别的概率，关键窗合并遮挡后的平均下降为 **0.2277**；同模态、同长度随机窗为 **0.0127**。728 条配对差均值为 0.2150，样本 bootstrap 95% 区间 `[0.1994, 0.2318]`；92.86% 样本的关键窗影响大于随机窗。绝对强度变化均值为 0.2280，对照为 0.0739。这说明选中的窗口对当前模型输出有可重复的扰动敏感度；由于选窗本身基于单窗扰动，该验证不能解释为独立的语义真值验证。合并遮挡是额外检查窗口的联合作用。

验证集平均 Router 权重按 Text/Vision/Audio 为 `0.420/0.339/0.241`，单独整模态遮挡后的平均预测类别概率下降为 `0.0767/0.0370/0.0721`。Router 最大模态与遮挡影响最大模态仅在 **35.0%** 样本一致，说明两类数值不可混用，也不支持单凭 Router 宣称可靠解释。图见 [`validation_explanation_summary.png`](results/q3/validation_explanation_summary.png)，完整指标、置信区间、混淆矩阵与边界说明见 [`validation_explanation_report.json`](results/q3/validation_explanation_report.json)，728 条逐样本数据见 [`validation_explanations.csv`](results/q3/validation_explanations.csv)。

## 附件 4 输出

20 条均完成极性、强度、Router 权重和局部扰动计算：Positive 10、Neutral 4、Negative 6。这是**预测分布**，不是准确率。Router 主导模态为 Text 12、Vision 8；正向整模态遮挡影响主导模态为 Text 4、Vision 5、Audio 6、无明显正向影响 5。二者衡量对象不同。19 条有视觉关键槽位及近似帧，13 号视觉特征全零而留空。

- [`attachment4_summary.csv`](results/q3/attachment4_summary.csv)：20 条便于论文展示的主要字段，包括预测、Router 权重、扰动贡献度、文本摘录、音频近似时段、视觉近似关键帧及原视频路径。
- [`attachment4_predictions_explanations.csv`](results/q3/attachment4_predictions_explanations.csv)：完整逐样本输出，包括三类概率、各模态有效槽位数、关键窗索引、每窗概率/强度变化及整模态遮挡影响。

## 复现

在 E 题目录运行：

```powershell
& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' EMOE_repro/q3_prepare.py
$env:HF_HOME=(Resolve-Path 'EMOE_repro/hf_cache').Path
$env:HF_HUB_OFFLINE='1'
$env:MPLCONFIGDIR=(Resolve-Path 'EMOE_repro/results/q3').Path
& EMOE_repro/.venv/Scripts/python.exe EMOE_repro/q3_explain.py --sample-batch 16 --variant-batch 48
```

依赖现有 `results/q2/refined/full.pt` 和 `hf_cache` 中 BERT 的 `vocab.txt`。显卡为 RTX 5060 Ti 16 GB 时，上述小批量运行可行；本任务没有再次训练 EMOE。若论文需要**精确**音频起止秒或原视频关键帧，应补充词级强制对齐/原始特征提取时间戳，然后重新映射；现有附件仅支持上文明确标注的近似位置。
