# E 题问题 3：冻结 P-RMF-E 的预测与解释

> 实测日期：2026-09-24。此版本解释 **Q2 原 P-RMF-E 三种子集成** 的输出，不解释 EMOE，也不是后来余弦学习率单模型或 Text 专项分支的输出。原 EMOE 问题 3 结果保留在 [Q3_README.md](Q3_README.md)。

## 1. 任务、输入与预测器

题目问题 3 面向三模态信息完整的场景，要求情感极性、连续情感强度、主要参考模态、模态作用程度、关键文本/语音/视觉证据；附件 4 是无标签专项集。问题 3 并非问题 2 缺失鲁棒性的验证，但本次让两问使用**同一组 P-RMF-E Q2 检查点**，重新解释其完整输入预测。只读取附件 2 `aligned_50` 的 valid 728 条和附件 4 对齐版 20 条，不进行训练或用附件 4 选模型。

- 输入：Text `B×50×768`、Audio `B×50×74`、Vision `B×50×35`，内部观测 mask 为 Text/Vision/Audio 顺序的 `B×3×50`。附件 4 使用题方现有 Text 连续特征，不重跑 BERT；不会人为加入新的缺失。原始 Audio/Vision 内容区全零行仍作为不可观测位置。
- 检查点：`results/dual_baseline_v2/prmf_e/seed_{1111,2222,3333}/best.pt`，与 Q2 原三模型集成相同。极性采用三票多数决，平票按平均类别概率决定；强度为三个截断到 `[-3,3]` 的回归输出均值。该浮点平票实现已与 Q2 `ensemble_full.json` 的 Clean 指标逐项核对。
- 完整输入输出：三分类和强度回归；P-RMF 内部三路高斯不确定度权重为 `3×B×8×128`，对代理 token、通道、三个种子取均值得到每样本三模态 **proxy allocation**。该量不是预测贡献百分比，也不是 EMOE Router。

## 2. 解释定义和验证

**时间窗。**在有效内容内把每个模态的可观测槽位分成最多 4 槽的连续窗口。逐窗置零并计算原预测类别平均概率变化和强度有符号变化；每路选综合影响最高的窗口，保留全部窗口分布。Text 窗通过 `text_bert` 与 `raw_text` 的 WordPiece offset 回查原文。Audio/Vision 只能精确回查到 **aligned_50 槽位**；秒数和原视频帧号按视频时长均匀插值，CSV 中均标 `approx_`，不可写成精确时间戳。

**主要参考模态。**对三个模态分别遮挡其全部可观测内容，取原预测类别概率的有符号下降值；把正向下降归一化为 `perturbation_contribution_*`。总正向下降不足 `0.005` 时输出 `None`，避免无证据时强行指定模态。CSV 另外保留三路 proxy 权重和 proxy 最大模态，便于与扰动证据对照。

**可靠性。**验证集先比较选中三路窗口的合并置零与同模态、同长度随机窗口；再固定所选窗口，改用**同预测类且有效长度相近的另一条验证样本特征**替换，保留原观测 mask，与匹配随机窗比较。独立替换只使用模型预测，不用真实类别挑 donor。两种干预均只说明预测敏感性，不能证明人类语义上的因果贡献；同长度随机替代窗不存在的样本在 donor 对比中排除。

## 3. 验证结果

附件 2 `valid=728`，完整输入；极性由 Q2 三模型多数决，强度取均值。

| 模型 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| 本次 P-RMF-E 三模型集成 | **0.6580** | **0.6289** | **0.5873** | **0.6419** |
| 先前 Q3 冻结 EMOE 单检查点 | 0.6085 | 0.5827 | 0.6287 | 0.6088 |

这两行可描述各自 Q3 运行状态；一个是三模型集成、一个是单模型，不能当作等规模结构优劣实验。P-RMF-E 的 Negative/Neutral/Positive 召回分别为 `0.7476/0.4076/0.7396`，中性类别仍是主要错误来源。混淆矩阵和逐样本真值、预测见 `results/q3_prmf_e/validation_explanation_report.json`、`validation_explanations.csv`。

- 内部 proxy 权重均值按 Text/Vision/Audio 为 **`0.3306/0.3329/0.3365`**；每样本最大与次大权重的中位差仅约 `0.00354`。因此“proxy 最大模态”受极小数值差支配，不能作为主要解释结论。
- 整模态遮挡对原预测类别概率的平均下降按 Text/Vision/Audio 为 **`0.1092/0.0150/0.0212`**；proxy 最大模态与最大遮挡影响只在 **19.23%** 的验证样本一致。本模型解释应以实际扰动敏感度为主要依据，proxy 权重仅作模型内部信息。
- 选中窗口合并置零的平均预测类概率下降 `0.04174`，匹配随机窗 `0.00932`；配对差 **`0.03242`**，样本 bootstrap 95% 区间 **`[0.03046,0.03442]`**。这项比较使用了和选窗相同的置零干预，存在选择偏差。
- 更换为 donor 特征替换后，可比较样本 `687/728`；选中窗相对随机窗的预测类概率下降配对差 **`0.02874`**，95% 区间 **`[0.02603,0.03148]`**，`83.70%` 的可比较样本为正。强度绝对变化的配对差 `0.01529`，区间 `[0.01007,0.02105]`。donor 组合可能超出真实分布，仍需原视频人工核验。

## 4. 附件 4 全量输出

20/20 个 ID 与其同名原视频路径对应，并通过原文与 `text_bert` token ID 一致性检查。预测分布：Negative `9`、Neutral `2`、Positive `9`；这是**预测数，不是准确率**。按整模态扰动定义的主要参考模态：Text `15`、Audio `3`、Vision `2`。内部 proxy 最大模态却有 `19/20` 为 Audio，且权重近似均匀，再次表明两种含义不能混用。13 号样本视觉特征内容全零，视觉关键帧字段留空。

与旧 EMOE 的附件 4 输出按 ID 比较，20 条中有 15 条极性一致、5 条不同；附件 4 没有标签，不能据此判断哪一个预测正确。

| 产物 | 内容 |
|---|---|
| [附件 4 摘要 CSV](results/q3_prmf_e/attachment4_summary.csv) | 20 条预测、主要参考模态、三路作用、关键证据和视频路径 |
| [附件 4 完整 CSV](results/q3_prmf_e/attachment4_predictions_explanations.csv) | 概率、强度、proxy 权重、有符号扰动、关键窗与近似时间/帧 |
| [附件 4 全窗口重要性 CSV](results/q3_prmf_e/attachment4_window_importance.csv) | 每条样本每模态各时间窗的重要性及是否选中 |
| [验证报告](results/q3_prmf_e/validation_explanation_report.json) | 四指标、混淆矩阵、模态统计和两种扰动检验 |
| [验证集总览图](results/q3_prmf_e/validation_explanation_summary.png) | proxy、整模态消融及窗口敏感度 |
| [典型样本解释卡](results/q3_prmf_e/cards/) | Negative/Neutral/Positive 各一例，含三路时间重要性分布 |

## 5. 复现与提交边界

在 E 题根目录执行：

```powershell
$env:HF_HOME=(Resolve-Path 'EMOE_repro/hf_cache').Path
$env:HF_HUB_OFFLINE='1'
$env:MPLCONFIGDIR=(Resolve-Path 'EMOE_repro/results').Path
& 'EMOE_repro/.venv/Scripts/python.exe' 'EMOE_repro/q3_prmf_e.py' --sample-batch 16 --variant-batch 48
```

若尚无 `results/q3/attachment4_aligned50.npz`，先按 [原 Q3 说明](Q3_README.md)执行 `q3_prepare.py`。当前三个 FP32 检查点各约 **29.77 MiB**，总量约 **89.3 MiB**，因此这次实验结果尚不是满足赛题约 50 MB 总附件限制的提交包；精简/量化及其预测一致性须另行验证。附件 4 无标签，不能报告 Accuracy、F1、MAE、Pearson，也不能用于选模型。
