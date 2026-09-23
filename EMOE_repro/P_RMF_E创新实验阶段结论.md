# P-RMF-E 增量创新与消融：阶段结论

状态：按用户要求，完成当前小验证后**暂停继续优化**。本轮只使用附件 2 aligned_50 的 train/valid；附件 2 test 与无标签附件 3/4 均未用于训练、选模或消融裁决。基线为 `results/dual_baseline_v2/prmf_e/`，协议为相同的 7 组合连续缺失增强、3 种子、8 epoch、26 场景评估与单一 Clean+A/V30 复合分数选模。完整代码在 `q2_prmf_innovation.py`，原始结果在 `results/prmf_innovation/`。

## 预设实验与结果

| 单因素变体 | 训练种子 | 相对同种子基线：Clean F1 / MAE | 相对基线：25 缺失场景平均 F1 / MAE | 结论 |
|---|---:|---:|---:|---|
| 缺失位置 embedding | 1111 | `−.0050 / +.0023` | `−.0070 / +.0005` | 单种子筛查未过，停止扩展。 |
| 覆盖率校正原不确定度融合权重 | 1111/2222/3333 | `−.0024 / −.0007` | `−.0049 / −.0007` | 3 seed 未过；不与 embedding 叠加。 |
| 冻结完整输入教师→局部缺失学生的轻量双头一致性 | 1111/2222/3333 | `−.0018 / +.0014` | `−.0027 / +.0029` | 3 seed 未过；不纳入 Q2 主干。 |

F1、MAE 差均为 **新变体 − 同 seed P-RMF-E baseline**，所以 F1 正值、MAE 负值才有利。预先规定的晋升门槛是缺失网格 F1 至少 `+.005` 且 MAE 不升，并保持 Clean F1/MAE；**三个候选均未达到**。没有为追求单场景峰值改选 epoch，也没有调第二套损失系数。Q2 当前可复核主干仍是 P-RMF-E 双基线版本；EMOE 仍为对照。此次结果不能称为创新点已成功提升。

## 机制与失败定位

- `mask`：三路 128 维缺失 embedding 加在原 Linear 投影后、token Transformer 前，零初始化；代码验证初始共享参数和双头预测与基线完全一致。训练后 L/V/A embedding 范数约 `.052/.078/.044`，但同 seed 的 A/V 中段 30% F1 低 `.0106`、三模态中段 50% F1 低 `.0145`，说明显式信号未自动转化为更好的分类。
- `coverage`：只在原 P-RMF 的 VAE 不确定度权重进入跨模态注入前做 `softmax(log w + β log c)`，不改变原 proxy 表征。三个最佳 checkpoint 的 `β` 为 `−.00495/−.00494/+.00178`，非常接近零，且符号不稳定。虽然部分场景 MAE 略降，Text 中段 50% F1 平均低 `.0159`，三模态中段 50% 低 `.0175`。不应将这种权重称为已验证的缺失置信度。
- `consistency`：同 seed 的已训练 P-RMF-E 只在 **train 完整输入**上缓存软目标；教师极性正确且强度误差≤.75 的样本参与 `0.05×[T=2 KL + SmoothL1]`，原 CE+L1 和 VAE/重建损失不变。seed 1111 教师 train 中约 53.9% 样本满足门控。三模态中段 30% 在一个种子曾提高 F1 `.0108`，但三种子平均该场景低 `.0014`、MAE 高 `.0045`；整体缺失网格也未过门槛。其 Clean Neutral 召回平均由基线 `.380` 到 `.406`，但极端强度 `|y|≥1.5` 的 Clean MAE 由 `1.128` 升至 `1.152`。因此局部类别收益伴随回归代价，不足以晋升。

本轮每次只变一项；`mask_coverage` 组合没有运行，因为两个单因素均未形成可靠综合收益。此前 EMOE 的 padding/覆盖率/Router 缩容/蒸馏调试结果也没有跨模型迁移为 P-RMF-E 成功结论。后续如果继续优化，应先用 train/valid 研究 Text 长段缺失、中性类别与强度收缩的共同瓶颈，再单独预设新假设和验收门槛；当前不继续试验。

## 可复查产物

- `results/prmf_innovation/{mask,coverage,consistency}/comparison_summary.json`：同 seed 清洁/缺失网格配对差。
- 各变体各 seed 的 `report.json`、`scenario_grid.csv`、`best.pt`：逐轮训练/选模、26 场景、参数与显存记录。`mask` 只有 seed 1111；其余各 3 seed。
- `results/prmf_innovation/slice_summary.json`、`sample_predictions.csv`：Clean、Text50、A+V30、三模态30/50 的中性召回、强情感 MAE 与逐样本输出。
- `P_RMF_E与EMOE_Q2双基线对比报告.md`：固定基线和附件 3 数据陷阱审计。本次新模型**没有**替换其 checkpoint 或无标签预测 CSV。
