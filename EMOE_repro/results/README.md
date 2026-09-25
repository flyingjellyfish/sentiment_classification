# 正式模型与结果索引

正式模型为 dual_baseline_v2/prmf_e 下 seed_1111、seed_2222、seed_3333 的 best.pt，各约 29.8 MiB。Q2 与 Q3 使用同一组三种子权重。其他候选权重不纳入本次仓库交付。

| 路径 | 用途 |
|---|---|
| dual_baseline_v2/summary.json、slice_report.json、scenario_mean.csv | Q2 双基线、缺失类型/率/位置/时长汇总 |
| dual_baseline_v2/prmf_e/seed_*/report.json、scenario_grid.csv | 正式 P-RMF-E 的逐种子训练选择与 26 场景指标 |
| dual_baseline_v2/sample_predictions.csv、failure_cases_seed1111.csv | 验证集逐样本预测与错误案例 |
| dual_baseline_v2/missing_rate_curves.png、missing_position_curves.png | Q2 缺失率和位置曲线 |
| dual_baseline_v2/attachment3_unlabeled_predictions.csv | 附件 3 全量 30 条 × 双基线 × 三种子的原始推理记录；正式集成使用其中 P-RMF-E 三种子 |
| dual_baseline_v2/attachment3_prmf_e_ensemble.csv | 附件 3 正式 P-RMF-E 集成的每样本一行预测，共 30 行 |
| q3_prmf_e/validation_explanation_report.json、validation_explanations.csv | Q3 同模型 Clean valid 四指标、模态和时间解释 |
| q3_prmf_e/attachment4_summary.csv、attachment4_predictions_explanations.csv、attachment4_window_importance.csv | 附件 4 全量 20 条预测、贡献和证据 |
| q3_prmf_e/validation_explanation_summary.png、cards/*.png | Q3 汇总和典型样本解释图 |

附件 2 valid 是有标签评估集；附件 3/4 没有标签，不能据其 CSV 计算 Accuracy/F1。Text 缺失的 [UNK] 重编码协议和旧后置 BERT 特征遮挡协议不能混比。正式集成的 Clean valid Acc/F1/MAE/Pearson 为 .6580/.6289/.5873/.6419，25 场景新协议均值为 .6283/.5937/.6043/.6179。详见[实验汇总](../../docs/实验尝试与后续工作.md)。
