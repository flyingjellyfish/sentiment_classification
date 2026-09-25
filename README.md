# E 题：P-RMF-E 多模态情感识别

本仓库保留一套正式模型：P-RMF-E 三种子集成（1111、2222、3333）。同一组权重完成问题 2 的局部连续缺失预测和问题 3 的完整输入解释。EMOE 与其他变体只作为论文中的对照和消融结论。

## 结果

| 附件 2 valid | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| Clean，三模型集成 | .6580 | .6289 | .5873 | .6419 |
| 25 个连续缺失场景均值，Text 先 [UNK] 重编码 | .6283 | .5937 | .6043 | .6179 |

分类采用多数决，平票按平均概率破局；强度取三个截断到 [-3,3] 的回归值均值。旧记录中的 Clean Acc .6593 采用另一种平票实现，不能当作额外涨点。附件 3/4 无标签，只有全量预测与解释，没有准确率。

## 核心文件

- [论文手工作报告](docs/Q2_Q3_论文手工作报告.md)：任务、模型结构、公式、对比与消融；开头注明了最终采用口径。
- [实验尝试与后续工作](docs/实验尝试与后续工作.md)：历次优化、失败案例与待验证方向。
- [Q2 pipeline SVG](docs/figures/P_RMF_E_Q2_pipeline.svg)及[预览图](docs/figures/P_RMF_E_Q2_pipeline_preview.png)：图中候选分支属于消融，不在正式模型中。
- [Q3 P-RMF-E 说明](EMOE_repro/Q3_PRMF_E_README.md)：同三权重的解释流程、可靠性验证和附件 4 结果。
- [产物索引](EMOE_repro/results/README.md)：三份正式权重、Q2/Q3 原始表、逐样本 CSV 与可视化。

## 代码与输入

主模型为 [prmf_e.py](EMOE_repro/prmf_e.py)，Q2 训练和评价入口为 [q2_dual_baseline.py](EMOE_repro/q2_dual_baseline.py)，附件 3 推理入口为 [q2_dual_attachment3.py](EMOE_repro/q2_dual_attachment3.py)，Q3 同模型解释入口为 [q3_prmf_e.py](EMOE_repro/q3_prmf_e.py)。P-RMF 上游代码由 [fetch_prmf_source.py](EMOE_repro/fetch_prmf_source.py)按固定 revision 下载；本地下载目录不入库。题方附件、视频、BERT 缓存也不入库。

附件 2 使用 aligned_50：Text 50×768、Audio 50×74、Vision 50×35；train/valid/test 分别 3395/728/727。标签 0 仅表示 Neutral，三分类为 Negative/Neutral/Positive。附件 3 只有 text_bert，需用冻结 bert-base-uncased 重建 Text 特征；已有的缺失段直接读取，推理不再二次遮挡。附件 4 是无标签完整输入解释集。

Windows RTX 5060 Ti 16 GB 本机已验证 Python 3.10、PyTorch 2.10.0+cu128。运行前将题方附件置于本地 E题数据/，安装 [requirements.txt](requirements.txt)，并执行：

```powershell
python EMOE_repro/fetch_prmf_source.py
python EMOE_repro/q2_dual_baseline.py --model prmf_e --seed 1111
```

另两个 seed 为 2222/3333。Q3 的具体运行命令和附件 4 准备步骤见其说明。三份 FP32 权重合计约 89.3 MiB，仓库保存的是可复核实验版本；若赛题提交附件限制约 50 MB，还需另行验证压缩方案。
