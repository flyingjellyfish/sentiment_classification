# E 题：多模态情感识别与局部缺失实验

本仓库保存 E 题 aligned_50 特征上的 EMOE / P-RMF-E 双任务建模、局部连续缺失评估，以及 EMOE 问题 3 的解释性分析。**当前 Q2 以 P-RMF-E 为正式主干，EMOE 保留对照；后续候选在严重文字缺失场景有效，但尚未整体晋升。**附件 3/4 无标签；不使用伪标签或 MOSI、IEMOCAP 等题外情感数据。实验生成的指标表、逐样本结果和可视化已整理在 [results/](EMOE_repro/results/README.md)。

## 已完成的结果

| 附件 2 valid，3 seed 均值 | EMOE | P-RMF-E |
|---|---:|---:|
| Clean Accuracy / Macro-F1 | .6140 / .5818 | **.6392 / .6069** |
| Clean MAE / Pearson | .6259 / .6119 | **.6025 / .6320** |
| 25 个局部缺失场景均值 Accuracy / Macro-F1 | .6123 / .5814 | **.6367 / .6033** |
| 三模态同段缺失 50% Macro-F1 | .5719 | **.5994** |

上表沿用旧的“完整 BERT 特征后置遮挡”协议。新一轮发现：同一冻结 P-RMF-E 在 Text 中段缺失 50% 时，先把词元换成 `[UNK]` 再重编码会使 valid F1 从 **.5869** 变为 **.4914**。据此训练的“条件式 padding 清零 + 缺失中性类别加权”候选，在附件 2 独立 test 的该场景使 Acc/F1 **.5571/.4845→.5740/.5358**；但 Clean test **.6717/.6133→.6657/.5977**，25 缺失场景平均只 **+.0013 Acc/+.0012 F1**，仍不替换正式主干。[论文手精简版](docs/Q2_Q3_论文手精简版.md)、[新协议实验报告](docs/Q2_UNK重编码与padding增量实验_20260924.md)。

## 数据与接口要点

- 附件 2 aligned_50：train 3395、valid 728、test 727；Text 50×768、Audio 50×74、Vision 50×35。只用 train/valid 训练和选模。
- 附件 3 的 30 个 pkl **有 text_bert 而没有 text**。推理需用相同的冻结 bert-base-uncased 重建 768 维 Text，已通过附件 2 可配对样本核验。
- 附件 3 有 27/30 条在有效 Text 位置出现 [UNK]，大多与 A/V 零值同段。旧版 EMOE 推理漏标 Text 缺失，旧 CSV 只作历史记录；正式入口 q2_dual_attachment3.py 用三路现有缺失，且不二次遮挡。
- 连续 regression_labels=0 是中性；分类 classification_labels=0/1/2 分别为负/中/正，逐条等于 sign(regression_labels)+1。
- 附件 3/4 无真值，不能计算测试 Accuracy/F1 或用预测分布选模型。内部 mask 通道顺序为 Text/Vision/Audio；对外报告需明确字段名。

赛题原始附件、特征、预训练 BERT 和 checkpoint 保持本地；**CSV/JSON 结果、逐样本预测、日志与 PNG 图已纳入 Git**，供复核与论文绘图。本地数据按赛题目录放在仓库根目录的 E题数据/ 下；结果目录内容与排除项见 [产物索引](EMOE_repro/results/README.md)。

## 代码地图

| 路径 | 用途 |
|---|---|
| EMOE_source/ | [EMOE 作者源码](https://github.com/fuyyyyy/EMOE)固定 Git submodule；本题修改在 patches/emoe_q2_local.patch。 |
| EMOE_repro/P_RMF_upstream/ | fetch_prmf_source.py 拉取的 [P-RMF 作者源码](https://github.com/hawksilent/P-RMF)固定 revision；下载目录不提交。 |
| EMOE_repro/missing_protocol.py | 50 步有效内容、源零值与连续窗口 mask 的共用协议。 |
| EMOE_repro/prmf_e.py | P-RMF 的 E 题双任务适配及可关闭的消融模块。 |
| EMOE_repro/q2_dual_baseline.py | Q2 公平训练与 26 场景评价主入口。 |
| EMOE_repro/q2_dual_*.py | 双基线汇总、案例、bootstrap 和附件 3 无标签推理。 |
| EMOE_repro/q2_prmf_innovation.py | 已完成的单因素消融；配套 _summary.py 和 _slices.py。当前暂停新训练。 |
| EMOE_repro/q2_unk_reencode_experiment.py | 文字 `[UNK]` 重编码、条件式 padding、中性加权的训练和同协议比较。 |
| EMOE_repro/q2_model_upgrade_experiment.py | masked attention、双头耦合、局部恢复的探索性单种子筛查；尚无晋升候选。 |
| EMOE_repro/q3_explain.py、q3_independent_faithfulness.py | EMOE Q3 模态/时间解释与独立扰动检验；见 [Q3 说明](EMOE_repro/Q3_README.md)。 |
| EMOE_repro/results/ | 已提交指标 JSON、逐样本 CSV、训练日志和可视化图；大权重、特征归档与缓存仍本地保存。 |
| docs/archive/ | 早期 EMOE 方案与复盘，供溯源；当前选择与流程以本 README 及正式报告为准。 |

## Windows 本机准备

已验证：Windows、RTX 5060 Ti 16GB、Python 3.10、PyTorch 2.10.0+cu128。作者仓库旧 torch 1.9 要求不适用于该显卡。先在独立环境安装适配显卡的 PyTorch，再安装 requirements.txt 中的非 Torch 依赖。

    git submodule update --init
    git -C EMOE_source apply --check ../patches/emoe_q2_local.patch
    git -C EMOE_source apply ../patches/emoe_q2_local.patch
    python -m pip install -r requirements.txt
    python EMOE_repro/fetch_prmf_source.py

仅运行探索性 `q2_model_upgrade_experiment.py --variant masked_attention` 时，还需在仓库根目录对下载后的 P-RMF 作者文件应用可审查补丁：

    git apply --check patches/prmf_masked_attention.patch
    git apply patches/prmf_masked_attention.patch

EMOE 固定 revision 为 c4759c748105e18354cd9082b1d8c6d310f1f9d8，P-RMF 固定 revision 为 769c31b50f173b3f7670404a0939a95388393610。训练直接使用附件 2 已给的 Text 特征，不重跑 BERT。附件 3 Text 重建需要本地缓存有 bert-base-uncased，可通过 HF_HOME 指向 EMOE_repro/hf_cache；它是公开预训练语言模型，不是额外情感数据。Q3 部分附件 4 pickle 需 NumPy 2 环境一次性准备，详见 Q3_README；无需升级 Q2 训练环境。

## 固定验证工作流

1. **接口审计**：运行 audit_q2_data.py、q2_unk_baseline_audit.py、q2_attachment3_missing_audit.py；确认字段、有效长度、源零值和标签映射。附件 3/4 只查接口。
2. **基线复核**：从仓库根目录运行 python EMOE_repro/q2_dual_baseline.py --model prmf_e --seed 1111；EMOE 改为 --model emoe，再各跑 2222/3333。每样本每 epoch 的连续窗口与 batch 顺序按同 seed 固定。最佳 epoch 只按 Clean + A/V 中段 30% 的复合 valid 分数选择。
3. **结果核对**：运行 q2_dual_summarize.py、q2_dual_cases.py、q2_dual_bootstrap.py；报告 26 场景的 Accuracy、Macro-F1、MAE、Pearson，缺失模态/位置/10–50% 和 Neutral/强情感切片，并记参数、显存和训练时间。标称与实际新增缺失率一起报告。
4. **新协议与创新验证**：运行 q2_unk_reencode_experiment.py，以先 `[UNK]` 再冻结 BERT 的同一缺失输入链比较原模型与新候选。只在附件 2 train/valid 开发；先单 seed 筛查，入选再做 3 seed、26 场景及四指标。附件 2 test 已用于一次冻结候选检查，不再用它调参。masked attention 和双头耦合仅有单种子负结果；局部恢复仍在探索，均不是已证实涨点。
5. **冻结后推理**：模型确定后才运行 q2_dual_attachment3.py；附件 3 不二次 mask，也不作有标签指标。Q3 Router/proxy 权重是内部融合量；时间证据须用独立扰动核查。无可信映射时，只把 50 槽索引称为准确定位，不能把线性换算秒数或帧号写成精确真值。

历史方法见[双基线统一协议](EMOE_repro/P_RMF_E双基线统一协议.md)和[创新预设协议](EMOE_repro/P_RMF_E创新实验预设协议.md)。正文引用指标前先确认对应的划分、输入协议和模型状态；附件 3/4 无标签输出只作展示。

## 文献与来源

- Fang et al., [EMOE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html)。
- Zhu et al., [P-RMF, ACL 2025](https://aclanthology.org/2025.acl-long.1075/)。
- 连续 span、双任务头、缺失 embedding、覆盖率校正和轻量一致性是本项目对 E 题的适配或探索，不是原论文的实验结果。

