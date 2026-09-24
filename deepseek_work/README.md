# deepseek_work —— DeepSeek Agent 独立工作区

> **本目录是 DeepSeek agent 独占的工作区**，用于避免与其他 agent 的文件互相覆盖。
> 其他人的 `docs/`、`EMOE_repro/`、`EMOE_source/`、`patches/` **一律只读**，本 agent 未写入、未修改。
> 所有实验在本机完成：Windows / RTX 5060 Ti 16GB / torch 2.10.0+cu128 / Python 3.10（只读复用 `EMOE_repro/.venv`）。

---

## ⚠️ 请先读：勘误

**`评审回应与勘误.md`** 是《问题2与问题3现有工作流独立评审》（`docs/Q2_Q3_pipeline独立评审.md`）的回应，记录了本文档集合中**确实存在的错误**及更正。**凡与勘误冲突的旧表述，以勘误为准。**

三个最重要的更正：

| # | 原结论（已撤回） | 更正 |
|---|---|---|
| A1 | "局部模态缺失近乎伪问题，没有可鲁棒的东西" | 该结论只成立于**对预计算上下文特征后置遮挡** `M⊙B(u)` 的协议。附件 3 的真实链路是**先 `[UNK]` 再经冻结 BERT 重编码** `M⊙B(C(u))`；同 valid 同窗口下，冻结 P-RMF-E 的 Text 中段 50% F1 由 `.5869` 降到 **`.4914`**，相对 Clean 降幅由 **.020 变为 .1155**（由另一位 agent 的 `q2_unk_reencode_experiment.py` 测得） |
| A2 | "附件 3 实际缺失约 68%，比训练上限 50% 更严重" | 68.4% 是**全 50 槽**零值比例；按**有效内容区间**为音频 **20.67%** / 视觉 **21.09%**，出现 UNK 的样本 **27/30**，有音频零行的样本 **27/30**，音频零段 **3.17 段/样本** |
| A9 | "推荐交付用 9 seed 集成" | 9 模型 FP16 约 **139 MB**，赛题总附件 **≤ 50 MB**。集成只能作实验对照，不是零成本交付方案 |

§4 还有一处场景数标注错误（`p6` 把 4 场景写成 25 场景），§5 的温度标定实验是恒真命题。详见勘误。

---

## 目录内容

### 报告（按建议阅读顺序）

| 文件 | 内容 |
|---|---|
| **`评审回应与勘误.md`** | **先读这个.** 对独立评审的逐条回应、撤回的结论、仍成立的结论、接受的改进建议 |
| `E题_赛题分析与问题重述.md` | 赛题背景、逐问要求拆解、已知条件核验、三问递进关系论证、完整问题重述 |
| `E题已有工作与Q2Q3建模现状及后续工作.md` | 读遍仓库 11 篇 md + 重算结果文件后的盘点：文档谱系、数据规律、Q2/Q3 现状、10 项缺口（第 7 节后续路线已被取代） |
| `Q2_pipeline与创新路线_评估意见.md` | 对 Q2 pipeline 的批评、64% 定位、5 个免费杠杆实测（⚠️ 部分建议已被实验否证） |
| `Q2_实验验证报告.md` | 第一轮 24 次训练：12 项判断的成立/否决表（🔴 核心结论 A1 已撤回） |
| `Q2_涨点与后续优化_最终报告.md` | 多 seed 集成涨点实验 + 原计划 pipeline 优化实验全部实跑（🔴 §5 场景数须更正） |
| `EBMC_思想适用性评估.md` | CVPR 2026 论文 EBMC 的适用性评估（🔴 §3 论证基础被削弱） |
| `papers/EBMC_2026.txt` | EBMC 论文全文抽取（`extract_pdf.py` 用带 pdfplumber 的 runtime python 生成） |

### 实验代码

| 目录 | 内容 |
|---|---|
| `exp/` | 第一轮：`e4_ensemble.py`（集成）、`e5_testbed.py`（受控消融台）、`e6_masking.py`、`e8_modality_sensitivity.py`、`e9_text_positions.py`、`e10_ensemble_replicate.py`、`e123_diagnostics.py`、`e7_noise_floor.py`、`aggregate.py` |
| `exp2/` | 第二轮：`p1_train_seeds.py`（保真复现 + 6 个新种子）、`p2_ensemble_curve.py`、`p3_pipeline_planned.py`（mask / mask_coverage）、`p4_ensemble_grid.py`、`p5_bootstrap.py`、`p6_pipeline_verdict.py` |
| `exp3/` | EBMC 相关：`eba_modality_only.py`（7 种模态组合）、`ebb2_ebmc_official.py`（官方机制 4 变体）、`ebc_paired_tests.py`、`ebd_q3_reliability.py` |
| `verify/` | 附件接口核验：`verify_data.py`、`verify_pkl.py`、`verify_pkl2.py`、`verify_unk.py` |
| 根目录 | `analyze_existing.py`（重算仓库结果规律）、`diagnose_q2_ceiling.py`、`ceiling_extra.py`、`test_deshrink.py`、`test_decision_fusion.py`、`test_ensemble_prior.py` |

### 结果文件（JSON / CSV）

`exp/out/`、`exp2/*.json`、`exp2/p4_ensemble_grid.csv`、`exp3/*.json`。
**checkpoint（`*.pt`，76 个约 714 MB）与缓存 `p4_cache.npz` 已由 `.gitignore` 排除，仅本地保留。**

---

## 关键数字（已按勘误校正）

### Q2 主干与集成（附件 2 valid，728 条）

| 方案 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| EMOE（正式双基线 3 seed 均值） | .6140 | .5818 | .6259 | .6119 |
| P-RMF-E（正式双基线 3 seed 均值） | .6392 | .6069 | .6025 | .6320 |
| P-RMF-E 原 3 seed 投票 + 回归平均 | **.6593** | **.6287** | .5873 | .6418 |
| P-RMF-E 9 seed 投票 + 回归平均 | .6525 | .6202 | .5861 | .6443 |
| 9 seed 集成在完整 26 场景网格（**最完整口径**） | 见 `exp2/p4_ensemble_grid.csv` | | | |

- **集成相对单模型均值**：逐场景配对 26/26 为正，ΔAcc 均值 **+0.0119**；按 **239 个 video_id** 成组 bootstrap 的条件区间 **[0.00343, 0.02608]**（评审口径，ΔAcc = 0.01496）。
- **3 seed 与 9 seed 的差别**落在抽样噪声内（26 场景仅 1 个显著）；`.6593` 是原有三元组抽得好的结果，9 选 3 子集平均为 `.6447`。

### 缺失协议的两种口径（**不可混用**）

| 同一冻结 P-RMF-E、同一 valid 窗口 | 后置遮挡 `M⊙B(u)` | `[UNK]` 重编码 `M⊙B(C(u))` |
|---|---:|---:|
| Clean F1 | .6069 | .6069 |
| Text 中段 50% F1 | .5869 | **.4914** |
| 三模态中段 50% F1 | .5994 | **.4996** |
| 相对 Clean 的降幅 | ≈ .020 | **≈ .1155** |

> 数据来源：另一位 agent 的 `EMOE_repro/q2_unk_reencode_experiment.py` 与其报告 `docs/Q2_UNK重编码与padding增量实验_20260924.md`。

### 模态组合各自能达到的水平（`exp3/eba_modality_only.py`，简化台 3 seed）

| 组合 | Macro-F1 | 组合 | Macro-F1 |
|---|---:|---|---:|
| T | .5911 | A+T | .5952 |
| A | .2841 | T+V | .6027 |
| V | .3510 | **T+A+V** | **.6026** |
| **A+V** | **.3523** | 多数类基线 | .2114 |

A/V 单独训练明显高于多数类（携带信息），但全部 A/V 对全模态只贡献 **+.0115**（**这是特定架构/预算下的已实现增量，不是贡献上限**）。

### 原计划 pipeline 优化实验（`exp2/p3_pipeline_planned.py` + `p6_pipeline_verdict.py`）

| 变体 | ΔClean F1 | Δ（本脚本 4 个缺失场景）F1 | 真 25 场景 ΔF1 | 过门槛 |
|---|---:|---:|---:|---|
| mask | −0.0065 | −0.0122 | 未评（只存 5 场景） | ❌ |
| coverage | −0.0024 | −0.0114 | **−0.00487** | ❌ |
| mask_coverage（此前从未运行） | −0.0052 | −0.0077 | 未评（只存 5 场景） | ❌ |
| consistency | −0.0018 | −0.0028 | **−0.00268** | ❌ |

**工程决定不变（暂不采用）；论文中的实验范围与数值须按上表更正。**

### EBMC 机制检验（`exp3/ebb2_ebmc_official.py` + `ebc_paired_tests.py`）

| 变体 | n | ΔMacroF1 | ΔMAE | 判定 |
|---|---:|---:|---:|---|
| twostage（两阶段课程） | 6 | **−0.0217** | **+0.0529** | 🔴 配对 t 检验**显著变差**，6/6 全为负 |
| emc | 12 | **+0.0089** | −0.0007 | ⚪ 不显著（8/12 为正） |
| imtd | 6 | +0.0048 | +0.0077 | ⚪ 不显著（5/6 为正，Pearson −.0122） |

### Q3 主要参考模态的信号比较（`exp3/ebd_q3_reliability.py`）

653/728 条有明确扰动主导模态；一致率：

| 信号 | 一致率 | Δ vs Router | 95% bootstrap |
|---|---:|---:|---|
| **Router argmax（现状）** | **0.3476** | — | — |
| IMTD 离散度（回归教师） | 0.3247 | −0.0230 | [−0.0750, +0.0337] |
| IMTD 离散度（概率教师） | 0.3216 | −0.0260 | [−0.0796, +0.0245] |
| 单模态预测的多种子标准差 | 0.3691 | +0.0214 | [−0.0383, +0.0827] |
| 全局最准模态 | 0.2328 | −0.1149 | [−0.1608, −0.0689]（显著变差） |

**IMTD 的原始构造并未超过现有 Router。** 该结论以"与旧 EMOE 消融一致"为目标，按评审提醒，此目标本身需重新论证。

---

## 运行方式

用其他 agent 已建好的虚拟环境（**只读调用，不安装、不修改**）：

```powershell
$py = "D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\EMOE_repro\.venv\Scripts\python.exe"
Set-Location "D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\deepseek_work"

# 接口核验
& $py -X utf8 verify\verify_pkl2.py
& $py -X utf8 verify\verify_unk.py

# 重算仓库已有结果的规律（无需训练）
& $py -X utf8 analyze_existing.py
& $py -X utf8 diagnose_q2_ceiling.py
& $py -X utf8 ceiling_extra.py

# 第一轮：受控台单因素消融（约 10 min）
Set-Location exp
& $py -X utf8 e5_testbed.py --variants base,dist,ordinal,amp,missing_emb --seeds 1111,2222,3333 --epochs 25 --tag e5
& $py -X utf8 e8_modality_sensitivity.py     # 会写 exp/out/e8_ckpt/
& $py -X utf8 e9_text_positions.py
& $py -X utf8 e123_diagnostics.py

# 第二轮：集成与 pipeline 优化（约 45 min）
Set-Location ..\exp2
& $py -X utf8 p1_train_seeds.py --seeds 1111,4444,5555,6666,7777,8888,9999 --tag prmf_extra --epochs 8
& $py -X utf8 p2_ensemble_curve.py
& $py -X utf8 p3_pipeline_planned.py --variants mask,mask_coverage --seeds 1111,2222,3333
& $py -X utf8 p4_ensemble_grid.py             # 首次约 7 min，之后走 p4_cache.npz
& $py -X utf8 p5_bootstrap.py
& $py -X utf8 p6_pipeline_verdict.py

# 第三轮：EBMC 机制（约 45 min）
Set-Location ..\exp3
& $py -X utf8 eba_modality_only.py 1111,2222,3333
& $py -X utf8 ebb2_ebmc_official.py 1111,2222,3333,4444,5555,6666 base2,twostage,emc,imtd
Copy-Item ebb2_ebmc_official.json ebb2_round1_4variants.json -Force   # 下一步会覆盖
& $py -X utf8 ebb2_ebmc_official.py 7777,8888,9999,10101,11111,12121 base2,emc
& $py -X utf8 ebc_paired_tests.py
& $py -X utf8 ebd_q3_reliability.py           # 依赖 exp3/ck_T|A|V_*.pt
```

论文 4 原始 PDF 文本抽取需用带 `pdfplumber` 的 runtime python：

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 papers\extract_pdf.py
```

---

## 环境与踩坑记录

- **沙箱限制**：把外部程序输出用管道捕获（`| Select-Object`）会被拒绝；`>` 重定向到文件不可靠。**请直接运行、不要接管道**。
- **NumPy 版本**：`.venv` 是 NumPy 1.26；附件 4 的 pkl 由 **NumPy 2** 保存，直接 `pickle.load` 会报 `No module named 'numpy._core.numeric'`。`verify/verify_pkl2.py` 用 `sys.modules` 别名兼容。需要 `pdfplumber`/`numpy 2` 时用上面的 codex runtime python。
- **`openpyxl` 缺失**：`.venv` 无法用 pandas 读附件 1 的 `label-100.xlsx`，改用解压 xlsx + 解析 XML。
- **不写其他目录**：所有训练 checkpoint、结果 JSON、缓存都写在 `deepseek_work/` 下，并在文件名/目录名上与本 agent 绑定。
- **磁盘占用**：本目录约 722 MB，其中 714 MB 是 checkpoint；上表的 `git` 排除项已覆盖。

---

## 与其他 agent 结论的关系

- 本工作区的定位是**诊断、复核与受控消融**，不替代仓库的正式主干决定。
- 正式 Q2 主干仍是 `EMOE_repro/results/dual_baseline_v2/` 下的 P-RMF-E；Q3 仍用旧 EMOE checkpoint。两者均由其他 agent 维护，本 agent 未改动。
- 本工作区**推翻过自己的若干结论**（见勘误），也**修正过仓库的两处统计口径**（附件 3 缺失率、p6 场景数）。这些更正已同步给评审文档。
