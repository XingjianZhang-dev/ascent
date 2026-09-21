# ASCENT (ARRAY-D-26-05300) 原始数据与代码位置审计

审计日期：2026-09-20  
项目根目录：`/Users/xingjianzhang0816/Documents/ChatGPT/Ascent -- TNNLS`  
逐项机器可读清单：`reports/ASCENT_REPRO_MATERIAL_LOCATIONS_2026-09-20.tsv`

> 第二遍证据审计已纠正 C-04、R-02、R-06、S-02、S-03 的首轮状态与 R-02 的 panel 引用。详情见 `reports/ASCENT_SECOND_PASS_EVIDENCE_AUDIT_2026-09-20.md`。以下计数已按第二遍结果更新。

## 1. 结论

原清单的 128 项（C 66、D 34、E 7、R 6、W 9、S 6）已逐项定位：

| 状态 | 数量 | 含义 |
|---|---:|---|
| `FOUND` | 100 | 有直接实现、原始产物或已保存的审计证据 |
| `DISTRIBUTED` | 3 | 内容完整，但分散在多个脚本/档案中 |
| `ARCHIVED_ONLY` | 1 | 完整内容只在最终灾备压缩档中 |
| `PARTIAL` | 17 | 有主要证据，但缺独立脚本、逐条表或完整节点快照 |
| `MISMATCH` | 2 | 当前真实实现/仓库状态与清单要求不同 |
| `ABSENT` | 5 | 没找到对应文件 |

论文主要数值所需的原始输入、逐行模型输出、冻结配置、分析 JSON、统计脚本和绘图输入均已找到。没有发现“论文核心结果只剩汇总数字、原始数据已丢失”的情况。

真正没有找到的是：`C-39` 功效脚本、`W-03` AI 辅助说明、`W-05` 代码许可证、`W-06` 产物许可证、`W-08` Zenodo DOI。第二遍已经保存了完整 reachable Git blob 与发布/备份档案的高信号密钥扫描记录。另有若干项目只做到部分满足，见第 7 节。

## 2. 最重要的总入口

| 用途 | 位置 |
|---|---|
| 核心 ASCENT 实现 | `ascent/` |
| 全部实验构造、推理、分析、作图 | `experiments/` |
| 冻结配置、模型 revision、权重分片 SHA-256 | `configs/` |
| 冻结输入面板 | `data/` |
| 原始逐行输出与分析产物 | `artifacts/` |
| 论文统一证据 CSV | `paper/data/evidence/` |
| 论文自动生成表格 | `paper/generated/` |
| 可移植环境/复现说明 | `reproduction/`、`supplementary/` |
| 逐项 C/D/E/R/W/S 映射 | `reports/ASCENT_REPRO_MATERIAL_LOCATIONS_2026-09-20.tsv` |
| 最终 GPU 灾备 | `backups/ASCENT_GPU_FINAL_20260823/ASCENT_GPU_FINAL_20260823.tar.zst` |
| 灾备全文件 SHA/大小清单 | `backups/ASCENT_GPU_FINAL_20260823/PROJECT_FILES_SHA256.tsv` |

最终灾备 SHA-256：

```text
431049c38a3c91e7b0fb1f8a3d5a52484aeae609a81a12af0c9c5272a5e00821
```

Array 补充材料包：`paper/submission/ASCENT_Array_Supplementary_Material_2026-08-23.zip`  
其 SHA-256：

```text
b6d14bf7dee8aa7162ba18a2a6856c758e506aaab5b87c1430ed4bd54617866b
```

补充包的源路径到包内别名映射由 `experiments/build_neural_networks_supplement.py` 定义；`experiments/build_array_supplement.py` 是 Array 构建入口。

## 3. 各核心实验的完整证据链

### 3.1 官方 16K BABILong

| 层级 | 精确位置 |
|---|---|
| 官方源数据及来源清单 | `artifacts/babilong_official_16k_qa23_source/` |
| 下载/核验代码 | `experiments/fetch_babilong_official_source.py` |
| 面板构造代码 | `experiments/prepare_babilong_panel.py` |
| 10 个冻结面板与 hash | `data/babilong_16k_canonical_confirmation/` |
| 冻结配置与模型 SHA | `configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json` |
| 推理代码 | `experiments/run_babilong_prompt.py` |
| 逐行原始输出 | `artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/canonical/{0p5b,1p5b,3b}/` |
| 3B raw 表示消融 | `artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/raw/3b/` |
| 统计分析 | `experiments/analyze_babilong_canonical_coscale_16k_confirmation.py` |
| 汇总结果 | `artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/analysis.json` |
| 文件校验和 | `artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/SHA256SUMS.txt` |

规模：输入 11 个文件、约 48 MB；原始输出/分析 42 个文件、约 3.1 MB。

### 3.2 语义 holdout

| 层级 | 精确位置 |
|---|---|
| training-generation 源 | `artifacts/babilong_train_8k_qa23_source/` |
| score-blind 构造与三元指纹排除 | `experiments/prepare_babilong_semantic_holdout.py` |
| 800 行、10 面板、零重叠记录 | `data/babilong_train_8k_semantic_holdout_confirmation/` |
| 冻结配置 | `configs/babilong_qwen2p5_canonical_semantic_holdout_8k_confirmatory.json` |
| 逐行原始输出 | `artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6/{0p5b,1p5b,3b}/` |
| 分析代码 | `experiments/analyze_babilong_canonical_semantic_holdout.py` |
| 汇总与零重叠 gate | `artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6/analysis.json` |

规模：输入 11 个文件、约 24 MB；输出 32 个文件、约 2.3 MB。

### 3.3 Qwen 3×3 精确 factorial

| 层级 | 精确位置 |
|---|---|
| categorical channel / exact posterior | `ascent/certified_channel.py` |
| 数据和配置构造 | `experiments/prepare_noisy_composition_sum_candidate.py` |
| 10 个面板（1 个开发、9 个确认） | `data/posttraining_noisy_composition_sum_numeric_candidate/` |
| 冻结配置 | `configs/posttraining_noisy_composition_sum_numeric_candidate.json` |
| candidate-normalized NLL 推理 | `experiments/run_noisy_composition_candidate.py` |
| 81 个端点-面板原始文件 | `artifacts/sum_numeric_candidate/confirmation/` |
| 统计/嵌套/13,824 检查代码 | `experiments/analyze_noisy_composition_candidate.py` |
| 汇总结果 | `artifacts/sum_numeric_candidate/confirmation_analysis.json` |
| 第二节点 7B/K5 256 行概率向量 | `artifacts/sum_numeric_candidate/audit_node2/rounds_5_sum_numeric_candidate_panel_1_qwen2p5-7b-instruct.json` |

规模：输入 11 个文件、约 564 KB；原始输出 81 个文件、约 29 MB。

重要边界：`confirmation_analysis.json` 保存了每个 transition 的计数（6 组 × 2,304 = 13,824）和总 gate，但没有另存一个 13,824 行的扁平 audit 文件。逐条重建所需的 81×256 原始记录仍全部存在。

### 3.4 7–8B breadth 与限制项 7B

| 层级 | 精确位置 |
|---|---|
| 冻结配置/全部权重 SHA | `configs/babilong_around7b_16k_extension_confirmatory.json` |
| 运行前冻结/审计 | `experiments/freeze_babilong_around7b_extension.py`、`experiments/audit_around7b_checkpoint.py` |
| 五模型逐行输出 | `artifacts/around7b_formal/{qwen2p5-7b,qwen3-8b,mistral-7b,falcon3-7b,granite-8b}/` |
| Qwen fixed3/raw4 对照 | `artifacts/around7b_formal/qwen2p5-7b/{canonical_fixed_slots_3,raw_slots_4}/` |
| 分析代码/结果 | `experiments/analyze_babilong_around7b_extension.py`、`artifacts/around7b_formal/analysis.json` |
| Qwen2.5-7B 跨节点精确审计 | `artifacts/around7b_crossnode/qwen7b-panel1-exact-audit.json` |
| 第三节点 Qwen3-8B 800 行复跑 | `artifacts/array_third_node_audit/` |
| 第三节点比较代码 | `experiments/analyze_array_third_node_audit.py` |

规模：正式 breadth 94 个文件、约 5.9 MB；第三节点 11 个文件、约 808 KB。

### 3.5 Q-RAG

| 层级 | 精确位置 |
|---|---|
| 冻结 manifest、checkpoint revision/SHA | `configs/babilong_qrag_direct_peer_frozen.json` |
| 检索运行器 | `experiments/run_babilong_qrag_retrieval.py` |
| frozen Qwen reader 运行器 | `experiments/run_babilong_qrag_reader.py` |
| 检索/reader 逐行输出 | `artifacts/remote_results/babilong_qrag_direct_peer_89a8280/{retrieval,readers}/` |
| 统计分析与汇总 | `experiments/analyze_babilong_qrag_direct_peer.py`、`artifacts/remote_results/babilong_qrag_direct_peer_89a8280/analysis.json` |

规模：24 个文件、约 3.1 MB。

### 3.6 RULER

| 结果族 | 配置 | 原始结果/分析 |
|---|---|---|
| CWE SmolLM2 | `configs/ruler_cwe_smollm2_8k_certified_confirmatory.json` | `artifacts/remote_results/cwe_confirm_c749f24/` |
| CWE Qwen2.5 16K | `configs/ruler_cwe_qwen2p5_16k_certified_confirmatory.json` | `artifacts/remote_results/qwen_cwe_confirm_cd8352f/` |
| NIAH Qwen2.5 16K multiquery | `configs/ruler_niah_qwen2p5_full_context_16k_multiquery_width_linear_confirmatory.json` | `artifacts/remote_results/qwen_niah_fullctx_16k_multiquery_width_linear_confirmation_372xxx.tar.gz` |
| NIAH SmolLM2 4K multiquery | `configs/ruler_niah_smollm2_full_context_4k_multiquery_width075_confirmatory.json` | `artifacts/remote_results/smollm2_4k_multiquery_width075_confirm_376101_376202_376303.tar.gz` |

Writer/metric/runner：`ascent/ruler_aggregation_memory.py`、`ascent/ruler_memory.py`、`ascent/ruler_qa_memory.py`、`ascent/ruler_metrics.py`、`experiments/run_ruler_aggregation_certified.py`、`experiments/run_ruler_niah.py`。

### 3.7 4K/8K、SmolLM2 factorial 与全部对照臂

| 物料 | 精确位置 |
|---|---|
| BABILong 4K 面板/结果 | `data/babilong_1k_4k/`、`artifacts/remote_results/babilong_4k_scale_confirm_c075cec.tar.gz` |
| BABILong 8K 面板/结果 | `data/babilong_1k_8k/`、`artifacts/remote_results/babilong_8k_confirm_440ef7f.tar.gz` |
| QA7/QA8 面板 | `data/babilong_official_qa78_8k/`、`data/babilong_structured_qa78_8k_certified_confirmation/` |
| SmolLM2 generative factorial | `artifacts/remote_results/babilong_8k_generative_factorial_audit_693c414/` |
| SmolLM2 certified factorial | `artifacts/remote_results/babilong_qa78_certified_factorial_ab4222a/` |
| BGE-M3 + byte-matched FIFO | `artifacts/remote_results/babilong_8k_neural_controls_072adbd.tar.gz` |
| same-FLOP static replay | `artifacts/remote_results/static_replay_same_flop_bbe577c.tar.gz` |
| irrelevant/corrupted controls | `artifacts/remote_results/babilong_4k_robustness_d93ff34.tar.gz` |
| certified/latent/fused 逻辑 | `ascent/fusion.py`、`artifacts/remote_results/babilong_qa78_certified_controls_8bb6f1d/` |

### 3.8 系统与 FLOPs

| 物料 | 精确位置 |
|---|---|
| 18 个 fresh-process 原始记录 | `artifacts/remote_results/babilong_4k_systems/{node1,node2}/` |
| 系统统计 | `experiments/analyze_babilong_systems.py`、`artifacts/remote_results/babilong_4k_systems/analysis.json` |
| state 字节统计 | `ascent/state_accounting.py`、`experiments/run_babilong_prompt.py` |
| FLOP profiler | `experiments/profile_babilong_flops.py` |
| FLOP 分析 | `experiments/analyze_babilong_flops_profile.py` |
| profiler 原始档/结果 | `artifacts/remote_results/babilong_flops_profile_e49eae9.tar.gz`、`artifacts/remote_results/flops_e49eae9/analysis.json` |
| latency-tail 复核 | `artifacts/remote_results/latency_tail_c87fa8c/` |

## 4. 论文数字到原始产物的直接映射

下表的“原始产物”是数字的权威来源；`paper/data/evidence/` 是用于作图的统一导出，而非替代原始产物。

| 数字组 | 原始产物 | 生成/复核代码 |
|---|---|---|
| 官方 16K：`.0563/.1613/.1050`、`.1888/.9775/.7888`、`.1613/.9875/.8263`、两组 Holm 对比 | `artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/analysis.json`，逐行为其 `canonical/` | `experiments/analyze_babilong_canonical_coscale_16k_confirmation.py` |
| 语义 holdout 三尺度、两组 Holm 对比、800 行零重叠 | `artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6/analysis.json`，逐行为同目录三尺度子目录 | `experiments/analyze_babilong_canonical_semantic_holdout.py`、`experiments/prepare_babilong_semantic_holdout.py` |
| 3B canonical `.9875`、raw `.8338`、15.38 点差与 Foundation 行级一致 | `artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/analysis.json` 的 `raw_3b_control` 及 `canonical/3b`、`raw/3b` | `experiments/analyze_babilong_canonical_coscale_16k_confirmation.py` |
| Factorial 对角 gain `.388/2.099/7.406`、增量、交互、7B K3→K5 `1.46`、13,824 | `artifacts/sum_numeric_candidate/confirmation_analysis.json`，逐行为 `artifacts/sum_numeric_candidate/confirmation/` | `experiments/analyze_noisy_composition_candidate.py` |
| 19.1× highlight | `7.406131652706986 / 0.3881704357070042`，均来自上一文件 | 表格/claim audit：`experiments/audit_manuscript_consistency.py` |
| 五个 7–8B gain 与区间、7B Foundation `.3325`、四槽 gain `.6675`、3B→7B `-.1588` | `artifacts/around7b_formal/analysis.json`，逐行为 `artifacts/around7b_formal/` | `experiments/analyze_babilong_around7b_extension.py` |
| Q-RAG `-52.5/+22.75/+26.25` | `artifacts/remote_results/babilong_qrag_direct_peer_89a8280/analysis.json`，逐行为 `retrieval/` 与 `readers/` | `experiments/analyze_babilong_qrag_direct_peer.py` |
| RULER CWE SmolLM2 `.2628/.4378/.9561` | `artifacts/remote_results/cwe_confirm_c749f24/analysis.json` | `experiments/analyze_ruler_aggregation_confirmation.py` |
| RULER CWE Qwen `.2417/.6144/.7100` 及跨零区间 | `artifacts/remote_results/qwen_cwe_confirm_cd8352f/analysis.json` | `experiments/analyze_ruler_aggregation_confirmation.py` |
| RULER NIAH Qwen `.0438/.1208/.3188` | `artifacts/remote_results/qwen_niah_fullctx_16k_multiquery_width_linear_confirmation_372xxx.tar.gz` | `experiments/analyze_ruler_niah.py`、`experiments/render_cross_task_table.py` |
| RULER NIAH SmolLM2 `.1042/.2167/.5229` | `artifacts/remote_results/smollm2_4k_multiquery_width075_confirm_376101_376202_376303.tar.gz` | 同上 |
| decode-time 比 `.4130/.4247/.2791`，state bytes `589.0/624.6/640.6`，state/model 比 | `artifacts/remote_results/babilong_4k_systems/analysis.json` 与 `node1/`、`node2/` | `experiments/analyze_babilong_systems.py` |
| FLOP 缩减 `83.7×/79.4×/78.2×`，wall-time `.9008/.8651/.7262` | `artifacts/remote_results/flops_e49eae9/analysis.json` | `experiments/profile_babilong_flops.py`、`experiments/analyze_babilong_flops_profile.py` |

统一绘图 CSV 由 `experiments/export_manuscript_evidence.py` 从上述权威 JSON 导出：

- `paper/data/evidence/primary_panel.csv`
- `paper/data/evidence/factorial_panel.csv`
- `paper/data/evidence/interaction_intervals.csv`
- `paper/data/evidence/around7b_panel.csv`
- `paper/data/evidence/systems.csv`
- `paper/data/evidence/evidence_profile.json`

论文表格由以下代码从分析产物重新生成：

- `experiments/render_paper_qwen_confirmation_table.py`
- `experiments/render_paper_additional_tables.py`
- `experiments/render_cross_task_table.py`
- `experiments/render_theory_evidence_map.py`

完整 CPU 复算命令位于 `supplementary/REPRODUCE.md`。

## 5. 模型版本与重新下载信息

原清单称下列版本“缺失”，实际均已保存：

| 模型 | revision | 位置 |
|---|---|---|
| Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | `configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json` |
| Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 同上 |
| Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 同上 |
| Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | `configs/babilong_around7b_16k_extension_confirmatory.json` |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | 同上 |
| Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 同上 |
| Falcon3-7B | `1e57a0ecd176c7c139f289c60a74e57f887c3dfb` | 同上 |
| Granite-3.3-8B | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | 同上 |
| SmolLM2-135M | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | `configs/babilong_qa78_certified_factorial_smollm2_8k_frozen.json` |
| SmolLM2-360M | `a10cc1512eabd3dde888204e902eca88bddb4951` | 同上 |
| SmolLM2-1.7B | `31b70e2e869a7173562077fd711b654946d38674` | 同上 |
| BGE-M3 / reranker | 见固定 revision 字段 | `configs/babilong_bge_m3_retrieval_protocol.json` |
| Q-RAG QA2 / QA3 | `b51e2b25…` / `d750d583…`，并含权重 SHA | `configs/babilong_qrag_direct_peer_frozen.json` |

统一重新下载记录在：

- `backups/ASCENT_GPU_FINAL_20260823/MODEL_REDOWNLOAD_RECORDS.json`
- `backups/ASCENT_GPU_FINAL_20260823/PUBLIC_MODEL_WEIGHTS_NOT_COPIED.json`

## 6. 环境与跨节点记录

可移植环境摘要：`reproduction/environment.json`、`reproduction/requirements-core.txt`。其中记录 Python 3.12.3、PyTorch 2.8.0+cu128、Transformers 4.57.6、CUDA 12.8、cuDNN 9.10.2、RTX PRO 6000 Blackwell 97,887 MiB，以及节点 1/2 的驱动版本。

完整 conda、pip、CPU、内存、NVIDIA-SMI、OS、git 状态与模型排除记录位于最终灾备内：

```text
backups/ASCENT_GPU_FINAL_20260823/ASCENT_GPU_FINAL_20260823.tar.zst
└── root/autodl-tmp/ASCENT_GPU_FINAL_20260823_META/BACKUP_METADATA/
    ├── CONDA_ENV_EXPORT.txt
    ├── CONDA_EXPLICIT.txt
    ├── PROJECT_VENV_PIP_FREEZE.txt
    ├── CPU.txt
    ├── MEMORY.txt
    ├── NVIDIA_SMI.txt
    ├── OS_RELEASE.txt
    ├── GIT_HEAD.txt
    ├── GIT_STATUS.txt
    └── MODEL_REDOWNLOAD_RECORDS.json
```

跨节点证据：

- Qwen2.5-7B：`artifacts/around7b_crossnode/qwen7b-panel1-exact-audit.json` 明确列出比较字段并确认 80 行预测完全一致。
- Qwen3-8B 第三节点：`artifacts/array_third_node_audit/analysis.json` 确认 10×80 行预测和 scientific fields 全部一致；该文件同时明确标为 post-hoc code audit、不可作 independent confirmation claim。
- factorial 第二节点记录：`artifacts/sum_numeric_candidate/audit_node2/rounds_5_sum_numeric_candidate_panel_1_qwen2p5-7b-instruct.json`；正确匹配的节点 1 记录是 `artifacts/sum_numeric_candidate/development/node1/rounds_5_sum_numeric_candidate_panel_1_qwen2p5-7b-instruct.json`。
- factorial 独立比对脚本/结果：`experiments/compare_factorial_cross_node.py`、`reports/FACTORIAL_CROSS_NODE_EXACT_AUDIT_2026-09-20.json`。256 条 prediction records 中两臂共 512 个 15-way 向量、7,680 个概率值与 768 个逐行 NLL 均在 JSON 解析后 IEEE-754 binary64 相等。

第三节点完整 CPU/RAM/pip 快照仍未找到，因此 R-05 仍为部分满足。R-02/R-06 已在第二遍补成直接、可重复的证据。

## 7. 需要特别注意的缺口与不一致

### 7.1 完全未找到

1. `C-39` 功效/样本量/MDE 脚本。
2. `W-03` `AI_ASSISTANCE.md`。
3. `W-05` 代码许可证。
4. `W-06` 产物/数据许可证。
5. `W-08` Zenodo version DOI。

### 7.2 有原料但缺独立交付物

- `C-08`：writer 在 API 结构上不接收 target，预检也验证 parser；没有统一的“每行阻止 target 字段访问”instrumentation。
- `C-26/D-17`：13,824 的原始行可从 81 个结果文件重建，汇总计数也在 analysis JSON；缺单独的 13,824 行明细表。
- `C-38/W-09`：冻结配置、hash、git commit 和时间线在配置及 `docs/EXPERIMENT_LEDGER.md`；没有独立的预注册声明文件。
- `C-59`：18 个进程结果在，但没有找到单独保存的进程编排脚本。
- `C-60`：能完整重算，但目前是三条 CPU 命令，不是一条统一命令。
- `D-29`：图可以确定性地由 CSV 生成，但没有额外的“每个图点到 CSV 行号”映射表。
- `E-03/R-05`：三个节点的完整硬件/环境快照不齐。
- `E-06`：dtype、greedy、seed、revision 固定；未找到全局 `torch.use_deterministic_algorithms`/TF32 政策文件。

### 7.3 清单要求与真实实现不一致

- `C-15`：当前及冻结备份中的 BABILong、factorial、RULER aggregation、RULER NIAH runners 对无效配置/产物均直接报错并停止，不是清单描述的“验证失败后退回 Foundation”。完整运行记录与失败日志审计见 `reports/C15_FAIL_CLOSED_VALIDATION_AUDIT_2026-09-20.md`。
- `S-05`：完整工作树本地确实包含 BABILong 源数据与面板正文；只有发布包是否重分发可以由构建器控制。因此不能把“仓库完全没有数据本体”标成通过。

## 8. 查找单个数字时应遵循的顺序

1. 先在 `paper/main.tex`、`paper/appendix.tex`、`paper/supplement.tex` 找 claim。
2. 用 `experiments/audit_manuscript_consistency.py` 或 `experiments/audit_paper_claims.py` 找它对应的分析字段。
3. 到本报告第 4 节所列的权威 `analysis.json`。
4. 再由该 JSON 的 provenance/source_files 字段回到逐行输出目录或 tar.gz。
5. 面板 ID、随机种子、hash 到对应 `data/*/MANIFEST.json` 和 `configs/*.json`。
6. 模型 revision/weight SHA 到冻结配置和 `MODEL_REDOWNLOAD_RECORDS.json`。

完整逐项位置仍以 `reports/ASCENT_REPRO_MATERIAL_LOCATIONS_2026-09-20.tsv` 为准；该表恰好包含 128 条物料记录。
