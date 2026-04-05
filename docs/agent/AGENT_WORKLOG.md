# AGENT 工作日志（跨窗口记忆）

> 记录规范见：/home/kt/ai/AGENT_MEMORY_LOG_PROTOCOL.md

## 2026-04-04 FP8 混合精度实现 + 模型扩容实验 (S2 系列)

### FP8 基础设施
- **新增** `model/fp8_linear.py`：FP8 forward (E4M3 tensor core GEMM via `_scaled_mm`)，backward 从 FP8 activation 反量化回 bf16 做 matmul。
- **关键限制**：RTX 5090 cuBLAS `_scaled_mm` 只支持 `A @ B.t()` 模式，backward 改用 bf16。
- **VRAM 优化**：save_for_backward 保存 FP8 activation (1 byte) + scale，不保存 bf16 (2 bytes)。312M 模型从 24GB 降到 ~13GB。
- **`convert_to_fp8(model)`**：自动替换 nn.Linear → FP8Linear，跳过 dim 不对齐 16 或参数量 < 4096 的层。
- **训练集成**：`train_luma_refactor.py` 新增 `--fp8` flag。

### Phase 6 误触发修复
- 发现 `run_experiment_matrix.py` 的 `BASE_ARGS` 被改为 `--phase 6`（World JEPA 开启 + self_rollout 强制为 0），导致 G5 加长训练假塌缩。
- **修复**：改回 `--phase 4`，删除 `build_phase6_config` 防止误触发。

### G5 加长训练 (5000 步, Phase 4)
- mode1 在 45-99% 大幅振荡，甜蜜点 step 1200-2500。DOD rank 始终=3。
- 结论：312M 模型 rank 上限=3，需要扩容突破。

### H Group 实验 (G5 + 第三类数据 5%)
- H1 (scifi): FAIL，mode1=93.8%，语义距离太远。
- **H2 (python): BEST**，mode1=87.8%，与 persona+math 语义兼容。
- H3 (ARC): FAIL，grid 格式与 token 预测不兼容，mode1=88.2% 但 loss 不收敛。
- 结论：312M 能承受 3 类数据，但语义距离比数量更重要。

### 模型扩容评估
- 基于 294M → 24GB (VRAM 41x param) 的观测校准 6 个方案。
- **推荐 S2 (635M)**：h=1024, layers=32, FP8 bs=4 → 29GB，留 3GB 余量。
- 报告：`docs/reports/Model_Scaling_Evaluation_20260404.md`

### S2 扩容实验（进行中）
| 实验 | 配置 | 参数量 | 最终 loss | mode1 轨迹 | 状态 |
|------|------|--------|-----------|-----------|------|
| S2-A | h=1024, L=32, depth=2, loops=12, bs=4 | 660M | 1.41 | [89→96→84→98→73→**50%**] | ✅ 完成 |
| S2-C1 | h=1024, **L=24**, depth=2, loops=12, bs=4 | 533M | (running) | step200: 80.8% | 🔄 运行中 |
| S2-C2 | h=1024, L=24, **depth=3**, loops=12 | ~570M | — | — | 待跑 |
| S2-C3 | h=1024, L=24, depth=2, **c_t=128, meta=192** | ~540M | — | — | 待跑 |
| S2-C4 | h=1024, L=24, **depth=3, c_t=128, meta=192** | ~580M | — | — | 待跑 |

- **S2-A 关键发现**：32 层 compress zone 导致前期 mode1 持续走高 (89→98%)，但 step 1200+ 后大幅恢复到 49.7%。梯度比 compress:reasoning ~13:1。
- **S2-C1 初步**：24 层 compress zone 在 step 200 时 mode1 更低 (80.8% vs 89.5%)，早期动力学更健康。

### VRAM 优化计划
- 写入 `docs/reports/VRAM_Optimization_Plan_20260404.md`
- P0: Optimizer CPU Offload (~2.6GB)
- P1: Reason Loop Checkpoint 检查 (~0.5-1GB)
- P2: Activation Offload 前 N 层 (~1-3GB)

### 其他产出
- `docs/reports/G5_Extended_5000_Report_20260404.md`
- `docs/reports/H_Group_and_G5_Extended_Report_20260404.md`
- `docs/reports/Model_Scaling_Evaluation_20260404.md`
- `docs/reports/VRAM_Optimization_Plan_20260404.md`
- `.gitignore` 更新保护敏感数据
- Git push to main (清理 2.3GB artifacts + persona 数据)

## 2026-04-03 11:50 FP32 严格防炸链路切换（用于后续重建工作流）
- 阶段: dynamics constraints foreground 长链重启
- 背景:
  - 之前链路存在“分数非有限值被容错兜底”的行为，不符合当前“不要容错 NaN”的执行要求。
  - 需要在 CUDA 上用全链路 FP32 降低数值炸裂概率，并把 non-finite 变成硬失败信号。
- 已完成改造:
  - 修改 `/home/kt/ai/luma-architecture/minimind/scripts/run_luma_stage12.py`
    - 新增 `--force-fp32` 开关。
    - CUDA 下可强制 `dtype=torch.float32`（不再默认 BF16）。
    - 报告字段新增 `force_fp32`、`model_dtype`，用于回溯实验精度口径。
  - 修改 `/home/kt/ai/luma-architecture/minimind/scripts/run_dynamics_constraints_iter100_fg.py`
    - 默认透传 `--force-fp32` 到 stage12 评估。
    - 遇到 `score=NaN/Inf` 直接判该次评估失败（不再 finite fallback）。
    - 遇到 `first_nonfinite_step` 非空直接判失败（不再作为软惩罚继续）。
    - `hard_pass` 与 objective 去掉 `score_finite` 容错逻辑，改为严格失败路径。
- 校验:
  - `python3 -m py_compile` 已通过：
    - `minimind/scripts/run_luma_stage12.py`
    - `minimind/scripts/run_dynamics_constraints_iter100_fg.py`
    - `minimind/scripts/run_dynamics_candidate_eval.py`
- 当前活跃链路（可直接接管）:
  - 主进程:
    - `PID=332037`
    - `/home/kt/ai/.venvs/luma-global/bin/python scripts/run_dynamics_constraints_iter100_fg.py --iterations 100 --score-tolerance 0.15 --min-rollout-nonzero 0.03`
  - 子训练进程（当前）:
    - `PID=332071`
    - `run_luma_stage12.py ... --force-fp32 ... --stage2-steps 4096`
  - 当前产物目录:
    - `/home/kt/ai/luma-architecture/minimind/artifacts/luma_dynamics_constraints_fg_20260403_115036`
- 重建工作流（晚点重启直接用）:
  1. 停旧链:
     - `pkill -f run_dynamics_constraints_iter100_fg.py`
     - `pkill -f run_dynamics_candidate_eval.py`
     - `pkill -f run_luma_stage12.py`
  2. 语法检查:
     - `python3 -m py_compile minimind/scripts/run_luma_stage12.py minimind/scripts/run_dynamics_constraints_iter100_fg.py minimind/scripts/run_dynamics_candidate_eval.py`
  3. 重启严格链:
     - `cd /home/kt/ai/luma-architecture/minimind`
     - `/home/kt/ai/.venvs/luma-global/bin/python scripts/run_dynamics_constraints_iter100_fg.py --iterations 100 --score-tolerance 0.15 --min-rollout-nonzero 0.03`
  4. 运行监控:
     - `ps -eo pid,ppid,cmd | rg 'run_dynamics_constraints_iter100_fg.py|run_dynamics_candidate_eval.py|run_luma_stage12.py' | rg -v rg`
     - `ls -lt /home/kt/ai/luma-architecture/minimind/artifacts/luma_dynamics_constraints_fg_* | head`
     - `tail -n 80 /home/kt/ai/luma-architecture/research-results.tsv`
- 判定口径（当前冻结）:
  - 先看“可训练性”：无 non-finite > 再看 rank/rollout > 最后看 score/bucket。
  - 非有限值不再归类为“可容错波动”，一律按失败处理。

## 2026-04-02 15:05 SIGReg 解耦矩阵切换（M0->M1）已落地并启动
- 阶段: dynamics autoresearch 切换执行（单卡，后台长跑）
- 本步目标:
  - 将训练流程切换到 `M0 最小闭环 -> M1 逐项复耦 -> Stage2 Top2*强度扫描`
  - 增加 SIGReg 位点/精度/warmup/诊断，先验证 world 分支可训练性
- 已完成:
  - 修改 `minimind/model/model_minimind.py`：
    - 新增 loss 解耦开关：`world_jepa_weight/self_jepa_weight/self_rollout_weight/exit_aux_weight/disable_self_jepa`
    - 新增 SIGReg 稳定开关：`sigreg_world_source/sigreg_world_fp32_only/sigreg_world_warmup_steps/world_delta_weight`
    - 新增诊断输出：`world_sigreg_loss_step`、`sigreg_source_mean/std`
  - 修改 `minimind/scripts/run_luma_stage12.py`：
    - 新增 CLI 透传与报告字段：`grad_norm_total/grad_norm_world_encoder/first_nonfinite_step/world_sigreg_loss_head/tail/max`
    - 修复 `bucket_probe_from_mixed_model` 缺少 `sigreg_world_steps` 定义导致的运行错误
  - 修改 `minimind/scripts/run_dynamics_candidate_eval.py`：
    - summary/guard 接入 `nonfinite_ok` 与 `sigreg_source_std_ok`
  - 修改 `minimind/scripts/run_dynamics_autoresearch_local.py`：
    - TSV 增列上述新诊断字段
  - 重写 `minimind/luma_stage0/dynamics_autoresearch_program.json`：
    - 新阶段链路：`4096 -> 10240 -> 14649 -> 78125`
    - 新候选集合：M0(A1-A4+encoder位点) + M1递进复耦 + low/med/high 强度
- 验证:
  - `python3 -m py_compile`（4 脚本 + model）通过
  - CUDA smoke 通过：
    - `m0_a1_cosine`
    - `m0_a2_cosine_sigreg_online`
  - smoke 产物包含：
    - 分桶分数（含 `arc_agi`）
    - Layer2（`POD/DMD/forcing-response`）
    - 新诊断字段
- 运行状态:
  - 已停旧链并启动新链：
    - output: `/home/kt/ai/luma-architecture/minimind/artifacts/autoresearch_sigreg_decoupled_m0m1_20260402`
    - 进程: `run_dynamics_autoresearch_local.py` -> `run_dynamics_candidate_eval.py` -> `run_luma_stage12.py`
  - 当前在跑首个候选：`A2-progress_shape_v1-h3+progress_exit_readout+m0_a1_cosine`（4096）
- 注意事项:
  - `research-results.tsv` 仅在候选完成后追加；运行中主要看 `autoresearch-runtime.json` + `metrics/*.jsonl`
  - `autoresearch-state.json` 当前仍显示 `starting`（脚本行为），以进程树和 runtime 为准判活跃状态

## [2026-03-28] Mamba3 工程落地 + Stage0 架构搭建 + 短程验证全线

### 核心架构里程碑
- **Mamba3 论文三要素工程落地**：在 `model/mamba3_module.py` 实现复数相位旋转、MIMO 多分支门控融合、Heun/Trapezoid 两阶段更新。底层保持 `mamba_ssm.Mamba2` 优化算子。MIMO 在 5090 上遇 TileLang 限制，已加 `auto_fallback_on_mimo_error` 兜底。
- **Stage0 脚手架落地**：`luma_stage0/` 配置骨架、指标接口、模块映射、约束校验、短程 harness。
- **Luma 骨架模块落地** (`model_minimind.py`)：`LumaConfig`、压缩区全套（`CompressionMambaLayer`、`CompressionKimiDeltaAttentionLayer`(官方FLA KDA)、`CompressionBlockAttentionResiduals`）、推理区（`ReasonMambaLayer`、`GatedDiffAttnFoXSWA`）、自省流（`IntrospectionStateStream`、`SelfJEPAResidualPredictor`）、世界模型（`WorldLatentJEPA`）、退出（`ExitController`）、`FactorizedLMHead`、`LumaBackbone`、`LumaForCausalLM`。
- **mHC 规格冻结**：多残差流动态混合 (`Sinkhorn-Knopp + Birkhoff`)，`n_streams=4, apply_zone=reason_loop_only`。
- **参数量收敛**：`381.470M` → `266.7M`（FactorizedLMHead），0.3B 档。

### 退出机制与动力学迭代
- **退出控制器**：纯 `delta_h` → 联合可学习 (`delta_h+self_error+world_error`) → 下一轮联合收益 (`LM proxy+self+world`)。
- **关键发现**：硬退出 → `loop_var=0`。软退出/采样退出后首次 `soft_loop_var=0.139`。冻结策略：**训练软退出，推理硬退出**。
- **Self-JEPA rollout**：4-step > 2-step（`self_rollout_tail 0.81 vs 1.02`），但退出分布问题独立。
- **快环/慢环**：`slow_k=1`（快环）综合最强。world JEPA 关闭后退出分布明显变差。
- **长 horizon**：`10x15` 有效（`hard_loop_var=7.0`），`10x20` 无额外收益，瓶颈在 exit policy。
- **数据消融**：从 TinyShakespeare → GSM8K+DailyDialog → 竞赛数学+对话+情绪，`loop_var=0` 非样本同质性导致。

### 预训练 Trainer + 优化器
- **`train_luma_pretrain.py`**：支持 `full+self_check`、参数分组、checkpoint/resume、`8-bit Muon + 8-bit AdamW`。
- **reason_shared_depth**：`depth=1` 适合 mixed 底座，`depth=2` 适合情感专项。

### Autoresearch 结论
- **Iter2**（one-step continuation gain）为最稳 baseline。Iter4-7（logit gating/two-step value/uncertainty-gated）未超越。
- **Iter9 JEPA Bundle**：world mask+SIGReg+surprise+JEPA coupling。Mixed 改善但 math 回退，ExpD(math adapter)为最优修复。
- **One-step + light two-step**：确立新默认（rollout `0.068→0.053`）。
- **Uncertainty head**：三变体均塌缩，当前插入点不可行。
- **产品方向**：转向 `full+self_check`（符合"聪明聊天伙伴"定位）。

## [2026-03-29] 动力学中长程筛选 + DataMix v1

- **E1~E12 新矩阵**接线 + smoke。`A2-progress_shape_v1-h3+progress_exit_readout` 为唯一通过 4096→10240 两轮的动力学增强主候选。
- **2048 dynamics prescreen**：5 候选晋级 → 4096 midcourse → 仅 A2+progress_exit_readout 存活。
- **DataMix v1**：重组 `luma_dataset/`，冻结 `50% smarter-first` 组成。
- **文献回顾**：扩展 `Dynamics_Literature_Midcourse_Plan`，新增 local_rollout_head / progress_exit_readout 等结构候选。
- **纯本地 watchdog runner** (`run_dynamics_autoresearch_local.py`) 不再依赖外部 Codex worker。
- **Nightly matrix12**：12 候选长链 systemd 启动。

## [2026-03-30] Matrix12 收尾 + ARC-AGI 接入

- **Matrix12 结果**：有效停在 2048+1条4096。`hier_block_token` / `double_p` 为实现 bug（shape mismatch），非结构失败。`memory_tiered_routing` 唯一进入 4096 但 guard 失守。
- **Rescue 重构**：修复 `block_score_head` 接口、新增 soft-tier/防塌缩损失/routing entropy 正则。
- **ARC-AGI 接入**：从 AI2 ARC 切换为 Chollet ARC-AGI，`run_luma_stage12.py` 新增 `--enable-arc-agi`。
- **防 bug 规则冻结**：改动后必须 `py_compile` → 单候选 smoke → 批量实验。

## 2026-04-02 Workspace consolidation + handoff hardening
- 路径收敛（关键）：
  - 当前唯一执行主线固定为：`/home/kt/ai/luma-architecture/minimind`
  - `minimind_runtime_dynamics` 已移除，不再作为任何 runner 的有效入口
  - `parameter-golf` 当前仅保留可选参考角色，默认实验链不依赖
- 运行状态确认：
  - `sigreg8` 8组合矩阵链仍在运行（`run_dynamics_autoresearch_local.py` + `run_luma_stage12.py`）
  - 活跃输出目录：`/home/kt/ai/luma-architecture/minimind/artifacts/autoresearch_sigreg8_15m80m_20260402`

## 2026-04-02 Artifacts cleanup (traceability-first)
- 先补报告再删文件，避免清理后失去可追溯证据。
- 新增清理与补表报告：
  - `/home/kt/ai/luma-architecture/docs/reports/Luma_Artifacts_Cleanup_20260402.md`
  - 其中补齐了 `retest_summary_slocalfloor_*` 的分桶分数、guard、有效性边界。
- 已删除：
  - 旧重复目录：`/home/kt/ai/luma-architecture/artifacts`
  - stale run 目录：`autoresearch_dynamics_15m80m_20260402`、`autoresearch_dynamics_15m80m_layer2_20260402`
  - 0B 无效 service/launcher 日志
  - 无用补测权重：`retest_summary_slocalfloor_20480_fixv1.pt`、`retest_summary_slocalfloor_20480_fixv2_fresh.pt`
- 保留：
  - `autoresearch_sigreg8_15m80m_20260402`
  - `autoresearch_dynamics_matrix12_20260329_234325`
  - `autoresearch_dynamics_rescue13_arcagi_20260330_083823`
  - 以及已被报告直接引用的工件

## 2026-04-02 Reports consolidation (reduce file count)
- `docs/reports` 从大量散报告收敛为少量 canonical 报告 + 生成矩阵工件。
- 新增：
  - `Luma_Stage12_Consolidated_Report_20260402.md`
  - `Luma_Dynamics_Consolidated_Report_20260402.md`
  - `docs/reports/README.md`（报告总入口）
- 删除：
  - 已被并入的旧同类散报告（stage12、rollout、uncertainty、r_t、mid/long 分拆报告等）
- 当前 reports 入口建议：
  1. `docs/reports/README.md`
  2. `Luma_Stage12_Consolidated_Report_20260402.md`
  3. `Luma_Dynamics_Consolidated_Report_20260402.md`
  4. `Luma_Dynamics_Matrix13_ARCAGI_Report_20260330.md`
  5. `Luma_Artifacts_Cleanup_20260402.md`

## 2026-04-02 Reference + Plan update (code-aligned)
- 新增 reference 入口：
  - `/home/kt/ai/luma-architecture/docs/reference/README.md`
- 更新 `README.md`：
  - 推荐阅读顺序改为先看 execution plan / reference index / reports index
- 新增执行计划：
  - `/home/kt/ai/luma-architecture/docs/plans/Luma_Execution_Plan_20260402.md`
- 更新主计划：
  - `/home/kt/ai/luma-architecture/docs/plans/Luma_v0.7.2_Agent_MasterPlan.md`
  - 顶部增加 `2026-04-02` 高优先执行更新
  - 将训练固定项改为当前 trainer 真实口径（不再写旧的“固定2-step + 简化总损失”）

## 2026-04-02 Loss reference rewrite (implementation-true)
- 彻底重写：
  - `/home/kt/ai/luma-architecture/docs/reference/Luma_Loss_Reference.md`
- 对齐到 `model_minimind.py` 当前真实损失组装：
  - `L_total = L_lm + L_world + L_self + w_rollout*L_self_rollout + w_exit*L_exit_aux + L_rollout_zone + L_routing_entropy + L_trajectory_vitality`
- 明确了三条 SIGReg 干预点与注入位置：
  - `world_online` -> `L_world`
  - `rollout_state_preds[:3]` -> `L_self`
  - `pred_delta_c` -> `L_self`

## 2026-04-02 Handoff note for next agent
- 接手第一优先检查：
  1. `ps` 活进程是否仍是 `sigreg8` 主链
  2. `autoresearch-runtime.json` + `watchdog-heartbeat.json` 是否一致
  3. 仅在 `luma-architecture/minimind` 路径下执行 runner/cleanup
- 判断实验有效性时，继续执行当前边界：
  - stale runtime 不算有效结果
  - 实现 bug 污染不直接判结构失败
  - NaN 分数不可直接用于 keep/kill 排序

## 2026-04-02 Agent protocol upgraded with mechanical overrides
- 更新文件：
  - `/home/kt/ai/luma-architecture/docs/agent/AGENT_MEMORY_LOG_PROTOCOL.md`
- 目标：
  - 把 Mechanical Overrides（Step0、分阶段、强制验证、上下文衰减防护、编辑完整性、重命名安全搜索等）固化成可执行协议，降低后续接手风险。
- 关键新增：
  - `STEP 0 Rule`：>300 行结构改造前先清死代码，并独立记录。
  - `PHASED EXECUTION`：每 phase 最多 5 文件，阶段间需验证与确认。
  - `FORCED VERIFICATION`：按项目类型执行强制校验，不得把写文件当成功。
  - `CONTEXT/READ/EDIT` 三类防错：10+轮重读、>500行分块读、编辑前后复读确认。
  - 记录模板新增“有效性边界”字段，强制标注证据是否可用于结论。
- 接手收益：
  - 新 agent 可直接按协议执行，减少“改了但不可交接”的隐性失败。

## 2026-04-02 Compression dynamics constraint probe on R3/R4
- 代码实现（可复现）：
  - `/home/kt/ai/luma-architecture/minimind/model/model_minimind.py`
  - `/home/kt/ai/luma-architecture/minimind/scripts/run_luma_stage12.py`
- 新增配置项：
  - `compression_dynamics_weight`
  - `compression_block_drift_floor`
  - `compression_block_var_floor`
- 新增训练项：
  - `compression_dynamics_loss = w * (relu(drift_floor - compression_block_drift_mean) + relu(var_floor - compression_block_var_mean))`
  - 已并入总损失与 `aux_loss`，并写入 stage2 指标输出：
    - `compression_dynamics_loss_tail`
    - `compression_block_drift_tail`
    - `compression_block_var_tail`
- 冒烟实验目录：
  - `/home/kt/ai/luma-architecture/minimind/artifacts/smoke_sigreg_compression_r3r4_20260402`
- 结果边界（当前版本）：
  - R3/R4 均可稳定跑通，无 `nonfinite_abort`
  - 但当前一侧下界约束常出现“loss=0”（压缩区漂移/方差天然高于 floor），说明约束已接线但在部分配置下未形成有效训练压力。
  - 后续若要让约束持续生效，应改为区间约束（low/high）或按统计自适应 floor。
