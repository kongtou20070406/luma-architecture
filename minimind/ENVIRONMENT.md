# Luma 训练环境备忘

## 如何运行训练脚本

**必须从 `trainer/` 目录运行**（因为 `--tokenizer_path` 默认是相对路径 `../model/qwen3_5_tokenizer`）：

```bash
cd /home/kt/ai/luma-architecture/minimind/trainer
/home/kt/ai/.venvs/luma-global/bin/python train_luma_refactor.py --phase 4
```

Python 解释器路径：`/home/kt/ai/.venvs/luma-global/bin/python`

## 常见 Phase 命令

```bash
# 后台训练（推荐）
nohup /home/kt/ai/.venvs/luma-global/bin/python train_luma_refactor.py --phase 4 > /tmp/phase4_train.log 2>&1 &

# Smoke test（只跑 5 步验证不崩溃）
/home/kt/ai/.venvs/luma-global/bin/python train_luma_refactor.py --phase 4 --iters 5 --batch_size 4
```

## Triton autotune crash 问题

**现象**：清空 Triton 缓存（`~/.triton/cache/`）后，mamba3 的 Triton kernel 在 autotune 时因 `K < 16` 的 `AssertionError` crash。

**根因**：`mamba_chunk_size` 默认值必须 ≥ 16（Triton `tl.dot` 最小要求）。  
之前默认是 4，训练能跑是因为 Triton 有 autotune 缓存。缓存一旦清除就必定 crash。

**修复**：`train_luma_refactor.py` 里 `--mamba_chunk_size` 默认值已改为 16。

另外在 Triton autotuner 打了补丁（`triton/runtime/autotuner.py`），让 `_bench` 也 catch `CompilationError` 和 `AssertionError`，防止坏配置导致整个 autotune crash。

**不要随意清 Triton 缓存**。如必须清除，第一次运行可能稍慢（autotune 重新计时）。

## Artifacts 路径

- Phase 3 metrics：`artifacts/refactor/phase3_metrics.jsonl`
- Phase 3.5 metrics：`artifacts/refactor/phase35_metrics.jsonl`  
- Phase 4 metrics：`artifacts/refactor/phase4_metrics.jsonl`
- 动力学报告：`artifacts/refactor/phase{N}_dynamics.json`

## 训练结果参考

| Phase | final loss_lm | DOD rank | mode1_energy | DMD radius | verdict |
|-------|--------------|----------|--------------|------------|---------|
| 3     | 1.24         | 3        | 65.1%        | 0.43       | FAIL (target≥4) |
| 3.5   | 1.31         | 3        | 65.8%        | 0.23       | FAIL (target≥4) |
| 4     | 运行中       | -        | -            | -          | - |

> Phase 3/3.5 DOD rank=3 而非目标 4，原因：只有 3 维梯度向量（compress/shared/reasoning），理论上最大 rank 就是 3。目标应调整为 ≥ 3。
