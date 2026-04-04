# Luma Stage-0（不含数据处理）实施说明

## 已落地内容
- 模块映射：`luma_stage0/module_mapping.yaml`
- 配置骨架：`luma_stage0/config_schema.py`
- 日志/指标接口：`luma_stage0/metrics_schema.py`
- 阶段0配置校验：`scripts/luma_stage0_validate.py`
- 阶段0短程可复现 harness：`scripts/luma_stage0_harness.py`

## 运行方式
```bash
cd /home/kt/ai/minimind
source /home/kt/ai/.venvs/luma-global/bin/activate
python scripts/luma_stage0_validate.py
python scripts/luma_stage0_harness.py --device cuda --seq-len 128 --batch-size 1
```

## 输出工件
- 配置快照：`artifacts/stage0_config_snapshot.json`
- 指标日志：`artifacts/stage0_metrics.jsonl`

## 说明
- 当前阶段仅完成架构冻结相关脚手架，不包含数据清洗与数据混配。
- Mamba3 MIMO 路径在本机已完成 kernel 编译验证；训练反向仍需根据 tilelang 共享内存限制单独调参。

