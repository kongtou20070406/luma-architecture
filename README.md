# luma-architecture

公开分享版的 **Luma architecture** 研究仓。

## 主项目位置（唯一）

- 本地主项目目录固定为：`/home/kt/ai/luma-architecture`
- 从现在起以这个仓作为唯一主线，不再并行维护“内外两套同名仓”。

这个仓主要用于和朋友分享当前进度，内容以：

- 架构设计文档
- 实验报告
- 关键代码实现
- 可复现实验脚本

为主。

它**不包含**：

- 私有 `luma_dataset/` 语料
- 训练权重 / checkpoint / `*.pth`
- 本地虚拟环境
- 运行缓存与日志残留

## 基线前缀命名

为了方便区分实验演进，这个仓现在采用“基线前缀 + 变体名”的记录方式。

- `A0`
  - 早期纯 `one-step` continuation skeleton
- `A1`
  - `one-step main + light two-step auxiliary` 升级线
- `A2`
  - 当前正式长程基线
  - 对应旧文档里常写的 `iter2`

命名示例：

- `A2-core`
- `A2-predictor_progress`
- `A2-progress_shape_v1`
- `A2-local_consistency_v2`

这样可以一眼看出：

- 它属于哪一代基线
- 它是在那代基线下加了什么变体

## 当前内容

- `docs/plans/`
  - Luma 总规划与当前冻结决策
- `docs/reports/`
  - 各轮结构实验、JEPA 实验、depth / self-check / r_t 报告
- `docs/reference/`
  - loss 说明、环境说明等参考资料
- `docs/agent/`
  - agent 工作日志与记忆协议
- `minimind/`
  - 当前 Luma 主实现底座与实验脚本
- `parameter-golf/`
  - 小规模机制验证脚本与参考材料的精简版

## 代码现状

当前主线更接近：

- `A2-core` 长程基线
- `full world JEPA`
- `depth2 shared reasoning block`
- `self_check_k = 2`
- `one-step main + light two-step auxiliary`

但这仍然是**研究实现**，不是已经冻结的最终正式预训练版本。

## 不包含的数据与权重

这个公开仓默认不带：

- `luma_dataset/`
- `minimind/checkpoints/`
- `minimind/out/`
- 任何训练权重文件

如果你要在本地完整复现，需要自行准备：

1. 私有或公开训练数据
2. WSL2 + CUDA 环境
3. Python 依赖包（如 `bitsandbytes`、`muon-optimizer`），以及按需编译第三方内核

详细见：
- [Luma_Experiment_Implementation_Checklist.md](docs/reference/Luma_Experiment_Implementation_Checklist.md)

## 推荐阅读顺序

1. [Luma_v0.7.2_Agent_MasterPlan.md](docs/plans/Luma_v0.7.2_Agent_MasterPlan.md)
2. [Luma_Loss_Reference.md](docs/reference/Luma_Loss_Reference.md)
3. `docs/reports/` 下的各轮实验报告
4. `minimind/model/model_minimind.py`

## 说明

- 这个仓以 **WSL2 + RTX 5090** 的实验环境为主。
- 某些依赖路径当前仍偏研究仓风格，环境安装说明里已经写了需要注意的地方。
- 如果你只是想理解设计，不需要把所有实验都复跑一遍；先看 `docs/plans` 和 `docs/reports` 就够了。

## License

Top-level repository license: `Apache-2.0`.

Notes:

- The main `minimind/` implementation and Luma integration are shared under `Apache-2.0`.
- The included `parameter-golf/` slice retains its own `MIT` license in `parameter-golf/LICENSE`.
- Third-party and inherited notices should be kept when redistributing this repository.
