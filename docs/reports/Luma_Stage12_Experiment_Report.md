# Luma Stage1-2 Experiment Report

## 1. 这份报告在回答什么

这份报告解释最近围绕 `c_t`、`Self JEPA`、`rollout loss`、动态退出做过的实验，到底分别在测什么，以及当前可以得出的结论。

一句话总结：

- `c_t` 现在已经是“完整慢环状态流”，不是单纯隔步刷新的占位量
- `Self JEPA` 已经按 `delta` 目标训练，而不是直接预测绝对 `c_t_next`
- `2-step rollout` 已接入，用来约束慢环动力学一致性
- 退出控制已经从“纯 `delta_h` 阈值器”升级到“联合信号 + 改善量目标”
- 训练时软退出、推理时硬退出，是目前更合理的默认策略

---

## 2. 我们到底试了什么

### 实验 A：慢环是否真的存在

目标：

- 验证 `c_t` 不是静态常量
- 验证 `c_t` 注入是否真的会影响主流 `h`

做法：

- 把自省流升级成完整慢环
- 新增跨循环 `meta_state_1 / meta_state_2`
- 每 `slow_k=2` 步更新一次慢环
- 非更新步沿用已有 `c_t`

现在代码位置：

- [IntrospectionStateStream](/home/kt/ai/minimind/model/model_minimind.py:842)
- [LumaBackbone](/home/kt/ai/minimind/model/model_minimind.py:1188)

---

### 实验 B：Self JEPA 是否真在学“变化值”

目标：

- 验证 `Self JEPA` 不是在学 `c_t_next`
- 而是在学 `delta_c = c_t_next - c_t`

做法：

- 把一阶主损失改成：
  - `pred_delta_c` 对齐 `target_delta_c`
- 不再把一阶主损失直接对齐到绝对 `c_t_next`

现在代码位置：

- [SelfJEPAResidualPredictor](/home/kt/ai/minimind/model/model_minimind.py:918)
- [one-step self loss](/home/kt/ai/minimind/model/model_minimind.py:1247)

---

### 实验 C：rollout loss 是干什么的

`rollout loss` 的作用不是“多一个辅助 loss”这么简单。

它主要防的是：

- 模型学成“输出很小的 delta 最安全”
- 或者学成“我不再变化，所以看起来很稳定”

我们这里的 rollout 含义是：

- 第一步：从 `c_t` 预测 `delta_1`，积分到 `c_t+1_pred`
- 第二步：再从 `c_t+1_pred` 预测 `delta_2`，积分到 `c_t+2_pred`
- 然后看这个 2-step 动力学轨迹是否仍然合理

所以 rollout loss 测的是：

- “你预测的变化，在连续展开后还能不能站得住”

不是只看一步好不好看。

---

### 实验 D：为什么要看 `loop_var`

`loop_var` = 不同样本执行的推理循环步数的方差。

如果：

- 所有样本都在同一步退出

那么：

- `loop_var = 0`

这通常意味着：

- 退出机制被硬阈值钉死了
- 或者退出分数虽然在动，但没有强到改变离散退出步数

你真正想要的是：

- 简单样本少想几步
- 难样本多想几步

所以 `loop_var` 是“动态退出是否真的形成深度分布”的直接观察量。

---

## 3. 我们用过哪些数据

### 第一轮

- `tinyshakespeare`
- 作用：快速机制 smoke test
- 问题：样本类型太单一，不足以很好区分“推理深度”

### 第二轮

- `GSM8K` 数学题
- `DailyDialog` 对话
- 混合成“数学 + 对话”短样本
- 作用：更接近 Luma 真实想处理的任务形态

当前混合数据文件：

- [luma_stage12_math_dialogue.json](/home/kt/ai/minimind/artifacts/luma_stage12_math_dialogue.json)

当前实验报告：

- [stage12_report.json](/home/kt/ai/minimind/artifacts/stage12_report.json)

---

## 4. 每个指标到底表示什么

### `mean_kl`

含义：

- 打开 `c_t` 注入和关闭 `c_t` 注入后，输出 logits 的平均 KL 差异

解释：

- 越大，说明 `c_t` 对输出分布影响越明显

不是：

- “越大越好”的绝对指标

它更像“`c_t` 有没有进入主流”的存在性指标。

---

### `mean_hidden_delta`

含义：

- 打开和关闭 `c_t` 注入时，隐状态 `h` 的平均差异

解释：

- 越大，说明 `c_t` 对内部表示的影响越明显

---

### `c_t_var`

含义：

- `c_t` 在不同样本上的方差

解释：

- 如果接近 0，说明 `c_t` 塌缩成几乎固定常量
- 当前非零，说明慢环有在表达差异

---

### `self_loss_head / self_loss_tail`

含义：

- 短程训练开始时和结束时的一阶 `Self JEPA` 损失

解释：

- 如果尾部下降，说明 `delta_c` 预测在变好

---

### `self_rollout_head / self_rollout_tail`

含义：

- 短程训练开始时和结束时的 rollout loss

解释：

- 如果尾部下降，说明 2-step 动力学一致性在变好

---

### `exit_score_var`

含义：

- 退出分数本身是否有波动

解释：

- 如果它完全为 0，说明退出控制器几乎是常数
- 如果非零，说明退出控制器至少在区分样本

---

### `hard_loop_var`

含义：

- 硬退出规则下，循环步数的方差

解释：

- 为 0 说明所有样本都在同一步退出

---

### `soft_loop_var`

含义：

- 采样退出规则下，循环步数的方差

解释：

- 如果软退出非零而硬退出仍为 0，就说明“硬阈值在压平深度分布”

---

### `two_step_improvement_mean`

含义：

- `2-step` 相对 `1-step` 的平均收益改善量

这里当前用的是：

- `self_error - rollout_error`

直观理解：

- 如果这个值明显为正，说明“再走一步有收益”
- 如果接近 0 或为负，说明“再走一步没什么赚头”

这就是为什么它能用来做 `exit target`。

---

## 5. 当前最新结果表示什么

以当前“数学 + 对话”混合实验为准：

- `mean_kl = 1.1445`
- `mean_hidden_delta = 3.375`
- `c_t_var = 0.7205`
- `hard_loop_var = 0.0`
- `soft_loop_var = 0.1389`
- `two_step_improvement_mean = -0.0176`

### 这些结果的直观解释

#### 1. `c_t` 已经不是装饰品

因为：

- `mean_kl` 非零
- `mean_hidden_delta` 非零
- `c_t_var` 非零

这说明：

- `c_t` 已经真的进入主流并改变了内部计算

#### 2. 慢环已经存在

因为：

- `c_t` 没塌缩
- 自省流有跨循环 `meta_state`
- `delta` 预测开始收敛

所以现在的慢环不是“名义上的慢环”，而是“有状态的慢环”。

#### 3. 硬退出确实在压平深度分布

因为：

- `hard_loop_var = 0.0`
- `soft_loop_var = 0.1389`

这说明：

- 模型已经有一点点区分能力
- 但硬退出规则把这种差异压没了

#### 4. 2-step 改善量目前偏负

`two_step_improvement_mean = -0.0176`

表示：

- 在这组短样本上，平均来看多走一步没有额外收益

这不一定是坏事。

它可能意味着：

- 当前样本整体不算太难
- 或者当前 rollout 动力学还不够强

但它至少说明：

- “改善量型 exit target”这条路在数学上已经接上了

---

## 6. 目前最可信的结论

### 已经基本确认的

1. `c_t` 作为并行慢环状态流是可工作的  
2. `Self JEPA` 用 `delta` 目标比直接对齐绝对状态更符合你的设计  
3. `rollout loss` 对防止伪稳定器是必要的  
4. 训练时软退出、推理时硬退出，是目前最合理的默认策略  

### 还没有完全确认的

1. 硬退出如何学出稳定且有区分度的深度分布  
2. rollout 改善量如何和 `LM/world/self` 联合收益做更强绑定  
3. 在更复杂任务上，`two_step improvement` 是否会变成真正有信息量的正负分布  

---

## 7. 接下来最自然的方向

### 方向 A：保持当前默认策略

- 训练：软退出 / 采样退出
- 推理：硬退出

这是当前最稳妥的选择。

### 方向 B：把 exit target 继续升级成联合收益

不是只看：

- `self_error`
- `rollout_error`

而是进一步看：

- `LM 改善`
- `world 改善`
- `self 改善`

然后统一形成“多走一步值不值得”的监督。

### 方向 C：换更明确分层的任务集

比如专门构造：

- 直接回答型
- 单步计算型
- 两步推理型
- 多干扰条件型

这样更适合把循环深度真正拉开。

---

## 8. 最终一句话版本

到目前为止，Luma 的慢环、自省 `delta` 预测、2-step rollout 和动态退出已经不再是概念图，而是可以运行和测量的系统。

真正剩下的核心问题不再是“这些模块有没有用”，而是：

- 如何把这些连续分数
- 变成稳定、可信、可泛化的离散退出深度分布

