# Luma Loss Reference

## 1. 这份文档是干什么的

这份文档专门解释 Luma 当前实验体系中的各种 loss。

它回答四类问题：

- 每个 loss 在优化什么
- 为什么它存在
- 它可能带来什么副作用
- 在我们当前代码里，它是怎么落地的

这份文档偏工程解释，不是论文写作。

---

## 2. 总损失结构

当前 `LumaForCausalLM` 里的训练损失可以概括成：

```text
L_total
= L_lm
+ L_world
+ L_self
+ w_rollout * L_rollout
+ w_exit * L_exit_aux
```

其中：

- `L_lm`
  - 主语言建模损失
- `L_world`
  - world JEPA 损失
- `L_self`
  - self JEPA 一阶残差损失 + residual regularization
- `L_rollout`
  - self JEPA 的多步动力学 rollout 损失
- `L_exit_aux`
  - exit controller 的辅助监督损失

---

## 3. `L_lm`：语言建模损失

### 作用

这是主损失。

它保证模型最终仍然是一个能预测下一个 token 的语言模型，而不是只会做潜空间自监督。

### 当前形式

- 标准自回归 cross-entropy
- 用 `labels[..., 1:]` 对齐 `logits[..., :-1, :]`

### 它优化的是什么

- 语言流畅性
- token 级预测正确性
- 最终可生成性

### 风险

如果只有 `L_lm`：

- `c_t` 很容易沦为装饰件
- `world JEPA` 不会真正学成世界态建模
- 退出控制会变成启发式规则，而不是被训练的行为

---

## 4. `L_world`：world JEPA 损失

### 作用

它要求主流隐状态不仅服务于 token 预测，还要在 latent 空间里保留“世界结构”。

简单说：

- 不是只会接词
- 还要会补回被 mask 掉的 latent world information

### 当前有两种实现

#### 4.1 `scaffold`

当前丐版 world JEPA：

- observer -> online latent
- EMA target
- masked latent prediction
- predictor 比较轻

它更适合：

- 阶段0/1/2 快速验证
- 低成本测试

#### 4.2 `LeWorldModel-style full`

更完整 world JEPA：

- 连续 span mask
- 更强 context predictor
- latent variance regularization
- 更像真正的 latent world modeling 分支

它更适合：

- 中程验证
- 更长 rollout
- 更难数据集

### 当前损失本质

#### `scaffold`

主项是 masked latent cosine loss：

```text
L_world_scaffold
= 1 - cos(pred_masked_world, target_masked_world)
```

#### `full`

主项仍然是 masked latent 对齐，但额外加入：

- `delta_loss`
  - 预测 masked latent 相对 visible summary 的变化
- `sigreg_loss`
  - 保持 latent 方差不要塌缩

大致可写成：

```text
L_world_full
= L_cosine
+ a * L_delta
+ b * L_sigreg
```

### 风险

如果 `L_world` 太弱：

- world 分支会退化成“后挂配件”
- 对退出与长程推理帮助有限

如果 `L_world` 太强：

- 会拖累 LM 主损失
- 训练更慢
- 容易让模型偏向 latent matching 而不是最终生成

### 当前实验判断

- `world JEPA` 应该保留
- `scaffold + self_check` 目前综合更稳
- `full` 在更长 rollout / harder math 上更有潜力

---

## 5. `L_self`：一阶 Self JEPA 残差损失

### 作用

这是自省流的核心训练压力。

它不是让模型预测绝对的 `c_{t+1}`，而是预测：

```text
Δc_t = c_{t+1} - c_t
```

### 为什么要预测 `delta`

因为如果直接预测 `c_{t+1}`，很容易出现惰性解：

- 直接复制当前状态
- 输出低变化
- 看起来稳定，其实没有学会动力学

预测 `delta` 的好处：

- 更接近真实状态变化
- 尺度更统一
- 更适合 rollout 展开
- 更容易发现“别动就是最安全”的伪稳定器

### 当前形式

预测头输入：

- `c_t`
- `delta_h`

输出：

- `pred_delta_c`

目标：

- `target_delta_c = c_{t+1} - c_t`

损失：

```text
L_self_main
= 1 - cos(pred_delta_c, target_delta_c)
```

### residual regularization

当前 `L_self` 里还包含一个很轻的正则项：

```text
L_residual_reg
= ||pred_delta_c||
```

它的作用是：

- 早期防止 `delta` 爆得太大

它的风险是：

- 如果太强，会把自省流训成“少动最安全”

所以这个项必须轻，不能喧宾夺主。

### 当前代码里的实际组合

当前 `aux["self_jepa_loss"]` 实际上是：

```text
L_self
= L_self_main
+ λ * L_residual_reg
```

---

## 6. `L_rollout`：多步自省动力学损失

### 作用

这是为了防伪稳定器。

如果只有一步 `delta` 预测，模型可能学会：

- 下一步尽量少变
- 这样一步损失不一定很差

但一旦要求它继续展开：

```text
c_t -> c_t+1 -> c_t+2 -> ...
```

单纯“少动”就不一定还能成立。

### 直觉

`L_self` 问的是：

- “下一步会怎么变？”

`L_rollout` 问的是：

- “你说的这个变化，继续往前推几步后还站得住吗？”

### 当前形式

当前实现里：

- 用 `SelfJEPAResidualPredictor.rollout()` 先预测多步 `delta`
- 再积分得到未来 `c` 状态
- 等真实未来慢环状态成熟后对齐

损失大致可写成：

```text
L_rollout
= 1 - cos(pred_c_t+k, true_c_t+k)
```

### 为什么这个损失很重要

它直接对应你最关心的问题之一：

- 模型是在“真正有动力学地推理”
- 还是“学会提前停住”

### 什么时候 `rollout` 下降是真变好，什么时候只是被压扁

不能只看 `self_rollout_tail` 越低越好。

更健康的下降通常同时满足：

- `L_self` 也在合理下降
- `c_t` 仍然保持非塌缩方差
- `hard_loop_var` 不是直接被压成死板常数
- `rollout_active_ratio` 仍然明显大于 `0`

其中：

- `rollout_active_ratio`
  - 有多少 step/样本真正形成了 rollout 监督
- `rollout_nonzero_ratio`
  - 形成的 rollout loss 里，有多少不是精确 `0`

如果出现：

- `self_rollout_tail` 接近 `0`
- 但 `rollout_active_ratio` 也接近 `0`

那更像是：

- 没有形成足够多的有效 rollout supervision
- 而不是模型真的把多步动力学全部学会了

### 风险

rollout 太短：

- 容易抓不住动力学

rollout 太长：

- 误差累积
- 梯度噪声变大
- 小模型短程训练时容易看起来更差

### 当前实验结论

我们已经看到：

- `4-step` 比 `2-step` 更好
- `5-step` 继续有收益
- `10-step` 在当前长 horizon 验证里仍有价值

### 如何判断 rollout 下降是“真的变好”还是“被压扁”

`rollout_tail` 变低，不能单独解读。

健康的 rollout 下降通常会同时伴随：

- `self_tail` 也合理下降
- `c_t` 仍然有非零漂移
- `intermediate_state_variance` 保持非零
- `world_summary_drift` 不会一起塌成静止
- 各个 bucket 不会同时被压成同一种极端数值

坏的 rollout 下降更像：

- mixed 和各桶一起掉到接近 `0`
- 伴随某个 gating / uncertainty / auxiliary 信号饱和
- bucket 间差异突然消失

这种情况通常不是“模型真的全学会了”，而是：

- rollout supervision 被压扁了
- 指标失去了区分能力

所以：

- 局部变低，可能是好事
- 全桶同时贴近 `0`，要高度怀疑是不是坏信号

---

## 7. `L_exit_aux`：退出控制辅助损失

### 作用

这个损失不直接优化语言建模，而是训练：

- 什么时候应该继续推理
- 什么时候应该退出

### 当前设计思路

退出不是只看一个 `confidence` 头。

当前 exit controller 会看：

- `delta_h`
- `self_error`
- `rollout_error`
- `world_error`
- `self_check_score`

然后形成一个 `exit_score`。

### 当前辅助监督的直觉

它现在试图学习：

- “继续一轮之后，`LM + self + world` 的联合目标到底还有没有真实收益？”

当前更准确的监督方式是：

1. 对每个 loop 状态，计算一个 loop-level `LM proxy`
2. 与该 loop 的：
   - `self_error`
   - `world_error`
   组合成联合分数
3. 比较当前 loop 和下一 loop 的联合分数变化

也就是：

- 如果下一轮联合分数明显更低，说明继续有收益，不该退出
- 如果下一轮收益已经很小，才监督为“可以退出”

### 为什么它只是辅助损失

因为退出本质上是离散决策，而且当前实现里它仍然用 BCE 去拟合一个离散目标：

- 很容易被硬阈值压平
- 很容易受采样/阈值影响

所以当前阶段更适合把它当成：

- 辅助训练信号
- 而不是绝对主目标

### 风险

如果 `L_exit_aux` 太强：

- 模型会为了“学会退出”而牺牲主任务

如果太弱：

- `hard_loop_var` 长期拉不开

---

## 8. `self_check` 不是单独 loss，但它影响退出

当前极简慢环自检流没有单独的显式损失。

它通过两种方式发挥作用：

1. 它产出 `self_check_score`
2. `self_check_score` 会进入 exit controller

所以它更像：

- 一个便宜的内部状态旁路
- 不直接作为 loss 项
- 但间接影响 `L_exit_aux`

这也是为什么它能明显拉高 `hard_loop_var`。

---

## 9. 各 loss 的分工总结

### `L_lm`

负责：

- 最终语言能力
- token 级生成目标

### `L_world`

负责：

- 世界态 latent 建模
- 让主流对外界结构敏感

### `L_self`

负责：

- 自省流的一阶变化建模
- 让 `c_t` 真正成为动态状态

### `L_rollout`

负责：

- 多步动力学一致性
- 防止伪稳定器

### `L_exit_aux`

负责：

- 训练“何时继续 / 何时停止”
- 帮退出控制不是纯硬规则
- 当前新版本更具体地说：
  - 它在学习“继续一轮后的联合收益是否还值得”
  - 当前默认口径已经不是“纯一步”，而是：
    - `one-step continuation gain` 作为主监督
    - `light two-step continuation auxiliary` 作为轻量辅助项

### 当前默认的 `exit policy` 解释

在最新 `512-step` A/B 里：

- `one-step only`
  - mixed `self_rollout_tail = 0.068359375`
- `one-step + light two-step auxiliary`
  - mixed `self_rollout_tail = 0.052734375`

这说明当前更合理的默认不是：

- 只学一步

而是：

- 一步做主
- 两步做轻量辅助

也就是说，当前默认 `L_exit_aux` 更像：

```text
L_exit_aux
= L_gain_1step
+ w_2 * L_gain_2step_aux
```

其中：

- `L_gain_1step`
  - 负责稳定、直接的 continuation gain 主监督
- `L_gain_2step_aux`
  - 不直接接管退出决策
  - 只作为“更远一步是否还有收益”的轻量辅助信号

---

## 10. 目前最值得盯的 loss 组合关系

当前最关键的不是单看某一个 loss，而是看这几组关系：

### 关系 A：`L_self` vs `L_rollout`

- 如果 `L_self` 好看但 `L_rollout` 很差
- 说明模型只会做一步假预测

### 关系 B：`L_world` vs `L_lm`

- 如果 `L_world` 下降但 `L_lm` 明显恶化
- 说明 world 分支过重

### 关系 C：`L_exit_aux` vs `hard_loop_var`

- 如果 exit loss 看起来正常
- 但 `hard_loop_var` 长期接近 0
- 说明 exit policy 仍然没真正学会“分配推理深度”

---

## 11. 当前工程结论

到目前为止，我们可以比较诚实地说：

- `L_self` 是活的
- `L_rollout` 是有价值的
- `L_world` 应该保留
- `L_exit_aux` 仍然是当前瓶颈之一

也就是说：

- 现在最弱的不是 world/self 这两条 JEPA
- 而是“怎么让更长推理 loops 真正被用起来”

这也是为什么下一阶段重点应该继续放在：

- 更难数据
- 更长 horizon
- exit policy

而不是一味继续堆更多 loss。
