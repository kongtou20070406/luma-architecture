# DataMix Source Status

这份表专门回答两个 review 问题：

- 哪些来源现在就值得纳入数据准备主线
- 哪些来源先保留为候选或 hold，不要急着下载进正式混合

## 当前结论

### Enabled

这些来源已经进入当前 `DataMix v1` 主思路，可以优先做小样下载、清洗和统计。

| Source | Bucket | Why keep it | Current stance |
|---|---|---|---|
| `EleutherAI/hendrycks_math` | `smart_math_reasoning` | 更难数学骨架，强化多步推理与中间状态维持 | `enabled` |
| `ricdomolm/MATH-500` | `smart_math_reasoning` | competition-style math，适合拉高 reasoning ceiling | `enabled` |
| `Maxwell-Jia/AIME_2024` | `smart_math_reasoning` | olympiad-like hard math，增强长链条思考 | `enabled` |
| `openai/gsm8k` | `smart_math_reasoning` | easier anchor，避免数学桶全是奥赛风格 | `enabled` |
| `OpenAssistant/oasst1` | `smart_reasoning_dialogue` / `dialogue_quality` | 高质量多轮树状对话，适合作为 reasoning+quality 双用途源 | `enabled` |
| `HuggingFaceH4/ultrafeedback_binarized` | `smart_reasoning_dialogue` / `dialogue_quality` | 回答质量偏好信号，利于对齐表达质量 | `enabled` |
| `facebook/empathetic_dialogues` | `empathy_support` | 共情表达基础源 | `enabled` |
| `thu-coai/esconv` | `empathy_support` | 支持性对话策略，和 Luma 的产品定位高度相关 | `enabled` |
| `local:minimind_python` | `smart_code_python` | 最贴当前工程栈的 Python 代码 | `enabled` |
| `local:parameter_golf_python` | `smart_code_python` | 机制验证与调试风格代码，适合增强 code reasoning | `enabled` |
| `local:wechat_pretrain` | `persona_seed` | 私有人格底色 | `enabled` |
| `local:pretrain` | `persona_seed` | 私有人格底色补充 | `enabled` |

### Candidate

这些来源值得保留，但现在不建议直接并入最终混合，要先做 license / 质量 / 去重审查。

| Source | Bucket | Why candidate |
|---|---|---|
| `bigcode/the-stack` (python subset) | `smart_code_python` | 代码规模大，但必须先做 permissive filtering 和去重 |
| `LooksJuicy/Chinese-Emotional-Intelligence` | `empathy_support` | 对中文情绪表达有潜力，但需要核对质量和 license |
| `Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset` | `empathy_support` | 情绪类别覆盖广，但需要核对质量和使用边界 |
| `wangrui6/Zhihu-KOL` | `dialogue_quality` | 中文长回答风格很有价值，但转载 / license 边界要谨慎 |

### Hold / Planned

这些来源现在先不要进入正式数据主线。

| Source | Bucket | Why hold |
|---|---|---|
| `BelleGroup/multiturn_chat_0.8M` | `dialogue_quality` | 先等 license 和再分发边界确认 |
| `local:pytorch_examples_seed` | `smart_code_python` | 还没整理成统一工件，先列为 planned |
| `local:tool_reasoning_traces` | `smart_reasoning_dialogue` | 以后 agent/tool 使用轨迹成熟后再纳入 |

## 当前下载策略

- 先下载 `enabled` 的**小样**，做：
  - 结构检查
  - 语言分布检查
  - license 记录
  - 粗去重统计
- `candidate` 只保留注册，不默认下载进正式混合
- `hold` 不下载，不进入任何自动 prepare 流程

## 一个重要说明

目前实验脚本里很多公开数据还属于“云端按需抓取”，不是已经全量落地到本地工作区。

也就是说，**现在真实实验使用的是混合状态**：

- `persona_seed` / 本地 Python 代码：本地数据
- `gsm8k / hendrycks_math / dailydialog / esconv / AIME / MATH-500`：通过 Hugging Face datasets-server 在线抓取样本

下一步的数据工作，就是把这种“在线取样验证”逐步变成“本地工作区可复现工件”。
