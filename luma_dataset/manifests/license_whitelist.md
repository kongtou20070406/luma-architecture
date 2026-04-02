# DataMix License Whitelist

正式进入一次性预训练混合前，优先允许：

- `Apache-2.0`
- `MIT`
- `BSD-3-Clause`
- `BSD-2-Clause`
- `CC-BY-4.0`
- `CDLA-Permissive-2.0`

需要人工复核后再决定是否进入正式混合：

- `unknown / missing`
- `research-only`
- `OpenRAIL` / 带行为限制的数据条款
- `copyleft`（如 `GPL`）
- 明确要求非商用、禁止再分发、来源不清的数据

当前策略：
- `BelleGroup/multiturn_chat_0.8M` 先列为 `review_required`
- `wangrui6/Zhihu-KOL` 若 license 或转载边界不清，也先列为 `review_required`
- 本地 `persona_seed` 只做私有训练使用，不进入公开仓
