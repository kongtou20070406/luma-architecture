### L-1: [labels: continuation-gain, one-step-lookahead] Iteration 2 keep: exit auxiliary loss now regresses one-step continuatio
- **Strategy:** [labels: continuation-gain, one-step-lookahead] Iteration 2 keep: exit auxiliary loss now regresses one-step continuation gain with loop progress, recent gain history, and remaining budget features; mixed self_rollout_tail improved from 0.09375 to 0.041015625 under 512-step validation.
- **Outcome:** keep
- **Insight:** [labels: continuation-gain, one-step-lookahead] Iteration 2 keep: exit auxiliary loss now regresses one-step continuation gain with loop progress, recent gain history, and remaining budget features; mixed self_rollout_tail improved from 0.09375 to 0.041015625 under 512-step validation.
- **Context:** goal=Upgrade Luma exit learning from exit/not-exit heuristics toward continuation gain while preserving emotion/persona/dialogue and keeping params <=0.35B; scope=model/model_minimind.py;,scripts/run_luma_stage12.py;,trainer/train_luma_pretrain.py;,docs/plans/Luma_v0.7.2_Agent_MasterPlan.md;,docs/reports;,docs/reference/Luma_Loss_Reference.md;,docs/agent/AGENT_WORKLOG.md; metric=mixed_self_rollout_tail_512; direction=lower
- **Iteration:** luma-continuation-gain#2
- **Timestamp:** 2026-03-28T11:30:44Z

### L-2: [labels: continuation-gain, pivot-away-two-step, guard-first] [PIVOT] Abandoning direct exit-logit gain gating and all c
- **Strategy:** [labels: continuation-gain, pivot-away-two-step, guard-first] [PIVOT] Abandoning direct exit-logit gain gating and all current two-step value variants. Across iterations 4-7 they either destabilized mixed outright or violated persona/emotion guards. Next strategy family should stay with retained one-step continuation gain and look for a fundamentally different leverage point, such as cross-file training-time scheduling or guard-aware weighting, instead of deeper value targets.
- **Outcome:** pivot
- **Insight:** [labels: continuation-gain, pivot-away-two-step, guard-first] [PIVOT] Abandoning direct exit-logit gain gating and all current two-step value variants. Across iterations 4-7 they either destabilized mixed outright or violated persona/emotion guards. Next strategy family should stay with retained one-step continuation gain and look for a fundamentally different leverage point, such as cross-file training-time scheduling or guard-aware weighting, instead of deeper value targets.
- **Context:** goal=Upgrade Luma exit learning from exit/not-exit heuristics toward continuation gain while preserving emotion/persona/dialogue and keeping params <=0.35B; scope=model/model_minimind.py;,scripts/run_luma_stage12.py;,trainer/train_luma_pretrain.py;,docs/plans/Luma_v0.7.2_Agent_MasterPlan.md;,docs/reports;,docs/reference/Luma_Loss_Reference.md;,docs/agent/AGENT_WORKLOG.md; metric=mixed_self_rollout_tail_512; direction=lower
- **Iteration:** luma-continuation-gain#8
- **Timestamp:** 2026-03-28T11:54:15Z
