# Luma Dynamics Experiment Matrix (Generated)

## Baseline
- `A2-progress_shape_v1-h3+progress_exit_readout`
- status: `current_keep`
- reason: Current long-run dynamics anchor after 2048/4096/10240 filtering.

## Screening Ladder
- `cuda_smoke`: `128`, single candidate sanity only, before batch launch
- `short_prescreen`: `2048`, promote top `6`
- `mid_rescreen`: `4096`, promote top `5`
- `long_round1`: `10240`, promote top `2`
- `long_confirm`: `20480`, promote top `1`

## Result Boundary
- 2048/4096/10240 are valid for keep/kill in this cycle
- historical 20480 stale runtime does not count as evidence
- bug-contaminated mid_rerun chain cannot be used as structural failure proof

## Experiments
### E1 `A2-progress_shape_v1-h3+progress_exit_readout+summary_chunk_film_v1_core`
- family: `summary_conditioned_chunk_film`
- priority tier: `tier1`
- design: chunk_size=32 summary-first weak FiLM/bias on ReasonMamba input; no direct token routing; no world_summary; no progress-state.
- target signal: pure summary-first baseline

### E2 `A2-progress_shape_v1-h3+progress_exit_readout+summary_chunk_film_v2_progress`
- family: `summary_conditioned_chunk_film`
- priority tier: `tier1`
- design: E1 plus progress_next/trend/plateau in chunk-control head; still Mamba-input-only weak FiLM/bias.
- target signal: progress-shape downlink to local control

### E3 `A2-progress_shape_v1-h3+progress_exit_readout+hici_construct_integrate_broadcast_v1`
- family: `hici_summary_hierarchy`
- priority tier: `tier1`
- design: local chunk construct -> global integrate -> chunk broadcast with [c_t, local, global, recent_block_repr].
- target signal: HiCI-style local-global-local control stability

### E4 `A2-progress_shape_v1-h3+progress_exit_readout+budgeted_summary_routing_v1`
- family: `budgeted_summary_routing`
- priority tier: `tier1`
- design: E1 plus budget head f(c_t, loop_idx, hard_loop_var_proxy) limiting strong-controlled chunk count each loop.
- target signal: control budget for anti-saturation

### E5 `A2-progress_shape_v1-h3+progress_exit_readout+budgeted_summary_routing_v2_progress`
- family: `budgeted_summary_routing`
- priority tier: `tier2`
- design: E4 with progress-aware budget f(c_t, progress_next, trend, plateau, loop_idx).
- target signal: progress-aware compute allocation

### E6 `A2-progress_shape_v1-h3+progress_exit_readout+hier_block_token_v1_block_only`
- family: `hierarchical_block_token_ct_routing`
- priority tier: `anchor`
- design: block-only gate with top-k blocks, weak residual scaling/bias in selected blocks; token gate disabled.
- target signal: hierarchical lower-bound anchor

### E7 `A2-progress_shape_v1-h3+progress_exit_readout+hier_block_token_v2_attn_bias`
- family: `hierarchical_block_token_ct_routing`
- priority tier: `tier1`
- design: block-select then token score inside selected blocks, token score only adjusts attention-side bias.
- target signal: attention-side sparse local focus

### E8 `A2-progress_shape_v1-h3+progress_exit_readout+hier_block_token_v3_residual_delta`
- family: `hierarchical_block_token_ct_routing`
- priority tier: `tier2`
- design: E7 upgraded to residual delta injection h_local = h_local + gate * delta_local within selected blocks.
- target signal: hierarchical upper-bound probe

### E9 `A2-progress_shape_v1-h3+progress_exit_readout+double_p_coarse_to_fine_v1`
- family: `double_p_coarse_to_fine`
- priority tier: `tier2`
- design: coarse top-p/capped top-k block screening then fine top-p/capped top-k token screening.
- target signal: adaptive sparsity by sample difficulty

### E10 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1`
- family: `memory_tiered_routing`
- priority tier: `tier1`
- design: tier-first routing over local chunk / loop history / block_repr / world_summary, then light local control.
- target signal: memory-source routing before token routing

### E11 `A2-progress_shape_v1-h3+progress_exit_readout+progress_focus_v1_chunk_query`
- family: `query_triggered_focus`
- priority tier: `tier2`
- design: focus query from c_t + progress-state selects chunks; selected chunks get stronger summary FiLM.
- target signal: lightweight query-triggered sparse focus

### E12 `A2-progress_shape_v1-h3+progress_exit_readout+progress_focus_v3_dense_sparse_hybrid`
- family: `query_triggered_focus`
- priority tier: `tier2`
- design: weak dense global FiLM fallback + strong sparse residual boost on query-selected chunks.
- target signal: dense+ sparse hybrid focus stability

## Required Metrics
### A_basic
- math/dialogue/emotion/persona_seed/python_code/mixed
- self_tail
- rollout_tail
- self_loss_tail
- mean_loss
- hard_loop_var
- avg_loop_depth
- exit_invalid_count

### B_rollout_progress
- rollout_nonzero_ratio
- rollout_active_ratio
- progress_next_mean/std
- progress_trend_mean/std
- progress_plateau_mean/std
- progress_vs_rollout_corr
- progress_vs_exit_corr

### C_state_dynamics
- c_t_var
- c_t_delta_norm_mean/std
- c_t_short_window_curvature
- pred_delta_c_cos_adjacent
- world_summary_drift_mean/std
- loop_to_loop_hidden_delta_mean/std

### D_control_stats
- gate_mean/std/saturation_ratio/nonfinite_gate_count
- selected_chunk_ratio
- chunk_gamma_mean/std
- chunk_beta_mean/std
- budget_value_mean/std
- budget_utilization_ratio
- selected_block_count
- selected_block_entropy
- selected_token_ratio
- topk_or_topp_mass
- delta_local_norm_mean/std
- tier_weight_local/loop_history/block_repr/world_summary
- tier_switch_rate
- focus_score_entropy
- focus_span_count
- focus_span_avg_width
- dense_path_gain
- sparse_path_gain

### E_numeric_safety
- exit_score_preclamp_nonfinite_count
- exit_score_postfix_clamped_ratio
- bernoulli_invalid_prevented_count
- nan_to_num_trigger_count
