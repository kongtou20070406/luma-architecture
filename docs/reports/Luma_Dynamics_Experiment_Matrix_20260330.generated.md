# Luma Dynamics Rescue Matrix (Generated)

## Baseline
- `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1`
- status: `provisional_keep`
- reason: Only line that reached mid in previous nightly run; now treated as rescue anchor, not final keep.

## Screening Ladder
- `cuda_smoke`: `128`, single candidate sanity only, must include a hierarchical path
- `mid_rescreen`: `4096`, promote top `5`
- `long_round1`: `10240`, promote top `2`
- `long_confirm`: `20480`, promote top `1`

## Result Boundary
- 4096 and above are valid keep/kill evidence in this cycle
- Any failed:1 with implementation traceback is marked bug-contaminated, not structural failure
- No stale runtime file can be used as alive-process evidence without service+process confirmation

## Experiments
### P0 `A2-progress_shape_v1-h3+progress_exit_readout`
- family: `formal_pretrain_readiness_anchor`
- priority tier: `anchor`
- design: No extra routing family enabled; validates readiness of the new baseline itself for formal pretrain lock.
- target signal: clean baseline dynamics before add-on controls

### R0 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1`
- family: `memory_tiered_rescue_anchor`
- priority tier: `anchor`
- design: Unmodified v1 anchor, used only as rescue baseline for 4096 comparison.
- target signal: anchor reference

### M1 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_lite`
- family: `memory_tiered_rescue`
- priority tier: `tier1`
- design: Soft tier weighting, world_summary cap, local share floor, no winner-take-most.
- target signal: anti early single-tier collapse

### M2 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_anti_collapse`
- family: `memory_tiered_rescue`
- priority tier: `tier1`
- design: M1 plus explicit entropy/local-share guards with regularized tier selection.
- target signal: tier entropy retention

### M3 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_lite+zone_loss`
- family: `memory_tiered_rescue`
- priority tier: `tier1`
- design: M1 + rollout vitality zone loss on nonzero/active/future-var ranges.
- target signal: rollout activity band preservation

### M4 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_anti_collapse+entropy_guard`
- family: `memory_tiered_rescue`
- priority tier: `tier1`
- design: M2 + stronger entropy floor and minimum local share penalties.
- target signal: dominant-tier suppression

### M5 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_lite+vitality_loss`
- family: `memory_tiered_rescue`
- priority tier: `tier2`
- design: M1 + trajectory vitality loss from c_t/world drift floors.
- target signal: trajectory anti-freeze

### M6 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_lite+local_floor`
- family: `memory_tiered_rescue`
- priority tier: `tier2`
- design: M1 + stronger local-floor routing so unmodulated local path cannot be eaten.
- target signal: local baseline preservation

### M7 `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1+m1_anti_collapse+anti_budget`
- family: `memory_tiered_rescue`
- priority tier: `tier2`
- design: M2 + anti-collapse budget floor (minimum active controlled mass).
- target signal: non-zero control budget floor

### S1 `A2-progress_shape_v1-h3+progress_exit_readout+summary_chunk_film_v2_progress+s_lite_control`
- family: `summary_rescue`
- priority tier: `tier1`
- design: Lower gamma/beta gain, residual-branch-first modulation, avoid direct dominance.
- target signal: summary over-control mitigation

### S2 `A2-progress_shape_v1-h3+progress_exit_readout+summary_chunk_film_v2_progress+s_local_floor`
- family: `summary_rescue`
- priority tier: `tier1`
- design: Summary control with stronger unmodulated local baseline floor.
- target signal: summary-local balance

### S3 `A2-progress_shape_v1-h3+progress_exit_readout+hici_construct_integrate_broadcast_v1+s_lite_control`
- family: `summary_rescue`
- priority tier: `tier2`
- design: HiCI route with lite-control modulation and capped gain.
- target signal: HiCI stability under weak control

### S4 `A2-progress_shape_v1-h3+progress_exit_readout+hici_construct_integrate_broadcast_v1+s_local_floor`
- family: `summary_rescue`
- priority tier: `tier2`
- design: HiCI route with local-floor preservation on non-selected chunks.
- target signal: HiCI anti-collapse local dynamics

## Required Metrics
### A_basic
- math/dialogue/emotion/persona_seed/python_code/arc_agi/mixed
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
- rollout_zone_loss_tail

### C_state_dynamics
- c_t_var
- c_t_delta_norm_mean/std
- pred_delta_c_cos_adjacent
- world_summary_drift_mean/std
- trajectory_vitality_loss_tail

### D_control_stats
- gate_mean/std/saturation_ratio/nonfinite_gate_count
- selected_chunk_ratio
- modulated_chunk_ratio
- chunk_gamma_mean/std
- chunk_beta_mean/std
- modulation_gain_low/mid/high_ratio
- unmodulated_path_energy
- tier_weight_local/loop_history/block_repr/world_summary
- tier_entropy
- dominant_tier_ratio
- tier_switch_rate
- routing_entropy_loss_tail

### E_numeric_safety
- exit_score_preclamp_nonfinite_count
- exit_score_postfix_clamped_ratio
- bernoulli_invalid_prevented_count
- nan_to_num_trigger_count
