# Luma Dynamics Matrix12 Nightly Report (2026-03-30)

## 1) Run Snapshot
- Run directory: `/home/kt/ai/minimind/artifacts/autoresearch_dynamics_matrix12_20260329_234325`
- Program: `/home/kt/ai/minimind/luma_stage0/dynamics_autoresearch_program.json`
- Status: `complete` (`autoresearch-runtime.json`)
- Time window (UTC+8): `2026-03-29 23:50:08` -> `2026-03-30 00:50:08`
- Duration: `3600s`
- Stage coverage:
  - `short_prescreen(2048)`: `12` candidates attempted
  - `mid_rescreen(4096)`: `1` candidate promoted and attempted
  - `long_round1(10240)`: `0`
  - `long_confirm(20480)`: `0`

## 2) Validity Boundary
- Valid structural evidence:
  - `8` short candidates completed with JSON report + summary.
  - `1` candidate (`memory_tiered_routing_v1`) passed short guard and entered `4096` mid rescreen.
- Implementation-bug contaminated (cannot be used as structural failure evidence):
  - `hier_block_token_v1_block_only`
  - `hier_block_token_v2_attn_bias`
  - `hier_block_token_v3_residual_delta`
  - `double_p_coarse_to_fine_v1`
  - Common runtime error in logs:
    - `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x419 and 291x1)`
    - Location trace points to `model_minimind.py` dynamic routing path (`block_score_head` input shape mismatch).
- Stale-runtime boundary:
  - This run itself is complete and consistent; no stale-runtime miscount detected for this nightly chain.

## 3) Candidate Outcomes (12 total)

| Candidate suffix | 2048 status | 2048 score | 2048 guard | 4096 status | 4096 score | 4096 guard | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `summary_chunk_film_v1_core` | ok | 0.07094 | fail | - | - | - | rollout_nonzero guard failed |
| `summary_chunk_film_v2_progress` | ok | 0.04797 | fail | - | - | - | rollout_nonzero guard failed |
| `hici_construct_integrate_broadcast_v1` | ok | 0.05369 | fail | - | - | - | rollout_nonzero guard failed |
| `budgeted_summary_routing_v1` | ok | 0.05701 | fail | - | - | - | rollout_nonzero guard failed |
| `budgeted_summary_routing_v2_progress` | ok | 0.06467 | fail | - | - | - | rollout_nonzero guard failed |
| `hier_block_token_v1_block_only` | failed:1 | - | - | - | - | - | impl bug (shape mismatch) |
| `hier_block_token_v2_attn_bias` | failed:1 | - | - | - | - | - | impl bug (shape mismatch) |
| `hier_block_token_v3_residual_delta` | failed:1 | - | - | - | - | - | impl bug (shape mismatch) |
| `double_p_coarse_to_fine_v1` | failed:1 | - | - | - | - | - | impl bug (shape mismatch) |
| `memory_tiered_routing_v1` | ok | 0.04851 | pass | ok | 0.03911 | fail | promoted to 4096, then rollout_nonzero guard failed |
| `progress_focus_v1_chunk_query` | ok | 0.07037 | fail | - | - | - | rollout_nonzero guard failed |
| `progress_focus_v3_dense_sparse_hybrid` | ok | 0.07814 | fail | - | - | - | rollout_nonzero guard failed |

## 4) Key Dynamics Reading
- `memory_tiered_routing_v1` is the only line that preserved enough short-horizon rollout activity to pass short guard.
- At `4096`, the same line improved score numerically (`0.04851 -> 0.03911`) but failed guard because `rollout_nonzero_ok` collapsed (`rollout_nonzero_max = 0.0`).
- Most summary/query lines show the same pattern: self-tail can look better, but rollout activity collapses early, so they are not acceptable as healthy dynamics lines under current guard.

## 5) Keep / Kill (Current Run Boundary)
- Keep (provisional):
  - `A2-progress_shape_v1-h3+progress_exit_readout+memory_tiered_routing_v1` as the only validly promoted line from this run.
- Kill by current implementation (not by idea):
  - `hier_block_token_*`, `double_p_coarse_to_fine_v1` current code version only, due to reproducible shape bug.
- Not promoted:
  - All other short-complete lines failed short guard on rollout-nonzero criterion.

## 6) Immediate Next Actions
1. Fix `block_score_head` feature-shape contract in `model_minimind.py` and re-run the 4 bugged candidates from `2048`.
2. For summary/query families, debug why rollout_nonzero collapses to `0.0` despite finite training.
3. Re-run the matrix with same progression policy (`2048 -> 4096 -> 10240 -> 20480`) after bugfix, then reassess long-run slots.

