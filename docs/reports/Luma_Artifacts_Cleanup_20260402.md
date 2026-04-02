# Luma Artifacts Cleanup Report (2026-04-02)

## 1) Cleanup Scope
- Target directory (canonical): `/home/kt/ai/luma-architecture/minimind/artifacts`
- Duplicate historical directory: `/home/kt/ai/luma-architecture/artifacts`
- Active run directory (must keep): `autoresearch_sigreg8_15m80m_20260402`

## 2) Keep Set (Baseline + Key Candidates)

| Category | Kept artifacts |
|---|---|
| Current active matrix | `autoresearch_sigreg8_15m80m_20260402` |
| Matrix12 key run | `autoresearch_dynamics_matrix12_20260329_234325` |
| Matrix13 key run | `autoresearch_dynamics_rescue13_arcagi_20260330_083823` |
| Baseline smoke anchors | `smoke_15m80m_baseline_128.*`, `smoke_layer2_sigreg_64.*` |
| Report-linked historical records | All files that are directly cited by existing reports under `docs/reports/` |

## 3) Added Summary For Previously Unreported Retest Logs

### 3.1 Candidate
`A2-progress_shape_v1-h3+progress_exit_readout+summary_chunk_film_v2_progress+s_local_floor`

### 3.2 Retest score table (from `retest_summary_slocalfloor_*`)

| Artifact | Guard all_ok | score | rollout_nonzero_max | c_t_var | hard_loop_var |
|---|---:|---:|---:|---:|---:|
| `retest_summary_slocalfloor_smoke_128.summary.json` | ✅ | `0.31025390625` | `0.125` | `0.7512394786` | `1.359375` |
| `retest_summary_slocalfloor_20480_fixv1.summary.json` | ✅ | `0.034906005859375006` | `0.125` | `325867104.0` | `0.0` |
| `retest_summary_slocalfloor_20480_fixv2_fresh.summary.json` | ❌ | `0.0271636962890625` | `0.0` | `0.7512394786` | `1.359375` |
| `retest_summary_slocalfloor_1_test.summary.json` | ❌ | `0.901953125` | `1.0` | `0.7512394786` | `1.359375` |

### 3.3 Bucket tails table

| Artifact | math | python_code | mixed | dialogue | emotion | arc_agi |
|---|---:|---:|---:|---:|---:|---:|
| `smoke_128` | `0.2705078125` | `0.369140625` | `0.328125` | `0.2568359375` | `0.3486328125` | `0.291015625` |
| `20480_fixv1` | `0.02008056640625` | `0.0570068359375` | `0.0203857421875` | `0.0184326171875` | `0.056640625` | `0.0589599609375` |
| `20480_fixv2_fresh` | `0.018798828125` | `0.03955078125` | `0.01458740234375` | `0.011138916015625` | `0.0439453125` | `0.0518798828125` |
| `1_test` | `0.8125` | `1.08984375` | `0.794921875` | `1.064453125` | `0.919921875` | `0.828125` |

## 4) Stale Runtime Runs (not counted as valid results)

| Run directory | Runtime state | Process alive | Valid for keep/kill? |
|---|---|---|---|
| `autoresearch_dynamics_15m80m_20260402` | `stopped_after_stage` | ❌ | No |
| `autoresearch_dynamics_15m80m_layer2_20260402` | runtime json shows `running` | ❌ (stale pid) | No |

## 5) Deletion Policy Applied
- Delete duplicate historical root artifacts tree: `/home/kt/ai/luma-architecture/artifacts`
- Delete stale run directories that are not valid evidence:
  - `autoresearch_dynamics_15m80m_20260402`
  - `autoresearch_dynamics_15m80m_layer2_20260402`
- Delete empty zero-byte launcher/service logs in canonical artifacts root.
- Keep all key run directories and report-linked artifacts listed in section 2.
