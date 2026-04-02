# Luma Layer-2 Dynamics Analysis

## POD
- effective_rank: 2
- total_energy: 40.839879

| mode | eigenvalue | energy_ratio | cumulative_energy |
|---:|---:|---:|---:|
| 1 | 40.784661 | 0.998648 | 0.998648 |
| 2 | 0.046576 | 0.001140 | 0.999788 |
| 3 | 0.008642 | 0.000212 | 1.000000 |

## DMD
- spectral_radius: 0.924963
- stable: True

| mode | real | imag | magnitude | angle |
|---:|---:|---:|---:|---:|
| 1 | 0.212361 | 0.000000 | 0.212361 | 0.000000 |
| 2 | 0.924963 | 0.000000 | 0.924963 | 0.000000 |
| 3 | 0.767634 | 0.000000 | 0.767634 | 0.000000 |

## Forcing-Response (Top |corr|)
| forcing | response | corr | abs_corr |
|---|---|---:|---:|
| progress_trend_mean | rollout_nonzero_ratio | 0.972491 | 0.972491 |
| progress_next_mean | mean_delta_norm | -0.952619 | 0.952619 |
| progress_trend_mean | self_loss_tail | 0.911138 | 0.911138 |
| progress_plateau_mean | rollout_nonzero_ratio | -0.902161 | 0.902161 |
| progress_plateau_mean | mean_delta_norm | 0.893314 | 0.893314 |
| progress_trend_mean | self_rollout_tail | 0.860932 | 0.860932 |
| progress_plateau_mean | self_loss_tail | -0.799848 | 0.799848 |
| progress_trend_mean | mean_delta_norm | -0.797135 | 0.797135 |
| progress_plateau_mean | self_rollout_tail | -0.734066 | 0.734066 |
| world_surprise_mean | rollout_nonzero_ratio | -0.714409 | 0.714409 |

## Metrics Digest
- num_records: 218
- num_failed_checks: 5
