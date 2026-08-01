# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/complete-highdim-leaderboard/phase2-ledh-repair1/lgssm/lgssm-t50-n10000-seed81120-score.json`
- Status: `completed`
- Row: `benchmark_lgssm_exact_oracle_m3_T50`
- Stage: `score-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'not_run_score_only'}`
- Memory: `{'score_memory_budget_pass': True, 'full_row_memory_gate_applicable': True, 'n10000_memory_pass': True, 'peak_mib': 41.75341796875, 'budget_mib': 14000.0, 'source': 'score_gpu_memory_info_after'}`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
