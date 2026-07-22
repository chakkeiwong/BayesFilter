# Compact LEDH Score GPU/XLA Artifact

- JSON: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/fixed-sir-t1-n4-seed81120-score.json`
- Status: `completed`
- Row: `zhao_cui_spatial_sir_austria_j9_T20`
- Stage: `score-only`
- Evidence class: `owner_designated_managed_session_visible_gpu_trusted`
- Score correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'not_run_score_only'}`
- Memory: `{'score_memory_budget_pass': True, 'full_row_memory_gate_applicable': False, 'n10000_memory_pass': None, 'peak_mib': 80.04736328125, 'budget_mib': 14000.0, 'source': 'score_gpu_memory_info_after'}`

## Nonclaims

- raw score and FD shards are not score admission
- prefix results are not full-row evidence
- segmented execution is not monolithic batch memory or runtime evidence
- not HMC readiness or posterior correctness evidence
- not a runtime or scientific superiority claim
