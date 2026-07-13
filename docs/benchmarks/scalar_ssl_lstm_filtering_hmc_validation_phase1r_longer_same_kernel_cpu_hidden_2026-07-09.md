# Scalar SSL-LSTM Filtering HMC Validation Phase 1R - 2026-07-09

## Decision

- phase1r_acceptance_repair_screen_passed: `True`
- vetoes: `[]`
- passed_seed_count: `3` / `3`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 1R result and refresh/review Phase 2 reference-agreement subplan

## Phase 1R Gate

- acceptance rates: `[0.921875, 0.734375, 0.578125]`
- acceptance range: `0.578125` to `0.921875`
- native divergence statuses: `['not_exposed_by_kernel', 'not_exposed_by_kernel', 'not_exposed_by_kernel']`
- native divergence interpretation: native divergence unavailable for at least one seed; unavailable is not zero divergences
- log-accept threshold used as native divergence: `False`

## Aggregate Summary

- max abs u by seed: `[7.3994829375628735, 7.858316919728933, 10.427120225910855]`
- target log-prob overall range: `-44.16301361531461` to `-37.79979096670376`
- log-accept max abs by seed: `[2.588266759698262, 128.65675528526683, 372.85150587307623]`
- interpretation: descriptive only; no ranking, convergence, posterior correctness, or default-readiness claim

## Seed Rows

| seed index | seed | status | vetoes | acceptance | finite samples | native divergence |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | [20260709, 6101] | passed_short_smoke | none | 0.921875 | 64 | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |
| 1 | [20260709, 6102] | passed_short_smoke | none | 0.734375 | 64 | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |
| 2 | [20260709, 6103] | passed_short_smoke | none | 0.578125 | 64 | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | passed |
| native_divergence | native divergence unavailable for at least one seed; unavailable is not zero divergences |
| zero_divergence_claim | not made |
| statistically_supported_ranking | none; no method comparison and no uncertainty interval |
| descriptive_only_differences | per-seed acceptance, target-log-prob range, log-accept range, sample range, and runtime |
| default_readiness | not assessed |
| gpu_xla_readiness | not assessed; CPU-hidden debug/reference exception |
| hmc_readiness | not assessed; Phase 1R finite/acceptance repair screen only |
| next_evidence_needed | reviewed Phase 2 scalar reference agreement before any posterior agreement interpretation |

## Run Manifest

| field | value |
| --- | --- |
| command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 720 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md` |
| git | `{'commit': 'f297b103303c64019302ed5d9b9aaf2c8f919b64', 'dirty': True, 'dirty_line_count': 25, 'dirty_preview': [' M bayesfilter/inference/hmc_diagnostics.py', ' M bayesfilter/linear/kalman_qr_tf.py', ' M tests/test_common_inference_runtime_contracts.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py', '?? docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json', '?? docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md', '?? docs/plans/bayesfilter-kalman-qr-analytic-vs-autodiff-score-scaling-subplan-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-result-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-subplan-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-result-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-subplan-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-subplan-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase3-gpu-xla-subplan-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase4-expansion-decision-subplan-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase5-closeout-subplan-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-execution-ledger-2026-07-09.md', '?? docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-gated-execution-runbook-2026-07-09.md']}` |
| environment | `{'python': '3.13.13', 'tensorflow': '2.20.0', 'cuda_visible_devices': '-1', 'tf_physical_devices': [{'name': '/physical_device:CPU:0', 'device_type': 'CPU'}], 'tf_logical_gpus': []}` |
| conda_env | `tfgpu` |
| cpu_gpu_status | CPU-hidden debug/reference exception |
| jit_compile | `False` |
| tf32_mode | disabled_by_cpu_hidden_debug_contract |
| native_divergence_telemetry_status | `['not_exposed_by_kernel', 'not_exposed_by_kernel', 'not_exposed_by_kernel']` |
| native_divergence_interpretation | native divergence unavailable for at least one seed; unavailable is not zero divergences |
| random_seeds | `[[20260709, 6101], [20260709, 6102], [20260709, 6103]]` |
| wall_time_seconds | `320.83368024101947` |
| output_artifacts | `['docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json', 'docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md']` |
| plan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md` |
| subplan_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-subplan-2026-07-09.md` |
| result_file | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-result-2026-07-09.md` |

## Nonclaims

- Phase 1R finite/acceptance repair screen only
- not HMC readiness evidence
- not HMC convergence evidence
- not posterior correctness evidence
- not a zero-divergence claim when native divergence is unavailable
- not a tuned-kernel claim
- not sampler superiority evidence
- not statistically supported ranking evidence
- not GPU/XLA production-readiness evidence
- not default-readiness evidence
- not Zhao-Cui source-faithfulness evidence
