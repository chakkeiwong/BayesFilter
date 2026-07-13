# Phase 2T Result: LGSSM TF32 Correctness Policy Repair

Date: 2026-07-09

Status: `PASSED_POLICY_SMOKE_PHASE2U_FULL_RUN_MAY_START_AFTER_REVIEW`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Adopt a disclosed separate-precision correctness arm for LGSSM TF32 production score runs. | Met for smoke scale: trusted GPU `N=256,T=3` passed same-scalar FD when production TF32 stayed enabled and FD correctness used a disclosed no-TF32 value-only scalar arm. | No target, route, memory, or hidden-precision veto in smoke. | Full `N=10000,T=50` runtime/memory remains untested after the repair. | Draft and review Phase 2U full LGSSM run subplan. | No full LGSSM score admission yet; no non-LGSSM conclusion; no HMC/posterior/scientific/runtime claim. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | What correctness policy can validly certify the compact no-tape LGSSM score when default TF32 execution fails direct same-scalar FD at smoke scale? |
| Baseline/comparator | Phase 2S TF32-fail/no-TF32-pass smoke pair and Phase 1 score validator. |
| Primary criterion | Met at smoke scale. The selected policy passes trusted GPU smoke and discloses that the FD correctness arm uses TF32 disabled while production score execution uses TF32 enabled. |
| Veto diagnostics | No hidden no-TF32 substitution; no exact Kalman overclaim; no tolerance-only pass; no historical route admission. |
| Explanatory diagnostics | Score-specific memory was populated; correctness metadata records production TF32 and FD-arm TF32 settings. |
| Not concluded | Full score admission, non-LGSSM score admission, HMC readiness, posterior correctness, runtime ranking, or scientific superiority. |

## Selected Policy

For LGSSM compact score admission attempts:

- production value/score execution remains `float32` with TensorFlow TF32
  enabled;
- same-scalar finite-difference correctness uses the value-only scalar route
  with TF32 disabled;
- the score artifact records:
  - `score_correctness.tf32_mode = "disabled"`;
  - `score_correctness.production_tf32_execution_enabled = true`;
  - `score_correctness.tf32_execution_enabled = false`;
  - `score_correctness.uses_disclosed_separate_precision_arm = true`.

This policy does not claim direct TF32 finite-difference parity. It certifies
the same finite scalar with a numerically stable correctness arm and discloses
the precision split.

## Code Changes

Edited:

- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
- `tests/test_ledh_lgssm_manual_score_phase4.py`

Changes:

- Added CLI option `--score-fd-tf32-mode {match,default,enabled,disabled}`.
- Added FD correctness metadata to `manual_score_diagnostic.same_scalar_fd`.
- Preserved production TF32 setting while temporarily setting FD-arm TF32 mode
  around value-only FD evaluations.
- Added correctness metadata into the Phase 1 score artifact.
- Added tests for disclosed no-TF32 correctness arm metadata.

## Checks

Focused tests:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
42 passed, 2 warnings
```

Trusted GPU smoke:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 256 \
  --time-steps 3 \
  --batch-seeds 81120,81121 \
  --transport-policy active-all \
  --sinkhorn-iterations 2 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 64 \
  --col-chunk-size 64 \
  --particle-chunk-size 64 \
  --score-mode compact-sensitivity \
  --score-fd-tf32-mode disabled \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output /tmp/bayesfilter-phase2t-lgssm-gpu-smoke-score.json
```

Result:

```text
score_runtime_gate_applicable = true
score_gpu_memory_info_after.peak = 139720704 bytes
same_scalar_fd.status = pass
same_scalar_fd.max_abs_error = 0.006190299987792969
same_scalar_fd.max_relative_error = 0.0004953071475028992
same_scalar_fd.tf32_mode = disabled
same_scalar_fd.production_tf32_execution_enabled = true
same_scalar_fd.tf32_execution_enabled = false
same_scalar_fd.uses_disclosed_separate_precision_arm = true
```

## Handoff To Phase 2U

Phase 2U may attempt the full LGSSM run only after its subplan is reviewed. It
must use:

- `--num-particles 10000`;
- `--time-steps 50`;
- `--batch-seeds 81120,81121,81122,81123,81124`;
- `--score-mode compact-sensitivity`;
- `--score-fd-tf32-mode disabled`;
- `--dtype float32`;
- `--tf32-mode enabled`;
- trusted GPU execution;
- score-specific memory admission;
- Phase 1 validator admission before leaderboard integration.
