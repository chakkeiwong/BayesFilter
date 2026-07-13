# Phase 2S Result: LGSSM Score Procedure And Memory Repair

Date: 2026-07-09

Status: `BLOCKED_FIXABLE_TF32_CORRECTNESS_GATE`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not launch the full `N=10000,T=50` LGSSM score run yet. | Not met: no validator-admitted full Phase 2S score artifact was produced. | Correctness veto active under default TF32 GPU smoke; score-specific memory instrumentation and value-only FD route repaired. | Whether TF32 full admission should use adjusted FD tolerances/step, a no-TF32 correctness arm, or a reviewed exact/reference correctness gate while keeping production value/score TF32. | Draft Phase 2T TF32 correctness-policy repair before any full run. | No LGSSM score admission, no leaderboard completion, no compact-math rejection, no HMC/posterior/scientific/runtime claim. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can LGSSM score admission be unblocked by repairing score procedure, score memory measurement, and value-only same-scalar FD checks before another full `N=10000,T=50` run? |
| Baseline/comparator | Phase 2/2R no-artifact runs, admitted LGSSM value artifact, Phase 1 validator, existing compact transport JVP tests. |
| Primary criterion | Not met. Phase 2S stopped before full run because trusted GPU TF32 correctness smoke failed. |
| Veto diagnostics | Default TF32 trusted GPU smoke failed same-scalar FD at `N=256,T=3`; full trusted run is therefore blocked. |
| Explanatory diagnostics | CPU tests passed, CPU prefix smoke passed, trusted GPU no-TF32 smoke passed, score memory fields were populated from score route. |
| Not concluded | The compact derivative is not rejected; the current blocker is a default-TF32 correctness/admission-policy issue. |

## Code Changes

Edited:

- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
- `tests/test_ledh_lgssm_manual_score_phase4.py`

Changes:

- Added `_same_target_value_from_components` so same-scalar FD uses the value
  route rather than recomputing the compact score/JVP route for each
  plus/minus scalar.
- Added score-specific GPU memory reset and
  `score_gpu_memory_info_before/after` around score diagnostics.
- Changed score artifact memory diagnostics to use
  `score_gpu_memory_info_after`, not `gpu_memory_info_after`.
- Tightened top-level compact score admission so score-specific memory must
  pass before the raw result status says `admitted_same_target_compact_score`.
- Added tests that fail if FD calls the score route and if score artifact memory
  is copied from the value route.
- Added nested `score_artifact` emission for full-row score runs so downstream
  integration can validate the Phase 1 artifact directly.

## Local Checks

Focused CPU tests:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_compact_transport_jvp.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/test_ledh_lgssm_manual_score_phase4.py -q
```

Result:

```text
52 passed, 2 warnings
```

CPU prefix smoke:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 64 \
  --time-steps 3 \
  --batch-seeds 81120,81121 \
  --transport-policy active-all \
  --sinkhorn-iterations 2 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 16 \
  --col-chunk-size 16 \
  --particle-chunk-size 16 \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /CPU:0 \
  --device-scope cpu \
  --expect-device-kind cpu \
  --output /tmp/bayesfilter-phase2s-lgssm-prefix-score.json
```

Result:

```text
score_status = executed_compact_score_fd_pass_but_material_gate_blocked
score_admission_status = blocked_material_gate_not_full_gpu_row
same_scalar_fd.status = pass
score_artifact_present = false
```

CPU value-only FD parity diagnostic at `N=256,T=3`:

```text
float32 base compact/value-only objective diff: 9.5367431640625e-07
float64 base compact/value-only objective diff: 0.0
value-only FD matched compact-objective FD up to float roundoff.
```

## Trusted GPU Smoke Diagnostics

Default TF32 smoke:

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
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output /tmp/bayesfilter-phase2s-lgssm-gpu-smoke-score.json
```

Result:

```text
score_runtime_gate_applicable = true
score_gpu_memory_stats_reset = true
score_gpu_memory_info_after.peak = 139720704 bytes
same_scalar_fd.status = fail
same_scalar_fd.max_abs_error = 3.4168434143066406
same_scalar_fd.max_relative_error = 0.21000058948993683
```

No-TF32 GPU smoke:

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
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode disabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output /tmp/bayesfilter-phase2s-lgssm-gpu-smoke-score-no-tf32.json
```

Result:

```text
score_runtime_gate_applicable = true
same_scalar_fd.status = pass
same_scalar_fd.max_abs_error = 0.0009069442749023438
same_scalar_fd.max_relative_error = 0.0010520674986764789
```

TF32 FD-step diagnostic:

```text
step 0.001: fail, max_abs_error 3.4168434143066406
step 0.003: fail, max_abs_error 0.7971048355102539
step 0.01: fail, max_abs_error 0.1394672393798828
```

The FD-step sweep was interrupted before larger steps because it exceeded a
reasonable bounded diagnostic window. It did not produce a passing default-TF32
configuration.

## Interpretation

Phase 2S fixed the two procedural issues identified by review:

1. FD no longer calls the compact score/JVP route.
2. Score admission no longer borrows value-route memory evidence.

The next blocker is now cleaner and smaller: default TF32 GPU score correctness
does not pass the current same-scalar FD gate at the smoke scale, while the same
GPU smoke passes with TF32 disabled. Because BayesFilter's production LEDH
route defaults to TF32, Codex must not silently switch the full admission run to
no-TF32 without a reviewed policy decision.

## Handoff To Phase 2T

Draft a Phase 2T subplan before further full runs. Phase 2T should decide and
test one of these admissible paths:

1. keep full production score/value execution in TF32 but use a reviewed
   no-TF32 or FP64 correctness arm for same-scalar FD;
2. adjust TF32 FD step/tolerance through a bounded, reviewed calibration if it
   remains mathematically meaningful;
3. use exact/reference correctness evidence for LGSSM while preserving the
   same finite LEDH target for value/score.

Any path must preserve the target scalar, admitted value artifact, row id,
seeds, `N=10000`, `T=50`, theta coordinate system, parameter order, compact
provenance, and score-memory gate.
