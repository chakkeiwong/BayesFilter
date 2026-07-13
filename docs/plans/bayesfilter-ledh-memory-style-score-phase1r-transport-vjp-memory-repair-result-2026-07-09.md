# LEDH Memory-Style Score Phase 1R Result: Transport VJP Memory Repair

Date: 2026-07-09

Status: `PARTIAL_PASS_WITH_FULL_SINKHORN_STEP_MEMORY_BLOCKER`

## Phase Objective

Repair the remaining memory blocker in the memory-style LEDH score path by
removing repeated full scatter/update allocation from the streaming transport
VJP pullbacks, while preserving the realized finite-`N` LEDH
`observed_data_log_likelihood_estimator` / `log_likelihood` scalar.

## Implementation Summary

Updated:

- `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
  - `_filterflow_streaming_softmin_vjp` now accumulates row/column block
    cotangents in TensorArrays and stacks/slices once at exit.
  - `_filterflow_streaming_transport_from_potentials_vjp` now uses block
    TensorArray accumulation for `s`, particle, query, key, and `f`
    cotangents instead of repeated `_scatter_axis1_add_2d` /
    `_scatter_axis1_add_3d` full-accumulator updates.
  - Existing JVP helpers avoid known `[B,R,C,D,P]` broadcast products through
    `_half_pairwise_squared_cross_jvp` and einsum contraction.
- `tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py`
  - Added source sentinels forbidding local `GradientTape`,
    `ForwardAccumulator`, `.gradient(`, old scatter helpers, and
    `tf.tensor_scatter_nd_add` in the two repaired VJP hot helpers.
- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
  - Refreshed stale runner wording so the default score route is described as
    memory-style reverse/VJP, with the compact forward-sensitivity route marked
    historical/diagnostic.

## Review

Claude read-only review was not used in this phase because the local escalation
reviewer had already rejected the bounded Claude repo-artifact disclosure path
for this program. Per the runbook fallback, a fresh local Codex read-only
review was used.

Local Codex reviewer verdict: `VERDICT: AGREE`.

Reviewer notes:

- no blocking findings;
- softmin VJP signs and epsilon/key/value cotangents looked correct;
- transport-from-potentials VJP normalizer cancellation and `d_g = 0` looked
  correct;
- no local production `GradientTape`/`ForwardAccumulator` or repeated full
  scatter-add updates were found in the reviewed hot helpers;
- residual nonblocking gap: the closest direct primitive oracle lives in
  `tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py`, while the phase
  sentinel file itself is mostly structural.

## Checks Run

Syntax:

```bash
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py
```

Result: `pass`.

Focused CPU-hidden contract suite:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py \
  tests/test_ledh_compact_transport_jvp.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result: `59 passed, 2 warnings`.

Primitive VJP oracle checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_softmin_vjp_matches_dense_and_tiny_autodiff \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_transport_from_potentials_vjp_matches_manual_and_tiny_autodiff -q
```

Result: `5 passed`.

## Trusted GPU Rungs

All GPU commands were run in trusted/escalated mode as required by the local
GPU/CUDA policy.

### Rung 1: `N=256,T=3` LGSSM Score-Only

Artifact:

`docs/plans/artifacts/ledh-memory-style-score-phase1r-lgssm-score-only-n256-t3-2026-07-09.json`

Result:

- emitted successfully;
- `score_route =
  memory_style_reverse_vjp_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`;
- `score_status = blocked_score_only_diagnostic_not_admitted`;
- `score_admission_status = blocked_score_diagnostic_stage_not_admitted`;
- `score_gpu_memory_info_after.peak = 69932032` bytes, about `66.69 MiB`;
- explicitly non-admitting because same-scalar FD was not run in score-only
  stage.

### Rung 2: `N=1000,T=10,Sinkhorn=2` LGSSM Score-Only

Artifact:

`docs/plans/artifacts/ledh-memory-style-score-phase1r-lgssm-score-only-n1000-t10-2026-07-09.json`

Result:

- emitted successfully;
- `score_route =
  memory_style_reverse_vjp_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`;
- `score_status = blocked_score_only_diagnostic_not_admitted`;
- `score_admission_status = blocked_score_diagnostic_stage_not_admitted`;
- `score_gpu_memory_info_after.peak = 275654656` bytes, about `262.88 MiB`;
- this shows the repaired VJP hot helpers can emit at medium particle/time
  scale when the Sinkhorn finite-step count is the small smoke setting.

### Rung 2b: `N=1000,T=10,Sinkhorn=10` LGSSM Score-Only

Command:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 1000 \
  --time-steps 10 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --annealed-scaling 0.9 \
  --annealed-convergence-threshold 0.001 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 256 \
  --col-chunk-size 256 \
  --particle-chunk-size 256 \
  --score-mode compact-sensitivity \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-memory-style-score-phase1r-lgssm-score-only-n1000-t10-s10-2026-07-09.json
```

Result:

- no artifact emitted;
- trusted `nvidia-smi` during the run reported about
  `15698 MiB / 16376 MiB`;
- this exceeds the reviewed score-memory budget of `14000 MiB`;
- the exact TensorFlow process `579194` was terminated after the memory gate
  failed; the process exited with code `143`;
- artifact absence was confirmed.

### Rung 3: `N=10000,T=50` Single-Seed LGSSM Score-Only

Command:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 10000 \
  --time-steps 50 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --annealed-scaling 0.9 \
  --annealed-convergence-threshold 0.001 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode compact-sensitivity \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-memory-style-score-phase1r-lgssm-score-only-n10000-t50-seed81120-2026-07-09.json
```

Result:

- no artifact emitted;
- trusted `nvidia-smi` during the run reported about
  `15695 MiB / 16376 MiB`, then `15733 MiB / 16376 MiB`;
- this exceeds the reviewed score-memory budget of `14000 MiB`;
- the exact TensorFlow process `573884` was terminated after the memory gate
  failed; the process exited with code `143`;
- artifact absence was confirmed.

## Root-Cause Update

The original Phase 1R root cause was real: repeated full scatter-add updates
inside the transport VJP were on the interrupted stack and have now been
removed. The repair is validated by focused VJP oracle tests and by the
`N=1000,T=10,Sinkhorn=2` emission under budget.

The remaining blocker is more specific than "LGSSM full history" alone. The
extra `N=1000,T=10,Sinkhorn=10` rung reproduces the about-`15.7 GiB` no-artifact
failure even before full `N=10000,T=50`. The next hot surface is therefore the
finite-Sinkhorn reverse over iteration history in
`_filterflow_streaming_finite_sinkhorn_potentials_vjp_total`, which stores five
full `[batch,N]` TensorArray histories for every finite Sinkhorn step and calls
`_filterflow_streaming_softmin_vjp` many times in the reverse loop. LGSSM
per-time history may still matter at full `T=50`, but it is now secondary to
the Sinkhorn-step-count memory gate.

## Decision Table

| Field | Decision |
| --- | --- |
| Primary criterion status | Partial pass: focused tests pass and `N=1000,T=10,Sinkhorn=2` emits under budget. |
| Veto diagnostic status | `N=1000,T=10,Sinkhorn=10` and full single-seed `N=10000,T=50,Sinkhorn=10` score-only violate memory/no-artifact. |
| Main uncertainty | Exact split between finite-Sinkhorn step-history lifetime, repeated softmin-VJP block allocation, LGSSM time history, and XLA allocator behavior. |
| Next justified action | Phase 1S: instrument and repair finite-Sinkhorn reverse memory lifetime first, then revisit LGSSM time-history checkpointing if needed. |
| Not concluded | No full LGSSM score admission, no full leaderboard score, no all-model readiness, no HMC readiness, no posterior correctness, no scientific superiority. |

## Next-Phase Handoff

Proceed to:

`docs/plans/bayesfilter-ledh-memory-style-score-phase1s-sinkhorn-reverse-lifetime-subplan-2026-07-09.md`

Phase 1S inherits:

- same scalar target:
  `observed_data_log_likelihood_estimator` / `log_likelihood`;
- memory-style reverse/VJP route as the default LGSSM score route;
- old compact forward-sensitivity route remains historical/tiny-only;
- shared transport VJP scatter/update repair is in place and tested;
- full-row Sinkhorn step count `10` still fails memory even at
  `N=1000,T=10`;
- score-only rungs remain diagnostic and cannot admit a leaderboard score row.
