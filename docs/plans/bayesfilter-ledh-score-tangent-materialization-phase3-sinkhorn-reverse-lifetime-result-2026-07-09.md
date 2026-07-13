# Phase 3 Result: LGSSM Sinkhorn Reverse Lifetime Repair

Date: 2026-07-09

Status: `PARTIAL_PASS_NEW_FULL_SIZE_NO_ARTIFACT_BLOCKER`

## Objective

Repair the remaining LGSSM memory-style score blocker by reducing finite
Sinkhorn reverse/VJP tensor lifetime at the production Sinkhorn step count,
without changing the realized finite-`N` LEDH
`observed_data_log_likelihood_estimator`, parameter order, seeds, transport
policy, or score admission criteria.

## Implementation

Updated:

- `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- `tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py`
- `tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py`

Added and wired:

```text
_filterflow_streaming_same_points_softmin_vjp
```

The total finite-Sinkhorn VJP now uses this helper in
`_filterflow_streaming_finite_sinkhorn_potentials_vjp_total` for all
same-points softmin pullbacks. This avoids the old generic route's separate
full `d_query` and `d_key` cotangent output stacks on the equal-chunk path used
by the memory-style LGSSM score route.

The generic `_filterflow_streaming_softmin_vjp` remains available for
non-identical query/key cases and as a fallback when row and column chunk sizes
are unequal.

## Review

Claude was not retried for this program because the approval reviewer rejected
the bounded Claude artifact disclosure path. A fresh Codex read-only review was
recorded in:

```text
docs/reviews/bayesfilter-ledh-score-tangent-materialization-phase3-codex-review-2026-07-09.md
```

Review verdict:

```text
VERDICT: AGREE
```

## Local Checks

```bash
python -m py_compile \
  experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py
```

Result: passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_softmin_vjp_matches_dense_and_tiny_autodiff \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_same_points_softmin_vjp_matches_generic_route \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_transport_from_potentials_vjp_matches_manual_and_tiny_autodiff \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
68 passed, 2 warnings in 29.68s
```

## Trusted GPU Rung 1: Required Blocker Gate

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
  --score-mode manual-reverse \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-score-tangent-materialization-phase3-lgssm-score-only-n1000-t10-s10-2026-07-09.json
```

Result artifact:

```text
docs/plans/artifacts/ledh-score-tangent-materialization-phase3-lgssm-score-only-n1000-t10-s10-2026-07-09.json
```

Key result:

| Field | Value |
| --- | --- |
| `num_particles` | `1000` |
| `time_steps` | `10` |
| `sinkhorn_iterations` | `10` |
| `score_route` | `memory_style_reverse_vjp_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot` |
| `score_status` | `blocked_score_only_diagnostic_not_admitted` |
| `score_peak_mib` | `262.915283203125` |
| `score_memory_budget_mib` | `14000.0` |

Interpretation: the original `N=1000,T=10,Sinkhorn=10` memory/no-artifact
blocker is repaired for the score-only diagnostic rung.

## Trusted GPU Rung 2: Full-Size Single-Seed Diagnostic

Because rung 1 emitted under budget, the subplan permitted the single-seed
full-size diagnostic:

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
  --score-mode manual-reverse \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-score-tangent-materialization-phase3-lgssm-score-only-n10000-t50-seed81120-2026-07-09.json
```

Observed result:

- TensorFlow initialized GPU and XLA.
- The original command process disappeared after roughly 12 minutes.
- No output artifact was written at the requested path.
- A trusted `nvidia-smi` check during/after the anomaly showed high GPU memory
  residency but no running compute process.
- The execution handle remained open briefly after the child process was no
  longer present.

Missing artifact:

```text
docs/plans/artifacts/ledh-score-tangent-materialization-phase3-lgssm-score-only-n10000-t50-seed81120-2026-07-09.json
```

Interpretation: do not classify this as a mathematical score failure or as the
old finite-Sinkhorn reverse memory failure. The repaired blocker rung passed.
The full-size rung has a new procedural/runtime blocker: no atomic artifact or
failure record is emitted when the large run exits anomalously.

## Evidence Contract Status

| Requirement | Status |
| --- | --- |
| Same finite-`N` scalar preserved in code path | Supported by focused VJP/source tests |
| No production autodiff tape introduced | Passed |
| No exact Kalman score substitution | Passed |
| `N=1000,T=10,Sinkhorn=10` emits under 14 GiB | Passed |
| `N=10000,T=50,Sinkhorn=10` emits under 14 GiB | Blocked: no artifact emitted |
| Full score admission | Not attempted; still blocked by score-only diagnostic stage |

## Nonclaims

This phase does not claim full score admission, full leaderboard completion,
HMC readiness, posterior correctness, scientific superiority, or cross-model
score readiness. The full-size rung did not emit a score artifact and therefore
cannot be used as admission evidence.

## Handoff

The next phase should repair the full-size runner procedure before another
large run:

- write atomic progress/failure artifacts before and after value execution,
  before score execution, after score execution, and on exceptions;
- record process/runtime/GPU-memory state even when score-only execution fails;
- add a small CPU-hidden or tiny GPU smoke test for the failure-artifact path;
- rerun the full-size single-seed diagnostic only after artifact emission is
  guaranteed.

Next subplan:

```text
docs/plans/bayesfilter-ledh-score-tangent-materialization-phase3r-lgssm-fullsize-artifact-procedure-subplan-2026-07-09.md
```
