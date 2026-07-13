# Phase 5 Repair Result: Actual-SV Streaming-Flow Parity

metadata_date: 2026-07-07
status: `PASSED_TINY_PARITY_REPAIRED_FULL_NOT_ADMITTED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-repair-streaming-flow-parity

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the actual-SV score route differentiate the exact same finite-`N` streaming-flow scalar used by the admitted value route at tiny diagnostic scale? |
| Baseline/comparator | `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py` with `streaming_tf.batched_ledh_flow_streaming_particles_tf` and raw streaming transport. |
| Primary criterion | Score-route forward scalar matches the value route before all-coordinate FD score evidence is used. |
| Veto diagnostics | Matrix-flow primal, manual-VJP primal that differs from value transport, tape/autodiff, stopped partials, target substitution, nonfinite values, artifact over-admission. |
| Not concluded | Full `N=10000,T=1000` score admission, HMC readiness, posterior correctness, runtime ranking, scientific superiority, or other-model score readiness. |

## Review

- The initial `claude_review_gate.sh` attempt returned `REVIEW_STATUS=probe_timeout`.
- Direct tiny Claude probe returned `CLAUDE_PROBE_OK`.
- Direct bounded read-only packet review returned `VERDICT: AGREE`.
- Claude's caution was preserved: parity must remain tied to the exact padded streaming forward route, not relaxed into algorithmic similarity.

## Root Cause Repaired

Two same-target-forward-scalar defects were found and repaired:

1. The score adapter's LEDH flow primal used the matrix aux primitive instead of the admitted streaming value flow primitive.
2. The score adapter's transport primal used the manual finite VJP transport forward, whose finite schedule did not match the admitted raw streaming transport forward.

The repaired score adapter now:

- uses a streaming-flow-with-aux helper that mirrors `batched_ledh_flow_streaming_particles_tf` chunking, padding, core call order, and posterior covariance arithmetic;
- uses raw streaming transport for the forward particles/log weights to match the value route;
- retains manual no-tape VJP helpers for reverse cotangents only;
- starts from the same fixed initial particles built by the actual-SV value adapter.

## Code And Test Artifacts

- Updated: `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- Updated: `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- Tiny diagnostic JSON:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-parity-tiny-score-diagnostic-2026-07-07.json`
- Tiny diagnostic Markdown:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-parity-tiny-score-diagnostic-2026-07-07.md`

## Checks Run

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py -q
```

Result:

```text
9 passed, 2 warnings
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
30 passed, 2 warnings
```

Tiny repaired diagnostic command:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  --batch-seeds 81120 --time-steps 2 --num-particles 64 \
  --transport-policy active-all --sinkhorn-iterations 2 \
  --sinkhorn-epsilon 1.0 --annealed-scaling 0.9 \
  --annealed-convergence-threshold 1.0e-3 \
  --transport-plan-mode streaming \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --transport-ad-mode full --row-chunk-size 16 --col-chunk-size 16 \
  --particle-chunk-size 16 --dtype float64 --tf32-mode disabled \
  --fd-step 1.0e-5 --score-fd-atol 5.0e-3 --score-fd-rtol 5.0e-3 \
  --source-value-artifact docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json \
  --output docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-parity-tiny-score-diagnostic-2026-07-07.json \
  --markdown-output docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-parity-tiny-score-diagnostic-2026-07-07.md
```

Result:

```text
score = [-0.13676240070260542, 0.38478843496586546]
fd_score = [-0.1367604994584326, 0.38480405004648327]
max_abs_error = 1.561508061781458e-05
max_rel_error = 4.057930423530709e-05
score_admission_status = tiny_score_diagnostic_not_admitted
```

The command was CPU-hidden by design. The CUDA initialization warnings are sandbox/framework noise for this CPU-hidden tiny diagnostic and are not GPU readiness evidence.

## Decision

The Phase 5 streaming-flow parity repair passes the tiny local gates and repairs the same-forward-scalar blocker at diagnostic scale.

Full actual-SV score remains not admitted. The next phase must refresh and review the full-row score/memory subplan before any full `N=10000,T=1000` trusted GPU run.

## Nonclaims

- This is not full actual-SV score admission.
- This is not memory evidence for `N=10000,T=1000`.
- This is not HMC readiness evidence.
- This is not posterior correctness evidence.
- This is not a runtime ranking or scientific superiority claim.
