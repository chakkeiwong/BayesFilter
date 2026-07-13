# Phase 3 Result: Actual-SV Compact Score Port

Date: 2026-07-08

Status: `PASSED_TINY_COMPACT_GATE_PHASE4_MAY_START_AFTER_REVIEW`

## Question

Can actual-SV compute the same transformed finite-`N` LEDH
`log_likelihood` score using compact forward sensitivity instead of the
historical reverse-record/manual-total-VJP route?

## Decision

Passed for the Phase 3 tiny gate.

The actual-SV default score route is now:

```text
compact_forward_sensitivity_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot
```

The historical route remains present only as a diagnostic comparator:

```text
manual_total_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot
```

and cannot full-admit through the shared score contract.

## Implementation Anchors

- Shared compact provenance allowlist:
  `bayesfilter/highdim/ledh_score_contract.py`.
- Actual-SV compact transport value/JVP helper:
  `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`,
  `_compact_forward_transport_jvp_tf`.
- Actual-SV compact LEDH flow JVP:
  `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`,
  `_compact_streaming_flow_jvp_tf`.
- Actual-SV compact forward loop:
  `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`,
  `_compact_value_and_score_from_components`.
- Default across-seed score dispatch now calls the compact route:
  `_manual_value_and_score_across_seeds`.
- Score artifact provenance is read from the computed route and defaults to
  compact actual-SV provenance.

## Evidence Contract Result

| Field | Status |
| --- | --- |
| Same transformed actual-SV value scalar | Passed tiny guard; compact score `log_likelihood` matches value route at `<= 1e-12`. |
| No tape/autodiff runtime sentinel | Passed. |
| Default route avoids reverse records | Passed static source test for compact/default dispatch: no `records.append`, `reversed(records)`, `_transport_vjp_tf`, `GradientTape`, or `ForwardAccumulator`. |
| Coordinate FD | Passed tiny all-coordinate check. |
| Full admission | Not claimed; tiny artifact remains `tiny_score_diagnostic_not_admitted`. |
| Memory evidence | Not run in Phase 3; `n10000_memory_pass=false` in the tiny artifact. |

## Local Checks

Command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
38 passed, 2 warnings
```

Tiny diagnostic command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  --source-value-artifact docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json \
  --output docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-tiny-compact-score-2026-07-08.json \
  --time-steps 2 --num-particles 9 --batch-seeds 81120 \
  --particle-chunk-size 4 --row-chunk-size 9 --col-chunk-size 9 \
  --sinkhorn-iterations 1 --transport-policy active-all \
  --dtype float64 --tf32-mode disabled
```

Result:

```text
score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot
score_admission_status = tiny_score_diagnostic_not_admitted
max_abs_error = 6.049662710727599e-06
max_rel_error = 2.1665722010856176e-05
```

Artifact:

```text
docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-tiny-compact-score-2026-07-08.json
```

## Transport Note

The compact actual-SV score loop carries the value state and parameter
tangents forward. To preserve the exact admitted value scalar, the transported
particles/log weights used for the scalar are produced by the raw streaming
value route. The tangent is produced by the no-tape finite streaming transport
value+JVP helper and is checked against centered same-scalar finite
differences at the tiny gate.

This is sufficient for the Phase 3 tiny diagnostic. It is not full-row
admission evidence.

## Nonclaims

- This phase does not admit the full `N=10000,T=1000` actual-SV score.
- This phase does not provide GPU memory evidence.
- This phase does not establish HMC readiness, posterior correctness, runtime
  ranking, or scientific superiority.
- This phase does not port fixed-SIR, predator-prey, generalized-SV, or KSC-SV.

## Handoff

Phase 4 fixed-SIR may start after read-only review confirms no unresolved
material boundary issue. Phase 4 must not rename the current fixed-SIR
manual-total-VJP route as compact. It must either implement a real compact
forward-sensitivity route for fixed-SIR or write a blocker result.
