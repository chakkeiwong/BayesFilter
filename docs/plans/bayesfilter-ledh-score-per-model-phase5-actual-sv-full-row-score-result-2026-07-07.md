# Phase 5 Repair Result: Actual-SV Full-Row Score/Memory Gate

metadata_date: 2026-07-07
status: `CLOSED_BLOCKED_SAME_ALGORITHM_PARITY_GAP`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-repair-full-row

## Phase Objective

Decide whether the tiny-passing actual-SV no-tape score route can proceed to a
full `N=10000,T=1000` score/memory run.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Full actual-SV score/memory run is blocked. |
| Primary criterion status | Not met: the tiny score route is close to, but not exactly the same algorithm as, the admitted value route. |
| Veto diagnostic status | Same-algorithm parity gap vetoes full-row execution and score admission. |
| Main uncertainty | Need a streaming-flow aux/VJP route, or another reviewed mechanism, that differentiates the exact same streaming-flow value algorithm used by the admitted actual-SV value artifact. |
| Next justified action | Create a repair subplan for actual-SV streaming-flow parity and manual VJP before any full-row score/memory ladder. |
| What is not concluded | No full actual-SV score admission; no rejection of the tiny route mathematics; no HMC readiness, posterior correctness, runtime ranking, or scientific superiority claim. |

## Memory-Risk Audit

The current score route stores per-time reverse records:

- `particles`;
- `proposal_mean`;
- `proposal_covariance`;
- `noise`;
- `post_flow`;
- `corrected_log_weights`;
- `normalized_log_weights`;
- `mask`;
- full matrix-flow `flow_aux`.

For scalar actual-SV at `T=1000,N=10000,B=1,float64`, a conservative tensor
payload estimate is:

- explicit non-flow records: about `0.458 MiB` per time step;
- flow auxiliary records: about `0.992 MiB` per time step;
- total retained tensor payload: about `1.45 MiB` per time step;
- full `T=1000` payload: about `1.42 GiB` per seed before TensorFlow/object
  overhead.

This estimate does not by itself veto full execution under the `14000 MiB`
budget. It also does not admit the score, because memory is not correctness
evidence and the same-algorithm parity gate failed.

## Same-Algorithm Parity Audit

The admitted value artifact was produced by:

```text
docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py
```

using:

```text
streaming_tf.batched_ledh_flow_streaming_particles_tf
```

The tiny score diagnostic route used the matrix aux/VJP primitive:

```text
core_tf._batched_ledh_linearized_flow_with_aux_tf
core_tf._batched_ledh_linearized_flow_vjp
```

This was useful for a bounded no-tape VJP diagnostic, but it is not the exact
same implementation block as the admitted value route.

Targeted parity probe at `T=2,N=64`, seed `[81120]`, `float64`:

```text
value_route   = [-3.603378542893297]
score_forward = [-3.6034006908781437]
diff          = [-2.214798484656555e-05]
```

Interpretation:

- The score route is close to the value route.
- It is not exactly the same forward scalar/algorithm.
- This is wrong relative to the runbook's same-algorithm gate for admission.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the tiny-passing actual-SV no-tape total score route scale to full `N=10000,T=1000` with replayable correctness and memory evidence? |
| Answer | Not under the current implementation. The route must first close the same-algorithm parity gap. |
| Baseline/comparator | Tiny actual-SV score diagnostic, admitted actual-SV value artifact, Phase 1 score artifact validator, and same-forward-scalar parity probe. |
| Primary criterion | Failed/not met because exact same-algorithm parity with the admitted value route is not established. |
| Veto diagnostics | Same-algorithm parity gap. |
| Explanatory diagnostics | Memory estimate suggests the stored-record payload may be bounded, but it cannot override the parity veto. |
| Artifact | This blocker result plus the Phase 5 tiny diagnostic and score code/tests. |

## Required Repair

Draft and review a subplan to make the actual-SV score route use the exact same
streaming-flow algorithm as the admitted value route, while exposing enough
manual VJP information for no-tape total derivatives.

Candidate repair directions:

1. Add a streaming-flow-with-aux primitive that mirrors
   `batched_ledh_flow_streaming_particles_tf` block-by-block and stores only
   the block aux needed for reverse scan.
2. Implement a streaming-flow VJP that replays block-level matrix VJP in the
   same padded block order as the value route.
3. Add same-forward-scalar parity tests between the value route and score route
   before any FD score tests.
4. Only after parity passes, rerun tiny all-coordinate FD and revisit the
   full-row memory ladder.

## Nonclaims

- Full actual-SV score is not admitted.
- Tiny score correctness is not invalidated by this result, but it is
  insufficient for admission.
- Memory estimate is not correctness evidence.
- No KSC, generalized-SV, raw Gaussian, or augmented-noise score claim is made.
- No HMC readiness, posterior correctness, runtime ranking, scientific
  superiority, or all-algorithm comparison is claimed.
