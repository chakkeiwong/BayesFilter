# LEDH Memory-Style Score Phase 1S Subplan: Sinkhorn Reverse Lifetime

Date: 2026-07-09

Status: `READY_FOR_REVIEW_BEFORE_EXECUTION`

## Phase Objective

Repair the remaining full-row Sinkhorn-step-count memory blocker in the
memory-style LEDH score route by reducing finite-Sinkhorn reverse lifetime in
`_filterflow_streaming_finite_sinkhorn_potentials_vjp_total`, without changing
the realized finite-`N` LEDH `observed_data_log_likelihood_estimator` /
`log_likelihood` scalar or the score parameter order.

The immediate target is `N=1000,T=10,Sinkhorn=10` score-only emission under
budget. If that passes, the next target is a single-seed `N=10000,T=50`
LGSSM score-only diagnostic that emits under the reviewed memory budget. Full
score admission remains a later phase because same-scalar FD and fixed-seed
aggregation are not part of this phase.

## Entry Conditions Inherited From Previous Phase

- Phase 1R transport VJP repair passed focused CPU-hidden tests.
- Primitive softmin and transport-from-potentials VJP oracle tests passed.
- Trusted GPU `N=256,T=3,Sinkhorn=2` and `N=1000,T=10,Sinkhorn=2`
  score-only diagnostics emitted under budget with memory-style score
  provenance.
- Trusted GPU `N=1000,T=10,Sinkhorn=10` exceeded memory by `nvidia-smi`
  observation and emitted no artifact.
- Trusted GPU `N=10000,T=50` single-seed score-only exceeded memory by
  `nvidia-smi` observation and emitted no artifact.
- Old compact forward-sensitivity route is historical/tiny-only.

## Required Artifacts

- Updated implementation, if repair proceeds:
  `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- Updated implementation, if LGSSM history repair is also needed:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- Updated or added focused tests:
  - `tests/test_ledh_lgssm_manual_score_phase4.py`
  - `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
  - optional new checkpointing test module if the patch creates a named helper
- Phase 1S result record:
  `docs/plans/bayesfilter-ledh-memory-style-score-phase1s-sinkhorn-reverse-lifetime-result-2026-07-09.md`
- Trusted GPU score-only artifacts or blocker records under:
  `docs/plans/artifacts/`

## Required Checks, Tests, And Reviews

Review:

- Use Claude read-only review only if the bounded review command is locally
  permitted.
- If Claude is unavailable or policy-blocked, use a fresh Codex read-only
  review and record the limitation.

Pre-implementation diagnostics:

1. Audit finite-Sinkhorn VJP retained state in
   `_filterflow_streaming_finite_sinkhorn_potentials_vjp_total`:
   - list `running_ta`, `a_y_ta`, `b_x_ta`, `a_x_ta`, and `b_y_ta` shapes;
   - count softmin VJP calls as a function of finite `steps`;
   - identify whether all five histories are required or can be recomputed by
     a checkpoint/replay schedule.
2. Audit whether `_filterflow_streaming_softmin_vjp` can expose a more direct
   cotangent mode for repeated Sinkhorn reverse calls so it avoids allocating
   unused blocks.
3. Keep LGSSM retained-history audit as a secondary diagnostic only after
   `N=1000,T=10,Sinkhorn=10` emits under budget.

CPU-hidden tests after implementation:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_softmin_vjp_matches_dense_and_tiny_autodiff \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_transport_from_potentials_vjp_matches_manual_and_tiny_autodiff -q
```

GPU rungs, trusted/escalated only:

1. `N=1000,T=10,Sinkhorn=2` score-only regression.
2. `N=1000,T=10,Sinkhorn=10` score-only blocker rung.
3. `N=10000,T=50,Sinkhorn=10` single-seed score-only only if rung 2 emits
   under budget.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can memory-style score avoid full-row Sinkhorn-step-count no-artifact memory failure by reducing finite-Sinkhorn reverse lifetime rather than changing the target scalar? |
| Baseline/comparator | Phase 1R `N=1000,T=10,Sinkhorn=2` pass, `N=1000,T=10,Sinkhorn=10` no-artifact blocker, and `N=10000,T=50,Sinkhorn=10` no-artifact blocker. |
| Primary criterion | Trusted GPU `N=1000,T=10,Sinkhorn=10` score-only emits under the reviewed `14000 MiB` score-memory budget; then one-seed `N=10000,T=50,Sinkhorn=10` emits under budget. |
| Correctness criterion | Focused CPU tests pass; tiny/prefix score values stay same-scalar and same-route; no production autodiff is introduced. |
| Veto diagnostics | Target scalar drift; exact Kalman substitution; parameter-order change; production `GradientTape`/`ForwardAccumulator`; stopped partials; nonfinite score; no artifact; memory over budget. |
| Explanatory diagnostics | Runtime, peak GPU memory, Sinkhorn step count, retained-state key list, checkpoint/replay size, whether remaining memory scales with `steps`, `T`, or `N`. |
| Not concluded | Full score admission, all-model score readiness, HMC readiness, posterior correctness, scientific superiority, or fixed-seed aggregate validity. |

## Implementation Plan

1. Build a retained-state ledger for the finite-Sinkhorn total VJP path:
   - classify each per-step TensorArray as required, recomputable, or removable;
   - record expected memory scaling in `steps`, `N`, and `batch`;
   - identify which softmin VJP returned blocks are used by each caller.
2. Prefer a checkpoint/replay design over storing all per-step potentials:
   - store only sparse checkpoints or final state;
   - replay finite Sinkhorn forward state for a short reverse segment;
   - reverse through that segment and discard local state;
   - preserve the exact finite iteration schedule and arithmetic target.
3. If a smaller softmin VJP API patch is enough, add specialized return modes
   so repeated Sinkhorn reverse calls avoid allocating unused `d_key`,
   `d_values`, `d_query`, or epsilon cotangent blocks.
4. Add tiny equivalence tests comparing old full-history total VJP and new
   checkpointed/minimal-lifetime total VJP at small `N,steps`.
5. Only after `N=1000,T=10,Sinkhorn=10` emits under budget, audit LGSSM
   per-time retained state and decide whether checkpointed LGSSM segment replay
   is also needed.
6. Preserve the public runner surface and score artifact fields.
7. Run CPU-hidden tests.
8. Run trusted GPU rungs in sequence and stop on first no-artifact or
   memory-over-budget rung.
9. Write the Phase 1S result record and refresh the next subplan.

## Forbidden Claims And Actions

- Do not claim score admission from score-only diagnostics.
- Do not replace LEDH score with exact Kalman score.
- Do not change seeds, `N`, `T`, parameter order, target output field, or
  target scalar after seeing results.
- Do not use production `GradientTape`, `ForwardAccumulator`, stopped partials,
  or the historical compact forward-sensitivity route as admission evidence.
- Do not run five-seed aggregation, FD-only correctness, or full
  `N=10000,T=50` before the `N=1000,T=10,Sinkhorn=10` blocker rung emits under
  budget.
- Do not edit unrelated HMC, Neutra, or leaderboard files during this repair.

## Exact Next-Phase Handoff Conditions

Handoff to score-admission re-entry only if:

- focused CPU-hidden tests pass;
- trusted GPU `N=1000,T=10,Sinkhorn=10` emits under budget;
- trusted GPU one-seed `N=10000,T=50` score-only emits under budget;
- result record states that the output is diagnostic-only until same-scalar FD
  and fixed-seed aggregation are run.

Handoff to an LGSSM time-history checkpointing subplan if:

- `N=1000,T=10,Sinkhorn=10` emits under budget;
- full `N=10000,T=50,Sinkhorn=10` still breaches memory with no artifact;
- the remaining stack/evidence points to `_manual_value_and_score_from_components`
  per-time history rather than finite-Sinkhorn reverse lifetime.

Handoff to a lower-level TensorFlow/XLA memory diagnosis subplan if the
BayesFilter-owned finite-Sinkhorn and LGSSM retained-state ledgers are both
minimal but memory still exceeds budget.

## Stop Conditions

Stop and write a blocker result if:

- review finds an unpatched material flaw;
- a focused correctness test fails;
- the proposed patch changes the scalar identity or parameter order;
- source sentinels find production autodiff or old historical compact route use
  in the admitted score path;
- any GPU rung does not emit, exceeds memory budget, or shows nonfinite score;
- continuing would require package installation, network access, external
  service access, or unrelated dirty-worktree edits.

## Skeptical Audit Before Execution

- Wrong baseline checked: exact Kalman is not a score target here.
- Proxy metric checked: score-only emission is diagnostic, not admission.
- Hidden assumption checked: Phase 1R repaired transport VJP scatter updates
  but did not prove finite-Sinkhorn reverse at `steps=10` is memory-safe.
- Environment checked: all GPU rungs require trusted execution.
- Artifact sufficiency checked: a result or blocker record is mandatory.

Audit status: `PASS_FOR_READ_ONLY_REVIEW_BEFORE_IMPLEMENTATION`.
