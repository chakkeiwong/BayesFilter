# LEDH Reduce-Only Score Recurrence Subplan

Date: 2026-07-09

Status: `DRAFT_NEXT_REPAIR`

## Phase Objective

Design and implement the next repair after the shared contraction patch:
a reduce-only or checkpointed no-tape score recurrence for LEDH compact scores
that preserves the same realized finite-`N`
`observed_data_log_likelihood_estimator` / `log_likelihood` scalar but avoids
full parameter-axis particle tangent storage at full `N=10000,T=50` scale.

## Entry Conditions Inherited From Previous Phase

- Shared transport JVP contraction repair passed focused tests:
  `53 passed, 2 warnings`.
- `N=256,T=3` and `N=1000,T=10` LGSSM score-only GPU diagnostics emitted.
- Full `N=10000,T=50` LGSSM score-only still exceeded the memory gate and
  emitted no artifact.
- No score artifact is admitted.

## Required Artifacts

- Design note or derivation under `docs/plans/` explaining the recurrence and
  its scalar identity.
- Implementation changes in the narrowest possible files, likely:
  - `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`;
  - `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`;
  - possibly a new helper module only if reuse across models is clear.
- Tests proving tiny equivalence to the current compact forward-sensitivity
  route.
- Result record:
  `docs/plans/bayesfilter-ledh-reduce-only-score-recurrence-result-2026-07-09.md`
- Review packet under `docs/reviews/`.

## Required Checks, Tests, And Reviews

Before implementation:

1. Write the mathematical recurrence with the exact scalar, carried state,
   adjoints or directional accumulators, and what tensors are no longer
   materialized.
2. Review the derivation with MathDevMCP if it can audit the labeled LaTeX or
   plan context; otherwise use a fresh Codex read-only review packet. Claude may
   be used only if local policy allows bounded repo artifact disclosure.
3. Skeptically audit for target drift, exact-Kalman substitution, hidden
   stopped partials, and proxy admission.

Implementation tests:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_compact_transport_jvp.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Additional tests to add:

- tiny deterministic LGSSM reduce-only score equals current compact
  forward-sensitivity score within existing tolerance;
- same-scalar FD agreement on a tiny deterministic LGSSM case;
- source/runtime sentinels reject `GradientTape`, `ForwardAccumulator`, stopped
  partials, historical `manual_total_vjp`, and exact Kalman score substitution;
- the reduce-only full-scale path cannot be admitted from score-only diagnostics.

GPU rungs:

1. Trusted GPU `N=256,T=3` score-only reduce-only smoke.
2. Trusted GPU `N=1000,T=10` score-only reduce-only diagnostic.
3. Trusted GPU `N=10000,T=50` single-seed score-only reduce-only diagnostic.
4. Only if rung 3 emits under budget: return to score+FD aggregation flow.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the LEDH score be computed for the same finite-`N` scalar without carrying full `batch x N x state_dim x param_dim` particle sensitivities at full scale? |
| Baseline/comparator | Current compact forward-sensitivity route on tiny deterministic cases; same-scalar FD on tiny cases; Phase shared-kernel full-scale no-artifact blocker. |
| Primary engineering criterion | Full LGSSM single-seed `N=10000,T=50` score-only reduce-only diagnostic emits under the reviewed memory budget. |
| Correctness criterion | Reduce-only tiny outputs match current compact route and same-scalar FD. |
| Admission criterion | No admission until full fixed-seed score+FD aggregation validates with `validate_ledh_score_artifact(..., require_admitted=True)`. |
| Veto diagnostics | Target drift; changed seeds or row identity; exact Kalman score substitution; production autodiff; stopped partials; FD mismatch; nonfinite score; memory over budget; no artifact. |
| Explanatory diagnostics | Runtime, peak memory, tensor shapes, checkpoint count, chunk sizes, and whether final `d_transported` is avoided at full scale. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, runtime ranking, or cross-model admission. |

## Candidate Technical Directions

Prioritize in this order:

1. Reverse accumulation through the time loop with bounded checkpointing of
   value states only, using existing no-tape transport VJP primitives for
   transport adjoints and accumulating scalar parameter cotangents. This avoids
   a full parameter axis on all particles.
2. Direction-by-direction score reconstruction only as a diagnostic fallback;
   it may reduce peak memory but likely multiplies runtime by `param_dim`, so it
   is not preferred for full admission unless the reverse recurrence is blocked.
3. Model-specific analytical LGSSM-only recurrence only if it still
   differentiates the finite LEDH estimator and does not substitute Kalman
   likelihood; this must not be called a general cross-model score repair.

## Forbidden Claims And Actions

- Do not claim score admission from score-only artifacts.
- Do not replace LEDH finite estimator score with exact Kalman score.
- Do not change row identity, target scalar, output field, seeds, `N`, `T`,
  transport settings, parameter order, or admission criteria after seeing
  results.
- Do not use production `GradientTape`, `ForwardAccumulator`, stopped partials,
  or historical `manual_total_vjp*` evidence.
- Do not proceed to other models until LGSSM has either an admitted artifact or
  a reviewed blocker that explicitly hands off.

## Exact Next-Phase Handoff Conditions

The next score admission phase may start only if:

- reduce-only derivation is reviewed;
- tiny equivalence and FD tests pass;
- full single-seed LGSSM score-only emits under budget;
- a result record and next subplan are written and reviewed.

If any condition fails, write a blocker result with the smallest next repair.

## Stop Conditions

Stop if:

- the recurrence cannot be shown to compute the same finite LEDH scalar;
- tiny equivalence to the compact route fails;
- same-scalar FD fails;
- full score-only reduce-only rung exceeds memory or emits no artifact;
- local review does not converge after five rounds;
- continuing requires package installation, network/data fetches, credentials,
  destructive git actions, or changing pass/fail criteria after seeing results.

## Skeptical Audit Before Execution

- Wrong baseline checked: exact Kalman is not the target score.
- Proxy criterion checked: score-only emission is not admission.
- Hidden assumption checked: avoiding `d_transported` is necessary but not
  sufficient; potential tangents/checkpoints must also be bounded.
- Environment checked: GPU rungs require trusted execution.
- Artifact sufficiency checked: the result must record emitted artifacts or the
  precise no-artifact/memory blocker.

Audit status: `READY_FOR_REVIEW_BEFORE_IMPLEMENTATION`.
