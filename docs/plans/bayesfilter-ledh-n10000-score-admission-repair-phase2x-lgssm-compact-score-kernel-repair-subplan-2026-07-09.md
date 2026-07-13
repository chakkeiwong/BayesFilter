# Phase 2X Subplan: LGSSM Compact Score Kernel Repair

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Repair the full-scale compact score pass itself. Phase 2W showed that even a
single score-only `N=10000,T=50` shard does not emit within the bounded window,
so the next phase must reduce tensor lifetime or compute the required score
without materializing full transported-particle tangent arrays.

## Entry Conditions Inherited From Previous Phase

- Phase 2V implemented and tested exact seed-sharded aggregation.
- Phase 2W implemented and tested score/FD split diagnostics.
- Trusted GPU score-only and FD-only smokes passed at `N=256,T=3`.
- Full trusted GPU score-only seed `81120` at `N=10000,T=50` did not emit an
  artifact after about 6.75 minutes.
- No LGSSM score artifact is admitted yet.

## Required Artifacts

- Phase 2X result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2x-lgssm-compact-score-kernel-repair-result-2026-07-09.md`
- Implementation diff in one or both of:
  - `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`;
  - `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`.
- Focused tests in:
  - `tests/test_ledh_lgssm_manual_score_phase4.py`;
  - `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`;
  - optionally `tests/test_ledh_compact_transport_jvp.py`.
- Trusted GPU diagnostic logs:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2x-*.log`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-n10000-score-admission-repair-phase2w-result-phase2x-subplan-review-bundle-2026-07-09.md`

## Required Checks, Tests, And Reviews

Before implementation:

- Review this subplan after the Phase 2W result.
- If Claude is blocked by local policy, use fresh Codex packet-only review and
  record the limitation.

Implementation checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_compact_transport_jvp.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Focused tests to add or refresh:

- new reduced/tensor-lifetime-repaired score path matches the current compact
  score on tiny deterministic LGSSM cases;
- new path matches same-scalar FD on tiny deterministic cases;
- source audit still finds no `GradientTape`, `ForwardAccumulator`, stopped
  partials, or historical `manual_total_vjp` admission path;
- if transport JVP internals change, existing compact transport JVP tests still
  pass;
- new path cannot admit without score correctness and memory diagnostics.

Trusted GPU rungs:

1. `N=256,T=3` score-only smoke under the new path.
2. `N=1000,T=10` score-only diagnostic.
3. `N=10000,T=50` single-seed score-only diagnostic.

Do not run later rungs if an earlier rung fails to emit or fails correctness.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the compact no-tape LGSSM score pass be made to emit at full single-seed `N=10000,T=50` scale without changing the target scalar? |
| Baseline/comparator | Phase 2W full score-only single-shard no-artifact blocker and current compact tiny/smoke correctness. |
| Primary criterion | Full single-seed `N=10000,T=50` score-only artifact emits under trusted GPU, with finite compact score, GPU runtime gate, and bounded memory diagnostics. |
| Correctness criterion | New/repaired path matches current compact score and same-scalar FD on small deterministic cases. |
| Admission criterion | Still no full admission until FD correctness and all fixed seeds aggregate through `validate_ledh_score_artifact(..., require_admitted=True)`. |
| Veto diagnostics | Target drift; score differs from current compact tiny reference beyond tolerance; FD mismatch; non-GPU score output; memory above budget; no artifact; historical route; tape/autodiff; stopped partial derivative; hidden exact-Kalman substitution. |
| Explanatory diagnostics | Tensor shapes, peak memory, runtime per rung, chunk sizes, and whether tensor stack sites were removed or made reduce-only. |
| Not concluded | HMC readiness, posterior correctness, exact Kalman score equality, runtime ranking, or non-LGSSM admission. |

## Candidate Repair Directions

Prioritize the smallest correct repair:

1. Add a reduce-only LGSSM score mode that carries particle tangents through the
   filter but avoids storing/stacking extra transport JVP blocks beyond the next
   particle/tangent state.
2. If the blocker is inside transport JVP stacking, replace TensorArray stack
   patterns in `_filterflow_streaming_transport_from_potentials_jvp` with a
   chunked write/consume route, preserving exact output.
3. If full tangent propagation remains too expensive, derive a score-only
   recurrence that accumulates the score contribution and necessary tangent
   state with smaller lifetime, but still differentiates the same realized
   finite-`N` LEDH scalar.

Any repair that changes the scalar or replaces the LEDH finite estimator with
exact Kalman likelihood is forbidden.

## Forbidden Claims And Actions

- Do not claim LGSSM admission from tiny/smoke tests.
- Do not admit score-only artifacts.
- Do not change target scalar, seeds, `N`, `T`, parameter order, source value
  artifact, transport policy, Sinkhorn settings, production TF32 policy, or
  disclosed no-TF32 FD policy.
- Do not use `GradientTape`, `ForwardAccumulator`, stopped partials, or
  historical `manual_total_vjp*` evidence.
- Do not silently move to later model phases while LGSSM remains blocked.

## Exact Next-Phase Handoff Conditions

Phase 2Y or Phase 3 may begin only if:

- Phase 2X writes a result record;
- ledger is updated;
- either full single-seed score-only emission is repaired and the next subplan
  returns to full fixed-seed score/FD aggregation, or Phase 2X writes a precise
  blocker with the next smallest kernel repair;
- the next subplan is reviewed.

## Stop Conditions

Stop if:

- the repaired path cannot be shown to compute the same compact score on tiny
  deterministic cases;
- trusted GPU rungs fail to emit;
- memory exceeds the reviewed budget;
- local tests fail and cannot be fixed without changing the contract;
- review does not converge after five rounds;
- continuing requires package installation, network/data fetches, credentials,
  destructive git actions, or changing pass/fail criteria after seeing results.

## Skeptical Audit Before Execution

- Wrong baseline checked: exact Kalman likelihood is not the score target.
- Proxy metric checked: full score-only emission is diagnostic, not admission.
- Hidden assumption checked: any reduced path must match current compact score
  on small deterministic cases before full-scale use.
- Environment checked: GPU rungs require trusted execution.
- Artifact sufficiency checked: every rung writes logs and durable artifacts or
  a precise blocker.

Audit status: ready for read-only review before implementation.
