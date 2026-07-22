# Phase 4 Subplan: Exact Adapter And Trainer Integration

Date: 2026-07-14

## Phase Objective

Expose the admitted batch materializer/kernel/prior through the exact LGSSM
adapter's fixed `neutra_batch_log_prob_and_grad_status` method, allow the
repository binding factory to inspect and bind it, and prove the generic trainer
calls it without changing scalar HMC/parity methods.

## Entry Conditions Inherited From Phase 3

- Batch materialization and SVD/eigh graph-status kernel pass same-regime scalar
  parity and per-row isolation gates.
- Current trainer already fails closed before side effects when the method is
  absent or ineligible.
- Scalar adapter methods remain compatibility and parity authority.

## Required Artifacts

- Focused exact-adapter method in
  `bayesfilter/testing/deterministic_lgssm_exact_target_tf.py`.
- Binding and trainer integration tests.
- Phase 4 result and reviewed Phase 5 certification subplan.

## Integration Contract

The method must accept rank-2 `[B,18]` only, call the batch materializer, batch
SVD/eigh kernel, and batch prior directly, and return value `[B]`, score
`[B,18]`, and all required status fields `[B]`. It must apply the scalar target's
same final rule: invalid/nonfinite rows return NaN value/score and
`valid_pre_regularized_score=False`.

The method must not call another adapter method, map rows, provide a scalar
fallback, or accept caller-stamped capability metadata.

## Required Checks

1. Binding factory accepts the real exact adapter and records bound source.
2. Direct batch method matches scalar rows in eager and same-regime CPU-XLA.
3. Source audit and graph audit show no sample map/loop or callback.
4. Existing scalar exact-adapter identity, legacy parity, and XLA tests pass
   unchanged.
5. One-step actual exact-target trainer integration runs through the binding and
   emits batch provenance; if CPU cost is excessive, a compiled objective/update
   probe may replace artifact writing but must still use the real adapter.
6. Missing/forged/detached binding tests remain passing.
7. Python compile and `git diff --check`.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Question | Is the exact batch callable correctly bound and consumed without scalar fallback or scalar-API drift? |
| Pass criterion | Direct parity, binding, scalar regression, and real trainer integration pass. |
| Hard veto | Binder rejection, adapter delegation/map, status/NaN mismatch, scalar regression, fallback, or output before binding failure. |
| Explanatory only | CPU compile/runtime. |
| Artifact | Tests and Phase 4 result. |
| Nonclaims | Integration does not establish trusted GPU performance, target-specific recipe quality, HMC readiness, or scientific validity. |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Keep scalar map methods | active HMC/parity compatibility | callers may confuse them with training route | binder fixes a different method name | frozen compatibility |
| Direct batch method source | Phase 0 source audit | helper delegation could hide scalar mapping | binder inspection plus source test | reviewed requirement |
| Adapter signature remains target identity | current strict harness | new capability could be detached from identity | binding adds callable source hash | reviewed compatibility choice |

## Skeptical Subplan Audit

- Identity drift: target signature and scalar adapter signature remain stable;
  batch binding separately hashes the actual method source.
- Proxy promotion: method existence is insufficient; real invocation, parity,
  graph, and trainer integration are required.
- Hidden fallback: the binding's direct-source audit rejects adapter delegation
  and mapping primitives.
- Phase leakage: no trusted GPU performance or recipe selection occurs.

Audit verdict: **PASS**. The integration boundary is narrow and retains
independent scalar authority.

## Forbidden Claims And Actions

- Do not delete or redirect scalar HMC methods.
- Do not change target or adapter signatures merely to advertise batching.
- Do not add target-specific trainer branches or a scalar fallback.
- Do not run serious training or infer GPU speed.

## Exact Next-Phase Handoff Conditions

Phase 5 starts after the real adapter binds, same-regime parity passes, scalar
regressions pass, and a real adapter reaches at least one compiled optimizer
update through the generic trainer.

## Stop Conditions

Stop only for target identity/status ambiguity, scalar compatibility breakage,
or an unrepairable binding/integration mismatch. Compile or harness failures
trigger local repair.

## Phase-End Procedure

1. Run required local checks.
2. Write Phase 4 result/close record.
3. Draft or refresh Phase 5 certification subplan.
4. Review Phase 5 suitability and continue when no real blocker exists.

