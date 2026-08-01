# Phase 5 Subplan: Batch-Native Correctness Certification

Date: 2026-07-14

## Phase Objective

Repair batch capability dependency identity and certify the complete exact
adapter-to-optimizer path across numerical, status, graph, reproducibility, and
policy ledgers before any trusted GPU performance work.

## Entry Conditions Inherited From Phase 4

- Exact batch method binds and reaches a real optimizer update.
- Scalar target identity and HMC/parity methods remain unchanged.
- Direct method source identity is present, but repository helper dependency
  closure is a known provenance blocker.
- Trusted GPU performance has not been measured.

## Required Artifacts

- Repository-derived dependency-closure fields in `NeuTraBatchTargetBinding`.
- Adversarial tests proving helper source affects closure identity and callers
  cannot forge or detach it.
- Certification tests/result covering the full Phase 5 matrix.
- Reviewed Phase 6 performance/repair subplan.

## Dependency-Closure Contract

The binder must inspect direct global function calls in the bound method,
select repository-owned `bayesfilter.*` callables, hash their owning module
sources, sort/deduplicate the module records, and hash the normalized closure.
The adapter cannot provide or override these values. TensorFlow/library version
identity remains environment provenance, not repository source closure.

## Required Checks

1. Dependency closure contains the batch materializer and batch SVD kernel
   modules and changes if an inspected helper/module source changes.
2. Forged, detached, mapped, delegated, non-XLA, and singleton routes remain
   rejected.
3. Eager/eager and CPU-XLA/CPU-XLA exact value, score, and status parity over
   multiple rows.
4. Mixed invalid-row NaN/status behavior and row isolation.
5. Objective gradient versus finite difference on an exact-target flow weight.
6. One-step update and five-step deterministic state equality across fresh
   output roots with identical seeds.
7. Graph inventory: training time loop plus target time loop, no sample map or
   host callback; target/optimizer invocation counts recorded.
8. No NumPy/Python sample loop in active algorithmic/training modules.
9. Existing scalar, binding, training, and kernel suites.
10. Python compile and `git diff --check`.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Question | Is the full batch-native exact-target training path reproducibly identified and correct before performance work? |
| Pass criterion | Dependency identity, numerical/status parity, gradients, deterministic updates, graph/policy, and regressions pass. |
| Hard veto | Unbound helper drift, scalar mismatch, status/NaN mismatch, nondeterministic same-seed state, map/callback fallback, missing XLA, or broken scalar API. |
| Explanatory only | CPU compile/runtime and loss trajectory. |
| Artifact | Tests and Phase 5 result. |
| Nonclaims | CPU certification does not establish trusted GPU speed, transport quality, posterior correctness, HMC convergence, recipe ranking, or scientific validity. |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Whole owning-module source hashes | simple repository closure mechanism | unrelated edits in module change identity | expected conservative invalidation test | reviewed reproducibility choice |
| Direct global calls only | bound method is intentionally thin | deeper helper dependency changes may be missed if not in hashed owning module | helpers live in whole hashed modules; cross-module direct calls audited | reviewed bounded closure |
| Five-step batch-2 CPU smoke | smallest real deterministic training check | too small for transport behavior | explicitly engineering-only | diagnostic baseline |
| Same-regime numerical parity | Phase 3 execution-order finding | mixed eager/XLA comparisons misdiagnose rounding | preserve separate eager and XLA ledgers | reviewed evidence rule |

## Skeptical Subplan Audit

- Identity theater risk: closure is derived by the repository binder, not
  supplied by the adapter, and adversarial mutation is tested.
- Proxy promotion: five steps prove state/reproducibility mechanics only.
- Missing downstream check: objective gradient and optimizer state are both
  included, not inferred from target parity.
- Environment mismatch: CPU-hidden XLA is correctness evidence only; Phase 6
  requires trusted GPU evidence.
- Hidden default: smoke geometry and batch size two are labeled diagnostic and
  cannot nominate a training recipe.

Audit verdict: **PASS**. The phase directly repairs the only material Phase 4
provenance gap and provides the last correctness gate before bounded GPU work.

## Forbidden Claims And Actions

- Do not alter scalar target identity or promote CPU timing.
- Do not treat five-step loss as transport-quality evidence.
- Do not start the batch-size/performance ladder before all correctness vetoes
  pass.
- Do not weaken source audits or parity tolerances to make the phase pass.

## Exact Next-Phase Handoff Conditions

Phase 6 starts only when all certification ledgers pass and its subplan defines
trusted GPU commands, warmup/timing separation, batch rungs, parity rechecks,
repair order, versioned outputs, and the 45-minute aggregate GPU budget.

## Stop Conditions

Stop only for an unrepairable identity, mathematical, status, deterministic
state, or XLA mismatch. Test/harness defects trigger focused repair.

## Phase-End Procedure

1. Run required local checks.
2. Write Phase 5 result/close record.
3. Draft or refresh Phase 6 performance subplan.
4. Review Phase 6 suitability and continue when no real blocker exists.

