# Phase 5 Subplan: Batched Analytical Score Contract

Date: 2026-07-09

## Phase Objective

Define the exact batch-native analytical QR score API, shape contract,
limitations, and reference baselines before implementation.

## Entry Conditions Inherited From Previous Phase

- Dtype-polymorphic scalar QR value and analytical score paths pass.
- Benchmark artifacts can verify requested/observed dtype.
- Existing batched-static QR value path is available as a value/autodiff
  comparator.

## Required Artifacts

- Contract note or API doc in the Phase 5 result.
- Tests or source-contract checks that distinguish true batch-native score from
  scalar wrappers.
- Refreshed Phase 6 implementation subplan.

## Required Checks, Tests, And Reviews

Run:

```bash
git diff --check -- docs/plans tests bayesfilter/linear
```

Read-only review is required for the contract before implementation.  Because
the Claude review gate is unavailable for this run unless separately approved,
use a fresh bounded Codex substitute review and label it weaker than Claude
review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the batch-native analytical score contract precise and implementable without confusing batch size `B` and parameter dimension `P`? |
| Baseline/comparator | Existing scalar analytical score and batched-static QR value/autodiff gradients. |
| Primary criterion | Contract states inputs `[B, ...]`, derivative tensors `[B, P, ...]`, outputs `[B]` and `[B, P]`, dtype behavior, time-invariant limitation, and scalar/autodiff references. |
| Veto diagnostics | Contract permits `tf.vectorized_map` scalar wrapper as final kernel, leaves dtype ambiguous, or lacks parity baseline. |
| Explanatory diagnostics | Planned shape examples and source-contract checks. |
| Not concluded | Implementation correctness or performance. |
| Artifact | Phase 5 result and reviewed Phase 6 subplan. |

## Forbidden Claims And Actions

- Do not implement production batch-native score before contract review.
- Do not treat vectorized scalar score as the optimized target.
- Do not broaden to time-varying tensors unless explicitly reviewed.

## Exact Next-Phase Handoff Conditions

Advance to Phase 6 only if the contract is reviewed and names exact source,
test, and fallback comparator paths.

## Stop Conditions

Stop if shape/dtype contract cannot be made unambiguous or if implementation
would require a new mathematical derivation not covered by existing scalar
score equations.
