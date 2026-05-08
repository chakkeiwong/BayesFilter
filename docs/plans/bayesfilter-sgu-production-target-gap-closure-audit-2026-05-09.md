# Audit: SGU production-target gap closure plan

## Date

2026-05-09

## Plan Under Audit

`docs/plans/bayesfilter-sgu-production-target-gap-closure-plan-2026-05-09.md`

## Audit Stance

Pretend the plan was written by another developer.  Check whether it can be
executed without premature production promotion, hidden BayesFilter economics,
missing stop rules, or tests that fail to answer the current SGU blocker.

## Verdict

Approved as a gated diagnostic and decision plan.

The plan correctly identifies the remaining blocker: residual-closing causal
anchors exist, but no tested anchor satisfies the locality gate needed for a
production predictive filtering target.  It keeps BayesFilter backend/filter,
derivative/JIT, and HMC work downstream of a model-owned causal value target.

## Required Guardrails

1. The Phase 1 derivation must name equations and variables explicitly.  It
   cannot merely repeat that `static_foc` is closest.
2. Phase 2 may design candidates, but must not relax the `2e-2` locality gate
   after seeing diagnostics unless a scale-aware derivation is written first.
3. Phase 3 helpers must stay diagnostic unless a candidate passes residual,
   locality, timing, rank, and stress gates.
4. Two-slice projection must keep:

```text
sgu_two_slice_projection_diagnostic_passed
blocked_sgu_two_slice_projection_not_filter_transition
```

5. If no causal production target passes, BayesFilter code changes are limited
   to planning/provenance/reset-memo updates.
6. Derivative, compiled, and HMC phases must be recorded as blocked gate
   decisions unless a production value target exists first.
7. The final commit must stage only scoped SGU planning/provenance/client
   artifacts.  Existing QR/linear derivative files and Windows
   `Zone.Identifier` files are out of scope for this SGU pass.

## Missing Points Found

No blocking missing point was found.

Two non-blocking clarifications are required during execution:

- Phase 1 should record per-equation residual and sensitivity evidence for
  `H[3]` through `H[6]`.
- Phase 4 should state explicitly whether a production API was added or
  deliberately skipped.

## Stop Rules

Stop before production API work if Phase 1 or Phase 2 cannot define a causal
candidate with predeclared locality semantics.

Stop before BayesFilter backend/filter changes if Phase 4 does not produce a
causal production target.

Stop before derivative/JIT/HMC work unless the same model/backend pair has
value evidence, derivative evidence, and compiled parity evidence in that
order.
