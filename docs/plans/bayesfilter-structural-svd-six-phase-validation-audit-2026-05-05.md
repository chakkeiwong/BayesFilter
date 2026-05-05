# Audit: structural SVD six-phase validation plan

## Date

2026-05-05

## Plan reviewed

- `docs/plans/bayesfilter-structural-svd-six-phase-validation-plan-2026-05-05.md`

## Audit stance

Pretending to be another developer, I audited whether the plan actually closes
the right phase goals rather than blindly following the untracked handoff.

## Findings

1. The six-phase plan correctly treats the untracked handoff as partially stale.
   BayesFilter already has structural metadata, sigma-point, particle,
   degenerate Kalman, factor-backend, and spectral derivative gates.
2. The plan is valid because it validates existing code before proposing
   changes.  That avoids unnecessary rewrites of the structural SVD reference
   backend.
3. The dependency order is sound:
   hygiene -> BayesFilter core -> DSGE metadata -> blocker reconciliation ->
   final validation -> commit.
4. The plan correctly separates adapter-readiness from model-specific
   structural promotion.  Rotemberg/SGU completion bridges are not enough to
   claim final nonlinear structural correctness.
5. The plan correctly keeps SVD/eigen derivative, JIT, and HMC promotion out of
   scope for this pass.
6. The plan has adequate tests for the phase goals: structural core tests,
   exact Kalman tests, DSGE adapter tests, client metadata contracts, full
   pytest, source-map parse, diff check, and LaTeX.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Client `/home/chakwong/python` tests fail for environment reasons | Record as client-owned blocker unless BayesFilter tests fail. |
| Blocker register over-promotes DSGE metadata | Use `client_metadata_ready_for_structural_tests`, not `filter-correct`. |
| Untracked PDF/templates accidentally staged | Stage explicit paths only. |
| Full pytest is slower or warns via TensorFlow Probability | Treat warnings separately from failures. |
| Handoff plan remains untracked | Either leave it untracked as external handoff input or explicitly stage only if user wants it recorded. |

## Decision

Execute the six-phase plan.  Continue automatically through V5 if tests pass.
Stop and ask for direction only if:

- a tracked BayesFilter code file is unexpectedly dirty before execution;
- BayesFilter structural core tests fail and cannot be fixed locally;
- DSGE metadata tests fail in a way that contradicts the expected current
  client commit;
- final validation fails after a scoped fix attempt.
