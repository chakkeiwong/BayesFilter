# Phase 2 Subplan: Generic Candidate Value and Total-JVP Core

Date: 2026-07-21

Status: `READY_EXECUTION`

## Question

Can the candidate route expose one model-independent finite value program and
its complete recursive forward sensitivity without TensorFlow autodiff or
runtime finite differences?

## Scope

This phase implements only the adapter contract and a generic candidate core.
It uses a toy nonlinear model for validation. It does not wire actual SV,
KSC-SV, predator-prey, generalized SV, or Austria SIR into the leaderboard.

## Evidence Contract

| Item | Contract |
|---|---|
| Primary criterion | Candidate value and recursive score are the derivative of the same finite program on fixed randomness and fixed design. |
| Baseline | Same candidate value callable evaluated by central finite differences at a small diagnostic fixture. |
| Hard vetoes | Nonfinite output, value/score mismatch, missing tangent dependency, runtime autodiff/FD, reset/marginal failure, or changed canonical route. |
| Explanatory diagnostics | Per-time increments, score increments, reset residuals, Sinkhorn residuals, and toy-model wall time. |
| Pass boundary | Toy nonlinear adapter passes all coordinates and replay; generic route remains candidate-only. |
| Nonclaims | No nonlinear production-row validity, full-horizon feasibility, XLA readiness, default promotion, or leaderboard admission. |

## Skeptical Audit Before Execution

- The toy model must use fixed stateless noise and a fixed residual design so
  FD compares the same finite scalar rather than a resampled estimator.
- The adapter must expose both value and tangent callbacks; a value-only model
  is not sufficient evidence for total-score correctness.
- The reset must include source moments/weights and transport dependencies; a
  transported-cloud-only derivative is not admissible.
- This phase uses a static Python time loop as a diagnostic implementation.
  Phase 5 must replace or validate it with staged/loop-native XLA before any
  default claim.

## Artifacts And Stops

- Implementation: `bayesfilter/highdim/cubature_genut_filter.py`;
- Tests: `tests/highdim/test_cubature_genut_filter.py`;
- Result: `docs/plans/bayesfilter-cubature-genut-nonlinear-default-phase2-result-2026-07-21.md`.

Stop if the generic core cannot pass the toy same-scalar derivative gate, or if
the implementation requires modifying the canonical Contract E route.
