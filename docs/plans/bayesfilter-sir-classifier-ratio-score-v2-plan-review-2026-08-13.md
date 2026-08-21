# Skeptical Review: Classifier-Ratio Score V2

Date: 2026-08-13  
Reviewed plan: `bayesfilter-sir-classifier-ratio-score-v2-plan-2026-08-13.md`  
Verdict: `PASS_WITH_IMPLEMENTATION_TESTS_REQUIRED`

## Audit Findings

| Risk | Finding | Disposition |
|---|---|---|
| Wrong baseline | V1 selected a nonlinear head across incompatible location and scale tasks; V2 restores convex linear and quadratic baselines | repaired |
| Proxy promoted to criterion | Validation loss selects controls only; exact score error on untouched data remains primary | pass |
| Missing stop condition | Any of nine exact cells failing blocks SIR; no V2 scientific retry is authorized | pass |
| Unfair comparison | All candidates receive identical full paths, splits, optimizer budget, calibration, and tests | pass |
| Hidden assumption | Coordinate-specific tuning and zero initialization are explicitly hypotheses with failure diagnostics | pass |
| Stale context | V1 failures and its one repair are preserved and cited; rejected Fisher artifacts remain excluded | pass |
| Environment mismatch | Direct `tftwogpu`, GPU/XLA, memory growth, TF32-off, and wrapper provenance remain required | pass |
| Artifact cannot answer question | Per-head rows plus exact-score intercept/error and loaded-module audit directly answer the oracle gate | pass |

## Mathematical Review

Balanced class odds still identify

`log p(theta+epsilon e_j,y) - log p(theta-epsilon e_j,y)`.

Every candidate produces only a learned classifier logit. Centered quadratic
features expand the classifier function class but do not supply a density,
likelihood, score, latent state, or exact coefficient. Dividing the calibrated
logit by `2*epsilon` is therefore unchanged. Coordinate-specific classifier
selection changes finite-sample estimation, not the mathematical target.

## Required Pre-Run Tests

1. convex classifier heads initialize every kernel and bias at exactly zero;
2. the quadratic logistic head exposes `[z,z**2-1]` and no hidden layer;
3. selection keys include horizon and coordinate and never final domain;
4. exact and SIR stages use the same candidate names and fitting API;
5. source/runtime dependency vetoes and the exact score-expression test remain;
6. all existing focused tests pass deliberately CPU-only before trusted GPU
   execution.

## Review Verdict

The revision addresses an observed tuning/optimization defect without changing
the requested filter-independent estimator or weakening evidence gates. It is
proportionate and answers the same research question. Execute only after the
required tests pass. A V2 exact-oracle failure is a terminal campaign result,
not permission for another silent architecture, epsilon, or threshold change.

## Attempt 01 Conformance Failure

Post-run source audit found that `linear_full_path` still used Glorot
initialization although V2 froze zero initialization for both convex heads. The
pre-run test checked only `linear_full_path_quadratic`, so the test suite did not
enforce required item 1 above. This invalidates attempt 01 as evidence for V2;
it is an implementation/test conformance failure, not a failed execution of the
reviewed scientific protocol.

Repair both the source and the parametrized test, preserve attempt 01 as invalid
evidence, rerun the focused suite, and execute one replacement exact-oracle run
in a fresh output directory. This correction does not add a candidate, change a
split, threshold, epsilon, optimizer, target, or score formula. It is not the
scientific retry forbidden by the verdict above.

## Replacement Attempt 02 Result

The replacement run used the repaired source and parametrized zero-
initialization tests. Runtime dependency audit passed with no forbidden module
loaded. It still failed the exact-oracle gate: 43 of 108 heads passed all
head-level gates, but only `T=20` log-scale reached three admitted epsilons and
passed its cell tolerance. The other eight cells did not reach three admitted
epsilons. Failure counts overlap and were Platt slope `42`, ECE `30`, signal
`17`, and AUC range `7`.

This is a valid V2 scientific failure, not permission for another unreviewed
repair. SIR execution is vetoed.
