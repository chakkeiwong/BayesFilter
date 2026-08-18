# Experiment plan: actual-SV dense Gaussian-mixture refinement vs Zhao-Cui

## Question
Is the observed gap between the Zhao-Cui transformed-likelihood approximation and the dense Gaussian-mixture / Kalman approximation mainly due to finite Gaussian-mixture approximation budget, or does it persist after refining the mixture enough that a bug or approximation-family bias becomes the stronger explanation?

## Mechanism being tested
We compare two approximations to the same actual-SV likelihood after exact log-square transformation and exact Jacobian correction back to the raw-`y` likelihood representation:
- Zhao-Cui Lane A transformed-likelihood approximation.
- Dense Gaussian-mixture / Kalman approximation to the transformed observation model.

For the dense Gaussian-mixture route, we will refine the approximation family by replacing the KSC 7-component log-chi-square mixture with fitted Gaussian mixtures of 7, 14, and 28 components to the exact `log(\epsilon^2)` density. Then we will recompute value and score in the raw-`y` representation and measure whether the dense-Kalman approximation changes materially.

## Scope
- Variant: actual-SV tiny deterministic fixtures, dimensions 1/2/3.
- Objective: assess whether the dense Gaussian-mixture approximation is already converged enough for comparison to Zhao-Cui.
- Seed(s): the fixed tiny actual-SV fixtures from the existing p43 tests.
- HMC/MCMC settings: none.
- XLA/JIT mode: CPU-only diagnostic run.
- Expected runtime: a focused offline comparison under a few minutes.

## Success criteria
- Build 7 / 14 / 28-component Gaussian-mixture approximations to `log(\epsilon^2)`.
- Recompute the dense Kalman approximation in the same raw-`y` likelihood representation.
- If moving from 7 -> 14 or 14 -> 28 components changes the dense-Kalman likelihood or score by less than 1%, then the Gaussian-mixture approximation is empirically stable at that budget.
- Record whether the Zhao-Cui vs dense-Kalman gap shrinks materially under refinement.

## Diagnostics
Primary:
- raw-`y` log-likelihood values for Zhao-Cui and dense-Kalman at component counts 7 / 14 / 28,
- relative likelihood change of dense-Kalman under 7 -> 14 -> 28 refinement,
- relative score change of dense-Kalman under 7 -> 14 -> 28 refinement,
- Zhao-Cui vs dense-Kalman gap after each refinement.

Secondary:
- fitted mixture quality to the exact `log(\epsilon^2)` density,
- whether the score gap is much smaller than the value gap,
- monotonicity / stabilization behavior of the dense route under refinement.

Sanity checks:
- exact Jacobian correction is applied identically at each component count,
- no route semantics change in Zhao-Cui,
- no comparison is made in mismatched transformed/raw representations.

## Expected failure modes
- The fitted 14/28-component mixtures may not materially improve the approximation because the dominant error comes from another part of the Kalman approximation.
- The dense-Kalman route may stabilize quickly but remain biased relative to Zhao-Cui, suggesting approximation-family bias rather than insufficient mixture resolution.
- If refinement changes the dense route erratically, that would increase suspicion of an implementation bug.

## What would change our mind
- If dense-Kalman changes by less than 1% under refinement but remains far from Zhao-Cui, then the main issue is probably not mixture resolution.
- If the gap shrinks materially under refinement, then the current value difference is likely a finite approximation-budget issue, not a bug.
- If refinement causes unstable or contradictory behavior, then we should suspect a bug or a deeper numerical issue.

## Command
```bash
# Fit 7/14/28-component Gaussian mixtures to log(epsilon^2),
# rerun dense Kalman in transformed space, convert back to raw-y,
# and compare values/scores to Zhao-Cui.
```

## Interpretation rule
- If the dense-Kalman approximation changes by <1% under 2x/4x mixture refinement, treat it as empirically converged at this fixture.
- If Zhao-Cui remains several percent away after convergence, that is stronger evidence of approximation-family bias than of a mixture-budget problem.
- If convergence fails or behavior is erratic, investigate for bugs.
