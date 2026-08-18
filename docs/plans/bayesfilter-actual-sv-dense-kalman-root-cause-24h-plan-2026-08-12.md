# Experiment plan: actual-SV dense-Kalman vs Zhao-Cui 24-hour root-cause campaign

## Question
Why does the likelihood gap between the Zhao-Cui transformed-target approximation and the transformed-back dense Gaussian-mixture / Kalman approximation remain several percent on the tiny actual-SV fixtures, even though the score gap is much smaller?

## Mechanism being tested
We compare two approximations to the same actual-SV likelihood after exact log-square transformation and exact Jacobian correction back to the raw-`y` likelihood representation:
- Zhao-Cui Lane A transformed-likelihood approximation.
- Dense Gaussian-mixture / Kalman approximation to the transformed observation model.

We then refine the Gaussian-mixture approximation budget (7 / 14 / 28 components) and test whether the dense-Kalman value and score stabilize to within 1%. This lets us distinguish:
1. dense-mixture approximation budget problem,
2. Zhao-Cui approximation problem,
3. approximation-family bias,
4. bug.

## Budget / campaign authorization
This campaign is authorized to use up to 24 hours of compute and analysis time.

The campaign should prefer the cheapest discriminating artifact first:
- value refinement before score refinement,
- only escalate to the score ladder if the value ladder does not already isolate the root cause.

## Scope
- Fixtures: tiny deterministic actual-SV fixtures, dimensions 1/2/3.
- Comparison objects:
  - Zhao-Cui transformed-target approximation,
  - Zhao-Cui dense same-target transformed reference,
  - dense Gaussian-mixture / Kalman approximation with 7 / 14 / 28 components.
- Runtime target: CPU-only unless a later justified speedup is required.

## Hypotheses
### H1. Dense-mixture budget problem
The dense Kalman approximation is still under-resolved. Increasing the Gaussian-mixture component count from 7 to 14 to 28 will materially change the likelihood and score, and the Zhao-Cui gap will shrink.

### H2. Zhao-Cui approximation problem
Zhao-Cui is itself materially biased relative to its own dense same-target transformed reference, so part of the cross-method gap is attributable to Lane A.

### H3. Approximation-family bias
Both methods are internally coherent, but they converge to materially different finite approximations at these budgets, and the dense-Kalman approximation stabilizes quickly while remaining far from Zhao-Cui.

### H4. Bug
One of the routes behaves erratically under refinement or violates internal consistency checks.

## Success criteria
- Compute the transformed-back raw-`y` likelihoods for Zhao-Cui and dense-Kalman at 7 / 14 / 28 mixture components.
- Measure the relative dense-Kalman change from 7->14 and 14->28.
- If both relative changes are <1% in value, treat the dense-mixture approximation as empirically converged in value on these fixtures.
- Compute the score ladder only if the value ladder does not already settle the diagnosis.
- Produce a result note naming the most likely root cause with explicit evidence.

## Diagnostics
Primary:
- Zhao-Cui vs Zhao-Cui dense same-target value gap,
- dense-Kalman raw-`y` likelihood values for 7 / 14 / 28 components,
- relative dense-Kalman value change from 7->14 and 14->28,
- Zhao-Cui vs dense-Kalman raw-`y` value gap after each refinement level.

Secondary:
- score gap between Zhao-Cui and dense-Kalman if needed,
- relative dense-Kalman score change from 7->14 and 14->28 if needed,
- whether score gaps remain much smaller than value gaps,
- any instability or non-monotone refinement behavior.

## Execution order
1. Reconfirm the transformed-back comparison convention.
2. Compute Zhao-Cui vs its own dense same-target gap.
3. Run 7 / 14 / 28 dense-Kalman value ladder in raw-`y` representation.
4. If dense-Kalman value changes are both <1%, stop the value ladder and interpret the result.
5. Only if value refinement is ambiguous, run the 7 / 14 / 28 score ladder.
6. Write a result note with the root-cause diagnosis.

## Interpretation rule
- If Zhao-Cui is already very close to its own dense same-target reference, then Zhao-Cui is unlikely to be the dominant source of the several-percent gap.
- If dense-Kalman changes by <1% under 7->14 and 14->28 refinement but remains several percent away from Zhao-Cui, approximation-family bias is the leading explanation.
- If dense-Kalman changes materially under refinement and moves toward Zhao-Cui, the dense-mixture approximation budget is the leading explanation.
- If either route behaves erratically or violates internal consistency checks, suspect a bug.

## Skeptical audit
- Do not compare transformed-space and raw-`y` likelihoods without the exact Jacobian correction.
- Do not declare convergence after only one refinement step.
- Do not interpret different finite approximations disagreeing as a bug without checking refinement behavior.
- Stop the campaign early if the value ladder already isolates the cause; do not spend score budget unnecessarily.
