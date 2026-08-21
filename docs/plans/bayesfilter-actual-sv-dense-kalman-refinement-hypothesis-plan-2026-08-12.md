# Experiment plan: actual-SV dense-Kalman vs Zhao-Cui hypothesis test

## Question
Why is the likelihood gap between the Zhao-Cui transformed-target approximation and the dense Gaussian-mixture / Kalman approximation still several percent after exact transformation correction back to the raw-`y` likelihood representation?

## Hypotheses
1. **Dense-mixture budget problem**
   The dense Kalman approximation is still using too crude a Gaussian-mixture approximation to `log(\epsilon^2)`, so the gap should shrink materially when the mixture size is increased from 7 to 14 to 28 components.

2. **Zhao-Cui approximation problem**
   The Zhao-Cui TT route may itself be materially biased relative to the exact transformed-target dense reference, so part of the gap may come from Lane A.

3. **Approximation-family bias**
   Even after refinement, the dense Kalman approximation and Zhao-Cui route may remain separated because the Kalman/Gaussian-mixture approximation family loses higher-order shape information that Zhao-Cui retains.

4. **Implementation bug**
   One of the routes may be computing the wrong likelihood/score for its intended approximation family.

## Mechanism being tested
- Compare the Zhao-Cui transformed-target route against its own dense exact-transformed reference on the tiny deterministic actual-SV fixtures. If that gap is tiny, Zhao-Cui is likely not the source of the several-percent discrepancy.
- Build richer Gaussian-mixture approximations to `log(\epsilon^2)` with 7, 14, and 28 components and recompute the dense Kalman approximation.
- Convert the transformed-space log likelihoods back to the raw-`y` likelihood representation using the exact data-only Jacobian correction.
- Compare values and scores across refinement levels and against Zhao-Cui.

## Scope
- Fixtures: tiny actual-SV deterministic fixtures for dimensions 1, 2, 3.
- Comparison objects:
  - Zhao-Cui TT transformed-target approximation.
  - Dense exact-transformed reference for Lane A.
  - Dense Gaussian-mixture Kalman approximation with 7 / 14 / 28 components.
- Score comparison: finite differences on the same approximation object when needed.
- Runtime target: CPU-only.

## Success criteria
- We quantify the Zhao-Cui gap to its own dense exact-transformed reference.
- We quantify the dense-Kalman raw-`y` likelihood and score under 7 / 14 / 28 component mixtures.
- We determine whether the dense-Kalman approximation changes by less than 1% under refinement.
- We identify which hypothesis best explains the observed Zhao-Cui vs dense-Kalman gap.

## Diagnostics
Primary:
- Zhao-Cui vs exact-transformed dense reference value gap and score gap.
- Dense-Kalman 7/14/28 raw-`y` log-likelihood values.
- Dense-Kalman 7/14/28 score vectors.
- Relative change of dense-Kalman value and score when moving 7 -> 14 and 14 -> 28 components.
- Zhao-Cui vs dense-Kalman value and score gaps after transformation correction.

Secondary:
- Whether value gaps are much larger than score gaps.
- Whether refinement stabilizes quickly.
- Whether any route shows erratic, non-convergent behavior suggestive of a bug.

## Interpretation rule
- If Zhao-Cui is already extremely close to its exact-transformed dense reference, then Zhao-Cui is not the main source of the gap.
- If dense-Kalman changes by less than 1% under 7 -> 14 -> 28 refinement, then mixture resolution is probably not the main issue.
- If dense-Kalman stabilizes but remains several percent away from Zhao-Cui, then approximation-family bias is the leading explanation.
- If either route behaves erratically or fails internal consistency, treat that as a bug signal.

## Skeptical audit
- Do not compare transformed-space and raw-`y` likelihoods without the exact Jacobian correction.
- Do not interpret two different approximation families failing to match exactly as a bug by itself.
- Do not declare convergence just because one refinement step is small; require the 7 -> 14 and 14 -> 28 changes both to be small.
- Keep score comparisons on the same mathematical object used for the value comparison.

## Command
```bash
CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
# fit 7/14/28 Gaussian mixtures to log(epsilon^2)
# compute Zhao-Cui, dense exact-transformed, and dense Kalman value/score comparisons
PY
```
