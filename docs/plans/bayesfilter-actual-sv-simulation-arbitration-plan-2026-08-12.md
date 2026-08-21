# Experiment plan: actual-SV simulation arbitration of Zhao-Cui vs Dense Kalman

## Question
On simulated data from the exact actual-SV model, is the remaining gap between Zhao-Cui and dense Kalman explained mainly by Dense Kalman error, by Zhao-Cui error, or by finite-budget approximation error that disappears under refinement?

## Mechanism being tested
We simulate paths from the exact actual-SV model
\[
y_t = \beta e^{x_t/2}\epsilon_t,\qquad x_t = \gamma x_{t-1} + \sigma \eta_t,
\]
then evaluate, on the **same simulated datasets**:
1. Zhao-Cui transformed-target approximation,
2. Zhao-Cui dense same-target transformed reference,
3. dense Kalman / Gaussian-mixture approximation with 7 / 14 / 28 components.

All likelihoods are compared in the same raw-`y` representation by applying the exact log-square transformation correction.

## Hypotheses
### H1. Dense Kalman finite-budget problem
The dense Kalman approximation is under-resolved at low component count. Refining from 7 to 14 to 28 components materially changes the result and shrinks the gap to Zhao-Cui.

### H2. Zhao-Cui problem
Zhao-Cui itself is materially biased relative to its own dense same-target transformed reference.

### H3. Dense Kalman problem
Zhao-Cui matches its own dense target well, but dense Kalman remains materially different even after refinement, indicating the issue lies on the dense Kalman approximation side.

### H4. Bug
One route behaves erratically or fails its own internal same-target consistency checks.

## Scope
- Simulated paths: exact actual-SV model with `(gamma, beta, sigma) = (0.6, 0.4, 1.0)`.
- Dimensions: 1 / 2 / 3.
- Horizons: short deterministic prefixes consistent with the existing tiny-fixture harness.
- Comparison objects:
  - Zhao-Cui TT transformed-likelihood approximation,
  - Zhao-Cui dense exact-transformed reference,
  - dense Kalman / Gaussian-mixture approximation with 7 / 14 / 28 components.
- Runtime target: CPU-only.

## Success criteria
- Fixed-variant actual-SV batch TT is compared to its own dense same-target reference on the simulated datasets.
- Exact-transformed Zhao-Cui is compared to its own dense same-target reference on the simulated datasets.
- KSC-surrogate Zhao-Cui is compared to its own dense KSC reference on the simulated datasets.
- Dense Kalman is compared to Zhao-Cui after exact transformation correction back to raw-`y` likelihood.
- 7 / 14 / 28 component refinement changes are measured.
- The experiment identifies which hypothesis best explains the observed gap.

## Diagnostics
Primary:
- Zhao-Cui vs Zhao-Cui dense same-target value gap and score gap,
- dense-Kalman raw-`y` likelihood values at 7 / 14 / 28 components,
- relative dense-Kalman value change under 7->14 and 14->28 refinement,
- Zhao-Cui vs dense-Kalman raw-`y` likelihood gap after each refinement.

Secondary:
- relative score changes under refinement,
- whether Zhao-Cui’s self-gap remains negligible under simulation,
- whether dense-Kalman stabilizes before or after the gap closes.

## Interpretation rule
- If the fixed-variant actual-SV batch TT route remains extremely close to its own dense same-target reference, that route is unlikely to be the dominant source of the cross-method gap.
- If exact-transformed Zhao-Cui remains extremely close to its own dense same-target reference, Zhao-Cui is unlikely to be the dominant source of the transformed-target gap.
- If dense Kalman changes by <1% under 7->14 and 14->28 refinement but remains materially away from Zhao-Cui, the issue is on the dense Kalman side rather than Zhao-Cui.
- If dense Kalman moves materially toward Zhao-Cui under refinement, the issue was largely finite mixture resolution.
- If either route behaves erratically or fails same-target checks, suspect a bug.

## Skeptical audit
- Do not compare transformed-space and raw-`y` likelihoods without the exact Jacobian correction.
- Do not infer a Zhao-Cui failure from cross-method disagreement alone; first require Zhao-Cui to fail against its own dense same-target reference.
- Do not infer dense-Kalman convergence from one refinement step only; require both 7->14 and 14->28 changes to be small.

## Command
```bash
CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
# simulate exact actual-SV paths,
# compare Zhao-Cui to its own dense same-target reference,
# compare dense Kalman 7/14/28 to Zhao-Cui after exact transformation correction.
PY
```
