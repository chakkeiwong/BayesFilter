# Actual-SV canonical transformed-likelihood comparison plan

## Question
Can we compare the Zhao-Cui actual-SV route and the dense Kalman / Gaussian-closure route in a mathematically valid way by first moving both into the same transformed-observation likelihood formulation?

## Mathematical target
The exact actual-SV likelihood is the observed-data marginal density
\[
L(\theta)=p_\theta(y_{0:T}).
\]
The canonical transformed observation is
\[
z_t = \log(y_t^2),
\]
with exact transformed residual
\[
r_t(x_t;\theta)=z_t-2\log\beta(\theta)-x_t,
\]
and exact transformed observation density
\[
p_\theta(z_t\mid x_t)=\frac{1}{\sqrt{2\pi}}\exp\!\left(\tfrac12 r_t-\tfrac12 e^{r_t}\right).
\]
The Jacobian of the change of variables is exact and \(\theta\)-independent, so it does not change the score with respect to \(\theta\).

## Mechanism being tested
Both candidate numerical routes are to be compared as approximations to the **same transformed exact target**:
1. **Lane A**: Zhao-Cui exact-transformed direct likelihood quadrature.
2. **Dense Kalman / Gaussian-closure lane**: dense Gaussian-mixture / Kalman-style approximation to the transformed model.

The goal is not to prove the finite approximations equal. The goal is to ensure both are tied to the same exact transformed likelihood and then compare their value/score approximation quality against common references.

## Scope
- Variant: actual-SV transformed-observation comparison on tiny deterministic fixtures.
- Objective: compare value and score approximation quality after exact change-of-variables accounting.
- Seed(s): the existing actual-SV deterministic fixtures and the comparator seed used in the repository comparison tests.
- HMC/MCMC settings: none.
- XLA/JIT mode: eager/CPU validation first.
- Expected runtime: targeted tests only.

## Success criteria
- The transformed-observation likelihood is written once in a canonical form and used as the comparison target.
- Lane A and the dense Kalman / Gaussian-closure route are each compared against their proper transformed-target reference.
- The comparison reports explicit approximation gaps, not same-scalar equivalence.
- The score comparison includes the exact Jacobian handling and the \(\theta\)-independence of the transform.

## Diagnostics
Primary:
- lane-local value gap to the transformed exact/dense reference,
- lane-local score residual or finite-difference error,
- cross-lane approximation gap.

Secondary:
- transformed-observation consistency,
- exact Jacobian / score invariance under the log-square transform,
- finite value and finite score on all tested fixtures.

Sanity checks:
- no raw-vs-transformed mismatch,
- no claim that a finite approximation equals the exact likelihood,
- no claim that two different approximation families are the same scalar.

## Expected failure modes
- Comparing Lane A and the dense Kalman route as if they were the same scalar.
- Omitting the exact change-of-variables / Jacobian correction when moving between raw and transformed variables.
- Treating a finite Gaussian-closure approximation as exact likelihood computation.
- Merging Lane A and Lane B documentation so the scalar semantics become ambiguous.

## What would change our mind
- If the transformed-target formulation cannot be written consistently with the current code, the comparison is ill-posed and we should stop.
- If the dense Kalman / Gaussian-closure lane does not approximate the same transformed exact target, it should be treated as a different model approximation and not compared as a same-target estimator.
- If the tests show the score is not invariant under the exact transform as expected, the implementation or interpretation is wrong.

## Files likely to update
### Primary plan / result notes
- `docs/plans/bayesfilter-actual-sv-canonical-transformed-likelihood-comparison-plan-2026-08-11.md`
- `docs/plans/bayesfilter-actual-sv-canonical-transformed-likelihood-comparison-result-2026-08-11.md`

### Reference code paths
- `bayesfilter/highdim/sv_mixture_cut4.py`
- `tests/highdim/test_p41_exact_transformed_sv_zhaocui_ladder.py`
- `tests/highdim/test_p43_sv_value_gradient_cut4_zhaocui.py`

## Verification
1. Compile the affected test modules.
2. Run the actual-SV transformed-likelihood comparison tests.
3. Record lane-local value and score gaps against the appropriate dense reference.
4. Confirm that the comparison is documented as approximation-gap evidence only, not same-scalar equivalence.

## Command
```bash
CUDA_VISIBLE_DEVICES=-1 python -m py_compile \
  tests/highdim/test_p41_exact_transformed_sv_zhaocui_ladder.py \
  tests/highdim/test_p43_sv_value_gradient_cut4_zhaocui.py

CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/highdim/test_p41_exact_transformed_sv_zhaocui_ladder.py \
  tests/highdim/test_p43_sv_value_gradient_cut4_zhaocui.py
```

## Interpretation rule
- If the transformed-observation exact target is consistent and both routes are internally stable, report the approximation gaps and stop claiming cross-lane identity.
- If either route fails the transformed-target consistency checks, treat it as a modeling or implementation blocker before drawing approximation conclusions.
