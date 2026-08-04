# SVX-ZC Monograph Fixed-Branch Reset Memo

Date: 2026-07-31

## Restart Point

The active authority is `docs/main.tex` and included chapters ch36b, ch37, and
ch38. The target is the BayesFilter fixed adjacent-state TT/KR approximation,
not the Zhao-Cui author MATLAB implementation. Its route classification remains
`extension_or_invention`; source mismatch is not an active veto.

## Current Evidence

- Focused route regression: `5 passed` in
  `tests/highdim/test_zhao_cui_fixed_adjacent_tt_tf.py`.
- The T>0 target now explicitly removes only the current-axis reference density
  from the active adjacent basis; the carried marginal supplies the previous
  density measure.
- Attempt 03 admission artifact:
  `docs/plans/artifacts/bayesfilter-svx-zc-monograph-admission-20260731/attempt03/result.json`.
- Ranks 1, 2, 4, and 6 all pass finite/positivity/coordinate/Jacobian/mass
  closure/conditioning/branch-FD/no-grid checks but fail rank-saturation because
  max fit residual is `0.0564383` or larger versus the declared `1e-8` veto.
- The dense exact-SV comparison is correctly wired in attempt 03; gaps are about
  `0.101` per observation for ranks 4 and 6 and remain descriptive evidence.

## Active Registry State

`SVX-ZC` is not executable for NeuTra. Its active state is
`TARGET_BLOCKED_FILTER_ADMISSION`, reentry rung `fixed-branch numerical
admission`. No target identity, tuning artifact, training, or HMC result exists
for this route.

## First Restart Action

Design one bounded capacity repair ladder under the same data, event order,
coordinate map, positive-defense convention, and total compute budget. Vary
only predeclared degree/order/rank or ALS capacity. Preserve the `1e-8`
rank-saturation veto and all existing structural checks. Write a fresh attempt
directory; never overwrite attempts 01-03.

## Do Not Do

- Do not relabel the route source-faithful.
- Do not relax the residual veto after observing the failed ladder.
- Do not launch NeuTra training or HMC before a fixed-branch admission pass and
  a separate batch-native target-adapter plan.
- Do not use the old source-route blocker as evidence against the monograph
  approximation.
