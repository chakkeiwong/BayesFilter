# Pairwise-Moment GenUT Cross-Model Reset Memo

Date: 2026-07-30

## Current Verdict

Pairwise off-diagonal co-skewness/co-kurtosis matching is implemented and
mechanically valid, but it is not a universal GenUT default.

The LGSSM result is a candidate-screen failure, not an implementation failure.
The generic map targets finite empirical weighted cross moments rather than the
known Gaussian values `(co-skew=0, co-kurtosis=1)`. Because the cubature reset
itself has off-diagonal co-kurtosis zero, the nonzero map injects a stochastic,
chart-dependent deformation on every reset. Repeated over `T=50`, this can
increase score variance even while the pairwise residual decreases. The
selection screen also used only two tuning seeds per validation trajectory and
had an any-coordinate ratio `>1` veto, so its rejection is conservative and
low-power. The exact Kalman post-run diagnostic nevertheless showed the
zero-step arm had the lowest validation aggregate score RMSE among the tested
arms.

Current model evidence:

```text
Austria SIR d=18: strong score-variance reduction, material finite-value shift;
                  promising opt-in repair, not promoted.
LGSSM d=3:        every nonzero arm failed at least one validation
                  score-coordinate variance veto; retain diagonal-only.
KSC-SV d=1:       zero pair constraints; exact structural no-op.
Predator-prey d=2: selected arm finite/value-stable, but claim variance ratio
                  0.953 with CI [0.832, 1.128]; no supported improvement.
```

## Canonical Cross-Model Evidence

Plan:

`docs/plans/bayesfilter-pairwise-moment-genut-lgssm-ksc-predator-prey-trial-plan-2026-07-30.md`

Result:

`docs/plans/bayesfilter-pairwise-moment-genut-lgssm-ksc-predator-prey-trial-result-2026-07-30.md`

Artifact:

`docs/benchmarks/artifacts/pairwise_moment_genut_cross_model_20260730/attempt01/result.json`

Successful smoke:

`docs/benchmarks/artifacts/pairwise_moment_genut_cross_model_20260730/smoke_attempt02/result.json`

Preserved XLA failure:

`docs/benchmarks/artifacts/pairwise_moment_genut_cross_model_20260730/smoke_attempt01/failure.json`

## Important Reference Hierarchy

- LGSSM has an exact affine Kalman value and analytical score oracle.
- KSC-SV has a converged sequential dense transformed-mixture value reference;
  its centered-FD derivative is diagnostic only. Runtime GenUT scores remain
  manual recursive scores.
- Predator-prey has no exact same-target score oracle in this campaign.
- SGQF is not a truth oracle for nonlinear SIR, KSC-SV, or predator-prey.
- Fixed Zhao-Cui happens to agree almost exactly with the dense KSC reference
  on the short scalar target, but it remains a diagnostic method route.

## Engineering State

The pairwise tangent projection in
`bayesfilter/highdim/higher_moment_contract_e.py` uses broadcasted reductions
instead of an equivalent `einsum` because the latter triggered an XLA GEMM
autotuner layout failure at LGSSM shape `[1008,5,3] x [3,3]`. The rewrite
passed independent forward-autodiff JVP parity.

Scalar state dimension now skips the pairwise loop exactly. Test
`test_pairwise_moment_controls_are_exact_structural_noop_for_scalar_state`
guards this property.

Focused tests: `35 passed` after both repairs. Final GPU/XLA campaign passed
with approximately `128.3 MiB` TensorFlow allocator peak and `464.8 s` wall
time.

## Next Justified Step

Do not expand pairwise tuning for LGSSM, KSC-SV, or predator-prey based on this
result. The next useful pairwise work is Austria-specific: localize the
`log_kappa_scale` discrepancy and explore the value/variance tradeoff with an
independent same-target score teacher if one can be constructed.

For KSC-SV, any future improvement must target the scalar distribution itself,
such as stronger one-dimensional moment or mixture matching. Off-diagonal pair
moments cannot help without changing the state representation.
