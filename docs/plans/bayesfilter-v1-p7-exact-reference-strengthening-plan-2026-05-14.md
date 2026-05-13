# BayesFilter V1 P7 Exact Reference Strengthening Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase P7 / R8 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Entry Gate

P7 starts only if P3 identifies a claim that needs stronger nonlinear
reference evidence than the current exact affine and dense one-step projection
oracles.

## Motivation

Models B-C are nonlinear, so dense one-step Gaussian projection diagnostics do
not certify exact full nonlinear likelihoods.  Short-horizon dense quadrature
or high-particle seeded SMC can strengthen evidence without becoming a
production dependency.

## Candidate References

Allowed:
- dense quadrature for low-dimensional short horizons;
- seeded high-particle SMC as diagnostic Monte Carlo;
- stored reference artifacts with reproducibility metadata.

Not allowed without a separate plan:
- production dependency on SMC packages;
- unseeded stochastic references;
- treating Monte Carlo estimates as exact.

## Primary Gate

P7 passes if the new reference artifact improves claim clarity for Models B-C
and labels exact, deterministic approximation, and Monte Carlo evidence
correctly.

## Veto Diagnostics

Stop and ask for direction if:
- stochastic references lack seeds or reproducibility metadata;
- dense projection is mislabeled as exact full nonlinear likelihood;
- reference dependencies enter production imports;
- reference artifacts are used to justify HMC or GPU claims without those
  gates.

## Expected Artifacts

```text
docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-result-2026-05-14.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Tests or benchmark artifacts are optional and should remain default-CPU safe.
