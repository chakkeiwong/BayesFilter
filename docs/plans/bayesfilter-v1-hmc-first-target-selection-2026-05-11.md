# Decision: BayesFilter v1 First HMC Readiness Target

## Date

2026-05-11

## Decision

The first HMC readiness target should be a linear Gaussian QR derivative
fixture, not SVD-CUT and not a live DSGE target.

Selected target:

```text
linear_qr_score_hessian_static_lgssm
```

## Rationale

The QR derivative target has the strongest current evidence:

- BayesFilter-local QR value tests pass;
- QR score/Hessian tests pass;
- compiled parity tests pass;
- optional live MacroFinance QR value and derivative compatibility passes on
  commit `0e81988957ef1f8b520014929bea32ffee3881f4`;
- no spectral-gap or active-floor branch is required for the QR factor path.

SVD-CUT remains a later target because derivative/HMC claims require
active-floor and spectral-gap branch diagnostics.  Rotemberg remains later
because it is an optional live external DSGE fixture and should not be the
first HMC readiness proof.

## Required Evidence Ladder

Before claiming HMC readiness for the selected target, collect:

1. exact model and parameterization;
2. value parity against existing QR likelihood result;
3. score parity against finite differences or TensorFlow autodiff;
4. Hessian symmetry and curvature diagnostics;
5. compiled/eager parity for value, score, and Hessian;
6. nonfinite objective and gradient event counts on a fixed parameter box;
7. short fixed-seed TFP HMC or NUTS smoke;
8. acceptance, step-size, divergence, and nonfinite diagnostics;
9. CPU benchmark artifact for the target shape;
10. GPU artifact only if escalated GPU probes and matching shapes exist.

## Blocked Claims

Do not claim:

```text
hmc_ready_all_filters
hmc_ready_svd_cut
hmc_ready_dsge
```

The first HMC target is deliberately narrow.  Passing it would prove one
well-controlled QR derivative path, not the whole BayesFilter filtering stack.
