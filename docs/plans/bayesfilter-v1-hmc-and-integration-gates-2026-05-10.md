# Plan: BayesFilter v1 HMC And Future Integration Gates

## Date

2026-05-10

## Purpose

This note defines when BayesFilter v1 may claim HMC readiness and when client
integration may begin.  It keeps external compatibility certification separate
from MacroFinance or DSGE switch-over.

## HMC Readiness Gate

HMC readiness is target-specific.  A backend is not HMC-ready merely because a
generic derivative test passes.

Required target evidence:

1. exact model/backend pair;
2. value parity against a reference where available;
3. score parity against finite differences or autodiff where appropriate;
4. Hessian symmetry and local curvature diagnostics if Hessian is used;
5. active floor frequency;
6. weak spectral-gap frequency;
7. deterministic-completion residuals for structural targets;
8. support/rank diagnostics;
9. compiled parity for the target shape;
10. short fixed-seed sampler smoke with acceptance, nonfinite events, and
    divergence diagnostics.

## SVD-CUT HMC Policy

SVD-CUT score/Hessian remains branch-gated:

- derivative target is the implemented regularized law;
- active floors block derivative claims;
- weak spectral gaps block derivative claims;
- separated-spectrum, inactive-floor behavior must dominate the target region
  before HMC experiments are meaningful.

## Linear SVD/eigen Derivative Policy

Linear SVD/eigen derivatives remain deferred.

They become eligible only if:

- a real external compatibility target requires them;
- QR derivatives are insufficient;
- repeated spectra and active floors can be detected reliably;
- a testing-only prototype matches finite differences/autodiff on smooth
  branches.

## Future v1 Integration Checklist

Do not create MacroFinance or DSGE switch-over branches until:

1. v1 API freeze criteria pass;
2. local v1 fixture tests pass;
3. optional live external checks pass on recorded external commits;
4. CPU benchmark artifacts exist;
5. GPU artifacts exist for any GPU claim;
6. HMC artifacts exist for any HMC claim;
7. dependency direction is documented:

```text
client project imports BayesFilter
BayesFilter does not import client project
```

8. rollback plan exists;
9. client defaults remain unchanged until parity and rollback are proven;
10. the client project owner explicitly approves integration timing.

## Output Artifacts For Future Integration

```text
docs/plans/bayesfilter-v1-macrofinance-integration-readiness-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-dsge-integration-readiness-YYYY-MM-DD.md
```

## Veto Diagnostics

Stop integration if:

- BayesFilter v1 public API is still moving;
- compatibility relies on external source edits;
- production dependency cycles appear;
- HMC readiness is inferred from generic tests;
- GPU claims lack escalated evidence;
- client defaults would change without rollback tests.
