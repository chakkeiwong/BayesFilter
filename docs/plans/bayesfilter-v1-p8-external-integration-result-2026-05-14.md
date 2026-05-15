# BayesFilter V1 P8 External Integration Result

## Date

2026-05-14

## Governing Plan

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p8-external-integration-plan-2026-05-14.md
```

## Phase Scope

P8 prepares external-client integration.  It does not switch MacroFinance or
DSGE to BayesFilter, does not edit external source trees, and does not add
production dependencies on external projects.

## Plan Tightening

No plan edit was needed.  The P8 scope is already clear:

- external clients remain compatibility targets;
- bridges start as test-only or optional live checks;
- MacroFinance and DSGE ownership stays outside this lane;
- SGU economics are not promoted unless the structural/DSGE lane is reopened.

## Independent Audit

Audit question:

```text
Can an integration plan be reviewed without changing external source trees or
creating production dependency cycles?
```

Audit findings:

- BayesFilter already has local adapter modules under `bayesfilter/adapters`,
  but P8 does not modify them.
- Existing optional tests cover MacroFinance-style linear compatibility and
  DSGE adapter gates, but P8 keeps optional external checks out of default CI.
- P1-P7 produced current V1 evidence for nonlinear values, scores, branch
  diagnostics, tiny HMC smoke, GPU/XLA diagnostics, Hessian deferral, and
  exact-reference deferral.
- MacroFinance should remain an external project until BayesFilter V1 has a
  stable release tag and client-side owners explicitly open a switch-over
  branch.
- DSGE should remain design/test-only until a client-owned fixture provides a
  causal structural filtering law and residual contract.  SGU remains blocked
  as a production filtering target in this V1 lane.

Veto diagnostics checked:

| Veto | Result |
| --- | --- |
| BayesFilter imports MacroFinance or DSGE as production dependency | clear; no code changes |
| External source is modified from V1 lane | clear; no external edit |
| SGU economics promoted without reopening structural/DSGE lane | clear; SGU remains blocked |
| Optional live checks described as default CI | clear; optional status preserved |

## Integration Plan

### First Client Target

MacroFinance linear QR compatibility should remain the first external client
target because:

- the linear QR value/score/Hessian path has the strongest current evidence;
- compatibility tests already exist in BayesFilter-local tests;
- integration can be staged without touching nonlinear Model C, GPU, or SGU
  decisions;
- rollback is simple: keep MacroFinance on its current implementation while
  BayesFilter runs as an optional comparison backend.

### Bridge Type

The first bridge should be test-only or optional-live:

```text
BayesFilter fixture/adapters -> MacroFinance-compatible expected arrays
```

Production client imports should wait until:

- BayesFilter V1 has a tagged release or pinned commit;
- default CPU and focused V1 tests pass on that commit;
- optional live MacroFinance compatibility passes on a clean recorded external
  checkout;
- MacroFinance owner approves a client-side switch-over branch.

### API Surface Ready For Clients

The stable-enough BayesFilter V1 surface for client review is:

- public import gate in `tests/test_v1_public_api.py`;
- linear QR value/score/Hessian API;
- structural affine conversion and deterministic-completion diagnostics;
- nonlinear SVD cubature, SVD-UKF, and SVD-CUT4 value/score functions at the
  documented branch scope;
- testing-lane nonlinear Models A-C and branch diagnostics.

The following are not production-ready client claims:

- nonlinear Hessians;
- exact full nonlinear likelihood for Models B-C;
- broad GPU speedup;
- HMC convergence for nonlinear targets;
- DSGE SGU production filtering.

### Required Parity Tests

Before any client switch-over:

- BayesFilter default CPU suite passes;
- focused V1 nonlinear suite passes;
- MacroFinance optional live compatibility passes on a clean external checkout
  with recorded commit hash;
- no production import from MacroFinance or DSGE appears in BayesFilter public
  modules;
- any client-side adapter maps parameter names, order, dtype, and Hessian sign
  conventions explicitly;
- deterministic-completion residuals and regularization diagnostics are
  exposed in the client-facing result object or diagnostic log.

### DSGE Integration Boundary

DSGE integration remains future/test-only:

- Rotemberg is the preferred first optional live DSGE fixture if the DSGE lane
  supplies a causal local filtering law and residual contract;
- EZ can be a metadata fixture after its all-stochastic or structural
  completion contract is explicit;
- SGU remains blocked until the DSGE/structural lane owns the economics,
  timing, and residual closure.

BayesFilter should not import DSGE economics into production modules.

### Rollback Path

Every client integration trial must be reversible:

- keep the client default backend unchanged until parity gates pass;
- add BayesFilter as an opt-in backend or comparison mode first;
- write artifacts with commit hashes for both repositories;
- keep client-specific failures classified as external evidence rather than
  BayesFilter default-CI failures unless the same issue reproduces on
  BayesFilter-local fixtures.

### Ownership

BayesFilter owns:

- public filtering APIs;
- diagnostics and metadata contracts;
- BayesFilter-local fixtures and tests;
- optional compatibility artifacts.

MacroFinance/DSGE owners own:

- client-side switch-over timing;
- client economics and parameter conventions;
- production default changes;
- client repository edits and rollback branches.

## Gate Result

P8 primary gate passes:

- the integration plan can be reviewed without external source changes;
- all external work remains optional/test-only until a client lane is opened;
- no veto diagnostic fired.

## Next Phase Recommendations

Recommended next phases after final validation:

1. Freeze the V1 release candidate after focused/default CPU tests pass.
2. Run an optional live MacroFinance compatibility check on a clean external
   checkout and record commit hashes.
3. Open a separate client-owned MacroFinance switch-over plan only if that
   optional check passes.
4. Keep DSGE integration design-only until a client-owned Rotemberg or EZ
   fixture contract is supplied.
5. Use larger-shape or batched benchmarks before making any GPU performance
   claim.
