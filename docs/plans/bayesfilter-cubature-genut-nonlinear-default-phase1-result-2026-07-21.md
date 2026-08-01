# Cubature/GenUT Nonlinear Default Program: Phase 1 Result

Date: 2026-07-21

Status: `PASS_PHASE1_CANDIDATE_DESIGN_IDENTITY_LAYER`

## Outcome

Implemented a candidate-only TensorFlow design and scope-identity layer in
`bayesfilter/highdim/cubature_genut_candidate.py`. The canonical Contract E
route and the LGSSM benchmark runner were not changed.

The module provides:

- replicated positive spherical-radial Cubature designs;
- unconstrained GenUT axis construction with explicit feasibility checks;
- a strict positive-weight gate;
- exact equal-weight replication only when masses are representable at `N`;
- standardized-to-physical design mapping; and
- immutable candidate scope/route identities with digest validation.

The Gaussian GenUT specialization is explicitly constructed as the canonical
`2d` Cubature design with equal weights. It is not inferred from a generic
`2d+1` floating-point construction, avoiding ordering and tiny-central-weight
artifacts.

## Checks

| Check | Result |
|---|---|
| Candidate plus legacy LGSSM tests | `17 passed` in `6.14 s` |
| Python compilation | Pass |
| Scoped `git diff --check` | Pass |
| Runtime NumPy dependency | None in candidate module |
| GPU use | Deliberately hidden; no GPU evidence claimed |
| Canonical route changes | None |

The initial test attempt found two implementation issues and was preserved as
phase evidence: Gaussian GenUT's generic `2d+1` ordering did not match Cubature,
and float roundoff made an analytically zero central weight slightly negative.
The repair made the Gaussian alias explicit and canonical. The final suite
passes the corrected behavior.

## Decision Table

| Decision | Status |
|---|---|
| Candidate design mechanics | Passed moment, mapping, positivity, and replication tests |
| Candidate identity mechanics | Passed deterministic digest and tamper rejection tests |
| Generic finite value/score | Not implemented; Phase 2 pending |
| Nonlinear model readiness | Not established |
| Canonical/default/leaderboard readiness | Not established and not changed |
| Next justified action | Write and review Phase 2 generic finite value/total-JVP subplan |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Passed for Phase 1 candidate mechanics |
| Statistically supported ranking | None; no stochastic claim run |
| Descriptive differences | None used |
| Default readiness | False |
| Evidence needed next | Generic model adapter, finite scalar/JVP, branch tests, same-scalar FD audit |

## Nonclaims

This result does not establish exact filtering likelihood, posterior score
accuracy, nonlinear full-horizon validity, XLA/TF32 readiness, high-dimensional
scaling, method superiority, leaderboard admission, HMC readiness, or a NAWM
result.
