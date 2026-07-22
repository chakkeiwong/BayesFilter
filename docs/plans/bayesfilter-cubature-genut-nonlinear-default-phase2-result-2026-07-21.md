# Cubature/GenUT Nonlinear Default Program: Phase 2 Result

Date: 2026-07-21

Status: `PASS_PHASE2_GENERIC_FINITE_VALUE_TOTAL_JVP_TOY_GATE`

## Outcome

Implemented the candidate-only generic finite value/total-JVP core in
`bayesfilter/highdim/cubature_genut_filter.py` with a model adapter contract
for initial state, nonlinear transition, observation log density, and all
corresponding tangents.

The core includes:

- fixed-randomness recursive nonlinear transition and observation evaluation;
- finite unrolled Sinkhorn and its cost-scale/floor tangent;
- weighted normalization and weight tangents;
- Contract E moment/residual/Cholesky reset and complete reset JVP;
- per-time value and score increments; and
- reset/marginal diagnostics.

No runtime TensorFlow autodiff, finite difference, or NumPy computation is
used. The canonical Contract E route and the LGSSM benchmark runner were not
modified.

## Test Gate

The toy adapter is nonlinear (`tanh` transition and squared-observation
likelihood), uses fixed stateless innovations and a fixed Cubature design, and
checks the recursive score against central FD of the same finite value scalar.

| Check | Result |
|---|---|
| Phase 2 plus Phase 1 and legacy LGSSM suite | `19 passed` in `6.24 s` |
| Python compilation | Pass |
| Scoped `git diff --check` | Pass |
| Runtime autodiff/FD/NumPy scan | Pass: none in candidate core |
| CPU/GPU | Deliberately CPU-hidden; no GPU evidence claimed |

During implementation review, the ancestor/tangent ordering and the toy
observation tangent were corrected before accepting the final gate. The final
test exercises parameter dependence through initialization, transition,
observation, normalization, transport, reset, and later recursion.

## Decision Table

| Decision | Status |
|---|---|
| Generic finite value/JVP mechanics | Passed toy same-scalar gate |
| Total derivative claim | Correct for the tested fixed finite toy program; not generalized to production rows yet |
| Nonlinear model adapters | Actual SV/KSC-SV/predator-prey/SIR not yet wired to this core |
| XLA/TF32/high-dimensional scaling | Not established |
| Canonical/default/leaderboard readiness | Not established and not changed |
| Next justified action | Phase 3 adapter pilots with target-law and same-target checks |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Passed for the toy finite program |
| Statistically supported ranking | None |
| Descriptive differences | None used |
| Default readiness | False |
| Evidence needed next | Model-specific target adapters, full-horizon claims, and Contract E paired comparison |

## Nonclaims

This result does not establish exact nonlinear filtering, model-row validity,
full-horizon feasibility, score precision, XLA/TF32 readiness, high-dimensional
scaling, method superiority, leaderboard admission, HMC readiness, or a NAWM
result.
