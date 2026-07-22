# Phase 8 Subplan: Common Proposal/Weight Identity

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Continuation ID: `contract-e-canonical-gradient-migration-continuation-20260714-115526`

Status: `EXECUTION_ACTIVE`

## Objective And Research Intent

Determine whether the common LGSSM affine proposal, Jacobian correction, and
manual JVP compute the exact conditional predictive contribution they claim.
This is the smallest diagnostic that distinguishes a shared implementation
error from finite-particle error after the paired 16-seed audit found nearly
the same Kalman discrepancy in Contract E and no-reset arms.

The candidate mechanism is a missing, duplicated, or incorrectly differentiated
term in

```text
log p_Q(x_t - F x_{t-1})
+ log p_R(y_t - H x_t)
- log p_Q(z_t - F x_{t-1})
+ log |det(dx_t/dz_t)|.
```

For the implemented exact affine conditional proposal, this must equal, for
every fixed previous particle and transition-noise draw,

```text
log N(y_t; H F x_{t-1}, H Q H' + R).
```

The expected failure mode is a value or derivative mismatch in the shared
flow/density/Jacobian path. A mismatch is a repair trigger. Equality instead
rules out that local mechanism and leaves finite-particle recursion error or a
later shared-state bug as the next hypothesis.

## Entry Conditions And Owner Decisions

- Owner-selected center gradient boundary: `delta_grad=0.05`.
- Owner-selected audit count: `16`, exploratory with no power claim.
- The paired audit completed at `T=2,N=128` and classified the reset effect as
  `mixed_or_inconclusive`; both arms failed the same small-shape Kalman screen.
- The exact Kalman likelihood remains the LGSSM oracle.
- The production factory remains empty and all work here is diagnostic only.
- Existing source closure hashes matched before this diagnostic.

## Skeptical Plan Audit

| Risk | Audit disposition |
| --- | --- |
| Wrong baseline | The comparator is the analytic conditional predictive density for the same fixed previous particles, not the marginal Kalman likelihood. |
| Proxy promoted to a scientific gate | This identity can veto or clear only the local proposal/weight implementation; it cannot establish Kalman equivalence. |
| Hidden randomness | Fixed deterministic tensors are used and serialized; no stochastic inference is made. |
| Stale timing assumption | The identity is conditional on the previous particle and is valid under the canonical transition-first loop. Stationarity explains why the generated observation-first data do not create a comparator mismatch here. |
| Environment mismatch | CPU-hidden float64 is an explicit reference diagnostic; it is not GPU/XLA readiness evidence. |
| Unanswerable artifact | The artifact retains value residuals, autodiff Jacobians, manual JVP residuals, source hashes, command, and tolerances. |
| Missing stop | Stop on source drift, nonfinite output, invalid flow chart, or any identity failure requiring code repair. |

Audit verdict: `PASS_FOR_FIXED_IDENTITY_DIAGNOSTIC_ONLY`.

## Evidence Contract

Run fixed float64 tensors with `B=2,N=4,d=3,p=5` at the frozen center
`theta=(0.72,0.55,0.35,0.35,0.45)`. Compare:

1. the exact code expression used in the canonical logits;
2. the analytic conditional predictive log density;
3. TensorFlow autodiff Jacobians of both expressions; and
4. the canonical manual JVP of the code expression.

Also verify that adding uniform log weights and normalizing reproduces the
log-mean-exp of the analytic per-particle predictive densities at `T=1`.

Pass criteria are float64 numerical identity at `atol=2e-11, rtol=2e-11` for
values and all Jacobian/JVP entries, plus finite outputs and a valid flow chart.
These tolerances are implementation-identity tolerances based on float64
roundoff through small Cholesky solves; they are not statistical or scientific
accuracy thresholds. The artifact records raw maxima so the tolerance is not
used to hide scale.

## Default And Assumption Audit

| Choice | Provenance and status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Frozen center | Existing Phase 8 center; reviewed default for this diagnostic | Could miss off-center chart failures | No off-center claim |
| `B=2,N=4` fixed tensors | Minimal shape covering batch/particle axes; convenience choice | Could miss scale-dependent numerical behavior | Exact per-entry residuals only |
| Float64 CPU-hidden | Repository reference-check policy | Does not test GPU/TF32/XLA | Explicit nonclaim |
| `2e-11` identity tolerance | Roundoff allowance for small float64 factorization chain; hypothesis | Could be too loose for a grossly scaled quantity | Report absolute and relative maxima |

## Required Artifacts And Checks

- exclusive JSON under the continuation Phase 8 log root;
- focused tests for value, autodiff, manual-JVP, and normalization identities;
- explicit float64 correction of the paired audit Student critical value using
  its preserved per-seed arrays, written as a new artifact without overwriting
  the original;
- Python compilation, JSON parse, and `git diff --check`;
- a Phase 8 result/close record and exact next hypothesis.

## Forbidden Claims And Actions

- Do not claim Kalman equivalence, finite-particle adequacy, reset correctness,
  primary-shape validity, canonical admission, HMC readiness, leaderboard
  readiness, default readiness, or release readiness.
- Do not tune the fixed tensors, tolerance, center, audit count, or gradient
  boundary from the output.
- Do not rerun the 16 stochastic arms merely to correct reporting precision.
- Do not cross the immutable continuation deadline or launch a larger campaign.

## Handoff And Stop Conditions

If any local identity fails, repair only the shared proposal/weight/JVP defect,
add a focused regression, and repeat this diagnostic before stochastic work.
If all identities pass, close this hypothesis and next test the carried
weighted recursion against an independent finite-mixture recursion at `T=2`.
Only after the common recursion is cleared may a separately budgeted larger
`N`/multi-seed Kalman study be proposed. Stop immediately on source drift,
invalid chart, nonfinite output, missing evidence, or deadline exhaustion.
