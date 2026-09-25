# Contract-E UKF Covariance-Carry Repair

Date: 2026-09-07

Status: IMPLEMENTED; FOCUSED_TESTS_PASS; GPU_RERUN_PENDING

## Blocking discrepancy

The documented canonical lifecycle says that resampling moves the particle
triple `{state, covariance, weight}` and explicitly says that resetting only
states while leaving covariances in their old order breaks Li--Coates
Algorithm 1.  The executed analytical endpoint previously reset the state with
Sinkhorn plus Contract-E and the dual-cap correction, then assigned
`covariances = post_covs` unchanged.  The helper-level categorical
`triple_gather` test did not exercise this endpoint.  That implementation was
wrong relative to the documented canonical covariance lifecycle.

This repair is a prerequisite for interpreting any Phase 4A KDM score result:
positive-bandwidth weights change the transport, so leaving covariance rows in
their source order would feed mismatched local UKF covariances into the next
time step.

## Mathematical repair

Let `Pi[target, source]` be the Sinkhorn coupling produced inside the existing
reset and let

```text
A_ri = Pi_ri / sum_j Pi_rj
```

be the same row-stochastic target-by-source barycentric transport already used
for the reset state.  Carry the attached UKF covariance and its tangent as

```text
P'_r  = sum_i A_ri P_i
dA_ri = dPi_ri / m_r - Pi_ri dm_r / m_r^2
dP'_r = sum_i dA_ri P_i + sum_i A_ri dP_i,
m_r   = sum_i Pi_ri.
```

The carried matrices are symmetrized against finite-precision skew.  No ridge,
clipping, eigensystem projection, or between-particle scatter term is added.
A convex combination of valid positive-definite local covariances remains
positive definite; a numerical guard may report or reject invalid input but
must not silently alter it.

This is the repository's established
`same_transport_barycentric_covariance_carry` semantics.  It is a BayesFilter
OT extension of categorical triple resampling, not a claim that Li--Coates
derived deterministic OT covariance carry.  Adding
`(x_i-x'_r)(x_i-x'_r)^T` would turn local conditional covariance into mixture
total covariance and double-count spread already represented by the particle
locations, so that alternative is not the selected target.

## Evidence contract

| Field | Requirement |
|---|---|
| Question | Does the canonical Contract-E call chain carry each post-update UKF covariance with the exact state transport and propagate its total tangent? |
| Baseline | Current state-only reset, retained only as a failing regression fixture; it is not an eligible scientific baseline. |
| Primary pass | Closed-form transport/covariance value parity and central finite-difference tangent parity, followed by a two-step endpoint test proving the carried covariance enters the next UKF prediction. |
| Vetoes | Source-order covariance survives a non-identity reset; transport orientation differs between state and covariance; covariance or tangent term is omitted; symmetry/finiteness fails; zero-bandwidth canonical/KDM parity or existing analytical finite differences regress. |
| Explanatory only | Magnitude of covariance movement, conditioning, runtime, and TF32 drift. |
| Nonclaims | The repair does not prove Kalman agreement, KDM score improvement, TF32 readiness, HMC readiness, or source-faithfulness of the OT extension. |
| Artifact | This plan, focused tests, and the updated KDM reset/result notes. |

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|---|
| Same `A[target,source]` for states and covariances | Existing repository Algorithm-1 OT implementation and ch19c triple-carry contract | Moves attached local state with the deterministic resampling map | Transpose error attaches the wrong covariance | Marked covariance and explicit matrix-product test | Reviewed repair target |
| Linear barycentric covariance carry | Existing route identifier `same_transport_barycentric_covariance_carry` | Deterministic relaxation of gathering an attached covariance | Averaging may differ from categorical resampling at finite `N` | Report movement and compare only within the declared OT program | BayesFilter extension |
| No between-cloud scatter term | Meaning of `P_i` as local UKF covariance | Particle locations already carry between-component spread | Omitting scatter would be wrong if `P_i` meant total mixture covariance | Lifecycle/type assertion and linear-Gaussian oracle later | Reviewed semantic choice |
| No PSD projection or ridge | Convexity of the SPD cone | Avoids changing accepted covariance values | Invalid upstream covariance reaches a later Cholesky | Explicit finite/symmetry/eigenvalue test in focused diagnostics | Fail-closed policy |

## Skeptical pre-execution audit

- **Wrong baseline:** the authority is the documented canonical lifecycle and
  actual Contract-E transport, not the categorical helper or historical
  pre-2026-08-21 benchmark results.
- **Proxy promotion:** a passing unit finite difference proves implementation
  consistency only; the later Kalman score comparison remains the scientific
  criterion.
- **Hidden assumption:** the meaning of the covariance is fixed as
  particle-local conditional UKF state.  The selected formula would not be
  correct for a total mixture covariance.
- **Stale context:** historical numerical results are not reused.  Only the
  established route semantics and current LaTeX contract are used.
- **Environment mismatch:** focused derivative tests are deliberate CPU
  references; the complete GPU/XLA/TF32 gate is rerun separately.
- **Artifact adequacy:** the endpoint test must show next-step predicted
  covariance changes when the carry is disabled; merely testing a standalone
  matrix formula is insufficient.
- **Stop condition:** stop before an oracle campaign if the endpoint tangent,
  zero-bandwidth parity, or existing canonical suite fails.

The audit passed.  The repair is narrowly defined, has an existing repository
semantic authority, and directly closes a documented call-chain violation.

## Execution sequence

1. Expose the existing reset's normalized transport and tangent in a private
   shared core; preserve the public state-only wrapper for compatibility.
2. Add a triple-reset wrapper that carries `P` and `dP` with that exact
   transport, without a second Sinkhorn solve.
3. Wire the canonical analytical executor to the triple-reset wrapper and
   expose post-reset covariance trace fields.
4. Add explicit value/orientation, tangent finite-difference, and two-step
   call-chain tests.
5. Rerun the focused canonical/KDM suite and the complete Phase 4A GPU/XLA
   calibration.  Update the LaTeX/code crosswalk with the result.

## Execution result

The shared reset core now returns the normalized target-by-source transport and
its tangent.  The canonical analytical endpoint calls the triple wrapper with
the post-update UKF covariance and tangent, and the next time step consumes
the carried covariance.  The trace records both covariance and transport
outputs, making the call chain inspectable.

The focused CPU command passed `19/19` tests before the broader rerun and the
combined canonical/KDM suite subsequently passed `50/50`.  The tests cover
transport orientation/value, central finite-difference covariance tangents,
two-step feedback into UKF prediction, zero-bandwidth KDM parity, score
recursion, and registered endpoint resolution.  GPU/XLA evidence is rerun
separately after this code change; no scientific result is inferred from the
CPU tests.
