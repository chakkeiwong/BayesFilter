# Phase 1 Subplan: Normative Contract E Mathematics And Evidence-Design Freeze

Date: 2026-07-13

Status: `REVIEWED_ACTIVE`

Master program:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-master-program-2026-07-13.md`

## Phase Objective

Define, derive, and independently check the exact finite Contract E--Chol value
program and its total derivative before schema-v2 or production implementation.
Freeze every mathematical convention, nonsmooth branch, numerical veto, finite-
difference safeguard, and LGSSM uncertainty design that later implementation and
promotion runs must obey.

This phase must answer what quantity is canonical. It must not infer that target
from whichever current helper happens to run.

## Entry Conditions Inherited From Phase 0

- `contract_e_chol_v1` is the only reset semantics eligible to seek canonical
  status.
- The total derivative identifier is
  `contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`.
- All v1/raw artifacts are historical, non-upgradeable, and canonically
  ineligible; no raw fallback exists.
- Canonical identity will be factory-issued in Phase 2 and cannot be caller
  stamped.
- The Phase 0 route-freeze manifest parses, focused revocation tests pass, and
  every known raw/Contract E route and consumer has an exact anchor.
- Existing dense Contract E code is reference evidence only, not the normative
  authority.
- Dirty work from other lanes remains untouched.

## Claimed Target And Quantity Computed

The claimed target is one deterministic finite program conditional on prepared
model inputs and fixed random streams. At each filtering time it computes the
declared likelihood increment before reset, performs streaming positive transport
with explicit row-mass quotient, applies Contract E--Chol moment restoration, and
passes equal weights and reset particles to the next time.

The claimed gradient is the total derivative of that same program with respect
to the declared parameter coordinates. The specification must prove that the
adjoint includes direct source-moment and source-weight dependence as well as the
transported-cloud path. A `Y+`-only adjoint is a partial derivative and is wrong
relative to this target.

## Required Mathematical Specification

Write
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase1-normative-mathematics-spec-2026-07-13.md`
in repository notation. It must include checked derivations for:

1. Filtering time order: state proposal, flow, corrected normalized weights,
   current observed-data likelihood increment, reset, and next-time state.
2. Weight domain and coordinates: whether VJPs are with respect to normalized
   weights, log weights, or logits, and the exact normalization pullback.
3. Moment conventions: weighted source mean/covariance and equal-weight reset
   mean/covariance, including the exact `1/N` versus `1/(N-1)` denominators.
4. Streaming transport: numerator, row mass, quotient
   `Y_i+ = numerator_i / row_mass_i`, its JVP/VJP, and a fail-closed zero/small-
   row-mass policy. A hidden denominator floor is forbidden unless it is part of
   the stated target and its active-set derivative is defined.
5. Fixed residual design: centering, scaling, orientation, seed/hash identity,
   parameter dependence, and whether any design derivative exists.
6. Contract E forward equations: residual injection, covariance gap, Cholesky
   factors, triangular-solve orientation, affine restoration, and the precise
   covariance object restored when ridge is nonzero.
7. Ridge semantics: a predeclared parameter-independent prepared input, or a
   fully differentiated policy. A stopped adaptive ridge cannot support the
   total-gradient claim. The spec must say plainly whether the unridged or ridged
   covariance is the exact target and what diagnostic applies to the other.
8. Complete JVP/VJP: direct moment and weight terms, transported-cloud terms,
   row-quotient terms, residual-design terms if any, ridge terms if any, and the
   normalization pullback. It must explicitly derive
   `G_X = G_X^moments + G_X^transport` and
   `G_w = G_w^moments + G_w^transport`.
9. Active sets: row-mass guard, Cholesky/ridge branch, clipping, temperature,
   masks, and any support transform. The spec must distinguish smooth-chart
   derivatives from branch changes.
10. Same-scalar identity: the value returned beside JVP/VJP and the value used by
    every FD center/endpoint are the same callable and prepared inputs.

Every categorical mathematical claim needs either a derivation in this notation
or a checked source anchor plus a logical derivation for the current setting.

## Required Numerical Veto Design

Write a machine-readable freeze at
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase1-numerical-statistical-design-freeze-2026-07-13.json`.

Before observing canonical promotion results, it must define dtype- and scale-
aware vetoes for:

- weighted/equal-weight mean residual;
- covariance residual for both the stated ridge target and the disclosed
  unridged diagnostic;
- row and column marginal residuals;
- minimum row mass and quotient activity;
- Cholesky diagonal/positive-definiteness failure;
- condition number or backward-error proxy;
- realized ridge magnitude and any branch activity;
- nonfinite values/gradients; and
- chunk invariance for forward and gradients.

Thresholds may come from floating-point backward-error bounds, an independent
float64 reference discrepancy, or a predeclared engineering requirement. They
must record units, scale, dtype, derivation, and role. A number selected only
because a current output happens to pass is forbidden. If a defensible threshold
cannot be derived, leave the affected promotion gate blocked and escalate rather
than inventing it.

## Required FD Design

The owner-directed `0.05 * sqrt(p)` relative rule remains only an implementation
screen for an individual coordinate/direction. It is not a confidence level,
Kalman-equivalence margin, or general stochastic-gradient tolerance.

Freeze before implementation:

- a representable central-difference step ladder per coordinate, expressed in
  parameter scale and dtype;
- identical fixed data, random streams, masks, residual design, ridge input, and
  compiled callable at center and both endpoints;
- exact center-primal identity;
- endpoint separation and endpoint-collapse checks;
- a near-zero absolute scale derived from value scale, parameter scale, dtype,
  and/or the stable ladder plateau rather than an arbitrary constant;
- the ladder-convergence decision and selected plateau rule; and
- active-set/branch identity at center and endpoints.

Passing the relative screen without ladder stability and branch identity is not
derivative evidence.

## Required LGSSM Statistical Design

Before Phase 8 results, freeze:

- the finite-seed estimands for total value bias and each of the five gradient
  components relative to the TensorFlow float64 Kalman oracle;
- common prepared observations and paired/common-random-number seed identity;
- either a fixed seed count or a precision-stopping design with an interval
  construction whose coverage remains valid under that stopping rule;
- confidence level and interval construction with assumptions stated;
- a simultaneous or multiplicity-aware rule for the five gradient components;
- an absolute rule for oracle components near zero and a relative rule only
  where the denominator is scientifically meaningful;
- uncertainty treatment for the owner-accepted `0.1%` `d=3,T=50` value-bias
  boundary; and
- the exact decision table for equivalent, non-equivalent, and inconclusive.

The historical `1%` gradient screen is not an automatic margin. Derive an
equivalence region from the intended numerical/HMC use, with units and
consequences, or preserve an explicit owner-decision blocker before Phase 8.
Sign reversal, nonfinite output, or order-one relative disagreement remain hard
vetoes and do not become acceptable merely because an interval is wide.

## Required Artifacts

- Normative mathematics specification named above.
- Numerical/statistical design-freeze JSON named above.
- Phase result:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase1-normative-mathematics-result-2026-07-13.md`.
- Focused derivation-check log under
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase1/`.
- Phase 2 schema/factory subplan:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase2-schema-v2-factory-subplan-2026-07-13.md`.
- Updated ledger and stop handoff.

## Required Checks, Tests, And Reviews

1. Inspect the Contract E LaTeX sections and dense helper line by line; record
   every agreement and mismatch without treating either as authority by default.
2. Re-derive the fixed-ridge forward and pullback independently.
3. Run CPU-hidden float64 tiny checks with fixed tensors comparing the dense
   manual VJP to TensorFlow autodiff and directional central differences. These
   checks are diagnostic only; implementation parity remains Phase 3.
4. Perturb source particles and weights separately so missing direct moment/
   weight terms cannot be hidden by the transport path.
5. Check the quotient derivative independently at nonuniform row masses.
6. Validate the JSON freeze and all source/math anchors.
7. Run `git diff --check` over phase artifacts.
8. Obtain one bounded fresh read-only review focused on the mathematical target,
   complete pullback, non-arbitrary thresholds, and statistical coverage. The
   platform-blocked Claude disclosure path is not retried or bypassed.

CPU reference commands must set `CUDA_VISIBLE_DEVICES=-1`. No trusted GPU or
production benchmark is authorized in this phase.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the canonical finite Contract E--Chol value program and its total derivative fully and unambiguously specified before implementation? |
| Baseline/comparator | Independent derivation, dense Contract E helper, TensorFlow float64 autodiff, directional FD, existing LaTeX, and exact Kalman only for later statistical-design definition. |
| Primary criterion | All forward conventions and total pullback terms are derived; ridge and active-set semantics are explicit; numerical/FD/statistical gates are frozen without inspecting canonical promotion outputs. |
| Promotion vetoes | Missing direct moment/weight term; ambiguous covariance denominator/orientation; stopped adaptive ridge; hidden row floor; arbitrary threshold; invalid coverage under optional stopping; missing multiplicity or near-zero rule; or documentation/code treated as authority without checking. |
| Repair triggers | Algebra mismatch, autodiff/VJP mismatch, FD ladder instability, incomplete active-set definition, or review-identified design ambiguity. |
| Continuation vetoes | The selected Contract E target is internally inconsistent; no admissible ridge chart can be defined; a required scientific threshold needs owner judgment; five nonconvergent material repair rounds; or campaign budget exhaustion. |
| Explanatory only | Runtime of tiny checks, raw-route outputs, and descriptive historical seed scatter. |
| Not concluded | Production implementation correctness, streaming feasibility, Kalman agreement, nonlinear validity, HMC readiness, leaderboard completeness, or release readiness. |

## Forbidden Claims And Actions

- Do not implement schema v2, the canonical factory, or production Contract E in
  this phase.
- Do not edit the dirty model-specific compact harnesses from the other lane.
- Do not use current raw-route results to tune a gate.
- Do not reuse the FD `0.05 * sqrt(p)` rule for Kalman agreement.
- Do not reuse actual-SV `6%` outside its declared model-specific role.
- Do not silently restore the historical `1%` Contract E gradient margin.
- Do not call covariance restoration sufficient for gradient correctness.
- Do not call an adaptive ridge fixed if it changes with the differentiation
  candidate.
- Do not launch GPU, HMC, nonlinear, leaderboard, detached, or long commands.

## Exact Next-Phase Handoff Conditions

Phase 2 may begin only if:

- the normative spec uniquely defines the forward value and complete total
  derivative;
- the row quotient, weight coordinates, covariance denominators, Cholesky
  orientation, residual design, ridge target, and active sets are explicit;
- independent float64 autodiff/VJP/FD checks agree on fixed tiny fixtures;
- direct source-particle and source-weight perturbations are covered;
- every numerical veto has a scale/dtype derivation or remains an explicit
  promotion blocker;
- the FD ladder and near-zero policy are frozen;
- the paired statistical design has valid coverage under its stopping rule;
- the Kalman-gradient margin is derived or remains an explicit pre-Phase-8 human
  gate, not an invented number;
- focused checks and bounded mathematical review pass;
- the result states target, computed quantity, equality status, and remaining
  nonclaims directly; and
- the Phase 2 schema/factory subplan is drafted and reviewed for consistency.

## Stop Conditions

Stop and write a blocker result if the mathematical target is inconsistent, a
fixed-ridge chart cannot state what covariance it restores, the complete
pullback cannot be derived, a threshold requiring owner judgment would otherwise
be invented, an unexpected concurrent edit appears, five material repair rounds
do not converge, or the campaign budget cannot support a valid close record.

## Phase-End Protocol

1. Run all derivation and focused CPU-hidden checks.
2. Write the Phase 1 result or blocker result with a decision table.
3. Separate mathematical-target failure, reference-code failure, numerical-chart
   failure, and evidence-design failure.
4. Draft or refresh the Phase 2 schema/factory subplan.
5. Review the Phase 1 result and Phase 2 handoff when material.
6. Update the ledger and stop handoff.
7. Advance only when every exact handoff condition passes.
