# Claude review adjudication: tempered reverse-KL transport ensemble

Date: 2026-08-28  
Status: `ALL_SUBSTANTIVE_FINDINGS_ADJUDICATED_ACTIVE_PLAN_REPAIRED`

Reviewed replies:

- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-math-review-reply-2026-08-28.md`;
- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-plan-review-reply-2026-08-28.md`.

Revised plan:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`.

## Verdict

I agree with Claude's `AGREE` verdict on the mathematical note. I also agree
that the plan needed repair. All two implementation blockers and all six
serious-campaign blockers identify real missing obligations. The plan has been
amended without changing the proposal: training still uses fresh IID Gaussian
reverse-KL queries, maps remain categorical rather than averaged, and sampling
still uses frozen charts with exact corrected kernels and a proper replica-
exchange bridge.

Two suggested remedies were not adopted literally:

1. A finite grid of beta values and theta points cannot prove bridge
   integrability. It is useful only for numerical stress testing. The active
   plan instead requires a q=20-specific analytic bound and a source-bound
   receipt.
2. Retrying fresh Gaussian batches until one is wholly valid conditions the
   stochastic training measure on numerical admission. The active plan uses a
   fixed pre-optimizer latent screen and a finite deterministic reference-
   affine/scale initialization repair ladder, then fails closed on invalid
   training rows.

## Mathematical-review findings

| Finding | Disposition |
|---|---|
| C1: explain `log Z` | Agree as optional clarity. The existing proof is correct and the implementation plan never differentiates `Z`. No mathematical repair is required. |
| C2: make bridge integrability explicit | Agree with the concern and with the note's explicit assumption `0 < Z_beta < infinity`. Do not adopt the suggested statement that common support alone is sufficient: it is not. For q=20, the plan now proves a stronger target-specific sufficient condition, namely a normalized prior and a positive uniformly bounded finite-horizon likelihood. |
| C3: continuous state-dependent gamma | Agree as optional clarity. The note already requires a separate Metropolis, Gibbs, or augmented-state argument for any state-dependent selector. The implementation plan now rejects such a selector structurally and retains an exact negative fixture. |

The LaTeX mathematical authority was not changed by this adjudication, so the
Claude `AGREE` verdict and prior MathDevMCP audit continue to apply to the same
source hash.

## Implementation blockers

| Finding | Agreement | Repair |
|---|---|---|
| I1: no intermediate-beta properness check | Agree with the missing obligation, not with treating a finite sample as proof. | Phase 0 now binds a proof to the actual q=20 construction. With finite horizon, positive fixed observation variance `R`, and nonnegative unscented covariance weights, each Gaussian innovation variance is at least `R`; therefore `0 < L(theta) <= M`. Thus `0 < integral g0 L^beta <= max(1,M) < infinity` for every beta. Held-out points are separately classified as numerical stress evidence. |
| I2: blind initialization can deadlock | Agree. | Each initialization is screened before optimizer creation on one fixed stateless standard-base-Gaussian bank, disjoint from training and folded by component identity. It may traverse only a predeclared finite reference-affine/scale ladder, with no optimizer mutation. Exhaustion is explicit failure. Once training is admitted, invalid rows are archived and fail the update; no fresh-batch selection manufactures a valid update. |

## Serious-campaign blockers

| Finding | Agreement | Repair |
|---|---|---|
| S1: gamma semantics not exercised | Agree. | Phase 6 verifies `pi K = pi` exactly for fixed uniform and nonuniform gamma, rejects state-dependent gamma at configuration, and verifies the two-state counterexample exactly. Stochastic moment checks remain explanatory. |
| S2: quadratic work measured but ungated | Agree with the missing budget gate; disagree that it represents `K^2 B` SSL-LSTM target calls. | The target is evaluated on `K B` outer samples; `K^2 B` counts flow inverse/cross-density work. Phase 8A derives an admissible joint-arm envelope from measured memory/update time and the arm's allocated campaign budget. |
| S3: `K` search and confirmation conflated | Agree. | Phase 8A uses only calibration/validation random streams; Phase 8B freezes one candidate per arm and consumes no confirmation stream. Phase 9 alone consumes untouched confirmation streams. The observed data and posterior target are never split or changed. |
| S4: no learned-map reliability screen | Agree, strengthened. | Phase 4 implements and Phase 8 applies round-trip, inverse/forward log-determinant cancellation, score-finiteness, and conditioning screens on self-component, every cross-component, reference, and declared diagnostic banks. A self-only check is insufficient for multi-chart HMC. |
| S5: undefined direct cold mode transition | Agree with the undefined gate; reject an arbitrary Euclidean-distance substitute. | The q=20 protocol uses the strict sign of `observation_weight.0.0`, whose half-spaces contain the two known sign-separated MAP representatives. It requires per-chain visits/crossings and an MCSE-aware binary-indicator diagnostic while explicitly denying formal basin or exhaustive-mode identification. Initialization forgetting uses start-stratified equivalence intervals rather than a nonsignificant difference test. |
| S6: physical replica baseline may use a different bridge | Agree. | Phase 5 makes the physical and charted routes consume the same proper bridge object, endpoint identities, status/cache rules, and swap code. The diagnostic pure-power implementation is ineligible. |

## Non-blocking findings

| Finding | Disposition |
|---|---|
| N1: oracle arm unspecified | Added as an optional discovery ceiling using the known sign-separated representatives. It cannot select, rank, or promote the blind candidate. |
| N2: tuning-artifact reuse ambiguous | Clarified: reuse across different scopes is forbidden; recovery of a preserved checkpoint for the exact same scope and lineage is allowed. |
| N3: optional joint-arm failure unclear | Clarified: budget excess, nonfinite computation, or declared component collapse rejects the arm. Alpha following the derived approximation-error-biased optimum is correct behavior, not failure. |
| N4: hot basin forgetting undefined | Replaced by a predeclared hot declared-region protocol using sign-region visits, indicator ESS/MCSE, replica identity, and start-stratified equivalence. It is not called formal basin evidence. |
| N5: swap call accounting unclear | Clarified by actual operation. The q=20 bridge can recombine cached `log g0` and `log L` at any beta without another filter call; a generic bridge may need two cross-beta calls. Cache hits, recombinations, and new target calls are counted separately and identically across arms. |

## Source checks supporting the repairs

- `bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py:405-420`
  binds the fixed observation covariance, while lines `451-477` construct the
  q=20 likelihood and add the four-dimensional Gaussian prior.
- `bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py:1933-1943`
  selects `alpha=1`, `beta=2`, `kappa=0`; lines `2344-2349` form each
  innovation covariance as a weighted observation-point covariance plus the
  fixed observation covariance; and lines `2400-2432` evaluate the Gaussian
  log-density.
- `bayesfilter/nonlinear/sigma_points_tf.py:154-188` gives the corresponding
  covariance weights, which are nonnegative at those settings.
- `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-result-2026-08-09.md`
  supports only the two known MAP-containing observation-weight half-spaces;
  it explicitly does not establish formal basin membership.
- `bayesfilter/testing/distributed_replica_exchange_tf.py` is diagnostic and
  uses the pure-power route, so it cannot be promoted as the matched physical
  proper-bridge comparator.

A focused CPU configuration diagnostic was run with GPU devices deliberately
hidden. It instantiated the q=20 target and its exact declared unscented rule,
and returned: augmented sigma-point dimension `80`, horizon `30`, observation
dimension `1`, minimum covariance weight `0.00625`, all covariance weights
nonnegative, and fixed observation variance
`0.35788974483896524`. This checks the current source configuration used by the
properness derivation; it is not itself the proof and must be rerun when any
bound source changes.

## Readiness

The corrected plan is ready for Phases 0--7 implementation and routine
fixtures. No serious q=20 campaign is authorized by this revision. That
campaign remains blocked until its subplan freezes a bounded compute budget,
component and temperature candidates, optional-joint-arm feasibility envelope,
ESS/MCSE and declared-region travel criteria, confirmation streams, and attempt
cap.
