# P3 Subplan: KSC SV Principal-Square-Root UKF

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `READY_FOR_BOUNDED_TARGET_ADMISSION_IMPLEMENTATION`

## Phase Objective

First repair and admit `KSC-UKF`, the scalar KSC seven-component
Gaussian-mixture transformed-SV posterior defined by the component-enumerated
principal-square-root UKF recurrence. Only after a complete repository-issued
typed identity may the cell proceed to same-target plain HMC, target-specific
NeuTra screening, fresh 5,000-step GPU/XLA training, and confirmation.

## Inherited Entry Conditions

- P0 attempt 04 and the P1 typed-identity/state/archive/device harness are valid.
- P2 is complete with both exact-SV cells `TARGET_BLOCKED`; neither P2 result is
  evidence for or against this KSC cell.
- `KSC-UKF` enters as `TARGET_BLOCKED`; the P0 inventory identity is ineligible
  for HMC, training, transport loading, or claims.
- Use the same preserved seed-81101 T=1000 raw observation trajectory as a
  shared data source, with observation hash
  `5e2423149e4f59eb588ccc7f16ec6d9ee984ccc4710a3ae07a3dbcf5c37db748`.
  KSC applies the distinct transform `log(y^2 + 1e-8)` and a distinct mixture
  likelihood, so its target signature cannot equal either P2 signature.
- Freeze physical truth `(gamma,beta,sigma)=(0.6,0.4,1.0)` as explanatory only;
  infer `gamma,beta` with `sigma=1.0` fixed.
- Freeze the source prior `gamma,beta ~ Uniform(0.1,0.9)` independently and the
  two-probit chart `gamma=0.1+0.8 Phi(u0)`,
  `beta=0.1+0.8 Phi(u1)`, including the complete chart Jacobian. This is the
  same source prior family used for the shared synthetic model, not evidence
  transferred from the P2 filter.
- Freeze `ksc_1998_log_chi_square_mixture()` including its weights, shifted
  means, variances, source string, and dependency closure.
- Existing anchors are `independent_panel_sv_mixture_ukf_filter` and
  `independent_panel_sv_mixture_ukf_score` in
  `bayesfilter/highdim/sv_mixture_cut4.py`. They are tiny-fixture reference
  routes: they contain Python time/component loops and eager validation and are
  not eligible as the serious batched/XLA target.

## Target And Cell Scope

Mandatory: `KSC-UKF` only. Optional KSC SGQF or Zhao-Cui controls require
separate P0 registry rows/signatures and cannot change the primary cell result.

## Target-Admission Implementation

Implement a repository-owned scalar graph-native adapter with these fixed
properties:

- TensorFlow `float64`, input `[batch,2]`, no scalar fallback;
- a `tf.while_loop` over T=1000 and tensorized enumeration of all seven KSC
  components; no active Python time, component, or batch loop;
- direct scalar Gaussian moment algebra that is mathematically equal to the
  component principal-square-root UKF update for this affine component model;
- total source-coordinate value/score/status for prior, KSC-UKF likelihood, and
  chart Jacobian;
- status telemetry for finite values/scores, strictly positive predictive and
  innovation variances, normalized mixture weights, and covariance validity;
- semantic target identity bound to data, transform offset, mixture identity,
  initial stationary law, moment-collapse rule, prior/chart/Jacobian, dtype,
  adapter callable closure, status callable closure, and independent posterior
  recomposition.

The equality claim is narrow: for each fixed Gaussian-mixture component the
state transition and observation are affine, so sigma-point moments equal the
Gaussian predictive moments and the principal-square-root UKF update equals the
scalar Kalman update. The mixture moment collapse remains the declared KSC-UKF
approximation. A checked derivation and reference parity test are required; the
direct recurrence must not be called exact KSC filtering.

## Required Artifacts

- KSC target-repair ledger and mathematical equivalence derivation.
- KSC/UKF target replay, route/dependency, mixture identity, and transform
  manifests.
- Graph/reference parity, score FD, dense-filter comparison, component
  enumeration, variance/status, batch, CPU-XLA, and trusted GPU/XLA artifacts.
- Tuned same-target plain-HMC result and separate archives.
- Target-specific training/default audit, recipe screen, selected fresh
  5,000-step GPU/XLA training, frozen transport, and manifest.
- Fresh NeuTra tuning/confirmation, comparator agreement, health/convergence,
  phase result, repairs, ledgers, and refreshed P4 subplan.

## Required Checks And Reviews

1. Replay the raw data hash and independently verify the KSC transform hash,
   offset, seven mixture weights/means/variances, stationary initial law, and
   deterministic component order.
2. Derive and test the graph recurrence against the existing
   principal-square-root score wrapper on T=1 and T=2 at fixed audit points.
   Value parity to the historical SVD-factor value wrapper is explanatory only;
   the admissible identity binds the new repository-owned affine-equivalent
   principal-square-root recurrence and its mathematical contract.
3. Require batch singleton/parity/permutation, finite/status, centered-FD score,
   CPU XLA, trusted GPU XLA, memory growth, and no active NumPy/callback/Python
   time-component-sample loop.
4. Before execution, use fixed source-chart audit points `(-1,-1)`, `(-1,1)`,
   `(0,0)`, `(1,-1)`, `(1,1)`, and transformed physical truth.
5. Compare the KSC-UKF likelihood against
   `scalar_sv_mixture_dense_reference` on the first 20 observations using
   Legendre order 401 and radius 8. Require maximum absolute per-observation
   value gap `<=1e-3` and maximum absolute source-coordinate score gap to a
   centered finite difference of the dense reference `<=1e-2`. Freeze the
   source-coordinate centered-FD step lattice to `1e-4`, `3e-5`, and `1e-5`.
   Before interpreting the score gap, require all estimates finite, maximum
   coordinate disagreement between the `3e-5` and `1e-5` order-401 estimates
   `<=2e-3`, and maximum coordinate disagreement between order 401 and order
   601 at step `3e-5` `<=2e-3`; use the order-601 `3e-5` estimate as the score
   reference. These are filter-admission gates, not NeuTra promotion metrics.
   Dense-reference order 601 must also agree with order 401 within `2e-4` per
   observation at the same points or the reference is invalid and must be
   repaired before admission.
6. Pass R1B with independent recomposition of KSC likelihood, prior, full chart
   Jacobian, total unconstrained value/score, and wrong-substitution tests for
   exact-SV transform, altered mixture, altered offset, and cross-cell adapter.
7. Only if all target/filter gates pass, issue the typed identity and plan the
   same-target plain-HMC comparator. HMC and training are not authorized by the
   target-admission artifact alone.
8. If admitted, execute R2-R4 with target-specific tuning/training settings and
   preserve every tried, selected, rejected, unavailable, and untried family.

## Evidence Contract

| Field | P3 contract |
| --- | --- |
| Question | First, can the frozen KSC principal-square-root-UKF approximate posterior be implemented and admitted without route, target, score, batching, XLA, or filter-reference drift; if so, can target-specific NeuTra sample it consistently with same-target plain HMC? |
| Comparator | Tuned plain HMC on `KSC-UKF` exactly |
| Filter reference | P0-frozen component/dense/focused KSC reference; never the exact non-Gaussian SV target |
| Primary pass | Target rung: graph/reference parity, score/status/batch/XLA, dense-filter gate, and independent posterior identity all pass. Cell rung: selected 5,000-step training, modern HMC/health, and simultaneous comparator agreement pass. |
| Vetoes | Exact-SV substitution; mixture/offset drift; unproved affine-equivalence route; nonpositive variance; value/score/filter gate failure; target mismatch; HMC diagnostic/health/agreement failure |
| Explanatory only | Loss, acceptance, runtime, filter gaps, truth distance, optional-control metrics |
| Not concluded | Actual/exact SV validity, mixture approximation adequacy to actual SV, filter/recipe superiority, calibration, broad robustness, production readiness |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| seed-81101 T=1000 raw trajectory | P0 candidate and P2 frozen replay | isolates target/filter change on reproducible data | cross-target result reuse or transform confusion | raw and transformed hashes | frozen data source, not transferred evidence |
| two-probit Uniform prior/chart | Zhao-Cui synthetic experiment and P2 source audit | source-grounded bounded inference problem | KSC target could require another scientific prior | prior+Jacobian standard-normal identity | baseline target choice |
| fixed sigma 1.0 | synthetic data-generating model | retains two-dimensional target and existing score surface | understates uncertainty | explicit target signature/nonclaim | reviewed scope choice |
| KSC seven-component mixture and offset 1e-8 | repository KSC route | defines the requested distinct target | mixture/offset drift changes posterior | exact payload/hash tests | required |
| affine-equivalent direct scalar recurrence | checked component mathematics | removes Python loops while preserving principal-square-root component moments | algebra or collapse derivative differs from wrapper | T=1/T=2 parity and FD | hypothesis requiring admission |
| dense prefix order 401/radius 8 with order-601 and FD-lattice checks | P2 dense-reference design plus review repair | bounded independent filter diagnostic | truncation, quadrature, or differencing error masquerades as UKF bias | value order convergence, FD step stability, and score order convergence | reviewed diagnostic design |
| `1e-3` value and `1e-2` score filter margins | P2 value scale and looser derivative approximation allowance | prevents a materially drifting filter posterior from reaching HMC | strict margin may reject useful approximation | fixed six-point prefix audit | hypothesis, frozen before result |
| plain dense IAF only | capability inventory | only implemented learned family | failure cannot reject all NeuTra | family ledger | baseline; cell-wide rejection unavailable |

Architecture, optimizer, batch, heldout, seeds, HMC tuning, and posterior
equivalence margins remain deliberately unfrozen until target admission. They
must receive a KSC-specific protocol before R2-R4; LGSSM or exact-SV settings are
warm-start hypotheses only.

## Repair Triggers

Graph/reference or score failure re-enters target implementation. Invalid dense
order convergence repairs only the reference. A valid recurrence that fails the
frozen dense-filter margin leaves `KSC-UKF` `TARGET_BLOCKED`; it does not permit
post-result margin changes or HMC. Training and sampler failures use the
standard cell-local taxonomy. A failed training recipe is `RECIPE_REJECTED`,
not a cell-wide NeuTra rejection.

## Forbidden Claims And Actions

- No actual/exact-SV evidence substitution or claim.
- No hidden switch to SGQF, Zhao-Cui, exact SV, or a differently defined Kalman
  mixture. Direct scalar Gaussian algebra is admissible only with the checked
  affine principal-square-root equivalence contract and parity evidence.
- No post-result component/UKF setting or margin tuning.
- No active-path NumPy/host callback/sample loop, CPU serious training, archive
  pooling, or artifact overwrite.

## Handoff Conditions

P4 begins when `KSC-UKF` has an honest terminal state, the result/manifest and
ledgers are complete, every failed attempt is classified, and P4 is refreshed
with separately frozen predator-prey SGQF/UKF/Zhao-Cui routes and budgets.
`TARGET_BLOCKED` after the frozen filter gate is a valid terminal P3 state and
does not stop P4.

## Stop Conditions

Block `KSC-UKF` for invalid mixture target, unrepairable graph/principal-square-
root equivalence, failed frozen filter gate, absent same-target comparator,
bucket exhaustion, or three materially identical failed repairs. Continue to
P4 unless shared harness/evidence validity is implicated.

## Compute And Attempt Budget

Ceiling: 40 trusted GPU wall-hours plus 8 CPU reference hours. Reserve two
15-GPU-hour family arms: plain dense IAF and one P0-frozen KSC-specific enhanced
family. Each arm permits one screen, one selected fresh 5,000-step training,
one NeuTra confirmation, and arm-local retries. A separate 6-hour bucket funds
plain HMC and comparator retries; 4 hours fund trusted R0/R1/R1B cell admission,
cell-specific adapter/artifact emission, and their repairs. Common
harness/schema/reporting defects reopen and charge P1 only. Three localized
repairs apply per identical failure within the owning bucket.

## Skeptical Pre-Execution Audit

Decision: `PASS_FOR_BOUNDED_TARGET_ADMISSION_IMPLEMENTATION_ONLY`.

| Required challenge | Finding and control |
| --- | --- |
| Wrong baseline | Dense KSC-mixture latent-state integration is the filter comparator; exact log-chi-square and Kalman projection are not substituted. |
| Proxy promotion | Wrapper parity and FD establish engineering correctness only; dense-prefix agreement is separately required for filter admission. |
| Missing stop conditions | Failed target/filter gate blocks only this cell; shared corruption/budget is the only program veto. |
| Unfair comparison | Raw data may be shared, but KSC transform, mixture, prior, callable closure, and identity are bound independently. |
| Hidden assumptions | Data, prior/chart, sigma, offset, mixture, audit points, dense orders/radius, margins, dtype, device, and route equivalence are explicit above. |
| Stale context | The legacy route is tiny-fixture/Python-loop code and cannot be promoted directly. P2 supplied process lessons only. |
| Environment mismatch | CPU is limited to references/tests; trusted GPU/XLA with memory growth is mandatory for the admission canary and all later serious work. |
| Artifact insufficiency | A target identity is issued only after graph, reference, filter, recomposition, and substitution-negative evidence. |
| Misleading successful command | A compiling recurrence that fails dense-filter admission remains blocked and cannot reach HMC. |

Pre-mortem: the graph route could pass parity while both implementations share
a moment-collapse error, so the dense reference is independent and mandatory.
The dense reference could itself be under-resolved, so its values must converge
from order 401 to 601 and its centered-FD scores must pass both the frozen step
lattice and order-convergence gates before any score gap is interpreted. The
UKF approximation may simply fail the frozen filter margin; that is a cell
result, not a repairable harness failure or a reason to stop P4-P7.
