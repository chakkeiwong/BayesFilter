# P4 Subplan: Parameterized Predator-Prey

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `READY_FOR_CELL_LOCAL_TARGET_REPAIR`

## Phase Objective

Independently repair and test the parameterized predator-prey posteriors
`PP-SGQF` and `PP-UKF`, and close `PP-ZC` honestly as
`TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH`. No Zhao-Cui HMC or training is in scope
because no production-admissible source route exists.

## Inherited Entry Conditions

- P0 registry and P1 shared harness are admitted. P2/P3 are complete with
  blocked cells and are not predator-prey evidence.
- All three P4 cells enter `TARGET_BLOCKED`; no P0 scope identity is eligible
  for HMC or training.
- Freeze the existing seed-81104 T=20 trajectory from
  `_predator_prey_dataset(81104)`: state hash
  `ebd1caca85d589bfa61801e92b112b7a8e0b9d5504763cdb67b82100422f7da2`
  and observation hash
  `dc63294b6e77913aef0c92796dd2d3c7a1721a766f976fcc392cd02a70754387`.
- Freeze the six physical parameters `(r,K,a,s,u,v)` and the model-declared box
  `[(0.1,1.1),(110,130),(20,30),(0.1,1.1),(0,1),(0,1)]`.
  Use independent physical Uniform distributions on that box and a six-probit
  chart `lower_i + width_i Phi(u_i)` with the complete Jacobian. Truth
  `(0.6,114,25,0.3,0.5,0.5)` is explanatory only.
- Freeze model settings: initial `N((50,5),I2)`, RK4 delta 2.0, internal step
  0.1, process covariance `4 I2`, observation covariance `4 I2`, and
  `diagnose_negative_after_noise`. Negative states are allowed and occur in the
  frozen trajectory; positivity projection is forbidden.
- Freeze the generator's time convention: `y0` observes the initial Gaussian
  state before any RK4 transition. Assimilate `y0` analytically under
  `x0~N((50,5),I2)` and `y0|x0~N(x0,4I2)`, then run RK4 plus process noise for
  `y1:T-1`. Historical P47 wrappers transition before their first observation
  and are wrong relative to this frozen data-generating target.
- Existing relevant anchors include `PredatorPreySSM` in
  `bayesfilter/highdim/models.py` and SGQF/UKF closures and value/score tests in
  `tests/highdim/test_p47_predator_prey_filtering.py`.
- `PP-ZC` has checked paper/math and author-source anchors or remains blocked.

## Target And Cell Scope

- `PP-UKF`: first repair, using the existing production batched
  principal-square-root UKF forward/reverse engine with graph-native RK4
  transition and analytical state/source-coordinate Jacobians.
- `PP-SGQF`: second repair, using a repository-owned fixed-cloud graph recurrence
  with `tf.while_loop`, tensorized batch/cloud axes, and manual forward
  sensitivities. Candidate levels must be frozen before its admission run.
- `PP-ZC`: terminal `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH`. The only current
  route is the generic all-axes retained-grid diagnostic, which project policy
  makes production-ineligible; no source-route substitute may be invented here.

Each cell has an independent signature, comparator, transport, archives, and
result. Same model/data does not make filter posterior samples interchangeable.

## Required Artifacts

- Per-cell posterior closures with prior/chart/data/filter identity and complete
  dependency hashes.
- SGQF and UKF dense/focused value/score and branch/status admission artifacts.
- Zhao-Cui source-anchor/classification blocker record; no Zhao-Cui target
  binding, HMC, or training artifact is required or allowed.
- Per-admitted-cell plain-HMC, recipe/default audit, recipe screen, selected fresh
  5,000-step training, frozen transport, NeuTra confirmation, separate archives,
  agreement, health/convergence, and repair artifacts.
- P4 result/run manifest, state/budget/hash ledgers, and refreshed P5 subplan.

## Required Checks And Reviews

1. Record `PP-ZC` blocked without executing the generic retained-grid route as
   production evidence.
2. Implement `PP-UKF` and `PP-SGQF` as separate batch-native `[B,6]` posterior
   adapters. Active target paths may not contain Python time/batch/cloud loops,
   NumPy, callbacks, eager conversions, or scalar row mapping.
3. Verify six-probit prior plus Jacobian equals independent standard-normal
   density in source coordinates and bind all data/model/filter tensors and
   callable dependencies.
4. Require analytic `y0` update checks, centered-FD score stability, batch
   permutation, status/finite/domain diagnostics, CPU XLA, and trusted GPU XLA
   with memory growth. Compare the historical transition-before-first-
   observation wrapper as a negative control and require a nonzero mismatch;
   do not use historical T=2 parity as admission evidence.
5. The historical positive-state tensor grid is invalid for T=20 because the
   frozen path includes negative states. Do not use it as a production oracle.
   Before either filter-admission run, create a bounded TensorFlow bootstrap-PF
   reference design over fixed audit points with stateless seeds, particle-count
   ladder, repeated seeds, paired uncertainty intervals, ESS/resampling/domain
   diagnostics, and a predeclared practical gap. PF evidence is a stochastic
   filter diagnostic, not an exact likelihood oracle.
6. A filter may pass only if the PF ladder itself stabilizes and the filter gap
   satisfies the predeclared simultaneous uncertainty/equivalence rule. If the
   reference does not stabilize within its frozen budget, record
   `TARGET_BLOCKED_REFERENCE_INCONCLUSIVE`; do not tune a filter threshold after
   inspection.
7. For Zhao-Cui, preserve the checked route classification and blocker. Do not
   execute R1-R4 or attempt target binding for `PP-ZC`; the generic retained-grid
   diagnostic cannot become the production evaluator.
8. Pass R1B independently per eligible SGQF/UKF cell, recomposing prior, filter likelihood, full
   chart Jacobian, total unconstrained value/score, and substitution-negative
   tests without calling the production final target assembler.
9. Execute R2-R4 separately for SGQF/UKF cells that pass R1B.
10. Audit capacity, optimizer, scaling, and tuning choices per cell. One
   predator-prey filter's settings may only warm-start another.
   Preserve tried/selected/rejected/untried candidate-family ledgers.
7. Apply cell-specific comparator equivalence and filter-reference roles frozen
   in P0; do not rank three cells from descriptive summaries.

## Evidence Contract

| Field | P4 contract |
| --- | --- |
| Question | Which predator-prey filter posteriors can be sampled by target-specific NeuTra consistently with identical-target plain HMC? |
| Comparator | Separate tuned plain HMC for each exact filter-posterior signature |
| Filter reference | Stabilized multi-seed TensorFlow bootstrap-PF likelihood diagnostics plus T=2 dense/route parity and declared domain invariants; PF is not called exact |
| Primary pass | Each eligible SGQF/UKF cell independently passes R1-R4 and its frozen agreement gate; `PP-ZC` closes only with its explicit source-route blocker |
| Vetoes | Incomplete posterior binding; wrong parameter order/chart/prior/data; filter value/score/reference failure; source-ungrounded/generic-grid Zhao-Cui; target/artifact mismatch; health or convergence/agreement failure |
| Explanatory only | Loss, acceptance, runtime, truth distance, descriptive SGQF/UKF/Zhao-Cui gaps |
| Not concluded | Filter ranking/superiority, calibration, exactness, broad predator-prey robustness, production readiness |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| seed-81104 T=20 trajectory | existing production benchmark row | one synthetic path becomes universal evidence | fixed hashes and explanatory-only truth | frozen fixture |
| independent Uniform parameter-box prior and six-probit chart | model-declared support | scientific prior may be too convenient | prior/Jacobian identity and nonclaim | baseline target choice |
| fixed model/noise/RK4 settings | `PredatorPreySSM` contract | inherited fixture settings may not represent another task | bound manifest and domain replay | frozen for this cell only |
| production batched principal-square-root UKF | existing public engine | RK4 adapter/derivatives or time-zero semantics drift | analytic y0 update, manual RK4 derivative FD, whole-target FD, XLA | reviewed engine plus new target adapter |
| graph-native fixed SGQF | existing scalar math, new execution topology | branch, time-zero, or derivative drift | analytic y0 update, recurrence FD, XLA | hypothesis requiring admission |
| bootstrap PF reference | need for negative-state T=20 diagnostic | Monte Carlo noise or degeneracy misclassifies filter | particle/seed ladder and intervals | design must be frozen before run |
| plain dense IAF | current capability inventory | insufficient six-dimensional capacity | target-specific screen | only available learned family; cell rejection unavailable |
| enhanced learned family | capability inventory | family does not exist in current code | fail-closed family inventory | unavailable and not executable |

Training architecture, optimizer, heldout design, HMC settings, and posterior
equivalence margins remain unfrozen until a cell obtains a typed target identity.

## Repair Triggers

Posterior-binding, recomposition, or score failures re-enter R0/R1/R1B. A
filter approximation veto records `FILTER_CANDIDATE_REJECTED` unless a
predeclared repair exists. Zhao-Cui source or route mismatch blocks `PP-ZC`
only. A failed training recipe records `RECIPE_REJECTED`; cell rejection
requires the runbook's complete candidate-family accounting. Never tune the
transport to conceal an invalid filter value/score route.

## Forbidden Claims And Actions

- No cross-filter transport/comparator/archive reuse.
- No generic retained-grid Zhao-Cui route promoted to production evaluation.
- No source-faithfulness without checked anchors.
- No post-result search expansion/ranking, NumPy or sample loops in active
  paths, CPU serious training, warm-up pooling, or overwrite.

## Handoff Conditions

P5 begins only when all three cells have honest terminal states, including the
predeclared `PP-ZC` blocker, admitted artifacts are hash-bound, P4
checks/result/manifest and repair history are complete, and
P5 is refreshed with a graph-native structural posterior design, identity
vetoes, negative control, exact commands, and remaining budget.

## Stop Conditions

Block individual cells for target/reference/source/comparator invalidity,
unrepairable implementation, three identical failed repairs, or cell budget
exhaustion. Continue other P4 cells, but do not enter P5 until every P4 cell has
a recorded terminal state and the P4 phase result is complete. Stop program-wide
only for shared harness contamination, corrupted common data/evidence, or
program budget veto.

## Compute And Attempt Budget

Aggregate ceiling: 120 trusted GPU wall-hours plus 24 CPU reference hours. Each
admitted SGQF/UKF cell may use one 15-GPU-hour plain dense-IAF arm for one
screen, one selected fresh 5,000-step training, one NeuTra confirmation, and
arm-local retries. The unavailable enhanced arm is not executed, its nominal
budget is not reassigned, and failure of the only available family cannot yield
cell-wide NeuTra rejection. `PP-ZC` consumes no HMC/training arm. A separate
6-hour bucket funds plain HMC and comparator retries; 4 hours fund trusted
R0/R1/R1B cell admission, cell-specific adapter/artifact emission, and their
repairs. Common harness/schema/reporting defects reopen and charge P1 only.
Three localized repairs apply per identical failure within the owning bucket.
P3 refresh freezes command-level allocation.

## Skeptical Pre-Execution Audit

Decision: `PASS_FOR_UKF_THEN_SGQF_IMPLEMENTATION; PF_DESIGN_REQUIRED_BEFORE_FILTER_RUN`.

The old two-step positive-state grid is a wrong baseline for the T=20 trajectory
and is explicitly retired from production admission. Proxy parity cannot issue
an identity. The PF reference must declare uncertainty and stabilization before
execution. `PP-ZC` is blocked by policy rather than relabeled. Cell-local
failure or inconclusive reference does not halt the remaining P4 cells or P5.

Code-level audit addendum: the generator/wrapper time-order mismatch was found
before implementation. The plan now binds initial-observation assimilation and
forbids preserving the historical transition-before-y0 route merely for parity.
