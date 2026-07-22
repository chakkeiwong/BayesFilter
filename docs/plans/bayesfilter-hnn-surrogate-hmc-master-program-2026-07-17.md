# BayesFilter Corrected Neural-Force HMC Master Program

Date: 2026-07-17

Program ID: `bayesfilter-corrected-neural-force-hmc-20260717`

Status: `COMPLETE`

Supervisor/executor: Codex in the active BayesFilter session.

Advisory reviewer: Claude may perform bounded, read-only reviews. Claude does
not edit, execute, authorize, or change a scientific boundary.

Execution runbook:
`docs/plans/bayesfilter-hnn-surrogate-hmc-execution-runbook-2026-07-17.md`.

## Objective

Implement the narrower, proved neural-force HMC construction in Chapter 48 and
test it first on every BayesFilter model/filter configuration that has already
completed retained NeuTra HMC under the current sampler-health and truth-tail
standards. Reuse the admitted target adapters, typed identities, training/HMC
infrastructure, and diagnostic policy rather than rediscovering them.

The program asks two separate questions:

1. Does a frozen position-only learned force, used in a symmetric
   kick-drift-kick map and corrected with the true joint endpoint energy,
   preserve and recover the declared filter posterior?
2. After charging training and true endpoint values, does that corrected kernel
   pass the predeclared descriptive computational-viability screen against
   true-gradient HMC on the same target?

A validity pass does not establish a speed advantage. A speed observation does
not establish validity.

## Binding Method

The primary candidate is deliberately narrower than a general Hamiltonian
neural network.

- Work in the target-specific frozen NeuTra coordinate chart `z` when an
  admitted chart exists.
- Define the true chart potential by
  `U_z(z) = U_F(T(z)) - log|det dT(z)/dz|`. Every endpoint correction in
  `z` uses this complete scalar. Evaluating only `U_F(T(z))` is wrong relative
  to the transformed-target claim.
- Learn a scalar residual potential
  `U_hat(z) = 0.5 z^T M0^{-1} z + r_phi(z)` and obtain its position-only force
  by TensorFlow automatic differentiation.
- Freeze network weights, normalization, chart, mass matrix, step size, and
  leapfrog count before retained sampling.
- Propose with a symmetric kick-drift-kick map, apply a momentum flip, and use
  the actual target value plus both initial and final kinetic energies in the
  Metropolis ratio.
- Cache the current true target value and evaluate one new true endpoint value
  per proposal.
- Do not implement the historical missing-stage-one delayed-acceptance route.
- Do not use an arbitrary neural state update or a momentum-dependent learned
  Hamiltonian as the primary candidate.

The primary candidate is the pure corrected neural-force kernel. A later
fixed-probability mixture with another valid kernel is mathematically allowed
only as a separately labeled follow-up after the pure kernel is classified. A
mixture can receive `MIXTURE_VALIDATED`, but never
`HNN_VALIDITY_CONFIRMED`. State-dependent rescue or early termination is
outside the proved construction.

## Research Intent Ledger

| Field | Binding intent |
| --- | --- |
| Main question | On which already-admitted NeuTra model/filter targets does corrected neural-force HMC pass target validity, sampler health, truth-tail, and the separately declared descriptive performance screen? |
| Candidate | Frozen scalar residual potential in the admitted target-specific NeuTra chart, symmetric position-force integrator, true joint endpoint MH correction |
| Mechanism | Replace every expensive interior filtering-score call by a learned-force call while retaining one true endpoint posterior value |
| Exact baseline | Tuned raw-coordinate plain HMC on the same target plus true-gradient HMC in the same admitted NeuTra chart; matching preserved evidence may be reused after identity/diagnostic replay |
| Additional baseline | Zero-residual Gaussian force in the same chart; it tests whether learning adds anything beyond chart geometry |
| Expected failures | Wrong endpoint scalar, omitted kinetic term, non-involutive executed map, hidden momentum dependence, poor force generalization, low acceptance, nonconvergence, target/chart mismatch, training or XLA failure, or no amortized efficiency |
| Validity criterion | Per cell: hard validity and health gates pass, final modern R-hat and ESS gates pass, the truth-tail rule passes where generating truth exists (otherwise the frozen target-appropriate reference rule passes), and the exact cost ledger is complete. This yields `HNN_VALIDITY_CONFIRMED` independently of performance. |
| Performance criterion | `DESCRIPTIVE_PERFORMANCE_SCREEN_PASS` requires validity plus observed reuse-scenario end-to-end seconds per minimum bulk ESS no greater than same-chart true-gradient HMC, and observed post-compilation sampling-only seconds per minimum bulk ESS strictly lower. The reuse scenario charges residual-force training/tuning and HNN sampling; any NeuTra chart cost is treated identically in both chart arms. This is a one-campaign descriptive screen, not a statistically supported superiority claim. Otherwise record `PERFORMANCE_NOT_DEMONSTRATED`. |
| Promotion veto | Target/signature mismatch; repeated endpoint nondeterminism; map/reversal/Jacobian test failure; undefined force or target status; invalid full-energy ratio; divergence/energy veto; R-hat/ESS cap failure; or severe `p_truth < 0.003` |
| Continuation veto | Shared kernel or target-adapter invalidity; corrupted irreplaceable evidence; unavailable trusted GPU; exhausted program budget; or a repair requiring target, method, criteria, hardware, privacy, environment, or scientific-scope change |
| Repair trigger | Local implementation, XLA, memory, serialization, training, tuning, sampler, or reporting failure under the unchanged contract |
| Explanatory only | Training loss, force RMSE/cosine, acceptance, step size, energy-error quantiles, endpoint/gradient count, and runtime without uncertainty support |
| Must not be concluded | Filter exactness, calibration, universal HNN validity, superiority, production/default readiness, or validity of the broader L-HNN map criticized in Chapter 48 |

## Scope Matrix

### Tier A: Mandatory BayesFilter confirmations

These are the model/filter configurations with retained learned-NeuTra HMC and
preserved truth-tail evidence. They are the binding implementation matrix, not
a claim that their historical evidence has one uniform strength. The current
HNN campaign applies one prospective contract to all five and does not inherit
an earlier threshold exception.

| Cell | Model/filter target | Existing evidence | HNN program role |
| --- | --- | --- | --- |
| `LGSSM-KF` | LGSSM, exact Kalman likelihood | central one-seed truth-tail pass; independent fixture evidence | correctness and performance pilot |
| `PP-UKF` | parameterized predator-prey, fixed UKF likelihood | qualified: six-mean confirmation and retrospective noncentral truth-tail pass | nonlinear approximate-filter cell; fresh HNN evidence uses current gates |
| `PP-SGQF` | parameterized predator-prey, fixed SGQF likelihood | qualified: six-mean confirmation and retrospective noncentral truth-tail pass | quadrature-filter cell; fresh HNN evidence uses current gates |
| `SIR-SGQF` | parameterized spatial SIR, fixed SGQF likelihood | three-mean confirmation; central one-seed truth-tail pass | spatial nonlinear cell |
| `STR-UKF` | Chapter 18b structural model, structural UKF likelihood | qualified: owner-adjudicated one-seed pass with folded R-hat | deterministic-structure stress cell; no prospective ESS exception |

All five use their existing repository-issued target identity and admitted
adapter. A filter-defined posterior remains filter-defined.

### Tier B: Requalification, not automatic evidence reuse

The cross-repository ledger records exactly eight further historical
configurations:

| Tier B ID | Configuration | Phase | GPU ceiling |
| --- | --- | --- | ---: |
| `FUNNEL` | paper-scale funnel | P6 | 6 h |
| `ILLGAUSS` | ill-conditioned Gaussian | P6 | 6 h |
| `GERMAN-LR` | German logistic regression | P6 | 6 h |
| `NK-ANALYTIC` | NK-like analytic surrogate | P7 | 6 h |
| `NK-REAL` | small/real NK local posterior | P7 | 6 h |
| `NK-SVD-UKF` | small/real NK principal-square-root SVD-UKF posterior | P7 | 6 h |
| `ROT-KF` | Rotemberg linear-Kalman posterior | P7 | 6 h |
| `ROT-SVD2` | Rotemberg second-order SVD posterior | P7 | 6 h |

The NK and Rotemberg pairs remain materially distinct posterior-target
configurations even when counted within one model family. Each Tier B ceiling
includes its one repair attempt; unused time is not transferred automatically
to another configuration.

Tier B begins only after Tier A infrastructure passes. Each route must first be
ported or bound to the corrected BayesFilter kernel, identify its exact target
and chart, and satisfy current diagnostics. Qualified historical status remains
qualified. Missing truth or target identity yields `REQUALIFICATION_BLOCKED`,
not a fabricated pass. Tier B may be split into standard analytic targets and
DSGE/filter targets for cost control.

Previously blocked NeuTra cells such as exact SV, KSC-UKF, Zhao-Cui routes, and
SIR-UKF are excluded. HNN tuning cannot repair a target/filter/source-route
blocker.

## Evidence Contract

| Requirement | Contract |
| --- | --- |
| Target | Identical prior, data, parameter transform, Jacobian, filter settings, dtype, and repository-issued identity across all compared arms. In NeuTra coordinates the endpoint scalar is `U_F(T(z))-log|det dT/dz|`, not the unadjusted raw target. |
| Executed map | Fixed position-only force; exact symmetric kick-drift-kick sequence; momentum flip; no clipping, early stop, state-dependent step, or host callback |
| Acceptance | `min(1, exp[-U(z')-K(p')+U(z)+K(p)])` from actual initial/final momentum and deterministic true target values |
| Engineering | TensorFlow/TFP; batched chains and training; GPU/XLA default; memory growth; no NumPy or Python sample-axis loop in active paths |
| Training | Target-specific protocol with objective/scaling check, capacity/LR screen, heldout force/value diagnostics, fresh selected run, frozen artifact, and downstream nomination |
| Tuning | Candidate selection uses disjoint tuning draws and full sampler/energy diagnostics, not acceptance alone; frozen kernel is reverified before retained evidence |
| Sampling | At least four chains; retained warm-up; recent-window modern R-hat `<=1.05`; retained growth to at most 10,000 draws per chain; final modern R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`. These are hard prospective gates. A post-result user threshold change creates a separately labeled `QUALIFIED_POSTHOC_ADJUDICATION`, not `HNN_VALIDITY_CONFIRMED`. |
| Truth tail | Where generating truth exists, use one seed first. All `p_truth >=0.05`: pass. Any `0.003 <= p_truth <0.05`: one fresh data seed. Any `p_truth <0.003`: failure/investigation. It is a posterior-tail diagnostic, not a frequentist p-value. Real-data targets require a separately frozen reference rule and no truth-tail claim. |
| Comparator | Mandatory tuned raw-coordinate plain HMC, true-gradient HMC in the same NeuTra chart, and zero-residual Gaussian-force arm. Matching preserved raw-HMC evidence may be reused only after target, policy, and diagnostic replay. |
| Performance | Report compilation, chart cost, residual training, tuning, endpoint values, true-gradient calls, learned-force calls, fallback, wall time, minimum bulk ESS, seconds per minimum bulk ESS, ESS/sec, and amortization horizon. Apply the validity-independent screen above. No superiority ranking without uncertainty support. |
| Artifacts | Versioned attempt root, run manifest, frozen target/force identities, raw warm-up and retained tensors, traces, diagnostics, cost ledger, result, repair record, and hashes |

## State Machine

```text
UNINVENTORIED
  -> TARGET_REPLAYED
  -> KERNEL_MECHANICS_ADMITTED
  -> TRAINING_PROTOCOL_ADMITTED
  -> FORCE_ADMITTED
  -> TUNING_ADMITTED
  -> HNN_VALIDITY_CONFIRMED
```

Side exits are `TARGET_BLOCKED`, `IMPLEMENTATION_BLOCKED`,
`TRAINING_RECIPE_REJECTED`, `TUNING_BLOCKED`, `SAMPLER_INCONCLUSIVE`,
`TRUTH_TAIL_FAILURE` and `REQUALIFICATION_BLOCKED`. Performance is an
orthogonal cell field with values `DESCRIPTIVE_PERFORMANCE_SCREEN_PASS` or
`PERFORMANCE_NOT_DEMONSTRATED`; it cannot change a validity failure into a
pass. A side exit is cell-local unless it invalidates the shared kernel or
target-binding harness.

## Phase Map

| Phase | Objective | Dedicated subplan |
| --- | --- | --- |
| P0 | Freeze method, Tier A/Tier B scope, identities, defaults, commands, and budgets | `docs/plans/bayesfilter-hnn-surrogate-hmc-p0-scope-subplan-2026-07-17.md` |
| P1 | Implement and prove-by-test the corrected batched neural-force HMC kernel | `docs/plans/bayesfilter-hnn-surrogate-hmc-p1-kernel-subplan-2026-07-17.md` |
| P2 | Implement target-specific scalar-force training, tuning, archival, and campaign integration | `docs/plans/bayesfilter-hnn-surrogate-hmc-p2-training-harness-subplan-2026-07-17.md` |
| P3 | Run LGSSM exact-Kalman pilot and decide whether multi-model execution is justified | `docs/plans/bayesfilter-hnn-surrogate-hmc-p3-lgssm-pilot-subplan-2026-07-17.md` |
| P4 | Run predator-prey UKF and SGQF cells | `docs/plans/bayesfilter-hnn-surrogate-hmc-p4-predator-prey-subplan-2026-07-17.md` |
| P5 | Run spatial SIR SGQF and structural UKF cells | `docs/plans/bayesfilter-hnn-surrogate-hmc-p5-sir-structural-subplan-2026-07-17.md` |
| P6 | Requalify standard non-filter targets: funnel, Gaussian, logistic | `docs/plans/bayesfilter-hnn-surrogate-hmc-p6-standard-targets-subplan-2026-07-17.md` |
| P7 | Requalify NK/Rotemberg configurations under explicit target adapters | `docs/plans/bayesfilter-hnn-surrogate-hmc-p7-dsge-subplan-2026-07-17.md` |
| P8 | Synthesize cell decisions, cost evidence, limitations, documentation, and reset memo | `docs/plans/bayesfilter-hnn-surrogate-hmc-p8-synthesis-subplan-2026-07-17.md` |

P4 and P5 may classify one cell and continue the other. P6 and P7 are Tier B;
they do not block a complete Tier A result if requalification is impossible
within the frozen budget.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Position-only scalar force | Chapter 48 project theorem | Gives an explicit reversible, volume-preserving discrete map | accidental momentum/network-state dependence invalidates theorem | P1 source/API guards and reversal test | binding method |
| NeuTra coordinates | existing admitted target-specific transports | isolates residual geometry and supplies strong baseline | inherited chart may make residual trivial or hide raw-coordinate cost | zero-residual arm and full cost ledger | primary chart, not proof of speedup |
| Residual Gaussian baseline | Chapter 48 | tests benefit beyond chart geometry | learned arm selected by loss but not downstream behavior | P2/P3 downstream screen | mandatory baseline |
| Frozen fixed-step HMC | Chapter 48 and shared HMC policy | matches proved proposal | dynamic stopping/fallback breaks symmetry | negative tests and trace schema | binding |
| Reverse-KL plus score/value supervision | historical surrogate and NeuTra work | plausible target-specific training start | mode collapse or local force accuracy only | heldout shell/tail points and downstream HMC | hypothesis, P0 must freeze weights |
| Tier A first | current truth-tail ledger | highest-quality reusable target evidence | easy LGSSM overgeneralized | P3 promotion only unlocks, never proves later cells | reviewed schedule |
| One seed first | owner cost policy | controls expensive GPU use | fixture luck mistaken for reliability | exact one-seed label and conditional second seed | binding diagnostic policy |
| Reuse-scenario performance screen | program objective and existing admitted NeuTra charts | asks whether adding a residual force pays in the actual knowledge-transfer setting | one campaign mistaken for statistical superiority; chart sunk costs hidden | report reuse and from-scratch ledgers separately and label the result descriptive | reviewed decision rule, not a universal threshold |
| Existing NeuTra artifacts | completed campaigns | avoid needless retraining of charts | stale signature or artifact reuse across target | identity/hash replay | reusable only after replay |
| GPU/XLA and memory growth | repository policy | required learned-workload route | sandbox or allocator artifact misdiagnosed | escalated device probe and canary | required |

P0 must freeze the objective terms, normalization, architecture grid, learning
rates, batch sizes, training-step ladder, training region, seeds, tuning grid,
energy thresholds, and performance accounting. Inherited choices are
warm-start hypotheses until target-specific evidence supports them.

## Compute And Attempt Budget

Initial ceiling for the complete program is 22 CPU wall-hours and 120 trusted
GPU wall-hours. P0 must issue command-level sub-budgets before serious launch.
Planning allocation:

| Phase | CPU ceiling | GPU ceiling | Repair attempts |
| --- | ---: | ---: | ---: |
| P0 | 4 h | 0 h | 1 |
| P1 | 4 h | 2 h | 3 |
| P2 | 4 h | 6 h | 3 |
| P3 | 2 h | 12 h | 2 |
| P4 | 2 h | 24 h | 2 per cell |
| P5 | 2 h | 28 h | 2 per cell |
| P6 | 1 h | 18 h | 1 per configuration, included in each 6 h ceiling |
| P7 | 1 h | 30 h | 1 per configuration, included in each 6 h ceiling |
| P8 | 2 h | 0 h | 1 |

Unused budget is not a target. A clean early stop conserves it. Materially
expanding the total ceiling, hardware class, model matrix, or method requires
user direction.

## Phase-End Contract And Repair/Continue Rule

At the end of every subplan, the supervisor must:

1. Run its required local checks and serious-run validations.
2. Write a phase result or blocker/repair close record with manifest, decision
   table, inference-status table, raw evidence paths, wall time, and remaining
   budget.
3. Draft or refresh the next subplan from actual evidence rather than the
   original forecast.
4. Review the next subplan for wrong baselines, proxy promotion, missing stop
   conditions, unfair comparisons, stale assumptions, environment mismatch,
   feasibility, artifact coverage, and filtering-boundary safety.
5. Continue automatically when no true continuation veto fired.

For a localized failure under an unchanged scientific contract:

1. Preserve the failed attempt in its versioned directory.
2. Classify it as harness, target identity, map mechanics, training, tuning,
   sampler, truth-tail, performance, or reporting.
3. Write a concise repair record with root-cause hypothesis, smallest patch,
   focused regression, invalidated rung, budget charge, and rollback rule.
4. Patch narrowly, run the focused regression, and retry in a fresh directory.
5. After the phase-specific identical-failure ceiling, block only the affected
   cell or phase and continue independent work.

A candidate failure is not a research-direction failure. Stop the program only
if the target, shared kernel, data, mathematical proof assumptions, evidence,
hardware, or budget is invalid, or the needed repair crosses a human boundary.

## Skeptical Pre-Execution Audit

| Risk | Resolution |
| --- | --- |
| Wrong baseline | Same-target true-gradient HMC and zero-residual force are mandatory; an untuned raw HMC is not the sole comparator. |
| Proxy promotion | Loss, RMSE, force cosine, acceptance, and short-chain output cannot promote a force. Only corrected downstream chains can. |
| Missing stop conditions | Target/map/energy/health/truth/cap vetoes and phase budgets are explicit. |
| Unfair speed claim | Compile, training, endpoint values, gradient/force calls, fallback, warm-up, retained sampling, and ESS uncertainty are charged. |
| Trivial chart advantage | All HNN arms share the same admitted NeuTra chart; zero-residual and true-gradient chart baselines isolate the learned residual. |
| Both methods share wrong target | Reuse only repository-issued target identities and recomposition evidence; repeat target replay before each cell. |
| Chart Jacobian omitted at correction | Bind and test the complete transformed endpoint scalar `U_F(T(z))-log|det dT/dz|`; raw posterior energy alone is a hard target-identity failure. |
| Filtering boundary drift | Exact Kalman is latent-model exact; UKF/SGQF results are exact only for their named deterministic filter posterior. |
| Broader L-HNN smuggled in | API forbids momentum input and direct state-update maps; Chapter 48 is the binding theorem. |
| One failed recipe overinterpreted | `TRAINING_RECIPE_REJECTED` is not HNN rejection; untried frozen arms remain open or budget-blocked. |
| Tier B historical inflation | Every historical configuration must be requalified; historical labels and missing truth remain visible. |
| Valid mixture hides pure-kernel failure | Pure HNN is classified first. Any fixed mixture is a separate follow-up and cannot receive `HNN_VALIDITY_CONFIRMED`. |
| Prospective gate changed after output | Current R-hat/ESS thresholds are hard. A later user change is disclosed as `QUALIFIED_POSTHOC_ADJUDICATION` and cannot retroactively produce the primary validity label. |
| Expensive run before mechanism check | P1 mechanics and P3 LGSSM pilot gate all later model runs. |

Audit decision: the program is suitable for bounded execution after P0 freezes
the command-level protocol and after one advisory plan review. The artifacts
produced by P1-P3 directly answer whether the proved method was implemented and
whether broader compute is justified.

## Review Record

Local skeptical review found and repaired three material issues before external
review: Tier B was initially miscounted, tuned raw-coordinate plain HMC had
incorrectly been optional, and the transformed endpoint scalar did not yet
state the mandatory NeuTra chart log-Jacobian explicitly.

Claude Opus max-effort bounded read-only review round 1 returned
`VERDICT: REVISE`. It identified five program-contract defects: missing
performance decision semantics, overstated Tier A evidence uniformity,
ambiguous Tier B scope-to-budget mapping, possible contamination of the pure
HNN claim by a mixture kernel, and an unconstrained posthoc sampler-gate
exception. The same file was patched visibly to resolve all five.

Claude round 2 reviewed the repaired master file only and returned
`VERDICT: AGREE`, with no remaining material scientific, mathematical,
feasibility, budget, evidence, or phase-handoff defect. The exact review record
is `docs/plans/bayesfilter-hnn-surrogate-hmc-plan-review-record-2026-07-17.md`.

## Completion Semantics

`TIER_A_CELL_COMPLETE` means every Tier A cell is either
`HNN_VALIDITY_CONFIRMED` or has
a precise blocker/failure with the earliest re-entry rung and no unspent
predeclared repair. `PROGRAM_COMPLETE` additionally classifies every Tier B
configuration as confirmed, failed, or requalification-blocked and completes
P8. Neither state implies superiority, calibration, exactness of approximate
filters, production readiness, or universal HNN validity.
