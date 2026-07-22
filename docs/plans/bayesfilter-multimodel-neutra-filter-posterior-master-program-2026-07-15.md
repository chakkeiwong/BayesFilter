# BayesFilter Multi-Model NeuTra Filter-Posterior Master Program

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `COMPLETE_WITH_PRECISE_BLOCKERS`

Terminal result:
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-terminal-result-2026-07-16.md`.

Reset memo:
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-reset-memo-2026-07-16.md`.

P7 attempt 01 found and provisionally classified `PP-UKF` and `PP-SGQF` as
`EVIDENCE_BLOCKED_TUNING_ADMISSION`. The runbook repair rule reopened their R4
tuning rung under
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p4-r4-tuning-admission-repair-subplan-2026-07-16.md`.
Both repairs passed fresh disjoint tuning verification and fresh downstream
confirmation. P7 attempt 02 reverified three narrow confirmations and eight
precise blockers; the terminal result and reset memo are final for this runbook.

Supervisor and executor: Codex in the active BayesFilter session.

Advisory reviewer: Claude may perform one bounded, read-only material-plan
review and one terminal-result review. Reviewer availability or procedural
disagreement is not execution authority and is not a continuation veto.

## Program Objective

Determine, independently for each declared model/filter cell, whether a
target-specific batched GPU/XLA NeuTra transport followed by the shared
sequential HMC controller can sample the exact posterior induced by that
cell's frozen filter likelihood. Build missing graph-native posterior routes
where necessary, preserve the structural constraints of Chapter 18b, and
transfer the admitted LGSSM training/HMC procedure without treating LGSSM
settings as automatically valid defaults.

This program deliberately separates two questions:

1. Does NeuTra agree with tuned plain HMC on the *same filter-defined
   posterior*?
2. Is the filter-defined likelihood an adequate approximation to the intended
   model likelihood?

A pass for question 1 does not answer question 2. Dense, exact, or focused
filter references answer question 2 only in their declared scope.

## First-Read Context

- The completed LGSSM foundation is recorded in
  `docs/plans/bayesfilter-neutra-hmc-core-consolidation-and-robustness-reset-memo-2026-07-15.md`.
- The shared HMC policy is `bayesfilter_neutra_sequential_hmc_v1` in
  `bayesfilter/inference/neutra_hmc.py`.
- Modern R-hat is the maximum of rank-normalized split and folded
  rank-normalized split R-hat.
- Warm-up and retained draws are separately archived and may each grow to
  10,000 draws per chain.
- Every learned NeuTra transport is target-specific. Cross-cell transport,
  tuning, architecture, optimizer, and affine-preconditioner reuse is a
  warm-start hypothesis, never evidence.

## Research Intent Ledger

| Field | Program intent |
| --- | --- |
| Main question | For which declared model/filter posteriors does target-specific batched GPU/XLA NeuTra plus shared sequential HMC pass same-target convergence, health, and plain-HMC agreement gates? |
| Candidate | A separately trained frozen dense-IAF NeuTra transport for each admitted cell. |
| Mechanism under test | Whether a learned transport removes target geometry sufficiently for a fixed HMC kernel while preserving the exact frozen filter posterior through the change of variables. |
| Expected failure modes | Wrong target binding; filter value/score mismatch; cross-target artifact reuse; unbatched or host-bound execution; weak architecture/optimizer; invalid affine chart; HMC tuning failure; nonconvergence; comparator disagreement; structural-noise leakage. |
| Promotion criterion | The individual cell reaches `NEUTRA_CONFIRMED` under its frozen evidence contract. There is no program-wide all-or-nothing promotion criterion. |
| Promotion vetoes | Target-signature mismatch; invalid value/score; filter-route veto; nonfinite/status failure; missing same-target comparator; R-hat/ESS failure at the retained cap; comparator disagreement; structural-identity violation; wrong GPU/XLA provenance. |
| Continuation vetoes | Invalid shared harness or target taxonomy; corrupted or unreproducible evidence; missing required comparator with no scoped repair; required hardware unavailable; phase budget exhausted; or a repair would change the scientific question, data, method, privacy boundary, hardware class, or total campaign budget. |
| Repair triggers | Local infrastructure, serialization, graph, value/score, filter, tuning, training, sampler, or reporting failure under an unchanged cell contract. |
| Explanatory diagnostics | Training loss, acceptance, step size, runtime, peak memory, filter-reference gaps, posterior summaries, truth distance on synthetic data, and per-seed differences unless a subplan explicitly promotes one. |
| Must not be concluded | Broad NeuTra robustness; filter exactness; calibration; superiority; universal recipes; production readiness; or correctness of an untested cell. |

Before stopping after a candidate failure, the supervisor must answer whether
the result invalidates the target, data, harness, comparator, math, or artifact,
or merely rejects the current training/sampler candidate. Only the former class
can stop unaffected cells.

## Frozen Cell Matrix

P0 issues the actual target signatures; these IDs freeze program scope but do
not attest that a route is admitted.

| Cell ID | Model target | Filter likelihood | Classification | Initial state |
| --- | --- | --- | --- | --- |
| `SVX-SGQF` | Exact transformed non-Gaussian SV | Fixed SGQF direct-likelihood route | BayesFilter approximation to exact transformed target | `UNINVENTORIED` |
| `SVX-ZC` | Exact transformed non-Gaussian SV | Zhao-Cui fixed TT/SIRT route | Must be paper/source anchored | `UNINVENTORIED` |
| `KSC-UKF` | KSC Gaussian-mixture SV | Principal-square-root UKF | Primary KSC lane; distinct from exact SV | `UNINVENTORIED` |
| `PP-SGQF` | Parameterized predator-prey | Fixed SGQF | Existing value/score components need posterior binding | `UNINVENTORIED` |
| `PP-UKF` | Parameterized predator-prey | Structural UKF | Existing value/score components need posterior binding | `UNINVENTORIED` |
| `PP-ZC` | Parameterized predator-prey | Zhao-Cui fixed TT/SIRT | Must be same-target and source anchored | `UNINVENTORIED` |
| `STR-UKF` | Chapter 18b quadratic structural model | Structural UKF | Intended structural route | `UNINVENTORIED` |
| `STR-ZC` | Chapter 18b quadratic structural model | Zhao-Cui fixed route | `extension_or_invention`, never source-faithful reproduction | `UNINVENTORIED` |
| `SIR-SGQF` | Parameterized spatial SIR | Fixed SGQF | Full observed-data posterior route currently missing | `UNINVENTORIED` |
| `SIR-UKF` | Parameterized spatial SIR | Structural UKF | Scout route currently; posterior admission missing | `UNINVENTORIED` |
| `SIR-ZC` | Parameterized spatial SIR | Zhao-Cui fixed TTSIRT | Current evidence is local/complete-data scoped | `UNINVENTORIED` |

Optional KSC SGQF or Zhao-Cui controls require new cell IDs and P0 target
records. They must never be folded into `KSC-UKF` or used to substitute for it.

## Structural Target Boundary

The Chapter 18b model is

```text
m_t = rho m_(t-1) + sigma epsilon_t
k_t = phi k_(t-1) + gamma m_t^2
y_t = m_t + k_t + e_t.
```

For every propagated structural point the mandatory identity is

```text
k_t - phi k_(t-1) - gamma m_t^2 = 0.
```

`STR-UKF` must not inject artificial process noise into `k_t`. A deliberately
naive full-state UKF that does inject such noise is required as a negative
control, gets a different diagnostic signature, and is permanently ineligible
for posterior admission. The current NumPy worked fixture is a reference only;
the admitted parameter posterior must be graph-native TensorFlow.

## Cell State Machine

```text
UNINVENTORIED
  -> TARGET_FROZEN
  -> VALUE_SCORE_ADMITTED
  -> POSTERIOR_IDENTITY_ADMITTED
  -> COMPARATOR_ADMITTED
  -> TRAINING_SCREENED
  -> TRAINING_ADMITTED
  -> NEUTRA_CONFIRMED
```

Side exits are `TARGET_BLOCKED`, `IMPLEMENTATION_BLOCKED`,
`FILTER_CANDIDATE_REJECTED`, `COMPARATOR_BLOCKED`, `RECIPE_REJECTED`,
`SAMPLER_BLOCKED`, `CELL_CANDIDATE_REJECTED`, and `EVIDENCE_BLOCKED`. A side
exit is cell-local unless it proves the shared harness or target taxonomy
invalid. `RECIPE_REJECTED` rejects only the named topology/optimizer/training
recipe. `CELL_CANDIDATE_REJECTED` is legal only after the predeclared
target-specific candidate families and optimizer ladders have all executed and
been rejected under their frozen gates. A continuation veto, budget exhaustion,
or an untried arm produces a precise blocked state, not scientific candidate
rejection. Re-entry must cite a versioned repair record and preserve every
prior attempt.

## Whole-Program Evidence Contract

| Field | Binding contract |
| --- | --- |
| Scientific question | Does each target-specific NeuTra candidate sample its exact frozen filter posterior consistently with tuned plain HMC, and what separate evidence exists about the filter approximation? |
| Exact comparator | Tuned plain HMC using the identical target signature, unconstraining transform, prior, observations, dtype, filter settings, and value/score callable as NeuTra. |
| Filter comparator | Exact or dense reference where feasible; otherwise a focused value/score and invariant ladder whose limitations are explicit. It cannot replace same-target plain HMC. |
| Engineering admission | Batched TensorFlow value/status and value/score; XLA compilation; finite outputs; deterministic replay; status and shape validity; focused score check; no active-path NumPy, host callback, or Python sample-axis loop. |
| Posterior identity admission | Before plain HMC, a per-cell dossier states the constrained posterior decomposition, unconstraining map and complete log-absolute-Jacobian, prior/data/filter settings, and independently recomposes the total log density and total unconstrained score. Negative tests must reject wrong data, prior, filter, dtype, chart, Jacobian, and signature substitutions. Production and independent recomposition routes may share checked mathematical primitives but not the final target-assembly callable. |
| Training admission | Target-specific architecture/capacity and optimizer/hyperparameter screen; batched GPU/XLA; TensorFlow memory growth; fresh selected 5,000-step run; heldout and downstream nomination checks; frozen artifact bound to the target signature. |
| HMC confirmation | At least four chains; fresh fixed-kernel tuning; retained warm-up; recent-window warm-up modern R-hat `<=1.05`; cumulative retained sampling up to 10,000 per chain; final modern R-hat `<=1.01`, minimum bulk ESS `>=1000`, minimum tail ESS `>=400`; all health/status vetoes clear. |
| Comparator agreement | P0 freezes target-specific primary estimands, simultaneous uncertainty method, and practical equivalence regions before serious results. Descriptive posterior gaps or overlapping marginal intervals are not sufficient. |
| Truth recovery | Synthetic truth distance is explanatory for one dataset unless a repeated-simulation calibration design is separately approved. It is not a cell veto by itself. |
| Artifacts | Target ledger, dependency hashes, exact commands, environment, GPU/CPU provenance, seeds, wall time, filter and comparator results, recipe ledger, training state, warm-up archive, retained archive, diagnostics, result, repair records, and phase manifest. |
| Nonclaims | A cell pass establishes only the declared target, fixture/data, filter, recipe, sampler, and diagnostics. It does not establish superiority, calibration, production readiness, or another cell. |

## Baseline Ladder Per Cell

| Arm | Role | Admission status |
| --- | --- | --- |
| Dense/exact/focused filter reference | Tests filter implementation or approximation in its stated scope | Reference or explanatory; never the same-target HMC comparator |
| Prior or affine-only transport | Naive NeuTra baseline and geometry diagnostic | Comparator only |
| Tuned plain HMC | Same filter-posterior reference | Mandatory comparator |
| Plain target-specific dense-IAF NeuTra | Proposed method | Candidate |
| Target-specific enhanced family | Second predeclared candidate after plain-recipe failure | Candidate only when its family, optimizer ladder, selection rule, and reserved budget are frozen before any recipe outputs |

Passing a hard screen means a candidate remains viable. No stochastic ranking
among viable recipes or filters is allowed without predeclared uncertainty
evidence. Every result lists candidate families and optimizer ladders tried,
rejected, selected, and still untried. A `RECIPE_REJECTED` result cannot be
reported as NeuTra rejection for the cell.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Shared sequential HMC | Completed LGSSM program | Already enforces retained warm-up and modern diagnostics | Cross-model adapter violates shared interface | P1 synthetic adapter and exact target-signature guards | Reviewed foundation, revalidated in P1 |
| Two-arm candidate ladder | Existing dense-IAF baseline plus one P0-frozen target-specific enhanced family | Tests a bounded architecture repair rather than equating first-recipe failure with cell failure | Both arms may still be too narrow to characterize all NeuTra | Separate family screens and downstream confirmations | Frozen candidate scope, not universal NeuTra scope |
| 5,000 selected training steps | Successful LGSSM protocol and owner request | Serious common budget rung | Too few or wasteful for a new target | Learning/heldout curves plus downstream screen | Required common rung; not proof of adequacy alone |
| Four chains and HMC thresholds | Shared BayesFilter convergence policy | Consistent modern diagnostic floor | Short-chain diagnostics can still miss pathologies | Retained growth to cap plus health/comparator gates | Required minimum |
| Plain HMC as comparator | Same-target sampler principle | Separates transport from target/filter differences | Both samplers could share a wrong target | Independent target/filter admission before HMC | Mandatory but not sufficient |
| Existing filter settings | Prior repository work | Useful starting points | Inherited settings fail on posterior parameter region | P0 provenance table and P2-P6 target-specific ladders | Warm-start hypotheses |
| Exact SV before KSC evidence reuse | Target mathematics | Prevents Gaussian-mixture evidence substitution | Accidental target blending | Signature negative tests | Required |
| SIR last | Current route inventory | Largest observed-data target gaps | Work may remain blocked after other families pass | P6 admission rung before training | Scheduling choice |
| GPU/XLA training with memory growth | Repository policy | Required learned-workload route | CPU/non-XLA evidence mislabeled serious | Trusted device manifest and fail-closed runtime check | Required |
| Multicore CPU external sample generation | Repository policy | Efficient independent data generation | Non-reproducible worker seeds or Python bottleneck | P1 seed partition and batch-throughput check | Default generation route |
| No NumPy or sample-axis Python loops | Repository policy | Preserves TF/XLA batched execution | Hidden host transfer or retracing | AST/source guard plus graph profiler | Required active path |

P0 must extend this table for every prior, transform, observation set, filter
setting, target region, architecture grid, optimizer grid, seed policy, and
equivalence margin. Weakly justified choices remain hypotheses and cannot be
silently promoted.

## Phase Map

| Phase | Purpose | Dedicated subplan | Exit |
| --- | --- | --- | --- |
| P0 | Freeze target/route identities and assumptions | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p0-target-route-freeze-subplan-2026-07-15.md` | Machine-readable ledger and commands/budgets frozen |
| P1 | Build the shared multi-model campaign harness | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p1-shared-harness-subplan-2026-07-15.md` | Generic adapter, manifests, guards, and smokes admitted |
| P2 | Exact transformed SV: SGQF and Zhao-Cui | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p2-exact-sv-subplan-2026-07-15.md` | Both cells confirmed or honestly cell-local blocked/rejected |
| P3 | KSC SV principal-square-root UKF | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p3-ksc-ukf-subplan-2026-07-15.md` | KSC cell confirmed or honestly classified |
| P4 | Predator-prey: SGQF, UKF, and Zhao-Cui | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p4-predator-prey-subplan-2026-07-15.md` | Three cells independently classified |
| P5 | Chapter 18b structural model | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p5-structural-model-subplan-2026-07-15.md` | Structural UKF and Zhao-Cui extension independently classified |
| P6 | Parameterized SIR: close target gaps before training | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p6-parameterized-sir-subplan-2026-07-15.md` | Three cells independently classified; no scout evidence promoted |
| P7 | Cross-cell synthesis, terminal audit, and reset | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p7-synthesis-subplan-2026-07-15.md` | Cell-complete report and reset memo |

Execution is governed by
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-execution-runbook-2026-07-15.md`.

## Phase-End Continuation Contract

Every phase, including a partial phase, must:

1. Run the checks named in its subplan.
2. Write a versioned phase result or blocker record and serious-run manifest.
3. Update the cell-state ledger without overwriting prior evidence.
4. Classify every failure as harness/infrastructure, target identity,
   value/score implementation, posterior recomposition, filter approximation,
   tuning, training recipe, sampler, evidence/reporting, filter-candidate
   rejection, recipe rejection, or cell-candidate rejection.
5. Repair a localized contract-preserving failure in a fresh attempt directory,
   run the focused regression, then re-enter at the earliest invalid rung.
6. Refresh the next subplan from actual evidence, including budgets and open
   defaults.
7. Audit next-phase consistency, feasibility, artifact coverage, and boundary
   safety; continue if no true continuation veto fired.

The default repair ceiling is three repair attempts for one materially identical
failure within a phase. After that ceiling, mark the affected recipe or cell
blocked and continue independent work. Do not use the repair ceiling to skip an
untried predeclared candidate family or call the whole cell rejected. Raising a
phase compute ceiling or changing the scientific contract requires a refreshed
plan and user direction.

## Program Compute Budget

P0 must replace estimates with command-level ceilings before launch. Initial
planning ceilings are: P0 8 CPU-hours; P1 16 CPU-hours plus 2 trusted GPU-hours;
P2 16 CPU-hours plus 80 trusted GPU-hours; P3 8 CPU-hours plus 40 trusted
GPU-hours; P4 24 CPU-hours plus 120 trusted GPU-hours; P5 24 CPU-hours plus 80
trusted GPU-hours; P6 32 CPU-hours plus 120 trusted GPU-hours; and P7 8 CPU-hours.
The whole-program ceiling is therefore 136 CPU wall-hours plus 442 trusted GPU
wall-hours. These are aggregate wall-time ceilings on the local hardware, not
commitments to consume the full amount.

Each cell reserves two 15-GPU-hour candidate-family arms: (A) plain dense IAF
and (B) one target-specific enhanced family frozen in P0 before recipe outputs.
Each arm permits one bounded optimizer/capacity screen, one selected fresh
5,000-step training, one fresh NeuTra confirmation, and arm-local retries charged
to that arm. A separate 6-GPU-hour bucket funds the same-target plain-HMC tuning,
confirmation, and comparator-local retries once per cell. A separate 4-GPU-hour
cell-admission/infrastructure bucket funds trusted R0/R1/R1B value-score,
posterior-recomposition, identity, batch, XLA, and device canaries plus
cell-specific adapter serialization/artifact emission and their local repairs.
Common harness, common schema, and shared serialization/reporting defects reopen
P1 and are charged only to P1's shared budget; they never consume or classify a
cell-local admission bucket. The exact cell ceiling is
therefore 40 GPU-hours. Arm B is not consumed when Arm A confirms. At most three
localized repair attempts are allowed per materially identical failure and all
attempts are charged to their owning bucket. Exhausting the admission bucket
before `POSTERIOR_IDENTITY_ADMITTED` yields `TARGET_BLOCKED` or
`IMPLEMENTATION_BLOCKED`; exhausting the comparator bucket yields
`COMPARATOR_BLOCKED`; exhausting a family arm yields `RECIPE_REJECTED` or
`SAMPLER_BLOCKED` only when its scientific gate was actually answered, otherwise
a budget blocker. No mandatory-bucket exhaustion permits
`CELL_CANDIDATE_REJECTED`.

P1's 2-GPU-hour shared bucket covers the trusted common canary and common-harness
repairs. A common defect discovered in P2-P6 reopens P1, pauses affected
downstream cells without changing their state, and consumes only the remaining
P1 budget. If that shared bucket is exhausted before repair, the program stops
with the shared-harness budget continuation veto rather than cell blockers.

## Skeptical Plan Audit

Decision: `PASS_FOR_DESIGN; EXECUTION_REQUIRES_P0_FREEZE`.

| Required challenge | Finding and control |
| --- | --- |
| Wrong baseline | Plain HMC is required on the identical filter posterior. Dense/exact filters answer a separate approximation question. |
| Proxy promoted | Loss, acceptance, smoke chains, truth distance, and filter-reference gaps are explicitly non-promoting unless a cell contract assigns a narrower veto. |
| Missing stop conditions | Cell promotion vetoes, program continuation vetoes, repair triggers, attempt ceilings, and compute ceilings are separate. |
| Unfair comparison | Target signatures bind prior, data, transforms, likelihood, filter settings, dtype, and callable dependencies for both samplers. |
| Hidden assumptions | P0 must audit every material inherited default; target-specific training search is mandatory. |
| Stale context | The LGSSM result is a reusable controller/procedure only. It is not cross-model scientific evidence. Existing SIR scout/local-density work remains inadmissible until repaired. |
| Environment mismatch | Serious training is trusted GPU/XLA with memory growth. CPU is explicit for reference and external generation only. |
| Artifact insufficiency | Every state transition names a required artifact; fresh attempt roots and manifests prevent overwrite and target substitution. |
| Commands do not answer question | P0 freezes per-cell commands only after value/score and comparator routes exist; a successful generic smoke cannot move a cell beyond its own rung. |
| Candidate failure stops research | Cell-local exits preserve independent progress and invoke planned repair phases unless a real continuation veto fires. |
| Filter accuracy conflated with sampler validity | Separate engineering, filter/numerical, and sampler/scientific ledgers are mandatory. |
| Both samplers agree on a wrong posterior | `POSTERIOR_IDENTITY_ADMITTED` requires independent constrained/unconstrained recomposition, full Jacobian/prior/data/filter binding, total-score checks, and substitution-negative tests before plain HMC. |
| One failed recipe called NeuTra failure | Recipe and cell rejection are separate; the budget reserves two family arms and cell rejection requires both to execute and fail. Budget/contract boundaries create blockers, not rejection. |
| Structural target corrupted | Pointwise deterministic identity and no-artificial-noise veto precede likelihood or NeuTra claims; naive full-state UKF is negative control only. |
| Source-faithfulness overclaimed | Zhao-Cui cells require paper and author-code anchors; Chapter 18b application is labeled invention. |
| Stochastic ranking unsupported | Cell passes do not rank filters or recipes. Simultaneous uncertainty/equivalence rules are frozen before serious outputs. |

## Pre-Mortem

The campaign could pass while misleading us if both samplers bind the wrong
posterior, if exact SV and KSC evidence are mixed, if inherited training settings
are accepted after a low loss, if a short HMC run nominates itself, or if the
structural route injects noise into a deterministic coordinate. Target-signature
negative tests, independent filter admission, target-specific search, final
modern diagnostics, and the structural veto address these risks.

It could fail for engineering rather than science through XLA compilation,
memory, multiprocessing, serialization, graph shape, or artifact errors. Those
are repair triggers. A scientifically valid target whose first transport fails
remains a candidate-rejection result, not evidence against NeuTra across other
cells or against the model family.

## Completion Semantics

The program is `CELL_COMPLETE` when every mandatory cell is either
`NEUTRA_CONFIRMED`, `CELL_CANDIDATE_REJECTED`,
`FILTER_CANDIDATE_REJECTED`, or has a precise blocker with consumed budget and
next discriminating action, and P7 passes its integrity audit. A cell-level
candidate rejection must include a tried/selected/rejected/untried family ledger
and prove that no frozen candidate arm remains untried. This rejects only the
declared two-arm ladder, not all conceivable NeuTra architectures.
`ALL_CELLS_CONFIRMED` is allowed only if all eleven mandatory cells reach
`NEUTRA_CONFIRMED`. Neither phrase implies production readiness, superiority,
calibration, or exactness of approximate filter likelihoods.
