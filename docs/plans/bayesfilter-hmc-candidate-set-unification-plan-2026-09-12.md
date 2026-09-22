# HMC candidate-set unification and guidebook rewrite

## Current disposition: 2026-09-15

The supported public ordinary, fixed-transport, and conditional position-field
routes now use the shared candidate-set controller, including proposals,
repairs, further evidence, retention, and restart. The active NeuTra consumer
uses explicit members and the separate posterior controller. The tuning guide
and book were rewritten to describe that implementation. R-hat is reporting-only
during tuning. The [whole-procedure repair plan](bayesfilter-hmc-whole-procedure-repair-plan-2026-09-14.md)
and [execution audit](bayesfilter-hmc-whole-procedure-repair-execution-audit-2026-09-14.md)
are the current completion record and list the remaining target-specific limits.

The September 12/R5 text below is preserved as historical planning context.
Its statements that the numerical bridge and public/consumer migration are
deferred no longer describe current library implementation. Target-scale
qualification and a real MacroFinance campaign remain separate work.

## Historical R5 plan

Date: 2026-09-12.
Status: revision R5 with the 2026-09-14 R-hat role correction.
See `bayesfilter-hmc-tuning-rhat-role-repair-plan-2026-09-14.md`. R5 preserves the
MacroFinance candidate-specific epsilon-repair correction and adds durable
resume, budget conservation, replay-integrity, and guide/reference alignment
requirements. The pure controller, checked artifact boundary, and typed
ordinary/fixed-transport bridge are implemented; the numerical adapter
qualification, active-consumer migration, and public-default migration remain
deliberately deferred until their gates pass.
Inspected BayesFilter commit: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`.
Owner request: plan one sensible tuning procedure that explores a broad set of
leapfrog counts, retains every viable candidate, and carries those candidates
through further tuning and verification; include rewriting the guidebook.

The R2 audit and the R3 clarification are planning evidence; the implementation
and guidebook migration now proceed in bounded phases below. The plan proposes
the successor to the September 1 and September 5 repairs. Their code remains
the current implementation until each migrated surface is qualified; their
historical results and documents remain preserved.

Audit instructions: [R2 Claude handoff](bayesfilter-hmc-candidate-set-unification-claude-audit-handoff-r2-2026-09-12.md).
The [initial Claude audit](bayesfilter-hmc-candidate-set-unification-claude-audit-2026-09-12.md)
and the [thorough R1 audit](bayesfilter-hmc-candidate-set-unification-claude-thorough-audit-2026-09-12.md)
are preserved unchanged. The R1 thorough audit returned `AGREE`, but it also
described plan sections and budget policies that do not exist in this file;
those descriptions are not implementation or planning evidence. Section 12
records the R2 repair and this source-fidelity limitation.

## 1. Intended outcome and scope

BayesFilter should expose one recommended tuning entry point, one candidate
lifecycle, and one set-valued result. Ordinary coordinates and frozen nonlinear
transports require different preparation, but the subsequent search, measured
refinement, retention, verification, budgeting, and reporting must use the same
controller. A compatibility name must not own a second selection procedure.

The proposed entry point is the existing `tune_hmc_kernel`, extended with typed
preparation for exact ordinary and exact transformed targets. Keep
`tune_fixed_transport_hmc_kernel` as an explicitly classified compatibility
wrapper into that entry point. It must not independently select, stop, repair,
or issue a different kind of result. This is an intentional public-interface
migration, not an assertion that the two current functions are interchangeable.

Each controller invocation owns exactly one immutable frozen `scope_id` and
returns exactly one `HMCTuningCandidateSetResult` for that scope. A collection
of such results is a reporting view, not a scheduler or another tuner. This
choice applies to dispatch, results, resumes, budgets, and replay.

Retain `select_fixed_transport_candidate_set` only as a deprecated diagnostic
helper for historical callers, with `interface_kind="diagnostic_helper"`,
`artifact_authority=False`, explicit diagnostic result semantics, and
`tune_hmc_kernel` as its replacement. Its package re-export remains for that
compatibility role. Its callback-driven rungs never supply evidence or handoffs
to the active controller. Remove it from the recommended workflow, classify it
explicitly in registry/discovery checks, and migrate all active in-repo callers.
Cross-transport nomination uses the read-only result collection in section 4.5;
it does not call this diagnostic helper or launch fresh validation callbacks.

The typed deterministic proposal-field branch must use the shared lifecycle
where it is supported, while retaining its mechanics-only status and exact
endpoint-potential requirements. It must never become an exact-score route by
dispatch convenience. Migration is incomplete if a package-reachable active
branch still silently performs a powers-of-two/first-pass search.

Scope includes the active tuning dependency paths, result/replay interfaces,
registry, repository-owned consumers, examples, and the tuning-related content
of the guidebook built from `docs/main.tex`. It does not require rewriting
unrelated filtering chapters or expanding this into NUTS, learned-transport
training, or a new posterior study.

## 2. Checked starting point

The following are source observations, not numerical comparisons. Line numbers
refer to the inspected commit; use the named function if they move.

| Evidence | Exact source to inspect | Observation and consequence |
| --- | --- | --- |
| E1 | `docs/reference/hmc-tuning-interface.md:128–170` | Ordinary tuning measures a primary/refinement grid but starts at most two fresh verifications per attempt and stops on a pass. Preserving candidate records does not ensure continued candidate evaluation. |
| E2 | `bayesfilter/inference/hmc_kernel_tuning.py::_select_joint_l_epsilon_candidate`, lines 22211–22291 | Both selection branches prioritize distance from target acceptance among eligible candidates. That order is not an efficiency ranking. |
| E3 | `bayesfilter/inference/hmc_kernel_tuning.py::_run_phase7_direct_candidate_queue`, lines 25148–25468 | The queue records unstarted candidates and stops at `first_admission`; its start quota can leave eligible candidates unverified. |
| E4 | `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py:79–106, 842–1094, 1425–1592, 1992–2075` | Measured joint-grid tuning owns a separate selector and verifier. Out-of-band acceptance is not a validity veto under the measured policy. Audit the actual selection and held-out logic, not only its policy name. |
| E5 | `bayesfilter/inference/fixed_transport_candidate_selection.py:121–219`; `docs/chapters/ch21b_hmc_tuning_interfaces.tex:446–507` | The helper retains all validated candidates, but callers first tune each frozen transport. It is not already an all-pairs tuner or an authoritative replay builder. |
| E6 | `bayesfilter/inference/tuning_contract.py:939–1081`; `bayesfilter/inference/hmc_tuning_dispatch.py:29–100` | The registry describes two active tuner names and different preparation, epsilon, and verification rules. Typed force dispatch is another conditional mechanics branch. |
| E7 | `bayesfilter/hmc_ordinary_selection_policy.py:1–105` | The current ordinary grid, cap of 25, and single midpoint-refinement barrier are centralized but specific to the ordinary policy. |
| E8 | `docs/chapters/ch21_hmc_for_state_space.tex:55–143`; `docs/chapters/ch22_mass_matrices.tex:36–101` | The earlier HMC chapter still selects a “best pair” and describes a different refinement sequence. The mass chapter presents budget formulas whose coefficients still need provenance. Updating only chapter 21b would leave conflicting guidance. |
| E9 | `docs/chapters/ch26b_neutra_transport_hmc.tex:316–367` | NeuTra describes another selection/validation sequence. It must distinguish transport training restarts from multiple HMC settings per frozen transport. |
| E10 | `tests/test_hmc_kernel_tuning_outer_loop.py:3760–3900, 4160–4235`; `tests/test_fixed_transport_candidate_selection.py:1–80` | Tests currently protect both first-admission/quota behavior and all-survivor retention in different interfaces. Migration requires changing the intended assertions, not preserving both as active defaults. |
| E11 | `/home/ubuntu/python/MacroFinance/docs/plans/bayesfilter_candidate_specific_epsilon_repair_handoff_memo_2026_09_12.md` and its listed Phase 14 JSON/event artifacts | The current downstream run can switch from one `L` after inconclusive/directional verification, compute a repair for another candidate, and terminate without executing that repair. This is a control-flow and handoff-lineage defect, not evidence that either candidate is statistically invalid. |

The motivating M4 file is outside this checkout, at
`/home/ubuntu/python/MacroFinance/results/hmc/daily_asset_midas_phase14_end_to_end_master_20260911_attempt03/tuning_attempt01/tuning_summary.json`.
Its run manifest records BayesFilter commit
`d2124d425b0ea0ae0e3e5f4246bd6b03ff8a2170`; do not pretend it ran the current
HEAD. At JSON path
`bayesfilter_payload.tune_verify_repair_loop.attempts[0].verification_diagnostics.phase7_direct_candidate_queue`,
the recorded counts are three eligible candidates, two starts, and one unstarted
candidate with reason `start_quota_exhausted`. Attempt 1 starts its two eligible
candidates; attempt 2 has no eligible handoff and starts no verification. The
top-level result is `budget_exhausted`, `passed=false`, with an empty
`hard_vetoes` list. These observations establish incomplete tuning and repair
requests, not valid posterior sampling, invalidity of every candidate, or a
ranking of the reported pairs. Preserve this as a motivating control-flow
example; use a small synthetic fixture for regression tests.

## 3. Research intent and evidence contract

| Item | Contract |
| --- | --- |
| Main question | Can one explicit procedure carry all viable measured candidates through bounded refinement and fresh verification without coordinate-specific changes in retention semantics or candidate-specific repair identity? |
| Mechanism under test | One shared controller and typed preparation adapters, with staged allocation to every survivor, same-L directional epsilon repair, set-valued replay, and explicit incomplete status. |
| Expected failure mode | A renamed facade hides old first-pass selectors, starves later candidates, collapses epsilon alternatives within an L, switches L before a required repair, or presents an unexecuted repair as an admitted kernel. |
| Comparator | Exact current code at the commit above, including the two-start queue, acceptance-distance order, separate fixed-transport selection, and existing set-validation helper. Compare control-flow obligations and the same frozen transition mechanics. The M4 artifact is not a performance baseline. |
| Primary engineering pass criterion | Deterministic adversarial tests demonstrate complete survivor retention, fair stage allocation, correct scope-bound replay for each verified member, unchanged target/transition contracts, bounded repairs/resume, and matching guide/examples/registry. |
| Promotion veto | Lost viable candidate; unmeasured or unverified member exported as verified; computed-but-unexecuted repair treated as a handoff; target/score mismatch; invalid scope reuse; missing required diagnostics; unsupported backend; divergent active procedures; unsupported scientific wording; failing relevant tests or documentation contradictions. |
| Continuation veto | Shared target/coordinate/geometry corruption, invalid execution environment, corrupted or cross-wired evidence, or exhausted total budget. Local kernel failure and insufficient mixing evidence are not shared invalidity. |
| Repair trigger | Candidate-local numerical failure with an identified repair, one-sided directional epsilon evidence, acceptance needing exploration, inadequate grid coverage, or localized infrastructure failure. Each trigger has a typed bounded next action; inconclusive evidence alone is not directional repair evidence. |
| Explanatory diagnostics | Acceptance, trajectory length, geometry spectrum, movement, runtime, compilation cost, and descriptive R-hat/ESS/MCSE summaries. Posterior gates apply only after tuning handoff. |
| Unsupported conclusions | No universal optimal L range, epsilon, mass, acceptance target, convergence guarantee, sampler superiority, or M4/NeuTra posterior readiness follows from controller tests or short smokes. |
| Preserved result | Versioned test and smoke outputs, exact command/environment manifest, candidate and repair tables, before/after reproduction trace, code diff, rewritten source/PDF, result note, and Claude audit findings. |

This is primarily an engineering migration. It does not need an expensive
sampler-ranking experiment to prove that the controller retains its members.
Default API activation nevertheless requires the numerical/compilation checks
below; fixing the orchestration must not conceal the ordinary route's known
NumPy/runtime-policy debt. Any later assertion that the procedure finds better
kernels requires a separate target-specific comparison with uncertainty.

## 4. The single procedure

### 4.1 Prepare and freeze a scope

Validate the target measure, matching exact score or explicitly typed proposal
field, data identity, dimensions, dtype/backend, coordinate transforms, and
telemetry. Ordinary HMC may perform qualified windowed mass adaptation. A frozen
transport supplies its exact Jacobian-corrected target and declared latent mass.
Do not silently add ordinary mass adaptation to identity-mass NeuTra.

Freeze the resulting geometry and an explicit dispersed chain bank before
comparing epsilon/L candidates. Before a serious run, record how that bank is
constructed, its location/scale or mixture components in named coordinates,
random seeds, and the target-specific coverage check. For example, a controlled
multimodal fixture checks representation of its known modes; a general target
records checked dispersion and the remaining unknown coverage. No universal
dispersion threshold or sufficiency claim is introduced here.

The repository issues `scope_id` from the frozen target/data/score contract,
transform and mass, dimension, start-bank design and realized bank, backend,
dtype, execution mode, and bound numerical/source dependencies. A fresh search
also has a versioned `search_id`, preserved on resume, so independent calls on
the same frozen scope cannot collide. Each candidate ID includes scope, search,
and stable creation ordinal. A candidate record is immutable and contains its
exact `L`, epsilon, target/adapter signature, mass signature, warmup protocol,
scope/search IDs, and a repository-issued `candidate_record_hash`. Equal numeric
(epsilon,L) values in different scopes or independent searches are distinct
records. A verification receipt additionally contains a distinct
`verification_attempt_id`, source candidate-record hash, stream ID, draw range,
and seed lineage. An execution attempt is not a new candidate and must not
change the candidate record hash.

An epsilon repair changes epsilon, so it necessarily creates a new immutable
child candidate ID and record hash. To express the requested "same candidate"
semantics without contradicting that identity rule, the child retains the
parent's `candidate_family_id` and `parent_candidate_id` and must preserve the
same scope, target, exact `L`, mass signature, coordinate system, start-bank
design, and warmup protocol. The family identifies the fixed-geometry,
fixed-`L` repair lineage rooted at one measured candidate; two independently
proposed epsilons at the same `L` start different families. The candidate ID
identifies one exact epsilon setting. A child may be deduplicated only against
an exact existing child with the same family, settings, warmup protocol, and
evaluation design, while preserving every requesting parent link.
An `L` replacement is a new candidate family. A retry of one unchanged
candidate increments only `verification_attempt_id`.

Reuse the same declared start-bank design and stage budgets within a scope, with
distinct candidate/stage/replication random streams. No result or resume record
can contain states or verification evidence from a second scope, and no repair
child can inherit a parent's verified status or failed holdout.

A geometry repair ends reuse of that scope's qualification. Return the proposed
repair and its reason; a subsequent call performs fresh preparation under a new
scope, linked as a successor for reporting. There is no hidden multi-scope mass
repair loop. A multi-transport or geometry-repair campaign allocates explicit
per-scope budgets within its total authorized budget and invokes the same tuner
for each scope. The fairness contract below is within a scope; there is no
global barrier over incompatible latent coordinates. The reporting collection
records allocation, spending, and unfinished scopes without moving their
reserves or scheduling them. Any later campaign-budget reallocation is explicit
and cannot exceed the existing total authorization.

### 4.2 Declare broad coverage and a finite proposal budget

One common configuration declares the primary L grid, per-L epsilon proposal
budget, permitted refinement L values or bounded expansion rule, stage ladder,
repair limit, and total compute limits. Serious calls require these choices and
their provenance explicitly. Named convenience fixtures remain nonclaiming.
There is no hidden universal cap of 25 and no separate ordinary versus NeuTra
search algorithm selected by a config type.

The existing six-value L grid is an inherited starting hypothesis. A scope may
reuse it with justification or provide broader coverage within its declared
budget. Positive distinct integers are necessary but do not alone establish
that a grid is broad enough for the target. Record coverage limitations and
predeclare how boundary evidence can propose additional L values. No expansion
may exceed the declared maximum or total budget without a revised plan.

For every primary L, independently qualify epsilon with the same bounded pilot
proposal routine. Dual averaging may propose settings; it does not verify them.
Explicit per-L epsilon proposals may seed that routine, including the existing
fixed-transport grid. Every resulting pair must be measured with a frozen
kernel before it can enter the survivor set. The same epsilon number may arise
at different L values, but its qualification never transfers across L.

Schedule the bounded per-L pilot tasks before the primary measurement cohort,
with the L list, work bounds, and stream identities fixed in advance. Pilot
tasks have a proposal-family ID because an immutable epsilon/L candidate may
not yet exist. Charge and checkpoint these tasks in the same search ledger.
After the pilot cohort finishes, freeze its complete primary pair list in L
and proposal order. An unfinished pilot remains incomplete coverage; a failed
pilot is not evidence that every epsilon at that L would fail. Any permitted
replacement pilot or exploratory proposal has its own bounded deferred work.

Preserve multiple viable epsilon values at one L. Do not collapse them because
one has acceptance closest to a target. Enumerate permitted refinement around
all survivors, deduplicate exact scope/settings matches, and give newly proposed
pairs their own measurement and validation. Epsilon proposals and adaptation
updates are hypotheses; do not assume fixed-L acceptance is monotone or treat
an unmeasured multiplied/divided epsilon as qualified.

The broad grid is the initial coverage cohort. A candidate-specific repair is a
different operation from proposing another primary or refinement `L`: it is
triggered only by a typed directional result for an already measured candidate
and remains in that candidate's fixed-`L` family. It must be measured and freshly
verified before it can produce a qualified handoff.

Each scope declares a positive finite epsilon domain, a directional repair
factor or proposal rule, and a maximum number of repairs per candidate family.
The rule must produce a finite positive epsilon different from its parent and
remain inside the declared domain. Repair-depth exhaustion is a typed terminal
condition (`repair_limit_exhausted`) for that family; it does not authorize a
different `L` repair and does not erase the parent's evidence. These are scope
configuration hypotheses with recorded provenance, not universal HMC constants.

### 4.3 Separate health, adequacy, and repair

| Observation | Role and next action |
| --- | --- |
| Exact target/score mismatch or shared coordinate/geometry corruption | Shared continuation veto; stop the affected scope and repair its definition. |
| Nonfinite candidate trajectory, declared divergence, invalid target status, or missing required telemetry | Veto that candidate's promotion. Distinguish a local step/geometry problem from shared implementation failure before scheduling a measured repair. |
| Finite acceptance outside a target band | Descriptive tuning/efficiency diagnostic. It is not a validity veto and does not by itself create a repair; a repair requires the typed directional evidence rule below. |
| High acceptance with absent movement | Movement/health failure under the declared screen; acceptance cannot rescue it. |
| R-hat/ESS/MCSE insufficient or unavailable during tuning | Explanatory evidence; retain the diagnostic without rejecting, repairing, or delaying a tuning handoff solely for this reason. Posterior assessment may request its own declared evidence extension after handoff. |
| Declared R-hat, bulk/tail ESS, MCSE, or target-specific posterior mixing screen fails | Blocks posterior admission only where the downstream posterior route declares that screen. It is not a tuning promotion veto or an epsilon repair trigger. |
| Runtime/device failure | Infrastructure classification; preserve the failed attempt and retry locally if the scope and remaining budget permit. |
| Total campaign budget exhausted | Stop new work, preserve completed work and pending candidates, and report incomplete coverage. No scientific rejection is inferred. |

Choose numeric posterior mixing thresholds and target-specific mode/observable
checks before posterior assessment. They do not gate tuning handoff. Modern
rank-normalized split/folded R-hat, bulk/tail ESS, and
MCSE must be computed in the coordinates and observables relevant to the claim,
including model parameters when latent diagnostics do not answer that question.
Record mean Metropolis probability separately from realized binary acceptance.

#### Directional epsilon repair and candidate replacement

The verification decision is not itself a queue selector. The controller applies
the following typed mapping:

| Verification decision | Permitted action | Identity and ordering requirement |
| --- | --- | --- |
| `repair_step_higher` with one-sided valid support and no opposing direction | Create a new child with higher epsilon. | Preserve `scope_id`, target, exact `L`, mass signature, coordinates, start-bank design, and warmup protocol. Give the child the triggering family's repair priority and a fresh verification stream before any later-`L` replacement work. |
| `repair_step_lower` with one-sided valid support and no opposing direction | Create a new child with lower epsilon. | Apply the same fixed-`L` and fresh-stream requirements as the higher repair. |
| `inconclusive_evidence` | Keep the candidate pending or schedule its predeclared evidence extension. | Do not change epsilon, switch `L`, or call another candidate an epsilon repair. Preserve the original candidate and evidence. |
| `inconclusive_conflict` | Keep the candidate evidence unresolved or apply the declared conflict policy. | Do not derive a directional epsilon repair or silently promote a different `L`. Any exploratory proposal is separately identified and cannot be described as repairing this candidate. |
| `inconclusive_trajectory`, resonance, stalled/oscillating, or explicit geometry/trajectory veto | Apply only the declared trajectory repair. | An `L` replacement is permitted only when the scope policy names that veto as sufficient; it creates a new candidate family and cannot inherit epsilon or verification evidence. |
| `passed` or a finite out-of-band observation without typed directional support | Retain the candidate and continue the declared stages. | Acceptance distance cannot select a winner or create a repair. |

Directional support must be computed from valid, candidate-local evidence under
the scope's declared aggregation rule. A single noisy acceptance estimate or a
confidence interval merely touching a band boundary is not enough unless the
policy explicitly declares it directional. The repair action records the parent
candidate, old epsilon, new epsilon, exact `L`, mass signature, source decision
and source verification hash. A repaired child is never the same candidate
record as its parent; it is the same fixed-`L` candidate family.

### 4.4 Advance every survivor through declared stages

Use the closed-cohort stage scheduler specified below: within one frozen scope,
give every member of the active cohort its declared allocation before advancing
to another cohort. A first pass does not end the scope's campaign. The old
two-start quota becomes an execution-batch limit at most; it must not erase the
rest of the queue or force another mass adaptation before they receive service.

The default fairness contract is comparable per-chain transition allocations
within a stage, with measured gradient counts and time recorded. Longer L costs
more and that cost consumes the total budget. Preflight checks must budget the
full declared cohort and reserve final validation work. If the minimum coverage
cannot fit, report an under-budgeted design before launch; do not silently run
the first few candidates. Runtime overruns leave an explicit partial result.

A directional epsilon repair creates a new candidate record with parent lineage
but preserves the triggering candidate's scope, exact `L`, mass signature,
target, coordinates, start-bank design, and warmup protocol. Only epsilon and
the repair action identity change. It does not overwrite the old evidence or
discard other healthy candidates. Local epsilon repair does not rerun mass
adaptation. A trajectory-specific `L` repair must be explicitly authorized by
the scope policy, creates a new candidate family, and cannot inherit epsilon or
verification evidence. A geometry change creates a new scope and requires fresh
qualification; old verification cannot be transferred to it.

Within a frozen validation sequence, longer rungs may continue from saved chain
states. Record cumulative draw ranges so they are not counted twice or called
independent replications. After adaptation or any kernel change, restart the
declared warmup/validation sequence and use fresh validation streams. Keep all
adaptation, discovery, comparison, and validation draws out of later posterior
estimation. Preserve them as separately labeled evidence.

Every survivor must face the declared fresh final verification, independent of
the data used to choose its settings. A failure can trigger a bounded repair
with a new reserved verification stream; the failed holdout remains recorded.
Do not repeatedly inspect a fixed-size confidence interval at arbitrary stops
and claim nominal coverage. Predeclare assessment rungs and use a justified
sequential uncertainty method if inferential optional stopping is intended.

When a directional repair is requested, the controller records a repair action
with `repair_execution_status` (`not_started`, `running`, or `executed`),
`repair_verification_status` (`pending`, `passed`, or `failed`), and terminal
`qualified_repair_status`. The terminal qualified status is exactly one of
`executed_and_verified` or `not_executed_with_reason`; the latter means that no
qualified repair handoff was issued and includes reasons such as
`repair_budget_exhausted`, `repair_not_scheduled`, `repair_infrastructure_failure`,
or `repair_verification_failed`. An executed but failed child therefore remains
visible as executed/failed while its qualified handoff status is
`not_executed_with_reason`. Computing a new epsilon never changes that status.

#### Candidate and work-item transitions

Candidate evidence state and work execution state are separate. The candidate
stores its last completed stage even when work is interrupted or budget-limited.

| Candidate evidence state | Transition rule |
| --- | --- |
| `proposed` | Admitted for measurement with a scope-bound ID and reserved work; unmeasured and ineligible for replay. Measurement passes to `screened`; a health failure goes to `promotion_failed`. |
| `screened` | Mechanics passed; eligible for declared validation. Beginning validation gives `validating`. A refinement proposal creates a separate child and does not replace a healthy parent. |
| `validating` | A permitted evidence extension advances to the next predeclared rung with the same candidate and frozen kernel. Only completion of that candidate's own fresh final-verification requirements gives `verified`; a terminal screen failure gives `promotion_failed`. |
| `verified` | Terminal candidate success; retain its evidence and handoff. It grants no success or evidence to a child. Shared scope invalidity disables replay even if this earlier evidence state was verified. |
| `promotion_failed` | Terminal for that candidate's declared validation sequence, with an explicit health or evidence-insufficiency reason. A permitted repair creates a new child; the parent's failed record is never erased or turned into a pass. A directional repair child must retain its parent family and fixed-`L` identity. |

`repair_requested`, `proposal_budget_pending`, `budget_incomplete`, and
`repair_budget_exhausted` are separate work/disposition fields; none overwrites
the evidence state. An unchanged kernel may extend its existing sequence only
at predeclared rungs. After exhausting final verification, it cannot restart
the same test repeatedly until it passes. A settings/warmup-protocol repair
creates a new candidate and fresh validation; a geometry change instead
requires a new scope. The scope identity binds the available start design; a
repair cannot import another scope's chain endpoints or failed-holdout
endpoints into the fresh start bank.

Each candidate work item contains `scope_id`, `search_id`, `candidate_id`,
`candidate_family_id`, `candidate_record_hash`, `cohort_id`, stable ordered
`work_item_id`, stage/rung and replication identifiers, predecessor IDs,
`repair_action_id` when applicable, `verification_attempt_id`, stream ID and
draw range, reserved-work reference, and checkpoint/result reference. Dispatch
changes `pending -> running`; recording a complete receipt changes `running ->
completed`, including an adverse scientific outcome. A verification receipt
must echo the candidate record hash, exact `L`, epsilon, mass signature, repair
action ID, and attempt ID that were dispatched.
Infrastructure failures change `running -> interrupted`, not candidate success
or numerical rejection. Retry adds a distinct attempt ID to the same logical
work item. If no complete checkpoint exists, repeat only its uncommitted chunk
from the saved start/RNG state and count the failed computation too. Such a
repeat is not an independent replication or new evidence.
Pilot work uses the same execution record, with `proposal_family_id` and a null
`candidate_id`; its completed proposals acquire candidate IDs at the boundary.
If an infrastructure failure cannot be retried under the declared policy,
return `paused_infrastructure` with the unresolved work preserved, or
`partial_budget` if unrelated required work is exhausted. If the unresolved
next work is a required directional repair, also set the typed stop reason
`repair_budget_exhausted` when any declared total, attempt, stage, or wall-time
budget prevents its execution, or
`repair_infrastructure_failure` (when retry is forbidden), naming the parent,
child proposal, old/new epsilon, exact `L`, and mass signature. Shared
invalidity aborts the cohort and disables scope replay immediately rather than
waiting for its remaining work. No such stop is disguised as a completed
verification or qualified repair.

#### Cohort closure, child insertion, and reserves

1. After the pilot prelude in 4.2, freeze the complete primary proposal list and
   its deterministic ordinal order before measurement. Measurement is the first
   candidate stage; configured validation and fresh-verification rungs follow
   in a finite stage order.
2. At a boundary, choose the lowest unfinished stage among admitted ready
   candidates. Close that cohort's membership and ordered work-item list before
   dispatch. Within the cohort, order by candidate creation ordinal and then
   replication ordinal. Acceptance, runtime, and completion order do not reorder
   work. Bounded parallel execution must reduce outcomes in that same order.
3. During a partially completed cohort, append repair/refinement requests to a
   deferred proposal list. Neither its membership nor existing reservations
   change. Process requests only after the active cohort closes, in parent
   work-item order and then the predeclared proposal order; allocate child IDs
   at that boundary. Deduplicate only identical scope, settings, warmup protocol,
   and evaluation design, recording every requesting parent. Never deduplicate
   by L alone or across scopes, or reuse a failed holdout as a fresh attempt.
4. After the active cohort's already-reserved work closes, directional epsilon
   repair children have priority over any not-yet-admitted work for a different
   `L` and over advancement of the triggering family's parent to a longer stage.
   If several repairs are requested at one boundary, order them by parent
   work-item order and then proposal order. Existing reservations for other
   candidates are honored and cannot be stolen; this is the fairness exception
   that makes the same-`L` repair deterministic without preempting running work.
   The repair-priority interval ends when the child receives its declared fresh
   verification or becomes budget/infrastructure pending. A later-`L` candidate
   is never described as the repair of the earlier candidate.
5. Admit children from the separate declared exploration/repair reserve, after
   reserving their measurement and required validation through final verification.
   They never inherit a parent's verification or consume another active
   candidate's reservation. Unspent reservations released by a terminal candidate
   return to the free pool at the boundary, with an explicit accounting entry;
   a child needs its own allocation. If a required repair cannot be executed
   within the declared total, attempt, stage, or wall-time budget,
   preserve it as `proposal_budget_pending`, set its qualified repair status to
   `not_executed_with_reason`, and record `repair_budget_exhausted` when the
   budget is the cause. Continue already reserved work.
6. New children enter measurement. Selecting the lowest ready stage lets them
   catch up after the previous cohort has finished, before unrelated incumbents
   advance further. An evidence extension of an unchanged candidate occupies its
   next configured rung, not a newly inserted lower-stage candidate. Finite total
   proposal, refinement, repair, and rung limits prevent endless insertion.
7. Exceeding an execution estimate stops work at a recorded bound; it does not
   borrow another member's final-verification reserve silently. Commit observed
   costs and conservative bounds for interrupted work whose exact cost is
   unavailable. Required work is derived from each candidate's frozen stage
   schedule; stages not yet materialized as work items still count as pending.
   A terminal candidate failure may skip its remaining stages with a recorded
   reason. A scope is `complete` only when every required admitted work item is
   terminal or validly skipped and no in-design proposal remains pending. A completed
   finite design may have an empty verified set. `partial_budget` records
   unfinished required work; `shared_invalidity` disables that scope's replay.
   `has_verified_candidates` is independent of campaign completion. A scope's
   public `final_status` may be `repair_budget_exhausted` when the next required
   action is an unfunded directional repair; its `completion_status` remains
   `partial_budget`, and the repair payload names the exact blocked candidate.

Persist the active cohort and its remaining ordered work-item IDs, deferred
requests, creation counter, closed-stage decisions, reserves/spending, and
completed evidence IDs together with the candidate checkpoints. Resume consumes
this recorded order rather than rebuilding it by sorting diagnostic outcomes.
Ordinary atomic checkpoint replacement and checksums suffice; this is not a
launch-authorization or output-reservation protocol. Completed raw-draw receipts
cannot be credited twice; cumulative diagnostic windows may reference already
committed receipts without adding draw or work credit. A partial receipt cannot
satisfy a final-verification rung.

The required mutation trace is: cohort `[A:r, B:r]` starts; A finishes and
requests a directional same-`L` repair child A1 and a refinement child A2 while B
is pending. Neither child starts nor uses B's reserve. Interrupt and resume at
that point; B still runs next because its work was already reserved. Once that
cohort closes, admit A1 before any not-yet-admitted different-`L` work, then
measure A1 with a fresh stream and verify it before advancing A beyond the
repair barrier. Admit A2 only under the declared refinement order. B's
recorded outcome remains intact. Repeat with insufficient child funding: the
requested A1 record names old/new epsilon and exact `L`, has
`not_executed_with_reason` plus `repair_budget_exhausted`, and the result reports
partial coverage rather than a completed repair.

### 4.5 Return a set; nominate separately

The proposed versioned result, `HMCTuningCandidateSetResult`, has exactly one
immutable `scope_id`, one `search_id`, and contains:

- all candidate records and their complete scope/settings/parent identities;
- mechanically screened survivors, pending repairs, rejected candidates, and
  candidates left incomplete by budget;
- `verified_candidates`, each with its own frozen settings, final states,
  diagnostic evidence, and replayable repository-issued handoff;
- immutable `candidate_record_hash` values and repair records containing the
  parent/child IDs, `candidate_family_id`, old/new epsilon, exact `L`, mass
  signature, source verification hash, execution/verification statuses, and
  qualified repair status;
- separate campaign coverage/completion and per-candidate evidence status;
- optional `nominee_id`, nomination criterion, and uncertainty status; and
- total work, remaining budget, resume state, source/environment identity, and
  explicit posterior/scientific authority fields.

Both public entry names return this same single-scope result. A resume accepts
one matching scope/search/result and its checkpoint. The prepared scope uses a
repository-issued typed object; callers cannot self-stamp a scope ID.
Replay accepts the result plus its explicit scope-bound member ID and verifies
the corresponding private geometry. It rejects a candidate, state, evidence
receipt, or transport supplied from any other scope, even when numeric settings
and target names match.

`viable_candidates` has one definition across adapters: non-vetoed members with
evidence state `screened`, `validating`, or `verified` in a non-invalid scope.
`verified_candidates` is the subset with its own completed final verification;
an unfinished budget does not erase either set. Shared invalidity disables both
eligible sets but retains every prior state/receipt in the historical candidate
records. A result can have verified members while the campaign remains incomplete.
Report that fact without
claiming the full search completed. Replay requires an explicit verified member
ID; neither list position nor an implicit acceptance winner selects a kernel.
An unexecuted, executed-but-failed, or budget-pending repair cannot appear in
`verified_candidates`. A computed epsilon is a proposal only. When a required
repair is not run, the result includes
`qualified_repair_status="not_executed_with_reason"` and the typed reason,
including `repair_budget_exhausted`; it never emits a repaired-kernel handoff.

Default behavior returns the set without a nominee. Optional efficiency
nomination uses predeclared observables, budgets, independent replications, and
an explicit objective such as minimum relevant ESS per measured target-gradient
evaluation or per wall time. These objectives are not interchangeable. A
descriptive nominee may be useful when uncertainty is inconclusive, but its
choice never removes other verified members or establishes superiority. A
statistical ranking requires an appropriate uncertainty and multiplicity
analysis for the declared candidate family.

For several frozen transports, `HMCTuningScopeCollection` is a read-only
collection of already-issued single-scope results, keyed by scope and
search IDs. It preserves all members and per-scope completion/authority; it can
summarize cost and nominate descriptively among already verified members of the
same physical target when
observables and comparison designs are comparable. It cannot run callbacks,
adapt, validate, repair, merge chain states, issue a new handoff, or transfer
authority. Its nominee is an explicit `(scope_id, candidate_id)` reference;
replay still uses the original result and scope. Even collection-wide
completion is derived from the declared expected scopes and their statuses,
not inferred from the first available verified result. Cross-scope statistical
ranking remains a separate declared comparison, not a collection feature.

Selecting a verified kernel is separate from posterior estimation. Existing
sequential NeuTra retained-sampling requirements remain applicable. This change
does not introduce adaptive switching or pooling across kernels inside a
retained chain. Such a mechanism would need its own sampling argument.

## 5. Default and numerical-assumption audit

Numbers below are inherited observations or proposed bounded engineering
choices, not calibrated scientific defaults.

| Choice | Provenance and status | Failure mode | Early check or decision |
| --- | --- | --- | --- |
| L = (3,5,9,13,18,25), current upper bound 25 | September 5 owner-directed ordinary policy; inherited coverage hypothesis for the new procedure | Missed useful integration scales; accidental universal cap | Explicit scope grid, boundary/coverage report, measured permitted extensions; remove route-specific hardcoding from the new controller. |
| Target acceptance 0.70 and band [0.65,0.75] | Existing repository settings; tuning heuristics | Noisy band crossing mistaken for invalidity; poor movement at high acceptance | Separate acceptance and movement, uncertainty, and final mixing evidence; no acceptance-distance elimination. |
| Four-chain bank | Existing ordinary/validation convention; inherited baseline | Common starts conceal poor exploration or modes | P0 records a target-specific construction in named coordinates, realized-bank provenance, and a cheap dispersion/mode-coverage check with explicit limits as specified in 4.1; no universal sufficiency claim. |
| R-hat 1.01; helper ESS 400/200 and MCSE/SD 0.10 | Existing interface/helper screens; not newly calibrated here | Insufficient evidence or inappropriate observable coverage | Scope-specific validation specification, recorded limits, and declared extension/repair behavior. Do not silently copy helper constants into a new universal policy. |
| Fixed-transport 4+16 transition convenience screens | Current config, short diagnostic budgets | Short chains mistaken for mixing or efficiency proof | Keep mechanics-only fixtures explicit; serious configurations require their own ladder. |
| Single survivor-midpoint refinement barrier | Existing ordinary engineering rule; an explicit scope option, not a settled common default | Sparse grid or resonance missed; repeated refinements explode cost | Scope declares its proposal bound and coverage checks; children enter closed cohorts under 4.4 rather than mutating the active queue. |
| Per-stage equal transition allocation | Proposed fairness convention | Expensive L values consume budget; finish order biases nomination | Preflight whole-cohort cost and final-verification reserve; record actual gradients/time and incomplete work. |
| Same-`L` directional epsilon repair priority | New owner-requested lifecycle rule | A later `L` can replace the triggering candidate before its required repair; strict priority can delay other work | Complete already-reserved cohort work, then run the repair-priority child first; bound repair count and preserve other candidates' reserves. |
| Candidate family versus candidate record identity | New identity clarification | Treating an epsilon-changing repair as the same immutable record permits hash/evidence confusion; treating every retry as a new candidate duplicates evidence | Keep immutable candidate ID/hash per exact `(L, epsilon, mass, scope, protocol)`; keep family/parent lineage for same-`L` repairs and attempt IDs for retries. |
| Directional repair factor and epsilon bounds | Existing route mechanics, not calibrated here | Repeated or oversized repairs can leave the declared search or budget | Require a finite per-candidate repair limit, positive bounded epsilon, recorded old/new values, and fresh verification for every child; treat factors as scope hypotheses. |
| Repair terminal and budget statuses | New result contract | Computed or unrun repair appears as a completed handoff, or generic budget exhaustion hides the blocked candidate | Require execution/verification status, `executed_and_verified` versus `not_executed_with_reason`, and typed `repair_budget_exhausted` payload with candidate identity. |
| Mass freezing and identity mass in declared latent coordinates | Existing target/geometry contracts | Comparing changed kernels; transferred stale epsilon bounds | Same-scope signatures and same-kernel replay/parity fixtures; requalify after geometry changes. |
| TensorFlow/TFP, GPU, XLA, memory growth | Applicable repository direction; per-adapter XLA readiness remains unproved | Nominal shared controller still calls NumPy or enables unqualified compilation by default | P2 builds stable signatures and an explicit qualification mode; P5 checks the complete call chain and qualifies each adapter before default activation. Preserve non-admission or a documented reviewed non-XLA exception when qualification fails. |
| Scientific tolerances and random seeds | Not selected for a new scientific campaign by this planning task | Arbitrary constants acquire scientific authority | Preserve existing named fixture values where justified; future target-specific protocols must state their own derivation or provenance before running. |

## 6. Implementation work packages and acceptance

| Phase | Work and principal files | Completion evidence |
| --- | --- | --- |
| P0: freeze the migration surface | Use `scripts/inventory_hmc_tuning_routes.py`, `scripts/audit_ordinary_hmc_migration_surface.py`, registry/exports, examples, benchmark callers, and replay consumers. Explicitly inventory `select_fixed_transport_candidate_set` and all its callers for the diagnostic-only disposition in section 1. Read the active dependency closure through ordinary/typed executors, state, selection, artifact construction, and replay. Save current guide sources/PDF identities and freeze fixture/start-bank specifications. | Every reachable route has a migration classification; helper discovery cannot omit a `select_*` export. NumPy/runtime debt, old failures, and source/fixture assumptions are explicit. No whole-repository cleanup. |
| P1: common policy and result | Implement the single-scope controller and candidate/work-item transitions, immutable candidate records/hashes, candidate-family lineage, typed directional epsilon repairs, repair-priority queue, child insertion, reserve accounting, persisted order, typed repair/budget statuses, and set result from 4.1/4.3/4.4/4.5. Keep the optional result collection read-only. Proposed modules are `bayesfilter/inference/hmc_candidate_set_tuning.py` and `hmc_candidate_set_artifacts.py`; names may change without semantic changes. | Deterministic traces prove same-`L` repair identity, direction mapping, priority, non-inheritance, terminal statuses, and accounting before numerical integration. No launch tokens or custom authorization system. |
| P2: typed numerical adapters | Extract/reuse ordinary preparation and frozen-transport target construction; migrate the touched NumPy path to TF/TFP and host bookkeeping to Python standard types. Route all supported config branches through P1, including the mechanics-only force branch. Implement stable `tf.function` signatures and an explicit, non-default XLA qualification mode. This phase changes no public execution default. | Same frozen target, scores, transforms, and transitions agree with eligible references. Inspect imports/calls through state, selection, artifact construction, and replay for NumPy contamination. Explicit reference/debug graph mode is non-admitting; no pfor and no claim of XLA qualification yet. |
| P3: replay and consumers | Update dispatch, exports, `tuning_contract.py`, `hmc_route_contract.py`, artifact/replay builders, wrappers, and in-repo consumers to the same single-scope result and member IDs. Register `select_fixed_transport_candidate_set` as a deprecated diagnostic helper, retain only its diagnostic re-export, reject its result at active replay boundaries, and replace its active callers with per-scope tuning plus read-only collection. Carry candidate hashes, repair lineage, attempt IDs, typed budget outcomes, and no-handoff rules through replay. | Direct shared-controller wiring and identical injected work traces are checked. Every verified member replays within its own scope; cross-scope/pending/stale/computed-only repairs fail. Old payloads remain in their declared legacy/mechanics role. |
| P4: guidebook and examples | Apply section 7, explicitly remove the legacy helper from recommended workflows, and show separate per-transport scope results and a non-executing collection. Explain same-`L` directional repair, inconclusive evidence, repair priority, computed-versus-verified status, and typed budget exhaustion. Regenerate registry tables and example/test contracts. | Consistent rendered/source book and examples, substantive baseline comparison, and recorded reader-review limitation. Described activation and repair status match the actual P5 evidence. |
| P5: qualify, integrate, and activate | Run focused tests, including the MacroFinance reproduction trace, same-kernel/known-target checks, per-adapter XLA compatibility/equivalence/peak host and device memory/compile cost/steady-state checks, downstream static audit, and book builds. Recheck the full active dependency closure. Only then activate qualified adapter defaults and align documentation; finish with result/reset and audit disposition. | Default activation follows recorded passing qualification for each supported adapter class. A failed class stays non-admitting unless an explicit reviewed non-XLA exception defines its limits; never label an unqualified adapter XLA-ready. A directional repair cannot be called complete without fresh execution and verification. A remaining divergent procedure or runtime-policy violation prevents declaring the migration complete. |

No automatic edits, lock updates, or M4 reruns in MacroFinance/dsge_hmc are part
of this BayesFilter implementation plan. Produce concrete migration instructions
and static compatibility findings for those repositories. Their owners choose
when to adopt the versioned API. Existing experiments cannot change procedure
mid-run because the library's default changed.

## 7. Guidebook rewrite specification

Rewrite the tuning explanation as scientific exposition, followed by a compact
API reference. The reader should understand why geometry, epsilon, integration
length, mixing, and computation cost are separate choices before seeing config
fields. Explain one common procedure once; use ordinary and transformed targets
as examples of its preparation step.

| Surface | Required rewrite |
| --- | --- |
| `docs/chapters/ch21_hmc_for_state_space.tex` | Replace the stale “best pair”/local-refinement algorithm with the set procedure and a worked candidate table. Preserve exact target/score requirements, same-`L` directional epsilon repair, mass-change requalification, and Metropolis mechanics. Make clear that an epsilon repair is a new child record in the same fixed-`L` family, not an unverified replacement. |
| `docs/chapters/ch21b_hmc_tuning_interfaces.tex` | Explain one scope/result per call, broad coverage, independent per-L epsilon proposals, measured pairs, closed-cohort stages, same-`L` directional repair priority, inconclusive evidence versus directional evidence, fresh verification, set replay, typed budget exhaustion, optional nomination, resume, and posterior separation. Put `select_fixed_transport_candidate_set` only in diagnostic migration notes, with the replacement workflow. |
| `docs/chapters/ch22_mass_matrices.tex` | Explain geometry preparation/freezing and when a genuine geometry repair invalidates old tuning. Audit budget formulas/coefficients; preserve justified mathematics while labeling heuristic allocations accurately. |
| `docs/chapters/ch25_diagnostics.tex` | Align acceptance, movement, divergence, modern R-hat/ESS/MCSE, observable coordinates, uncertainty, and candidate-versus-campaign failures. Explain why an early short-chain failure can require more evidence. |
| `docs/chapters/ch26b_neutra_transport_hmc.tex` | Preserve change-of-variables mathematics and sequential retained sampling. Show one tuner call/result per frozen transport scope, multiple retained kernels within each, and a read-only cross-transport collection. Distinguish per-scope budgets, chain states, completion, and replay; no second validation callback or latent-state pooling. |
| `docs/chapters/ch26c_hnn_surrogate_hmc.tex` | Repair tuning references only; preserve the exact endpoint-potential and proposal-mechanics assumptions and the branch's weaker authority. |
| `docs/reference/hmc-tuning-interface.md`, `docs/generated/hmc_tuning_route_table.{md,tex}` | A concise operational reference generated/aligned with the new registry, including compatibility mapping, candidate IDs, incomplete status, and replay semantics. Generated tables remain an inventory, not an alternative procedure menu. |
| `docs/examples/hmc_tuning_*.py`, `docs/examples/fixed_transport_candidate_selection.py` | Demonstrate the shared call, multiple retained pairs, explicit scope/member replay, interrupted-cohort resume, and pure cross-scope reporting. Rewrite the named selection example around the collection; historical diagnostic-helper examples, if kept, must be explicitly named/classified as diagnostics. Mark illustrative configuration values. |
| Other active book/API cross-references discovered in P0 | Correct only confirmed contradictory guidance, including claims about default XLA and publicly visible tuning summaries. Full local reports must expose the candidate set; externally redacted reports may preserve actual privacy boundaries. |

Required worked example: start with several L values and multiple measured
epsilon values, show more than one survivor, let the first candidate pass while
later candidates still receive their stages, then show a directional high-
acceptance result that creates a higher-epsilon child with the same L and mass.
Run that child before later-L replacement work, show an inconclusive result that
does not change L, and show an unexecuted repair marked with
`repair_budget_exhausted`. Return the verified set with parent/child hashes and
repair statuses. Use explicitly synthetic values or captured fixture output. Do
not disguise invented numbers as the M4 run, combine attempts across changed
geometries, or call a nominee the best sampler without uncertainty support.

Before making mathematical/literature claims, inspect the relevant method,
theory, and appendix sections and official implementation where available.
Check the existing Neal/Hoffman-Gelman/Betancourt citations and the sources for
modern diagnostics and any acceptance-optimality claims actually used. Keep
local tractable paper copies and exact source/equation anchors. Acceptance
targets derived for a specific asymptotic setting must not become universal
finite-target rules. This plan itself makes no uninspected optimality claim.

Protect the actual current chapter sources, bibliography, examples, and PDF
identity before rewriting. Track every removed equation, derivation, citation,
assumption, finding, or qualification and explain its disposition. Inspect the
rendered PDF for equation explanations, tables, captions, cross-references,
transitions, and stale policy language; successful compilation is insufficient.
Naturalness and final reader acceptance remain subject to human feedback;
provisional drafting and technical integration may continue before it arrives.

The separate `docs/fable-rewrite/monograph` tree exists and currently does not
include chapter 21b. Preserve it as a separate manuscript unless its active
status is established in P0. Do not silently copy the rewrite into protected
historical/staging manuscripts. Identify which PDF and source tree the delivery
actually updates; `docs/main.pdf` is not assumed to match today's source.

## 8. Verification, budget, and commands

### Focused engineering checks

Create meaningful controller/replay tests in the proposed
`tests/test_hmc_candidate_set_tuning.py` and
`tests/test_hmc_candidate_set_artifacts.py`. Required adversarial cases are:

1. The first candidate passes, a later candidate also passes, and both complete
   the declared stages and remain replayable.
2. More candidates survive than the old two-start quota; later members receive
   work before an earlier member advances to a longer stage.
3. Multiple epsilon values at one L survive; refinement includes every survivor
   and no pair is admitted without being measured.
4. Out-of-band finite acceptance alone does not reject a candidate; high
   acceptance with no movement does not pass its declared health screen.
5. High or unavailable R-hat/ESS/MCSE cannot reject, repair, or delay a tuning
   candidate; posterior assessment retains its separately declared gates.
6. Local failure/repair preserves other candidates and frozen geometry; shared
   invalidity stops the affected scope; changed geometry requires fresh tuning.
7. Budget exhaustion/resume preserves partial progress, RNG/seed lineage,
   endpoint states, draw ranges, work charges, and pending order without
   repeating completed chunks or granting verification to unrun members.
8. Every verified member replays its own settings; wrong ID/scope, stale source,
   changed transport/mass/backend, or pending/failed status is rejected.
9. All supported entry names use the same controller; compatibility wrappers,
   typed force bindings, diagnostics, and legacy payloads cannot gain authority
   by relabeling. Assert wiring to the same controller object and inject a
   deterministic outcome/cost schedule at that shared boundary. For equivalent
   prepared inputs require the same ordered work, stages, child insertion,
   stream domains/draw ranges, budget charges, and stop reasons through each
   compatible entry. Normalize only issued scope/search namespace prefixes and
   adapter identity/authority fields when the compared calls legitimately differ
   in those fields; never drop, sort, or relabel divergent work or stop reasons.
   For distinct adapter classes use equivalent prepared fixtures and the same
   injected outcomes while checking their distinct authority. A shared result
   schema or identically named function is not a sufficient oracle.
10. Discovery/validation draws never enter posterior estimates, and uncertainty
    summaries do not treat continuation chunks as independent replications.
11. Replay the partially completed `[A:r, B:r]` mutation trace in 4.4, with both
    funded and unfunded repair/refinement children and an interruption after A.
    Assert exact resumed order, stable creation IDs, unchanged B reserves,
    lowest-stage catch-up, and candidate-versus-campaign terminal states.
12. A failed final holdout is preserved. A repaired child has fresh stream IDs,
    scope-bound evidence IDs, and disjoint `(stream_id, draw_range)` identities;
    it cannot use the failed holdout's states as a fresh start, inherit verified
    status, or become verified without its own completed final rung. Reject
    duplicate newly credited evidence IDs and overlapping newly credited raw
    draw ranges within a stream; cumulative windows can reference committed
    evidence without counting it again. Do not reject
    identical draw-index numbers in distinct independent streams. A retry of an
    uncommitted infrastructure chunk contributes at most one scientific receipt
    while every execution attempt consumes work budget.
13. Two scopes with identical numeric epsilon/L and matching physical target
    names retain different candidate IDs, banks, evidence, and private handoffs.
    Cross-wire each scope/member/state/receipt at resume and replay and require
    rejection. Independent searches on the same scope also have different
    search/candidate IDs; only explicit resume retains them. The collection
    preserves every scope/search, reports partial scopes
    honestly, and never calls a runner or creates a handoff/verified status.
14. Discover `select_fixed_transport_candidate_set` in registry/export tests,
    enforce its diagnostic-only role and replacement, reject its payload in
    active replay, and detect any active consumer/example still using it as the
    required second validation lifecycle.
15. P2 qualification mode cannot activate the new public XLA default or erase
    backend blockers. P5 activation requires matching per-adapter evidence and
    full dependency eligibility; a failure cannot be hidden by another adapter's
    passing receipt or a caller-supplied readiness flag.
16. A valid `repair_step_higher` creates a child with higher epsilon but the
    same scope, exact L, mass signature, target, warmup protocol, and
    `candidate_family_id`; `repair_step_lower` has the analogous lower-epsilon
    invariants. The child has a new candidate ID/hash and a parent ID; retries of
    one unchanged candidate retain its candidate ID/hash and change only the
    verification attempt ID.
17. `inconclusive_evidence` and `inconclusive_conflict` preserve the candidate,
    do not change epsilon or L, and cannot cause a different L to be labeled its
    repair. An L replacement is accepted only from the explicit trajectory-veto
    policy, with a new candidate family and no inherited verification.
18. A directional repair requested during a cohort is deferred without
    mutating the active cohort, then receives repair priority after already
    reserved work closes and before not-yet-admitted different-L work. Its
    stream, draw range, old/new epsilon, exact L, mass signature, and source
    verification hash are deterministic on resume.
19. A computed-but-unexecuted repair, an executed-but-failed repair, and a
    budget-pending repair cannot produce a verified member or replay handoff.
    The repair exposes execution and verification states plus exactly one
    terminal qualified status: `executed_and_verified` or
    `not_executed_with_reason`.
20. When a required directional repair cannot be funded, the result contains
    `final_status="repair_budget_exhausted"`, `completion_status="partial_budget"`,
    and a structured repair record naming the parent/child IDs, old/new epsilon,
    exact L, mass signature, and remaining budget. This is not reported only as
    generic candidate exhaustion.
21. A deterministic fixture reproduces the MacroFinance failure: a candidate at
    one L receives directional high/low epsilon evidence, a different L is also
    present, and the trace proves the same-L child is scheduled and freshly
    verified first. The before/after trace preserves the failed parent and never
    calls a computed repair an executed handoff.
22. Epsilon repair factors stay inside the declared positive epsilon domain and
    candidate-family repair limit. Limit exhaustion records
    `repair_limit_exhausted`, preserves the parent, and does not switch to a
    different L under the name of repair.

Use the existing public API, outer-loop, fixed-transport, selection, dispatch,
route, replay-authority, and documentation tests as regression sources. Replace
old first-admission/quota assertions for the new version; preserve them only
where they test an explicitly historical reader. Do not leave contradictory
active behaviors merely to keep old assertions passing.

Known-target checks should exercise a small exact Gaussian and a smooth
nonlinear target with an explicit invertible transform, in ordinary and frozen
coordinates. Compare the same frozen finite transition/value/score to eligible
independent references. Add controlled resonance/no-movement and candidate-local
failure fixtures plus the candidate-specific repair fixture in test 21. These
establish mechanics and diagnostics, not a stochastic ranking or reliable
posterior estimation from a short run. Choose dimensions,
seeds, transition counts, and numeric tolerances from inspected fixtures or
checked error arguments in the P0 execution note before launching them.

### Proposed execution envelope

The planning task uses no HMC or GPU work. For later execution of this plan,
the proposed engineering envelope is 60 minutes cumulative CPU-command wall
time, 20 minutes cumulative GPU-process wall time, and at most four GPU smoke
launches including retries, on one existing local GPU at a time. These are
convenience-chosen cost ceilings for interface qualification, not statistical
sample-size claims. Compilation and failed attempts consume the same envelope.
Limit each GPU smoke to five minutes, derived from the total ceiling divided
by the launch bound. A run that cannot answer its engineering check within the
remaining budget stops as under-budgeted; do not weaken its check to fit.

Before those GPU launches, P0 must record the exact smoke cases, commands,
settings, and tolerances to the execution note under this contract. This is a
bounded specification task, not approval-token machinery. No serious M4,
NeuTra training, posterior, or sampler-ranking run is included. Such runs need
their own target-specific tuning and statistical evidence budgets.

Use a fresh output root for every execution/repair, proposed as
`docs/plans/artifacts/hmc-candidate-set-unification-2026-09-12/<UTC-run-id>/`.
Record Git commit and relevant dirty-file diff, command, environment, device and
memory-growth provenance, XLA/dtype settings, seeds, actual wall time, outputs,
plan/result paths, and remaining budget. Use ordinary checksums and atomic
checkpoint replacement where needed; do not introduce one-use authority files.

Future commands below are templates, not commands already run. Set the
task-specific `HMC_UNIFY_RUN_ROOT` to a fresh absolute output root and use the
inspected TensorFlow environment
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; verify its versions before tests.
The new test files must exist before their commands are used.

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/inventory_hmc_tuning_routes.py --check
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/audit_ordinary_hmc_migration_surface.py --downstream-root /home/ubuntu/python/MacroFinance --downstream-root /home/ubuntu/python/dsge_hmc --output-dir "${HMC_UNIFY_RUN_ROOT:?set a fresh output root}/consumer-audit"
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_hmc_candidate_set_tuning.py tests/test_hmc_candidate_set_artifacts.py tests/test_hmc_tuning_dispatch.py tests/test_hmc_tuning_contract.py tests/test_hmc_tuning_documentation_contract.py tests/test_hmc_tuning_policy_replay_authority.py
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_hmc_kernel_tuning_public_api.py tests/test_hmc_kernel_tuning_outer_loop.py tests/test_fixed_transport_hmc_tuning.py tests/test_fixed_transport_candidate_selection.py
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/render_hmc_tuning_interface_docs.py --check
```

CPU commands intentionally hide GPU devices. Tests importing TensorFlow must
establish that setting before import. GPU qualification checks require trusted tool
execution and `TF_FORCE_GPU_ALLOW_GROWTH=true` before import plus verified
memory growth before device initialization. Use the installed read-only GPU
probe for readiness, then a separate bounded smoke launcher. No nontrusted GPU
failure proves that the hardware or method is broken. Pure numerical kernels
need stable graph signatures. P2 constructs an explicit qualification path;
P5 runs and records compatibility, numerical equivalence, peak host/device
memory, compilation cost, and steady-state observations for each adapter class
before the public default is changed. If the budget cannot qualify all classes,
report partial qualification and leave those classes non-admitting; a reviewed
non-XLA exception must be explicit and cannot masquerade as GPU/XLA readiness.
Short checks do not establish performance superiority. No pfor or unreviewed
backend substitution is permitted.

Regenerate tables before their check. Build the guide from working directory
`docs`, using an absolute output directory outside the protected PDF path:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="${HMC_UNIFY_RUN_ROOT:?set a fresh output root}/book" main.tex
```

Inspect the resulting PDF and compare it with the saved
source baseline. Record pre-existing unrelated book build failures separately
and repair only those required for this delivery; do not quietly report a
chapter-only build as a successful full guidebook build.

## 9. Skeptical audit and pre-mortem

The initial skeptical self-audit preceded any implementation or experiments.
The naive “reuse the candidate-set helper” plan fails: the helper consumes
already tuned transports and cannot issue per-pair tuning handoffs. Merely
removing the early `break` also fails: the start cap, result schema, replay
selection, local repairs, budget reservations, and documentation still assume
one chosen pair. Both errors are addressed by P1–P4.

| Risk examined | Resolution or remaining limitation |
| --- | --- |
| Wrong baseline or stale context | Pin current code and the distinct M4 commit; compare control flow and same-kernel mechanics, not historical speed or acceptance outcomes. |
| Proxy metrics promoted | Acceptance affects proposals/description; mixing screens have declared roles; optional efficiency ranking needs its own uncertainty evidence. |
| Missing stop conditions | Candidate, scope, and total-budget failures are separate; incomplete members survive in the result/resume state. |
| Unfair comparison | Stage allocation precedes further allocation to earlier candidates; bind geometry/start design/budgets; report differing cost and compilation explicitly. |
| Hidden numerical defaults | Inventory above exposes inherited grids, thresholds, chain counts, and screen budgets; serious scope values require provenance. |
| Environment mismatch | Plan includes the NumPy dependency migration, CPU-hidden tests, and separate trusted GPU/XLA checks. A wrapper around the old ordinary runtime is insufficient. |
| Artifacts cannot answer the question | Tests must inspect execution traces and replay every verified member. A candidate list in JSON or one successful selected kernel is insufficient. |
| Candidate-specific repair is lost or misattributed | A later L is selected after directional evidence, or a computed epsilon is reported as a handoff | Immutable candidate/family/attempt hashes, typed direction mapping, repair-priority ordering, fresh child verification, and explicit `repair_budget_exhausted` status are required before admission. |
| Unlimited retention means unlimited computation | Preserve all records; allocate additional work only under the declared proposal/rung/repair and total budgets. Budget exhaustion is explicit incompleteness. |
| Guide rewrite loses rigor | Compare substantive source changes and inspect the rendered full book; preserve target/score, transform, uncertainty, and qualification arguments. |

Updated disposition: Claude's initial audit returned `REVISE`; the thorough R1
audit returned `AGREE`, but its report included references to absent plan
sections and unsupported budget formulas. R2 incorporates only source-grounded
parts of those reviews and the MacroFinance candidate-specific repair finding.
Scientific fixture values and GPU commands still belong in P0; the cost ceilings
do not determine them. Neither revision nor audit establishes implementation
validity.

A misleading success would be two adapters returning the same schema while
retaining different early-stop behavior. The cheapest discriminator is a shared
injected outcome schedule with several survivors exercised through each public
entry. A misleading failure would be GPU compilation timeout or short-chain
R-hat failure being blamed on candidate retention. Distinguish these with pure
controller tests and same-kernel reference checks before broadening runtime.

## 10. Completion and reset note

Completion requires one shared active procedure; all declared survivors either
receive their required work or are explicitly incomplete; all verified members
can be replayed; repair/resume works without lost evidence; legacy and typed
authority boundaries remain truthful; eligible TF/TFP execution passes the
scoped checks; and the book, reference, registry, and examples tell the same
story. A nominee is optional. Posterior inference remains separate.

The implementation result note must include a decision table with primary
criterion status, veto status, main uncertainty, next action, and unsupported
conclusions. Any stochastic summaries also need an inference-status table:
hard vetoes, viable candidates, supported ranking (normally none for smokes),
descriptive differences, default-readiness limits, and next evidence needed.
Keep engineering correctness, numerical validity, and scientific interpretation
separate. Add the strongest alternative explanation and evidence that would
overturn each material conclusion.

One material Claude plan audit and one terminal implementation/result audit
are the intended independent reviews. Findings must identify concrete source,
mathematical, numerical, cost, or migration risks. Review is advisory under the
current repository governance; formatting disputes or reviewer unavailability
do not create new launch authority requirements.

Reset for the next agent: this is the planned successor to “one tuner per target
class,” with same-`L` candidate-specific epsilon repair added in R2. Do not
execute a third parallel procedure, rerun M4, overwrite old results, or rewrite
unrelated dirty work. The next document check is the bounded R2 review specified
in the handoff; P0 begins after implementation is requested.
Current unrelated dirty files concern the
SSL-LSTM recovery runtime and sigma-point work; none were edited for this plan.

## 11. R1 response to Claude's initial audit

All five findings are accepted as specification or validation gaps. The audit
is unchanged; these dispositions describe revisions made by Codex, not a new
Claude approval or passing implementation evidence.

| Finding | R1 disposition and exact location | Remaining evidence |
| --- | --- | --- |
| 1. Public survivor helper unclassified | Section 1 and P0/P3/P4 explicitly retire `select_fixed_transport_candidate_set` from active tuning, retain a diagnostic compatibility export, and replace active consumers. Section 4.5 defines the non-executing collection; test 14 enforces the boundary. | Implementation inventory and consumer/registry/replay tests. |
| 2. Scope cardinality ambiguous | Sections 1, 4.1, and 4.5 require exactly one immutable scope per call/result/resume and scope-bound member replay. The chapter 26b rewrite uses separate calls and a read-only collection. Test 13 attempts cross-scope replay and collection misuse. | Per-adapter scope construction, signatures, and executable replay tests. |
| 3. Mutable queue fairness unspecified | Section 4.4 defines candidate/work-item states, closed cohorts, deterministic ordering, deferred children, fresh child reservations, evidence extension, terminal statuses, and persisted resume order. The explicit A/B/A1/A2 trace is required by test 11. | State-machine implementation and interrupted/funded/unfunded trace tests. |
| 4. XLA default before qualification | P2 now builds stable graph signatures and a non-default qualification mode. P5 alone activates defaults after per-adapter compatibility, equivalence, memory, compile-cost, steady-state, and complete dependency checks. Test 15 enforces the sequencing. | Actual per-adapter qualification; unqualified classes cannot be promoted. |
| 5. Weak cross-entry/holdout oracles | Tests 9 and 12 require controller-object identity plus exact injected work traces, failed-parent preservation, fresh child streams, duplicate/overlap rejection, and independent final-rung completion. Tests 11/13 cover mutation and scope isolation. | Focused tests once the code exists. |

The secondary default findings are also incorporated: section 4.1 requires an
operational start-bank recipe and coverage diagnostic in P0; the midpoint rule
is an explicit scope option rather than an unexamined common default. Budget
ceilings and scientific scope remain as in R0.

The exact audited R0 plan and initial handoff are saved under
`docs/plans/artifacts/hmc-candidate-set-unification-2026-09-12/plan-revision-r1-20260911T203548445004Z/`
as `audited-plan-r0.md` and `initial-audit-handoff.md`. This also records the
unchanged Claude audit checksum. The useful next check is a bounded re-read of
the revised sections above against the five findings; repeat source inspection
only to resolve a new concrete mismatch. There is no need for another broad
audit packet or an experiment at this planning stage.

## 12. R2 repair after the thorough audit and MacroFinance handoff

Claude's thorough R1 audit returned `AGREE`, but it was not fully source
faithful: it cited sections 3.1-3.4, 4.6, 6.4, 8.1-8.5 and geometry-scaled
budget formulas that do not exist in the R1 file. That audit remains preserved
as historical review evidence; its claims about absent text are not treated as
proof that R1 contained those rules.

The MacroFinance handoff identified a concrete failure mode in the current
implementation: after an inconclusive or directional verification, the
orchestration can select another `L`, compute an epsilon repair for that other
candidate, and finish without executing the recorded repair. R2 repairs the
plan at the normative level:

| Gap | R2 repair |
| --- | --- |
| Candidate identity was not explicit enough | Section 4.1 defines immutable candidate records and hashes, `candidate_family_id` for fixed-`L` repair lineage, parent IDs for children, and separate verification attempt IDs. |
| Directional repair could change L implicitly | Sections 4.3 and 4.4 require `repair_step_lower/higher` to change only epsilon while preserving scope, exact L, mass, target, coordinates, start bank, and warmup protocol. L replacement requires an explicit trajectory veto and a new family. |
| Inconclusive evidence could trigger a different-L repair | Section 4.3 maps `inconclusive_evidence` and `inconclusive_conflict` to pending/extension only; neither may change epsilon, switch L, or label another candidate as the repair. |
| Queue order did not prioritize the repair | Section 4.4 defers mutation until the active cohort closes, honors existing reservations, then gives the triggering same-L repair priority before not-yet-admitted different-L work and before the parent advances beyond its repair barrier. |
| Computed repair could look complete | Section 4.4 separates execution, verification, and qualified-handoff statuses. Only `executed_and_verified` yields a qualified repair handoff. |
| Budget failure was too generic | A required unfunded repair has typed `repair_budget_exhausted` metadata naming parent/child, old/new epsilon, L, mass signature, and remaining budget, while campaign completion remains `partial_budget`. |
| Evidence lineage was incomplete | Work items and result payloads bind candidate hashes, repair action IDs, attempt IDs, stream/draw ranges, source verification hashes, and fresh child evidence. |
| Regression coverage was too broad | Tests 16-22 add direction, identity, ordering, inconclusive-result, unexecuted-repair, typed-budget, repair-limit, and MacroFinance reproduction cases. |

The phrase "same candidate" in the external handoff is interpreted as the
same fixed-`L` candidate family, not the same immutable record: changing epsilon
must change the candidate record hash. This distinction is required for replay
and evidence integrity. The next audit should verify that this interpretation
is scientifically acceptable and that no remaining text contradicts it.

R2 is a plan repair, not a claim that the current Phase 14 artifact has been
repaired. The old artifact remains historical evidence of the orchestration
failure. No downstream consumer may treat its recorded `higher_epsilon` value
as executed or verified without a new BayesFilter result carrying the statuses
and hashes above.

## 13. R3 status and reserve-accounting clarification

The R2 audit is accepted as a coherent implementation review, but its budget
discussion used `partial_budget` both as a completion state and, in one place,
as a possible public final status. R3 makes the distinction normative:

| Field | Allowed values and precedence |
| --- | --- |
| `completion_status` | `complete`, `partial_budget`, `paused_infrastructure`, or `shared_invalidity`; it describes campaign completion independently of candidate evidence. |
| `final_status` | `complete` when `completion_status=complete`; `repair_budget_exhausted` when the next required work is a directional repair blocked by budget; `paused_infrastructure` when retry is forbidden and unresolved work is infrastructural; `shared_invalidity` when scope-wide validity is lost; otherwise `partial_budget` for unfinished non-repair work. |
| `qualified_repair_status` | Exactly `executed_and_verified` only after the child’s own fresh final verification passes, or `not_executed_with_reason` for every other repair state. |

`final_status` is not inferred from the presence of a verified member. A result
may have verified members and still be `partial_budget`,
`repair_budget_exhausted`, or `paused_infrastructure`.
`repair_budget_exhausted` is reserved for a directional repair that was
actually proposed and could not be funded; an ordinary incomplete candidate
set uses `partial_budget`. Shared invalidity dominates all budget statuses for
replay authority. The repair payload must record the blocked child, parent,
exact `L`, old/new epsilon, mass signature, remaining budget, and the reason
that selected the final status.

The reserve oracle is also explicit. A terminal parent releases only its
unspent reservation to the free pool with an accounting entry. A repair child
receives a new allocation from the declared repair/exploration reserve (which
may draw from that pool under the configured accounting policy); it never
inherits the parent work item, verification receipt, or reservation by object
identity. Test 7 and the P1 controller checks must assert these accounting
entries and the child’s fresh allocation.

## 14. R3 execution disposition

P0 inventory and P1 pure-controller implementation are complete under the
versioned execution root
`docs/plans/artifacts/hmc-candidate-set-unification-2026-09-12/execution-r3-20260912T120244Z/`.
R4 boundary repairs and their checks are recorded under
`docs/plans/artifacts/hmc-candidate-set-unification-2026-09-12/execution-r4-20260912T125338Z/`.
The controller/artifact tests, durable resume checks, registry discovery,
generated documentation check, and full guidebook compilation pass. The R4
artifact boundary also validates candidate hashes, parent/child scope identity,
unique verification attempts, non-overlapping credited draw ranges, and budget
conservation. P4 source updates cover chapters 21 and 21b, the reference
interface, the legacy-example classification, and generated route tables.

P2 numerical adapter binding, P3 replacement of the two legacy numerical route
implementations and downstream callers, and P5 per-adapter TensorFlow/TFP/XLA
qualification remain open. The current public numerical routes therefore retain
their existing authority classification until those phases provide evidence;
the new pure controller is explicitly diagnostic-only at this boundary. This is
an intentional partial execution status, not a claim that the legacy runtime
has already adopted the unified procedure.

## 15. R4 post-implementation audit and repair

The R3 implementation was re-audited against the actual source rather than the
earlier Claude summaries. Three material inconsistencies were found and
repaired before further numerical work:

| Finding | Failure mode | R4 repair and evidence |
| --- | --- | --- |
| Confirmed artifact/replay gap | A checksummed payload could still claim a verified child without proving its state, receipt, parent identity, or qualified repair status; shared-invalidity evidence could be replayed. | `hmc_candidate_set_artifacts.py` now validates issued candidate hashes, scope/backend/execution identity, receipts, parent/child fields, directional epsilon, repair status, and rejects shared-invalidity replay. The controller artifact is explicitly replayable mechanics evidence (`replay_authority=True`) while retaining `artifact_authority=False` and `numerical_handoff_authority=False` until adapter qualification. Focused artifact tests cover scope/hash and invalidation cases. |
| Confirmed resume gap | The controller could resume only from a live Python object; persisted work-item order, reserves, and attempt IDs were not reconstructible. | `HMCTuningCandidateSetController.from_result_payload` and `resume_hmc_candidate_set` reconstruct immutable records, active cohorts, repair queues, reserve ledger, and attempt ordinals from a checked artifact. The interrupted write/load/resume regression proves completed work is not repeated. |
| Confirmed guide contradiction | The reference interface still described `select_fixed_transport_candidate_set` as the recommended second validation procedure, contradicting Sections 1 and 7. | The reference guide now presents per-scope candidate-set results plus read-only cross-scope reporting and labels the selector/example diagnostic-only. Documentation contract tests reject the old recommendation. |
| Confirmed budget-accounting gap | Work items were recorded as reserved but never charged, so `remaining_budget_units` overstated the available campaign budget and an unfunded exploration could report `complete`. | Dispatch charges one unit from the candidate reservation; terminal release is recorded separately; result exposes used/reserved/remaining units and marks unfunded proposals partial. Conservation and unfunded-exploration tests pass. |

R4 intentionally does not claim that these changes migrate the two existing
TensorFlow/TFP numerical implementations. The current public ordinary and
fixed-transport routes still contain independent selection and handoff logic;
they remain the P2/P3 migration surface. The pure controller's artifact is an
engineering record, not posterior, convergence, sampler-ranking, GPU, XLA, or
scientific evidence. Numerical adapter wiring must preserve exact target,
transition, mass, source-closure, and authority contracts before any public
default changes.

The next smallest discriminating action is P2 adapter integration with a
shared injected-work trace and known-target parity fixtures. P3 then replaces
legacy active replay/selection consumers, and P5 qualifies each adapter's
TensorFlow/XLA path before activation. A remaining legacy route or failed
qualification is a partial migration status, not permission to weaken the
candidate-set contract.

## 16. R5 post-implementation audit and execution

The R4 source was audited again before treating the typed bridge as usable. The
audit found three authority-boundary defects and one test-only contract failure:

| Finding | Failure mode | R5 repair and evidence |
| --- | --- | --- |
| Confirmed artifact-state gap | A caller who recomputed the outer checksum could relabel an inconclusive receipt as verified, or omit the repair action for a verified child. | Artifact validation now requires candidate-state/verified/viable-ID agreement, passing verification decisions with no hard veto, one repair action for every child, directional source receipts, and qualified child status. Adversarial replay tests cover both mutations. |
| Confirmed resume-order gap | Persisted work items were not checked for unique contiguous order, candidate family/hash binding, receipt-to-work identity, or active running status. | The loader validates work-item order, candidate identity, cohort scope, verification-attempt uniqueness, and repair-action links before reconstruction. A reordered persisted queue is rejected. |
| Confirmed budget-ledger gap at the artifact boundary | A rehashed payload could report used or reserved units inconsistent with its append-only accounting events. | Artifact validation reconciles charged, allocated, and released units with the reported budget fields; the existing controller conservation tests remain green. |
| Confirmed receipt/work gap | A receipt could be attached to a pending or non-verification work item, making unexecuted evidence look replayable. | Receipt validation now requires a completed verification work item with matching candidate and attempt identity; over-reserved budgets and retained viable members on shared invalidity are rejected. |
| Test-only dispatch guard | The migration branch left the public dispatch module at the existing strict source-size boundary. | Removed one redundant blank line; the dispatch contract remains unchanged and passes at 149 source lines. |
| Fixed-transport caller policy declaration | The identity-admission benchmark instantiated the fixed-transport config without naming its diagnostic policy. | Added `legacy_directional_diagnostic_v1` to the identity fixture; this records its non-authoritative role without changing its stale-identity rejection behavior. |
| Checkpoint policy restoration | The Phase 9B restore call constructed a fixed-transport config without an explicit policy keyword, so the caller-policy contract could not audit it. | Restore the serialized policy explicitly, with `measured_joint_grid_v1` only as the backward-compatible default for older payloads. |

The typed bridge is now an explicit migration path. `tune_hmc_kernel` accepts
`HMCControllerConfig` only with a repository-issued typed adapter, and the
fixed-transport compatibility entry delegates to the same controller. Scope
identity binds target preparation, transition, backend, dtype, adapter, and
source closure. Both the typed run and its result remain
`numerical_handoff_authority=False` and `artifact_authority=False`; controller
execution is mechanics evidence, not target parity, convergence, XLA readiness,
or posterior evidence. The existing ordinary and fixed-transport numerical
defaults still retain their legacy authority and selection paths until P2/P3/P5
qualification is complete.

The R5 focused suites pass, including the dispatch, controller, artifact,
adapter, contract, and documentation checks. No GPU/HMC campaign was run and
no downstream MacroFinance or `dsge_hmc` caller was changed. P2 known-target
parity and TensorFlow/XLA qualification, P3 active-consumer migration, and P5
default activation remain open. A remaining legacy selector or an unqualified
adapter is therefore a partial migration status, not a reason to rank or
promote a candidate.
