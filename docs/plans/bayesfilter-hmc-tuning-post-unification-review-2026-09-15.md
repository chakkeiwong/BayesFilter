# HMC tuning after unification: code and guide review

Review baseline: `19ca6973` on `main`, committed and pushed before this review.
Question: Are public entry points and internal logic consistent, how can a
reasonable tuning problem still fail, and what refactoring is justified?

## Review plan and evidence contract

Trace each supported public call through preparation, proposals, measurement,
repair, evidence extension, verification, completion, persistence, and retained
sampling. Compare those paths with the reference, capability registry, guidebook,
and actual consumers. Inspect both success and failure transitions. Measure code
structure and dependency boundaries rather than judging quality by file size alone.

The comparator is the documented common candidate-set procedure and the actual
public API at this commit. A reproducible invariant violation, ignored relevant
option, missing necessary call path, or contradictory normative guidance is a
finding. Existing passing tests are evidence only for the cases they exercise.
No test in this review will establish target suitability, posterior convergence,
or sampler superiority. R-hat remains explanatory during tuning.

Use source inspection, deterministic controller fixtures, focused fault injection,
and tiny CPU numerical diagnostics only if necessary to distinguish explanations.
The convenience diagnostic ceiling is 15 CPU minutes; no GPU run, target-scale
campaign, package change, or runtime refactor is included. CPU numerical processes
hide GPUs before framework import and use the existing `tfgpu` Python environment.
Stop a diagnostic if it requires target-scale evidence, violates its assumptions,
or exhausts this ceiling; report the untested boundary rather than infer success.
Preserve commands, outputs, source anchors, and timing under
`docs/plans/artifacts/hmc-tuning-post-unification-review-2026-09-15/`.

Skeptical audit before diagnostics: the preceding repair's conclusion is not the
baseline oracle. Distinguish optional search-policy choices from genuine branch
divergence, candidate rejection from controller failure, and conservative
qualification from false scientific rejection. Check restart and budget semantics
at intermediate states, not only final successful checkpoints. Synthetic settings
are mechanics fixtures, not transferred scientific defaults. An unchanged
runtime source baseline permits unambiguous reproduction. This review plan passes
for that bounded engineering question.

One additional diagnostic exercises the automatic ordinary preparation/search
on a two-dimensional standard Gaussian, using the standard preparation config
with only an explicit CPU/non-XLA debugging exception. The existing broader
tests mostly supply epsilon grids or pause after the first automatic pilot.
This check asks whether the automatic path completes and preserves every
candidate's disposition, not whether its settings are statistically good.
Use the existing default tuning seed and budgets, an origin start, explicit
Gaussian lineage, and a 180-second external diagnostic timeout within the
15-minute review ceiling. A timeout is inconclusive engineering evidence,
not a scientific rejection. No runtime source changes accompany the check.

Follow-up diagnostic, audited before execution: the automatic Gaussian search
returned zero verified candidates and exhausted three repairs per L despite
opposite directional acceptance evidence. Reuse its exact frozen layers, start
bank, execution policy, target, and epsilon domain in a fresh search. Measure
the three log-spaced interior points between its observed 1.4386148 and
2.0345086 proposals for each original L, with the original evidence ladder and
no further proposal refinement. These points are derived from the failed
search, not suggested defaults. Preserve separate seeds through a new search
identity. A verified member establishes only that the automatic proposals missed
a candidate that passes the existing tuning screen. It cannot establish
posterior convergence, a preferred L, or a general success rate. Use a
180-second external ceiling within the existing total budget. Also test the
same proposal behavior with a deterministic monotone outcome fixture, and test
whether a controller result containing an in-band acceptance decision and a
separate promotion veto can survive its own writer/reader round trip. These
checks distinguish proposal and persistence defects without changing policy.

## Findings and answers

The procedure is unified at the public scheduling layer, but it is **not fully
consistent or robust**. The previous repair removed real problems; its passing
tests did not cover several interactions found here. This review leaves runtime
code unchanged and records the remaining repairs explicitly.
Read the earlier execution audit's closed findings as statements about its
tested cases; this review reopens the affected proposal, evidence, budget, and
documentation work with new counterexamples. F1–F10 remain open at review close.

| User question | Answer |
| --- | --- |
| One clear entry point for each case? | Yes at the supported facade level: two public tuner functions cover the declared cases and share one controller. Configuration translation, the registry, and generated guide text still contain inconsistencies. |
| Fully consistent internal logic? | No. Reproduced failures concern step proposals, acceptance/veto translation, artifact loading, budgets, seed freshness, and collection lookup. |
| Can reasonable tuning problems still fail? | Yes, both for legitimate finite-evidence/geometry reasons and avoidable implementation reasons. The automatic ordinary search returned zero verified members on a two-dimensional Gaussian; a fresh search of interior epsilons under the same frozen geometry verified 13 pairs. |
| Well structured; should it be refactored? | The shared controller and immutable candidate records are useful foundations. Active preparation still depends on a 30,188-line historical module, while duplicated execution and untyped observations have already drifted. Fix the demonstrated defects and then refactor in bounded stages. |

### Public routes and what is already consistent

| Actual target and coordinates | Supported entry point | Scope of its result |
| --- | --- | --- |
| Ordinary exact value/score, automatic geometry | `tune_hmc_kernel` with `HMCKernelTuningConfig` | Every verified exact pair is retained for checked numerical replay. |
| Ordinary exact value/score, already frozen geometry | `tune_hmc_kernel` with `HMCControllerConfig` and an issued numerical adapter | Same lifecycle and replay checks. |
| Exact transformed target under a supported frozen transport | `tune_fixed_transport_hmc_kernel`, with a frozen payload or issued numerical adapter | Same lifecycle in the declared latent coordinates. |
| Frozen position-only proposal field with exact endpoint potential | `tune_hmc_kernel` with `TensorFlowHMCKernelTuningConfig` and a runner binding | Same scheduling, conditional mechanics evidence; it cannot issue an exact-score retained member. |

The dispatch paths are `hmc_tuning_dispatch.py:29`,
`hmc_candidate_set_public.py:101,181`,
`fixed_transport_hmc_tuning_tf.py:823`, and
`hmc_candidate_set_position_field.py:152`. All reach
`run_typed_hmc_candidate_set` and `HMCTuningCandidateSetController`.
Preparation factories and historical selectors remain helpers, not additional
authoritative tuning procedures. Different target/coordinate preparation is
necessary; sharing scheduling must not erase those mathematical distinctions.

The active controller preserves distinct `(L, epsilon)` candidates, repairs an
epsilon through a new same-L child, independently verifies measurement survivors,
extends inconclusive evidence under declared limits, and retains all verified
members. A first passing member does not end the search. `nominee_id` is `None`.
The exact binding preserves the final geometry and start bank and checks
numerical evidence before retained replay. Shared invalidity disables replay.

**R-hat is reporting-only in the active tuning paths reviewed here.** It does
not select, reject, repair, or postpone candidates. The common evidence evaluator
does not consume it; `_report_rhat` catches unavailable/error diagnostics.
`runner.run_sequential` passes a selected member to the separate posterior
controller, where R-hat/ESS requirements still apply. That separation is correct.
The new Gaussian failure was not caused by R-hat.

### Remaining defects and repair priorities

P1 below means a correctness or usability repair to do before relying on the
affected path. P2 means an important bounded-search, configuration, or
maintenance repair. These are engineering priorities, not stochastic rankings.

#### F1 — P1: epsilon repair can discard useful opposite-direction evidence

At `hmc_candidate_set_tuning.py:1427`, `_request_repair` multiplies or divides the
current epsilon, clips it to the domain, and uses one geometric interior point
only when the resulting exact pair was already tried. It does not maintain the
nearby opposite-direction observations. After one interior test, the next
directional repair can leave the observed interval and spend the last repair on
a less informative boundary.

This is reproducible even with a deterministic monotone fixture: epsilons
between 1.6 and 1.8 pass, smaller values request an increase, and larger values
request a decrease. With three repairs, the controller tests
`1 -> 2 -> sqrt(2) -> 2.2`, retains no member, and reports `complete` with 96 of
100 call units unused. The defect does not require nonmonotone HMC acceptance.

The automatic ordinary Gaussian diagnostic exercised real preparation and TF/TFP
transitions using the standard config, with an explicit CPU/non-XLA exception.
All six L values followed variants of this sequence:

| Proposal | Observed role in this run |
| --- | --- |
| `1.0172542790` | Pilot acceptance above target at every L; increase requested. |
| `2.0345085580` | Acceptance below target, or initially inconclusive and subsequently below target. L=3 reached fresh verification and failed its acceptance screen. |
| `1.4386147978` | Acceptance above target; increase requested. |
| `2.1948885047` | Clipped domain boundary, beyond the already measured `2.0345085580`; acceptance below target. |

All six families exhausted the inherited three-repair limit. The result was
`complete`, zero verified members, 24 candidates, and 28 dispatched work items
out of 300 available units. Survivor refinement at
`hmc_candidate_set_tuning.py:1094` considers only verified or terminally
inconclusive candidates, so it did not reopen these directional failures.
`complete` correctly means the configured queue is exhausted; it is not a
success flag or evidence that no admissible kernel exists.

A diagnostic follow-up measured three log-spaced interior proposals,
`1.5688205613`, `1.7108109533`, and `1.8656525737`, at the original L values.
It preserved the original target, mass, coordinates, start bank, warmup,
execution, evidence policy, source closure, and epsilon domain; their nine
identity fields were checked for equality. A new search ID supplied fresh
random streams. Optional survivor refinement was disabled; the existing
directional repairs and evidence rungs remained available. The follow-up
verified **13 pairs: 10 initial interior pairs and 3 repair children**, covering
all six L values. It used 89 call units and also preserved one inconclusive
member. The complete records are in the linked results below.

This is evidence of missed viable tuning settings, not evidence that a new
algorithm is statistically superior. The two searches have different proposals,
seeds, and work counts; no speed or reliability ranking is supported.

Repair: persist all directional evidence at each L and prioritize unvisited
interior hypotheses between nearby opposing observations before returning to
external/boundary exploration. Real fixed-L HMC acceptance need not be monotone,
so this is a proposal policy, not permission to infer unmeasured passes or
permanently discard everything outside an interval. Declare finite fallback
coverage when all families fail before useful refinement. Review the inherited
three-repair limit in this new lifecycle rather than simply raising every cap.
Tests must include the deterministic reversal sequence and complete automatic
preparation/search on simple targets, not just explicit-grid successes.

#### F2 — P1: a rejected member can make a mixed result unloadable

The observation model separates an acceptance decision from promotion vetoes.
An in-band acceptance decision can coexist with `native_divergence_positive`.
The position-field adapter preserves these separate fields at
`hmc_candidate_set_position_field.py:118`. The controller correctly rejects
promotion, but stores the receipt's acceptance decision as `passed`.

`hmc_candidate_set_artifacts.py:247` rejects **any** receipt combining a passing
decision with a hard or promotion veto, even if that candidate is explicitly
`promotion_failed`. The writer at line 390 does not apply the same validation.
The deterministic reproduction wrote a result with one rejected member and one
verified peer, then the normal loader raised
`verified verification receipt contains a hard veto`. Position-field restart
uses this validator too.

Repair: give acceptance outcome and candidate qualification separate typed
fields. A rejected receipt with valid acceptance evidence must be serializable;
a verified member must still have its own passing, veto-free verification.
Apply the same invariants in writers, readers, and live replay. Test mixed
success/failure inventories and real position-field observation translation.

#### F3 — P1: native divergence has different repair roles across adapters

The common evaluator treats native divergence as a promotion veto while
preserving a supported directional acceptance repair
(`hmc_verification.py:1346,2207`). The position-field route follows that rule.
The exact adapter adds native divergence to `health_failures` at
`hmc_candidate_set_execution.py:459`, then at line 468 converts the entire
decision into `failed` and disables repair.

A synthetic finite trace with acceptance 0.2 and one native divergence produced
`repair_step_lower`, valid evidence, and a promotion veto in the common policy;
the exact adapter returned `failed` with `repair_eligible=False`. The parent must
remain unpromotable in both routes, but the child-repair eligibility should not
change merely with the adapter. Default TFP HMC often exposes no native
divergence bit; this reproduction proves a conditional classification defect,
not its frequency in actual targets.

Repair: share the observation translation and explicitly separate malformed
execution, candidate health, current promotion vetoes, and permitted repairs.
Do not relax the divergent parent's admission screen.

#### F4 — P1: retained block sampling omits tuning chunk seeds from freshness checks

`HMCCandidateRetainedRunner.run` checks tuning evidence base seeds and predecessor
block seeds at `hmc_candidate_set_retained.py:295`, but not independently seeded
later chunks within those tuning stages. `run_sequential` already checks both
base and chunk seeds at line 198.

An actual verified Gaussian member accepted retained seed
`(444194925, 1333419708)`, the seed of a second tuning chunk, and wrote a retained
archive. This violates the declared separation of tuning and retained
randomness. It does not establish a measured amount of posterior bias.

Repair: use one seed-inventory function for block sampling, sequential sampling,
exports, and continuation checks. Test every recorded tuning chunk, including
chunks completed before a restart. Avoid introducing elaborate launch tokens;
the existing recorded seed inventory is sufficient for this accidental-error
problem.

#### F5 — P2: an unaffordable peer blocks affordable verification

At `hmc_candidate_set_tuning.py:1279`, stage/cohort ordering places measurements
before verification. At line 1586 the first work item that exceeds remaining
gradient budget immediately stops the entire queue.

A fixture with L=3 and L=25 completed the L=3 measurement for 300 units.
The L=25 measurement needed 2,500 units, while L=3 verification needed 300.
Under a 600-unit cap the controller returned `partial_budget`, with zero
verified members and 300 units still available. Candidate reservations protect
call units, not the full numerical cost needed to reach verification.

This is an avoidable scheduling limitation rather than an invalid scientific
rejection. Repair: estimate and reserve mandatory numerical work at admission,
or explicitly defer an unfundable peer while completing affordable reserved
verification. Preserve the deferred candidate and incomplete search status;
never quietly reduce its required evidence. Test the interaction among cohort
fairness, heterogeneous L costs, repairs, and multiple budget types.

#### F6 — P2: chunk resume charges the whole stage again

`hmc_candidate_set_execution.py:500` estimates full-stage cost on every call.
`observe` resumes from saved chunks at line 519, but controller accounting at
`hmc_candidate_set_tuning.py:1600` adds that full estimate again.

An injected resource interruption preserved a 64-draw chunk of a 136-transition
stage (128 measurement draws plus 8 warmup, four chains, L=3). The first attempt
charged 2,176 work units. Resume completed only the remaining chunks but charged
another 2,176. At the 4,352-unit ceiling that would otherwise fund measurement
and verification, the candidate was screened but verification stayed pending.

The present accounting is conservative and the guide calls it an estimate;
this is premature exhaustion, not evidence of overspending. Attempt-count
charges can legitimately repeat, and interrupted native work may have uncertain
cost. Repair: distinguish completed chunk work, bounded attempted/uncertain
work, and reservations for unexecuted chunks. Do not charge completed chunks
again as newly scheduled numerical work. Test interruption before, within, and
after a chunk and compare accounting with an uninterrupted control.

#### F7 — P2: collection lookup omits search identity

`HMCTuningScopeCollection` permits distinct `(scope_id, search_id)` results at
`hmc_candidate_set_tuning.py:758`. Its `member` method at line 786 stops at the
first matching scope, even if the requested verified member belongs to a later
search. The reproduction accepted both searches, then rejected the second
search's valid member as unverified. With `expected_scope_ids`, `complete` at
line 777 similarly keeps only the last result for each scope.

Repair: use explicit scope/search/member identity consistently, or reject
ambiguous collections. Declare whether completeness refers to every search or
one explicitly designated search per scope; it must be independent of incidental
tuple order. Test multiple searches of one scope with different completion
states and disjoint verified members.

#### F8 — P2: configuration checks and preparation budgets remain incomplete

The ordinary wrapper starts operational preparation at
`hmc_candidate_set_public.py:132` before validating `search_config` at line 154
or constructing the execution config. A dispatch sentinel confirmed that
`search_config=object()` entered preparation before encountering the invalid
search option. The position-field route has the same ordering at
`hmc_candidate_set_position_field.py:169,180`.

Source inspection also shows that the common deadline takes effect only after
preparation returns. Recording preparation elapsed time makes later accounting
honest, but does not bound all preparation calls or preserve failed preparation
through the new numerical checkpoint. Native tracing/compilation cannot be
preempted by a Python deadline, as the guide correctly explains; the additional
gap is the uncheckpointed preparation phase before the shared scheduler exists.

Repair: preflight all public options and coordinate/capability compatibility
before numerical work. Pass a deadline and progress recorder through preparation
where interruption is supported, and preserve phase-failure information. State
which operations can only be bounded by an external process timeout. This needs
focused preflight/failure tests, not a new long target campaign.

#### F9 — P2: accepted options can be ignored or contradict another config

The typed fixed-transport branch at
`fixed_transport_hmc_tuning_tf.py:844` drops `search_config`, `execution_config`,
`target_lineage`, and `source_paths` instead of forwarding or rejecting them.
A dispatch spy reproduced those four omissions. The branch also does not
validate a redundant `frozen_transport_payload`. The ordinary typed branch
rejects duplicate search/execution options but ignores supplied lineage/source
overrides. An issued binding already fixes provenance; conflicting overrides
should be rejected explicitly, not appear to take effect.

Source-inspected migration inconsistencies:

- `TensorFlowHMCKernelTuningConfig.step_adaptation_results` remains a required,
  validated, serialized option (`hmc_tensorflow_tuning.py:270,342`). Its numerical
  use is in the historical candidate verifier at lines 844 and 850. Active
  preparation returns before that path at line 1022; the shared adapter uses
  `verification_results` for every pilot/measurement/verification stage. Changing
  the promised adaptation count changes identity metadata without controlling
  the advertised work.
- Ordinary and position-field `max_leapfrog_steps` filter the automatic primary
  grid (and its derived refinement grid) but are not checked against supplied
  search, refinement, or expansion grids. The maximum's scope is therefore
  unclear.
- The position-field config still defaults `use_xla=False`
  (`hmc_tensorflow_tuning.py:289`), while the ordinary config defaults to XLA.
  The route's conditional mechanics authority does not make this the owner-
  requested normal GPU/XLA policy. A debug exception should be explicit.

Repair: use a small preparation config plus the shared search and execution
configs. Translate each legacy option exactly once, reject unsupported or
conflicting options before execution, and explicitly mark inactive historical
fields. Define whether a supplied search replaces a legacy L bound, or require
all initial/refinement/expansion L values to obey it. Test the chosen semantics
through public entry points.

#### F10 — P2: the guide and registry still disagree about the position-field grid

The main reference and Chapter 21b correctly describe the shared broad-grid
procedure. The capability registry entry for
`bind_neural_force_hmc_tuning_runner` still says
“powers-of-two candidate screen ... not the ordinary broad-grid policy”
(`tuning_contract.py:1242`). That statement is generated into both
`docs/generated/hmc_tuning_route_table.md:33` and `.tex:64`, which the guidebook
includes. The active route uses the common broad L grid.

The guide also describes available native divergence among evidence-usability
checks (`ch21b_hmc_tuning_interfaces.tex:162`), while the common policy assigns
it a promotion veto. Clarify its role together with F2/F3.

Repair the registry source, regenerate both tables, and reconcile the narrative
with the repaired decision model. A generator consistency check alone will not
catch a stale claim shared by the registry and both generated files. Add a
semantic check against actual dispatch/search settings and inspect the rendered
changed pages when implementing that repair. This review changed no book prose
or TeX, so it did not rebuild the book again.

### Legitimate failures versus avoidable failures

Even a repaired tuner cannot guarantee success for every reasonable model in a
finite budget. The relevant distinctions are:

| Observed outcome | Meaning and justified response |
| --- | --- |
| No stable or useful pair within the declared L/epsilon domain | That search is exhausted. Revisit geometry, coordinate preparation, or explicitly expand the declared domain; do not declare the model untunable. |
| Acceptance intervals remain inconclusive at the evidence cap | Evidence is insufficient under the declared screen. Preserve the member as inconclusive; longer fresh evidence requires the corresponding budget. |
| Fixed-L return phases, rejection immobility, or heterogeneous chain behavior | Reject promotion of that setting and use supported step/trajectory repair. Acceptance is neither convergence nor an efficiency ranking. |
| Invalid target/score, missing required telemetry, inconsistent accepted states, or changed frozen dependencies | Respect the declared candidate or shared invalidity scope. Repair the implementation/evidence before reuse. |
| Weak local geometry, insufficient mass-adaptation information, or starts exploring an unrepresentative region | Preparation can fail; tuning can also pass locally without a useful posterior sampler. Improve preparation and assess posterior behavior separately. |
| Resource limits, unsupported XLA target operations, native compilation, or exhausted compute | Preserve an infrastructure/budget outcome. These do not reject the scientific model. |
| F1, F5, or F6 occurs | An implementation/search limitation may explain an empty set even when viable pairs exist. Repair the identified mechanism before expanding every budget. |
| R-hat fails in the posterior assessment | This can veto posterior use while leaving tuning membership unchanged. It is not a reason to revive an R-hat tuning gate. |

The acceptance band, compatibility interval, evidence ladder, L grid, initial
epsilon, and repair caps are operational choices with finite evidence. They do
not prove optimality, global coverage, or convergence. In particular, retaining
all verified candidates means all measured candidates that pass their own
checks, not every possible point in the continuous epsilon domain.

### Code structure and a bounded refactor

The [source and structure inventory](artifacts/hmc-tuning-post-unification-review-2026-09-15/source-and-structure.json)
records hashes, line counts, and largest functions. Selected measurements:

| Module | Lines | Structural concern |
| --- | ---: | --- |
| `hmc_kernel_tuning.py` | 30,188 | 551 functions including nested definitions; historical loops of 1,374 and 854 lines coexist with the active preparation path. |
| `hmc_candidate_set_tuning.py` | 1,714 | State machine, proposal policy, several budgets, persistence reconstruction, result validation, and reporting collection are combined. |
| `hmc_candidate_set_execution.py` | 693 | A useful numerical binding, but it also handles source closure, serialization, telemetry classification, chunking, and timing. |
| `hmc_tensorflow_tuning.py` | 1,958 | Active preparation shares a module/config with historical candidate verification. |
| `fixed_transport_hmc_tuning_tf.py` | 2,765 | Small active wrapper surrounded by historical selection/verification and a large legacy config. |
| `hmc_verification.py` | 2,349 | Central evidence policy is useful, but adapters independently translate its fields. |

File size alone is not the verdict. The concrete problems are dependency
direction and duplicated responsibilities. Active preparation imports private
helpers from the old monolith. Exact and position-field paths separately
implement chunks, seeds, deadlines, evidence dictionaries, and restart behavior.
Public options pass through `Any`-typed wrappers. The controller accepts string
decisions in mappings; the reader reconstructs their meaning independently.
The F2/F3/F4 defects are examples of the resulting drift.

Retained replay has a further scale limitation: `_validate_member` rechecks the
entire search's evidence inventory for each member and block
(`hmc_candidate_set_retained.py:59,222`). Exports embed that full inventory, and
archive continuation recursively reads preceding archives. Chunking bounds a
transition call, but completed traces and inventories still accumulate in host
memory. These source-visible costs have not been benchmarked at MacroFinance
scale; small successful tests do not establish scalable replay.

The recommended sequence is:

1. Add failing behavioral regressions for F1–F9 and fix them without reorganizing
   unrelated code. Make F2/F3 use one explicit observation vocabulary. Keep
   candidate rejection, evidence insufficiency, scope invalidity, infrastructure
   interruption, and queue completion distinct.
2. Extract a typed prepared-target result and a small preparation module. Keep
   historical configs/readers as explicit compatibility boundaries and remove
   active imports of old selection loops. Preserve target, coordinate, metric,
   warmup, and source identities.
3. Share the chunk execution lifecycle, seed inventory, timing, and restart cost
   accounting behind a narrow transition interface. Exact-score and position-
   field transitions retain separate capability/health requirements and replay
   authority; they need not share their numerical transition implementation.
4. Separate proposal generation and budget scheduling from state transitions.
   Persist the proposal evidence needed after a restart. Test small generated
   state sequences, varied candidate ordering/costs, interruption points, and
   writer/reader round trips.
5. Simplify public configs and generate route documentation from checked case
   definitions. Update the reference and book together. Preserve historical
   artifacts without silently upgrading them when source identities change.
6. Evaluate memory/I/O at representative inventory sizes before redesigning
   persistence. Prefer shared immutable evidence files and ordinary checksums;
   avoid another security/control subsystem for this trusted academic workspace.

Acceptance thresholds should remain fixed during these correctness repairs.
Proposal/default-policy changes need their own bounded evidence contract.
Use deterministic mechanics tests first, then full automatic tiny Gaussian and
anisotropic checks, followed by targeted GPU/XLA compatibility checks where
numerical execution changes. Larger target validation belongs in a separately
budgeted campaign. A whole-module rewrite would mix too many behavior changes
with untested extraction and is not justified.

### Evidence, assumptions, and review limits

| Material choice | Provenance and role | Failure mode / early check |
| --- | --- | --- |
| Standard Gaussian adapter | Existing `tests/test_hmc_candidate_set_execution.py::GaussianTarget`; independent small mechanics fixture. No dataset. | An explicit-grid-only test could hide automatic proposal failure; the first numerical search uses full standard preparation. |
| Automatic seed `(20260621, 8)`, broad grid, epsilon, three repairs, and evidence ladder | Inherited from the actual standard public config/preparation. Baseline, not newly justified defaults. | One-seed observations cannot estimate success probability or rank policies. Persist complete candidate histories. |
| Interior epsilons | Derived by log interpolation between observed opposite-direction proposals. Diagnostic hypotheses. | Different geometry or thresholds could falsely explain success; nine frozen-scope fields were checked equal. |
| Four-chain acceptance compatibility screen | Existing reviewed API policy; pass/fail criterion for this engineering existence check. | Repeated looks and short chains do not establish nominal sequential coverage or posterior convergence. |
| Deterministic budgets, monotone fixture bounds, interruption point, reused seed | Convenience mechanics constructions or exact recorded seed reuse. | They demonstrate state/role violations, not incidence on real models. The resource interruption is explicitly injected. |
| CPU/non-XLA and 180-second command ceilings | Explicit small debugging exceptions and convenience cost limits. | No GPU performance, XLA equivalence, production readiness, or numerical default promotion follows. |

Inspection covered the public facades, active preparation bindings, common
controller, evidence policy, exact and position-field adapters, result readers,
checkpoints, retained bridge, active NeuTra consumer, capability registry,
reference, Chapter 21b and neighboring mass/NeuTra guidance. Source inspection
and focused reproductions are not a proof over every historical helper or
arbitrary consumer. No new target-scale or GPU experiment was run.

The prior implementation audit reports 314 passing latest tests across its
progressive suites and a GPU/XLA explicit-grid smoke. This review did not rerun
that entire unchanged suite. Its new counterexamples show missing combinations:
automatic search through terminal completion, mixed receipt roles, heterogeneous
cost scheduling, interruption with partial work, multi-search collections, and
seed freshness across every tuning chunk.

| Check | Evidence | Result |
| --- | --- | --- |
| Seven targeted cases | [reproduction-r1/result.json](artifacts/hmc-tuning-post-unification-review-2026-09-15/reproduction-r1/result.json) | Reproduced adapter-role, affordable-work, collection, preflight, forwarding, restart-charge, and seed-freshness defects. Numerical body: 12.72 seconds after imports. |
| Full automatic ordinary Gaussian | [automatic-ordinary-r1/result.json](artifacts/hmc-tuning-post-unification-review-2026-09-15/automatic-ordinary-r1/result.json) | Returned normally, complete, zero verified out of 24 candidates; 31.82 seconds including imports. |
| Deterministic proposal and receipt tests; frozen-scope interior search | [followup-r1/result.json](artifacts/hmc-tuning-post-unification-review-2026-09-15/followup-r1/result.json) | Both deterministic failures reproduced; 13 numerical pairs verified, one inconclusive; 90.59 seconds including imports. |
| Protected unrelated changes | [protected-changes-audit.json](artifacts/hmc-tuning-post-unification-review-2026-09-15/protected-changes-audit.json) | All twelve pre-existing tracked file hashes unchanged. |

Scripts, complete numerical evidence, logs, exact commands, environment, seeds,
and baseline hashes are retained in the artifact directory. Numerical command
time remained well within the 15-minute diagnostic ceiling. The first script's
timer excludes imports; the other two include them. A combined
[run manifest](artifacts/hmc-tuning-post-unification-review-2026-09-15/run-manifest.json)
preserves those timing distinctions.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject a claim of full implementation consistency | Multiple reproducible invariants fail. | Engineering findings block that claim. | Additional untested combinations may exist. | Repair F1–F10 and add interaction coverage. | The entire implementation or prior repair is worthless. |
| Classify empty automatic Gaussian search as a search limitation | Same frozen scope supports independently verified interior pairs. | Default queue has no verified member; follow-up retains its own vetoes and failures. | Frequency across seeds and other targets is unmeasured. | Repair directional proposal memory, then assess the revised bounded search. | Gaussian/HMC failure, posterior correctness, or policy superiority. |
| Recommend staged refactoring | Duplicated responsibilities coincide with demonstrated drift. | Unbounded rewrite and silent authority changes are unacceptable. | Target-scale memory/I/O behavior is not measured. | Fix behavior, extract typed boundaries, then test compatibility and scale. | Refactoring alone establishes numerical validity. |

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | Failed or inconclusive settings remain unpromoted; the follow-up has 13 verified pairs under the existing screen. |
| Statistically supported ranking | None; no candidate or search-policy ranking was attempted. |
| Descriptive-only differences | Candidate counts, acceptance values, work counts, and runtime from these particular runs. |
| Default-readiness | Not established; the automatic search defect is sufficient to reject a blanket robustness claim. |
| Next evidence needed | Behavioral regressions, repaired automatic-path checks, and separately planned target/seed/GPU validation for any broader default claim. |

Post-review red team: a different search seed or a larger proposal budget could
explain some Gaussian differences. The deterministic monotone fixture establishes
the proposal-memory limitation independently of that explanation, and equality
checks exclude changed geometry as the cause of the follow-up's viable members.
The weakest evidence is extrapolation to expensive real targets; this report
makes no such success-rate or performance claim. The guide's current broad-grid
and no-R-hat-tuning direction remains appropriate. The open work is to make its
implementation and remaining documentation agree.
