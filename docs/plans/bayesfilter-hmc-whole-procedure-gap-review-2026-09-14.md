# Whole HMC tuning procedure: remaining-gap review

Date: 2026-09-14. Review baseline: `9329cadf` on `main`, including numerical
bridge implementation `9d06fbef`. Scope: preparation, public dispatch, shared
candidate lifecycle, acceptance/health decisions, verification, persistence,
retained replay, current consumers, tests, registry, and the guidebook.

Outcome: **unification remains incomplete**. Nine priority groups are detailed
below. The strongest new finding is an actual HMC execution that requests a
smaller epsilon but is discarded without repair. The review also reproduces
inconclusive searches incorrectly marked complete and missing failure persistence.

## Review intent and evidence contract

The question is whether the executed procedures now satisfy the user's original
request: broad L coverage, independent epsilon qualification, retention and
further tuning of every viable candidate, and R-hat reserved for posterior
assessment. The comparator is the revised September 12 unification plan and the
September 14 numerical-bridge contract, checked against current executable code.

This is an engineering review, not a sampler comparison. A reproducible
contract violation or a checked missing call path is a finding. A passing old
regression suite is supporting evidence for its tested behavior, not proof of
complete unification. Numerical rankings, convergence, MacroFinance target
validity, and new GPU qualification are outside this review.

Validation is bounded to source inspection, deterministic controller traces,
synthetic numerical-health/acceptance evidence, fault injection, and focused
existing tests. CPU devices are used deliberately with GPUs hidden before
TensorFlow import. No long HMC campaign or new scientific default is authorized
by this review. The diagnostic script and actual outputs will be preserved in
`docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/`.

The local diagnostic budget is ten CPU minutes, a convenience ceiling for
engineering checks rather than a scientific threshold. Stop a diagnostic if it
requires a target-scale campaign, package changes, GPU execution, or evidence
outside its stated contract. Synthetic constants and injected failures are
mechanics fixtures; they do not estimate how frequently these failures occur.

## Skeptical audit before diagnostics

The baseline distinguishes the newly implemented numerical bridge from the
still-active compatibility configurations. The review follows public dispatch
and consumers rather than inferring integration from exported names. Acceptance
screens and numerical health have separate roles; neither R-hat nor a short
Gaussian smoke can stand in for tuning correctness or posterior validity.
Checks will assert actual queue state, repair direction, error handling, and
durability rather than merely matching implementation strings. Source-only
claims and executed reproductions will be labeled separately.

The audit passes for this bounded engineering review. Remaining uncertainty is
the behavior of real target-scale consumers, which these checks cannot resolve.
Unrelated dirty files and historical artifacts will be preserved.

The deterministic role mismatch warrants one additional tiny CPU numerical
check under the same ten-minute ceiling: four chains, 64 measurement draws,
zero discarded warmup, standard two-dimensional Gaussian, L=3, epsilon=3.
The draw count is the existing acceptance policy minimum; L and epsilon are
deliberately unstable diagnostic fixtures, not scientific defaults. Run the
public candidate-set call and inspect whether its actual low-acceptance
measurement preserves the policy's smaller-epsilon repair. Preserve raw traces
and exceptions in a fresh `gaussian-oversized-step-r1` directory. Passing or
failing this fixture supports only the repair-routing finding; it cannot rank
samplers or qualify a real target. The audit passes because the observation
under test is dispatch behavior, not the deliberately poor candidate's quality.

## Findings and disposition

**The numerical bridge is implemented, but the complete tuning procedure is
still not unified. There are correctness gaps in the shared lifecycle as well
as unfinished migration and contradictory guidance.** The previous bridge audit
was too narrow to establish completion of the original unification objective.
Its successful replay checks remain useful; they do not answer the gaps below.

Priority P1 means repair before calling the procedure complete or relying on it
for unattended broad-candidate campaigns. P2 means a bounded implementation,
configuration, or documentation repair needed for a dependable public workflow.
These are engineering priorities, not scientific rankings.

| ID | Priority | Confirmed finding | Evidence |
| --- | --- | --- | --- |
| G1 | P1 | Public configuration types still choose different tuning algorithms and result contracts. The ordinary route still stops at first admission. | Public dispatch witness, existing queue regressions, source call chain. |
| G2 | P1 | The numerical adapter collapses promotion vetoes, repair triggers, and shared invalidity into one failure. A stuck, low-acceptance candidate loses its valid smaller-epsilon repair. | Synthetic role checks and an actual Gaussian HMC counterexample. |
| G3 | P1 | Inconclusive candidates become `validating`, but the result says `complete`; resume performs no further work. | Both inconclusive decisions reproduced through durable resume. |
| G4 | P1 | Partial numerical tuning has no complete durable checkpoint/resume path. Measurements and failures can lose their explanations; framework and reporting exceptions escape before persistence. | Fault injection, artifact inspection, source call chain. |
| G5 | P1 | The shared path measures caller-supplied pairs but does not implement the planned per-L pilot, refinement, expansion, or evidence-rung sequence. | Complete controller/config/evaluator inspection. |
| G6 | P2 | Repair/config validation admits reversed repair factors; proposal history permits duplicate settings and oscillating repairs. | Deterministic reproductions, including rejection of the controller's own issued result at replay. |
| G7 | P2 | Budget units count calls, not actual transition/gradient/time cost; a valid one-unit candidate reservation can strand verification with ample free budget. | Source accounting and durable-resume reproduction. |
| G8 | P1 | R-hat policy is not consistently separated from tuning across optional compatibility behavior, agent instructions, and the NeuTra chapter. | Executed fixed-transport regressions and checked source text. |
| G9 | P2 | The guide, registry, examples, and tests mix the intended procedure with current compatibility behavior; ordinary preparation still defaults to non-XLA. | Code/prose comparisons and passing inventory/documentation tests. |

### G1. Configuration still selects a different procedure

The public name alone does not identify a common algorithm:

| Public input | Actual executed procedure | Returned outcome |
| --- | --- | --- |
| `HMCControllerConfig` plus numerical binding | Shared candidate controller: supplied-pair measurement, one verification, bounded multiplicative repair. | All verified IDs in `HMCTuningCandidateSetResult`, with a separate numerical retained bridge. |
| `HMCKernelTuningConfig`, or omitted config | `_run_canonical_hmc_tuning`: operational warmup, independent per-L ladders, broad/refinement selection, older direct verification queue. | One final kernel through the older result type. |
| `FixedTransportHMCKernelTuningConfig`, or omitted fixed-transport config | Separate measured joint grid, replicated efficiency nomination, held-out verification of one nominee. | One selected candidate/final kernel. |
| `TensorFlowHMCKernelTuningConfig` | Separate position-field procedure with powers-of-two L candidates and first-passing selection. | One mechanics handoff; weaker authority remains correctly declared. |

[Public dispatch](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_tuning_dispatch.py:50)
has three explicit branches. The ordinary
[queue](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_kernel_tuning.py:25491)
sets `stop_reason="first_admission"` and breaks. Its older budget policy permits
two or three verification starts per outer attempt
(`hmc_kernel_tuning.py:18535,18608`). The fixed-transport
[selector](/home/ubuntu/python/BayesFilter/bayesfilter/inference/fixed_transport_hmc_tuning_tf.py:2346)
chooses one row, then verifies only that nominee at line 2380. The position-field
branch [skips later candidates](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_tensorflow_tuning.py:1089)
once a candidate passes.

The executed ordinary queue tests still explicitly expect a two-start limit
and later candidates marked `not_run` with reason `first_admission`
(`tests/test_hmc_kernel_tuning_outer_loop.py:3780,3828`). These tests demonstrate
the current mismatch; they must be reclassified as compatibility tests or
replaced when migration activates.

Repair: move supported preparation/config translations into the shared
lifecycle, preserving each branch's target and authority restrictions. A
position-only force must remain a distinct numerical adapter with weaker
authority, but it need not retain an independent scheduler. Keep old replay
readers and deliberately selected historical execution separate from the new
default. Test public calls with all ordinary candidates passing and require the
entire declared cohort to execute and remain replayable.

### G2. Typed evidence loses its meaning at the numerical boundary

The acceptance policy deliberately gives a supported smaller-epsilon repair
priority when rejections also cause poor movement
([hmc_verification.py:2243](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_verification.py:2243)).
Poor movement continues to veto promotion of the current kernel. Those two
roles are compatible: reject the current kernel, then test a repaired child.

In contrast,
[the numerical adapter](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_execution.py:447)
merges `candidate_promotion_vetoes` and engineering invalidity into `hard_vetoes`
and replaces the policy decision with `failed` whenever any veto exists.
[Measurement processing](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_tuning.py:1292)
then terminates that candidate before repair can occur. The controller's
verification branch also handles `hard_vetoes` before directional decisions,
so preserving the decision string alone would not fix the full call chain.

The actual Gaussian diagnostic used L=3 and epsilon=3. Its measured mean
acceptance was approximately `3.19e-297`; the policy returned
`repair_step_lower`, but the adapter returned `failed` with
`movement_gate_failed`. The public call executed one measurement, created zero
repairs, and reported `complete`. This is a real numerical counterexample to
the repair-routing contract, not a claim that epsilon=3 should pass.

The same normalization also discards the distinction between
`shared_execution_invalid` and candidate-local failure. Injecting corrupted
accepted draws makes the evidence evaluator explicitly return
`shared_execution_invalid` (`hmc_verification.py:1307,1317`); the adapter returns
ordinary `failed`, and a controller fed that observation can still complete and
verify another candidate. Scope-level invalidity must disable that scope's
replay. Merely rejecting the damaged candidate is insufficient under the
existing evidence contract.

Repair: carry validity scope, promotion vetoes, repair eligibility, diagnostic
alerts, and infrastructure status separately through observations and receipts.
Schedule the supported same-L repair while keeping the parent unpromotable.
Propagate genuinely shared invalidity to the scope's stop state with its
reason. Test both the numerical adapter and controller together, including
low acceptance plus stalled chains, true resonance, candidate-local domain
failure, and shared execution corruption. Do not reinterpret every domain
rejection as shared target invalidity.

### G3. Inconclusive is terminal in practice but labeled unfinished internally

[Both verification branches](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_tuning.py:1328)
set an inconclusive candidate to `validating` and release its remaining reserve.
They enqueue no extension or retry. The
[completion calculation](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_tuning.py:1412)
checks pending work and proposals but ignores `validating` states.

For both `inconclusive_evidence` and `inconclusive_conflict`, the reproduction
returns `complete`, one viable but unverified candidate, no pending work, and
18 of 20 budget units unspent. Durable resume calls the evaluator zero times
and returns the same result. There is no public evidence-extension operation.
Increasing numerical draw counts instead changes the binding's transition and
scope identity (`hmc_candidate_set_execution.py:510,515`); it is not a supported
same-candidate continuation.

Repair: introduce a predeclared finite sequence of evidence allocations and an
explicit inconclusive disposition. A pending candidate must have resumable work;
a candidate stopped at a declared evidence cap must have an explicit terminal
inconclusive status and reason. Completion must account for required stages,
including those not yet materialized as queue entries. Retain verified members
independently of completion, and do not promote an inconclusive candidate.
Test interruption/reload/extension and cap exhaustion for both ordinary and
repaired candidates. Repeated looks must not be described as nominal confidence
coverage without an appropriate predeclared uncertainty rule.

### G4. Durable retained replay works; durable numerical tuning is unfinished

[Numerical observations](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_execution.py:474)
are stored in `binding._evidence`, an in-memory dictionary. The public typed
runner [writes only after controller return](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_adapters.py:224).
Its result contains verification hashes but no measurement observation records.
A candidate rejected during measurement can therefore appear only as
`promotion_failed` on disk, without the acceptance evidence or failure reason.
The reproduction confirms that its controller payload contains neither
`movement_gate_failed` nor the underlying acceptance evidence.

Full numerical evidence can be persisted through
[retained-member export](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_retained.py:167),
which requires an already verified member. There is no public export/load path
for an incomplete execution binding and its complete evidence inventory. A
completed unsuccessful search also lacks that export path. Controller-only
resume correctly preserves queue order, but it cannot recover numerical
evidence that was never saved. Restarting with a new binding leaves earlier
verification hashes unresolved at the retained validator.

The controller catches only `HMCSharedInvalidity` and
`HMCInfrastructureFailure` (`hmc_candidate_set_tuning.py:1383`). The numerical
evaluator does not translate TensorFlow resource/runtime exceptions into those
types. Fault-injected `ResourceExhaustedError` escapes the public typed runner
without a result file. Likewise, an R-hat reporting exception at
`hmc_candidate_set_execution.py:470` occurs before evidence insertion and escapes
without preserving even the already computed measurement. This is not evidence
that a high R-hat value gates tuning; it is an error-handling gap in a nominally
reporting-only computation.

Finally, [the result writer](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_artifacts.py:386)
atomically replaces any existing destination. A second independent tuning call
can overwrite the first result in the same directory. The reproducer confirms
replacement. Atomic checkpoint updates and immutable final run artifacts need
distinct write semantics.

Repair: persist the execution specification, every measurement/verification
receipt and numerical evidence reference, controller state, and completed work
at bounded checkpoints. Offer a public numerical resume operation. Save failed
and zero-verified searches too. Use ordinary versioned tensor files and atomic
checkpoint replacement; no launch-token protocol is needed. Classify known
resource failures without disguising programming errors or target failures.
Save numerical evidence before optional diagnostics; record a reporting error
as unavailable diagnostic evidence. Refuse existing final destinations before
starting a new run. Tests should kill/reload between cohort members, inject a
resource failure after completed work, and resume through retained export.

### G5. Independent per-L proposal tuning and refinement remain caller work

[HMCControllerConfig](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_tuning.py:511)
requires an already supplied `epsilon_by_l`. It has no pilot/adaptation
schedule, refinement rule, expansion bound, replication schedule, or evidence
rungs. `_admit_primary_candidates` directly creates those pairs and enqueues
measurement. The only work stages are `measurement` and `verification`
(`hmc_candidate_set_tuning.py:464`); the evaluator rejects other stages.

Each supplied pair is independently measured. This is correct and must be
preserved. It does not implement the unification plan's independently bounded
per-L pilot prelude, all-survivor refinement, and further evidence stages
(`bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md:208,215,224,388`).
`add_exploration_candidate` is a manual controller operation, not a shared
public search policy. Numerical `repair_trajectory` is converted to failure;
there is no declared trajectory-repair implementation.

Repair: complete the common proposal/stage policy within this controller.
Explicit epsilon grids should remain valid initial hypotheses. A pilot may
propose epsilon, but fixed-kernel measurement and fresh verification must still
qualify each exact pair. Carry target-specific L coverage and expansion limits
in the config. Refinement must use all surviving families; it must not return
to nearest-acceptance winner selection. Use fixtures with several surviving L
values and several surviving epsilons at one L, and test complete proposal and
stage coverage through the public API.

### G6. Repair validation and proposal history need small concrete fixes

[Scope validation](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_tuning.py:124)
accepts any positive repair factor other than one. The repair rule multiplies
for `repair_step_higher` and divides for `repair_step_lower`, so factors below
one reverse the requested direction. With factor `0.5`, the controller created
and marked verified a higher-step child changing epsilon `0.4 -> 0.2`. Its own
replay validator then correctly rejected it: `higher repair did not increase
epsilon`. Validate `repair_factor > 1` before any work for this parameterization,
or use an explicit proposal rule with checked direction.

With a valid factor of two and opposing directional observations, the controller
visited `0.4 -> 0.8 -> 0.4`, then exhausted the repair limit. Manual exploration
can also admit the exact same L/epsilon twice as independent candidates. The
plan calls for exact-setting deduplication and recorded requesting parents.
Revisiting settings can be legitimate as a declared fresh evidence extension,
but silently creating new proposal identities does not implement that policy.

Repair: retain proposal history; distinguish a new setting, an explicitly
declared replication, and an evidence extension. Use a bounded alternative
proposal after direction reversal instead of cycling blindly. Every proposed
pair must still be measured; do not assume fixed-L acceptance is monotone.
Also reject fractional/bool L and budget inputs instead of silently applying
`int(...)` (`hmc_candidate_set_tuning.py:65,227,538`). Tests should cover reversed
factors, duplicate proposals, direction reversals, and invalid scalar types.

### G7. Budget semantics do not yet support the planned campaign contract

Every work item [costs one unit](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_candidate_set_tuning.py:471),
regardless of L, measurement/verification draw count, discarded warmup, target
cost, or compilation. The numerical evidence saved by `observe` omits the
runner's runtime metadata and measured elapsed time. The config has no wall-time
or gradient-work cap; a running numerical work item cannot be interrupted at a
controller checkpoint. These unit counts bound dispatched calls, which is
useful, but not the compute ledger promised by the plan.

Preflight also permits `candidate_reserve_units=1` even though a surviving
candidate requires at least measurement plus verification. In the reproduction,
one successful measurement consumes that reservation; verification cannot
start, despite 19 free units. Durable resume performs zero work because it
neither reallocates reserve nor rejects the invalid design earlier. The default
reservation of three avoids this particular first-pass case, but retries can
still exhaust a candidate's reserve while global budget remains.

Repair: derive minimum reservation from the declared stages, and make retry and
extension allocations explicit. Preserve both call counts and numerical work
(transitions/gradient evaluations, with elapsed time and compilation cost
reported). Enforce total campaign limits through bounded execution chunks and
record partial work. Include preparation cost in the campaign's accounting or
expose it clearly for the caller to combine. Test unequal L/draw costs and
resource retries; do not silently penalize longer L by granting it fewer draws
within a stage meant to give equal transition evidence.

### G8. R-hat is still contradictory outside the ordinary/shared pass decision

The ordinary high-R-hat regression passes, and the shared evaluator does not
read R-hat to choose its decision. Those repairs are intact. Three remaining
surfaces conflict with the user's requested tuning/posterior separation:

1. [Fixed-transport compatibility config](/home/ubuntu/python/BayesFilter/bayesfilter/inference/fixed_transport_hmc_tuning_tf.py:105)
   still accepts `require_modern_rank_normalized_verification=True`.
   `_classify_verification` adds a hard R-hat veto at line 2061. This option is
   **off by default**; the finding is the retained opt-in tuning gate, not a
   claim that every fixed-transport call requires R-hat. The executed regression
   `test_modern_verification_folded_rhat_vetoes_in_band_acceptance` proves the
   behavior. R-hat also remains a nominee tie-breaker at line 2360.
2. [The NeuTra chapter](/home/ubuntu/python/BayesFilter/docs/chapters/ch26b_neutra_transport_hmc.tex:331)
   says modern R-hat may veto members in the shared tuning screen. Lines 363–365
   permit a caller-declared R-hat validity screen/tie-breaker in the same tuning
   discussion. That does not match the new evaluator.
3. [AGENTS.md:297](/home/ubuntu/python/BayesFilter/AGENTS.md:297)
   still says “Tuning admission uses modern R-hat <=1.01” in the older NeuTra
   policy. It needs explicit posterior-stage terminology to avoid repeating
   the earlier agent confusion.

Repair: remove active tuning R-hat rejection/delay/ranking semantics, or isolate
them as historical readers/procedures with no new-tuning authority. Preserve
R-hat reporting, including unavailable/error status, and retain appropriate
R-hat/ESS checks in downstream posterior assessment. Update agent instructions
and all affected chapters together. Test high, unavailable, nonfinite, and
computation-error R-hat through each supported tuning adapter.

### G9. The guide and registry overstate current integration

The most direct contradiction is in the newly labeled ordinary compatibility
section of [chapter 21b](/home/ubuntu/python/BayesFilter/docs/chapters/ch21b_hmc_tuning_interfaces.tex:181):
lines 198–200 state that every retained member receives fresh verification and
a passing member does not terminate the cohort. The executed compatibility
queue does exactly that termination in G1. Lines 207–215 also describe the
shared immutable-child/extension lifecycle as current compatibility behavior.

The reference's [Ordinary Default Policy](/home/ubuntu/python/BayesFilter/docs/reference/hmc-tuning-interface.md:285)
mixes “migration target” with present-tense claims of complete-cohort
verification. Its fixed-transport budget section calls
`acceptance_target_distance` the default branch at line 652, while the actual
default is `replicated_min_bulk_ess_per_gradient`.

[The capability registry](/home/ubuntu/python/BayesFilter/bayesfilter/inference/tuning_contract.py:988)
advertises automatic per-L epsilon ladders and survivor-midpoint refinement
under the same public function that also accepts the supplied-pair shared path.
Its top-level artifact-authority flag and policy fields do not by themselves
resolve the chosen config branch's result and authority. The new nested bridge
metadata helps, but does not make the older fields true for every call.

The older examples `hmc_tuning_ordinary.py` and `hmc_tuning_fixed_transport.py`
still demonstrate separate configs. Several in-repository consumers, including
`neutra_end_to_end.py:1826`, call the ordinary compatibility route. The checked
benchmark inventory contains old fixed-transport callers; some are explicitly
historical and should remain historical, not be mechanically reactivated.

`HMCKernelTuningConfig.use_xla` still
[defaults to false](/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_kernel_tuning.py:6944).
The recommended preparation factory constructs `.standard()` when no config is
supplied (`hmc_kernel_tuning.py:24237`), inheriting that mismatch even though the
new measurement/retained binding defaults to XLA. This is existing default-policy
debt, not evidence that the new binding secretly disables XLA.

Repair: document actual behavior by config until migration is complete, then
make current normative prose describe the common procedure. Align the registry,
examples, public signatures, and active consumer wiring. Complete target-class
XLA qualification before changing preparation defaults; a tiny Gaussian smoke
alone does not qualify all state-space targets. Update the original plan's
completion ledger: numerical evaluation/replay now exists, but proposal/rung
completion, all-branch migration, consumer integration, and default activation
remain open.

## Checked components and remaining validation limits

| Component | Assessment |
| --- | --- |
| Frozen ordinary geometry | Both affine layers and checked post-warmup starts are captured/reconstructed by the new preparation binding. Earlier real-preparation and finite-difference regressions support this; no contrary result found here. |
| Per-pair kernel execution | The numerical runner receives the candidate's exact L and epsilon. Measurement and verification use distinct derived seeds. The new real counterexample exercised this path. |
| Multiple verified members | Shared results retain all verified IDs and no implicit nominee. The retained bridge requires explicit member identity and the member's own numerical verification. |
| Repair ancestry/replay | A selected child must pass; ancestors need not. Existing live/durable validation covers changed settings/scope/geometry/evidence. G2/G6 concern scheduling and input validation, not permission to skip verification. |
| Endpoint health | Accepted/proposed states and targets, endpoint score finiteness, momentum, Metropolis state consistency, declared endpoint status, and available native divergence are checked. Intermediate integrator target events remain dependent on the target's telemetry contract. |
| Retained continuation | Export/reload and predecessor endpoint/seed continuity are implemented and have earlier same-process/fresh-process tests. Tuning draws remain excluded. |
| Downstream posterior assessment | The new retained runner deliberately has no posterior-convergence authority. It reports per-block diagnostics in active coordinates and does not itself enforce cumulative model-coordinate R-hat/ESS or the sequential warmup schedule. There is no demonstrated repository integration here from this new runner to the existing sequential posterior controller. That integration needs a consumer test before posterior use. |
| Scalability | All tuning tensors are base64-encoded in an in-memory JSON inventory; each member export embeds the inventory; retained validation recomputes every verification receipt and recursively rereads predecessors. These are checked design properties, not a measured performance failure. Target-scale memory, restart latency, and long continuation remain unqualified. |
| Target identity and mathematics | Adapter signatures, source paths, lineage, frozen geometry, and start probes detect many accidental changes. Completeness of target/data/prior identity and correctness of the value/score still need consumer evidence. No cryptographic authentication expansion is recommended. |

The final-metric epsilon bound is also not explicitly carried by the new
preparation-binding specification: `_fixed_mass_step_upper_bound` feeds the old
per-L ladder, whereas the new binding takes a caller-supplied epsilon domain.
This is a missing policy/validation connection to resolve with G5. A bound
qualified at one L must not be asserted valid at every L; independent proposals
and measured qualification remain necessary. This review has not established
a numerical failure caused specifically by that omission.

## Current MacroFinance context

The September 14 memo describes an older pinned snapshot. Today's checked
MacroFinance source is newer: its typed worker calls
`bind_prepared_candidate_execution` and passes `binding.typed_adapter` to the
public tuner (`scripts/run_daily_asset_midas_phase14_typed_candidate_worker.py:95,106`).
Its support scanner now distinguishes implementation availability from exact
target qualification. It is no longer correct to describe that scanner as
unconditionally returning `supported=False`.

The current consumer proposes three factors around the operational warmup
epsilon for each broad L, then measures those pairs separately
(`daily_asset_midas_phase14_candidate_set_adapter.py:490,550`). Reusing a numeric
proposal across L is allowed because each pair is measured independently; it
does not transfer qualification. However, this illustrates G5: proposal choice
and its numerical assumptions still live in the consumer. The current 64-draw
measurement/verification design also directly encounters G3 when evidence is
inconclusive, and final export still depends on having a verified member (G4).

This is a source-wiring observation, not an executed MacroFinance campaign or
target qualification. CPU/non-XLA settings there are explicitly labeled as a
consumer exception; this review has not checked the approval provenance or
scientific adequacy of that campaign. No MacroFinance file was changed.

## Validation performed and what the tests establish

Artifacts:

- [Deterministic/fault-injection reproducer](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/reproduce_gaps.py)
  and [13-case output](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/reproduction-results.md).
  Final expanded execution took 3.015 seconds; the initial 12-case execution
  took 2.882 seconds before adding the shared-invalidity case.
- [Actual Gaussian counterexample](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/gaussian-oversized-step-r1/result.md),
  with its candidate result and full numerical evidence beside it. Execution
  took 4.783 seconds; the command and exact seed/config are recorded there.
- `focused-regressions.xml`: **53 passed**, 5.25 seconds; controller, artifact,
  adapter, documentation, and three fixed-transport R-hat tests.
- `legacy-routing-regressions.xml`: **3 passed**, 3.68 seconds; two direct-queue
  behavior tests and ordinary high-R-hat reporting regression.
- `route-inventory.json`: **20 discovered / 20 registered**, no unclassified
  or stale function names. This checks inventory consistency, not convergence
  of the different configuration branches onto one procedure.

All numerical checks used
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python` with `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, and `BAYESFILTER_TEST_DEVICE_SCOPE=cpu`.
The resulting CUDA-no-device messages are expected for deliberately hidden
devices and say nothing about GPU health. No GPU run, package change, benchmark
ladder, or real-target campaign was performed. Runtime implementation and guide
sources were not edited during this review.

Exact validation commands are preserved in
[validation-commands.md](/home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/validation-commands.md).
The twelve pre-existing dirty tracked files match their preservation hashes.

The existing 56 focused regressions passing does not contradict the findings.
They cover identity, replay rejection, and implemented queue behavior; several
explicitly preserve compatibility behavior now at odds with the intended
unification. The missing regression is often the interaction between individually
tested components: policy repair plus movement veto, inconclusive status plus
resume, numerical execution plus crash persistence, or public dispatch plus
complete-cohort coverage. The earlier 394-test bridge validation was not rerun
or represented as new whole-procedure evidence.

| Decision | Primary criterion | Veto/status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not declare full unification complete. | G1/G3/G5 fail the specified lifecycle. | Existing passing members retain their validated local status unless shared invalidity applies. | Real-target incidence and cost are unmeasured. | Repair the common controller and migrate remaining branches. | No rejection of HMC or the candidate-set design. |
| Repair the smaller-epsilon decision path. | Actual policy returned `repair_step_lower`; public controller created no child. | The deliberately oversized Gaussian parent correctly fails the movement screen. | The child's eventual acceptance is not evaluated here. | Preserve the parent's promotion veto and execute a measured child. | The parent should not be admitted; no claim that any particular child will pass. |
| Keep posterior assessment separate. | Ordinary/shared tuning does not require R-hat. | Optional compatibility gate and stale guidance remain. | No posterior study performed. | Remove conflicting tuning semantics and test downstream assessment explicitly. | No convergence, accuracy, efficiency, or default-readiness claim. |

There is no statistically supported sampler ranking in this review. Acceptance
from the tiny Gaussian diagnostic is used only to reproduce a declared repair
decision; the parent is not a research-method comparison. Default readiness for
all adapters remains unestablished.

## Recommended repair order and terminal review

1. Repair G2/G3/G6 together: preserve typed roles, make inconclusive work
   resumable or explicitly terminal, and validate repair proposals before
   execution. Start with the reproduced failures as regressions.
2. Add complete numerical checkpoints and failure classification (G4), then
   stage-derived reserves and compute accounting (G7). Prove interrupted
   numerical tuning can resume and export a verified member in a new process.
3. Complete the common per-L proposal/refinement/evidence policy (G5), then
   migrate supported public configs and active consumers to it (G1), preserving
   target-specific numerical and authority differences.
4. Align R-hat roles, agent instructions, guidebook, registry, examples, and
   tests (G8/G9). Qualify affected preparation/adapter defaults before activation.
   The final end-to-end fixture must start from preparation, create multiple
   survivors and a repaired child, pause/resume, and replay more than one member.

The strongest alternative interpretation is that the new controller was only
intended as a fixed supplied-grid evaluator. That interpretation explains its
two stages but does not satisfy the original unification plan or the guide's
claims about automatic proposals, further tuning, and completion. It also does
not explain the role-mapping bug, reversed repair factor, or missing failure
persistence. The weakest evidence here concerns real-target frequency and
performance; those remain unmeasured. The findings should be closed by public
call-chain regressions and revised behavior, not by adding another wrapper or
renaming a configuration.
