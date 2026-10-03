# HMC tuning follow-up review

Baseline: `5139f151`, after the candidate consistency repair. The user asks
whether public routes and the guide are consistent, whether internal logic is
consistent, how reasonable tuning problems can still fail, and whether the code
needs further refactoring. This is a review; runtime code stays unchanged.

Review status: completed. The unification is real, but complete consistency is
not established. This review reproduced five implementation gaps, found a broken
guide example and an important omitted explanation, and confirmed unnecessary
growth in persistence and replay work. The detailed findings below supersede any
interpretation of the previous repair's passing tests as a clean bill of health.

## Review plan and skeptical audit

Trace the supported public routes through configuration, preparation, proposals,
measurement, repair, verification, completion, persistence, resume, and retained
sampling. Compare executed behavior with the capability registry, reference,
guide chapters, examples and current consumer calls. Check defaults and failure
transitions as carefully as successful explicit-grid examples. Inspect numerical
evidence translation, repair budgets and immutable record relationships, then
assess dependency structure and test gaps.

The comparator is the actual committed code and the current documented
procedure, not the previous audit's closed labels. A finding requires a source
trace or a focused reproducible counterexample. Use deterministic fixtures first;
tiny CPU debugging checks are allowed when they discriminate a numerical or
persistence hypothesis. Passing existing tests will not be treated as proof of
complete consistency, and a finite-search miss will not automatically be called
an implementation bug. Separate invalid evidence, candidate rejection, unsupported
claims and untested behavior.

The convenience review ceiling is 15 CPU minutes for diagnostics, with no GPU
launch, model-scale campaign, threshold change or runtime refactor. Numerical
processes hide GPUs before imports and use the existing `tfgpu` interpreter.
Record commands/results under
`docs/plans/artifacts/hmc-tuning-followup-review-2026-09-15/`. Stop a diagnostic
that requires target-scale evidence or exceeds the ceiling, and state that
boundary. All unrelated worktree changes remain untouched.

Skeptical audit: previous tests cover useful counterexamples but cannot establish
cross-route equivalence, arbitrary resume histories, true numerical cost, or
posterior validity. Synthetic traces test interpretation, not correctness of the
underlying integrator. Read-only replay of existing final-source artifacts can
test the live validator without changing the target or spending a new sampling
budget. Shared scheduling does not justify merging exact-score and arbitrary
position-field authority. These distinctions make the bounded review meaningful;
proceed.

## Focused diagnostic contract

Use deterministic controller observations to test whether released reservations
reactivate affordable work within the same invocation. The comparator is an
unchanged resume of exactly the same controller state; no extra search budget or
new evidence policy is supplied. A difference in completion caused solely by
calling resume is an engineering finding, not a stochastic comparison. Fixture
budgets and decision sequences are convenience choices derived to exercise the
reservation boundary. Also inspect candidate-local TensorFlow exceptions and
replay an existing unit-Gaussian checkpoint without new sampling, if its source
identity remains valid. Replay success tests artifact consistency only; it does
not establish the correctness or convergence of the Gaussian sampler. Record
commands, outputs, timing and source baseline in the artifact directory above.

For the retained boundary, test two legal ordinary inputs separately: a scalar
exact Gaussian and a batched exact Gaussian without optional target-status
telemetry. Use the existing regression's 128-draw evidence allocation, four
chains, epsilon 1.3, L=3, and wider mechanics-only acceptance band; these are
inherited debugging fixtures, not default-policy evidence. Require an actual
verified member before exercising the posterior bridge. A 16-draw warmup cap
only tests dispatch and never establishes posterior readiness. Stop after these
two fixtures or on absent verification; do not tune fixtures to manufacture a
desired result. Inspect derived posterior seed sequences without running chains,
and exercise declared target exceptions through an injected native-call failure;
that injection establishes exception handling only. The total 15 CPU-minute
ceiling remains unchanged.

## Verdict and answers to the four questions

| Question | Verdict |
| --- | --- |
| One clear entry point per case? | Yes at the tuning facade: ordinary exact-score inputs use `tune_hmc_kernel`; supported frozen transports use `tune_fixed_transport_hmc_kernel`; the conditional proposal-field case uses `tune_hmc_kernel` with its typed config/binding. Prepared geometry enters the same controller. The whole tuning-to-posterior path is not yet consistent: `run_sequential` adds unadvertised target requirements. |
| Fully consistent internal logic? | No. Shared-invalidity history, derived posterior seeds, native target-exception handling, released reservations and posterior dispatch each have a reproduced counterexample. |
| Can reasonable tuning problems still fail? | Yes, through these defects and through legitimate limits of finite search, frozen geometry, local starts and finite evidence. Empty verification is not evidence that the target has no usable HMC kernel. |
| Well written and structured; refactor? | The controller/decision/proposal split is useful. The implementation as a whole still mixes historical and active contracts, duplicates runtime/persistence logic and repeatedly processes complete histories. A staged refactor is justified after the focused correctness fixes. |

The desired procedure remains sensible: freeze the target and geometry, search
each exact `(L, epsilon)` pair, repair with fresh candidates, verify every
survivor, and retain every verified member. No new winner selection or R-hat
tuning requirement is needed. In the active paths inspected, R-hat is
reporting-only, including diagnostic computation failures. Posterior R-hat/ESS
assessment remains separate.

## Confirmed findings

### R1 — The posterior bridge changes the target execution contract (high priority)

`hmc_candidate_set_execution.py:409` runs independent scalar chains through the
issued numerical binding. `hmc_candidate_set_retained.py:131` and `:196` hand
the selected member to the separate sequential controller, which builds a
batched program at `neutra_hmc.py:398` and `:580`. It also calls target-status
telemetry whenever the wrapper exposes a callable, without preserving the
tuning binding's `target_status_trace_policy='none'`.

Two actual Gaussian mechanics runs demonstrate the consequence. Each completed
tuning with one numerically verified member:

| Legal tuning input | `run_sequential` result |
| --- | --- |
| Exact target accepting rank-one inputs, with scalar-chain tuning | `ValueError`: target received rank two, shape `(4, 2)`. |
| Batched exact target without optional target-status telemetry, tuned with status policy `none` | `TypeError: base_adapter must expose target_status_telemetry`. |

The affine wrapper's always-present telemetry method is the immediate cause of
the second failure (`batched_value_score.py:498`). Both targets satisfy the
ordinary tuning path's actual requirements. These failures occur before
posterior convergence assessment; R-hat did not cause them. The direct
`member.run` archive path uses the original binding and is a separate case.

Repair: let the sequential controller consume the checked member's numerical
transition and health evaluator, preserving the supported scalar/batched
topology and declared telemetry policy. If a different execution contract is
required, expose and qualify it explicitly before sampling. Merely adding
batch support to the scalar test target would hide this library boundary defect.
Test both targets through actual public tuning, member construction and the
first posterior chunk, plus declared status failures and coordinate transforms.

### R2 — Shared invalidity makes a previously verified repair history unreadable (high priority)

After a repaired child verifies, its action correctly records
`executed_and_verified`. If a later peer reports shared invalidity, the
controller clears all replayable membership at
`hmc_candidate_set_tuning.py:1710`, preserving the earlier action as history.
The artifact validator nevertheless requires that action's child to remain in
the now-empty verified set (`hmc_candidate_set_artifacts.py:329`).

A deterministic two-L example returns `completion_status='shared_invalidity'`
and no verified IDs, but validation raises
`ValueError: qualified repair child is not verified`. The final result writer
uses this validator; the numerical checkpoint writer can save the inconsistent
payload, which the numerical loader then rejects. The failure cannot be saved
and reloaded through the normal result interface. Replay stays blocked, and
the underlying immutable numerical evidence remains on disk.

Repair: distinguish the fact that a child passed before invalidation from its
current eligibility for replay. A shared-invalid result must preserve its
history, serialize and reload, while allowing no numerical handoff. Add a
round-trip test for a verified repair followed by shared invalidity, as well as
the same event after an unrepaired verified candidate. Do not delete the old
passing evidence or re-enable the member to satisfy the validator.

### R3 — Posterior seed validation checks roots, not the executed chunk schedule (high priority)

`hmc_candidate_set_retained.py:209` checks only the supplied warmup and retained
root seeds against the complete tuning seed inventory. The posterior controller
actually executes seeds derived at `neutra_hmc.py:652` and `:730`, using
`(a, b + 1009 * (chunk_index + 1))` (`:1536`).

The review loaded an unchanged, previously verified member and supplied a root
whose first derived seed equals a recorded tuning seed. The bridge accepted
and delegated it. The downstream call was mocked, so this establishes the
validation gap; it does not establish identical tuning/posterior random draws,
because their runner topologies differ.

There is also a direct overlap within the posterior controller itself:
`warmup_seed=(17, 0)` and `retained_seed=(17, 1009)` are accepted, but the second
warmup chunk and first retained chunk both use `(17, 2018)` with the same
batched runner when chunk sizes match. Distinct roots do not ensure distinct
streams. No chains were executed for this seed arithmetic check.

Repair: derive phase-separated seeds through one shared implementation and
validate the complete bounded chunk inventory against tuning and the other
posterior phase. Validate int32 bounds too. Preserve historical seed schedules
when reading old evidence; changing a live schedule requires a new policy/run.
Tests should cover later chunks, unequal chunk sizes, maximum allocations,
attempted tuning chunks, and restart. This is an evidence-independence fix,
not a reason to use R-hat in tuning.

### R4 — A declared target-domain exception pauses every candidate (medium priority)

The numerical observer catches only a short list of resource exceptions around
the native call (`hmc_candidate_set_execution.py:541`). An adapter-classified
`InvalidArgumentError` is not translated into candidate-local failure.
The controller treats it as an unexpected exception, marks the work interrupted,
and rethrows (`hmc_candidate_set_tuning.py:1662`). Its unchanged resume attempts
the same work and seed again.

A native-call fault injection raised an error for which the bound adapter's
`classify_target_exception` returns true. The public tuner propagated the
`InvalidArgumentError`, wrote `paused_infrastructure`, and never attempted the
other admitted L. This tests exception handling, not a new numerical failure
of a real filtering model. The preparation code already distinguishes declared
target-domain failures from unclassified TensorFlow failures
(`hmc_warmup.py:2699`), so the active candidate executor has lost an existing
distinction at this boundary.

Repair: preserve a typed candidate execution-failure record, charge the attempt,
reject that pair and continue its funded peers. Only a specifically declared
target failure may receive this treatment. Shape errors, broken execution and
unclassified TensorFlow exceptions must not be silently called low acceptance.
Missing acceptance evidence must not manufacture a directional repair; any
additional proposal needs its own declared rule and measurement. Test the
failure record through checkpoint loading and export of an unaffected member.

### R5 — Released reservations do not reactivate deferred work (medium priority)

`_defer_unfunded_work` puts a work item into `_budget_deferred`
(`hmc_candidate_set_tuning.py:1577`). `_next_cohort` excludes it for the rest of
the invocation (`:1310`). Releasing another candidate's unused reservation
does not reconsider it. When no nondeferred work remains, the controller returns
`partial_budget` (`:1607`).

The deterministic reproduction has two primary candidates and seven call units.
A repaired child needs three measurement rungs, exhausting its reservation.
Its verification is deferred; the other candidate then verifies and releases
one unit. The first invocation stops with six units spent, one available, and
one verification costing one unit pending. An unchanged checkpoint resume
finishes at seven units and retains both candidates. No budget increase or new
search policy was needed.

Repair: reconsider deferrals when the relevant available resource changes, or
perform one progress-sensitive rescan before stopping. Do not repeatedly retry
a permanently unaffordable gradient allocation. Test released call budget,
permanent gradient exhaustion, and mixtures of both without busy loops or
duplicate charges.

### R6 — The guide contains a broken invocation and omits the exact acceptance rule (medium priority)

The proposal-field example included by the book declares
`step_adaptation_results=1` (`docs/examples/hmc_tuning_neural_force_binding.py:66`)
and calls the public tuner without disabling the default pilot. The shared
execution config requires at least 64 pilot decisions. Calling the documented
`tune_deterministic_field` function fails before preparation with
`pilot_num_results must be an integer >= 64`. Existing documentation tests run
its payload-construction `main`, which never calls that function
(`tests/test_hmc_tuning_documentation_contract.py:303`).

The reference and book mention a compatibility interval but do not state the
actual pass rule at `hmc_verification.py:2207`. After conflict and pathology
checks, the interval must overlap the practical band, lie inside the repair
band, and have every chain mean inside the repair band. A pooled mean outside
the practical band can therefore pass. A synthetic interpretation check with
chain means `(0.73, 0.75, 0.77, 0.79)` gives mean `0.76`, interval approximately
`[0.7296, 0.7904]`, and `passed` under the default policy. This is not HMC or
posterior evidence; it shows precisely what the policy computes.

That behavior follows the current code. It should be explained explicitly,
because the original user confusion concerned means just above `0.75`.
Readers should not infer a hard pooled-mean cutoff from the band alone.
Document chain conflict, temporal conflict, directional decisions, movement
vetoes and the separate fresh verification in a compact decision table.
Repair the example with a declared eligible pilot allocation or an explicit
pilot-disabled search, then test its real dispatch. Keep the existing
reporting-only R-hat explanation.

There is smaller navigation/configuration debt: the registry's replacement for
`select_fixed_transport_candidate_set` points to `tune_hmc_kernel`, although
the normal frozen-transport replacement is `tune_fixed_transport_hmc_kernel`.
Legacy config names and payloads still expose old selection/verification fields.
For example, `FixedTransportHMCKernelTuningConfig.require_all_chain_movement`
is accepted by the config but is not read by the active public translation;
movement is controlled by the shared acceptance policy. Each such field should
be translated, rejected on override, or explicitly identified as historical.

### R7 — Persistence and replay repeatedly process the entire search (medium priority; structure/performance)

The read-only replay of the complete automatic Gaussian search succeeded: 162
numerical evidence records, 18 verified members. Inspection of the actual
writer and a dry call showed that one checkpoint revisits all 162 records
(`hmc_candidate_set_checkpoint.py:38`). For existing immutable evidence,
`_write` serializes it again, reads the existing file and parses both JSON
documents (`:22`). Every later chunk and work boundary repeats this work.

Building one retained member also recomputes all 162 analyses, including unrelated
peers (`hmc_candidate_set_retained.py:58`). `export`, each subsequent `run`, and
member loading invoke that full validation again. Every member export embeds
the whole evidence inventory. The active NeuTra consumer exports every verified
member separately (`neutra_end_to_end.py:1861`). The inspected inventory alone
occupies 9,937,304 JSON bytes for a two-parameter fixture. This is a measured
inventory size and a counted operation pattern, not a target-scale performance
benchmark.

With similarly sized receipts, repeated full-inventory processing grows
quadratically with the number of completed work items; exporting V members
duplicates approximately V inventories. Retained predecessor validation also
recursively rereads the preceding archive chain. These choices preserve useful
evidence but can make host memory, JSON work and restart time dominate larger
problems. The guide correctly says target-scale memory/restart performance is
unqualified; it does not establish that this implementation scales.

Repair: use one shared immutable evidence store with compact member references
and an explicit portable-bundle export. Persist newly completed evidence once;
update only the checkpoint index and ledger. Validate a complete immutable
inventory once per load, preserving scope invalidity and ancestry checks, then
reuse that checked state until its inputs change. Use iterative predecessor
validation. Avoid adding new approval or cryptographic infrastructure: the
existing checksums and versioned files are sufficient for this trusted workspace.

## Remaining legitimate ways a reasonable tuning problem can fail

| Cause | Meaning and next justified action |
| --- | --- |
| Preparation fails: invalid starts, bad curvature, rejected metric update, bootstrap or target-domain error | Candidate search may never begin. Inspect `preparation_progress.json`, geometry and declared target health; correct the inputs or the preparation mechanism in a fresh run. This does not reject the posterior model. |
| Frozen geometry is inadequate across the explored region | A single covariance can fit local curvature poorly, especially with strong scale variation or separated modes. Another geometry/transport is another scope; acceptance tuning alone cannot repair the map. |
| Finite L coverage or epsilon domain misses viable regions | The convenience grid and final-metric epsilon ceiling are bounded proposal hypotheses. Fixed-L acceptance can be nonmonotone. A valid kernel may lie outside the declared domain or between tested points. Expand a justified finite search; do not infer instability everywhere above a local ceiling. |
| Family, candidate or evidence caps are reached | Default repairs and `(1, 2, 4)` rungs do not guarantee discovery or a decisive screen. The default ordinary candidate evidence begins at 64 decisions per chain; repeated fresh rungs are not cumulative posterior samples. A new budget or search needs a new declared run. |
| Chain/block disagreement, recurrence or target-local numerical invalidity | A finite pooled acceptance mean is insufficient. Some candidates legitimately fail while others remain usable. Inspect typed reasons and follow the permitted repair; never turn invalid data into acceptance feedback. |
| Time, gradient or invocation budget is exhausted | `partial_budget` can coexist with verified members. Native calls/compilation cannot be preempted by the cooperative deadline. Distinguish a true resource limit from R5's stranded affordable work. |
| Fresh verification differs from measurement | These are different finite stochastic runs. Survival is a screen, not superiority or a guarantee of repeated passing. Do not cherry-pick a seed or silently relax the criterion. |
| Starts and all short chains explore the same limited region | All candidate screens can pass while global mixing remains poor. Separate model-coordinate posterior diagnostics and target-specific evidence remain necessary. |
| Resume or replay rejects a source, target, geometry or environment change | This is intentional identity protection. It is not a newly failed numerical candidate. Use a preserved compatible checkout or a fresh run; unlisted data/prior dependencies still require consumer provenance. |
| Posterior R-hat/ESS or downstream scientific checks fail | Reject that posterior use while preserving the tuning history. This is not authority to reintroduce R-hat as a tuning gate. |

The default acceptance target/bands, four-chain/temporal-block design, movement
thresholds, evidence rungs, broad L grid and refinement factors are inherited
operational policy. This review does not recalibrate them or establish their
universal adequacy. The guide already disclaims nominal repeated-look confidence
coverage; the missing exact decision rule in R6 should be added alongside that
qualification.

## Refactoring recommendation

Keep the one controller and the existing separation between acceptance evidence,
promotion eligibility and posterior assessment. The pure decision/proposal
modules and immutable candidate records are improvements worth retaining.

The active implementation is still tightly coupled to a 30,047-line
`hmc_kernel_tuning.py`, a 7,472-line `hmc_warmup.py`, a 1,966-line
`hmc_tensorflow_tuning.py`, and a 2,773-line fixed-transport module containing
both the facade and historical selection. The controller itself is 1,757
lines. These are measured line counts at this baseline; line count alone is not
the verdict. The important problems are mixed responsibilities, private
cross-module imports, parallel checkpoint implementations, loosely typed
observation dictionaries and repeated interpretation of the same evidence.

Use the following order:

1. Add regressions and repair R1–R5 without changing the acceptance policy.
   Introduce a shared typed execution-failure record and preserve the historical
   fact/current-eligibility distinction during invalidation.
2. Make one numerical transition/health protocol serve candidate execution and
   retained sequential chunks, with explicit topology and telemetry capability.
   Keep exact-score and conditional proposal-field authority distinct.
3. Extract active preparation configs and numerical primitives from the historical
   module behind compatibility imports. Make the facade translate each legacy
   option exactly once; document or reject retired overrides. Do not move 30,000
   lines wholesale or rename public APIs without a migration boundary.
4. Consolidate evidence storage, checkpoint indexing, loading and member export
   as described in R7. Preserve ordinary checksum-based reproducibility.
5. Update the guide examples and decision table with the same change, and test
   the actual public examples rather than only imports or expected strings.

The repair tests should cover combinations, not only isolated branches:
verified repair then shared invalidity; candidate-domain exception beside a
valid peer; released reservations beside permanent budget exhaustion; repeated
interruptions during measurement and verification; scalar/batched targets with
and without telemetry; exact and proposal-field routes; and full derived seed
inventories before and after export/reload. A parameterized lifecycle test that
round-trips every reachable terminal/interrupted state would materially improve
coverage. A storage test should assert bounded repeated reads without declaring
a new scientific runtime threshold.

## Evidence, limits and terminal review

Artifacts are under
`docs/plans/artifacts/hmc-tuning-followup-review-2026-09-15/`:

- `controller_checks.py` / `controller-checks.json`: reservation and shared-invalidity counterexamples.
- `boundary_checks.py` / `boundary-checks.json`: two actual numerical tuning-to-posterior checks, classified native-failure injection, guide invocation, small member replay and seed arithmetic.
- `replay_checks.py` / `replay-checks.json`: unchanged complete automatic-search replay and counted evidence traversal.
- `acceptance_check.py` / `acceptance-check.json`: synthetic interpretation of the acceptance predicate.
- `run-manifest.json`: commands, baseline, environment, timing, seeds, source hashes and preservation checks.

The new numerical/diagnostic processes used deliberately hidden GPUs and the
existing `tfgpu` interpreter. Their recorded harness wall time totals about 20.3
seconds, including imports and the full-history replay, within the 15 CPU-minute
review ceiling; interpreter teardown and source inspection are excluded.
No new GPU or model-scale run was attempted. The earlier 322 regression tests,
two complete automatic-search tests and CPU/XLA smoke remain prior evidence;
they were not rerun or treated as proof against these counterexamples.

The source audit traced active facade dispatch, preparation translation,
candidate scheduling, evidence decisions, numerical health, checkpoint writing
and loading, retained replay, the posterior bridge, the active NeuTra consumer,
capability registry, reference, book tuning/mass chapters and their examples.
Historical helpers were checked for their authority boundary and active
dependencies. This is not a fresh proof of every legacy numerical primitive,
arbitrary user target, CUDA kernel, transport codec or posterior diagnostic.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain candidate-set unification | Shared public lifecycle and all-member retention confirmed on inspected paths | No finding requires abandoning this design | Untested targets and larger searches | Repair boundary defects within this design | Universal tuning success |
| Reject a claim of full internal consistency | Five implementation counterexamples reproduced | Library behavior fails its stated boundary/persistence/seed rules | Native domain case used fault injection | Add focused regressions and repairs | Every existing artifact is invalid |
| Keep R-hat outside tuning | Active tuning uses reporting-only R-hat | No new R-hat tuning gate found | Posterior adequacy is a separate question | Preserve separate posterior assessment | Convergence from acceptance |
| Refactor in stages | Concrete coupling and repeated processing confirmed | Target-scale performance remains unqualified | No large memory/throughput benchmark | Correctness fixes, then transition/storage extraction | Speedup size or new default readiness |

| Inference status | Result |
| --- | --- |
| Hard veto screen | Engineering counterexamples support repair; both new Gaussian tuning members passed their configured mechanics screens. |
| Statistically supported ranking | None sought or established. |
| Descriptive-only differences | Tiny-run timings and prior Gaussian candidate counts describe these fixtures only. |
| Default-readiness | Not established; no change to numerical defaults is proposed by this review. |
| Next evidence needed | Regressions for each repaired interaction, then bounded GPU and target-scale validation where relevant. |

Terminal skeptical review: the strongest alternative explanation for the new
failures is that they concern deliberately constrained fixtures rather than
common production inputs. That limits prevalence claims, but does not remove
the defects: scalar targets, optional telemetry, classified domain failures,
late shared invalidity and released budget are all representable in the current
public contracts. The injected exception establishes control flow only, and
seed equality across different runner topologies does not prove equal random
draws. The scalar/telemetry failures and posterior-phase seed overlap have
independent direct source explanations. Passing the corresponding new tests
after repair would overturn these implementation findings; it would still not
prove posterior validity or universal finite-search coverage.
