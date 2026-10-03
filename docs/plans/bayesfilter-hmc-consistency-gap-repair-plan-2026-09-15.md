# HMC consistency gap repair

Baseline: `8275b497` (runtime baseline `19ca6973`). The user authorized planning,
skeptical review, and execution to close F1–F10 in
[the post-unification review](bayesfilter-hmc-tuning-post-unification-review-2026-09-15.md).
Preserve all unrelated dirty files and all prior diagnostic results.

## Objective and evidence contract

Make the shared candidate-set procedure obey its stated search, evidence,
budget, replay, and configuration semantics through every supported facade.
Retain every verified pair, preserve failed and inconclusive evidence, and keep
R-hat reporting-only in tuning. Exact-score and position-field authority remain
different. This is an engineering repair, not a sampler ranking campaign.

The comparator is the committed implementation and its reproduced F1–F10
counterexamples. Passing requires behavioral regressions for those failures,
appropriate existing tests, complete automatic tiny-target checks, consistent
reference/registry/book text, and a successful document build. Numerical health,
scope identity, fresh verification, replay integrity, and GPU memory policy are
vetoes where applicable. Runtime, acceptance values, R-hat, and candidate counts
are explanatory beyond their existing declared tuning roles. No posterior
convergence, general success rate, speed superiority, or target-scale replay
performance will be concluded.

## Implementation sequence

1. **F1: remember directional evidence.** Extract a pure proposal helper that
   uses the latest valid observation for each exact pair. At a given L,
   prioritize an unvisited geometric interior between the current pair and a
   nearby opposing directional observation. Persist evidence through existing
   receipts; reconstruct it on resume. Geometric interiors are measured
   hypotheses, not a global monotonicity assumption. Keep the existing family
   repair cap. During the already declared finite refinement rounds, also
   investigate unresolved opposing-direction intervals, including when no
   initial family survived. Preserve bounds, deduplication and candidate caps.
2. **F2/F3: one typed decision vocabulary.** Share acceptance/validity/veto/repair
   translation between adapters, receipts, controller and artifact validation.
   Preserve a finite divergent parent's promotion veto while permitting an
   otherwise supported smaller-step child. A rejected receipt must round-trip;
   a verified member still needs its own passing veto-free fresh verification.
   Apply semantic validation at writes as well as reads.
3. **F4/F6: shared chunk mechanics.** Centralize chunk seeds and their inventory,
   deadline checks and accounting hooks. Numerical adapters quote remaining
   stage work for scheduling, but charge each attempted native chunk before its
   call and checkpoint that charge. Saved completed chunks are not charged
   again on resume. A failed native call remains conservatively charged;
   invocation-count budgets also charge retries. Record that distinction and
   test both failure between chunks and failure inside a charged native call.
   Retained block, sequential and archive checks use the full tuning seed
   inventory, including partial chunks.
4. **F5: finish affordable mandatory work.** Explicitly defer work that cannot
   fit the remaining numerical/call budget; preserve its pending status and
   reservation. Close its cohort as budget-deferred and dispatch other funded
   mandatory stages. Do not reduce evidence requirements or steal a peer's
   reservation. Stop as partial when nothing further can fit. Persist enough
   state to make restart deterministic and prevent busy loops.
5. **F7: scope/search lookup.** Resolve members by scope plus candidate ID across
   all matching searches, with an optional explicit search ID. Collection
   completeness requires every included search to be complete and every
   explicitly expected scope to be present. It is independent of tuple order.
6. **F8/F9: preflight and preparation.** Validate config types, conflicting
   overrides, all L grids, evidence counts and execution compatibility before
   numerical preparation. A bound adapter rejects redundant provenance/payload
   options. All supplied grids obey the preparation config's explicit L cap.
   Add a shared preparation progress/deadline recorder; check between supported
   preparation phases/windows and preserve failures. Native calls require an
   external timeout for hard preemption. Preparation failures are retried in
   fresh output directories; numerical resume begins only after a frozen scope
   exists. Extract the active ordinary preparation orchestration from the
   historical monolith behind its existing import compatibility boundary.
   Give the position-field pilot its declared `step_adaptation_results` budget
   as a fixed-pair pilot allocation (no adaptation-derived admission), document
   the translation, and use the shared execution config for explicit stage
   counts. Position-field XLA defaults on; non-XLA use needs an explicit debug
   reason on the active facade. Historical readers remain readable.
7. **F10: guide and registry.** Correct the registry's position-field search
   description and generate both interface tables. Update the reference and
   book with repaired evidence roles, proposals, deferral, chunk accounting,
   preparation limits, option semantics, and status interpretation. Test actual
   route behavior as well as generated-text consistency. Build the guide and
   inspect changed rendered pages.
8. Audit the execution against every finding. Record exact tests, commands,
   environment, timing, source state and outcomes in an execution note. Commit
   and push only task files, following the user's continuing commit/push
   authorization.

The refactor is limited to shared decision/proposal/chunk/preparation boundaries
needed by these repairs. Remaining historical helper extraction and large-scale
evidence-store redesign are separate maintenance work; this plan will not claim
those performance uncertainties are closed by small tests.

## Assumptions, defaults, and test budget

| Choice | Provenance / status | Reason and failure check |
| --- | --- | --- |
| Acceptance policy, four chains, evidence ladder, L grid, family cap | Existing API policy; unchanged baseline settings. | Isolate correctness from relaxed thresholds. Test near-band, veto and inconclusive outcomes separately. |
| Opposing-observation interiors and unresolved-interval refinement | Derived repair for F1; proposal hypothesis. | Retain nonmonotone evidence and require measurement/verification. Test reversal, noisy/nonmonotone fixtures, bounds, restart, and caps. |
| Attempted chunk cost `(L+1)*chains*transitions` | Existing conservative work model applied at the native-call boundary. | Distinguish unknown failed-call work from unexecuted work; compare uninterrupted and interrupted histories. |
| Position-field pilot count translation | Existing required legacy count, now explicitly a fixed-pair proposal budget. | Validate minimum evidence and prove the configured count actually controls pilot calls; retain explicit stage config override with conflict checks. |
| XLA default | Owner's existing TensorFlow/XLA policy. | Test defaults, explicit debug exceptions, and bounded trusted GPU compatibility. No automatic target qualification. |
| Gaussian and anisotropic fixtures, existing seeds | Existing small test targets and seeds; engineering baselines. | Exercise automatic preparation/search to completion. Do not rank policies or interpret one seed as a reliability estimate. |
| 45 CPU minutes and 10 GPU minutes total; at most four GPU attempts | Convenience repair-validation ceilings, not scientific settings. | Focused tests before broader suites. Reuse only unchanged passing checks. Stop numerical campaigns at the ceiling and report remaining evidence honestly. |

Artifact root: `docs/plans/artifacts/hmc-consistency-gap-repair-2026-09-15/`.
Use distinct attempt directories and logs. CPU tests deliberately set
`CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true
BAYESFILTER_TEST_DEVICE_SCOPE=cpu` before framework imports and use
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. GPU checks require trusted access,
the installed readiness probe, memory growth before imports, and recorded
device/XLA/allocator provenance. No training or MacroFinance target campaign is
included. The final note will preserve exact commands as run.

## Skeptical audit before implementation

The initial tempting fixes have material flaws: merely raising three repairs
does not fix lost evidence; dropping divergence vetoes would weaken admission;
charging only successful chunks would hide failed native work; letting a costly
peer terminate the whole queue would retain F5; and accepting a `passed` receipt
without examining its separate vetoes would create false authority. The design
above rejects all five shortcuts.

The baseline is the actual pushed code and its failing histories, not the old
"closed" labels. Explicit-grid success alone cannot pass the automatic-search
requirement. Search completion and verified membership are separate assertions.
CPU/non-XLA mechanics tests do not substitute for GPU/XLA qualification. Legacy
payloads cannot be silently resumed under a changed controller policy; increment
the policy version and keep historical reads distinct from executable resume.
Fresh source identities imply fresh numerical bindings after code changes.

Preparation hard preemption and large-scale replay are not silently promised:
phase checks preserve progress, while an external process limit handles a stuck
native call. Historical extraction will preserve signatures and tests rather
than rewriting numerical routines. The weakest assumption is proposal coverage
on targets outside the tiny fixtures; no universal success claim is made.

Audit verdict: **proceed with this revised, bounded design**. Each finding has a
behavioral acceptance check; evidence roles, stop conditions, target/coordinate
boundaries, artifact locations and resource limits are explicit. Unexpected
validity failures trigger localized repair within this scope. A failed candidate
alone does not stop the implementation plan.

### Terminal-audit refinements before final validation

The first complete automatic Gaussian check passed. The broader suite also
exposed four older fixed-transport tests that still invoke the retired callback
route or require a multi-point initial grid. Check them against the committed
baseline, then preserve their historical coverage under the named historical
helper and test the active facade's actual rejection/translation rules.

F4 requires an export/reload regression with both a verified member and unfinished
peer work. Completed evidence alone loses the latter's seeds. Reconstruct all
attempted chunk seeds from the already durable work/accounting records, including
failed native calls; use that inventory in both live and reloaded retained
sampling. No additional approval or launch-token mechanism is needed.

The new explicit position-field warmup allocation must not hide nonfinite warmup
states, log acceptance, or energy errors. Check full numerical health before
discarding warmup from acceptance statistics, and preserve energy errors in the
observation. Add a compiled synthetic-trace regression and a mixed receipt
checkpoint round trip. Label legacy configuration's historical candidate-policy
metadata explicitly so it cannot contradict the active shared route.

These refinements preserve thresholds, authority boundaries, and the original
compute budget. Skeptical review: derivation from charged work survives export
without duplicating partial traces; finite warmup is a health veto, not a new
acceptance or R-hat screen. Proceed with these bounded completeness repairs.

The trusted readiness probe subsequently reported no idle permitted GPU.
While other checks run, prepare the same bounded XLA smoke for either the next
idle permitted GPU or an explicit CPU/XLA debugging exception. The latter can
check compilation and replay mechanics but cannot satisfy GPU compatibility or
claim eligibility; report that gap if trusted GPU availability remains blocked.
Use at most 180 seconds for this tiny smoke within the original compute ceiling.
Repeated-interruption coverage is also included; the existing controller already
reconstructs the active cohort correctly, so no speculative identity refactor is
needed unless that behavioral regression fails.
