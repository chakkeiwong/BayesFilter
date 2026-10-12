# D4R4c adaptive comparison consumer repair

Date: 12 October 2026. This is the executable engineering dependency between
the [window ladder](bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md) and
the full-model comparison in the [master](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md).
The structural 1024/256 diagnostic is terminal with audited partial evidence.
The consumer and attainability repairs pass 59 focused tests; the corrected
GPU deployment has a 5580-second allocation. The current budget is in
`preconfirmation-01/settlement-01.json`; the final section records the next
command and its limits. No numerical sampler or tuning-admission default
changes here. Earlier execution entries below preserve their dated context.

## Question, baseline and scope

Can the comparison consumer faithfully execute a frozen proposal cohort with
steps above one and duration-preserving repair descendants while retaining
every member that the public tuner verifies? The baseline is the already
tested `run_adapted_mams_search` public-tuner/export/reload path. The older
`run_modern_hmc_method_fit.py` is historical D4b execution: its exact-count
assertion, domain `[.001,1]`, disabled repairs and old proposal/reference
loaders cannot support the new cohort. Do not reinterpret its old artifacts.

Factor the existing public dispatch, export and reload into one reusable
frozen-cohort consumer. MAMS continues to obtain its proposals from the shared
discarded pilot, then calls this consumer. Fixed HMC, randomized HMC and
SNAPER proposals can use the same frozen-cohort consumer with their own
explicit binding and search configuration. This is not a third tuner.

The consumer must validate the binding's actual epsilon domain, acceptance
policy and candidate allowance before creating an output directory. Reserve
the declared repair opportunities when computing the candidate/error cap.
Do not impose the old epsilon upper bound or assume that proposal and terminal
candidate counts agree. Do not use posterior diagnostics or short-path prices
to change membership. A rejected parent and its verified child both remain
in the result; a verified child must have a valid recorded parent chain.

Record requested primary pairs, uncreated primary pairs, every created
candidate's identity, parent, family, numerical parameters and disposition,
all repair actions, and every verified export. Primary membership is checked
by pair only for roots. Descendants keep identity even if a different family
has the same numerical pair. If the public tuner pauses or exhausts its
budget, preserve its actual inventory instead of fabricating passes or
requiring a completed cohort. Fresh public verification and checked replay
remain mandatory for every export.

## Evidence contract and implementation order

| Item | Role and decision |
| --- | --- |
| Real public tuning, export/reload and continuation on ordinary and fixed-transport bindings | Engineering pass criterion |
| Wrong epsilon domain, policy mismatch, unfunded candidate cap, unrequested roots, missing/invalid repair ancestry, export-set mismatch | Fail before dependent use; a shared defect stops affected execution |
| Numerical candidate rejection or incomplete search | Recorded candidate outcome; not a direction veto |
| Posterior reference, R-hat, ESS, MCSE, time and acceptance means | No consumer-membership authority; this engineering phase makes no method ranking |
| Unique source and outputs, unchanged shared tuning authority | Reproducibility requirement |

1. Implement the common frozen consumer and use it from the existing MAMS
   consumer, preserving the existing pilot/search directory layout and fields.
2. Test epsilon above one on actual HMC and MAMS, both coordinate routes,
   actual repair descendants, paused/uncreated inventories, deliberate
   missing exports and wrong ancestry. Reuse existing LGSSM and nonlinear
   accumulation integration checks to guard MAMS wiring.
3. Run the focused CPU reference/mechanics suite with GPUs deliberately
   hidden. Use XLA where the test requests it; graph-only fixture checks are
   explicit CPU mechanics exceptions. No GPU performance claim follows.
4. Commit the source and refresh the master. Only then build the full-model
   consumer on this tested foundation. Its fresh reference bridge must bind
   the final committed target closure; the old report must not be asserted
   current after source changes.
5. Before a posterior comparison, freeze and price a complete design for
   adequately searched fixed/randomized HMC, SNAPER and adaptive MAMS, plus
   the old MAMS grid as a causal diagnostic where funded. Account for every
   pilot, candidate, repair, failed trial and posterior member. Use fresh
   paired seed replications, model/tail quantities and uncertainty, and keep
   unfunded cells visible if the full design exceeds the remaining grant.

The pass criterion for this phase is correct routing and complete inventories,
not sampler convergence, a robustness gain, nominal MCSE coverage or release.
There is no new numerical threshold. Test epsilons, trial counts and seeds are
mechanics fixture inputs, inherited where possible; they are not scientific
defaults. Local tests use the existing 3600-second diagnostic reserve, whose
known use before this phase is 792.834369 seconds. Keep wall time and output
under `N5/generic-adaptation/preconfirmation-01/consumer-checks-01/` and settle
it once. Stop or reallocate within the grant before exhausting the reserve.

## Skeptical pre-execution review

The source of the problem is the consumer's old fixed cohort assumptions,
not evidence that the shared controller loses members. The chosen factorization
reuses that controller rather than reproducing its acceptance or repair logic.
The audit checks unfair comparisons: a successful adapted MAMS cohort alone
cannot establish improvement against unsearched classical controls. It checks
proxy metrics: short prices and pilot availability have no qualification
authority. It checks missing outcomes: repairs, paused roots and failures
remain explicit. It checks environment mismatch: the active GPU run retains
its immutable source; CPU mechanics do not become GPU evidence. Existing
reference compatibility is not silently transferred to new source.

Review verdict: proceed with steps 1–3 under the existing local reserve;
steps 4–5 remain necessary before launching claim-bearing comparisons.
The smallest misleading success would be exporting only the old proposal
count after a child verifies. The deliberate descendant and missing-export
regressions directly test that failure. A consumer test cannot establish
statistical calibration; the antithetic MCSE and stopped-chain reference
questions remain separate open work.

## Execution result

The common frozen consumer is implemented in
`bayesfilter/testing/inference_validation/modern_tuning_consumer.py` and the
existing MAMS consumer calls it. It preserves the existing pilot, tuning and
member directory locations; moving completed checkpoints was rejected in
review because those checkpoints contain absolute paths. Existing outputs
cannot be overwritten. All actual candidates, their parent/root identities,
repair actions, uncreated primary pairs and verified exports are recorded.

Twenty-four distinct regressions passed: four above-one HMC/MAMS tests across
both public coordinate routes; a real acceptance rejection followed by a
duration-preserving child, export and replay; fixed HMC; budget stop;
incomplete inventories; five ancestry/identity mutations; four configuration
failures before numerical work; overwrite prevention; missing-export detection;
and the existing ordinary/transport, LGSSM and nonlinear accumulation consumer
checks. The initial 22-test suite took 172.55 pytest seconds; conservatively
charge 180 seconds including startup. A follow-up had one incorrect test
expectation (`budget_bound` instead of the public `partial_budget` completion
status); after correcting only the assertion, the resource-stop test passed.
The two follow-up commands took 15.224024 and 5.304087 seconds. Total charge is
200.528111 seconds and known local usage is 993.362480 of the reserved 3600
seconds. The [test record](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/consumer-checks-01/result.json)
preserves commands, source hashes, environment and outcomes.

This closes steps 1–3 at the working-source level. The full-model executable
and fair, priced comparison design remain steps 4–5. The old fit driver has
not been silently repurposed. No posterior or statistical calibration claim
follows from these tests. The strongest remaining alternative is that a
correctly preserved adaptive cohort still explores only a local region;
independent posterior references and fresh cross-model comparisons must
answer that question.

## Full-model consumer implementation review

Step 4 will use an explicit frozen design file, with a typed transition,
controller and numerical execution configuration, geometry path, epsilon
domain, repair allowance, seed and optional posterior settings. Reconstruct
the existing prepared target and mass with those declared settings; use the
common frozen consumer for all methods. The design labels fixed HMC,
randomized HMC, SNAPER-proposed randomized HMC and MAMS, and verifies that
the label agrees with the executed transition. A label alone cannot establish
that a SNAPER/MAMS adaptation was performed; preserve its proposal provenance
and make no adaptation claim from an arbitrary frozen table.

The executable must validate the exact reference bridge before dependent
posterior work and assess every verified identity, including descendants.
Posterior settings and the finite-target agreement screen remain explicit
inputs. The inherited D4b screen, `(absolute error + 3 MCSE + reference
sensitivity)/reference SD <= .10`, may be used only when explicitly chosen
and labeled an operational screen without stopped-chain coverage guarantees.
Missing/nonpositive uncertainty cannot pass. Qualification-only operation
has no posterior authority and does not need a posterior reference. Resource
stops preserve public checkpoints and the complete result inventory.

Review before implementation: pass. This fixes the concrete consumer gap,
keeps the verified set intact, makes every numerical setting inspectable and
does not require changing target mathematics or the old archived loaders.
The final committed source still needs its reference audit and a priced
comparison allocation before serious posterior execution. Validate shared
binding construction, method labels, all-member posterior dispatch and
unavailable-reference outcomes with focused CPU mechanics tests.

The implementation audit found three issues before serious execution: imports
from `scripts` are not recursively included by the existing library-source
scanner; supervisor interruption can leave a stale running manifest; and
reference controls could otherwise be validated only after qualification.
Include the actual script helpers explicitly, validate posterior and reference
settings before target preparation, and preserve completed dispositions while
finalizing interrupted work. The reference screen must use the design's
declared precision estimator settings, rather than independently inheriting a
second batch-size or lugsail configuration. Exercise these failure paths and
one real full-horizon CPU CLI qualification smoke. This smoke tests routing and
serialization only; its small cohort, broad floor and short trials cannot
support a scientific admission, performance or posterior claim.

CPU reference execution may use XLA or the explicit graph-debug exception;
device choice and compilation choice are distinct. GPU campaign execution
still requires XLA. The first working-tree CPU CLI attempt was stopped after a
driver edit invalidated its source identity; preserve its 200.760842-second
charge and incomplete checkpoint. Repeat from a committed immutable source,
using CPU/XLA for the bounded full-horizon check. This is a local execution
repair, not evidence against a sampler or reason to relax qualification.

## Full-model consumer checks, 12 October

The explicit-design driver is implemented in
`scripts/run_modern_hmc_frozen_full_model.py`, with preparation and all-member
assessment in `modern_full_model_consumer.py`. Eighteen distinct regressions
pass: four actual full-data target preparations (PP-SGQF, PP-UKF, SIR and
structural), method-label rejection, shifted/constant reference controls,
explicit precision settings, peer continuation after a posterior veto,
resource stops, reference/member mismatch, interruption preservation of
completed members and repair descendants, source-helper inventory, seed and
policy preservation, and CPU/GPU compilation-policy separation. Posterior
dispatch tests are mocked engineering controls, not full-model posterior
results. The earlier common consumer's real public tuning and replay tests
remain the numerical routing authority for these checks.

The driver records the compiled operator, source helpers, exact design and
geometry inputs. It validates reference settings before qualification, keeps
missing uncertainty inconclusive, and finalizes supervisor stops without
discarding completed outcomes or uncreated roots. No numerical tuning default
or posterior threshold changed. The historical D4b driver remains unchanged.

Known local use is 1537.897999 seconds of the 3600-second reserve, including
the interrupted CLI attempt, all follow-ups and the 58.109641-second earlier
committed-source consumer check. Detailed commands and measured times are in
`preconfirmation-01/consumer-checks-01/`. The next action is a scoped commit,
immutable-source CLI check and fresh reference compatibility audit. The final
full comparison still needs its priced classical search and paired design;
the new driver alone does not establish adequate tuning, posterior delivery,
MCSE coverage, a method ranking or release readiness.

The committed `935e9d719` deployment has now passed reference compatibility on
all four full-data targets (169.225259 CPU seconds). Its first SIR CPU/XLA
qualification smoke reached the 300-second cap after returning one 64-draw
trial and starting a second. The returned call cost 119.482305 seconds including
its first compilation. The terminal supervisor correctly preserved the
candidate, partial evidence and incomplete qualification; this is not a
sampler failure. Do not extrapolate this CPU check into a GPU timeout policy.
Use the full-data PP-SGQF target for the next at-most-300-second CLI mechanics
check from the same immutable source. The prior four-target preparation tests
and existing public-route controls remain intact; a successful SGQF check
will not be reported as a completed SIR qualification. All counts and the
broad acceptance floor stay labeled mechanics fixtures. Local use before this
retry is 2007.425158 seconds, leaving 1592.574842 in the existing reserve.

The PP-SGQF CPU/XLA CLI attempt also exhausted its 300-second diagnostic
allocation, during public-binding validation before any tuning checkpoint.
Preserve it as an incomplete deployment check (300.150605 seconds), not a
failed candidate. These CPU allocations were convenience smoke bounds and
are demonstrably insufficient for the complete full-target consumer. Stop
repeating those short CPU launches. The code path repeatedly validates the
actual target at its fixed start bank; that validation is numerical work and
cannot be mistaken for a dead worker merely because no sampling checkpoint
has been written.

Use one bounded GPU/XLA PP-SGQF qualification-only deployment check from
`full-model-source-935e9d719`, with the exact eight-trial mechanics fixture,
same data and local factor, a fresh seed and fresh output. Reserve 1800
aggregate worker-seconds from the unallocated grant; this leaves 134275.474013
seconds unallocated while the existing reserves remain held. The 1800-second
cap is an engineering allocation hypothesis, not a predicted runtime or
deadlock threshold. This tests whether the canonical GPU backend can complete
the new CLI's preparation, public qualification and export/reload. It does not
test posterior convergence or make the .1 fixture floor a scientific default.
Use an idle permitted RTX 4080 SUPER if available; shared admission is allowed
under the already-used trusted memory-growth/headroom policy, with contention
recorded and no cross-device performance comparison. Do not interrupt another
worker. Stop at allocation exhaustion or an implementation/source/numerical
failure, preserving every candidate and partial trial. No automatic second
1800-second attempt is allocated by this note.

Skeptical retry audit: the full-data target and numerical checks are unchanged;
CPU and GPU outcomes will retain distinct execution identities. A missing
qualification or export remains incomplete. This launch answers deployment on
the repository's required backend, unlike a third short CPU retry. It does not
pretend to resolve the separate expensive-validation question; a shared
validation/profiling repair remains a follow-up if measured GPU cost is also
unacceptable. All four references are already compatible with this committed
source, although this qualification-only check does not consume posterior
reference outcomes. Known local usage is now 2307.575763 seconds.

The structural rung is terminal at 7200.434712 supervisor seconds. Its archived
supervisor finalized the result but left the worker manifest at `running`.
Repair future price-supervisor finalization and let the read-only terminal
auditor explicitly reconcile only this demonstrated deadline case (nonzero
exit, exhausted recorded cap and matching terminal result), recording the old
status rather than rewriting the old evidence. Audit completed pilot tensor
bundles as well as price bundles before using the saved partial diagnostics.
Deliberate status and checksum mutations must fail. This is a reporting and
provenance repair; the immutable worker source and pilot data stay untouched.

The parallel guide update documents the implemented full-member consumer in
the existing tuning chapter and incorporates the checked finite-AR(1)
antithetic counterexample in the existing modern-methods chapter. It creates
no second guide. Build these two chapters with the repository preamble and
bibliography and inspect the rendered edited pages. The exact AR(1) arithmetic
is already recorded in the analytic-control result; no new MCSE policy or
unverified literature claim is introduced. D4R5's final release-wide consistency
audit remains open pending the comparison and calibration results.

## Terminal review and attainable-evidence repair

Both GPU workers are terminal. The structural allocation spent 7200.434712
seconds. Sixteen complete pilot calls survive with 95160 actual and 196544
reserved gradient-work units; the seventeenth call did not return. Both first-
duration 1024-draw windows produced usable source IACT estimates (19.4152 and
20.3532), and all thirteen observables had usable estimates. Heavy shared-device
contention explains much of the second window's elapsed time. The run did not
finalize a proposal cohort or prices. Its read-only terminal audit checked 177
source hashes, two geometry inputs, the compiled operator and all 384 tensor
receipts in the sixteen completed bundles. The archived running manifest is
explicitly reconciled with the killed worker's terminal result, not rewritten.
Ten mutation and supervisor tests pass for the reporting repair.

The full-data PP-SGQF GPU deployment used immutable `935e9d719`, verified memory
growth and XLA, and spent 929.206570 supervisor seconds. All eight independent
measurement trials returned healthy evidence, with mean acceptance 0.99997038.
The public controller correctly returned `inconclusive_at_cap`: simultaneous
start intervals were [0,1]. It performed no verification and exported no member.
R-hat is reporting-only and did not cause this disposition. Reducing the
inherited sixteen-trial mechanics fixture to eight was a planning error: at
measurement alpha .0025, even eight perfect trial scores cannot cross the
implemented betting threshold. This is not evidence against MAMS or the target.

Before another numerical launch, add a model-independent attainability check
to the frozen-cohort research consumer and full-model driver. It must use the
actual interval implementation, scheduled evidence rungs, full statistical
candidate cap (including reserved repairs), and both measurement and fresh
verification error allocations. Scores equal to one give an optimistic upper
bound on every attainable lower confidence endpoint: the betting capital and
Hoeffding lower endpoint are monotone in each score. Scores equal to zero give
the corresponding optimistic upper-endpoint bound. Thus a failure even at the
relevant extreme proves that this design cannot qualify any candidate. Passing
this necessary check does not establish power, convergence, affordability or,
for a two-sided rule, simultaneous attainability of both endpoints.

The preflight records those limits and rejects proven-impossible designs before
full-target preparation. It does not modify the public tuners or their
qualification law; intentional inconclusive public-tuner tests remain valid.
Use TensorFlow's existing statistic, including its rounding margins and
Hoeffding allocation across looks, rather than a second approximate formula.
Exercise the failed eight-trial design, an attainable sixteen-trial design,
stricter verification allocation, unused repetition caps, larger candidate
caps, both interval methods and two-sided limits. Include a consumer call-chain
test proving that impossible designs stop before target preparation.

Skeptical pre-execution audit: the previous deployment could never answer its
export/replay question, despite correct execution. This revision repairs that
specific flaw without lowering the floor or error standard. Extreme synthetic
scores test mathematical attainability only; observed trial scores are not
reused as verification. A two-sided necessary bound must not be labelled a
proof of sufficient sample size. Scope, target, methods and total campaign
budget are unchanged. Proceed with focused CPU diagnostics, GPUs hidden, under
the existing local reserve; price and allocate any GPU retry explicitly after
the check passes. Preserve all failed attempts and source identities.

The two-chapter guide extract compiled to 81 pages in `guide-02`; the edited
pages were rendered and inspected. `guide-01` records the failed build with a
missing copied examples directory. Neither is a full-monograph release audit.
Known local usage is 2313.645604 of 3600 seconds before this continuation's
checks. Settling both GPU workers leaves 135145.832731 unallocated worker-
seconds, with the local reserve still held and no active GPU reservation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Structural partial diagnostic retained | Completed windows audited; cohort incomplete | No invalid completed bundles | Contended cost and second-duration estimates | Use completed diagnostics in pricing; retain omitted rows | Full structural adaptation or posterior validity |
| Deployment incomplete for export/replay | Qualification mathematically impossible at eight trials | Design cannot answer intended check | Cost of a feasible full execution | Add attainability preflight, then explicitly allocate a corrected check | MAMS failure or release readiness |
| Reporting and guide repairs implemented | Ten reporting tests and chapter build pass | No new numerical default | Full comparison and calibration remain open | Commit scoped repairs and continue | Full-book or statistical qualification |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Completed deployment evidence was healthy; the trial budget prevented qualification. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Acceptance, window IACTs and elapsed times. |
| Default-readiness | Not established. |
| Next evidence needed | Feasible fresh public verification/export, priced paired classical/MAMS comparisons, MCSE/stopping calibration. |

## Repair verification and corrected deployment allocation

The focused suite passed all 59 tests in 328.318985 wall seconds, covering the
new preflight, common consumer, four full-target preparations and terminal
reporting. The preflight leaves intentional inconclusive public-tuner behavior
intact and stops the full-model driver before target preparation. The archived
GPU deployment passes a separate CPU read-only audit of 190 source/input
hashes, checkpoint/evidence checksums, all eight independent seeds, native
traces and 8192 charged work units. This byte/work audit does not reconstruct
a GPU runtime or grant replay authority. Both the failed eight-trial and
proposed sixteen-trial designs were checked against the actual statistic.
The latter passes the necessary attainability bound in both stages.

The existing guide chapters now explain the guard and its limits. `guide-03`
compiled successfully in 4.968896 seconds and the edited page was rendered and
inspected. The exact antithetic derivation remains intact. The full monograph
and final release review remain separate unfinished work.

Allocate **5580 GPU supervisor seconds** for one corrected full-data PP-SGQF
deployment, `consumer-checks-01/gpu-deployment-02/`, from a fresh committed
immutable source. This is the previous 929.206570-second wall cost multiplied
by four (32 calls instead of eight), with an explicit 50% shared-load allowance,
rounded up to minutes. Linear scaling overcounts fixed preparation/compilation
but cannot bound future contention; the cap is a funded allocation, not a
deadlock rule or runtime guarantee. No automatic further attempt is funded.

The design restores the inherited sixteen-trial mechanics fixture at each
stage. It keeps the same .05 family error allocations, .1 mechanics floor,
64 transitions/trial, epsilon .15, maximum count one, full observation horizon,
warm-start geometry and target. These are explicitly engineering fixture
inputs, not a tuning recommendation or a method-comparison design. Fresh
target/tuning seed blocks 61310/61510 replace 61300/61500; all measurement and
verification streams are derived by the public controller. No old trial is
reused or reclassified. Required outcome: complete public measurement, fresh
verification and checked export/reload, with every disposition preserved.
Failure is reported by layer; an infeasible or healthy-but-inconclusive
candidate does not reject the research direction.

Launch from the immutable archive with trusted GPU access and verified memory
growth, using an idle permitted RTX 4080 SUPER if available or the previously
reviewed shared-headroom admission otherwise. Preserve actual device and load.
The executable command is `/home/ubuntu/anaconda3/envs/tfgpu/bin/python
scripts/run_modern_hmc_frozen_full_model.py --design <absolute deployment-02/design.json>
--output <absolute deployment-02/run> --max-wall-seconds 5580`, with
`CUDA_VISIBLE_DEVICES` set to the admitted GPU and `TF_FORCE_GPU_ALLOW_GROWTH=true`.
Record the exact resolved command and source commit in its manifest.

Skeptical retry audit: pass after correcting the impossible trial count. This
allocation answers deployment only; the fixture's high acceptance and short
travel cannot answer the original long-chain mixing question. It must not be
substituted for adequately searched classical controls or model/tail posterior
comparisons. Stop on source drift, invalid numerical evidence or exhausted
allocation; monitor progress and contention without killing a healthy worker
early. Reference compatibility for a later posterior comparison must bind the
new source. The old four-target compatibility result remains valid only for
its archived source. The next master refresh must settle this run, then freeze
the affordable paired design and MCSE/stopping calibration scope.

`settlement-01.json` records 2659.466401 known local seconds of the held 3600,
including a conservative ten-second rendering/inspection charge, and
129565.832731 unallocated worker-seconds after this new GPU reservation.

The repair is committed as `e68cdf1c4`; its isolated source archive passed
55 selected regressions in 78.392807 seconds (the four expensive full-target
preparations were already in the 59-test suite). The corrected GPU run began
at 08:45:34 UTC on host GPU 0, using the existing trusted shared-device
admission with over 27 GiB free at the probe. Its manifest verifies memory
growth and XLA; both attainability stages pass. Device 0 and 1 were busy, so
no idle-device performance inference is allowed. Tool session: 31349.

In parallel, repeat the existing four-target CPU reference-compatibility audit
against this exact committed source in `reference-consumer-03/`, under the
unchanged B0 contract. This is needed for dependent posterior work, not for
tuning membership. Use the same audit driver and original reference archives,
hide GPUs, preserve exact commands and cap this local diagnostic at 420
seconds within the existing reserve. The previous check cost 169.225259
seconds; the larger allowance accommodates CPU load and is not an accuracy
threshold. Known local use before this check is 2737.859207 seconds. Any
changed integrand/quantity/source finding stops the affected posterior
comparison, while the independent qualification-only deployment can continue.
