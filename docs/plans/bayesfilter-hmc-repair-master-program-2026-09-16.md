# HMC repair master program

## Active release checkpoint: independent confirmation running, October 5

The frozen **96-search P4 confirmation is running** as
`bayesfilter-hmc-v7-confirmation-20261005-r1.service`, launched at
2026-10-04 16:51:41 UTC (October 5 locally). Release remains **pending**.
The [execution plan](artifacts/hmc-v7-release-2026-10-02/confirmation-execution-2026-10-05.md)
and [passed design audit](artifacts/hmc-v7-release-2026-10-02/confirmation-design-01/preflight.json)
bind 32 independent frozen seeds each for QR LGSSM, nonlinear SSM and the
residual-whitened funnel. All 1,811 source files and the tested supervisor match.
Numerical configurations, source-23, original cores, all-member retention,
reporting-only tuning R-hat and statistical criteria are unchanged.

A separate **one-minute health monitor is now running** as
`bayesfilter-hmc-v7-monitor-20261005-r1.service`. Its
[live report](artifacts/hmc-v7-release-2026-10-02/monitoring-01/status.md) is the
current health authority between chat turns. At 18:10 UTC on October 4
(October 5 locally), two QR searches and one nonlinear search had completed,
two other slots had deferred before sampling under transient GPU contention,
and the next residual-funnel search was progressing.
The observer checks source/configuration/seed identity, completed-member and
memory/XLA records, progress age, service exit and storage. Desktop completion
and error notifications were successfully requested; new chat messages cannot
be issued by this observer.

The [monitoring plan](artifacts/hmc-v7-release-2026-10-02/confirmation-monitor-plan-2026-10-05.md)
allows one separately reported same-seed retry of each untouched pre-import
resource deferral after the primary service ends and only within the original
remaining GPU reservation. Original primary failures and all 96 slots remain
authoritative for the release calculation. Other errors require diagnosis;
healthy work is not restarted. The monitor passed 27 focused tests, and 77
distinct monitor/supervisor checks pass across the recorded selections, plus a
live-input check. A transfer of 3,600 already-authorized CPU seconds funds this
work and terminal audit; no new compute grant or GPU transfer was added.

The service has a 189,063-second supervisor ceiling plus at most 10 seconds of
cleanup, reserved inside the existing release allocation. The 40.78-hour budget
scenario includes the unchanged 20% margin and readiness/reporting allowances;
the earlier 60.49-hour slower scenario remains a risk. No outside GPU transfer
or new CPU grant was assumed. Design preparation cost 0.366486 CPU seconds and
the device probe 0.103582 GPU seconds; both are settled once. The running
service is reserved and will be charged once at termination, including waits
and failures, without adding nested worker times.

The first QR worker passed resource admission, verified memory growth before
GPU initialization and wrote actual GPU/XLA trial records. This establishes
successful execution startup, not completed confirmation or release evidence.
The live [progress record](artifacts/hmc-v7-release-2026-10-02/confirmation-run-01/progress.json)
is written after each slot. Every failure, timeout and resource deferral remains
in the denominator. Next: complete P4, audit raw/retained evidence and the
simultaneous family delivery bounds, then complete P5 source/guide/regression
review and the explicit scoped support decision.

## Prelaunch October 5 checkpoint: confirmation funded

The goal is a **supported, explicitly selected v7 release**. Release remains
**pending**. The [release plan](bayesfilter-hmc-v7-release-plan-2026-10-02.md)
controls P0–P5; the [execution record](bayesfilter-hmc-v7-release-execution-2026-10-02.md)
and [current continuation audit](artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-05.md)
preserve every result and charge. Defaults, all-candidate retention and
reporting-only tuning R-hat are unchanged.

Both complete source-23 price vectors pass. The resource-observed repeat
finished in 3,657.258328 enclosing GPU seconds and retained every verified
member. An independent audit found exact same-seed agreement for all 31,072
raw trials, candidate inventories and states, with complete source, device,
XLA/memory-growth, export/reload, checkpoint and attempted-work accounting.
No competing compute process was observed on the selected GPU in 361 snapshots.

| Family | Candidates | Verified members | Valid trials | First allocated seconds | Repeated allocated seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| QR LGSSM | 42 | 22 | 11,968 | 2,377.167430 | 1,468.070101 |
| Nonlinear SSM | 45 | 19 | 10,816 | 2,227.336864 | 1,319.514584 |
| Residual-whitened funnel | 32 | 17 | 8,288 | 900.360195 | 869.673643 |

The full repeated vector projects **32.51 GPU hours** for all 96 searches.
The predeclared 20% reserve, 96 × 60-second readiness allowance and 600-second
terminal allowance increase that to **40.78 hours**. The distinct October 5
grant adds 24 GPU hours, bringing the remaining authorized balance to **55.10
hours**, including 9,278.015433 seconds outside the release allocation. The
40.78-hour scenario fits inside the release allocation alone; across all funds,
**14.32 hours** remain beyond that scenario. No new CPU grant is assumed. The
earlier slower vector and 0/10/20/30/50% sensitivities remain preserved; the
slower vector with the same allowances requires 60.49 hours and still exceeds
the balance. No family-wise minimum selection or smaller denominator is
permitted. These are descriptive forecasts, not runtime guarantees.
Confirmation is unfrozen and unlaunched.

The host-affinity diagnostic is complete. It compared the original cores 8–11
with quieter same-socket cores 32–35 in original/candidate/candidate/original
order. All 36 native calls and 1,152 full trial records matched exactly, but
record assembly was slower in both candidate-affinity repetitions for every
family. Native times overlapped. The proposed placement change is not adopted,
and another complete price is not justified by this result. The first launch
had deferred before import under contention; its cost and the successful
177.608444-second retry are both charged. All reservations are released and
no GPU run remains active.

Source-23 remains numerical authority; the predicate optimization and the
source-24–27 concurrent analyzer remain withdrawn. Current-source evidence
includes 201 distinct scoped posterior/endpoint, preparation/negative-control
and documentation checks. The optional pre-import process-aware confirmation
admission passed 154 distinct focused tests and is frozen separately from the
numerical source. These counts overlap earlier selections and are not additive.
Admission readiness cannot guarantee post-launch resource availability.

| Gate | Current state and next action | Required evidence |
| --- | --- | --- |
| P0/P1 implementation and recovery | Scoped source/compatibility checks pass | Frozen dependencies/build, current readers and both public routes |
| P2 full procedure and numerical validity | Two complete vectors and exact raw-trial audit pass | Unchanged grid, starts, data, horizons, repairs, all members and checkpoint |
| P3 calibration | Compatible trial/boundary evidence remains valid | Unchanged allocations and vetoes; no hardware-wide theorem |
| P4 reliability and cost | Repeated-vector scenario funded; confirmation design and execution pending | Freeze complete design and count all 96 independent slots; simultaneous one-sided 95% family bounds > .80 |
| P4 posterior interface | Original K0 and negative controls pass | Warmup excluded; separate posterior assessment; no SBC claim |
| P5 guide and release decision | Guide/registry/examples pass; decision pending P4 | All scoped requirements, explicit support scope and exclusions |

Next-phase refresh: freeze 32 independent root seeds per family, disjoint from
development, and bind the complete configurations, tested supervisor, exact
allocation and fresh output root. Recheck source, GPU and storage readiness,
then execute all 96 searches and audit their delivery bounds and retained
artifacts. The October 5 continuation audit supports design preparation; no
additional speculative optimization or repricing is required. Keep all three
families, all 96 slots, every numerical setting and the original margin rule.
The hard campaign budget remains binding under slower execution. P5 then
requires the final source/evidence assembly, scoped regression/documentation
checks and explicit local support decision. No release or default promotion is
claimed. The October 4 funding gap is historical and cleared by the new grant.

## Earlier October 4 checkpoint: source-22

The goal remains a **supported, explicitly selected v7 release**. Release is
**pending**. The [release plan](bayesfilter-hmc-v7-release-plan-2026-10-02.md)
controls P0--P5, and the [execution record](bayesfilter-hmc-v7-release-execution-2026-10-02.md)
preserves every attempt and cost. Earlier entries below are historical.
All verified candidates remain retained; R-hat stays reporting-only in tuning.
Default promotion and public publication remain separate decisions.

The source-21 two-family queue is **finished**. The original residual-whitened
funnel completed in 1,381.226 worker seconds with all 18 verified candidates
exported/reloaded and its checkpoint reconstructed. The nonlinear SSM reached
its 1,800-second search cap with ten verified members and nine unfinished
candidates; all ten members passed closeout, but the search remains partial.
Its enclosing worker time was 2,208.007 seconds. The enclosing queue cost
3,589.238 GPU seconds and was charged once. The source-19 QR price completed
with 22 verified members in 2,318.380 seconds. Different source versions and
partial results cannot be combined into a complete confirmation forecast.

A settled-artifact audit checked all 15,136 source-21 chunks, full issued work
identities, source/configuration/base seeds, GPU/XLA/memory growth, member
inventories and charges. No numerical invalidity or unaccounted attempted
work was observed. These are scoped development outcomes, not independent
confirmation or posterior convergence evidence.

The new work-metadata audit found that a chunk could omit required identity
fields or add an unknown null field. The reader now compares the complete
issued record, ignoring only mutable work status. The preserved reproduction
failed before the repair; 77 focused checks pass afterward. Source-22 passes
186 frozen-source focused checks, 346-test collection and
two actual SSM reference checks; source-21 and its evidence remain immutable.
The maintained integration selection now includes the four recent replay and
confirmation-report test files. Collection and inventory/reporting checks pass.

| Gate | Checked evidence and remaining gap | Required next action |
| --- | --- | --- |
| P0/P1 source and migration | Source-21 recovery/replay checks passed; complete work-metadata fix passes 77 worktree checks; source-22 passes 186 focused and two SSM reference checks | Preserve checked dependency/build manifests and carry only compatible evidence |
| P2 full procedure | Broad QR and supplied residual-funnel searches have complete scoped results; nonlinear delivery is partial | Resolve the measured cost problem before another full nonlinear/current-source price; retain every failed attempt |
| P2 numerical boundaries | CPU/XLA and trusted GPU/XLA high-precision envelopes, original-stream parity, corruption and recovery controls pass in their recorded scopes | Carry only compatible evidence forward; preserve exact arithmetic and all validity checks |
| P3 statistical calibration | Independent-trial calibration and adversarial controls pass in their frozen scope | Preserve error allocations and reporting/admission separation; no universal guarantee is inferred |
| P4 reliability and cost | Full independent confirmation is unfrozen; current prices do not establish an affordable design | Measure the remaining work-item overhead, adopt a justified bounded repair if available, then obtain a complete source-consistent forecast and review replication power |
| P4 posterior interface | Original three K0 panels/seeds and missed-mode/endpoint negative controls pass | Tuning admission cannot bypass sequential warmup, precision or posterior-reference assessment |
| P5 guide and release decision | Agent reference/official chapter agree; documentation checks and book build passed | Once required gates pass, audit the final source/evidence matrix and publish a scoped local support decision |

The cost investigation has narrowed the nonlinear gap. Of its 1,750.990-second
search, recorded native batch-call shares total 224.074 seconds, and native
calls plus tensor serialization total 255.535. Outcome generation/application
spans total 1,718.556 seconds, leaving only 32.434 outside work-item calls.
The remaining 1,463.021 seconds inside work items includes analysis, inner
checkpoints and other uninstrumented work; no single cause accounts for all of it.
A source-21 saved-trial reconstruction passed exact equality for 32 trials.
Its cold instrumented timing is a localization measurement, not a full-price
forecast. Separate saved-summary/R-hat timings passed exact output comparison;
warm
calls around .02--.03 seconds do not support R-hat as the main cause. Two
fresh public first-work runs passed exact raw-output parity. Their profile
identified binding validation as a possible cost; direct measurement confirmed
8.954--9.529 seconds in its current-target probe per validation. The next
bounded repair hypothesis is native batch evaluation of the same start bank,
conditional on capability, exact scalar/batch parity and unchanged target/
geometry-mutation detection. No target-probe optimization is adopted yet.
Shared GPU load was observed but its causal delay is unquantified.

The inherited release criterion remains simultaneous one-sided 95% lower
bounds above 0.80 across three families. Even an all-success design requires
at least **19 searches per family**; that is a mathematical minimum, not the
chosen design. The proposed **32 per family** allows one failure each. Under
independent searches with an assumed true delivery probability of .95, its
chance of passing all three families is only .1406; at .99 it is .8829.
Those rates are sensitivity assumptions, not measured probabilities. Do not
reduce the denominator or launch an underpriced design to obtain a release.

The release allocation remains 21,600 CPU worker seconds and 60,000 GPU
seconds. The live ledger is
`artifacts/hmc-v7-release-2026-10-02/allocation.json`; after the source checks,
1,637.480 CPU and 29,502.265 GPU seconds remain,
including 250 CPU seconds reserved for reconciliation. All diagnostic workers
are terminal. The full-price queue is terminal; no confirmation is running.
The separate old-policy SSM service reports closed and remains outside the v7
evidence and allocation.

Execution continues through the existing phases in this order: preserve the
checked source; measure and validate the current-target probe cost repair or
quantify the funding gap; price the complete unchanged scientific procedure;
freeze an affordable and
statistically justified independent design; execute it with every planned slot
counted; and complete the terminal audit. If no supported design fits, preserve
release-pending status and state the precise funding/scope decision needed.
Missing/deferred/timeout slots remain unsuccessful on their original denominator.
No seed, panel, horizon, L grid, threshold or retention rule is changed to
rescue a failed result. Future repairs refresh the active gate/cost matrix,
not another independent tuning procedure.

## Historical development record

## October 3: measured throughput repair before confirmation

The original three source-08 development slots all ended incomplete. Their
failed outcomes and 5,825.951-second enclosing cost are preserved. Static versus
dynamic leapfrog graphs and health-transfer optimization did not supply a
complete search price. The next step therefore targets the measured native
small-call cost as well as preserving the completed host repairs.

The 107.093-second capacity diagnostic on QR/nonlinear shows that the existing
TFP row-batched kernel can process 32 four-start banks at much lower descriptive
cost per bank. Its shape-dependent streams cannot become trial evidence. The
new internal runner preserves each original TFP trial seed, seed splits,
integrator, MH arithmetic and target-status rows. Thirteen native tests pass
on CPU and in a correctly verified GPU/XLA run for two-trial batches on Gaussian,
QR, nonlinear and residual-funnel targets at L=3/25. A preceding test launch hid
GPU through pytest's default configuration; it is explicitly reclassified CPU
mechanics and is not GPU evidence. No admission or default changed.

The plan now requires maximum-batch parity, fresh-process loss/partial-save
recovery, unchanged all-member export/reload and full scientific configuration
before a batched full-search price. Default batch size 1 preserves historical
payloads; the optional size is bound into new execution identity. Full-search
stage timings distinguish tuning, export, reload, checkpoint reconstruction
and cost accounting. Confirmation remains unfrozen; partial prices and native
capacity checks cannot close the release gate.

October 3, 14:06 UTC refresh: maximum-size parity now passes for every one of
32 original streams on all four models, on both CPU and verified GPU/XLA.
The RNG-only native loop fixes the measured unrolled compilation cost while
keeping target evaluation batched. Failed and unsaved batch rows must retain
their attempted-work charges; both mutation controls pass. Final source-13
assembly and actual process-recovery checks precede the reserved single QR
full-search price. Keep original panels, seeds, six-L/M100 search and complete
all-member replay. No proxy/native timing is a complete price. This sequence
passes the skeptical audit: remaining questions are actual delivery, enclosing
cost and independent replication, with no changed admission thresholds.

## October 3: v4 scheduling and endpoint-health repair

The first GPU price used frozen `source-03`, whose controller policy was v3.
All three slots ran: QR and nonlinear exited with partial-budget results after
1,927.675 and 1,978.377 seconds; the funnel process timed out after 1,980.277
seconds. Total queue time was 5,886.334 seconds. The earlier administrative
closeout inferred termination from sandbox process visibility and was wrong;
`gpu-prices-01/closeout.json` is explicitly retracted. The genuine terminal
`result.json` supplies outcomes and charges. No slot was unstarted.

The reviewed v4 repair is now documented in the agent reference and this
chapter. It gives screened candidates their pending verification ladder before optional
extensions and repair descendants, while preserving broad primary coverage,
candidate-specific evidence, all verified members, and legacy checkpoint order.
This removes the observed verification-starvation failure mode but does not
make a full search cheap. The completed source-05 price measured v4 with the old
health profile; all three searches were incomplete (5,857.215 seconds total).
It cannot certify the later endpoint-health repair. The source-08 development
price tests that repair and the validated host-cost repairs. V7 stays pending
the affordability/delivery gate; no confirmation denominator is reduced.

The source-04 posterior regression exposed a genuine health-rule defect: a
moving Gaussian path happened to return close to its first point, triggering
the inherited endpoint-distance veto. Changing panels/seeds to obtain a pass
was rejected as a repair. The original failure was reproduced and preserved;
the explicit release profile now makes endpoint distance reporting-only while
retaining adjacent movement, repeated-state, short-cycle, divergence and finite
value vetoes. All 38 focused checks pass, including the original three panels
and seeds. Library defaults and old artifacts are unchanged. Fresh source-06
checks then passed, and source-08 pricing now measures the repaired profile
and host path. The release
plan records the derivation, positive/negative controls and 1,800-second bounded
repair allocation. No universal false-veto or posterior-coverage claim follows.

The [release execution record](bayesfilter-hmc-v7-release-execution-2026-10-02.md)
preserves every attempt. Full-search pricing exposed a real host-cost gap hidden
by the earlier single-pair tests: accumulated evidence was repeatedly serialized,
hashed and recomputed. Repairs preserve raw records, charge-before-call and
full validation at admission/export/reload boundaries. An isolated source build
also exposed and repaired the missing native Sylvester-library build step.
The persistence and scheduling repairs did not change numerical thresholds;
the explicit health profile is a separately recorded policy change. Release remains
pending full-search delivery, affordability, independent replication and the
terminal audit; successful launches and partial test progress do not close them.

The October 3 continuation also checked the existing static and dynamic
leapfrog graph options on the same QR target at L=3 and L=25. Exact samples and
traces agree on all eight paired seeds. The small shared-GPU timing comparison
is descriptive only and does not change the frozen price. The 225 distinct
recent regression tests and this parity check close their stated engineering
questions; they do not replace complete delivery or independent confirmation.

## October 2 acceptance robustness execution refresh

The [concrete acceptance repair plan](bayesfilter-hmc-acceptance-decision-repair-plan-2026-10-02.md)
and [matrix](bayesfilter-hmc-acceptance-robustness-matrix-2026-10-02.md) now include
the full requested test program. The controlling phase names remain P0--P5;
new evidence refines those phases rather than inventing a new tuner or silently
closing unmet conditions. The [execution note](bayesfilter-hmc-acceptance-repair-execution-2026-10-02.md)
and `artifacts/hmc-acceptance-decision-repair-2026-10-02/progress.json` carry
the current acceptance work independently of the active GPU SSM queue below.

| Phase | Latest checked evidence | Remaining closure condition |
| --- | --- | --- |
| P2-A evidence and recovery | 428 combined checks plus focused additions pass; individual chunk costs and missing charges are checked; cumulative trials deduplicate; actual serial/threaded streams and all four QR/nonlinear fresh-process interruptions pass | New target/backend combinations need their own numerical checks |
| P2-B numerical and input validity | Gradient/Jacobian/energy defects have independent tests; unknown generic SSM parameters fail before execution; matrix rows distinguish tested and unsupported inputs | Native hard-support rejection needs bound telemetry; supported diffuse/missing/singular variants need independent references |
| P3-A corners | Expanded 32 method/cells completed 16,384 searches; raw counts, streams and denominators audited; no observed false membership, wrong direction, false preparation or false temporal alert | Results remain conditional on the frozen laws; no universal calibration claim |
| P3-B full cap and temporal power | Another 1,024 searches exercised 100 actual candidates each; both methods delivered all 100 interior members, with independent verification; misleading pooled means retained none | Hoeffding temporal detection is weak at admission stopping time; absence of an alert cannot establish preparation adequacy |
| P4-A/B models | K0, QR/nonlinear, supplied exact/residual funnels and local mixture positive confirmations delivered/exported; all remaining named model families have bounded development measurements, including actual K6 after packaging repair | QR/residual-map/mixture confirmations and maintained positive/negative integrations pass; replicate target-specific delivery/cost before promotion |
| P4-C recovery and device parity | Four real-filter process recoveries, four CPU/GPU XLA parity checks and three GPU public-tuner deliveries pass; memory growth and actual chunk placement recorded | Longer-run resource behavior and expensive-model affordability remain separate |
| P4-D inference | K0 posterior mean/variance check passes; mixture missed-mode check correctly fails posterior assessment while tuning remains verified | Matched multi-replication coverage/SBC remains open; K6 lacks a joint posterior oracle |
| P5 guide | Official chapter, agent reference, capability-registry tables and regression inventory updated together; full book built and changed pages visually checked; documentation contracts pass | v7 remains experimental; 78 unresolved existing book citations are separate debt |

Whole-program review found and repaired two further root causes: aggregate
cost checks could miss a shortened native chunk after checksums were recomputed,
and unknown SSM parameters could describe a model different from the one run.
The snapshot model ladder also exposed missing QR provenance and K6 data files;
fresh-directory packaging retries preserve the failed attempts. A combined
recovery command hit its process ceiling after three tests, so bounded separate
processes on different CPU affinity supplied complete results without changing
scientific settings. These are implementation/harness repairs, not rejected
statistical methods or permission to relax acceptance criteria.

The acceptance continuation has a conservatively charged 28,800 CPU seconds
on top of its earlier 28,800- and 43,200-second allocations; the shared ledger
records 70,536.184 CPU worker seconds remaining. Do not debit these allocated
runs again as new spending. The trusted parity diagnostic separately reserves
600 GPU seconds from reconciled uncommitted balance, documented in
`expanded/continuation-01/gpu-parity-01/charge.json`. Three GPU public-delivery
checks reserve another 1,800 seconds in `gpu-delivery-01/charge.json`, leaving
67,612.317 GPU seconds uncommitted after the existing SSM reservation and
protected allowance. Their measured enclosing times are 22.691 and 727.615
seconds. Conservative charges remain in place. Shared execution, verified
growth and every native chunk's GPU/XLA placement were recorded. The active
SSM campaign keeps its frozen numerical policy and cannot validate v7.

Every next phase preserves failed candidates, exact configs and original
denominators, records cost and remaining work, and refreshes this program.
Unknown outcomes and unfunded cases remain open. R-hat and temporal diagnostics
do not control v7 tuning admission; posterior qualification remains separate.
No default policy has been promoted.

The terminal audit reconciles **441 distinct passing test IDs**, including
the focused corrections, without counting repeated suites twice. GPU model
delivery and statistical searches are separate results. The latest plan,
matrix and execution note above govern current work; dated status paragraphs
below preserve history and do not reopen completed work or authorize spending.

## October 2: authorized progress-aware state-space continuation

The owner added **48 CPU hours and 48 GPU hours** and requested repair, audit
and continuation. The [active amendment](bayesfilter-hmc-ssm-progress-continuation-plan-2026-10-02.md)
replaces the September 30 stage/attempt caps with one progress-aware queue.
It carries all seven incomplete existing main fits and five pilots, unlocks
unstarted main fits as matching pilots complete, and preserves the nine final
assessments and original 32-slot denominator. Both queue and child allocation
limits are repaired; numerical criteria and the frozen worker remain unchanged.

Phases are implementation and regression audit, trusted GPU continuation, then
terminal evidence/accounting review. Every allocation refreshes eligibility,
cost and remaining work. The skeptical pre-implementation audit is recorded in
the amendment; launch follows actual dispatcher and linear/nonlinear recovery
tests. The new service has a 48-hour ceiling including shutdown; old charges
and the protected unrelated allowance remain accounted for. Implementation
and launch are now complete. There are 156 distinct passing checks, including
actual linear/nonlinear checkpoint recovery. The official tuning chapter builds
and its changed page has been inspected.

The first launch found that checkpoint execution binds the exact GPU selection;
using another GPU of the same model was wrong for this checkpoint. Its 12 startup
failures changed no numerical evidence and consumed 97.995 seconds including
closeout. The diagnosed repair restores the original GPU and checks device
identity before dispatch, retaining all failure receipts and costs.

The corrected service `bayesfilter-hmc-ssm-progress-20261002-r2.service` started
**October 2 at 08:58:00 Shanghai**, with 172,702.005 seconds remaining under the
same campaign ceiling. Its latest end including shutdown is approximately
**October 4 at 08:56:22 Shanghai**; this is a ceiling, not a completion forecast.
The first K1 child passed reconstruction, GPU/memory-growth checks and XLA
compilation. The [execution result](bayesfilter-hmc-ssm-progress-continuation-result-2026-10-02.md)
records new durable K1 evidence beyond the original 147 observations, with
all previous numerical evidence and nine final assessments unchanged. The result
and `artifacts/hmc-ssm-progress-continuation-2026-10-02/r2/status.json` preserve
the current checkpoint. Machine-readable master progress refreshes after every
allocation and again at settlement. The new active CPU/GPU ledgers are under
that October 2 artifact root.

## October 1: acceptance uncertainty development complete; admission repair open

The [acceptance uncertainty plan](bayesfilter-acceptance-uncertainty-validation-plan-2026-10-01.md)
has been reviewed and executed. The experimental TensorFlow/XLA covariance and
MCSE diagnostics, adversarial calibration, seven-model integration checks,
eleven-case BGS replay, and guide update are complete. There are 171 passing
checks. Three bounded CPU calibration attempts consumed about 120 wall seconds
and no GPU allocation; no run remains active for this development phase.

The subsequent clean-checkout integration passes 201 checks after including
the required v6 policy reader and state-space fixture dependencies. The linked
result records this packaging repair separately from the calibration outcome.

The [result](bayesfilter-acceptance-uncertainty-validation-result-2026-10-01.md)
rejects statistical default promotion: strongly persistent stationary traces
produce false conflicts and poor interval coverage; merely increasing batch
sizes does not repair coverage. The existing optional v6 contrast also misses
opposing within-chain drifts. The new diagnostic remains experimental and cannot
issue receipts or change candidate membership. R-hat remains reporting-only
in ordinary tuning. These findings do not change or restart the separately
frozen DSGE, NeuTra, or state-space campaigns.

The October 2 [DSGE response and plan review](bayesfilter-dsge-acceptance-repair-response-2026-10-02.md)
confirmed the library-level gap. The subsequent
[code audit and concrete repair plan](bayesfilter-hmc-acceptance-decision-repair-plan-2026-10-02.md)
supersedes the earlier four-step sketch. It traces both public exact-score
routes, the position-field branch, controller, seeds, checkpoints and retained
export; 13 focused existing checks pass. It also identifies changing evidence
horizons, zero-width intervals from identical observations, insufficient model
tests, and a search simulation that does not match all-member retention.

The dependent implementation phases are now:

The owner's subsequent execute instruction makes every case in the
[robustness matrix](bayesfilter-hmc-acceptance-robustness-matrix-2026-10-02.md)
required work. The repair plan's **Execution amendment: required root-cause
coverage** is the controlling work breakdown: P2-A evidence/recovery and P2-B
numerical mutations; P3-A known-law corners and P3-B actual full-cap/temporal
calibration; P4-A public-model delivery, P4-B difficult/negative targets,
P4-C execution parity/recovery, P4-D independent posterior assessment; then P5
the official guide and release decision. These are subdivisions of the existing
program, not separate tuners or a new sequence of ad hoc phases.

Whole-program review found the old nominal candidate cap insufficient, missing
executable model-cost outcomes, uncalibrated temporal/preparation reports, and
mutation tests specified only in prose. Execution now addresses those root
coverage gaps. Gaussian/K0 require positive delivery; boundary, difficult and
negative cases have explicit other expected outcomes. Existing v5/v6 model
results cannot establish v7 coverage. Twelve additional CPU core-hours are
reserved from the existing grant for this bounded continuation; GPU work uses
only reconciled uncommitted resources. Each subdivision refreshes the same
progress record, records failures and chooses the next predeclared repair
without changing failed criteria. No acceptance result certifies convergence.

1. **P0: protocol and preflight.** Freeze starts, discarded prefix, measured
   horizon and weights. Define a broad qualification band separately from the
   preferred epsilon-tuning band. Price evidence before target work and reject
   structurally infeasible allocations. The versioned protocol, framework-free
   preflight CLI and focused contract tests are implemented; its public-route
   validation is included in P2.
2. **P1: health and statistics.** Preserve health semantics through an explicit
   shared evaluator. Implement independently checked bounded-trial confidence
   calculations, with a simple concentration reference and a derived sequential
   alternative. Four different starts are not repetitions; small-R t coverage
   and observed zero variance cannot be assumed. The shared health evaluator,
   trial summaries and intervals passed the 173-check P1 regression; new
   statistics remain experimental and do not change the default.
3. **P2: execution and durable integration.** Add independent fixed-horizon
   repetitions, complete-trial evidence rungs, persistent error allocation,
   cost accounting and exact resume across both numerical adapters. Update
   receipts, checkpoint recomputation and retained export together. Keep old
   policies and frozen campaigns under their original semantics.
4. **P3: actual-controller calibration.** Test all retained members and adaptive
   repair children, independent verification, repeated looks, invalid/aborted
   work, strong dependence and heterogeneous/opposing start behavior. Require
   useful decisions as well as error control; temporal diagnostics do not
   become convergence gates.
5. **P4: model delivery and affordability.** Exercise Gaussian controls, actual
   linear/nonlinear state-space targets, supplied whitened funnels, a mixture
   negative control and position-field mechanics. Price fresh BGS integration
   after cheap validation. No extra GPU reservation is taken from the active
   state-space campaign; the detailed plan declares bounded future allocations.
6. **P5: official guide and release decision.** Update the book, agent reference,
   capability registry and regression tiers together. Promote only a supported,
   useful and affordable policy scope. A statistically valid but unaffordable
   reference is not a completed practical repair.

The earlier P2 combined regression had 260 passing checks and no failures/skips,
covering repeated-trial collection, exact and mechanics resume, tamper detection,
receipt interruption, reserved verification work and positive multiple-member
export through both public exact-score tuners. The first P3 held-out slice
completed 512 searches in each of eleven cells for two methods (11,264 searches,
978.435 seconds). It observed no false membership or wrong direction assertions;
the simultaneous per-cell error-rate upper bounds are .015824. Betting met
the delivery criterion in all seven applicable cells. The Hoeffding reference
failed delivery on heterogeneous valid starts at this allocation. A subsequent
300-case CPU/XLA comparison found identical decisions after the statistics graph
refactor, with maximum absolute numerical difference 7.22e-16. See the
[execution note](bayesfilter-hmc-acceptance-repair-execution-2026-10-02.md).
The
[corner-case/model matrix](bayesfilter-hmc-acceptance-robustness-matrix-2026-10-02.md)
also distinguishes implementation tests, statistical calibration and posterior
assessment; model names or an all-inconclusive suite cannot close delivery.

The top-of-file refresh supersedes this earlier slice: M100/adversarial and
temporal calibration, scoped public-model delivery, GPU parity/delivery and
official book work have now completed. Replicated real-model delivery/cost,
matched posterior coverage/SBC and the release decision remain. The existing
v5/v6 default is unchanged; the experimental v7 policy is not promoted.
The separately authorized state-space resource
continuation retains its frozen admission policy and cannot establish that
these statistical repairs are complete. BGS supplies target-specific
preparation/geometry and affordability evidence alongside the library repair.

## Historical October 2 audit before the new amendment

The state-space pooled campaign ended normally at October 1 05:17:19 Shanghai.
**Nine of 32 original fits are complete**: one K0, all four K4 and all four K7.
Four K2 fits hit the three-attempt cap despite durable progress; three K0 fits
and five pilots were stranded at the repair-stage boundary. The incomplete
pilots leave 16 main slots unpriced. The service is inactive. After the separate
1,200-second allowance, 70,012.317 seconds (19.448 hours) of the grant remained
unspent; that balance does not renew an expired calendar deadline.

The [terminal audit](bayesfilter-hmc-ssm-pooled-repair-result-2026-09-30.md)
records the allocation diagnosis and numerical limits. All 14 predeclared
assessed members across nine fits pass their posterior checks, while one of
four exact-reference intervals misses and two of 48 numerical comparisons
only partially overlap the reference sensitivity interval. No ranking or
coverage-calibration claim follows. Preserve completed outcomes unchanged.

The next allocation repair should carry progressing pilots and fits across
phases in one dependency-aware queue, refresh prices/eligibility as pilots
finish, and permit further fair quanta within cumulative budgets and deadlines.
Keep no-progress/invalid-evidence stops and the original 32-slot denominator.
Test these cases through the actual dispatcher and frozen checkpoint recovery
before another launch. This is a required amendment, not an implemented repair
or a new launch. Statistical acceptance-uncertainty development above remains a
separate problem; this audit does not identify it as the cause of the timeouts.

## Historical state: September 30 pooled recovery repair

September 30 18:10 Shanghai update: the service is active in the main phase.
Six of 32 original fits have final assessments: one recovered K0, all four K4,
and one nonlinear K7. Their declared posterior checks pass; one of K0's four
exact-reference intervals misses the truth and remains reported. K5/K6 now
complete preparation and reach tuning, but their complete pilots remain
unavailable. K1/K3 pilots are also incomplete. The first interrupted K2/K7 main
fits are queued for automatic checkpoint continuation after peers. Numerical
source and coordinator files are unchanged, and the original enclosing end
ceiling remains in force. The current audit and detailed dispositions are in
the [execution note](bayesfilter-hmc-ssm-pooled-repair-result-2026-09-30.md).

The owner authorized the [reviewed pooled-repair plan](bayesfilter-hmc-ssm-pooled-repair-plan-2026-09-30.md).
It connects exhausted-cap checkpoint recovery to a shared queue, schedules
individual fits across models, and reports numerical-reference interval agreement
separately from exact coverage. K5/K6 require matching repaired development
workloads before main allocation; K3 data-B needs a matching reference price.
Completed negative assessments are preserved. The numerical worker remains the
verified September 29 frozen source; no tuning or posterior gates are relaxed.

The existing balance funds at most 150,410.488 GPU seconds, preserving the
separate 1,200-second reservation. The first repair/pricing stage is capped at
six GPU hours. T1 is September 30 04:52:19 Shanghai, with new compute/report
deadlines at T1+46/48 hours explicitly authorized by the execute request. The
old deadlines and charges remain historical facts. The active result/checkpoint
is [the September 30 execution note](bayesfilter-hmc-ssm-pooled-repair-result-2026-09-30.md).
Implementation and regression review are complete: 99 regression checks and two
real linear/nonlinear recovery tests pass, with 14 affected dispatcher checks
passing after final reporting changes. The shared-GPU continuation launched at
September 30 06:57:20 Shanghai as
`bayesfilter-hmc-ssm-pooled-20260930-r1.service`. Its first K0 child has verified
GPU/XLA/memory-growth provenance. The enclosing cap including shutdown ends
no later than October 2 00:44:11 Shanghai; it is not a completion forecast.

The saved diagnosis confirms K0's candidate counts of 84/70/98/98 versus 16 in
its pilot, dominated by extra measurement and directional-repair work. It does
not establish a controller bug or a causal GPU slowdown. Reassessment of the
three saved K4 fits gives ten intervals containing the empirical reference
sensitivity interval and two partially overlapping it. Those two remain
ambiguous; none of these twelve comparisons establishes exact or sequential
coverage. Original assessments are unchanged.

## September 29 baseline: main execution closed, validation incomplete

The measured main queue finished at 23:23:42 Shanghai on September 29. Seven
of 32 original fits ran: three K4 fits completed with declared posterior and
descriptive reference checks passed, and all four K0 fits exhausted their
contention-extended allocations during search. One funded K4 fit missed the
original latest-start cutoff; 24 slots were unfunded within the remaining wall
allocation. K4 interval-coverage records remain unavailable. No full-matrix
calibration or numerical default is established.

Both services are inactive and no campaign Python worker remains. The
18,893.277-second enclosing charge is settled exactly once, leaving
151,610.488 GPU seconds in the additive grant. The original latest-start
deadline has passed; that unspent balance does not reset this campaign's clock.
The [terminal result](artifacts/hmc-ssm-pilot-repair-2026-09-29/result.md) and
`artifacts/hmc-ssm-main-2026-09-29/r1/terminal-audit-2026-09-30.json` preserve
the complete inventory and receipt checks.

The next repair should first diagnose K0 candidate-work expansion and per-fit
pricing from its saved checkpoints. The 1.5 pilot pricing margin was inadequate
for all four main realizations; contention detection alone does not explain
the runtime. K1/K3 still need complete pilots, including K3's reference-workload
reconciliation; K5 needs a full repaired pilot and K6 preparation and reference
remain unresolved. K2 and nonlinear K7 have complete development evidence but
no funded main replications. No numerical work is currently queued. Preserve
all original failures and unstarted slots in any later continuation.

The saved-code audit confirms two concrete integration gaps: the main loop
does not assign additional common-pool time to a fit after its local cap is
spent, and stopped-interval coverage does not consume K4's numerical reference.
The latter currently looks up exact analytic functionals, which are absent for
K4; a repair needs explicit numerical-reference uncertainty. Across K0's 391
recorded observations, 245 directional epsilon decisions and 54 inconclusive
decisions account for substantial search work. All R-hat fields are explicitly
reporting-only. The [additional diagnosis](artifacts/hmc-ssm-pilot-repair-2026-09-29/result.md)
records these findings without attributing all runtime to contention.

The [September 30 budget assessment](bayesfilter-hmc-ssm-continuation-budget-assessment-2026-09-30.md)
records 42.114 GPU and 10.455 CPU worker-hours still available. It proposes
using at most six GPU and two CPU hours of that balance for diagnosis and
complete-fit repricing before requesting further compute. These are spending
caps, not completion forecasts. K5/K6 and the remaining search workload still
prevent a reliable full-completion quote; a new execution schedule remains to
be specified.

## Earlier September 29 nonlinear completion and measured main allocation

The first six-pilot recovery has finished. K0, K2 and K4 completed their full
declared workloads with cumulative prices 2,241.074, 4,841.006 and 2,533.406
seconds; assessed members passed their declared posterior checks. K1/K3 remain
in search at their resource caps. The last 3,332.905 seconds of the same repair
pool now fund one final K7 continuation. K7 has completed its search, retaining
all 21 verified candidates, and is performing the original two assessments.
K5 still needs a full repaired pilot; K6 preparation remains unresolved.

`bayesfilter-hmc-ssm-main-sequence-20260929-r1.service` waits for K7's normal
settlement, then starts the tested main allocator. It recomputes whole-lane
funding from complete cumulative costs and the actual remaining clock; the
obsolete 2,100-second per-fit convenience cap is no longer treated as measured
evidence. With 8.5 hours remaining, original K0/K4 lanes fit the allocation;
launch will recompute this, and each cell must start before 23:03:52 Shanghai.
All 32 original slots remain reportable. Main is still 0/32 while K7 runs.
The 56 focused main/allocation checks pass. A reference-workload mismatch
additionally excludes K3 from whole-lane pricing. Details and decisions are in
the [execution note](artifacts/hmc-ssm-pilot-repair-2026-09-29/result.md) and
[reviewed phase amendment](bayesfilter-hmc-ssm-pilot-repair-2026-09-29.md).

## Earlier September 29 campaign allocation repair and startup diagnosis

The [reviewed repair](bayesfilter-hmc-ssm-pilot-repair-2026-09-29.md) separates
additional campaign allocation from the original numerical design. It is
implemented with cumulative caps, bounded attempts, unchanged frozen workers,
and preservation of completed evidence. The focused selection passes 102
checks; two real linear/nonlinear CPU checkpoint continuations pass, and 43
startup/preparation-recovery regressions pass. The tuning chapter and reference
describe the new continuation path.
The live coordinator closeout repair subsequently passed 14 focused tests,
including a real timed-out CPU subprocess and normal receipt settlement.

The first GPU diagnosis passed both startup screens with smaller steps. K5
then failed a nonfinite mass-adaptation trace; K6 exhausted its first diagnostic
allowance during preparation. The completed second diagnosis established K5's
preparation repair and exposed K6's later warmup failure. The six independent
resource-stopped pilots use unchanged source/design checkpoint continuation.
Results, exact commands and budget accounting are in the
[execution note](artifacts/hmc-ssm-pilot-repair-2026-09-29/result.md).
The common additional repair pool is six GPU hours inside the existing grant,
36-hour SSM ceiling and original deadlines. All 32 main slots remain; no main
fit has started. This is the active phase; the stopped-run diagnosis below is
its baseline.

K5's follow-up has now completed preparation in 295.646 seconds with one
checked restart, 116 excluded discarded transitions and one metric update.
Its usable development configuration is explicit startup initialization with
20 rounds and up to three preparation restarts; this is not a numerical
default promotion. K6's longer preparation failed after 724.814 seconds on two
nonfinite warmup log-acceptance entries. The third K6 diagnosis discarded two
failed attempts, containing 315 and 450 transitions, after classifying rejected
nonfinite proposals with valid retained states. Its third preparation attempt
remained incomplete at the 1,775-second service limit; there is no successful
K6 handoff. K5's success is not evidence for K6 correctness.

That limit also exposed a coordinator closeout race. The service terminated
the coordinator before the nested timeout could be recorded, and the queued
sequence correctly refused the failure. The repaired launcher now reserves
30 seconds inside the service allowance for closeout and passes an absolute
numerical deadline. The original failed records remain preserved. After the
focused regression passed, direct continuation started at 12:49:11 Shanghai
in `bayesfilter-hmc-ssm-pilot-continuation-20260929-r1.service`. It resumes
K0, K7, then K1--K4 from their original checkpoints. Its remaining allocation
is at most 18,270.261 seconds including shutdown, within the six-hour repair
pool after all three diagnostic charges. No further K6 diagnostic is queued.
K0 has completed its full continued workload and both declared posterior
assessments. Its cumulative price is 2,241.074 seconds; the resumed portion
cost 464.420 seconds, and all preserved evidence passed validation. The queue
has moved to K7. Main remains 0/32 pending the complete affordability decision.
Main affordability must use cumulative continued-pilot costs. The inherited
per-lane ceilings were convenience allocations, and K0--K4's original 2,100
seconds per fit cannot contain even 1.5 times the already spent 1,775 seconds.
If a complete resumed price is otherwise affordable, the next phase must
explicitly reallocate the unspent main pool instead of treating that inherited
per-lane ceiling as a scientific failure. The 22-hour main, 36-hour SSM,
settled grant and original wall limits remain binding, and unfunded cases stay
in the original denominator.

## Earlier September 29 pricing failure diagnosis

Checked September 29 at 11:10--11:30 Shanghai: the shared-GPU service exited
at 05:38:30 after pricing failed. All eight GPU mechanics cells and all four
public-pipeline preflight cells passed. None of the eight complete-fit pilots
finished, so the main matrix has started 0/32 fits. The ledger is settled and
there is no active campaign worker. See the updated
[execution note](artifacts/hmc-shared-gpu-recovery-2026-09-29/result.md).

K0--K4 and K7 each consumed the 1,175-second nominal allowance and the full
600-second contention extension while continuing to make progress. Their
original fit allowance is exhausted, so automatic recovery cannot restart
them. K0 completed its search with five verified members and one of its two
declared posterior assessments. The other five searches remain incomplete.
K5 and K6 failed bootstrap with nonfinite proposal/log-acceptance diagnostics;
these are separate numerical startup failures, not contention timeouts.

The next repair must diagnose K5/K6 startup under the existing bounded
initialization mechanism, and reallocate pilot time explicitly within the
remaining repair budget before attempting a complete same-workload price.
Preserve completed candidate evidence and posterior chunks wherever the native
identity checks permit resume. Do not reset per-fit caps by rerunning the
unchanged queue, treat an incomplete pilot as a price, shorten the posterior
workload, or retry completed unfavorable assessments. Source/configuration
changes require fresh identities and cannot silently reuse numerical receipts.
Only complete affordable prices can admit the corresponding main cases; all
32 original slots remain in the report.

The GPU grant has 190,601.314 seconds (52.945 hours) left, including the
separate 1,200-second NeuTra reserve. Elapsed calendar time is now the tighter
constraint: the original compute cutoff remains September 30 at 03:03:52
Shanghai and the report cutoff is 05:03:52. There is no supported completion
forecast from the censored pilots, and the full matrix is not promised by those
deadlines. This status supersedes the launch-time descriptions below.

## Earlier September 29 shared-GPU recovery launch

The owner requires contention recovery rather than exclusive device availability.
The [shared-device repair](bayesfilter-hmc-shared-gpu-recovery-2026-09-29.md)
is implemented and tested. It admits trusted shared capacity at every layer,
keeps bounded progress-based extensions, and automatically resumes an incomplete
contention-interrupted fit at most once using its remaining original allowance.
Completed assessments, numerical failures and corrupt evidence are not retried.
All attempt costs and original deadlines remain binding. Preflight/pricing
now have explicit recovery allowances within the existing repair allocation;
main prices are limited by remaining wall time and cumulative SSM spending.

At launch, `bayesfilter-hmc-shared-ssm-20260929-r1.service` ran numerical preflight
on GPU 2 alongside foreign work, after passing 97 focused, 110 compatibility
and 50 final wiring checks (overlapping sets). Real linear/nonlinear state-space
subprocess tests preserved committed chunks through automatic recovery. The
official tuning chapter and reference are updated. The
[execution note](artifacts/hmc-shared-gpu-recovery-2026-09-29/result.md) records
commands, budgets and limits; live state is
`artifacts/hmc-shared-gpu-recovery-2026-09-29/runtime-r1/status.json`.
All ten C1 recoveries remain complete; no new C1 fits are scheduled. This is the
continuation that superseded the empty-device queue described below; its
terminal pricing state is recorded above.

## Earlier September 29 XLA preflight repair and continuation

All ten C1 timeout retries are complete. The post-hoc beta-binomial inventory
is 256/256 complete; original coverage failures and confirmation records remain
unchanged. All eight SSM GPU mechanics cells passed, but the four public
preflight cells stopped before compilation because the campaign adapter
declared no full-chain XLA capability.

The [bounded repair](bayesfilter-hmc-ssm-xla-preflight-repair-2026-09-29.md)
now has 69 passing regressions, including all eight full-chain XLA graphs,
deterministic graph/XLA leapfrog parity, public tuner execution, recovery reuse,
and corrected cumulative accounting. The target declares scoped diagnostic
capability only when XLA is configured. Actual GPU qualification is pending.

The restarted service `bayesfilter-hmc-ssm-xla-repair-20260929-r1.service` is
waiting for capacity; trusted telemetry reports foreign work on all three
GPUs. It reuses all ten C1 completions and automatically runs fresh GPU
mechanics/public preflight, complete-fit prices and funded main cases. Source,
status and commands are in
`artifacts/hmc-ssm-xla-repair-2026-09-29/`; the active GPU ledger remains
`artifacts/hmc-ssm-funded-2026-09-28/grant-ledger.json`.

The prelaunch balance is 57.878 GPU hours including the separate NeuTra reserve.
The 36-hour SSM ceiling, earlier charges and original clock are unchanged:
new work stops September 29 at 23:03:52 Shanghai, compute stops September 30
at 03:03:52, and reporting is due at 05:03:52. The
[execution note](artifacts/hmc-ssm-xla-repair-2026-09-29/result.md) distinguishes
engineering readiness, GPU qualification and posterior evidence. It supersedes
the dated operational status below.

## Earlier September 28 additional grant and capacity queue

The owner added 50 GPU hours. The new additive ledger at
`artifacts/hmc-ssm-funded-2026-09-28/grant-ledger.json` preserves all settled
charges and starts with 58.736 GPU hours including the NeuTra reserve. The
36-GPU-hour state-space plan is funded. Its original 42/46/48-hour clock and
numerical criteria remain unchanged.

The prior sequence stopped: fit 78 completed; fit 79 timed out under continuing
foreign GPU load despite its full allowance; eight later retries were deferred.
State-space preflight did no numerical work because its GPU stayed busy.
A separately bounded capacity queue now precedes numerical work. The
[continuation amendment](bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md)
preserves fit 78 and permits one fresh resource retry for each remaining seed,
then GPU preflight, complete-fit pricing and the unchanged 32-slot main matrix.
GPU qualification remains pending; new funding does not prove capacity or
convergence. Historical results below remain evidence, not current status.

## Earlier September 28 timeout recovery and state-space sequence

The owner authorized rerunning exactly the ten timed-out C1 fits and then
starting the state-space campaign. The [reviewed recovery plan](bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md)
is now the active execution sequence. The optional revised policy retains
earlier contention evidence, publishes earned allowance early, and waits
within budget before admitting a fit on a busy GPU. Tests passed: 103 focused
checks followed by 29 checks including actual Gaussian/beta-binomial isolated
pipelines (planner checks overlap). No numerical criteria changed.

C1 children use their unchanged frozen numerical source. Fresh recovery results
are post-hoc evidence and never overwrite the original 256-slot confirmation.
The new SSM source differs from its prepared baseline only in four scheduling
modules. The idle GPU 0 is the same hardware class as C1's currently occupied
GPU 1; device provenance is recorded. Recovery has an 80-minute enclosing cap
plus 30 seconds for shutdown/accounting, then automatic SSM preflight,
complete-fit pricing and funded main cases. The existing 9.185 usable GPU hours
limit the entire sequence; every unfunded SSM slot stays in the 32-slot report.
The historical outcomes and coverage failures below remain unchanged.

## Active September 28 C1 outcome and timeout finding

C1 terminated on September 27 at 06:13 Shanghai time after 39.176 GPU worker
hours. Gaussian completed 256 fits; beta-binomial completed 246 with ten
per-fit timeouts. Both delivery screens passed, but several coverage/joint
screens failed. The terminal receipt is settled exactly once; 9.519 hours
remain in the additional grant, or 9.185 after the separate NeuTra reserve.
The state-space campaign has not launched.

The [saved-record timeout audit](bayesfilter-hmc-c1-timeout-audit-2026-09-28.md)
identifies sustained foreign GPU processes and fixed 750+120-second fit caps.
Nine stopped during candidate search; one started its first posterior warmup
chunk. A current-only contention check denied fit 82 its extension after the
other process disappeared near the deadline. The current and frozen SSM
supervisors share these limitations. The next execution repair should cover
bounded admission waits, cumulative contention evidence and early allowance
decisions before a costly GPU launch. This audit did not change runtime code,
rerun C1, replace outcomes or allocate new compute. Model/reference preparation
is complete; the new scheduling finding and separate coverage diagnosis remain
open. The dated September 26 preparation state below is preserved as history.

## Active September 26 state-space preparation

The owner's model-relevance and time-budget questions are addressed in the
[reviewed 48-hour proposal](bayesfilter-hmc-state-space-48h-plan-2026-09-25.md).
It adds actual filtering-target integration, independent numerical references,
focused failure tests and a 32-fit state-space inventory, capped at 36 GPU and
16 CPU reference/test worker-hours including reserves. Full-fit pricing must
establish affordability before the main matrix runs; the deadline guarantees
a terminal report, not successful convergence of every model.

The owner instructed us to prepare this work while C1 finishes. C1 continues
on its frozen source. This supersedes the proposal's original recommendation
to end C1 early. There is no second GPU reservation: complete-fit pricing after
C1 settlement determines what fits in the actual grant balance. Every unfunded
slot remains in the 32-slot inventory. The official guide remains
`docs/main.tex` and its tuning chapter; this is an execution plan.

## September 26 continuation: K0--K7 implementation prepared

The 48-hour state-space work is now implemented through the existing
inference-validation executor. The active plan is
`bayesfilter-hmc-state-space-48h-plan-2026-09-25.md`; its CPU preparation
artifacts live under `artifacts/hmc-state-space-preparation-2026-09-26/`.
The rejected repeated-legacy-fixture draft is no longer the launch suite.
`docs/validation/state-space-48h.json` is the single 32-slot K0--K7 inventory,
with all verified tuning candidates retained and R-hat/ESS/MCSE confined to
posterior assessment. The planner includes independent reference checks,
Laplace/prior comparators, K7 approximation diagnostics, GPU/XLA canaries,
complete-fit pricing and a fresh-source launch route. The broad CPU selection
passed 142 tests; 24 overlapping focused checks cover the final controller and
strict K0 pipeline tests. All 14 oracle-backed datasets passed the declared
reference sensitivity checks; K6's posterior oracle remains unavailable.
The launch package and command are frozen in
`artifacts/hmc-state-space-preparation-2026-09-26/prepared-final-r1/` and the
active plan. The [preparation result](bayesfilter-hmc-state-space-preparation-result-2026-09-26.md)
records tests, failures, repaired tracing/status/shape issues, limitations and
the conservative six-CPU-hour preparation charge. These six hours are part of
the 16-CPU-hour ceiling, not an additional allocation. GPU execution requires C1 terminal
settlement, measured affordability and trusted GPU/XLA preflight; preparation
itself has made no GPU reservation and has not touched C1.

## Active update: September 25 execution-budget repair

The [bounded debug plan](bayesfilter-hmc-budget-debug-2026-09-25.md)
governs the current work. C1's isolated executor lost the development wrapper's
explicit graph-reuse setting. That option now propagates through the public
validation CLI and isolated child. Sequential suites can also carry unused
settled-cell time forward without enlarging the total or resetting a fit's
allowance. Both full GPU comparisons passed exact numerical/evidence parity;
70 current regression cases and two finalizer checks pass. Direct repricing
reduced the full C1 forecast to 37.473 hours within its existing 46-hour ceiling.
The complete 512-fit confirmation is running as
`bayesfilter-hmc-c1-repaired-20260925-r1.service`; its first fit completed
normally. See the [repair result](bayesfilter-hmc-budget-debug-result-2026-09-25.md).

C2 recovered one of its two cell-limited fits on its original source and now
has **255/256 complete**. Its remaining half-defect fit exhausted the original
cumulative allowance. All three conservative rate screens pass, but the
complete-study requirement remains unmet. The final missing slot is preserved,
not silently restarted. The recovery's 767.645 GPU seconds are charged to the
old C2 reservation. The debug GPU costs use the separate September 25 grant.
The [execution checkpoint](bayesfilter-hmc-gap-closure-execution-2026-09-24.md)
and machine progress hold the current service and balances.

## September 25 GPU allocation

The owner authorized **50 additional GPU hours**. The
[active allocation and reviewed execution decision](bayesfilter-hmc-gap-closure-phases-2026-09-24.md#september-25-additional-allocation-and-active-execution-decision)
supersedes the historical unfunded-C1 statements below. The original direct
price was unaffordable; the debug plan repairs the execution mismatch and
reprices the unchanged 256-Gaussian/256-beta-binomial confirmation before its
46-hour launch ceiling can be used. A separate additive-grant ledger prevents
double counting.
No candidate criteria, sample counts or historical outcomes are changed.

The September 25 NeuTra architecture policy also supersedes the former J2
pricing recipe. Use the canonical author-profile IAF for new learned-map
work; historical configured-map mechanics remain historical evidence. Its
target-specific training and downstream confirmation still need a priced
protocol. Exact MacroFinance inputs and terminal C1/C2 scientific review
remain separate obligations. The checkpoint and receipts are under
`artifacts/hmc-additional-gpu-2026-09-25/`.

Updated 2026-09-24. The active execution phases are now G–L in the reviewed
[closure amendment](bayesfilter-hmc-gap-closure-phases-2026-09-24.md), added at
the user's request to cover all remaining gaps. The preceding follow-on is the
[remaining-gap repair and validation program](bayesfilter-hmc-remaining-gap-program-2026-09-24.md).
Execution is underway. A reconciled all 272 historical fits, repaired a
tie-sensitive quantile cutoff and two compatibility imports, and passed the
independent-reference and affected pipeline checks. B completed its saved-array
diagnosis and all eight GPU development fits. C2's first confirmation attempt
stopped after three per-fit timeouts, with nine of 256 fits completed. Its
65,100-GPU-second ceiling remains cumulative within the existing allowance. The
[execution checkpoint](bayesfilter-hmc-remaining-gap-execution-2026-09-24.md)
records its frozen design and skeptical audit; the
[current results](bayesfilter-hmc-remaining-gap-results-2026-09-24.md)
separate completed engineering/development evidence from pending confirmation.
C1 coverage confirmation remains unfunded alongside C2. G–L start with actual
costs and mechanism checks and execute full confirmation only when its complete
design is priced and funded; an unavailable input or failed candidate does not
stop independent phases.

G has completed three full GPU fits and the fixed-workload diagnostic. H's
assessment explanations are implemented and tested. J's deterministic capacity,
inverse/Jacobian/score checks passed on CPU and GPU. K rechecked the missing
exact inputs. L repaired one compatibility import, aligned the official chapter
and rebuilt its bibliography successfully. A fresh, separately identified I
confirmation has launched under the existing C2 reservation. Its 128/64/64
inventory and statistical screens are unchanged; the historical nine outcomes
are not pooled. The [active checkpoint](bayesfilter-hmc-gap-closure-execution-2026-09-24.md)
records the decisions, and machine progress records live/terminal execution.
L's final scientific audit remains pending that confirmation.

The latest requested continuation adds **I2/J2/K2/L2** in the same amendment:
make the unchanged 512-fit C1 inventory executable and price its funding gap;
prepare and test configured-family GPU pricing with correct numerical
provenance; recheck exact consumer dependencies; and close narrowly supported
facade imports with actual caller tests. These CPU engineering steps execute
while I runs. They have a 1,800-CPU-second ceiling from the existing balance;
they neither spend I's reservation nor launch competing GPU work. Full C1 and
learned-map quality remain open until their entire designs are priced and
funded. The continuation's skeptical review is recorded before implementation
in the [amendment](bayesfilter-hmc-gap-closure-phases-2026-09-24.md#requested-continuation-i2-j2-k2-and-l2).

I2/J2/K2/L2 have now reached their reviewed engineering dispositions. The
512-fit C1 suite is generated; delivery, coverage and their joint event have
separate tested reports. Configured IAF/DSF pricing is executable and its six
CPU mechanics arms pass. Three active-consumer facade dependencies were
removed, and the stale LGSSM source pin was repaired after reconstructing its
old identity and checking numerical parity. The continuation has 168 distinct
passing scoped tests and a clean, visually inspected official-book build.
Its 1,153.045 CPU seconds, including failed attempts and conservative staging
charges, are reconciled; it used no GPU allocation. These counts overlap earlier
regressions and must not be added as independent scientific evidence.

I remains running; its terminal statistical audit is next. Full C1 is still
unfunded, configured-map GPU pricing and downstream quality remain open, and
K2 confirmed that the exact MacroFinance inputs remain absent. The
[execution checkpoint](bayesfilter-hmc-gap-closure-execution-2026-09-24.md#continuation-i2j2k2l2-execution)
and machine progress preserve these remaining obligations. No scientific gap
is closed by test counts or a successful service exit.

## Additional closure phases G–L

| Phase | Gap and deliverable | Closure condition |
| --- | --- | --- |
| G | Runtime under load: fixed workloads, full-fit prices, bounded execution settings. | Measured complete fits or explicit censored costs; honest campaign affordability decision. |
| H | Burn-in and uncertainty: explanatory readiness information, independent MCSE/stopping diagnosis and multi-model tests. | Mechanism checks pass; full C1 statistical calibration remains separately required. |
| I | C2 source/budget disposition and funded confirmation; C1 funding assessment. | Full original denominators and unchanged rate screens, or explicit incomplete/underfunded result. |
| J | Supplied-whitened geometry and learned-map capacity, then target-specific training/validation if affordable. | Capacity/inverse/Jacobian/score checks before any learned-map quality claim; global exploration evaluated in model quantities. |
| K | Exact MacroFinance reproduction and matching joint reference. | Exact inputs and reference, or a concrete missing-input record. |
| L | Bounded facade cleanup, official book/reference alignment, tests and final reconciliation. | Scoped regressions, rendered book and a disposition for every requirement. |

The skeptical review in the amendment rejects five shortcuts: extrapolating
supervisor success into fit completion, silently mixing source versions,
conflating lugsail with quantile/readiness rules, treating map capacity as
posterior quality, and adding bibliography entries before diagnosing the build.
Development caps total 10,500 CPU and 4,140 GPU seconds within the recorded
balance; the 60,022.255 GPU-second C2 reservation remains separate. These are
maximum debugging allocations, not new prices or defaults. Each phase records
results, charges and the next phase's revised decision before continuing.

### Active timeout-supervision amendment

C2's timeout diagnosis led to a completed engineering repair, governed by the
[timeout-aware supervision plan](bayesfilter-hmc-timeout-aware-supervision-plan-2026-09-24.md).
The three historical 890-second kills occurred during productive tuning; the
native tuner recorded no wall limit, and the isolated executor stopped each
cell after its first non-complete child. The repair propagates each fit's
remaining deadline through preparation and candidate-set search, observes
durable numerical progress, records trusted CPU/GPU workload telemetry, grants
only explicitly budgeted contention grace inside the cell ceiling, and
continues after a local timeout while preserving the full denominator. Unknown,
corrupt, or nonzero-exit failures remain cell-stop conditions until diagnosed.

The base fit allocation remains cumulative and cannot be reset. Existing C2
artifacts and the frozen suite are historical evidence and are not rewritten;
any continuation uses a fresh versioned output root and the remaining original
allocation. This amendment changes execution supervision and diagnostics only;
it does not make R-hat a tuning gate, alter candidate retention, or promote a
runtime observation into sampler or statistical evidence. Dynamic allowances
are separate from numerical identity, and budget-stopped posterior chunks stay
resumable. The reviewed amendment includes deterministic boundary tests, real
CPU pipeline/resume tests and, after those pass, a maximum 600-GPU-second
telemetry canary from the existing unreserved balance. This diagnostic uses a
fresh design/output and cannot complete or replace historical C2 confirmation.

The [terminal repair review](bayesfilter-hmc-timeout-aware-supervision-result-2026-09-24.md)
records 91 isolated-source tests and 41 later reporting/documentation checks
(overlapping counts), plus a GPU/XLA canary that granted 90 seconds with trusted
contention and numerical progress, then terminated at the declared final cap.
The fit remained unavailable; its 33 tuning observations and checksum survive.
Enclosing cost was 541.547 GPU seconds. The worker exited and no process remains.
This closes the timeout instrumentation and supervision gaps, not full-fit pricing
or confirmation. A changed source/policy cannot resume frozen C2 under its old identity,
and a new experiment cannot reset the three exhausted fit slots. Confirmation
closure requires a complete, explicitly accounted design, not a successful
smoke test or fewer missing fits. The next phase must price complete fits under
the intended sharing conditions and specify the old/new source strata, untouched
fit inventory and cumulative budgets before any confirmation continuation.

M31--M38 completed its funded engineering scope. Its
[result](bayesfilter-hmc-m31-m38-result-2026-09-23.md),
[terminal checkpoint](bayesfilter-hmc-m31-m38-execution-2026-09-23.md) and
[exact former master](bayesfilter-hmc-repair-master-m31-m38-archive-2026-09-24.md)
remain historical evidence. Do not restart them or append an automatic M39.
The earlier [history through M30](bayesfilter-hmc-repair-master-history-through-m30-2026-09-23.md)
is also preserved. Their old next-step instructions are not active.

## Current requirement state

| Requirement | Disposition and next work |
| --- | --- |
| R1: unified public procedure and all-candidate lifecycle | Engineering checks passed for the declared matrix. Preserve two public tuners, all verified members and posterior-only R-hat/ESS/MCSE. |
| R2: posterior delivery and uncertainty | H explains assessment requirements and failures. Repaired C1 is funded and running with its full 256+256 inventory; terminal delivery/interval review is pending. The 48-hour proposal would preserve an early stop as incomplete. |
| R3: full-procedure defect sensitivity | Fresh I/C2 has 255/256 complete after original-source recovery. All three conservative rate screens pass; the complete-study requirement remains unmet. Historical 9/256 results stay separate. Actual SSM failure-mechanism coverage is proposed, not yet demonstrated. |
| R4: global exploration | Supplied-map integrations pass. Difficult-mixture model-coordinate exploration remains open; J's capacity witnesses do not establish it. |
| R5: exact MacroFinance consumers | K rechecked the exact bootstrap paths and full-joint MIDAS reference dependency. Both remain unavailable; synthetic or fixed-loading results do not substitute. |
| R6: structure and reference coverage | Bounded compatibility repairs and independent ArviZ checks passed; one external fixture and legacy facade debt remain. The current catalogue's LGSSM target uses a dense likelihood, so actual filtering-target pipeline coverage needs the proposed factory and reference integration. |
| R7: learned-map quality | September 25's canonical author-profile policy supersedes the legacy J/J2 recipes. Historical capacity checks do not establish current learned-map quality. Current target-specific training and downstream validation remain separately priced work; the 48-hour SSM proposal does not claim to close them. |
| R8: guide and evidence alignment | The execution-budget repair has a built and visually inspected official chapter. C1 terminal review is pending; C2 incompleteness remains explicit. New SSM coverage will enter the guide only after implementation and evidence. |

The new audit recovered a material distinction: all four available-interval
Gaussian coverage screens passed, but missing warmup outputs caused the
all-planned-fit screens to fail. Beta-binomial fixed-count intervals also missed
the screen. Those observations do not establish a single stopping/MCSE cause.
Quantile MCSE uses a separate estimator from lugsail. Read the active plan's
cause table before changing defaults.

## Preceding A–F program: preserved history

This table records the predecessor work packages. Active instructions and
remaining work are G–L above; do not relaunch A–F from this historical table.

| Work | Deliverable | Dependency |
| --- | --- | --- |
| A | Existing-evidence inventory, independent reference tests and narrow structural repair. | First. |
| B | Mechanism diagnosis, bounded posterior repairs, frozen candidate policies and actual prices. | A. |
| C | Complete independent coverage/delivery and named defect-detector confirmation. | A/B and adequate funding for each entire frozen cell. |
| D | Supplied-map/global exploration tests and adequately priced learned-map protocols. | A; serious posterior/training claims also need B and their own evidence. |
| E | Exact consumer integration and local reply. | Exact inputs; mechanics and posterior-reference needs remain separate. |
| F | Affected tests, official guide, terminal requirement report and reconciliation. | Every package has a disposition; always runs. |

The preceding plan fixed statistical units, model/negative-control coverage and
evidence roles that G–L preserves. The active checkpoint refreshes results and
budgets between G–L phases. No shortened confirmation denominator or relabelled
smoke test closes a scientific requirement.

## Budget and authority

The A–F amendment opened with **71,946.615 CPU and 74,041.939 GPU
worker-seconds** from M38 (about 19.99 and 20.57 hours).
This was not additive to earlier grants. The fresh I campaign is active as
`bayesfilter-hmc-closure-i-20260924-r1`; consult machine progress for its terminal
state. Its inner ceiling is 59,990 seconds and its service contains all child
process groups within 60,000 seconds plus one second of shutdown grace.
The [live ledger](artifacts/hmc-remaining-gaps-2026-09-24/reconciliation-live.json)
charges completed and failed attempts; pending GPU work is reserved separately.

At G's opening, completed A–F and timeout-repair attempts had charged about 9,374 GPU seconds,
including the 5077.745-second C2 attempt and 541.547-second diagnostic canary.
About 64,668 GPU seconds remained, of which 60,022 were the unspent C2 allocation
for diagnosis/continuation and 4,646 unreserved. About 69,588 CPU
worker-seconds remained. The current G–L charges are in
`artifacts/hmc-gap-closure-phases-2026-09-24/closure-ledger-checkpoint.json`;
I's running cost is reserved and will be charged from its enclosing receipt.
Initial repair debugging receipts use pytest elapsed
excluding interpreter startup; later checks use enclosing wall receipts.
The ledger records exact accounting values and provenance; do not
double-charge nested child receipts or reset cumulative fit/cell ceilings.
G's new-source prices forecast **58,218.937 GPU seconds** for fresh C2, versus
its existing 60,022.255-second reservation. This small margin is unproved;
completion is not guaranteed. The older 56,724-second forecast remains
historical. C1's two no-fixed-comparator
scenarios total about 137,189 seconds, still unfunded alongside C2. The combined
point forecast is about 53.86 GPU hours before further training/consumer work.
These are descriptive forecasts, not guarantees. Preserve full denominators
and all failures if the actual run reaches its cap.

The earlier planning source was `622d9a9ed028659328b6386f9e9de7f368e7bba3`.
G/I use the recorded `source-g1` snapshot over
`de80aaff5812ebfbed551977476c0868551a2c88`; H/L's owned reporting and import
changes are separately tested in `source-l3`. Preserve each
historical experiment's own source identity. Exclude unrelated dirty Q20,
training, governance, chapter 26b and bibliography changes from owned
snapshots and edits. The active plan is an implementation/research amendment,
not another tuning guide: the book remains `docs/main.tex`, with
`docs/reference/hmc-tuning-interface.md` describing the same API.

[Machine progress](artifacts/hmc-repair-master-2026-09-16/program-progress.json)
tracks these same dispositions. `scientific_gaps_closed` remains false.
