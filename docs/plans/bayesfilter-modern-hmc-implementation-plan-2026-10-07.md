# Implementation and validation plan for modern HMC methods

Date: 7 October 2026. Status: **SIR bounded cohort complete; master program
in progress; checkpoint refreshed 20 October 2026**. Optional typed randomized
HMC, deterministic fully refreshed MAMS and discarded ChEES-rate/SNAPER proposals
are implemented for engineering validation. They remain experimental. Current
tests, repairs and remaining phases are recorded in the
[active execution record](bayesfilter-modern-hmc-execution-2026-10-07.md).

**Current checkpoint, 20 October 2026:** The
[full-model SIR phase](bayesfilter-modern-hmc-sir-evidence-phase-2026-10-17.md)
finished at **01:42 China time**. All four methods are terminal and the service
is inactive. The [terminal result](bayesfilter-modern-hmc-sir-result-2026-10-20.md)
records 24 verified tuning members, 23 completed posterior assessments and
21 posterior/reference deliveries.

| Method | Verified | Posterior checks passed | Posterior/reference deliveries | Other disposition |
| --- | ---: | ---: | ---: | --- |
| Fixed HMC | 8 | 7 | 6 | Member 1 reference failure; member 8 resource-censored |
| Randomized HMC | 8 | 8 | 7 | Member 1 reference failure |
| SNAPER-proposed randomized HMC | 6 | 6 | 6 | None in this cohort |
| Fully refreshed MAMS | 2 | 2 | 2 | None in this cohort |

All verified tuning members remain retained. Neither reference failure was
selectively retried. Posterior R-hat, ESS, MCSE and reference agreement remain
separate from acceptance/health-based tuning membership. These descriptive
counts do not rank methods; candidate sets and costs differ and independent-fit
replications remain missing. The reference applies to the declared finite-filter
target, not the exact nonlinear state-space posterior.

The [controller recovery](bayesfilter-modern-hmc-sir-controller-recovery-2026-10-18.md)
preserved the fixed prefix without replay; its exact orchestration snapshot
passed 185 tests. Numerical authority remains `full-model-source-e68cdf1c4`,
and all designs, seeds, thresholds and recorded GPU/XLA/memory-growth settings
remain frozen. Saved strict audits and a fresh closeout check passed: **5,530
unique source/evidence hashes** plus member decisions and accounting reconcile.
The fresh audit does not independently recompute TensorFlow ESS/R-hat/MCSE.

Inclusive numerical cost was `197315.48680924764` seconds. Controller accounting,
the previously outstanding submission receipt and measured closeout check leave
**`22375.605841124954` seconds** from the SIR allocation. Pricing and nested
worker time were not charged again. See the
[closeout settlement](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/sir-cohort-20261017-01/closeout-20261020-01/settlement.json).
The separate diagnostic pool remains **`131192.0584563613` seconds**; the
separate 3600-second hold and historical 1579.5929542018566-second balance remain
excluded. No new numerical study has been launched or allocated.

The SGQF cohorts are also terminal; SNAPER delivered five of six qualified
members. The [stationary-stress](bayesfilter-mcse-stationary-stress-result-2026-10-18.md),
[antithetic ESS](bayesfilter-mcse-antithetic-ess-result-2026-10-18.md),
[saved nonlinear replay](bayesfilter-mcse-saved-nonlinear-result-2026-10-18.md)
and [interval follow-up](bayesfilter-mcse-dependence-interval-result-2026-10-18.md)
are complete. The interval study's 768 records represent 384 independent latent
four-chain units: 22/108 fixed and 3/108 conditional stopped screens flagged
undercoverage among 231 adjusted screens. Dependent cells mostly never stopped.
No replacement estimator/default was validated, and stationary starts supply
no burn-in evidence. The
[documentation closeout](bayesfilter-mcse-interval-documentation-result-2026-10-18.md)
passed 15 contract checks, a diagnostics PDF build and changed-page inspection.

**Next N5 phase:** scope and price a predeclared multi-model/independent-fit
confirmation from measured all-member costs, with full failure/censoring
denominators and uncertainty analysis. Do not selectively retry failed members
or automatically spend the released balance. Calibrated precision/stopping and
actual MacroFinance/DSGE consumer qualification remain open. The
[N8 engineering slice](bayesfilter-modern-hmc-n8-consumer-audit-result-2026-10-17.md)
passed 171 tests, 15 documentation checks and an 82-page excerpt; it is not
external-consumer or release qualification. N6/N7 remain conditional branches,
separate from a scoped N2–N5 release. The
[active checkpoint](bayesfilter-modern-hmc-randomized-active-checkpoint-2026-10-15.md)
contains exact paths and next actions. Earlier observations below are historical
and do not supersede this terminal SIR status.

**Prior status, 15 October:** The fixed-HMC continuation is terminal: all
eight verified candidates were assessed, seven delivered and one failed the
declared precision/reference screen without a tuning veto. Its 48-hour grant
settled with 137,361.195109 worker-seconds available after the held local
reserve. The storage cleanup that preceded this refresh was performed to free
capacity for this same program; archived evidence was retained and no separate
campaign consumed the balance. The next prescribed comparator is the frozen
randomized-HMC SGQF cohort. Its minimum-count price exceeds the balance, so the
new phase is explicitly a bounded feasibility attempt with full root and repair
denominators, not a promised complete comparison. The executable phase is in
the [randomized-HMC feasibility plan](bayesfilter-modern-hmc-randomized-feasibility-2026-10-15.md).

**Historical execution update, 15 October:** The randomized-HMC phase passed its actual
immutable-source GPU preflight and launched as
`bayesfilter-modern-randomized-feasibility-20261015-01.service` on GPU 1. The
preflight performed no sampling and verified source commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`, the eight frozen roots,
the 18-quantity reference compatibility report, data and geometry identities,
XLA and memory growth. The launch receipt and allocation are under
`N5/generic-adaptation/priced-comparison-01/randomized-feasibility-01/`.
The numerical cap is 137,000 inclusive seconds from 137,361.195109 settled
seconds; the existing 3,600-second local reserve is already held and is not
charged twice. At that launch the worker was qualifying and resource censoring
was possible; no ranking, default, broad validity or release conclusion was
permitted. The recoverable phase checkpoint is
`bayesfilter-modern-hmc-randomized-active-checkpoint-2026-10-15.md`.

**Historical status, 14 October:** The fixed-HMC comparator stopped at its total
allocation at 11:21 China time. All eight primary candidates were independently
verified and retained; one passed posterior/reference checks, one has seven
saved sampling chunks, and six have not started posterior assessment. The
[fixed-HMC terminal audit](bayesfilter-modern-hmc-fixed-terminal-2026-10-14.md)
checks source/input and saved-chunk integrity, records the reporting repair,
and settles the allocation. The owner has now granted 48 additional hours.
The [posterior continuation plan](bayesfilter-modern-hmc-posterior-resume-2026-10-14.md)
reserves one hour for checks/closeout and up to 47 hours for the complete
unfinished cohort. A separate resume driver reuses the immutable numerical
source, all eight verified identities, completed outcomes, saved chunks and
original seeds. Forty-two continuation/consumer tests and six supplemental
checks pass; two additional fixed-trajectory tests and the actual-source GPU
preflight also passed. The detached continuation launched on GPU 1 at 17:45
China time on 14 October. Its inclusive 47-hour cap ends approximately 16:45 on
16 October; the supervisor records heartbeats and committed work separately.
The [active checkpoint](bayesfilter-modern-hmc-active-checkpoint-2026-10-13.md)
records its service, output and budget receipts. Descriptive saved-price scenarios are
25–42 hours before overhead or changed machine contention; completion within
the allocation is not guaranteed. No tuning work is repeated.

At the 14 October 23:37 China-time inspection, six members passed posterior
and reference checks, one reached the sample cap with insufficient precision,
and the eighth is sampling. The [interim result](bayesfilter-modern-hmc-continuation-progress-2026-10-14.md)
records the failed second-moment screen and a passing source/checkpoint audit.
The active cohort continues. Its failure is a posterior outcome, with tuning
membership preserved; no reference threshold, estimator or count cap changes.
The [closeout plan](bayesfilter-modern-hmc-cohort-closeout-2026-10-14.md)
adds tested terminal settlement and prepares the complete randomized-HMC
baseline next, subject to terminal audit and the actual released balance.

The bounded full SGQF MAMS continuation finished;
four of four verified candidates passed all declared posterior and finite-target
reference screens. The [terminal note](bayesfilter-modern-hmc-priced-mams-terminal-2026-10-13.md)
records the result and cost. N5 broader matched-model confirmation, broader
estimator/stopping validation, and N8 consumer/documentation/release checks remain open. N6 LAPS
has its own conditional development scope; N7 alternatives are triggered only
by a specified residual failure. No broad comparison or release is established.

The [October 13 continuation](bayesfilter-modern-hmc-comparator-precision-continuation-2026-10-13.md)
settles the MAMS cost and releases unused pricing allowance. It reserves 3,600
CPU-worker seconds for a fresh optional MCSE safeguard evaluation and allocates
79,364.235006 seconds to the complete eight-root fixed-HMC baseline under the
unchanged SGQF design and immutable numerical source. Forty-eight focused
settlement/report/launch tests pass. Other methods/models and independent
replications remain unfunded in this allocation. The precision experiment
cannot retroactively change the completed MAMS result or frozen comparator rules.

The fixed-HMC service ran from 13 October at 13:18 until 14 October at 11:21
China time, preserving 896 qualification calls. The optional `lugsail_max_batch_means` policy
is implemented with both component estimates and activation diagnostics; 78
focused CPU/XLA checks and nine GPU parity cases pass. Reassessment of all four saved MAMS members
reproduces the original precision values and preserves their first passing
looks (6000, 10000, 6000, 3000). The fresh 128-replication, six-condition
calibration is complete, with all 768 records audited. The
[terminal precision result](bayesfilter-mcse-safeguard-result-2026-10-13.md)
records both favorable antithetic variance behavior and unresolved coverage.
There is no supported paired coverage ranking or default promotion. Actual
CPU-study cost is 2019.235247 seconds, releasing 1580.764753 seconds. Charging
the fixed-HMC supervisor's 1.171799-second termination overhead leaves
1579.592954 seconds unallocated; the full local-check reserve remains held.
The terminal note records the accounting. This balance does not fund the
seven remaining posterior assessments; the October 14 grant above funds their
bounded continuation separately.
The book chapter and API guide document the optional policy.

**Earlier progress through 12 October:** D2f, D3, E3, D4a and the first D4b1
full SGQF cohort are terminal. All 15 frozen SGQF proposals completed fresh
tuning verification. SNAPER delivered 6/6 posterior and separate finite-target
reference screens; MAMS delivered 5/9, with one inconclusive warm-up and three
precision-limited members. The [D4 plan](bayesfilter-modern-hmc-n5-method-proposals-2026-10-10.md)
records the unchanged science contract; the
[local reporting repair](bayesfilter-modern-hmc-d4b-reference-repair-2026-10-11.md)
preserves the original failed diagnostic and assesses all nine MAMS members in
a new output directory. The refreshed twenty-cell inventory, repaired report,
provenance and budget ledger are in `N5/method-reference-repair-01/`.
Before D4R4b, the settled campaign balance was 151019.616 aggregate
worker-seconds after the existing overhead reserve, bounded D4R3 charge and
conservatively settled D4R4a cost preflight. After settling both D4R4b GPU
rungs and the first full-model consumer deployment, **135145.833 seconds
(37.54 aggregate worker-hours)** were unallocated, with the local 3600-second
reserve still held. Both GPU runs are terminal. A corrected deployment now
has also completed in 1859.353110 seconds. That historical settlement released
unused reserves and left **133819.625 seconds (37.17 worker-hours)** unallocated;
see `preconfirmation-01/settlement-03.json` and the consumer plan. No worker
from that deployment phase remains active. The October 13 continuation above
has since assigned that balance to terminal pricing/MAMS charges, the held
local-check and precision-study reserves, and the now-settled fixed-HMC ceiling.
It is not an additional unallocated balance and cannot fund another campaign.
The other 45 D4a proposals, fairer classical
search, independent paired-seed replications and broader nonlinear validation
remain open. The current development and validation phase is **D4R, generic trajectory adaptation**,
defined in the [reviewed repair subplan](bayesfilter-generic-trajectory-adaptation-repair-2026-10-11.md).
The completed MAMS cohort verified a frozen nine-point grid with pilots,
refinement and repair disabled; it did not execute section 6.3's MAMS length
adaptation. D4R separates requested physical travel from epsilon, implements
discarded method-specific proposals and frozen-scale exploration diagnostics,
then tests them across analytic and state-space targets. All verified members
remain retained; no R-hat or posterior-precision gate is added to tuning.
D4R1 is implemented: explicit duration compilation, discarded MAMS proposals,
fixed-policy diagnostics and versioned duration-preserving repairs. D4R2 has
recovery and both public consumer routes; Gaussian, LGSSM and nonlinear
accumulation mechanics pass. The [execution audit](bayesfilter-generic-trajectory-adaptation-execution-2026-10-11.md)
tracks the remaining analytic/observable tests and D4R3–D4R4 confirmation.
These engineering checks establish neither mixing nor a robustness gain.
The final isolated regression suite passed 99 tests. A public Gaussian CPU
smoke retained and assessed all three verified policies. The
[bounded D4R3 result](bayesfilter-mams-adaptation-feasibility-result-2026-10-12.md)
now records the completed eight-target GPU/XLA run: all 25 candidates were
verified and assessed, 21 passed the intentionally short local posterior
checks, four remained incomplete, and all four mixture members missed a mode
despite passing local checks. Seven mutation/oracle tests and 35 independent
adapter tests passed. There is no method ranking, global-coverage claim or
release promotion. The execution audit records the exact command, source
snapshot, hardware probe, charged 722.673-second supervisor run, and remaining
confirmation requirements. D4R4 remains open: independent references,
model-coordinate observables, paired baselines and priced bounds must be frozen
before a posterior launch. D4R4a priced all four full models in two bounded
attempts. Refresh D4c/F from those measured per-member costs; preserve all
unlaunched denominators. N6 LAPS and conditional N7 alternatives remain
separate method-development branches, followed by N8 consumer/book/release
review. No method ranking, exact nonlinear truth, optional release or default
promotion follows from this one-seed SGQF cohort.
The [D4R4a result](bayesfilter-mams-d4r4a-full-model-pricing-result-2026-10-12.md)
closes full-model cost preflight: the shared discarded MAMS proposer completed
two short prices on each of PP-SGQF, PP-UKF, SIR and structural UKF; all eight
short paths passed numerical health. The first committed snapshot omitted
SIR's paired condition diagnostic, causing a proper preparation veto; the
scoped source repair and all four isolated full-horizon price smokes passed.
The bounded SIR/structural retry finished with total GPU supervisor charge
1,191.476 seconds across both attempts. No model delivered a reliable learned
length from the two 64-draw diagnostic windows. These are cost-only rows from
two recorded source commits, not a same-source posterior or method comparison.
The old strict posterior reference loaders retain their historical source
checks. D4R4b has now completed a separate versioned compatibility audit for
all four references against `source-163d24b34`: source differences were
inspected, full-horizon values/scores and quantity charts passed parity, and
the final integration rungs passed reaggregation. Saved local factors remain
unadmitted warm-start hypotheses. Finish the longer discarded-window ladder
before freezing an adequately searched classical/MAMS cohort, model
quantities and paired uncertainty design.
The [D4R4b preconfirmation plan](bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md)
now makes those first dependencies executable: a versioned reference-source
audit and an all-model 256/64 diagnostic-window rung, with a conditional
1024/256 rung preserving original target seed indices. It suballocates 3,600
local-check seconds and two 7,200-second GPU allocations from the settled
balance. The first rung has now settled at 4144.142 seconds, releasing
3055.858 seconds. The structural second rung settled at 7200.435 seconds,
including its measured supervisor overrun. It retained sixteen audited pilot
calls and usable estimates at its first duration, but did not finish the
second duration or finalize a proposal cohort. No structural price or
confirmation is fabricated from these partial results.
The cost consumer gains explicit window/lag/seed-index arguments, with unchanged
historical 64/16 defaults; it remains a caller of the shared MAMS proposer.
The [execution record](bayesfilter-mams-d4r4b-execution-2026-10-12.md) records
the completed 165.772-second CPU reference check and audited all-model GPU
rung. PP-SGQF, PP-UKF and SIR each have usable source and slow-observable
estimates and four healthy prices. Structural has two healthy prices but
neither estimate, triggering the planned structural-only 1024/256 rung.
All fourteen prices pass the terminal source/seed/work audit. This does not
close posterior validation. The older D4b method-fit consumer cannot consume
the new proposal schema, epsilon domain or repair descendants; update its
consumer contract and reference bridge before a fresh comparison.
The [adaptive consumer repair](bayesfilter-modern-hmc-adaptive-consumer-repair-2026-10-12.md)
now supplies a common frozen-cohort caller for the existing public tuners.
The MAMS consumer uses it, and 24 focused CPU mechanics tests cover both
coordinate routes, epsilon above one, an actual duration-preserving repair
child, fixed HMC, budget stops and state-space integrations. The new explicit
full-model driver is now implemented with eighteen additional regressions,
including all four full-data preparations, repair-descendant inventories,
posterior peer continuation and supervisor interruption. All four references
passed a fresh compatibility audit against committed `935e9d719`. The GPU/XLA
deployment from that immutable source completed eight healthy trials in
929.207 seconds but retained no member: its reduced trial budget made passing
the statistical rule mathematically impossible. This is a planning failure,
not a R-hat rejection or evidence against MAMS. The consumer subplan now
requires a model-independent attainability preflight using the actual
measurement and verification statistics before full-target preparation.
The complete priced baseline/replication design and feasible fresh
verification/export remain; these consumer checks do not close D4R4
confirmation. The structural run's stale running manifest was explicitly
reconciled in a read-only audit, and future supervisor finalization is repaired.
Ten reporting regressions and a rendered two-chapter guide build pass. Known
local diagnostic use before the attainability repair is 2313.645604 of the
reserved 3600 seconds. Failed source-invalid and timed-out CPU attempts remain
charged and preserved. The guide's exact antithetic AR(1) calculation explains
why lugsail need not be conservative; it changes no numerical default.
The attainability repair is implemented and all 59 focused regressions pass.
The archived deployment's eight trials and 8192 work units pass the read-only
source/seed/trace audit. The corrected sixteen-trial-per-stage GPU check has
completed fresh verification and checked export/reload of its one verified
member. Its independent terminal audit passes 190 source/input hashes, all
32 trial streams, native traces and 32768 charged work units. Its question
was deployment, not mixing or a method ranking. Known local diagnostic use
settled at 3066.854670 seconds; the unused reserve has been released.
The plan requires a fresh priced comparison design and stopped-chain/MCSE
calibration after settling that deployment; neither is silently replaced by
this short engineering fixture.
The four references now pass fresh compatibility against `e68cdf1c4` in
`reference-consumer-05`, after two preserved source/metadata-reader failures.
The scientific-floor sensitivity table shows why 32 repetitions cannot be
assumed adequate: near-constant .70 acceptance requires the 64-trial rung
under the inspected allowances, and .65 can require 128 when repairs reserve
more candidate error budget. The next priced comparison must include those
scheduled-rung and fresh-verification costs; a mathematical attainability
check alone is not a power calculation. The corrected GPU deployment has
completed normally in 30.99 minutes. Thirty-three reference-reader tests
also pass; 55 selected consumer tests were repeated successfully from the
isolated committed source. The [terminal consumer result](bayesfilter-modern-hmc-adaptive-consumer-repair-2026-10-12.md#terminal-result-and-next-phase)
records the decision, limits, budget and complete next-phase requirements.
The next deliverable is an affordable, explicitly priced paired comparison
with adequate classical searches and model/tail posterior checks, plus the
remaining MCSE/stopping calibration. This successful deployment does not close
the original MAMS mixing/precision question or establish a release.
The [priced continuation](bayesfilter-modern-hmc-priced-comparison-2026-10-12.md)
now makes that next step executable. It freezes eight classical fixed, eight
randomized, six SNAPER and four actually adapted MAMS proposals on full SGQF,
reserves one repair per family (duration-preserving for randomized trajectories,
supported fixed-count repair for fixed HMC), and schedules 32/64/128
independent trials in each fresh qualification stage. All four deterministic
attainability preflights pass; this is not a power result. A 7200-second GPU
price allocation and 3600-second local allowance are reserved from the settled
37.17 hours. Remaining allocations must follow current all-member prices,
including validation/export and per-trial overhead sensitivity. The complete
twenty-cell comparison is not presumed affordable; unfunded cells and paired
replications remain explicit. No new numerical default is introduced.
The [MCSE/stopping diagnostic](bayesfilter-mcse-stopping-calibration-result-2026-10-12.md)
is complete: all 128 replications in each of six stationary Gaussian/Student-t
conditions, with four estimators and fixed versus first-passing endpoints.
The bounded first attempt completed 717 records; a checked continuation
preserved every completed record and finished the remaining 51. Total cost
is 1940.794223 CPU-worker seconds within the same 3600-second local reserve.
The full audit passes all 768 records, 384 independent latent four-chain units,
77 numerical source hashes and cumulative accounting. The exact Gaussian
calculation confirms that the unmasked configured lugsail variance estimate
can be downward biased under antithetic dependence (about 10.6% at rho=-.9,
10000 draws). Two fixed-endpoint Student-t estimates were unavailable and
remain counted. This closes the planned reference experiment, not estimator
calibration in general: no replacement default, stopped-coverage guarantee,
warm-up or sampler-correctness claim is supported. Further estimator work
requires a specified repair, fresh fixed-count replication and downstream HMC
checks; it must not revise the frozen MAMS posterior criteria after seeing
results.
Fourteen additional deterministic CPU/XLA regressions now compare the actual
batch estimator with exact covariance quadrature, preserve the antithetic
counterexample and test the negative-chain pooling guard. They pass against
the immutable source, bringing this phase to 78 distinct focused regressions.
The terminal-pricing continuation is now implemented and has passed 53 focused
receipt, budget, launch-boundary and MCSE-accounting tests. It checks the full
design when reusing prices after the narrow fixed-HMC repair correction and
preserves all roots, native receipts and statistical denominators. The completed
fixed-HMC cost scenarios already exceed the remaining balance for the full
comparison. After all prices finish, the continuation writes a bounded
allocation for the unchanged adapted MAMS cohort from the actual remaining
budget. The conservative minimum-count scenario remains a cost diagnostic;
it cannot veto a useful partial attempt. The consumer preserves every planned,
unfinished and delivered identity, with fresh qualification and unchanged
posterior/reference checks. Actual budget exhaustion prevents launch. A
forecast larger than the balance makes incomplete coverage an explicit risk,
not a claim that no execution is possible.
This is the previously declared scoped feasibility branch; broader methods,
models, paired replications, statistical confirmation and release remain open.
The original waiting continuation service was stopped before any child launch
to repair that planning veto; `continuation-01/settlement.json` records zero
child cost. The replacement uses the fresh `continuation-02` directory and
`bayesfilter-modern-priced-continuation-20261012-02.service`. Pricing is now
terminal: all 26 roots and 52 native calls completed at 13:56 UTC on October 12,
charging 6851.941203 GPU-worker seconds. The terminal audit passed source,
receipt, common-start, stream and memory-policy checks, with no numerical-health
veto in the short calls. The continuation recorded a 123019.624951-second
(34.17-hour) allocation and launched the immutable MAMS consumer on host GPU 1.
Its manifest confirms memory growth, TF32 and XLA; fresh public qualification
is active. Five 64-transition measurement trials across four chains were
checkpointed by 14:09 UTC; no candidate was verified at that observation.
All four roots and four reserved repairs remain in the design.
The 39.60-hour minimum-count forecast exceeds the allocation; this is bounded
feasibility with possible incomplete coverage, not a promised completion time.
Thirty-three focused budget/report tests pass, including
launch under an oversized conservative forecast, actual exhaustion and partial
child outcomes. Broader method/model comparisons and fresh replications remain
unfunded; no candidate has acquired posterior or release status from pricing.
The [priced execution record](bayesfilter-modern-hmc-priced-execution-2026-10-12.md)
tracks the completed MCSE study, terminal prices and active MAMS qualification.
The [analytic-control subplan](bayesfilter-mams-analytic-controls-2026-10-12.md)
has been executed within the existing local allowance. Its audit found and
repaired an incorrect inherited partial-funnel model map and different model
scales in the exact/partial fixtures. Forty-two distinct regressions and a
384-positive/192-negative reference calibration are recorded in the
[result](bayesfilter-mams-analytic-controls-result-2026-10-12.md). No negative
control falsely passed, but strong antithetic chains often lacked usable
lugsail MCSE; 32 replications per condition cannot establish nominal coverage.
A fresh CPU graph public-pipeline diagnostic has also finished: all six
Student-t/exact-funnel/partial-funnel members verified, exported/reloaded and
passed the short posterior checks, while one Student-t member raised a
separate second-moment reference alarm. Every tuning member remains retained.
The remaining D4R3 work is the estimator/stopping repair motivated by the
completed reference study and GPU/XLA confirmation, not another implementation
of the corrected map or tail checks. Historical feasibility rows are not
reinterpreted.

The bounded full SGQF MAMS continuation is now terminal. Pricing, launch,
qualification, export/reload and posterior assessment completed in
`N5/generic-adaptation/priced-comparison-01/continuation-02/mams-run/`. All four
frozen MAMS candidates were retained and verified; each passed warm-up,
retained-sample precision, independent finite-target agreement and numerical
health checks. The run used 2,000 warm-up transitions per chain and between
3,000 and 10,000 retained transitions per chain. The terminal audit checked 197
source/input hashes and 288 qualification calls. It charged 40,403.448742
worker-seconds and left 82,616.176209 seconds of that child allocation unused.
No candidate was nominated, no method ranking was computed, and release remains
false. This closes the scoped SGQF MAMS feasibility phase; it does not close the
master program.
Historical status snapshots below are dated records, not active instructions.

**Owner scope correction, 10 October:** NUTS is excluded, including diagnostic
comparators. The active matrix has twenty cells: five core models times fixed
HMC, randomized HMC, SNAPER-proposed HMC and MAMS. Previous NUTS results and costs
are historical and do not define remaining work. The
[exclusion repair](bayesfilter-nuts-exclusion-repair-2026-10-10.md) records
execution enforcement and the source-preserving current inventory. This scope
supersedes contrary earlier phase descriptions.

The recommended first delivery is randomized-trajectory HMC with optional
ChEES/SNAPER adaptation, followed by MAMS. LAPS follows a qualified MAMS
implementation and an actual measurement of ensemble cost on our state-space
targets. MALT is a conditional alternative, while target-evaluation improvements
can proceed independently. MEADS, delayed-rejection GHMC and particle methods
have explicit later entry conditions rather than being implicit promises of
the first release.

The current tuning **procedure** applies: prepare, propose, freeze, measure,
verify independently, retain every verified candidate, then assess posterior
equilibration and precision separately. Its present **fixed-epsilon/fixed-L
representation, acceptance policy, execution adapter and some tests do not
apply unchanged**. New methods must extend the shared procedure, not introduce
competing tuners or quietly reuse an incompatible HMC artifact.

This is an implementation plan, not another tuning guide. The official
exposition remains [the monograph](../main.tex), especially
[the modern-methods chapter](../chapters/ch26g_modern_hmc_methods.tex), and the
agent-facing instructions remain [the interface reference](../reference/hmc-tuning-interface.md).
The [literature/source review](../reviews/bayesfilter-hmc-methods-literature-survey-2026-10-07.md)
and [mathematical audit](../reviews/bayesfilter-modern-hmc-monograph-audit-2026-10-07.md)
provide the scientific starting point. The
[existing HMC repair master](bayesfilter-hmc-repair-master-program-2026-09-16.md)
continues to own existing-method repairs and release evidence. Its unfinished
requirements are not closed by this plan or inherited as evidence for a new method.

## 1. Decisions and applicability

| Component | Implementation decision | Tuning decision | First usable result |
| --- | --- | --- | --- |
| Randomized-length corrected HMC | Add a frozen, state-independent length distribution to the existing exact-score kernel | Tune epsilon and distribution parameters; freeze the law, not each realized length | Optional typed candidate through both existing coordinate entry points |
| ChEES/SNAPER | First implement a fixed-geometry adaptation profile using inspected TFP primitives | Adapt epsilon and trajectory scale on discarded pilot draws; freeze and measure fresh candidates | An initializer/proposer for the same randomized-HMC kernel |
| MAMS | Port source-mapped microcanonical subflows and their Jacobian correction to TF; deterministic fully refreshed variant first | Tune epsilon, length law and integrator; add within-path sphere refreshment only after its correction is checked | Optional corrected sampler, initially in whitened coordinates, dimension greater than one |
| LAPS | Port as a discarded ensemble initialization/adaptation procedure feeding corrected MAMS | Its unadjusted bias schedule is a separate adaptation rule, not an acceptance-qualified sampler | Optional initializer after ensemble diagnostics and cost have been qualified |
| MALT | Port the source-defined symmetric stochastic trajectory and path-work correction if residual resonance or repeated rejection justifies it | Tune epsilon, trajectory scale and damping; freeze all three | Optional alternative; no claim that a JAX implementation is already a TF runtime |
| Pathfinder | Separate optional initialization/geometry producer | Tune optimization/approximation choices on initialization evidence, then retune the final sampler | Initial states and frozen geometry; approximate draws do not become posterior draws |
| Equivalent parallel filtering | Optimize the same declared value and total score | Numerical equivalence comes first; changed numerical source requires new sampler scope | Faster target evaluation, if demonstrated on the actual target |
| MEADS, delayed-rejection GHMC | Later specialized branches with full auxiliary-state semantics | Method-specific proposals within the common lifecycle | Only after the need and augmented-state diagnostics are demonstrated |
| Particle-MALA/Particle-mGRAD | Separate latent-path inference program | Conditional-particle controls, not an epsilon/L translation | Not a replacement for a marginalized parameter sampler |

No method becomes the default merely because its paper reports fewer gradients.
No R-hat, posterior ESS, or MCSE threshold can reject, rank, repair, delay or
terminate a tuning search. These diagnostics belong to posterior assessment.
An adaptation objective defined by SNAPER or MAMS may propose settings; that
does not turn it into an admission or posterior-accuracy criterion.
Completed posterior failures may inform a separately versioned development
round with fresh tuning and confirmation evidence. They cannot change
membership, repairs, ranking or stopping inside the frozen search whose
posterior is being assessed. D4R makes this boundary explicit.

## 2. Inspected baseline and concrete implementation gaps

The published survey baseline is commit
`b1ccb8678c1ca68811eaa301bd4e1fd368e4bc23`, previously pushed to main. Planning
also inspected the shared checkout at `70a6d7e96` with substantial subsequent
working-tree changes. In particular, the local execution code/reference include
workload-aware allowance work. Before implementation, reconcile the chosen
baseline with main and record the actual source used. Preserve unrelated dirty
work and running campaigns; do not build release claims from an unidentified
combination of checkouts.

The inspected code has the following constraints:

| Existing location | Reusable behavior | Required extension or limit |
| --- | --- | --- |
| `bayesfilter/inference/tuning_contract.py`, `HMC_TUNING_INTERFACE_CAPABILITIES` | Two public artifact-authority tuners; explicit coordinate/target support | Register each new supported family and its target requirements; fail before work on unsupported combinations |
| `hmc_candidate_set_tuning.py` | Shared cohorts, repair children, independent verification, all-member retention | Candidates currently encode epsilon and leapfrog count; controller proposal/identity logic needs a typed transition specification |
| `hmc_candidate_set_execution.py` | Bound numerical runner, evidence persistence, work reservation | Fixed-step execution and cost estimates do not describe random lengths, several force stages or stochastic substeps |
| `hmc_candidate_set_retained.py` | Checked export/reload and retained bridge | Positional state and fixed HMC settings cannot represent every persistent or ensemble transition |
| `hmc_acceptance_protocol.py` | Explicit fixed-horizon trial estimand, separate error allocations, stream identity | v7 is experimental, currently requires four start weights and strict interior endpoints for both acceptance regions |
| `hmc_acceptance_statistics.py` | Bounded-score uncertainty calculations | Reuse arithmetic only after testing the new method's score, independence unit and one-sided boundary behavior |
| `testing/inference_validation/designs.py`, `procedures.py` | Existing models and pipeline assembly | Designs and member selection assume fixed L; `FrozenTransition` bootstraps from positions each call |
| `testing/inference_validation/engines/invariance.py` | Reference-anchored distributional tests | The two-arm random-position construction assumes a reversible kernel; this is not a universal augmented-kernel test |
| `mcmc_uncertainty.py` | Existing batch-means/lugsail posterior uncertainty | Current across-chain pooling assumes appropriate independence; coupled ensemble rows cannot be pooled as independent chains |

The current v5 policy is operational default, v6 is optional, and the replicated
v7 policy remains experimental. A new-method result cannot claim mature tuning
coverage by calling the v7 helper. This program must qualify its own policy
profile and finish applicable existing statistical requirements before release.

Installed TFP supplies SNAPER and trajectory-adaptation primitives. No integrated
BayesFilter public SNAPER, MAMS, MALT, LAPS or MEADS implementation was identified
in the inspected runtime. The available adaptive-MALT source is JAX/FunMC; the
BlackJAX sources are references for a TF port, not permission to introduce a
second runtime backend.

## 3. Research question, evidence and diagnostic roles

The question is whether these methods reduce the **total cost of delivering
accurate posterior estimates for declared state-space targets**, under the same
target, geometry and accuracy requirements, without making tuning less reliable.
Separate four possible causes of cost: target/score evaluation, trajectory
integration, adaptation/verification, and retained-sample dependence. A method
that addresses one cause need not address the others.

The principal comparator is the current fully tuned corrected HMC procedure,
using every verified member according to a declared downstream allocation rule.
Randomized HMC without learned length adaptation is the simple new-method
comparator. Add the plain proposed method before its enhancement: randomized
HMC before SNAPER, MAMS before LAPS, and serial filtering before parallel
filtering. Give compared samplers the same frozen geometry and model-coordinate
starts so a geometry improvement is not attributed to the transition method.

Separate a kernel comparison from a whole-procedure comparison. In the former,
fixed-length HMC also receives the new explicitly experimental, method-neutral
qualification profile and comparable proposal/evidence budgets. Otherwise removing
an upper acceptance bound only for a new method could manufacture a tuning-cost
gain. In the latter, include the current user-facing HMC policy as an additional
baseline and attribute policy, preparation and sampler changes separately. Existing
artifacts keep their original policy; this matched comparison requires fresh runs.

### Evidence contract and intent ledger

| Item | Required interpretation |
| --- | --- |
| Mechanism under test | A particular frozen transition, adaptation procedure, initializer or equivalent target evaluator, identified separately |
| Engineering pass | Correct target/score/correction, executable public-to-kernel wiring, valid state/RNG recovery, finite and accounted work, schema/replay compatibility |
| Sampler-validity pass | Checked invariant-measure argument, calibrated distributional tests with defect-detection power, and agreement with independent references for the declared target |
| Efficiency promotion | Accuracy-qualified complete-run comparisons with uncertainty supporting lower total cost or a higher probability of delivery within the declared resource cap; no accuracy or applicable heuristic veto |
| Promotion veto | Wrong target or correction, unexplained reference discrepancy, invalid numerical state, supported posterior failure, invalid MCSE, or supported underperformance against the applicable simple comparator in a claimed regime |
| Continuation veto | Invalid harness/reference, corrupted evidence, broken mathematical assumption, missing required evidence, or exhausted overall budget; repair the affected layer before its evidence can be used |
| Repair trigger | Inconclusive acceptance, poor geometry, failed adaptation, candidate failure, repeated rejection, no movement, recurrence, resource contention or compilation difficulty, classified at the correct stage |
| Explanatory diagnostics | Acceptance beyond the declared qualification criterion, adaptation objectives, posterior ESS/R-hat during tuning, energy/work distributions, target latency and batch utilization |
| Nonconclusions | No universal convergence guarantee, automatic burn-in determination, all-model speed ranking, exact nonlinear likelihood claim from a UKF target, or default change from smokes/author benchmarks |
| Preserved result | Versioned design, manifest, raw trial/health records, full candidate dispositions, posterior quantities, independent references and terminal decision note |

The roles are stage-specific. Numerical health is a tuning and posterior veto.
R-hat is reporting-only in tuning but may veto posterior readiness/assessment.
Source-defined moment/autocorrelation estimates guide adaptation; their failure
does not itself establish invalidity of a frozen corrected kernel. A rejected
candidate triggers the next planned repair unless a real continuation veto fires.

For any serious comparison the result must contain both a decision table
(criterion, veto, uncertainty, next action, nonconclusion) and an inference-status
table (hard vetoes, supported ranking, descriptive differences, default readiness,
next evidence). An absence of detected discrepancy is not proof of correctness.

## 4. Common architecture: extend the existing two entry points

Keep `tune_hmc_kernel` for ordinary coordinates and
`tune_fixed_transport_hmc_kernel` for a supported frozen transport. Add an
explicit typed method configuration to their configuration objects. A stage
helper, author-code adapter or bare kernel cannot issue canonical tuning
artifacts. Extend the capability registry and generate its documentation from
the same registry.

The typed interfaces below are the intended full design. The execution record
distinguishes the implemented full-refresh subset from persistent, stochastic
and ensemble extensions. Small modules implement the new responsibilities
without another large branch in `hmc_kernel_tuning.py`.

### Frozen transition specification

A proposed `FrozenTransitionSpec` must bind:

- family, source profile, integrator and composition coefficients/order;
- coordinate chart, map and metric, including the convention for kinetic energy;
- epsilon and complete trajectory distribution, units, randomization topology,
  rounding, minimum length and maximum support;
- momentum/direction refreshment, damping, persistence or slice policy when used;
- exact correction definition and required target/score capabilities;
- auxiliary-state initialization law and full-state checkpoint codec;
- dtype, relevant numerical backend/TF32/XLA policy and numerical dependency closure.

Keep immutable kernel settings separate from the changing chain state. Freeze
the law of a random trajectory, not one realized trajectory. Random lengths
drawn from the frozen law do not constitute runtime adaptation. State-dependent
lengths or refreshment are a different algorithm requiring their own argument.

The tuning evidence scope additionally binds the start bank, its labels/weights,
discarded prefix, measured horizon, independent trial definition and evidence
policy. These are not all properties of the posterior transition itself, but
they are necessary to interpret a qualification result. Runtime allowances and
host contention observations belong to execution provenance; they must not
silently change either the kernel or its statistical estimand.

Legacy `(epsilon, L)` candidates map to the explicit fixed-length HMC profile
with their original semantics. Keep old readers and replay behavior. Do not
reinterpret a legacy result as randomized HMC, change its evidence policy on
reload, or erase source-specific state to make a checkpoint load.

### Proposer, transition and telemetry contracts

A method-specific proposer may use pilot adaptation to emit immutable candidate
specifications. The existing controller owns scheduling, funding, deduplication,
measurement, fresh verification and retention. Repairs create child candidates;
they never edit a measured parent's kernel. Candidate identity includes the
entire frozen policy, not just epsilon and a nominal mean length.

A transition adapter must expose initialization, one transition, checkpoint and
restore of the **actual Markov state**. Position-only state is sufficient for
some full-refresh marginal kernels; persistent momentum, slice variables,
ensemble folds and random-sequence phases must be retained when the algorithm
uses them. Do not bootstrap away persistence at each diagnostic chunk.

Telemetry must distinguish:

- `log_mh_ratio`, acceptance probability and realized acceptance;
- endpoint potential change, Hamiltonian energy change when meaningful,
  deterministic integration work, Jacobian work and refreshment terms;
- finite/domain/target-status checks at the required substeps;
- realized length, truncation, movement and recurrence observations;
- actual value/score calls, batched rows, force substages and wasted masked work.

The field names are proposed. In particular, do not label MAMS's work correction
as ordinary endpoint Hamiltonian energy error. Existing energy-proxy thresholds
need method-specific meaning and calibration before reuse.

### Randomness, restart and work

Use stateless streams with recorded identities for candidate, stage, independent
trial, start/chain, transition, substage and purpose. Separate initialization,
momentum/direction, trajectory length, refreshment and MH uniforms. Record any
intended common randomness, such as one shared length across an ensemble. That
dependence is inside a trial; it does not create additional independent trials.

Checkpoint the next stream position and all auxiliary state. Restored execution
must preserve the specified law and, within a supported identical execution
configuration, reproduce uninterrupted work. Different device/batch reduction
orders need distributional/numerical parity, not an unpromised bitwise guarantee.
Changing a bound execution setting requires the declared compatibility decision.

Reserve a conservative maximum work amount before native dispatch and settle
observed work afterwards. Do not reuse `(L+1) * transitions` for a multistage
integrator, randomized length, delayed-rejection path or ensemble. Include force
reuse, extra endpoint scores, inactive rows still evaluated by a batched target,
and auxiliary/reverse proposals. Charge attempted work lost in a process failure;
retrying the same trial never creates another independent observation.

Implement typed specifications and state adapters first for existing HMC. Public
API-to-kernel wiring tests must demonstrate that both coordinate entry points,
export/reload, posterior runners and validation adapters actually call the shared
implementation. The existence of a new kernel class alone is not integration.

## 5. Tuning policy for corrected methods

### The shared lifecycle

1. Check the target, its score, supported coordinates and geometry. Prepare and
   freeze geometry before the first fixed-geometry profile. Use fresh verification
   starts representative of the declared application, with coordinate roundtrips.
2. Declare the finite primary candidate families, repair options, candidate cap,
   budgets, acceptance policy and trial design. Broad coverage now means several
   length scales/distributions and method controls, not only integer L values.
3. Run bounded, discarded pilots. ChEES/SNAPER, dual averaging or MAMS source
   adaptation may nominate settings. Every funded primary family receives its
   planned opportunity; the first successful pilot cannot stop the cohort.
4. Freeze each complete candidate. Measure it from the frozen start bank with
   fresh trial randomness, independent of its pilot. Changing the length law,
   metric, epsilon or refreshment requires a new candidate and new evidence.
5. Independently verify every measurement survivor, using reserved fresh streams
   and the declared verification error allocation. An incomplete verification is
   an explicit pending/inconclusive outcome, not an exported verified member.
6. Repair and refine all eligible families under the common bounded scheduler.
   Epsilon bracketing may propose hypotheses; nonmonotone evidence is retained.
   Never transfer acceptance evidence across trajectory laws or other controls.
7. Export **all verified candidates**, with no nominee or acceptance-distance
   winner. Retain failed, deferred and inconclusive records with their reasons.
8. Use new posterior streams. Discard all adaptation and tuning draws, assess
   equilibration separately, then grow retained sampling for declared quantities.
   Tuning success does not imply posterior readiness.

If many verified members are available, downstream testing/production must state
how it allocates work: all members in a funded comparison, a predeclared balanced
subset, or a deterministic representative without a superiority claim. Optional
efficiency comparisons occur after qualification with separate evidence. They
cannot retroactively remove other verified members or select a benchmark winner
using the same confirmation outcomes being reported.

### Acceptance targets and qualification are different

The ordinary 0.65–0.75 preferred region must not be imposed on every new method.
The installed SNAPER helper uses a dual-averaging target of 0.8; the MAMS paper
uses 0.9 in its practical tuning and reports 0.99 for a difficult funnel example;
LAPS reports targets depending on integrator order, including 0.7 and 0.9.
These are attributed proposal hypotheses, not universally justified defaults.

For the new corrected profiles, implement an explicit **one-sided qualification
option**: require sufficient evidence that each declared start's mean acceptance
probability exceeds a declared minimum, with admissible upper endpoint 1.
High acceptance alone is not a validity failure. A high-acceptance candidate may
spawn a potentially cheaper sibling while remaining retained if verified.
Movement, short-cycle, finite-state and correction checks remain separate.

This requires a new versioned policy: current v7 validates both endpoints strictly
inside (0,1). It is not achievable merely by passing `(minimum, 1.0)` to the
existing config. Preserve existing v5/v6/v7 meanings. Extend bound inversion,
decisions, artifacts, replay checks and boundary tests together. Do not promote
this policy globally or select an arbitrary minimum to make a run pass.

The minimum is an operational usability requirement, not a theorem about mixing.
Its provenance must be a stated application requirement or a priced calibration
decision made before confirmation. The preferred optimizer target can differ.
If no defensible floor is available, record that policy choice as unqualified;
do not disguise an inherited 0.7 target as a scientific necessity.

### Independent finite-horizon evidence

For corrected kernels a reusable score is

\[
 S_{rj}=\frac{1}{T}\sum_{t=1}^{T}\alpha_{rjt},\qquad
 a_j=\mathbb E[S_{rj}\mid\text{frozen kernel, start bank and initialization law}].
\]

Here r identifies an independent complete trial, j identifies a start, and the
fixed discarded prefix W precedes the T measured transitions. The acceptance
probability includes the correct method-specific MH ratio. Scores are bounded
in [0,1]; serial dependence within a trial and dependence between starts inside
one trial are allowed. Independence is required across the complete trials used
by the chosen evidence calculation. For augmented kernels, reset the auxiliary
state from its declared law as well as the positions.

This is a conditional fixed-horizon expectation, **not stationary acceptance**
or an estimate of sufficient burn-in. Extend evidence by adding independent
trials while W and T remain fixed. Increasing T or changing initial states
changes the estimand and needs a new scope. Consecutive chunks are not new trials.

Use simultaneous bounds for all required start expectations and the declared
candidate/inspection family. Search and verification have separate allocations;
do not claim their sum or product is a stronger theorem without a derivation.
Adaptive proposers must use fresh evidence after freezing; the statistical
argument must remain valid conditional on the past. If verification informs a
repair, that child gets fresh measurement/verification and its own allocation.
Prove the bound's applicable independence/filtration conditions, then calibrate
null behavior and power; simulation alone does not prove sequential coverage.

Make evidence affordability calculable before launch. For example, for independent
bounded trial scores, a two-sided Hoeffding interval at one predeclared look has
half-width `sqrt(log(2 / eta) / (2 * n))`. For K candidates, J starts and H fixed
looks, the simple conservative allocation `eta = alpha / (K * J * H)` gives
`n >= log(2 * K * J * H / alpha) / (2 * w^2)` to achieve half-width at most w.
This is a planning bound under its assumptions, not the only allowed estimator
or an assertion of power at the qualification boundary. A true mean separated
from the floor by more than `2 * w` can be qualified on the simultaneous coverage
event; a boundary mean may remain inconclusive. Price this number of complete
trials. If it is unaffordable, revise the prospective precision/candidate design
or qualify a better bounded-score method; do not count correlated transitions
as n or widen a criterion after seeing the confirmation data.

Generalizing the four-start v7 contract is a separate explicit schema change,
including weights and multiplicity. The first independent-chain profile may
retain four starts for compatibility, labeled as an inherited design choice.
Do not squeeze an ensemble of hundreds of dependent rows into a fictitious
four-chain independent test. An ensemble trial can instead produce declared
bounded summary scores with their own tested policy.

## 6. Method-specific implementation and tuning recipes

### 6.1 Randomized-trajectory HMC

Start with the existing exact-score, fixed-geometry, MH-corrected HMC kernel.
Independently sample a trajectory parameter from a frozen distribution before
each proposal. A state-independent mixture of invariant kernels is invariant;
randomizing the length is not a license to stop a path using an uncorrected
state-dependent rule.

For the first explicit IID profile, propose continuous uniform duration
`tau ~ Uniform(0, tau_max)` with `L = max(1, ceil(tau / epsilon))`. Record this
choice as a simple, testable design hypothesis, not the only valid distribution.
The induced integer law is generally not uniform because of rounding. Require
its support to fit the declared maximum integration count; reject incompatible
settings rather than silently clipping, redrawing or renormalizing. Other laws,
including a capped law, are separate explicit profiles.

Use stateless independent lengths per chain in this first frozen profile. A
shared batch length is another legitimate candidate law when defined correctly,
but changes dependence and hardware cost and must be separately identified.
Implement bounded TensorFlow loops with stable shapes. A masked row that still
incurs target work counts toward cost; masking must not move the inactive state.

Tune epsilon independently for each declared duration scale. Use a broad finite
set around a geometry-based/pilot scale, refined around every survivor. The
actual scales and count are priced design choices, not a new universal L grid.
Keep fixed-length controls at resonant and nonresonant scales. Verify length
histograms against the induced law, randomized-mixture invariance, proposal and
MH stream separation, maximum-work accounting and interrupted replay.

### 6.2 ChEES/SNAPER

Implement the source-defined squared-jump objectives and endpoint-derivative
adaptation as **proposers**. ChEES uses the squared radial observable; SNAPER
uses `f_w(q) = [w^T(q - mean)]^2` and a cost-normalized jump criterion along an
estimated principal direction. The chapter's `eq:bf-modern-snaper` and
`eq:bf-modern-snaper-derivative` state the definitions and derivative limits.
The endpoint derivative is not the exact derivative of the discontinuous,
rounded-and-rejected finite program. Preserve the inspected estimator and
acceptance treatment; do not silently replace it with autodiff through a ceiling.

The first profile is **SNAPER-style fixed-geometry adaptation**. Prepare and
freeze the ordinary metric, or preserve the supplied transport's identity latent
metric. Reuse/port the installed trajectory criterion and adaptation primitives
without enabling another internal mass adaptation. TFP's full SNAPER kernel
adapts a diagonal metric and trajectory length; its helper adds epsilon adaptation.
Calling that helper unchanged would duplicate ordinary geometry work and violate
the first fixed-transport profile. Label the local fixed-geometry profile as an
adaptation, not an exact reproduction of the full helper or paper algorithm.

An optional future native TFP-SNAPER preparation profile may own its diagonal
metric adaptation, but then must freeze and bind that geometry before common
measurement. It is a separate source-mapped mode; only one component owns the
metric. Do not introduce it merely to avoid implementing the clean first profile.

Use discarded pilot windows to estimate the center/direction, adapt epsilon
with the declared source-based target, and adapt trajectory scale. Preserve
the adaptation trace, caps and final averaging/freeze semantics. Freeze the last
declared averaged settings and evaluate a finite neighborhood in epsilon/scale;
all siblings need fresh measurement. Stable seeds, conservative pilot caps and
bound-hit diagnostics are required. A pilot hitting its cap can still emit a
clearly labeled frozen hypothesis; qualification decides whether it is usable.

Installed TFP trajectory adaptation defaults to **Halton jitter**, with a shared
scalar length. Do not silently transfer that behavior into the IID profile.
Explicitly configure the intended jitter and write parity tests for the chosen
profile. Supporting a deterministic low-discrepancy sequence later requires its
phase in checkpoints and a correct time-dependent/augmented-state interpretation.

Adversarial tests include one slow eigenvector hidden among many easy dimensions,
near-tied covariance eigenvalues, collapsed starts, rotated covariance, singular
empirical covariance, and nonlinear curvature. An attractive objective value
must not hide a stuck mode or become a posterior precision claim.

### 6.3 MAMS

Use whitened coordinates initially, dimension `d > 1`, and unit direction
`u = normal / norm(normal)`. Implement the force subflow and a source-mapped
symmetric second-order composition before a separately identified higher-order
composition. Each composition needs its actual coefficients, force-evaluation
count and negative-substep behavior checked against the pinned source.

For potential U, `e = -grad(U) / norm(grad(U))`, and a force-substep duration h,

\[
 D=\cosh\delta+(e^\top u)\sinh\delta,\quad
 \delta=\frac{h\|\nabla U\|}{d-1},\quad
 \log J=-(d-1)\log D.
\]

Accumulate the actual substep work `A = sum((d-1) * log D)` and use
`log_mh_ratio = -U(q_new) + U(q_old) - A`, with the source-defined reversal and
rejection convention. The force-map and surface-measure derivations are in the
chapter's `eq:bf-modern-force-map`, `eq:bf-modern-force-jacobian` and
`eq:bf-modern-mams-acceptance`. Ordinary HMC endpoint kinetic energy is wrong for
this correction. Stable hyperbolic/log arithmetic, zero force, near-parallel and
antiparallel directions, and overflow margins are first-class numerical tests.
Clipping or renormalization that changes the map needs an explicit analyzed
profile; silent numerical repairs cannot be assumed measure-preserving.

First deliver the deterministic trajectory with a fully redrawn unit direction
per outer proposal, as supported by the inspected adjusted-MCLMC source. Do not
invent cross-proposal persistence for this variant. Then implement the stochastic
within-path sphere refresh variant using the source's path correction and noise
conventions. Full outer refresh and partial inner refresh are distinct controls.

Tune epsilon, physical duration and integrator separately; then add the sphere
refresh scale for the stochastic variant. Locally call duration `tau` and reserve
`L` for the integer integration count: the paper also uses L for physical length.
The paper's whitened starting duration proportional to `sqrt(d)` and practical
acceptance target 0.9 are source hypotheses, not transferable HMC scales or final
settings. Use its autocorrelation-based adaptation only to nominate frozen
duration candidates. Apply common independent qualification to each full policy.
Freeze the preconditioner/transport and all refreshment parameters.

Test sphere norm and surface Jacobians in tangent coordinates, inverse subflows,
accumulated work and the low-step limit. Add stationary sphere refresh tests,
random-direction independence, and path reversal for the stochastic variant.
Mutating the Jacobian sign must fail an analytic/distributional oracle. Reject
d=1 explicitly; unsupported cases may use an explicitly chosen HMC method but
cannot silently turn a MAMS artifact into an HMC artifact.

### 6.4 LAPS

Treat LAPS as an ensemble warm-up method followed by corrected sampling, not as
a sampler with an acceptance probability of one during its unadjusted stage.
Separate unadjusted initialization, diagonal reparameterization, adjusted
adaptation, freezing, fresh qualification and posterior assessment in the state
machine. Discard every unadjusted or adaptive draw from posterior estimates.

Before coding, resolve a concrete source choice. The inspected paper uses an
error schedule expressed through `F(C * estimated_bias)`; the inspected BlackJAX
source uses `C * bias**(3/8)` with the paper expression commented out. The paper's
centered equipartition statistic and the source's uncentered statistic differ.
The source's fluctuation normalization and epsilon-freeze flag also need explicit
mapping. Check the paper algorithm's printed stopping inequality against its
surrounding text and the implementation. Port one named profile first, with every
equation and source operation identified. Do not create a hybrid and call it
faithful. Evaluate an alternative schedule as a separate candidate only if needed.

Tune the declared initial error scale, schedule controls, trajectory/refreshment
parameters, adjusted epsilon target and ensemble size under a bounded pilot.
Moment statistics can trigger a stage transition or an extension under that
profile; they cannot certify sufficient burn-in. A finite cap may move healthy
states to adjusted warm-up with the reason `unadjusted_cap_reached`. Nonfinite
states, invalid denominators or failed target evaluations require explicit
repair/reset; they cannot be hidden by the transition to the next stage.

Construct tests in which the moment diagnostic is misleading: shifted Gaussian
means with the correct centered variance, wrong off-diagonal covariance,
equal-variance distributions with different fourth moments, one-mode starts in
a mixture, heavy tails, near-zero denominators and shared random streams.
Test the centered and uncentered profiles against their actual formulas; the
mean-shift counterexample for one must not be wrongly attributed to the other.

Distinguish two experiments:

1. **Kernel qualification:** take the frozen corrected MAMS policy produced by
   adaptation and qualify it with independently reset trials. This answers
   whether the frozen policy meets tuning requirements on the declared starts.
2. **Initializer benefit:** repeat the entire LAPS ensemble procedure independently
   and assess downstream posterior delivery, including all adaptation cost.
   This answers whether LAPS helps initialization on that target.

Shared adaptation can correlate ensemble endpoints; freezing does not erase
that dependence. The independent statistical unit is a whole independently
started ensemble. Do not multiply the replication count by its chain count or
pool its rows using the current independent-chain MCSE formula. Initial support
can continue one predesignated endpoint from each independent ensemble for
ordinary independent-chain assessment, while retaining all other draws as
diagnostics. Using all ensemble rows for posterior inference requires a tested
cluster-aware variance/assessment path. Report total ensemble cost in either case.
Nested diagnostics cannot rule out bias common to all ensembles.

Do not import the paper's 256/4,096-chain choices as defaults. Measure target
batch latency and memory first. The source reports major gains in per-chain work,
but total ensemble gradients and end-to-end cost answer different questions.

### 6.5 MALT and later alternatives

MALT is the next sampler branch if qualified randomized HMC/MAMS still show a
specific resonance or rejection problem. Port the published symmetric splitting
and the inspected adaptive-MALT source into TF, with a named profile where they
differ. Use Gaussian OU refreshment with frozen damping, and accumulate **only
the deterministic integration energy errors** for the MH path correction.
Endpoint total energy includes OU heat and is the wrong correction. Tune epsilon,
duration and damping through discarded source-based adaptation plus bounded
frozen candidates. Check zero-damping and strong-refresh limits and rejection
conventions. MALT's full trajectory-boundary refresh does not by itself require
cross-trajectory momentum persistence; retain whatever the selected profile uses.

MEADS requires momentum/slice/fold state and a careful joint invariant law.
A first freeze-after-adaptation profile is a local variant and must be labeled
as such. An online folded-adaptation version is a separate task with product-law
and ensemble-diagnostic tests; common tuning success cannot establish it.

Delayed-rejection GHMC is considered only if measured state-dependent stiffness
persists after supplied whitening and ordinary refinements. Derive and test the
forward/reverse rejection probabilities, persistent state and ghost-proposal
work. Naively retrying a rejected proposal at smaller epsilon is not the
published corrected method. Tune the staged ladder as one frozen kernel.

Particle-MALA/mGRAD is considered only for a consumer actually sampling a latent
path. Require conditional-particle extended-target, resampling/ancestor and
local-weight tests against smoothing references. Separate particle number and
proposal-scale tuning from the marginal-parameter HMC interface. Generalize the
shared lifecycle if helpful, but do not register it as an HMC epsilon/L profile.

## 7. Target contracts, initialization and state-space cost

For MacroFinance, the inspected representative is the seven-parameter MIDAS
target with a batched masked, correlated-noise Kalman likelihood and its score.
Recheck the actual consumer endpoint before integration. Include parameter
constraints, every transformation Jacobian and the total derivative of the
likelihood actually returned. A dropped likelihood normalization term can alter
parameter inference even when a state-filter mean looks correct.

For dsge_hmc, the inspected BGS description declares `T_source_ukf97` and a
deterministic position-only proposal force, not its asserted exact score. Such a
force can be useful in a separately corrected proposal, but it does not justify
SNAPER score identities, MAMS equipartition/adaptation identities or a claim of
exact-gradient inference. New exact-score profiles must reject it at preflight
until its total score is qualified. Alternatively, a separate force profile needs
its own derivation, registry entry and tests. Do not route it through the
fixed-transport tuner simply because that tuner accepts a callable.

Sampling a UKF/sigma-point likelihood accurately does not remove its approximation
to the nonlinear model. Give every model two explicit identities where relevant:
the scientific generative model and the numerical posterior target sampled.

Profile target value and total score separately and together, including masks,
time horizon, batch size and dtype. For equivalent parallel Kalman filtering,
derive the Gaussian-factor composition with its log normalizers, missing-data
and correlated-noise handling. Compare values, scores, smoothing/forecast
quantities and end-to-end posterior output with a serial independent reference.
Changing the likelihood is not a performance optimization. Do not assume UKF or
QZ stages admit the same parallelization.

Pathfinder may supply dispersed initial states and a frozen geometry proposal.
Check L-BFGS curvature, positive-definite approximation handling, importance
weights and multimodal failure. A successful optimizer or high approximate ELBO
does not establish posterior coverage. Geometry changes require new tuning;
map/initializer draws remain separate from retained posterior draws.

Positive funnel tests use a supplied exact or residual-whitened map. Transport
learning remains an upstream task under the current NeuTra architecture policy;
this program does not revive historical maps or change the owner-selected
architecture. Analytic funnel fixtures are reference maps, not learned-map
quality evidence. Include model-coordinate tail quantities after transformation.

## 8. Test infrastructure and explicit corner cases

Extend the existing `bayesfilter.testing.inference_validation` package. Keep
lightweight deterministic regressions, calibrated statistical tests, actual
public-pipeline integration and research comparisons distinct. Do not run a
large stochastic ladder on every ordinary code change.

### A. Numerical and measure-level tests

| Case | Oracle/check | Defect that must be detected, and interpretation |
| --- | --- | --- |
| Value/score and transformations | Independent closed form or finite-difference reference in FP64; map inverse/forward and full Jacobian | Wrong gradient, omitted transform derivative, wrong likelihood constants; endpoint-corrected invariance alone may not detect a wrong force |
| Ordinary HMC / random lengths | Gaussian linear map, reversibility, MH ratio and exact induced length law | Off-by-one rounding, wrong mixture, reused MH/length stream, hidden clipping |
| MAMS force map | Tangent-space surface Jacobian, inverse subflow, sphere norm, stable zero/large-force limits | Jacobian sign omission, Euclidean determinant substituted for sphere measure, unstable cancellation |
| Stochastic MAMS | Uniform-sphere refresh law, path-reversal ratio and rejection handling | State-dependent unaccounted refreshment or reused substep noise |
| MALT | OU stationary law and summed deterministic work | Including OU energy in MH or silently dropping a deterministic substep |
| Persistent/slice profiles | Full augmented target and transition/reversal law | Position-only restart, wrong slice update or momentum flip |
| Equivalent filtering | Independent serial likelihood/score and smoothing oracle | Wrong mask/cross-covariance/normalizer, batch-only score fork |
| Guard behavior | Healthy no-fire controls and explicit failing domains | Silent ridge, clipping or reset that changes the claimed target/map |

Use NumPy only for explicitly independent diagnostic/reference calculations;
eligible runtime paths remain TF/TFP. SymPy/MathDevMCP algebra checks may support
Jacobians and identities, but the earlier chapter audit is not formal theorem
certification. Probability measures, reversal conventions and source fidelity
still require an explicit argument.

### B. Distributional correctness and mixing are different tests

Use exact/reference draws as independent anchors where available. For a proven
reversible marginal kernel, retain the existing Gandy–Scott random-position rank
construction with randomized ties. For a nonreversible augmented transition,
implement its genuine reverse/adjoint kernel for the backward arm or use an
independent stationarity/two-sample design. Running the same nonreversible kernel
in both arms is not a valid shortcut. A schedule driven by Halton phase also
needs its own correct construction.

Calibrate false positives on known correct kernels and detection power on
activated defects separately. Declare the test family, significance allocation,
replication count and randomization resolution before confirmation. No-op/identity
and deterministic two-cycle controls may preserve the target while mixing badly;
they must pass the appropriate stationarity property and fail the intended
movement/recurrence or posterior-quantity screen. A test that rejects every such
control as an invariance defect is testing the wrong property.

### C. Tuning statistics and adaptation tests

| Tricky case | Required result |
| --- | --- |
| Gaussian full-period return and half-period sign cycle | Acceptance near one cannot certify exploration; squared/radial quantities and cycle diagnostics expose the problem |
| Healthy independent Gaussian proposals with acceptance one | New one-sided policy can qualify them; no artificial upper-band failure |
| Acceptance expectation at/near the lower boundary | Correctly calibrated uncertainty; unresolved cases remain inconclusive at cap |
| Rare rejections or identical observed probabilities | No invented zero uncertainty or claim of stationary acceptance |
| Nonmonotone acceptance across length/epsilon | Preserve contradictory observations and independently test descendants; no evidence interpolation |
| Opposite behavior at different starts | Per-start qualification prevents a favorable pooled mean hiding a failed start |
| Opposite within-chain temporal drifts | Preserve explanatory drift diagnostics; do not make them an undeclared R-hat/acceptance veto |
| Strong temporal dependence | Fixed-horizon complete trials supply replication; chunks and draws do not inflate the sample size |
| Shared trajectory noise or an ensemble pilot | Dependence is recorded inside the independent trial/ensemble; cloned streams are detected |
| Many candidates and repeated inspection | Declared error allocation and candidate cap remain valid after optional proposals/restarts |
| Pilot/verification reuse or repair after verification | Contaminated evidence is rejected; child requires fresh streams |
| Change only R-hat/ESS/MCSE in otherwise identical tuning evidence | Candidate states, repairs, schedule and retention remain identical |
| SNAPER direction collapse, tied eigenvalues or cap hit | Explicit finite diagnostic/fallback hypothesis, no claim of optimal trajectory |
| LAPS favorable moments but wrong mean, dependence, mode or tails | Diagnostic limitation is exposed; posterior assessment with independent references rejects the stronger claim |

### D. Model matrix

The analytic cases test mechanisms. The state-space cases test the application.
Report results by regime rather than averaging an easy Gaussian with a difficult
SSM and treating the mean as evidence for the repository.

| Model/regime | Reference and quantities | Main purpose |
| --- | --- | --- |
| Isotropic Gaussian, resonant durations | Exact draws; means, variances, squared coordinates and return times | Separate acceptance, invariance and exploration |
| Rotated anisotropic Gaussian, one slow direction | Exact covariance and projections onto known eigenvectors | SNAPER adaptation, metric convention, hidden slow direction |
| Supplied exact-whitened and residual-whitened funnel | Analytic target/draws; model-coordinate scale and tail events | Geometry/sampler separation and transformed-score correctness |
| Student-t with declared finite moments; Cauchy separately | Exact CDF/quantiles; bounded events for Cauchy | Heavy tails and explicit rejection of unjustified finite-variance MCSE claims |
| Separated unequal mixture, starts in one mode | Known component weights/density and mode indicators | Common bias, inadequate mode exploration; no global-mixing guarantee |
| Positive parameters and simplex target | Independent transformed density/Jacobian | Constraint errors, boundaries and dimension checks |
| K0 LGSSM location, T=32 | Analytic parameter posterior | Cheapest exact end-to-end state-space test |
| K1 LGSSM interior persistence/noise, T=32 | Independent low-dimensional grid reference | Typical coupled parameter tuning and posterior accuracy |
| K2 near-unit persistence (rho=0.97), T=32 | Independent grid/reference | Slow direction and awkward boundary geometry |
| K3 small observation noise (0.02), T=32 | Independent grid/reference | Stiffness, numerical score and small-step behavior |
| K4/K5 two-noise models, T=16 and T=128 | Declared independent grid/reference with refinement checks | Parameter confounding, weak identification and horizon cost |
| K6 multivariate 18-parameter LGSSM, T=120 | Independent likelihood/score and predictive checks; no current independent joint posterior oracle | Scale, batching and consumer mechanics; cannot certify joint posterior accuracy alone |
| K7 nonlinear SSM, T=16 | Grid for its declared sigma-point approximate likelihood | Nonlinear sampler integration, explicitly not an exact nonlinear posterior reference |
| Additional masked/correlated-noise LGSSM | Independent joint-Gaussian/serial filter value and score; low-dimensional posterior reference | Match the representative MacroFinance likelihood features |
| Small nonlinear latent-state/SV target | Refined quadrature or independently qualified smoothing reference on a tractable problem | Distinguish latent-path inference from marginalized approximate-filter parameter inference |

K0–K7 names and numbers are inherited validation fixtures in
`ssm_campaign_profiles.py`, not universally representative model defaults.
New dimensions, horizons and severity ladders must be justified by the failure
being tested and costed before use. For high-dimensional models without an
independent posterior oracle, report that limitation rather than invent one.
Centered funnels are bounded stress/failure controls; a failure there alone does
not fail the positive supplied-whitening contract. K0 is one-dimensional: it is
a positive HMC/SNAPER pipeline case and an explicit unsupported-dimension test
for the initial MAMS/LAPS profiles. Their positive SSM tests begin with K1 and
other supported multidimensional targets. Do not pad K0 with an undocumented
auxiliary parameter just to make that test pass.

For nonlinear approximate targets, a matched-posterior reference tests sampler
error. SBC against the underlying exact generative model also includes filter
approximation error; diagnose those separately. Where no independent reference
is accurate enough, the corresponding accuracy claim remains open.

### E. Required nonlinear comparison program

Amendment, 7 October 2026, following the user's model-coverage question: the
original matrix put too much emphasis on the LGSSM ladder and a single reduced
nonlinear example. The repository already contains several nonlinear model
families and posterior adapters. They belong in the main comparison program.
Analytic targets establish mechanisms; they cannot substitute for this coverage.
Lack of a closed-form nonlinear posterior is a reason to construct and qualify a
numerical reference, not to omit the model from the program.

A read-only inventory inspected the
[model registry](../../bayesfilter/testing/neutra_model_registry_tf.py), its adapter
factories, the generic nonlinear adapter, model laws and existing reference tests.
This establishes which code and declared targets exist. It is **not** a fresh
GPU/XLA execution audit or a certification of the old registry's readiness labels.
Some registry settings point to historical geometry/results. N0 must reconstruct
current target contracts and obtain eligible frozen geometry; it cannot reuse
historical map or tuning authority merely because a factory still names it.

| Nonlinear case | Existing implementation and target | Comparison role and required checks |
| --- | --- | --- |
| NL-ACC: nonlinear accumulation, Model B | [Generic SSM adapter](../../bayesfilter/testing/simple_nonlinear_generic_target_adapter_tf.py), three sampled parameters, with explicit sigma-point filter routes; K7 is a related reduced two-parameter instance of this family | Required core case. Include K7's tractable posterior reference and the full generic adapter, short and longer horizons. Check accumulated-state dependence, total score and residual geometry. Do not count these as two independent model families. |
| NL-PP-UKF: predator–prey, six parameters | [UKF posterior adapter](../../bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py), nonlinear dynamics with the declared principal-square-root UKF likelihood | Required core case. Check ODE integration/time ordering, parameter correlations, positive-parameter chart/Jacobian, filter status and posterior/predictive quantities. |
| NL-PP-SGQF: the same predator–prey model | [Fixed-level SGQF posterior adapter](../../bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py) | Required core target variant. Compare samplers within this fixed likelihood separately from PP-UKF; the two filters generally define different posterior targets. |
| NL-SIR: epidemic model, three parameters | [SIR SGQF posterior adapter](../../bayesfilter/testing/sir_filter_neutra_target_design_tf.py) | Required core case. Check constrained states/parameters, epidemic-growth and decay regimes, timing, weak identification and predictive epidemic quantities. SIR-UKF is owner-excluded in the registry and is not a comparison arm. |
| NL-STR: nonlinear structural model, five parameters | [Structural UKF posterior adapter](../../bayesfilter/testing/structural_ukf_neutra_target_design_tf.py); the accumulated state includes a squared stochastic component | Required core case relevant to structural macro models. Check persistence/noise/curvature interactions, deterministic-state constraints, full score, and short versus longer observation horizons. |
| NL-GROWTH: standard nonlinear growth, Model C | [Existing model and derivative fixtures](../../bayesfilter/testing/nonlinear_models_tf.py); transition contains a nonlinear rational term and time forcing, with a squared observation function | Required adapter/reference extension for the broader nonlinear release claim. First declare a parameter posterior, priors and chart, for example for the existing process/observation noise scales. Current first-step projection tests are not a complete posterior oracle. Keep a latent-path experiment as a distinct target. |
| NL-SV: actual stochastic volatility | The registry includes a two-parameter SVX-ZC adapter in [the SV target module](../../bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py) | Required inventory and eligibility review, followed by comparison if its current source route and total score qualify. Identify the frozen finite approximation and test tail/volatility quantities. Historical TT settings do not establish current Zhao–Cui source-route eligibility or an exact SV posterior. |

The immediate core comprises NL-ACC, NL-PP-UKF, NL-PP-SGQF, NL-SIR and NL-STR:
five model/likelihood cells from four model families. NL-GROWTH needs an explicit
posterior adapter/reference extension; NL-SV needs a current route-eligibility
check. These are named work items, not silently missing rows. Existing blocked
SVX-SGQF/KSC-UKF or other Zhao–Cui entries remain visible with their actual target
or score problem; a tuning experiment cannot bypass that problem. The failure of
one filter route does not exclude the underlying model when another eligible
route exists. In particular, Model B's structural fixed-SGQF admission test does
not invalidate its separately declared generic sigma-point target.

#### Comparison design

For every eligible core cell, compare tuned fixed-length HMC, simple randomized
HMC, SNAPER-proposed randomized HMC and deterministic
MAMS as those implementations become available. Run corrected stochastic MAMS
and LAPS-plus-MAMS against the same cells when those phases open; LAPS is compared
as an entire initialization-to-posterior procedure, including ensemble cost.
A conditional MALT or later method must cover the nonlinear regimes for which
it claims usefulness. A matrix cell can report unsupported, repair-needed,
inconclusive, failed or completed; absent results cannot be recorded as success.

Hold observations, prior, parameter chart, numerical likelihood, score convention,
dtype and supplied frozen geometry fixed within each sampler comparison. Use the
same model-coordinate starting blocks, correctly mapped into sampler coordinates.
Compare affine preparation with a supplied qualified nonlinear map in a separate
geometry experiment; do not give only the new sampler a better map. N0 declares
scientific quantities, including model-relevant predictions and tail events.
R-hat remains excluded from tuning admission and belongs to posterior assessment.

Use two observation-horizon levels where the adapter supports them and a
nominal plus a justified difficult regime, such as near-boundary persistence,
low observation noise, weak parameter identification or nonlinear accumulation.
N0 chooses actual values from the model's domain and measured prices; do not
apply an arbitrary common severity multiplier across unrelated models. Begin
with the smallest complete nonlinear fit to expose plumbing defects, then
expand to the declared matrix. Preserve family-specific results and all missing
fits rather than letting easy cases dominate an aggregate performance score.

#### References and inference limits

There are three distinct comparisons:

1. **Numerical target and score:** compare the implemented likelihood and total
   derivative against an independently checked evaluator/derivative reference.
   Include transforms, filter normalizers, time ordering and covariance branches.
2. **Sampler output for that fixed target:** use refined quadrature on tractable
   low-dimensional cases and independent, carefully assessed numerical posterior
   references on larger ones. A conditional low-dimensional slice cannot certify
   the full six-parameter posterior. A reference chain must have its own dispersed
   starts, mode/tail checks and MCSE; two HMC chains agreeing is insufficient if
   they share an incorrect target or a missed mode. Compare retained estimates
   and intervals within the predeclared tolerances including reference uncertainty.
3. **Filter approximation to the generative model:** compare the UKF, SGQF or
   other approximation to a qualified quadrature/particle reference separately.
   For example, the existing
   [predator–prey bootstrap-PF reference](../../bayesfilter/testing/predator_prey_bootstrap_pf_reference_tf.py)
   estimates likelihoods at parameter points. It does not by itself supply a
   six-parameter posterior oracle or establish equality with the UKF posterior.
   Record particle error and likelihood-versus-log-likelihood bias correctly.

Closed-form solutions are unnecessary for including these comparisons. However,
a reference-limited case cannot support an accuracy-qualified speed ranking until
its uncertainty is resolved. It can still expose crashes, score mismatches,
failed tuning, resource costs and sampler disagreement, each with its actual
interpretation. A filter-approximation discrepancy is not automatically a sampler
failure; good sampling of an approximate target is not proof of exact-model
posterior correctness. SBC using the generative model must preserve this distinction.

#### Integration, budgeting and completion

N0 records for every nonlinear cell the actual adapter-to-value/score call chain,
current target identity, geometry status, reference, declared quantities and
repair owner. N1 connects those adapters to the common validation design rather
than calling historical private chain runners. N2 includes NL-ACC as an early
public-pipeline smoke; N3/N4 add the other core adapters as each sampler is wired.
N5 must report the complete core comparison matrix. N6 evaluates LAPS on that
same eligible matrix with independent ensemble replications. N8 closes the
broader nonlinear claim only to the extent supported by NL-GROWTH/NL-SV evidence
and documented exclusions.

Add a framework-free matrix-completeness regression that fails if a required
model/method cell disappears, is conflated with another filter target, or lacks
an explicit disposition. Add public tuner-to-retained-assessment integration
fixtures for each core adapter and its relevant failure case. Keep tiny mechanics
fixtures in routine tests; replication, reference construction and realistic
horizons belong to the separately priced campaign. Do not substitute the former
for the latter.

Price complete fits for these nonlinear families before assigning the main
comparison budget. Reduce redundant analytic performance sweeps before dropping
nonlinear application coverage; retain the cheap analytic correctness regressions.
If the authorized cap cannot fund all required comparisons, preserve unfunded
cells and restrict the release claim instead of declaring the nonlinear program
complete. This amendment changes the proposed plan only; it launches no runs and
makes no new claim about current sampler performance.

### F. Public pipeline, recovery and hardware

Each implemented family must have tests that call the public tuner, retain more
than one verified candidate when the fixture supports it, export/reload every
member, run posterior assessment and resume after interruption. Cover ordinary
and supported fixed-transport coordinates, supplied residual whitening and
state-space targets. Unit tests of a kernel method are insufficient.

Inject crashes before dispatch, after charged native work, during record save,
after a completed trial and during posterior continuation. Check source/state
binding, no duplicate evidence, preserved charges and correct RNG continuation.
Mutating the map, metric, length law, auxiliary-state law, source profile or
correction invalidates reuse. Mutating an operational allowance alone follows
the established compatibility policy and does not change numerical evidence.

Exercise ragged lengths, odd batch sizes, tail batches, long horizons, invalid
one-row targets, unavailable gradients and resource contention. Batch-native
targets must give the same rowwise values/scores as their independent reference,
including Jacobian terms. Changing one row must not change another row's target
value except where an explicitly declared coupled target requires it.

Use stable `tf.function` input signatures and bounded tracing. Pure numerical
hot paths target GPU/XLA; incompatibility is a reported qualification gap or
explicit debug/reference exception. No pfor or implicit vectorized Jacobian
route without the required approval. Configure/verify GPU memory growth before
initialization and record placement, dtype, TF32 and allocator behavior. CPU
reference/smoke processes hide CUDA before framework import.

Workload-aware supervision observes progress and contention, extends only the
execution allowance within the authorized total cap, and pauses/defer work when
necessary. It must never lengthen a frozen trial horizon, change tuning evidence
thresholds, restart with a new seed while calling it continuation, or treat a
timeout as scientific rejection. A stuck process with no supported progress is
not granted unlimited time. Test these decisions with injected resource events
before relying on a long shared-GPU run.

## 9. Posterior calibration, burn-in and performance comparisons

Retain the separate posterior controller. An initializer's stopping statistic
or an acceptance-qualified kernel does not determine burn-in adequacy. Use
independent dispersed starts (or the declared ensemble-aware construction),
archive discarded warm-up, and assess readiness on predeclared recent windows.
Current warm-up counts/windows and R-hat thresholds are operational inherited
settings, not universal sufficient sample sizes. Require task-specific evidence
for a changed policy; report `not_ready_at_cap` honestly.

After freezing and equilibration, assess rank-normalized split/folded R-hat,
bulk/tail ESS, numerical/movement health and precision for declared scientific
quantities. Existing lugsail/batch-means MCSE addresses asymptotic sampling
variance under its assumptions, not initialization bias. Check finite moments,
enough batches, batch-size sensitivity and independent-reference calibration.
Use separately validated methods for quantiles and rare events. Common bias,
missing modes and invalid infinite-variance mean estimates cannot be repaired by
reporting a smaller standard error.

Before confirmation, declare an accuracy tolerance for each scientific quantity
in its meaningful units (or a justified reference scale), including mode/tail
quantities that could fail while parameter means look stable. Carry independent
reference uncertainty into the assessment. Reference comparisons should support
agreement within these tolerances, with multiplicity handled across the declared
quantities; merely failing to reject equality is insufficient. If the reference
is too imprecise to decide, the outcome is reference-limited. Accuracy and MCSE
requirements are common across methods and cannot be loosened after their results
are seen.

For SBC, generate independent complete datasets/fits and keep truth assessor-only.
Ranks require the stated exchangeability treatment; correlated retained draws
cannot be treated as independent posterior samples by assumption. Use the
existing calibrated rank/dependence design or a separately validated replacement.
Run coverage/accuracy checks both at a fixed diagnostic sample count and under
the **actual sequential stopping rule**. The latter catches optional-stopping
and precision-policy defects the former misses.

Include every planned fit in the outcome ledger. Crashes, non-delivery, cap hits,
invalid references and missing estimates remain in the original denominator
for delivery claims; also report accuracy conditional on delivery distinctly.
Do not make a fast method look accurate by dropping the fits it cannot finish.

### Cost, uncertainty and simple adversaries

Measure total compile, preparation/map, pilot, measurement, verification,
posterior warm-up, retained and diagnostic time. Report cold and amortized costs,
total value/gradient work and per-chain/ensemble work separately. Measure target
batch latency instead of assuming linear or perfect scaling. Use matched target,
map, starting block, numerical precision and declared resource accounting.
An additional whole-system experiment may include different preparation methods,
but then their costs and effects must be included explicitly.

Use independent complete-run replications, paired by dataset/start block when
appropriate. The same integer seed across algorithms does not mean the same
random experiment. Cluster LAPS observations by independent ensemble; sibling
candidates and saved chunks are not independent fits. Predeclare uncertainty
methods for delivery probability and cost, including censored incomplete runs.
Use supported intervals for a cost difference/ratio or restricted cost-to-delivery
measure; do not rank from one seed, maximum runtime, median ESS or successful
fits alone. Insufficient precision means no supported ranking.

Construct these cheap adversaries on the applicable regimes before interpreting
a complex method: tuned fixed-length whitened HMC (existing baseline), simple
IID-randomized HMC without objective adaptation (resonance baseline), short
corrected HMC with a conservatively calibrated step (numerical robustness), and
an analytic/direct or grid method on tractable small targets (small-model cost
and accuracy reference). Specify the actual configurations and conditional regimes in
the campaign design, not just their names. For initialization compare the existing
dispersed-start/geometry preparation; for filtering compare the serial Kalman
reference. These are falsification checks, not tuning targets.

A statistically supported loss to an applicable cheap adversary vetoes an
improvement/default claim for that regime. Noisy apparent losses remain visible
and require more evidence before promotion; they must not be hidden by pooling
regimes. A loss does not make a mathematically correct optional kernel wrong.
Passing every adversary also does not prove superiority or all-model adequacy.

## 10. Phases, dependencies and completion criteria

Use one progress record for this program, created when implementation begins.
Names below are local to this new-method program and do not renumber the existing
HMC master. After every phase, record completed evidence, failures/repairs,
remaining budget and the exact next phase. Refresh the next phase from observed
costs and findings; do not invent an unplanned extension merely to continue work.

| Phase | Deliverable and main code targets | Required exit evidence | Next decision |
| --- | --- | --- | --- |
| N0: baseline and design | Reconcile source; consumer and nonlinear-model target/score contracts; capability matrix; complete-fit profiling design and model/reference inventory | Small independent value/score/transform checks; chosen references and actual environment recorded; no unsupported consumer route | Select priced initial scope and freeze the first test design |
| N1: shared infrastructure | Typed specifications, state codecs, telemetry, work accounting, method proposer interface; changes in candidate-set tuning/execution/retained and validation designs/procedures | Existing HMC API-to-kernel parity and replay; negative schema/RNG/work tests; one-sided policy tests; proper invariant-test dispatch | Enable new kernels only after existing behavior survives migration |
| N2: randomized HMC | IID duration profile and bounded TF/XLA execution through both public tuners | Analytic length law, reversible-mixture tests, resonance/identity controls, LGSSM public pipeline and crash recovery | Optional experimental candidate; price full tuning/retained fits |
| N3: ChEES/SNAPER proposer | Fixed-geometry source-mapped adaptation and immutable candidate emission | Objective/derivative reference, freeze/metric/jitter tests; anisotropic/funnel and core nonlinear pipelines; uncertainty-aware comparison with simple random lengths | Keep both methods; efficiency/default decision only under confirmation criteria |
| N4: MAMS | Force map/Jacobian, deterministic full-refresh kernel; then stochastic refresh/higher-order profiles separately | Mathematical/source audit, negative controls, sphere/path tests, independent posterior references and multi-model pipeline | Qualify optional deterministic profile before enabling later variants |
| N5: core confirmation | Frozen corrected families on the analytic/SSM suite, including the complete required nonlinear matrix in section 8E | Calibrated tests, actual-stopping posterior accuracy/delivery, uncertainty and budget reconciliation | Scope-specific release decision; failed candidates trigger recorded repairs |
| N6: LAPS | One source-resolved ensemble warm-up profile and MAMS bridge; ensemble-aware evidence path | Diagnostic counterexamples, stage/discard/freeze tests, measured batch scaling, independent ensemble comparisons | Keep experimental unless initialization benefit and downstream accuracy justify use |
| N7: targeted alternative | MALT first if the stated residual failure warrants it; later MEADS/DRGHMC only with a separate bounded amendment | Method-specific correction/state tests and improvement evidence against existing survivors | Release a supported optional profile, or record that current methods suffice |
| N8: consumer and documentation closure | Actual MacroFinance/BGS and nonlinear adapters within supported contracts; NL-GROWTH/NL-SV disposition; registry/reference/book updates; final integration audit | Public consumer-to-kernel wiring, runnable examples, chapter build, scoped limitations, migration tests | Optional release; separate decision for default promotion |

Target-evaluator optimization and Pathfinder are parallel **workstreams**, not
an instruction to launch additional agents. Their dependency is N0 target/score
qualification, and they join N5/N8 only after equivalence or initialization
evidence. Particle methods require a separate latent-path scope. A supported
N2–N5 release need not wait for every conditional branch, but must name exactly
which methods and target classes it covers.

Each implementation phase updates tests alongside code. Extend existing
`tests/test_hmc_candidate_set_*`, acceptance-policy/statistics, replay and
`tests/inference_validation/` suites; add focused proposed files for randomized
trajectories, SNAPER adaptation, MAMS subflows/work, MALT path correction and
LAPS ensemble state only as those components are introduced. No placeholders
passing unimplemented behavior, and no tests that merely restate a formula used
by the implementation as their only oracle.

Update `docs/chapters/ch21b_hmc_tuning_interfaces.tex` and the interface reference
when APIs become available, with the same semantics and generated capability
table. Update `ch26g_modern_hmc_methods.tex` where implementation findings change
its recommendations or source qualifications. Keep one operational guide.

## 11. Compute, numerical choices and executable handoff

This program does not recycle prior CPU/GPU campaign allowances. Previous
budgets and incomplete old campaigns are not a fresh allocation to this program.
Routine unit/compile/reference checks can follow implementation authorization;
serious runs need the concise phase design and a bounded authorized allowance.

Before a long run, price **complete fits**, including tuning/verification and
posterior stopping, on the cheapest analytic target, an LGSSM and a nonlinear
target. Separately measure cold compilation, steady-state batched score cost,
memory and contention. A few transitions price neither a full search nor an
ensemble experiment. Use those measurements to choose candidate counts, trial
replications, dataset counts and parallelism. Record timing uncertainty and a
reserve for localized repair, reference computation and terminal audit.

If a 48-hour wall window is proposed later, calculate feasibility from the task
graph and measured prices: CPU work, GPU device-hours, shared-device waiting and
wall time are different budgets. Fit N2/N3 and representative SSM confirmation
first if the complete all-method matrix does not fit. Do not promise a universal
48-hour campaign before prices exist. Do not shorten evidence horizons or discard
hard models to make a terminal report look complete. Explicitly defer unfunded
tests and reduce the release claim accordingly.

The campaign design must make the following numeric fields concrete before
launch: model dimensions/horizons, maps, epsilon/duration hypotheses, integrator
profiles, optimizer targets/windows/caps, acceptance floor, W/T, start weights,
candidate and trial caps, error allocations, posterior estimands/tolerances,
replication design, hardware policy, allowance limits and total attempt/resource
budget. This document specifies how to choose and test them; unsupported inherited
numbers are not silently assigned just to make a command runnable.

| Material choice | Provenance/status now | Risk and earliest discriminating check |
| --- | --- | --- |
| TF/TFP, GPU/XLA and memory growth | Owner policy | Port incompatibility, compilation or memory failure; tiny trusted native check before scale-up |
| Fixed supplied geometry for first profiles | Design decision isolating sampler effects | Residual curvature may dominate; transformed score and whitened/residual Gaussian/funnel controls |
| IID uniform duration and ceiling rule | Proposed simple baseline law | Resonance/cost distribution changes with rounding; analytic histogram/proposal-law check |
| SNAPER 0.8; MAMS 0.9 and reported 0.99 funnel; LAPS 0.7/0.9 | Inspected source settings, hypotheses only | Target-specific mismatch; pilot sensitivity and fresh qualification, no automatic default |
| MAMS `sqrt(d)` initial duration | Source starting hypothesis in whitened units | Poor transfer to low-dimensional marginal parameters; measured pilot and scale neighborhood |
| Acceptance floor and evidence design | Must be chosen in N0/N1 phase design | Inconclusive or expensive test; power/precision arithmetic plus cheap calibration before campaigns |
| Four-start initial trial bank | Inherited v7 compatibility choice, experimental | Unrepresentative starts or mistaken independence; heterogeneous-start and complete-trial tests |
| Existing posterior windows/R-hat thresholds | Inherited operational policy; not burn-in proof | Common bias or insufficient mixing; independent reference, separated-mode and actual-stopping tests |
| Lugsail/MCSE settings | Existing estimator under moment/mixing conditions | Too few batches, tail or ensemble dependence; calibrated covariance/coverage tests |
| K0–K7 fixture numbers | Existing validation catalog | Fixtures omit consumer features; masks/correlation and consumer-specific witnesses |
| Ensemble size, timeouts and replication counts | Unchosen until priced; not paper defaults | Wrong batch scaling/underpowered study; cost and power design before authorization |

No arbitrary epsilon clipping, ridge, damping or state reset may be added as
an invisible stability fix. A numerics-changing guard needs its own stated
mathematical target, non-harm checks on healthy inputs and failure-case evidence.
Observability and fail-closed domain guards should be added with cheap no-fire
regressions; they do not require another governance chain.

Existing CLI plumbing can be reused once new typed designs are supported:

```bash
# Framework-free enumeration of the current validation infrastructure.
python -m bayesfilter.testing.inference_validation list

# During implementation: CPU-only reference/contract checks, not GPU qualification.
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_hmc_acceptance_protocol.py \
  tests/test_hmc_acceptance_statistics.py \
  tests/test_hmc_candidate_set_execution.py \
  tests/inference_validation/test_references_and_mechanics.py
```

The current tests cover current behavior; they must be extended before they can
establish new-method support. Each phase must generate a concrete suite JSON
validated by the existing `plan` command before `run`. Do not copy a proposed
method label into an old fixed-L suite and infer support from a successful parse.
GPU launch commands require trusted execution and the repository's verified
memory-growth launcher. Record the actual command/environment in the manifest.

Use a fresh root under
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/<phase>/<attempt>/`.
Preserve Git/source identity, environment, CPU/GPU status, data version, seeds,
elapsed and charged work, full configuration, plan/result paths and failure
classification. Localized retries keep the scientific contract and remaining
budget, use fresh attempt directories and preserve earlier outcomes. Ordinary
Git, checksums and versioned directories suffice; no approval-token machinery.

## 12. Skeptical review and release conditions

The planning audit found and repaired these material hazards before recommending
implementation:

| Hazard | Repair incorporated in this plan |
| --- | --- |
| Treating current tuning as a drop-in epsilon/L recipe | Reuse lifecycle; extend full transition, state, proposer, work and evidence contracts |
| Favoring new methods by giving only them a less restrictive acceptance policy | Matched-policy kernel comparison plus a separate existing-policy whole-procedure baseline |
| Requiring every method to pass a one-dimensional fixture | Explicit MAMS/LAPS dimension rejection; multidimensional positive SSM cases |
| Rejecting efficient high-acceptance methods with the old upper band | Versioned one-sided qualification; independent movement/cycle checks |
| Presenting experimental v7 evidence as settled release coverage | Explicit experimental status and new-policy calibration/theory requirements |
| Double mass adaptation or changing fixed-transport geometry | Fixed-geometry SNAPER profile; one geometry owner; fresh scope for changes |
| Implicit Halton/shared jitter | Explicit IID law first; phase/topology bound for any later alternative |
| Applying a reversible test to every augmented kernel | Checked reversibility/adjoint or independent stationarity test |
| Dropping momentum/slice/fold state on chunk boundaries | Full Markov-state codec and public-pipeline restart tests |
| Using endpoint HMC energy for MAMS or MALT | Separate Jacobian work and deterministic path-work corrections |
| Counting ensemble rows as independent trials/MCSE chains | Independent complete ensemble replication and qualified clustered analysis |
| Promoting LAPS moment diagnostics to burn-in proof | Diagnostic counterexamples, discarded initialization, independent posterior evidence |
| Silently mixing paper and author-code LAPS variants | Named source profile; equation/source/freeze/stopping audit before port |
| Using an approximate BGS force as an exact score | Target-capability rejection or separately derived and registered force profile |
| Claiming nonlinear posterior accuracy from a K7 approximate-target grid | Separate sampler and filter-approximation errors; K6 oracle limit explicit |
| Treating one reduced nonlinear fixture as adequate application coverage | Required core matrix of accumulation, predator–prey UKF/SGQF, SIR and structural targets; explicit growth/SV work items and per-cell dispositions |
| Pricing a full search from short kernels or importing paper ensemble sizes | Complete-fit pricing, bounded contention recovery and honest deferrals |
| Benchmark winner chosen among only successful fits | Original denominator, censoring-aware cost/delivery analysis and fresh comparisons |
| Claiming a kernel is integrated because a class exists | Executable consumer-to-public-tuner-to-kernel and export/reload tests |
| Publishing another inconsistent tuning guide | Update the existing reference, registry and official book together |

The baseline ladder, target/geometry matching, uncertainty analysis and stage
roles address wrong comparators, unfair comparisons and proxy promotion. The
source/environment checks address stale context and backend mismatch. Explicit
caps, failure classifications and phase refresh address missing stop conditions.
Proposed commands are identified as existing plumbing, current tests or future
phase commands; none is presented as a new-method campaign already runnable.

**Audit disposition:** the program is coherent for implementation planning.
Numeric campaign settings remain intentionally unchosen where local evidence is
missing; those are N0/N1 deliverables and prevent launching an unspecified long
run, not routine infrastructure work. No performance, convergence, new-method
support or default-readiness claim is made by this document. This is a source/code
self-review, not a new independent Claude or MathDevMCP certification.

Release has two levels. An **optional supported method** needs checked mathematics
and source mapping, target capability enforcement, native TF/GPU/XLA evidence,
calibrated statistics, independent posterior references for its declared scope,
full public-pipeline/recovery tests and consistent documentation. **Default
promotion** additionally needs uncertainty-supported end-to-end advantages across
the claimed regimes, no applicable simple-adversary veto, adequate failure
behavior and an explicit owner decision. An optional correct method can remain
useful without being the fastest method or the new default.

The current execution record identifies the next unfinished checks and phases.
The user funded 72 additional aggregate CPU/GPU worker-wall hours on October 8.
The [active N5 campaign](bayesfilter-modern-hmc-n5-campaign-2026-10-08.md)
governs saved SIR recovery, complete-fit pricing, references and subsequent
nonlinear comparisons. This replaces the exhausted engineering allowance for
new work; it does not reset historical accounting or qualify unfinished cells.

**Historical state, October 9 (superseded by the refresh below):** D0, D1, E0 and E1 finished at 17:08:59 UTC on
October 8 (01:08:59 China time on October 9). A subsequent funded continuation
has completed D2a's sixteen healthy PP-UKF/SIR GPU price calls and both selected
structural replays. E2c resolved local finite-difference score discrepancies on
all 128 saved points; structural reference precision still fails. D2b completed
UKF fixed: L=9 passed posterior/reference checks and L=3 was precision-limited.
Its next fit failed preflight on changed GPU availability, before sampling.
D2d is now terminal: the concurrent SIR sequence finished at 13:53:27 UTC.
SIR fixed passed both original members; randomized passed L=9 while L=3
remained precision-limited at 10,000 retained draws. UKF randomized exhausted
its unchanged wall cap: L=3 completed but failed precision; L=9 was interrupted
and the overall fit result is absent. Both UKF tuning members remain verified.
A separate transient-service cleanup race prevented the observer's closeout.
D2e repaired and tested that race, verified unchanged numerical source/reference
hashes and preserved the honest incomplete outcome in
`N5/full-model-closeout-01/`. No numerical result was reconstructed or rerun.
Watcher 10 is terminal; no D2d worker remains active. The cleanup fix passed
54 focused regressions, including already-removed and still-live unit cases.
D2's full-model harness now supports target-specific
quantities and matched references for both targets. The
[terminal execution audit](bayesfilter-modern-hmc-execution-2026-10-07.md#n5-batch-terminal-audit-9-october-2026)
records results, source checks and the settled budget.

Historical initial complete-fit pricing covered fixed HMC, randomized HMC, SNAPER,
MAMS and the now-excluded TFP NUTS comparator on LGSSM and accumulation. All ten
fits passed their separate posterior controller; nine passed every member's
independent agreement screen. The seed-1000 randomized LGSSM agreement failure
remains preserved. The separately declared D0 replication finished all 32 fresh
fits: all 64 members passed posterior and independent agreement screens.
Eight seeds per target/method leave a wide working-binomial 95% success interval
[.6306, 1]; there is no supported performance or reliability ranking.

D1 completed both full twenty-observation predator–prey SGQF fits and retained,
exported and reloaded both tuning-qualified members of each. The fixed L=9 and
randomized maximum-L=9 members passed posterior and reference screens after
4,000 and 6,000 retained draws per chain, respectively. Both L=3 members reached
the unchanged 10,000-draw cap with insufficient lugsail precision. Their
warm-up, R-hat, ESS and numerical health checks passed. Thus 2/4 members and
2/2 fits' at-least-one-member outcomes pass, while 0/2 all-member fit outcomes
pass. These are posterior precision limits; all four tuning qualifications
remain intact. One paired seed and this narrow grid establish no method ranking.

Full structural, predator–prey UKF/SGQF and SIR local geometry proposals have
healthy short GPU/XLA trajectories and independent value/score parity. The
saved two-member SIR mechanics cohort finished verification/export/reload.
PP-SGQF, SIR and now PP-UKF pass their unchanged independent-reference empirical
refinement criteria. Structural E1 remains limited: source-square rho has
half-width .002707 and last-order change .002044 reference SD, both above .002.
The other fourteen quantities pass. Further blind sample-count enlargement is
not the planned repair; first diagnose quantity-specific importance weights
and proposal coverage, preserving every scramble and the unchanged target.

The 15:19:39 UTC terminal ledger charges 50.10 worker-hours, has no active
reservation, and leaves 21.73 hours unallocated after the unchanged ten-minute
overhead reserve. D2d consumed 63,820.863 additive worker seconds, in addition
to D2c's already charged 2,710.947-second prefix. Unused D2d caps were released
once; no grant reset or nested-worker double charge occurred.

The next steps at that historical checkpoint were to use the structural diagnosis to design a fresh
calibration/validation proposal repair; inspect supported checkpoint continuation
for the interrupted UKF member and plan a duration/precision study for the
count-limited members; price the missing nonlinear SNAPER/MAMS cells; and freeze a
fresh paired comparison with all outcomes preserved. The D1 precision-limited
members remain terminal evidence when planning duration or geometry studies;
posterior diagnostics cannot become tuning membership or ranking rules.
The complete nonlinear matrix and N5 statistical qualification remain open.
N6 LAPS and the conditional later phases retain their own source, correction,
tuning and posterior requirements. The
[bridge record](bayesfilter-modern-hmc-n5-bridge-2026-10-08.md) preserves the
earlier mechanics and parity evidence. None of these batch results establishes
exact nonlinear posterior accuracy, release readiness or default promotion.

### October 10 funded continuation

The owner's additional 48 aggregate worker-hours brings the active N5 grant to
120 hours without resetting its settled history. The
[supplemental continuation plan](bayesfilter-modern-hmc-n5-additional-48h-2026-10-10.md)
governs checkpoint recovery, missing full-model comparator pricing, fresh
structural-reference proposal repair and the subsequent measured-cost refresh.
D2f finished: the interrupted UKF randomized L=9 member passed posterior and
reference checks after 2,000 warm-up and 5,000 retained draws per chain. One
committed chunk was replayed and six new chunks were computed under the original
policy; the added occupied cost was 4,252.341 seconds. The completed L=3 precision
failure and original missing fit result remain unchanged. Eight native
continuation/failure-path tests pass. NUTS is excluded by the owner; its three
old full-model price rows are historical bookkeeping, not unsupported methods
awaiting repair. SNAPER/MAMS retain full internal filter-status coverage. After the supplemental
pre-run audit, E3 completed fresh influence-proposal calibration and untouched
power-19/21 validation under the original fifteen-quantity precision screen.
All fifteen quantities pass, with maximum half-width/SD .0018873181816258845 and
last-order change/SD .0014673134087065422. Independent native value/score parity
passes. This is empirical refinement of the same finite UKF target, not exact
nonlinear-model inference. Fifteen reference controls pass. All six full GPU 0/XLA integration tests
pass, including SGQF/UKF/SIR data/geometry/status and native SIR MAMS. The first
price queue failed device preflight after GPU 1 became idle, before numerical
work; its 0.318 occupied seconds remain charged. The same six-cell SNAPER/MAMS
price queue finished on GPU 1 in `N5/full-comparator-prices-02/`: all six
approved cells and twelve short candidate paths passed their numerical-health
screen in 1,408.462 occupied seconds. These are prices, not tuning membership
or posterior qualification. Four queue/accounting tests pass. Observer
`N5/supplement-watch-04/` is terminal and settled structural-reference monitoring against
`N5/nuts-exclusion-01/budget-ledger.json`, including 52.127 seconds of exclusion
regressions. The refreshed twenty-cell matrix is
`N5/method-study-refresh-01/nonlinear-comparison-matrix.json`; excluded historical
rows, costs and all approved-method failures are preserved. No campaign worker
remains active at this settlement. The ledger charges 193403.4474935018 seconds
against the cumulative 432000-second grant; after the existing 600-second reserve,
237996.5525064982 seconds (66.11 hours) remain.

### Active remaining program and D4 audit

The [D4 plan](bayesfilter-modern-hmc-n5-method-proposals-2026-10-10.md) addresses
the missing concrete design. D3's 32-step SNAPER pilots all proposed L1, and its
MAMS epsilon.2/L3/9 settings were borrowed hypotheses. Neither is adequate
method-specific tuning or a strong baseline. D4a prices 1000-step discarded
SNAPER proposals with a duration neighborhood and nine native MAMS hypotheses
on full SGQF, UKF, SIR and structural targets. It validates structural dimension5,
horizon100 and all fifteen quantities through a new consumer, preserving old
source closures. Short prices cannot change tuning membership or establish
posterior mixing. The plan declares provenance, caps, candidate-veto handling,
GPU/XLA/memory policy, complete denominators, and exact commands. Skeptical
self-audit passes for D4a after focused checks; D4b is refreshed from terminal
prices before launch, not left as an unspecified long command.

| Phase | Remaining deliverable | Refresh/continuation rule |
| --- | --- | --- |
| D4a | Target-specific modern proposals and full native prices | Preserve every hypothesis; settle costs and refresh D4b |
| D4b | Public fresh qualification, export/reload of every verified member, shared posterior and independent agreement | Variable denominators; continue valid peers after a member veto; separate posterior failure from tuning |
| D4R: generic adaptation repair | Independent requested travel/epsilon proposals, MAMS discarded adaptation, frozen-scale diagnostics, analytic and state-space regressions | Follow the October 11 repair subplan; preserve v1 policies and all verified members; reprice before fresh multi-model confirmation |
| D4c/F, N5 | Broader classical baseline, fresh paired replications, twenty-cell evidence and uncertainty | Freeze feasible cohort from measured costs; under-budgeted cells stay explicit |
| N6 | Source-resolved LAPS ensemble initialization/adaptation and corrected MAMS bridge | Own audited scientific contract; conditional on budget and qualifying implementation |
| N7 | MALT or other targeted alternatives if residual failures warrant them | Conditional branch, not an implicit implementation promise |
| N8 | Consumer and release audit, NL-GROWTH/NL-SV disposition, registry/reference/book consistency | Scoped optional release can cover N2–N5 without all conditional branches; default promotion remains separate |

Posterior R-hat/ESS/MCSE/reference checks do not select or repair tuning members.
Count-limited original members and failed references remain recorded failures.
All healthy modern proposal prices still require fresh measurement and
verification. No superiority, complete nonlinear-matrix qualification, release
or default-readiness claim follows from this refresh.

### D4a terminal audit and D4b1 allocation

Terminal inspection found all sixty proposal and measured records, no omitted
hypotheses, valid full target/data/geometry/status wiring, and unchanged source
and reference hashes. SNAPER now proposes multiple-step trajectories on SGQF,
UKF and SIR. The structural pilot proposed epsilon about.0030–.0038; its healthy
short paths are insufficient evidence of useful mixing. MAMS's nine hypotheses
on each target remain unqualified proposals despite high short-path acceptance.
No statistical method ranking is supported.

D4b1 follows the declared target/method order and retains **all** proposals in
both first SGQF cells. Full-set price extrapolations give about9.02/20.96 hours
for fresh tuning plus2000 warm-up and10000 retained transitions across each set;
both10000-count maxima would require more. These are workload hypotheses, not
changes to sequential stopping. Allocate12 hours SNAPER and27 hours MAMS, plus
bounded independent assessments and cleanup, within D4's40-hour first allocation.
All verified members must export/reload; every member retains its posterior
outcome, including vetoes, incomplete precision or resource stops. N5's other
full nonlinear cells, fair classical search and fresh paired replication remain
open. The detailed plan's evidence contract and skeptical audit govern launch
after focused checks pass.

## Source anchors for implementation

The [local source manifest](../../.localresources/hmc-methods-review-20261007/source-manifest.json)
records paper versions, inspected sections, hashes and author-code revisions.
Use its pinned snapshots for the initial port; record deliberate upgrades.

- MAMS: [arXiv:2503.01707](https://arxiv.org/abs/2503.01707), inspected v2,
  methods/tuning sections 3–6 and Appendices A–C;
  `blackjax-mcmc-adjusted_mclmc.py` and `blackjax-mcmc-integrators.py` in the
  archived source directory. Chapter anchors: `sec:bf-modern-mams`,
  `eq:bf-modern-force-jacobian`, `eq:bf-modern-mams-acceptance`.
- LAPS: [arXiv:2601.16696](https://arxiv.org/abs/2601.16696), inspected v1,
  sections 2–7 and Appendices A–F; `blackjax-adaptation-laps.py`,
  `blackjax-adaptation-laps_burn_in.py` and the archived step-size source.
  Chapter anchor: `sec:bf-modern-laps`; source differences are described there.
- ChEES: [Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html);
  SNAPER: [Sountsov and Hoffman, inspected v3](https://arxiv.org/abs/2110.11576),
  especially Appendix C; installed TFP 0.25.0 snapshots
  `installed-tfp-snaper_hmc.py` and
  `installed-tfp-gradient_based_trajectory_length_adaptation.py`.
- MALT: [Riou-Durand and Vogrinc, inspected v3](https://arxiv.org/abs/2202.13230);
  [adaptive MALT](https://proceedings.mlr.press/v206/riou-durand23a.html);
  archived `tfp-adaptive_malt.py`. Chapter anchor: `sec:bf-modern-persistence`.
- MEADS: [Hoffman and Sountsov, 2022](https://proceedings.mlr.press/v151/hoffman22a.html);
  Pathfinder: [Zhang et al., inspected v4](https://arxiv.org/abs/2108.03782);
  delayed-rejection GHMC: [inspected v2](https://arxiv.org/abs/2406.02741);
  particle-MALA: [inspected v1](https://arxiv.org/abs/2401.14868).
- Validation and precision: [existing validation guide](../validation/README.md),
  its Gandy–Scott/SBC/reference provisions, and the chapter's derivations and
  citations for modern R-hat, ESS, batch means, lugsail and fixed-width stopping.
  Apply their assumptions to the actual independent unit and declared estimand.
