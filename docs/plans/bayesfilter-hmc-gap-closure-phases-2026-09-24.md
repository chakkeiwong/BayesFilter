# HMC closure phases G–L

The active localized repair is the reviewed
[September 25 budget-debug plan](bayesfilter-hmc-budget-debug-2026-09-25.md).
It addresses the development/confirmation graph-policy mismatch and stranded
cell allocations within the existing grants. C2 currently has 255/256
completed fits; the last original allowance is exhausted. Both full GPU parity
checks passed. The repaired full C1 forecast is 37.473 hours; all 512 confirmation
fits are now running under the existing 46-hour ceiling. The
[repair result](bayesfilter-hmc-budget-debug-result-2026-09-25.md) and active
checkpoint supersede the earlier queue state without changing the scientific
criteria or denominators below.

## September 25 additional allocation and active execution decision

The owner added **50 GPU hours (180,000 GPU worker-seconds)**. This is an
additive grant, recorded separately from the September 24 ledger so that I's
existing finalizer cannot overwrite or double-charge it. The original I/C2
run keeps its source, 256-fit inventory, statistical criteria and reservation.
The following continuation spends only the new grant. CPU engineering checks
use at most 1,800 seconds from the previous CPU balance. This section supersedes
the older statements below that C1 is unfunded; it does not change its criteria.

| Work | Maximum new GPU seconds | Basis and next decision |
| --- | ---: | --- |
| Direct C1 pricing | 3,660 | Four full no-comparator fits, two per target; 900 seconds per fit plus 60 enclosing seconds. Same numerical settings as C1, disjoint development seeds. |
| Full C1 confirmation | 165,600 | A 46-hour convenience ceiling, not a measured price. Launch only if 256 times the slowest complete development-fit cost for each target, plus 120 seconds of coordinator overhead, fits this ceiling. Divide the available cell time proportionately to these measured target costs. |
| Canonical NeuTra initial pricing | 1,200 | Reserved only; revise and test the canonical architecture runner before launch. The former 48-arm configured-map recipe is superseded. |
| Localized repair and unallocated balance | 9,540 | Released time may remain unused. No automatic rerun or change to statistical thresholds. |

The pricing envelope keeps the inherited 750-second fit allowance, up to
120 seconds of measured-contention extension and one-second shutdown grace.
The four 900-second cells and 60-second enclosing allowance are convenience
containment margins, not runtime quantiles. The confirmation cell allocations
are execution-only; the prepared design's root seeds, 256 replications per
target, broad L grid, target/data, posterior policies and predeclared
first-verified-member rule remain unchanged. Development IDs/seeds are disjoint
from all 512 prospective confirmation fits and are never pooled into them.
The max-of-two pricing rule is a conservative descriptive price relative to
the observed fits, not a tail guarantee. Censoring cannot be treated as a
complete-fit price. Failure to fit the envelope triggers diagnosis and replanning
within the remaining grant, not shorter chains or fewer replications.

### Evidence contract and skeptical audit

The question is whether the already nominated ordinary-HMC Gaussian and
fixed-data beta-binomial posterior policies deliver valid intervals at their
actual stopping times. The exact comparator is each target's analytic model-
coordinate mean/quantile. Qualified delivery, unconditional coverage, and their
joint event retain separate pointwise exact 95% intervals over all 256 planned
fits per target; each declared lower bound must reach .90 (240 successes).
Missing outcomes remain unsuccessful. Conditional coverage, short-chain
diagnostics, costs and training losses cannot replace these criteria. No
simultaneous-coverage, universal burn-in, method-ranking or default-promotion
claim follows. Ordinary epsilon/L tuning still admits on acceptance/health,
retains every verified candidate, and does not gate on R-hat, ESS or MCSE.

The pre-execution audit checked the prepared 2D Gaussian and fixed observations
`[5,12]` for beta-binomial, removed-comparator design, actual isolated-child
execution and full-denominator reporting against the tested continuation
snapshot. Critical ordinary-runtime/reporting files match that snapshot.
The source for this continuation is a new bounded copy of its numerical
package with ordinary-route-only eligibility explicitly recorded; no archived
NeuTra implementation is used as current learned-map evidence. Versioned
outputs, exact source hashes, environment, selected GPU UUID, XLA and verified
memory growth remain mandatory. Unrelated dirty work is excluded.

The audit caught two stale assumptions. The subtraction forecast is not a
direct price, so four same-workload fits precede confirmation. Also the
September 25 canonical-architecture directive retires the previous J recipe:
future learned-map work must use `bayesfilter_neutra_iaf_author_v1` and current
source anchors, with target-specific training choices. Historical J capacity
witnesses do not become evidence for that architecture. No old-map GPU job is
queued. This ordinary C1 continuation does not depend on a learned transport.

The revised execution passes review for pricing followed by the frozen C1
study, conditional only on measured affordability and engineering validity.
The automated queue waits for I's service **and its finalizer** to stop, checks
I's terminal receipt is present, and waits for the selected GPU UUID to have
no compute processes. C2 statistical failure does not veto independent C1.
Invalid source, corrupt/missing required process artifacts, unexpected process
errors, missing hardware provenance, or the grant ceiling stop the affected
launch. A valid posterior-policy rejection remains a result, not a harness
error. Every expensive launch receives its own receipt; nested timing is not
charged twice. Queue waiting is recorded as wall time, not GPU compute time.

The run manifest and ledger are under
`artifacts/hmc-additional-gpu-2026-09-25/`. The tested queue prices each full
fit, makes the numerical affordability decision, preserves the complete C1
inventory, and writes a terminal summary requiring scientific review. A
service exit alone cannot close C1. The terminal review must inspect fit
identities, missingness, posterior-health failures, exact bounds, source,
hardware and all charged attempts. Exact MacroFinance inputs remain an
independent dependency. The official guide remains `docs/main.tex`.

Status: reviewed for bounded execution, 2026-09-24. This extends the existing
[master](bayesfilter-hmc-repair-master-program-2026-09-16.md). It does not
restart completed A–F work or replace the official tuning chapter.

## Research intent and evidence contract

The question is which remaining tuning, posterior-delivery, uncertainty,
geometry and consumer requirements can be closed with the current code and
remaining allocation. Engineering, sampler validity and statistical calibration
remain separate. The baseline is the frozen source and numerical designs in
the September 24 remaining-gap and timeout results. The new source includes
only owned repairs over the recorded Git baseline; concurrent Q20/NeuTra work
is excluded unless independently incorporated and tested.

Primary engineering criteria are invariant numerical behavior under execution
controls; interpretable readiness/precision diagnostics; checked transform,
inverse, Jacobian and score arithmetic; exact consumer reproduction; and an
aligned, buildable official book. Statistical closure keeps the existing C1
coverage/delivery and C2 null-size/power criteria and denominators. Passing a
smoke, a training loss or a local R-hat threshold does not close those studies.
R-hat remains outside tuning admission. All verified candidates are retained.

Corruption, wrong source/target, violated numerical identity, missing required
hardware provenance or an exhausted total allocation are continuation vetoes
for the affected experiment. A failed candidate, slow productive fit or poor
map is a repair trigger; it does not stop independent phases. Runtime and short
training diagnostics are explanatory only. No stochastic ranking is claimed
without the predeclared uncertainty analysis. Preserve every failure and output
under `artifacts/hmc-gap-closure-phases-2026-09-24/`.

## Phases and refresh decisions

| Phase | Work and completion criterion | Refresh before the next phase |
| --- | --- | --- |
| G: actual cost and workload | Inspect saved stage/chunk costs; run a bounded same-workload GPU diagnostic and up to three complete development fits with actual workload telemetry. Record preparation, tuning, posterior, serialization and termination separately where available. No reduced L grid, changed acceptance band or shared fitted state. | Price the complete inventory from all attempted fits, treating caps as censored costs. If current prices cannot fund confirmation, do not launch it; continue H/J/K/L. |
| H: readiness and uncertainty | Audit existing warmup and precision reports; add missing explanatory information without changing decisions. Check actual posterior paths on Gaussian, beta-binomial and supplied-map cases. Use independent arithmetic and archived arrays to distinguish missing delivery, initialization and MCSE/stopping concerns. | Freeze only mechanism-supported optional policy candidates. C1 remains a separate full-denominator confirmation and cannot be replaced by these diagnostics. |
| I: confirmation disposition/execution | Preserve historical C2 as 9/256 complete, with three exhausted slots. Review source changes and costs before any continuation. Prefer a separately identified fresh 128/64/64 confirmation if pooling old/new source outcomes is unjustified and its whole forecast fits the remaining reservation. Otherwise retain the incomplete historical result and a concrete funded/unfunded disposition. C1 retains 256 Gaussian and 256 beta fits. | Do not reopen exhausted historical fits, silently pool source versions or reduce denominators. If a whole confirmation is priced and funded, execute it under the frozen criteria and cumulative outer budget; otherwise record the exact funding gap. |
| J: geometry capacity and downstream validity | Test supported multi-stage/permuted IAF and conditional DSF capacity against the identified forced-affine first marginal; check frozen density, inverse and score mechanics. Keep supplied exact/residual funnel maps as positive controls. Prepare a target-specific training/holdout/downstream protocol only after the family passes capacity checks. | Capacity is not trained-map quality or mode exploration. If training plus independent posterior confirmation cannot be funded, stop that study as underfunded and preserve the checked capacity result. No new sampler is introduced here. |
| K: exact consumer coverage | Recheck named MacroFinance reproduction paths and matching joint reference; inventory required inputs and run the exact integration if present and affordable. Missing inputs retain an explicit dependency and local reply. | Never substitute synthetic results or a fixed-loadings target for the requested full joint target. |
| L: maintainability, book and terminal audit | Inventory remaining active legacy facade dependencies and remove a narrowly owned dependency only with its actual caller/parity test. Resolve book-build citation failures using existing sources. Run affected multi-model tests, compile the official book, inspect changed pages and reconcile all phases/budgets. | Update master and machine progress with completed, failed-candidate, underfunded or awaiting-input dispositions. No automatic additional phase. |

Each phase writes one result section, a receipt-backed resource charge and a
short next-phase decision into the same execution note and machine progress.
This is ordinary reproducibility, not a new approval chain. User authorization
to execute this program covers localized repairs/retries within its contract.

## Allocation and numerical provenance

Opening balance: 69,588.313 CPU and 64,667.835 GPU worker-seconds, inherited
from the live ledger. The 60,022.255 GPU-second C2 reservation remains separate.
The following are convenience **maximum development allocations**, not prices,
scientific defaults or additive grants:

| Work | CPU ceiling, seconds | GPU ceiling, seconds |
| --- | ---: | ---: |
| G diagnostics/development | 1,200 | 2,940 |
| H reports/reference checks | 3,600 | 0 |
| J capacity and supplied-map mechanics | 1,800 | 600 |
| K input inventory/integration preparation | 300 | 0 |
| L tests/book/audit | 2,400 | 0 |
| Localized development retry reserve | 1,200 | 600 |

Development GPU maximum is 4,140 seconds, below the 4,645.580 unreserved
balance; 505.580 seconds remain unallocated. C2 confirmation consumes at most
its remaining reservation. C1's old 137,189-second point forecast is unfunded;
no runtime observation here is evidence that this cost disappeared. Lower
observed cost can nominate a revised price, not a smaller statistical design.

G's complete fits use the existing baseline/quarter/half development designs,
one fit each, each enclosing cap 900 seconds (the inherited development margin,
not a tail guarantee). Set base allowance 750 and extension pool 120, leaving
30 seconds within 900 for coordinator/launch overhead. These execution-only
numbers are hypotheses to be evaluated, not new inference defaults. Up to
240 GPU seconds fund the fixed-workload diagnostic. A capped first full fit
triggers cost diagnosis before spending on its peers; it may render the full
confirmation unaffordable under the current sharing conditions. Do not create
competing load or terminate foreign processes. Never infer GPU identity from a
numeric CUDA ordinal. All GPU work uses trusted execution, selected UUID,
memory growth before import, batch-native TF/TFP and XLA; no pfor.

H uses existing saved arrays and numerical test fixtures with CPU explicitly
hidden from GPU. New explanatory window/information fields cannot change
acceptance, warmup readiness or posterior precision decisions. Numeric test
tolerances come from the existing independent-reference tests or exact fixture
arithmetic. J uses small configured-flow fixtures and deterministic seeds from
the current tests; their widths, stage counts and point grids test capacity and
mechanics only. They are not a training recipe or default family choice.

## Skeptical program review before execution

The review found and corrected five potential errors. First, a successful
supervisor is not a completed fit, so G requires complete-fit cost or explicit
censoring. Second, cross-source seed/artifact identities prevent silently
resuming C2 under new code; I must report source versions and original failures.
Third, mean lugsail, quantile MCSE and warmup readiness are distinct mechanisms,
so H cannot repair them by changing one threshold. Fourth, one nonlinear map
test cannot establish mixture representability or global exploration; J limits
its first claim to removing the identified forced-affine restriction. Fifth,
the allegedly missing bibliography entries already exist; L must inspect the
build before adding or changing citations.

The revised program passes for its declared engineering/development scope.
The original scientific comparators, statistical units and stop rules remain
unchanged. New-source/runtime failures have localized repair branches. Full
confirmation and trained-map quality remain conditional on measured cost and
adequate funding. No external messages, package changes or new paid compute
are part of this program.

## Execution commands and artifacts

Use the existing public `python -m bayesfilter.testing.inference_validation run`
on generated explicit suite JSONs in the phase directory. Save exact commands,
source snapshot/commit, seeds, data version, GPU policy, wall time and all exit
receipts. Use bounded launcher receipts to charge each enclosing run once.
Focused CPU tests use `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true`
before importing TensorFlow. Book checks use `latexmk -pdf -bibtex` in a fresh
versioned build directory; bibliography inspection decides whether a code/doc
repair is actually needed. Exact generated commands and outcomes are appended
to the execution note, avoiding a second independently maintained runbook.

## Refreshed execution and remaining J protocol

See the [execution checkpoint](bayesfilter-hmc-gap-closure-execution-2026-09-24.md)
for G's completed prices, H's reporting repair, I's fresh confirmation decision,
J's passed CPU/GPU capacity checks and L's actual regression/build evidence.
The fresh I study retains the original 128/64/64 design on `source-g1`, using
its existing reservation. Historical C2 is preserved separately. No C1 or
learned-quality claim follows from this launch.

Capacity checks remove one specific architectural objection; the next J work
stays in J and requires a separately priced target-specific protocol. The
following design is prepared for that pricing decision, not launched as a
quality study within the 600-second mechanics allocation:

| Element | Banana | Separated mixture |
| --- | --- | --- |
| Target and references | Existing exact banana density/score and its analytic unbending reference. Preserve central and tail cases separately. | Existing exact mixture density/score and independent component-mixture draws. Preserve balanced/imbalanced weights and overlapping/separated modes as distinct scopes. |
| Comparator ladder | Identity; separately fitted diagonal and dense affine maps; configured IAF; configured DSF. Unsupported analytic payloads stay independent references. | Same ladder; the single-stage IAF remains a deliberately capacity-limited baseline. |
| Capacity candidates | Full-reverse IAF compositions and conditional DSF, first with the dimensions of the checked fixtures and then the actual target dimension. | Full-reverse multi-stage IAF or conditional DSF; root-preserving permutations do not fix the demonstrated first-coordinate restriction. |
| Training and selection | Exact batch-native reverse-KL objective with checked parameter directional derivatives; select on disjoint training-validation streams. | Same arithmetic contract, plus validation in both modes; a favorable loss in one mode cannot justify promotion. |
| Downstream criterion | Fresh public candidate-set tuning after each frozen map; model-coordinate mean, variance and tail/reference agreement with MCSE and replicated uncertainty. | Same, adding mode probability and between-mode exploration from both mode-specific and dispersed starts. |
| Veto and repair | Invalid inverse, Jacobian, score, update, frozen payload or downstream numerical health vetoes the map; localization precedes a new fit. | Same; wrong mode probability vetoes map promotion and triggers capacity/objective/budget diagnosis, not more epsilon tuning by default. |

Use the earlier widths `(8,8)` and `(16,16)` and learning rates `1e-4,1e-3`
only as **inherited development hypotheses** from D's price study. Two and four
IAF stages are new capacity hypotheses, with full-reverse permutation; neither
is a reviewed quality default. Start with the existing 512-update pricing rung,
then compare 2,048 and 8,192 updates only after measured full-step cost and
validation curves justify the next rung. The fourfold growth is a convenience
budget ladder, not an adequacy guarantee. Batch size, Adam moments, numerical
precision, initialization scale and any gradient clipping must appear in the
priced design with their provenance; do not transfer Q20 settings implicitly.

Price each target independently, including preparation, every candidate's
fresh tuning, all selected posterior quantities and independent confirmation.
Freeze a prospective multi-seed inventory and its uncertainty criterion before
quality runs; determine the inventory from the desired uncertainty rather than
declaring a small seed count sufficient. Use three disjoint roles for training,
map selection and final reference checks. Check graph signatures, GPU placement,
memory growth, value/score scale and objective gradients before the first long
rung. Holdout loss and short-chain ESS remain explanatory/nomination evidence;
the downstream criterion above governs quality. No method ranking or default
promotion is justified by the present capacity witnesses.

This protocol remains unpriced for the larger configured families and complete
downstream confirmation. The old single-stage training price cannot establish
its affordability. C2's reservation is not available for this work while I is
running. If the complete protocol cannot be funded from the remaining allowance,
record the priced shortfall before launching it; do not replace it with another
short training demonstration.

## Requested continuation: I2, J2, K2 and L2

The owner's latest request authorizes these additional work packages within
the existing program and allocation. They do not restart G–L or the running
I confirmation. The official guide remains `docs/main.tex`.

| Phase | Action now | Completion and next decision |
| --- | --- | --- |
| I2 | Generate and validate the exact C1 suite from B's nominated Gaussian and beta-binomial policies, without the development-only fixed comparator. Save all 512 fit identities, quantity slots, statistical screens and forecast. | Executable design and missing-outcome tests pass. Launch only after the whole study has a measured affordable price and a separately frozen source. The current 137,189-GPU-second forecast is unfunded. |
| J2 | Correct width provenance; implement a configured-family pricing runner with batch-native gradient/update and frozen-map checks, disjoint seed roles, per-stage timing and explicit partial/censored results. Exercise its actual call chain with tiny CPU mechanics fixtures. | A tested pricing command is ready for the next idle GPU window after I. Its short training curves cannot establish map quality. Full target-specific selection and downstream confirmation must be priced before they run. |
| K2 | Preserve the exact MacroFinance input inventory and recheck the two named paths before proposing a consumer launch. | Exact invocation, joint target and matched reference are present, or the existing missing-input disposition is retained. |
| L2 | Remove unused facade imports and redirect configuration/preparation imports to their owners only where actual consumer tests support the change. Review I progress, reconcile receipts, and refresh this program's terminal obligations. | Caller wiring and numerical integration checks pass; concurrent training/Q20 edits and I's frozen source stay separate. Historical stage readers and type-only imports may remain. |

### Evidence, budget and assumptions

The engineering comparator is the tested `source-l3` snapshot. Preserve that
source, then overlay only the files owned by this continuation in a fresh
snapshot. Its tests intentionally hide GPUs. The active I run remains on
`source-g1`, with all 256 planned fits, original statistical screens and its
60,022.255-second reservation unchanged. No new GPU job runs alongside it.

This continuation has a convenience ceiling of **1,800 CPU worker-seconds**
for tests and design diagnostics, drawn from the 68,990.772-second balance.
Record enclosing receipts once, including failures. No GPU seconds are spent
in this continuation while I runs. At most 1,200 of the 3,930.430 unreserved
GPU seconds may later price configured families after a free-device and
whole-inventory review; this is a development cap, not a measured cost or a
new grant. No learned-quality claim or full C1 launch is funded by that cap.

C1 inherits 256 independent complete fits per target, pointwise two-sided 95%
exact intervals and a lower bound of .90, requiring 240 successes. Its
delivery, coverage and joint delivery/coverage remain distinct; every missing
slot is a failure. Two fresh root seeds will be recorded as convenience stream
allocations before execution. The existing posterior settings, data and broad
L grid remain fixed hypotheses. C1's old price excludes the comparator by
subtraction; this forecast requires direct no-comparator repricing before
launch, and contains no runtime-tail guarantee.

J2 retains D's widths 8/16, learning rates 1e-4/1e-3, batch 64, validation
batch 1,024, float64, initialization scale .02 and 512-update price rung as
development hypotheses. The earlier reference to an inherited width 32 was
wrong. Configured two/four-stage full-reverse IAF and conditional DSF capacity
are new hypotheses; fixture dimensions/mixture components are recorded, never
promoted to defaults. Gradient clipping is a numerics-altering control: retain
D's norm-10 setting only as an explicit comparator, report its activations,
and check clipped versus unprotected updates on the same state. Nonfinite
objective/score/update, inverse, Jacobian or freeze mismatch veto that arm.
Loss, short-chain diagnostics and timings are explanatory, not quality screens.
New optimizers, objectives, numerical policies and transport algorithms are
outside this continuation.

### Skeptical pre-execution review

The review corrected the width provenance and rejected four shortcuts: using
C2's scalar target for C1; calling a CPU mechanics run GPU pricing; training a
family without testing the runner actually constructs it; and claiming facade
removal from a static import count. The tests must execute the generated
design/parser and selected consumer/trainer paths. Invalid source, numerical
identity, missing mandatory diagnostics or exhausted allocation stops the
affected work. A poor map, unavailable consumer input or unfunded confirmation
does not stop independent engineering work. Existing historical readers do
not become new public tuners.

The revised continuation passes this review for engineering/design execution.
The outstanding scientific evidence remains C1, complete I, adequately trained
and independently assessed maps, and the exact consumer reproduction. Partial
I results are progress only. Its terminal review must inspect denominators,
missingness, exact bounds, source and cumulative cost before recording any
confirmation conclusion. The automatic finalizer records receipts but does
not replace that review.

Call-chain inspection found an I2 reporting gap before implementation: the
public validation engine already preserves unconditional interval coverage,
but does not aggregate the additional qualified-delivery-and-coverage event.
I2 therefore adds this diagnostic report, without changing sampler decisions,
and tests missing, capped, failed, duplicate and out-of-range replications.
The new screen is declared explicitly by the C1 design; no other study gains
an implicit .90 threshold. The running I source and outputs remain unchanged.

L2's broader tests also found the 18-dimensional LGSSM registry still pins a
target signature predating the September 17 compiled Kalman change. The target
loader correctly refuses that pin in both the isolated and current source.
Before changing it, reconstruct the old signature from the recorded source
hashes and verify exact target/data identity plus batch/scalar numerical parity.
Only a demonstrated stale source pin may be refreshed. Historical tuning and
mass archives retain their original identities; no past result is retagged or
admitted under the new signature. A mathematical/data mismatch instead leaves
this consumer unavailable pending a separate target-specific repair. This is
a localized L2 repair within its remaining CPU allocation, not a new campaign.

### Executable continuation checkpoint

I2's unchanged suite and 512 prospective fit records are in
`artifacts/hmc-gap-closure-phases-2026-09-24/i2-design-r1/data/`. Its parser
accepts both targets. The coverage report now emits all three declared events,
with exact pointwise bounds and missing-fit denominators. Scientific C1
confirmation remains unfunded; the 137,189-second GPU estimate is a subtraction
forecast requiring direct no-comparator pricing and enclosing overhead.

J2's executable pricing command is
`docs/benchmarks/price_hmc_configured_maps_2026_09_24.py --output <fresh-root> --seconds <reviewed-cap>`.
Use the existing trusted bounded launcher, a free GPU UUID and the reviewed
1,200-second development ceiling after I, with inner budget allowing launcher
overhead. The inventory has 48 arms: two development targets, three configured
families (IAF with two/four stages and DSF with two), widths 8/16, two learning
rates and two seeds. The targets are banana bend .5 and mixture separation 5,
weight .3. These are D's development target scopes, not the full future
central/tail and balanced/separated quality inventory. The tiny `--cpu-smoke`
route ran all six target/family combinations and is mechanics evidence only.

Configured-family initialization differs from D's legacy zero-output IAF:
`glorot_small_final`, final-weight scale .02, bounded-tanh scale cap 1 and
two DSF components are explicit new development hypotheses. The component
count and full-reverse permutation come from J's capacity fixtures; the
initializer comes from the existing supported implementation. The API's
float64 inverse tolerance 1e-11 and positive DSF slope floor 1e-6 are inherited
arithmetic hypotheses, checked by inverse and gradient diagnostics, not promoted
scientific defaults. Adam (.9, .999, epsilon 1e-8) and clip norm 10 reproduce
D's optimizer comparator. A paired identical-state update records whether
clipping changes parameters; every training activation is counted. The two
prospective seed roots are 2026092491/92, with separate initialization,
derivative, training and development-validation roles. No final-reference
stream or posterior fit is consumed by pricing.

L2's LGSSM audit reproduced the old signature exactly by substituting only the
pre-3582b4ac5 Kalman source hash. The registry pin is refreshed; the old signature
still rejects. Numerical batch/scalar, XLA, target/data identity and invalid-row
tests verify this repair. Three direct legacy imports remain: two historical
stage/resume modules and one type-checking-only dependency. No wholesale facade
removal is claimed. K2 again found both exact MacroFinance paths absent in both
checkouts. The terminal I audit remains the next scientific step, followed by
GPU pricing when hardware is free; no automatic new phase or scientific
promotion follows a successful service exit.
