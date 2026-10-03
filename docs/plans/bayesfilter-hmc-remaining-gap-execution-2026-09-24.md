# Remaining-gap program: execution checkpoint

The user authorized continued execution on 2026-09-24. A is complete for its
scoped engineering checks. B's diagnosis and all eight full GPU fits completed;
C2's first confirmation attempt stopped after three per-fit timeouts; nine of
256 planned fits completed. No confirmation process remains active.
D's 16 training-price arms and F's affected tests/book build completed; E's
exact inputs remain unavailable. See the
[current result](bayesfilter-hmc-remaining-gap-results-2026-09-24.md).
Execution remains within the same
[program](bayesfilter-hmc-remaining-gap-program-2026-09-24.md).

Pre-execution skeptical audit: the checkout remains at `622d9a9e`; unrelated
concurrent edits now include candidate execution, registry and NeuTra modules.
Use committed-base source snapshots with an explicit owned overlay for tests
and experiments. Preserve those unrelated edits. Do not attribute results from
the clean snapshot to untested concurrent changes.

Evidence contract for A: preserve all 272 M21 planned full fits, identify their
distinct failure and interval outcomes, and reproduce the audited aggregate
counts. Independent reference checks must compare the correct ESS estimator;
Stan/ArviZ and the named TFP precision estimator are distinct. Public pipeline
tests must establish actual quantile wiring, policy identity and unavailable
evidence. Arithmetic mismatches or missing historical rows block interpretation
until repaired. Existing-chain reanalysis is development, not confirmation.

The reference environment will be an isolated venv over the installed tfgpu
environment, with any added diagnostic packages pinned and installed only
there. CPU tests hide GPUs before imports. GPU tests require trusted execution,
growth and recorded device/XLA settings. No broad environment mutation.

Resource ceilings remain those in the program: A 12,000 CPU / 1,200 GPU seconds;
B 18,000 CPU / 9,600 GPU seconds. Opening total allowance is 71,946.61478971103
CPU and 74,041.93929888518 GPU seconds. Each numerical/check command receives a
receipt in `artifacts/hmc-remaining-gaps-2026-09-24/`; failed attempts count.
No new stochastic ranking or default promotion follows from A's tests.

This audit passes for A. Its assumptions are checked by the per-fit inventory,
the isolated reference environment and exact public-route regressions. The
next scientific experiment will use B's mechanism-specific design before launch.

## B1 saved-array diagnostic, declared before execution

A reconciled all 272 fits. The newly independent TFP quantile reference exposed
a tied-cutoff rounding bug (257 draws, four chains, .95 quantile): indicator
ESS 75.6345 versus independent 84.4542 for one quantity. Reusing the existing
tie-preserving pooled percentile is a mathematical repair, not an ESS-policy
change. The quantile method now has an explicit corrected identity. Other
failures in the first A suite were the snapshot's missing native library and
two tests for concurrently developed Q20 consumer code absent from the committed
baseline. Those are recorded as environment/source-scope limitations; the Q20
work is preserved and is not silently counted as validated.

The next diagnostic uses **all** saved Gaussian/beta-binomial selected fixed
and stopped arrays from A, with their checked tensor hashes. It compares the
recorded interval, corrected quantile MCSE symmetric interval, and independent
direct .025/.975 order-statistic interval on each identical array. No new HMC
draws, selection or stopping decisions occur. It also evaluates whole-window
R-hat/ESS for the seven Gaussian warmup caps versus their recorded recent-window
values. Budget: 1,200 CPU seconds within B. CPU hides GPUs; TensorFlow diagnostics
include existing CPU XLA helpers and reporting graphs, not a production HMC run.

Primary engineering criteria: every planned row/quantity remains represented;
read checksums and mathematical identities hold. Coverage/error/SE differences
on these development arrays explain or nominate repairs only. No ranking,
fresh coverage confirmation, new stopping result or default promotion follows.
Moment assumptions are valid for the declared Gaussian and beta laws. A
missing/corrupt input vetoes that comparison; time exhaustion leaves the
diagnostic incomplete rather than shrinking its denominator. Preserve the
original stop and candidate membership. Independent SciPy diagnostic references
are permitted only here. Record elapsed time, command, source and result.

Skeptical review passes: this discriminates tied-cutoff arithmetic, interval
asymmetry and recent-window information without claiming that a changed
estimator would stop at the same time. It cannot identify initialization bias
merely from favorable whole-window R-hat; that distinction remains explicit.

## B2 prospective GPU development inventory

The saved-array diagnostic completed all 256 planned Gaussian/beta fits.
The tied-cutoff fix changes none of their recorded median MCSEs. Direct 95%
order-statistic intervals change only one Gaussian stopped and one beta fixed
coverage outcome. This does not support promoting a different interval rule.
Five of seven Gaussian warmup caps pass the same 1.05 screen when evaluated
over all 10,000 archived warmup draws; two still fail. This supports testing
the existing longer-window/count candidate, not declaring burn-in complete.

Run eight fresh full GPU/XLA fits: two Gaussian and two beta-binomial baseline
fits, plus two of each using the already proposed allocation/member policy.
Preserve exact target/data and all tuning settings from M21. Import the
candidate's count/window settings from M29 and its first-verified member rule.
Freeze independent seeds 2026092401 through 2026092408, in model/arm/repeat
order. These are reproducible development identifiers, not statistical evidence.
The original MCSE tolerances are .05 for Gaussian and **.005 for beta-binomial**;
checking the actual JSON exposed and corrected the plan's shared-.05 wording.
Both use lugsail and the corrected quantile arithmetic. Retain all verified
members and independent fixed comparators. A new code version is not an exact
replay of the historical baseline. Different fitted members prevent attributing
all observed differences to a single count/window choice.

Each fit has a 900-second timeout, above the recorded 286--347-second full-fit
observations as a development margin, not a tail guarantee. Total B GPU ceiling
stays 9,600 seconds; eight fit ceilings use at most 7,200. One worker at a time
on the idle GPU selected by trusted preflight; memory growth and XLA are required.
Stop the affected run on invalid source, wrong data/target, corrupt outputs,
missing growth or health invalidity; caps are retained results. Numerical
candidate rejection does not prevent the next predeclared fit. Report runtime
components and member/count outcomes. These eight fits cannot establish a
coverage rate, ranking or new default. C's funding decision follows actual
prices; do not launch its complete study speculatively.

Skeptical audit: exact original JSONs fix the baseline and differing model
tolerances; genuine public full-fit calls preserve the statistical unit;
all outcomes remain in the development inventory. Fixed comparators test
mechanisms, not an independent replication count. The experiment is adequate
for execution/activation/pricing, not scientific confirmation.

## C2 diagnostic implementation while B runs

Implement the separately named normal-conjugate reference-mean engine in the
existing validation CLI. Each replication generates fresh data, calls the
ordinary public preparation/tuning/posterior pipeline and independently
computes lugsail MCSE on the selected model-coordinate draws. This diagnostic
NumPy calculation is confined to `testing/inference_validation`; it is not a
runtime decision or tuner. Check its complete-batch formula against the
TensorFlow estimator on persistent, antithetic, tied and non-divisible arrays.
Keep nonpositive per-chain variance and insufficient batches unavailable.

Freeze explicit alarm alpha and MCSE/known-posterior-SD target in the design.
Require the predeclared first-verified member, complete numerical evidence,
warmup exclusion and qualified posterior. Null unavailable slots count as
possible alarms; defect unavailable slots count as nondetections. Preserve
all planned IDs, source/design identity, data and stream identities. The
engine's repeated-fit size/power has not been established by arithmetic tests.

Skeptical audit passes for implementation and short integration checks:
independent reference/variance arithmetic, honest missingness and genuine
public fits answer the engineering question. The analytic reference never
reaches tuning; generated truth is assessor-only. An actual full-fit activation
will use C's development allowance after the focused tests pass. No confirmation
inventory is launched before its complete cost is affordable. C's confirmation
sample sizes, thresholds and scientific nonclaims are unchanged.

B1 continuation: compare the existing lugsail and autocorrelation mean-MCSE
options on the same saved Gaussian/beta arrays, with the original stop times,
references and full 256-fit inventory. Budget 1,200 CPU seconds; no new samples.
Preserve unavailable per-chain/batch evidence. Estimated-SE/empirical-error and
coverage counts are descriptive mechanism checks, not an estimator ranking or
a default decision. No dependence-adaptive bandwidth is introduced without its
selector source audit. The unchanged original AR(1) exact-covariance result
already establishes that sqrt(n) batches can be inadequate under strong
persistence; these saved arrays test its relevance to the actual failed fits.

C2 activation/pricing inventory, declared before launch: one null full fit, an
exact paired no-op full fit, and independent .25/.5 posterior-SD defect fits.
Use seeds 2026092411 (null/no-op pair), 2026092412 and 2026092413. Preserve M34's
normal law (tau=2, sigma=1, now explicit n=6) and native tuning. Generate fresh
data for each independent fit. Use B's baseline count policy explicitly:
warmup minimum 2000, window 1000, chunks 500, cap 10000; retained minimum 1000,
chunks 500, cap 10000. This is a C2 hypothesis, not a universal allocation.
The declared mean-MCSE/SD ratio .05 gives absolute .02; median precision keeps
the public controller's same absolute requirement. Use nominal alarm .01.
The paired no-op is an engineering control, not another null replication.
Four outer ceilings of 900 seconds plus at most 40 seconds termination grace
fit within C's existing reserve. One GPU worker follows B's queue, never
concurrent on that device. Numerical caps remain unavailable outcomes, while
source/target/health/archive invalidity stops the affected run. Artifacts are
under `c-r1`; cold full-fit costs determine whether a complete statistical
cell can be funded. These four activations establish neither size nor power.

## D target-specific learning-curve pricing contract

D's serious map-quality promotion remains separate. Replace M37's unpriced
48-fit-per-target Cartesian grid with a bounded developmental price/scale
study: banana bend .5 and mixture separation 5, weight .3, each with widths
8/16, two tanh hidden layers, learning rates 1e-4/1e-3 and two initialization
seeds (2026092421/22). These are explicit M37 capacity/optimizer hypotheses,
not defaults transferred from Q20. Batch 64 and cumulative update looks
32/128/512 are convenience choices to expose cost, gradient validity and
learning-curve shape; they are not assumed adequate training. The fixed
1024-row independent base validation bank measures loss behavior only.

Use the committed trainer behind the tested M37 composition on the same
isolated source, explicit complete configuration and batch-native target,
GPU/XLA with growth. Concurrent transport-core work is excluded, so these
prices and maps cannot establish readiness for that untested revision.
Sixteen arms have a combined 1600-GPU-second ceiling from D's 7200-second
allocation. Finite-difference the actual scalar objective along its computed
gradient at initialization, including h and h/2 checks (h=1e-5, a double-
precision diagnostic convenience). This one-off derivative check may call the
eager diagnostic API; every repeated update/validation uses stable graph
signatures. Veto nonfinite objectives/gradients, source mismatch, scalar batch
fallback, failed graph signatures, failed frozen reload/inverse/Jacobian or
budget exhaustion. Save every curve and map; a failed arm stays failed.

No loss-based winning map is promoted. No HMC run uses these maps in this
pricing study. Identity/untrained behavior provides the zero-update baseline;
fitted affine and analytic geometry comparators remain required before a
serious quality comparison. The result will revise target-specific adequate
training/allocation estimates and may show that the 512-update convenience
ladder is inadequate. It cannot establish model-coordinate posterior quality,
mode exploration, superiority or a universal learning protocol. This bounded
study answers a narrower cost/default-assumption question that M37 left open.

B2 resource repair: all four Gaussian fits delivered qualified posteriors.
The next beta fit failed its preflight before sampler execution because GPU 1
was acquired by a dsge_hmc worker (PID 1566809). Preserve that failure and
charge its 0.120 seconds. Continue the same four beta designs/seeds on GPU 2,
the same RTX 4080 SUPER hardware class, in `gpu-queue-r2`. No other process
is interrupted. Trusted inspection found only its remote-desktop process.
The fourth Gaussian's 730-second timing overlaps the other GPU-1 job and is
therefore confounded by contention; do not attribute that runtime solely to
its L=25 member or use it as an uncontended performance comparison.

F checkpoint: the complete validation suite ran 360 tests; 359 passed and the
coverage-renderer test could not load its omitted script from the isolated
snapshot. Earlier collection attempts similarly found absent historical
benchmark/script fixtures. Restore only those committed inputs, retaining the
same numerical source, and rerun the one affected renderer module. These are
snapshot packaging failures, not silently omitted tests. The full suite includes
six-model public tuning/replay, supplied exact/residual funnel maps, global
mixture diagnostics, constrained-target checks and training/freeze/retune
composition. CPU mechanics tests do not close scientific posterior-quality
claims. The official main.tex build passed after restoring its five archived
figure inputs; PDF pages 432, 474 and 475 were rendered and visually inspected.

C2 isolation audit: the new engine currently loops over complete fits in one
TensorFlow process. That is adequate for the one-fit activation designs, but
would revive the graph/cache accumulation already repaired for other engines.
Reuse the existing fit-process supervisor, identities, cumulative retry budget,
exit receipts and full-denominator aggregation. A child that writes a numerical
assessment and then exits abnormally remains unavailable; its saved numerical
record is preserved. Test real independent generated datasets in two child
processes, restart without replay, source rejection and failed/missing exits.
This is a bounded infrastructure repair; it changes neither the detector nor
its thresholds or scientific contract. Skeptical audit passes because separate
children preserve the complete-fit statistical unit and the same public sampler
path, while process failures remain distinct from numerical outcomes.

## C2 confirmation freeze and funding decision

B2 completed all eight fits. Every selected member passed its declared posterior
checks and every verified sibling remained retained. One longer Gaussian
stopped mean interval missed the exact mean. This is development evidence only;
two fits per arm cannot establish coverage or a ranking. B1 did not justify
replacing either the lugsail mean estimator or the quantile interval rule.

C2's baseline/no-op records and saved draws match exactly. The quarter- and
half-SD shifts executed in the sampled potential and were detected; all three
independent fits qualified. Full observed prices were 216.591, 242.501 and
210.633 GPU seconds (no-op 215.272). D completed its 16 training-price arms
in 60.995 GPU seconds, with gradient and frozen-map checks passing. Loss was
still changing at 512 updates, especially for the mixture, so this has not
established an adequate training budget or learned-map quality.

The complete C2 inventory is now affordable within the existing allowance.
Freeze `c-r1/confirmation-designs-r1/suite.json`: 128 null fits, 64 quarter-SD
fits, 64 half-SD fits; seeds 2026092431/32/33 are prospective reproducibility
identifiers, disjoint from development. Preserve the exact C2 target, native
search, first-verified rule, 2000/1000 warmup/retained minima, 10000 caps,
lugsail and .02 absolute MCSE (.05 of the exact .4 SD). Alarm alpha is .01.
Keep the two-sided 95% exact rate bounds: null upper <=.10, each defect lower
>=.80. No-op is excluded from the confirmation denominator. Missing null slots
count as possible alarms; missing defect slots count as nondetections.

Frozen numerical source is `c-r1/source-r7`, identity
`dfcdec3c3084b3ff8aa7c5fa3ad327e7601f0e042ccaedbc602999fbd5d4bf2e`.
The complete-fit isolation integration passed 60 focused tests, followed by
three response/public-isolation checks and 19 focused assessment tests. A
reporting repair ensures that an occasional nominal null alarm is interpreted
through the predeclared rate screen; capped completed fits are not labeled
complete assessments. These reporting changes do not modify the sampler.
All confirmation fits use fresh processes and fresh data. The parent never
initializes TensorFlow. Source/design changes invalidate resume; incomplete
processes retain their remaining cumulative allocation and cannot count as
detections. Prior numerical results and streams are never reused as fresh fits.

Measured development prices imply 56,724.217 GPU seconds for the entire C2
inventory. Allocate 32,000 / 16,500 / 16,500 seconds to its three cells and a
65,100-second outer ceiling including process cleanup. The approximately 14.6%
margin above that descriptive forecast is an administrative allocation, not a
runtime probability guarantee. Each fit retains its original 890-second cap;
the cell budget remains the hard aggregate ceiling. Unexpected caps leave all
planned slots represented. Transfer unspent A/B/E and shared GPU allocations
to C; do not add a new grant. Completed charges and this reservation leave
more than 5,000 GPU seconds for F and localized repairs. No confirmation
sample count or threshold is reduced to fit the budget.

C1 remains unfunded alongside C2. The no-fixed-comparator price scenarios
are 65,584.976 Gaussian plus 71,603.714 beta-binomial GPU seconds, using the
uncontended development observations and subtracting their fixed-arm time.
That subtraction is a forecast, not a new no-comparator timing experiment.
The full C1+C2 point forecast is about 53.86 GPU hours before further training
and consumer work. Funding C2 does not close C1 or reject its candidate.

Skeptical pre-launch audit passes for C2: the original normal-conjugate law,
actual shifted target and exact independent assessor agree; full-fit units,
missingness and prospective seeds are explicit; activation is not treated as
power; CPU tests are not relabeled GPU confirmation; the full inventory has
an explicit bounded price and source; and every posterior quantity remains
separate from tuning admission. The main uncertainty is run-to-run price and
posterior availability. Numerical rejection continues to the next declared
fit. Infrastructure/source/artifact invalidity stops the affected cell for
repair; no repeated confidence-bound looks can stop or enlarge a cell.

Execute the existing public validation CLI on GPU 2, XLA on, memory growth
verified in every child, one worker. The exact command and trusted preflight
are recorded in `c-r1/confirmation-run-r1/manifest.json`. Its output suite is
`c-r1/confirmation-run-r1/suite/`; the public CLI emits the run index, each
complete-fit assessment and the final report. The result must retain separate
null calibration, two defect-rate bounds and availability. Passing these
screens would establish only the named location-defect detector on this law;
it would not establish general posterior correctness, C1 coverage, learned-map
quality, or a new tuning/posterior default.
Concurrent chapter26b/bibliography work is excluded from that build.

## Publication checkpoint and stopped confirmation

Before the requested commit/merge/push, the terminal receipt showed that C2
ended at 2026-09-23 22:00:12 UTC with exit code 1 and 5077.745 GPU seconds.
Null fit 5, quarter-SD fit 4 and half-SD fit 0 each reached the 890-second
per-fit ceiling. Five null and four quarter-SD fits completed and qualified;
the respective alarm counts were zero and four. The full 128/64/64 denominators
remain intact and all three assessments say `calibration_incomplete`.

This is an execution failure requiring diagnosis, not evidence against the
detector or research direction. Retracing warnings alone do not identify the
cause. Preserve the failed attempts and frozen source. Before any continuation,
inspect saved timing/tuning evidence and the cumulative fit/cell budgets; do
not grant a fresh timeout to an exhausted fit or reuse completed fits as new
replications. The ledger charges the outer receipt once, without adding the
nested child times, and retains only the unspent C2 allocation for continuation.

The publication audit uses the already tested committed-base overlays as its
baseline. It includes owned code, tests, guide edits and compact evidence,
excludes unrelated concurrent Q20/NeuTra changes, and preserves local raw
draws/source snapshots/build trees. Test passes support the narrow repairs;
they do not close C2. Inspect the actual remote divergence and rerun focused
checks if merging changes tested behavior. Stop publication on unresolved
conflicts or failed checks. No new numerical policy or campaign is introduced
by this Git operation.
