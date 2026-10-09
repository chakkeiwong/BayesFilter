# Inference validation suites

The October 2 [acceptance repair plan](../plans/bayesfilter-hmc-acceptance-decision-repair-plan-2026-10-02.md)
and [robustness matrix](../plans/bayesfilter-hmc-acceptance-robustness-matrix-2026-10-02.md)
define the experimental v7 tests. `bayesfilter.testing.acceptance_validation_inventory`
maps root causes to exact test sources and model configurations. An inventory
entry is not a passing result. The CLI
`python -m bayesfilter.testing.acceptance_decision_validation` provides
`preflight`, `calibrate`, and `models` commands with explicit frozen JSON
configurations. Calibration checks known finite-trial expectations; model runs
call the public tuner and record terminal outcomes, all attempts, verification,
export and checkpoint reload. The official tuning chapter in `docs/main.tex`
and its [agent reference](../reference/hmc-tuning-interface.md) explain the same
procedure. Neither CPU engineering tests nor acceptance qualification certify
posterior convergence or a new default.

Fast regressions cover `test_hmc_acceptance_protocol.py`,
`test_hmc_acceptance_statistics.py`,
`test_hmc_acceptance_accounting.py`, `test_hmc_acceptance_adversarial.py`,
`test_hmc_acceptance_inventory.py`, and `test_hmc_acceptance_calibration.py`.
`test_hmc_acceptance_ssm_recovery.py` adds actual QR LGSSM and nonlinear SSM
fresh-process recovery, with uninterrupted same-stream numerical references.
They are included in `hmc-regression-tests.txt`. The numerical trial,
fresh-process recovery, prepared-model and scalar-layout checks are in
`hmc-acceptance-integration-tests.txt`; run with
`python -m pytest @docs/validation/hmc-acceptance-integration-tests.txt` in the
same CPU reference environment described below. These include informative
positive QR/nonlinear SSMs and exact/residual maps, explicit low-budget
abstention, and a qualified mixture whose separate posterior assessment fails
because of a missed mode. They are larger than the fast arithmetic checks.
The selection also covers emitted-analysis and trial-seed reuse, fresh scoped
member/checkpoint replay with corruption controls, and fixed-denominator
confirmation reporting with independent binomial references. These checks
also cover live target-probe graph reuse with changed data, model constants,
callbacks, source, geometry and starts. They
protect the recent cost repairs and the interpretation of future confirmation;
they do not replace independent full-search replications.
Keep multi-seed delivery, full-cap statistical calibration and GPU/posterior
checks in the separate bounded validation tier described by the matrix. The
`models` CLI is deliberately CPU-only; GPU delivery uses the reviewed GPU
runner and verifies native evidence-chunk placement, memory growth and XLA.

The active continuation is the
[HMC repair master program](../plans/bayesfilter-hmc-repair-master-program-2026-09-16.md).
Its current [scoped v7 release plan](../plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md)
requires full searches in addition to the earlier prepared single-pair checks.
`scripts/prepare_hmc_v7_release_source.py --output <new-directory> --build-custom-op`
assembles the tracked baseline plus explicit HMC overlays and builds the native
TensorFlow dependency in that tree. Use the selected TensorFlow environment to
run it; the source and build manifests identify all inputs. This is an isolated
validation assembly, not a claim that uncommitted overlays are already released.
`test_hmc_acceptance_numerical_envelope.py` checks independent high-precision
bounds; `test_hmc_acceptance_release_validation.py` covers v7 preparation and
repair/verification recovery; `test_hmc_acceptance_posterior_integration.py`
checks repeated exact-reference sequential handoffs. CPU is an explicit
reference tier. The release pricing script requires trusted GPU execution and
its documented budget; it cannot launch confirmation or mark release readiness.
Its [M0--M6 result](../plans/bayesfilter-hmc-repair-master-result-2026-09-17.md)
records the earlier bounded repairs. The [M7 execution note](../plans/bayesfilter-hmc-repair-m7-result-2026-09-17.md)
records verification, frozen GPU results, remaining work and cumulative budget.
The [M15 result](../plans/bayesfilter-hmc-repair-m15-result-2026-09-21.md),
[M16 result](../plans/bayesfilter-hmc-repair-m16-result-2026-09-21.md) and
[M17 result](../plans/bayesfilter-hmc-repair-m17-result-2026-09-21.md) contain the
latest replicated fits, diagnostic sensitivity and geometry/route/reference
matrix. The machine-readable `program-progress.json` beside the phase artifacts
records each terminal ledger and reviewed next design. Phase completion and
scientific-gap closure are separate fields.
`scripts/prepare_hmc_repair_suites.py` resolves the initial development suites
archived as `suites-r1`. The measured GPU continuation uses the master program's
`suites-r2/fresh-gpu.json`; the initial resource caps are historical allocations.
Both use this same executor. The `acceptance` engine's `controller` route uses
synthetic marks. Its `frozen` route measures stationary TFP-HMC acceptance;
`prepared` executes repeated public single-pair tuning searches with fresh
verification and an independent stationary-acceptance reference. These routes
answer different questions and must not be pooled.
Mixture designs can declare `global_quantities: ["left_mode_probability"]` and
use `mode_dispersed` starts when the supplied chain bank is preserved.
Stopped-interval reports compare that probability with the independent mixture
CDF and keep missing posterior outputs in its denominator.
Ordinary designs expose the optional `preparation_bound_expansion_steps` setting.
SBC summaries distinguish `missing_dataset_records` from known unstarted work;
`unstarted_datasets` is null when incomplete shards prevent determining it.

The September 18 continuation is M8 within that same master program, with the
additional 48 CPU and 24 GPU hours reconciled against all earlier costs.
`scripts/prepare_hmc_m8_suites.py --output <fresh-directory>` resolves its
centered/noncentered funnel, repeated public acceptance, defect-power and
stopped/fixed calibration designs. Versioned suites and immutable package
snapshots live under `docs/plans/artifacts/hmc-repair-master-2026-09-16/m8-r1/`.
The script resolves plans only; launches follow the master's pilot cost checks.

For the earlier M7 continuation,
`scripts/prepare_hmc_repair_suites.py --stage m7 --output <fresh-directory>`
resolves its energy checks, fixed-look kernel-power experiments and four
automatic-pipeline cost pilots. It launches no workers. The pilots use
beta-binomial, LGSSM location, funnel and rotated Gaussian targets, select one
member by identity before posterior draws, and keep all unassessed siblings.
One pilot per target supplies cost and failure information, not calibration.

These suites exercise distinct questions through shared target definitions,
independent references, the public HMC tuners, saved observations, and common
reports. Tuning retains every verified candidate; R-hat, ESS, and MCSE remain
posterior diagnostics. The package cannot issue a new tuning authority.

The architecture and source survey are linked from the
[execution specification](../plans/bayesfilter-inference-validation-execution-2026-09-15.md).
Each resolved design declares development or confirmation status. A completed
process is not necessarily a complete statistical experiment, and neither
status grants default readiness. A profile name describes execution scope, not
strength of scientific evidence.

The [phase-two plan](../plans/bayesfilter-inference-validation-phase2-plan-2026-09-16.md)
preserves the earlier continuation. Current completion, missing evidence and
execution order belong to the master program linked above.
The completed [funded campaign](../plans/bayesfilter-inference-validation-24h-campaign-2026-09-16.md)
used 21.80 charged CPU hours and 11.17 GPU worker-hours within separate 24-hour
allowances. Its [results](../plans/bayesfilter-inference-validation-24h-result-2026-09-16.md)
include complete prepared-route rank evidence, ordinary-search availability
failures and a missed-mode counterexample despite passing local posterior checks.
The offline command
`python scripts/audit_inference_validation_campaign.py <campaign-root> --output <new-file.json>`
reconciles saved attempt costs, result/tensor hashes and incomplete full-SBC
datasets without launching a sampler. A clean audit establishes consistency of
the saved records; it does not establish statistical calibration or fund another
campaign.

## Commands

Run from the repository using the existing `tfgpu` environment:

For the selected existing CPU regression tests (pytest 8.2 or newer):

```bash
CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_TEST_DEVICE_SCOPE=cpu \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 \
python -m pytest @docs/validation/hmc-regression-tests.txt
```

This selection checks candidate lifecycle, graph reuse, timeout/accounting,
posterior diagnostics, actual Kalman/nonlinear filter adapters, and one
prepared-route tuning/reload/posterior mechanics case. Its 127 tests passed in
161 enclosing seconds on September 25, 2026; runtime is machine-dependent.
The selected pipeline case uses the dense `lgssm_location` likelihood and does
not substitute for the actual-filter bridge. The bridge and stressed SSM
profiles are covered by the separate selection below. This regression
selection does not establish GPU readiness or calibration.

The actual-filter bridge has a separate bounded CPU selection:

```bash
CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_TEST_DEVICE_SCOPE=cpu \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 \
python -m pytest @docs/validation/hmc-ssm-regression-tests.txt
```

It exercises the QR Kalman and deterministic sigma-point adapters through both
ordinary and prepared public tuning routes, named persistence/noise/horizon
profiles, independent mechanics references, reload, retained-candidate
inventory, the K0--K7 campaign targets, reference-grid sensitivity, negative
controls and the unavailable-reference schema. The CPU selection checks
graph/XLA numerical equivalence in an intentionally GPU-hidden process; it
does not establish trusted GPU execution or posterior accuracy.
The September 26 broad selection passed 142 tests in 707.7 enclosing seconds;
24 overlapping focused checks passed after the final controller and K0 fixture
repairs. Tiny warmup limits deliberately leave most posterior assessments
inconclusive. The exact receipts and frozen launch command are in the
[preparation result](../plans/bayesfilter-hmc-state-space-preparation-result-2026-09-26.md)
and [48-hour plan](../plans/bayesfilter-hmc-state-space-48h-plan-2026-09-25.md#launch-after-c1-settlement).

The complete unpriced 48-hour inventory is
`docs/validation/state-space-48h.json`. `plan` is safe before the C1 service
settles; `run` must use a fresh versioned output root only after the terminal C1
receipt and additive GPU ledger are reconciled. The suite encodes the complete
K0--K7 32-slot inventory: K0--K5 each have two datasets and two sampler seeds,
K6 has four seeds on its checked multivariate fixture, and K7 has two datasets
and two seeds. Each slot retains every verified tuning candidate; K0 and K7
assess two predeclared distinct-L members and other slots assess one. The
nonlinear K7 route is a declared sigma-point target and still needs trusted
GPU/XLA preflight; its posterior comparison is against the same approximation,
while true-model integration is a separate short-horizon diagnostic.

The executable preparation and launch controller is
`scripts/prepare_hmc_state_space_campaign.py`. `prepare` creates frozen
datasets, reference sensitivity receipts, comparator diagnostics and unpriced
stage suites. `freeze` records the source used for GPU work. Invoke the frozen
copy of the launcher for GPU stages so live-checkout edits cannot change the
controller. After C1 has a
terminal receipt, `run --stage preflight-mechanics`, `run --stage
preflight-pipeline`, `run --stage pricing`, and `run --stage main` use the
existing inference-validation executor, workload-aware process supervision,
the additive grant ledger and fresh versioned outputs. A complete uncensored
pilot for each execution shape is required before a main slot is funded;
unaffordable slots remain in the original denominator. `report` preserves all
planned slot dispositions. No stage treats R-hat, ESS or MCSE as tuning
membership. Do not run the unpriced inventory directly: the campaign controller
writes the funded main suite after checking pilot receipts and the settled grant.

For the separately budgeted validation suites:

```bash
python -m bayesfilter.testing.inference_validation list
python -m bayesfilter.testing.inference_validation plan docs/validation/fast-cpu.json
python -m bayesfilter.testing.inference_validation run docs/validation/fast-cpu.json --output /tmp/my-new-validation-run
python -m bayesfilter.testing.inference_validation run docs/validation/fast-cpu.json --output /tmp/my-new-validation-run --resume
python -m bayesfilter.testing.inference_validation report /tmp/my-new-validation-run
python -m bayesfilter.testing.inference_validation assess /tmp/my-new-validation-run --output /tmp/my-new-rank-assessment.json
```

Use a fresh versioned output directory for each evidence version. Resume checks
the suite and source identity, skips completed jobs, preserves failed attempts,
and charges consumed worker time. Native numerical checkpoints retain completed
tuning and posterior chunks. A coordinator interruption with unknown worker
completion charges the reserved budget conservatively. An ordinary advisory lock
prevents accidental concurrent coordinators. GPU runs require trusted execution;
workers verify memory growth before initialization. CPU profiles deliberately
hide GPU devices and use a documented non-XLA diagnostic exception.

When a suite omits `required_coverage`, every planned design has its own
requirement. Completing one epsilon or kernel-power cell cannot complete another
cell in the same category. An explicit category-level requirement asks for any
matching executed and assessed design. Historical plans without design IDs
retain that category-level meaning; inspect their individual design rows to
determine whether the whole planned suite completed.

`--max-workers N` supervises up to N separate worker processes with one index
writer. Budgets count summed worker wall time, including failed and cancelled
attempts, rather than elapsed calendar time. Predeclared SBC groups can combine
independently seeded dataset shards from the same suite and frozen source.
Aggregation verifies membership, numerical settings, source and result checksums;
missing groups and fits stay in their original denominator. Pilot, changed-source
and repaired confirmation experiments are not pooled.

For numerical search, accuracy and stopping designs, `options.isolate_fits`
can enable a new child process for each complete fit. Declare
`options.fit_process_timeout_seconds` within the total design budget. The
coordinator stays free of TensorFlow and keeps each child's normal exit or
failure, log, source/device manifest and memory/graph measurements. All tuning
members and posterior archives remain on disk. Total coordinator wall time
already includes these sequential children; do not charge their nested receipts
a second time. Isolation is optional, and unsupported engines fail validation.
An abnormal process with a completed assessment is retained for audit and is
neither automatically rerun nor counted as a successful replication. Partial
work may resume using native checkpoints, with previous attempts charged.

An explicit `timeout_policy.extension_mode="observed_intervals"` retains
trusted contention intervals observed during execution and publishes earned
allowance before the cooperative deadline, provided durable numerical progress
is recent. Extra time remains bounded by both `max_extension_seconds` and the
enclosing budget. This measures a scheduling allowance, not lost GPU capacity.
The compatibility default remains `latest_sample`. Optional
`gpu_admission_wait_seconds` bounds admission waiting. Its compatibility mode
`gpu_admission_mode="idle"` waits for foreign compute processes to leave.
`gpu_admission_mode="shared"` permits them when trusted device telemetry shows
available memory. Unknown telemetry cannot establish availability. Waiting
consumes the enclosing budget; shared admission does not promise throughput.

`max_contention_retries` permits a declared number of automatic checkpoint
retries after an incomplete resource stop with observed contention. All attempts
consume the original base-plus-extension cap, and a retry cannot renew it.
Final assessments and numerical failures are not retried. The state-space
campaign uses shared admission and at most one recovery attempt under the
[September 29 repair](../plans/bayesfilter-hmc-shared-gpu-recovery-2026-09-29.md).

An exhausted local fit cap can receive an explicit additional allocation from
the remaining campaign budget through
`campaign_recovery.continue_frozen_fit`. The original numerical worker, design,
seeds and checkpoint paths stay fixed; all prior attempts count toward the
enlarged cumulative cap. Completed assessments are reused, and numerical
failures remain a separate diagnosis. The executable
`scripts/run_hmc_ssm_pilot_repair.py` applies this mechanism to the preserved
state-space pilots under the [pilot repair plan](../plans/bayesfilter-hmc-ssm-pilot-repair-2026-09-29.md).
`scripts/run_hmc_ssm_priced_main.py` checks complete cumulative prices against
the original numerical workload before explicitly reallocating main-lane time.
It keeps all 32 original slots in the inventory, checks the latest-start cutoff
before each cell, and reports unfunded or unstarted slots. A timed-out tail,
changed reference workload, or successful preparation alone cannot set a price.
Its continued prices include original and resumed work. The regression list
includes actual K0/K7 CPU checkpoint continuations and tests of exhausted caps,
immutable evidence, numerical-failure exclusion and remaining campaign limits.
Its ten C1 original-seed recoveries are already complete and remain separate.

The active [October 2 continuation](../plans/bayesfilter-hmc-ssm-progress-continuation-plan-2026-10-02.md)
uses `scripts/run_hmc_ssm_pooled_campaign.py` for one queue of unfinished pilots
and main fits. Progressing jobs can receive more than three turns within the
global allowance; completed pilots unlock matching main workloads after each
turn. The prior stage-boundary caps are historical. Nine completed main fits
are preserved, and every unstarted/incomplete slot stays in the 32-slot report.
The current ledger and service are recorded in the master `program-progress.json`;
do not allocate against a superseded historical ledger.

An optional bounded sibling study uses `member_rule="shortest_verified_l"`,
`posterior_members="selected"` and explicit `posterior_member_count`. It selects
distinct verified L values in increasing order and the first candidate ID
within each, before posterior sampling. `member_slot_assessments` reports
coverage and fixed-comparator results separately for each ordinal slot, with
all planned fits in each denominator. Missing slots stay missing; sibling
outcomes cannot change tuning membership or nominate a posterior winner.

For ongoing accounting use `scripts/audit_inference_validation_campaign.py`
with `--live --cpu-seconds 86400 --gpu-seconds 86400`. Add `--verify-artifacts`
for result and tensor checksum checks. Its live ledger includes running and
queued reservations; an explicitly cancelled suite releases unstarted work.
Use `--overhead-cpu-seconds` and `--overhead-gpu-seconds` to include separately
accounted test/report costs or conservative charges in the allowance check.
Unstarted jobs retain their planned statistical denominator and zero worker cost.
This is accounting, not an assessment of statistical validity.

`assess` recalculates rank and permutation tests from preserved observations,
without launching a sampler. Other engines report `reassessment_not_supported`
and link their original assessments. Reassessment is exploratory and cannot
silently turn development evidence into confirmation.

## Profiles and interpretation

| Suite | Implemented exercise | Limit |
| --- | --- | --- |
| `fast-cpu.json` | All available target densities/scores; dense-metric leapfrog oracle; controller repair, retention, stream and restart checks; known faults | Numerical and controller checks, no convergence claim |
| `pipeline-cpu.json` | Automatic and prepared tuning, frozen transport, every-member posterior assessment, actual stopping, and independent complete-fit SBC units | Small CPU development runs; SBC and stopping estimates have wide uncertainty |
| `statistical-cpu.json` | Frozen-kernel random-position and two-sample tests across difficult target families, reference SBC, ignored-data controls, MCSE/R-hat arithmetic and primitive power | Primitive calibration does not establish full-pipeline power; no global all-model correctness test |
| `numerical-gpu.json` | GPU/XLA mechanics, frozen-kernel defects, ordinary and frozen-transport tuning, full-procedure SBC and repeated real-HMC defect tests | GPU coverage is only for the executed shapes, targets, settings and source version |
| `external-consumer.json` | Matched external observations and references for regression, eight schools and MacroFinance | Unavailable until actual bundles are supplied; no external fit or consumer numerical evidence is fabricated |

Execution validity, sampler findings and test response occupy separate fields.
A known wrong-energy kernel being rejected is successful defect detection and
an unfavorable sampler result. An exception is never counted as detection.
Identity and sign-flip two-cycle kernels preserve the symmetric Gaussian law;
their lack of exploration illustrates why invariance cannot establish mixing.
A position-only wrong score can still give an invariant Metropolis-corrected
kernel; the density/score oracle, rather than invariance alone, must detect it.

For frozen invariance, `options.kernel_power=s` composes s complete MH
transitions of the same frozen kernel. The default 1 preserves the original
transition stream. Powers use a stable TensorFlow signature, `tf.while_loop`,
independent stateless substep seeds and the declared XLA policy. Every substep's
state and log ratio must be finite; a failure remains invalid evidence rather
than a detected distributional discrepancy. The recorded log ratio is the last
substep's ratio, not a ratio for the composition. The experiment is the fixed-look
K^s extension in Gandy–Scott section 2.2; posterior/SBC draws are unaffected and
no sequential-testing guarantee is implied.

Gaussian mechanics additionally checks the actual MH log ratio using independent
analytic density and observed TFP endpoint momenta. Baseline, no-op and reversed
energy controls separate correct arithmetic from an activated mutation. Passing
this numerical oracle does not establish distributional sensitivity.
`options.profile_execution=true` writes `attempt-NNN-host.prof` on engine success
or failure. Inspect it with Python's `pstats`; missing profiles are reported
without masking the engine outcome. Host time includes framework compilation
and execution where they are not separately measured.

The reference SBC engine uses proper generative normal-normal, beta-binomial
and LGSSM-location models. The last model has one unknown location with a normal
prior and a known latent AR(1) covariance. Its reference uses a separate scalar
Kalman recursion. It is not a reference for arbitrary unknown state-space
parameters. Parameter, bounded-radius, dependency and data-dependent likelihood
quantities are evaluated where applicable. Truth is assessor-only: numerical
fits start from fixed data-independent positions.

Full SBC uses independent complete fits conditional on each dataset and takes
one fixed output per fit. It never calls chain draws or candidate siblings
independent replications. `declared_l_first` chooses the first ID within a
predeclared L group; `first_verified` chooses the first ID across the returned
set. Selection precedes posterior sampling and uses no truth or posterior
diagnostic. `posterior_members: selected` leaves sibling records explicitly
unassessed; `all` assesses every verified member. Missing fits stay in the planned denominator. Conditional ranks from
completed datasets cannot establish unconditional calibration when fits fail.

Ordinary designs with `native_search: true` pass no search override to the
public tuner. Its own preparation chooses the starting epsilon and broad search
configuration. Designs supplying `search` and `step_size` exercise a different,
explicitly configured procedure. An epsilon supplied before preparation can
exceed the final-metric bound, and native bounded search can return no verified
member. These are separate findings; neither is repaired by changing acceptance
thresholds after inspecting results.

Actual-stopping rows compare the final controller's means and medians with
available exact functionals. Nominal normal intervals formed from its MCSE are
evaluated at the actual stop, including cap outcomes. Binomial intervals use
complete pipeline replications and one declared candidate group, never sibling
counts. Failure to obtain a posterior output remains unavailable. These measured
intervals have no presupposed sequential coverage guarantee.

Designs declaring `fixed_comparator` additionally use the same verified member
for separately seeded, fixed discarded warmup and retained blocks. Both arms
are archived through the existing member runner. Exact-reference mean, median
and mixture mode-probability errors remain available even when runtime readiness
fails. A comparator exception preserves the stopped arm and records a missing
pair. Pointwise paired intervals and small-replication coverage estimates are
descriptive; they do not prove one stopping rule superior.

Reports distinguish controller calls from usable posterior outputs. A completed
call with no retained draws is unavailable for posterior accuracy; every
declared stopping quantity remains in the denominator even if all intervals
are missing. The generated table shows current-source executions, historical
executions and planned designs separately. Affine and dense nonlinear transport
evidence occupy different cells. Repeated executions of one design are repairs,
not additional independent replications.

## Allocation and assumption audit

| Choice | Provenance and justification | Failure diagnostic / interpretation |
| --- | --- | --- |
| Float64 densities and gradients | Reference regime; independent SciPy formulas and scale-aware centered differences | Relative error; GPU/XLA coverage is separately executed |
| FD step `eps_machine^(1/3) * max(1, abs(q))`; relative tolerance `100 * eps_machine^(2/3)` | Balances central-difference truncation and rounding; factor 100 is an engineering margin | Raw probes, values, scores and errors retained; not an exact-gradient theorem |
| Dense momentum mass `[[2,.3],[.3,.8]]` | Positive definite nonidentity diagnostic fixture, chosen to expose covariance/precision inversion | Compare TFP leapfrog with independently derived block-matrix powers and reversal |
| Density tolerance `1e-10`, leapfrog tolerance `1e-9` | Conservative float64 engineering margins on order-one fixtures | Quantitative errors retained; no transfer to ill-scaled consumers |
| R-hat arithmetic tolerance `1e-12` | Existing independent-rank regression tolerance, reused for the same float64 array experiment | Arithmetic mismatch is distinct from an unfavorable R-hat value |
| Target parameters | Named laws: Gaussian scales, banana bend, funnel scale, t degrees of freedom, mixture separation/weight and proper conjugate priors; complete values are in target/reference source and scenario JSON | Independently checked density/score laws; difficult settings are stress hypotheses, not tuning defaults |
| Initial offsets `[-1,-.3,.4,1]`, remote offset 8, mode offset -5 | Fixed, reproducible data-independent start hypotheses | Preserve actual preparation bank; these do not establish overdispersion or global mode coverage |
| Standard automatic preparation | Existing public preset under test | Smoke preparation failed to produce a dispersed Gaussian bank in development; standard is used for complete-path scope, preserving that failed smoke observation |
| Broad L grid `[3,5,9,13,18,25]`; supplied-pair `[3,5]` profiles | Existing broad policy versus explicit narrower prepared-route coverage | Do not label supplied-pair evidence as broad automatic coverage |
| Default search pilot/refinement and finite budgets | Existing controller with declared engineering allocations; no winner selection | Zero-member, capped, inconclusive and incomplete outcomes are reported |
| Prepared epsilon hypotheses 1.1/1.3/1.5; initial .4/.6/.8; wrong-energy 1.7 | Convenient Gaussian/reference exploration and strong named defect sizes, not transferred defaults | Every pair receives its own real acceptance and health checks; a failed hypothesis is not repaired by relaxing acceptance |
| Eight discarded evidence warmup transitions; at least 64 measured draws | Small engineering allocations; native acceptance policy minimum retained | Weak evidence can be inconclusive; no convergence inference |
| Prepared epsilon domain [.005,3], repair factor 2, maximum five repairs per family | Explicit finite engineering search hypotheses; automatic/frozen routes preserve their native domain derivation | Boundary or repair exhaustion stays a failure to find a member, not proof no member exists |
| Posterior chunks/caps and MCSE tolerance .1, bounded-function error tolerance .25 | Small development allocations in each JSON, chosen to exercise stopping and persistence | Report actual counts, error and uncertainty; tolerance comparison remains descriptive |
| Per-design alpha .05 with within-design Bonferroni multiplicity | Conventional diagnostic error allocation; each design is its own stated scientific question | No family-wide all-model correctness claim; negative-control designs are reported separately |
| Monte Carlo null counts 399 or 1999 | Plus-one p-values, with preflight resolution check against every tested quantity | Discrete resolution is preserved in each p-value |
| 128/256 invariance or reference-SBC replications; 256 primitive-power trials | Coarse development allocation; binomial SE is at most .03125 for 256 trials | Exact binomial bounds reported; continuous differences remain descriptive |
| Two datasets × two full fits for SBC; eight real-HMC power trials | Bounded complete-path demonstrations based on local measured preparation/compile cost | Too small for strong calibration or power claims; failed fits are explicit |
| AR(1) rho .8; transient offset `8 exp(-t/(n/3))`; constant and missed-mode controls | Known stationary finite-sample covariance and deliberately severe diagnostic stress | Compare runtime arithmetic with independent ranks and exact stationary variance; transient/missed-mode outputs do not inherit stationary CLT validity |
| Seed 20260916 with separately hashed phase/dataset/fit/member streams | Reproducibility choice, not a scientific default | Restart retains identity; no seed search or rerun-until-pass |

The historical CPU profile maximum worker reservations sum to 1,790 seconds. Its initial GPU
profile reserves 2,900 seconds, below the 3,600-second allowance. These are hard
engineering ceilings, not precision-based scientific allocations. Local failures
and retries consume the same total allowance. Primitive tests measure observed
cost before any larger confirmation allocation is proposed.

## Extending coverage

Register a target with its coordinate dimension, model quantity names, moment
conditions, generative capability and independent reference. Add target and
reference implementations separately, and test density/score/transform agreement.
Create scenarios and JSON designs using an existing engine. Add a named mutation
and a no-op control when a new mechanism needs a new oracle. Declare required
coverage cells explicitly; a controller double cannot satisfy numerical coverage.

The external reference reader accepts portable saved JSON observations with
target/data/prior/coordinate identity, quantity order, source/version/method,
uncertainty, dependency declaration, and sample checksum. This is an ingestion
boundary, not a claim that arbitrary external output is correct. A real consumer
still needs matched density/coordinate evidence and a reviewed reference.
Only a bundle explicitly declaring `sampling_structure: "iid"` receives an iid
reference standard error; MCMC or unknown dependence leaves that combined error
unavailable. Mean reporting additionally requires `finite_variance: true`.

The M8 campaign also has two explicit TensorFlow adapters in
`posteriordb_targets.py`: noncentered eight-schools and `sblrc-blr` regression.
They match the pinned posteriordb Stan laws, observations and coordinate
Jacobians. Their separate campaign runner calls the public ordinary tuner and
posterior controller, preserving the full candidate set and preselecting one
member before sampling. The ten-chain Stan reference is used only afterward;
the comparison includes both samples' lugsail uncertainty. These adapters do
not activate the generic `external-consumer.json` cells or supply a matched
MacroFinance target. Commands, exact upstream hashes, limits and results are in
the [M8 execution note](../plans/bayesfilter-hmc-repair-m8-result-2026-09-18.md).

M15 completed 246 fresh whole fits over 82 datasets with all outputs available,
using three independent fits per dataset. This extends automatic full-procedure
SBC beyond tiny mechanics tests, but its nonrejection is weak evidence against
subtle errors: M16's analytic normal controls at 32 datasets/three rank draws
detect quarter-SD and half-SD location shifts in 27/256 and 64/256 experiments.
That is statistic-level sensitivity, not whole-HMC defect power. M16 also
completed 576 numerical boundary searches and 128 complete three-arm sequential
validation experiments per device. The reversed-ratio defect is detected in
all trials; three null-control intervals remain too wide for the predeclared
size-precision screen. No default is promoted by these outcomes.

M17 completed 21 geometry/route cells: rotated Gaussian, centered/noncentered
funnel, Cauchy, mixtures with two start regimes, affine Gaussian/banana,
fixed dense-IAF banana/Dirichlet, and CPU Student-t. Eleven selected members
pass the full posterior checks; empty candidate sets and warmup/precision caps
remain reported. All six repaired conditional field cells reach independent
measurement/verification: Gaussian and beta-binomial on CPU/GPU, plus CPU LGSSM
and banana. They remain conditional mechanics and cannot issue exact-score
retained members. Three of four pinned regression/eight-schools fits pass the
full reference/posterior screen; CPU eight-schools passes all mean comparisons
but reaches the precision cap. These development cases do not resolve the
missing exact original MacroFinance reference, learned-transport training,
unknown-mode discovery, nominal stopping coverage or subtle whole-fit defect
power. Fixed nonlinear maps are not training evidence.

Dirichlet posterior reporting supports three
named probabilities from its two active coordinates. The CPU campaign pilot
assessed all 23 verified simplex members; a capped all-member GPU pilot remains
incomplete and cannot receive the CPU result's credit. A separate GPU subset
experiment assessed two selected members across two replications and retained
all 44 verified members; the other 42 remain unassessed for posterior accuracy.
Frozen invariance designs can now enable the separate Gandy--Scott Algorithm 3
wrapper with `options.sequential`. Each look uses a fresh complete experiment,
and sample count increases once after the first look. Its type-I bound is
conditional on independent look vectors and superuniform component p-values;
it does not cover tuning acceptance or posterior stopping. Fixed-look designs
remain available. None of the missing numerical cells is satisfied by counting
additional pytest cases or by an architectural review verdict.
