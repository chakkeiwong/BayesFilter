# Funded inference validation campaign

Date: 2026-09-16. The owner authorized 24 CPU hours and 24 GPU hours and asked
for execution. Status: completed within both allowances; see the
[terminal result](bayesfilter-inference-validation-24h-result-2026-09-16.md).
This is the funded numerical continuation of
[Phase 2](bayesfilter-inference-validation-phase2-plan-2026-09-16.md).
Earlier budgets and failed attempts remain preserved separately. Interpret each
new allowance as 86,400 summed worker wall-seconds on its declared device lane,
consistent with the previous campaign. Record CPU thread count (two intra-op,
one inter-op) and GPU identity; elapsed time is not a throughput comparison.

## Question and scope

Can the current public candidate-set tuning procedure produce complete inference
outputs across independent simulated datasets, and how do its actual stopping
decisions compare with predeclared fixed-length sampling against independent
posterior quantities? Separate numerical implementation, kernel invariance,
posterior calibration, availability, and test sensitivity.

The budget is sufficient for a substantial bounded campaign on the implemented
targets. Adequate sensitivity for a particular defect remains a measured
question. Missing external reference inputs cannot be replaced by a compute
allocation. No universal burn-in, all-model correctness, training-quality,
algorithmic superiority or default-readiness claim is planned.

The baseline is the current public tuner, including the already present
directional interior-epsilon repair. Previous runs used earlier source and are
cost/failure diagnostics only. Freeze a local copy of the package for each
campaign evidence version so other authorized workspace work cannot change
executing kernels. Preserve source hashes, Git commit and suite inputs. No
public tuning admission policy is changed for this validation campaign.

## Evidence and decision rules

| Role | Rule |
| --- | --- |
| Numerical pass | Independent density/score, transform, replay and inventory checks; health and finite output |
| Calibration finding | Predeclared rank tests of parameters and data-dependent likelihood; independent datasets, independent full fits within datasets; missing required fits invalidate an unconditional calibration claim |
| Stopping comparison | Both arms on the same target and verified member; independently seeded conditional on tuning; predeclared fixed counts; global reference error, availability, actual-stop coverage and cost with replication uncertainty |
| Test sensitivity | Null rejection and named activated defects, including subtle severity levels; binomial uncertainty; invalid execution is not detection |
| Promotion veto | Broken oracle, incorrect numerical identity, missing outputs, reference disagreement, insufficient statistical power or uncertain reference |
| Continuation veto | Invalid device/memory execution, corrupt artifacts, changed target/source during a job, or exhausted relevant allowance |
| Repair trigger | Incomplete tuning/posterior fit, shape incompatibility, insufficient job allocation, failed invariant or unmeasured compilation cost |
| Explanatory | Acceptance, local R-hat/ESS/MCSE, warnings and descriptive timing unless a posterior criterion explicitly declares their role |

All verified candidates remain in native records. Large replicated jobs may
assess a predeclared subset. A new `first_verified` reporting rule chooses the
lexicographically first verified candidate ID before posterior execution; it
uses no truth, accuracy or posterior diagnostic. This studies a predeclared
member rule applied to the complete returned set and avoids requiring an L
family the tuner is not obliged to produce. The historical fixed-L experiment remains failed/incomplete. Small
integration jobs continue to assess every member. Unassessed siblings remain
explicit and cannot count as validated.

## Work and allocation

Allocations are ceilings, not requirements to spend unused time. CPU and GPU
allowances cannot be exchanged. Every launch and local retry is charged.

| Phase | CPU seconds | GPU seconds | Deliverable |
| --- | ---: | ---: | --- |
| Prerequisites and cost pilots | 3,600 | 3,600 | Simplex report dimension, subset selection, timing, fixed comparator; independently checked CPU/GPU complete fits |
| Statistical primitive and severity calibration | 7,200 | 6,000 | Matched rank-test power at planned sizes; real-kernel defects and null controls |
| Fresh full-procedure SBC | 28,800 | 32,000 | Conjugate normal, beta-binomial and LGSSM regimes; separate device/route findings |
| Actual stopping and target/route expansion | 14,400 | 16,000 | Gaussian, mixture and constrained cases; repeated stopped/fixed arms; broader target matrix |
| Available external/reference work | 7,200 | 12,000 | Independent reference inventory and feasible exact-reference targets, no fabricated consumer bundle |
| Local repair and unallocated reserve | 25,200 | 16,800 | Focused repair/retest within the same evidence contract |
| Total | 86,400 | 86,400 | Hard campaign maximum per device |

Initial pilots use four independent full fits on each device, two datasets and
two fits each; these are cost/availability checks only. Candidate search retains
the native broad `(3,5,9,13,18,25)` grid, current acceptance policy, existing
standard preparation and bounded refinement. Posterior diagnostics use declared
counts and native thresholds, never loosened to produce SBC outputs.

The confirmation sizing candidates are 64 or 128 independent datasets with
seven independent complete fits each (eight possible ranks). These are planning
hypotheses: 64 and 128 bound a Bernoulli-rate standard error by 0.0625 and
0.0442. This bound does not prove rank-test power. Measure test sensitivity at
these exact dimensions and choose a funded size using pilot cost and independent
power controls before confirmation. If neither is defensible, record the
limitation and run development only. Do not rerun confirmation until it passes.

## Implementation prerequisites

1. Let the shared posterior controller report a fixed number of named model
   quantities different from the active dimension. Preserve draw/chain axes,
   active state/checkpoints, finite checks, empty shapes and continuation.
2. Add explicit validation-only member selection and posterior count settings.
   Record selection before posterior sampling, and keep all verified siblings.
3. Add a fixed-length comparator through the existing verified-member runner.
   Discard its declared initial block; use separate streams and native archives.
   Do not infer warmup sufficiency or promote fixed-count policy to runtime.
4. Record preparation/search, member construction, controller, first execution
   and repeated execution costs where measurable. Compilation and first execution
   remain combined unless the native metadata actually separates them.
5. Extend stopping assessment with global reference functionals and missing-pair
   accounting. Measure uncertainty across whole replications, not siblings.
6. Strengthen campaign resume, progress and source freezing for longer jobs.
   Use existing native checkpoints and ordinary JSON manifests; no launch tokens.

## Assumption audit and pre-mortem

The prior two-fit/two-dataset run is not a statistical baseline. Its failure to
return L=3 is not failure to retain valid L=5 candidates. The new reporting rule
changes the experiment explicitly and requires fresh data/randomness. Selecting
the first member can still select a poorly exploring kernel; SBC/global
references must expose this rather than condition on posterior success.

Default thresholds and preparation are subjects under test. Small pilot counts
are convenience allocations, not proof of sufficient burn-in. For replicated
stopping use inherited 500-transition chunks, 2,000 minimum warmup, a 1,000
window and 10,000 cap unless a target-specific pilot documents a different
development hypothesis. Fixed comparator warmup 2,000 and retained 4,000 are
predeclared cost/precision baselines, not correctness assertions. Mean quantities
require finite moments; mixture mode probabilities provide a global check.

An apparently successful run can mislead through success-conditioned ranks,
correlated fit outputs, missed modes, severe-only mutations, changing source,
or declaring compilation time to be sampling. The earliest checks are native
complete-fit inventories, fresh stream identities, independent exact references,
severity/null controls, source snapshot hashes and explicit cost categories.

The GPU readiness helper currently reports no device meeting its idle policy.
Check trusted device inventory before launch. Small memory-growth jobs can share
an underutilized device only with recorded resources and no speed comparisons;
do not terminate another user's process. GPU kernels must use TF/TFP XLA and
verified memory growth before initialization. CPU jobs deliberately hide GPUs.

The skeptical audit passes for prerequisite implementation and bounded pilots.
It does not preapprove a confirmation size before the cost/power evidence. The
new allowance authorizes routine local repairs and later funded phases under
this same question; it does not authorize a threshold change or default promotion.

## Execution and records

Use the existing `tfgpu` interpreter, no environment mutation. Output root:
`docs/plans/artifacts/inference-validation-24h-2026-09-16/`. Each source/design
version and attempt has a fresh directory; retain failed attempts. Worker
manifests record plan path, command, seeds, source, device/memory/XLA and wall
time. Keep a campaign ledger of allocated and consumed worker-seconds. Stop
launching when the remaining allowance cannot cover the next declared job.

After pilots, append measured allocations, resolved suite paths and the
power-based decision here before larger runs. Final results must separate
engineering pass, numerical findings, statistical support, unresolved coverage
and the next justified action. R-hat/ESS/MCSE remain posterior diagnostics.

## Pilot checkpoint and parallel execution audit

The CPU pilot completed four independent ordinary fits in 406.36 worker-seconds,
all 23 verified simplex members in 540.90 seconds, and two Gaussian stopped/fixed
replications in 317.43 seconds. The first two GPU fits required 250.80 and 241.33
seconds including preparation and tuning. These are planning measurements, not
CPU/GPU performance comparisons; GPU work shares a device and uses XLA, whereas
CPU work is an explicit non-XLA reference exception.

The independent primitive calibration used 256 experiments at each size and
severity, seven fit outputs, six normal observations and three simultaneous rank
quantities. At 64 datasets, the half-posterior-SD location defect was detected in
141/256 experiments (95% binomial interval 0.488--0.613); the one-SD defect was
detected in 256/256 (0.986--1). At 128 datasets, half-SD detection was 232/256
(0.864--0.939). Quarter-SD detection remained weak. Null rejection intervals
included 0.05. Consequently, a 64-dataset full fit test can address the named
large defect; non-rejection cannot exclude smaller bias. Analytic defect power
does not establish power against implementation mutations.

Before larger jobs, add bounded parallel job execution to the existing runner,
keeping one coordinator and existing per-worker timeouts. Dataset groups are
declared together in the suite before execution. Their independently seeded
shards use identical numerical settings and frozen source; aggregation checks
identities, checksums, expected membership and every missing dataset. It never
pools pilot, changed-source or retry outcomes. Parallelism reduces calendar time;
the budget remains summed worker wall-seconds, including failures. A maximum of
16 CPU and 4 GPU workers is a resource convenience hypothesis, subject to a
small integration check and measured resource use, not a throughput claim.

The pre-execution audit identified two reporting hazards to repair before the
new source freeze: an entirely missing stopped/fixed cohort must retain every
declared quantity in its denominator, and a fixed-arm exception must preserve
the stopped arm and record an unavailable comparator. No threshold or tuning
policy changes are involved. Focused corrupt/missing-group, concurrent worker,
and comparator-failure tests must pass before the larger launches. The audit
passes with these prerequisites; frozen pilot jobs continue unchanged.

## Resolved main allocation

All eleven new prerequisite checks passed, including two real concurrent SBC
workers, incomplete-group denominators, source/checksum rejection, simplex
reporting and an injected fixed-comparator failure. The main source identity is
`a1bdf4c334d519dbc8970ff082c10f92afa5cec8383177c24b7ff0d1743146be`, preserved
in `artifacts/inference-validation-24h-2026-09-16/source-main-r1/`.

The resolved suites supersede the preliminary phase allocation while retaining
the separate 86,400-second totals:

| Work | CPU maximum seconds | GPU maximum seconds |
| --- | ---: | ---: |
| Main ordinary normal SBC | 64,000 | 56,000 |
| Gaussian and single-mode mixture stopped/fixed arms | 7,600 | 10,200 |
| Beta-binomial and LGSSM full-fit development SBC | 3,600 | 4,000 |
| Actual HMC null/no-op/energy-defect calibration | 2,700 | 3,000 |
| Thirteen independent density/score mechanisms | 780 | 1,950 |
| Frozen affine and additional target cases | 2,700 | 5,100 |
| Main suite total | 81,380 | 80,250 |

CPU primary confirmation comprises 64 datasets in 16 predeclared shards, seven
fresh full fits per dataset (448 fits). Its only confirmatory family comprises
the parameter, bounded radius and data-dependent log likelihood, with Bonferroni
family size three at 0.05. GPU has 32 datasets and seven fits each, explicitly
developmental: its smaller size has no calibrated subtle-defect power. The two
device groups are never pooled. All other model/test comparisons are development
evidence without a campaign-wide rejection or ranking claim.

Both devices run eight Gaussian and eight deliberately single-mode mixture
replications, each with the predeclared fixed comparator. Eight replications can
expose a gross stopping failure but cannot establish nominal 95% coverage.
The kernel calibration repeats 32 complete invariance experiments at each of
epsilon 0.3, 0.6 and 1.0, with baseline, no-op and reversed-MH-ratio controls;
128 exact-reference anchors and seven transitions per experiment. The epsilon
ladder is an explicit severity hypothesis, not a sampler default. Binomial
intervals describe sensitivity and null rejection separately at each setting.

Main CPU reservation plus completed pilots/power is about 82,693 seconds before
focused regression time, leaving over 3,000 seconds for measured test costs and
localized repairs. Main GPU reservation plus all pilot ceilings is 83,550
seconds, leaving 2,850 seconds even if every pilot exhausts its cap. Timeout
cleanup and every failed attempt are charged; actual unused reservations become
available after completion. No later launch may exceed the remaining lane budget.

Commands run from the frozen source directory with the existing `tfgpu` Python:

```bash
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  -m bayesfilter.testing.inference_validation run suites/main-cpu.json \
  --max-workers 16 --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/main-cpu-r1

env CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  -m bayesfilter.testing.inference_validation run suites/main-gpu.json \
  --max-workers 4 --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/inference-validation-24h-2026-09-16/main-gpu-r1
```

The final skeptical audit accepts the primary CPU design for detecting the
calibrated large location defect, with no equivalence or universal-validity
claim. GPU and small secondary models remain development work. Budget shortfall,
lost outputs, fixed-arm failures and missing modes remain findings rather than
reasons to discard replications. External reference bundles are still absent;
they receive no invented execution or correctness credit.

## Main CPU finding and bounded repair

The CPU primary group finished all 448 attempted fits, but only 15/64 datasets
had seven outputs. Seventy-one fits raised `epsilon proposal exceeds the declared
final-metric safety bound`; fifteen returned no verified member. This invalidates
unconditional calibration. The explicit 0.6 pre-preparation proposal is a harness
choice, not the native automatic initializer, so the result cannot be presented
as the failure rate of untouched default initialization. A checked zero-member
fit additionally exhausted a bound of 1.14744 with acceptance above the repair
region for every L; that is a genuine bounded-search limitation in this scope,
not a reason to relax acceptance or remove the bound without further evidence.

The same large GPU normal experiment uses the confounded initialization. Stop
that main-suite attempt, preserve partial fits and charge actual worker time.
This is a wrong-baseline continuation veto, not rejection of HMC or the research
direction. Resume the other predeclared GPU questions in a fresh output root;
the unused normal reservation is released. Do not combine partial GPU ranks
with later corrected fits.

Repair the validation adapter with an explicit `native_search` option for
ordinary preparation. It passes no search override and records the resulting
native config, thereby using the public broad pilot/refinement and its own
initial epsilon. The original supplied-proposal case remains available and its
failure remains evidence. Test dispatch and run eight fresh native-search
datasets with two independent fits each, per device, as a bounded development
check before considering another large calibration. No acceptance, R-hat,
preparation safety-bound or default numerical policy is changed. Reserve at most
4,000 CPU seconds and 6,000 GPU seconds for this repair check; actual main CPU
cost was 39,253.37 seconds, leaving ample room within the original lane totals.

The skeptical audit accepts this repair because it removes the identified
baseline mismatch, retains native authority, uses fresh seeds and labels the
small result developmental. Any remaining zero-member or missing posterior
outcome is still a finding; it will not be repaired by seed search, reused
successful datasets or a relaxed threshold.

## Complementary prepared-route check

Native-search failures are now also observed without a supplied epsilon. In
checked failures, the pilot starts at the preparation ceiling, every L requests
a higher epsilon, and no higher proposal is permitted. This is an availability
finding for the ordinary route. It does not invalidate the independent target
reference or justify changing the ceiling during confirmation.

Use the remaining CPU allowance to discriminate this preparation/search issue
from failure of the shared candidate-set/posterior machinery: first run two
fresh normal-conjugate datasets with two independent fits through the existing
prepared route, explicit identity geometry, broad L search and unchanged
acceptance/posterior policies. This route uses no generating truth or analytical
posterior geometry. Reserve 600 seconds. If all four fits return outputs and
cost supports it, run a separately predeclared 64-dataset, seven-fit experiment,
with 16 shards capped at 2,400 seconds each (38,400 seconds total). Those ceilings
plus the existing CPU ledger and current native-search reservation fit the
86,400-second allowance. Otherwise preserve the cost pilot and defer that larger
experiment. Its ranks never substitute for automatic-preparation evidence.

The comparator is the same declared posterior law, tested via an explicitly
different public entry route. The primary test remains the three-quantity rank
family at 0.05 with the previously measured large-defect sensitivity. Failure
or missing output invalidates unconditional calibration; no ranking against
ordinary preparation is planned. This is complementary route coverage, not a
runtime repair or promotion of identity geometry. The audit accepts this small
discriminating check and the conditional budgeted continuation.

The prepared pilot's first fit took 139.74 seconds, so 448 fits are not supported
by the remaining CPU allowance. Do not launch the authored 64-dataset prepared
suite. Calibrate the same three-quantity, seven-output rank test at 32 datasets
using 256 fresh analytic experiments (at most 300 CPU seconds). A 32-dataset
prepared confirmation may proceed only if the pilot returns all four outputs and
the 95% lower binomial bound for detecting the predeclared one-posterior-SD bias
exceeds 0.8. The 0.8 threshold is a conventional power-planning criterion, chosen
before this new calibration and not a correctness threshold for HMC. Use fresh
confirmation seeds and 16 two-dataset shards at 2,400 seconds each, maintaining
the 38,400-second ceiling. This revision reduces cost while retaining a measured,
explicitly limited sensitivity claim. Smaller biases remain unresolved.

The 32-dataset analytic calibration detected the one-SD defect in 232/256 trials
(95% interval 0.864--0.939), with null rejection 6/256 (0.00865--0.05031).
It passes the declared power-planning screen. The native CPU development group
completed all 16 attempted fits but returned seven zero-member fits, leaving
4/8 complete datasets. No epsilon-bound exceptions occurred in that corrected
group. The provisional artifact audit checked 59 result hashes and 1,100 tensor
archives without a mismatch. Regression checks passed 187 tests; the separate
coverage-source rendering regression also passed. A conservative 1,000-second
CPU allowance is reserved for all focused tests, reporting and guide-build
overhead in addition to indexed numerical workers; terminal accounting will
report this reserve separately from measured numerical time.

The prepared pilot completed all four fits and both datasets in 584.80 indexed
worker-seconds. The 32-dataset suite is therefore launched as declared, with
224 independent full fits and fresh seed 2026091614. The updated CPU accounting
before launch is approximately 42,981.70 numerical worker-seconds consumed,
38,400 reserved for confirmation and 1,000 for test/report overhead, leaving
about 4,018 seconds uncommitted. This is the final large CPU launch; missing
outputs or a cap remain findings and cannot authorize rerunning until pass.

## Terminal accounting audit

Before reporting the campaign, audit every planned job, including work never
started after the confounded GPU suite was cancelled. The existing terminal
reader iterates only indexed jobs and could omit those missing datasets from
its availability inventory. Repair that reporting omission, retain their zero
consumed cost, and test cancellation with unstarted SBC datasets. Keep the
conservative 1,000 CPU seconds for tests, reports and guide builds separate from
measured numerical worker time, but include it in the campaign limit check.
Verify source snapshots, worker device/memory manifests, candidate retention and
result/tensor checksums before final interpretation. This deterministic audit
does not change executing snapshots or statistical thresholds. The skeptical
audit accepts the repair: it closes an accounting/denominator gap and cannot
turn missing output into calibration evidence.

## Frozen nonlinear transport coverage

The remaining GPU allowance can cover one additional declared route question:
does the current frozen dense-IAF codec, transformed target and retained bridge
still execute correctly with the full posterior controller? Run two fresh
replications of the synthetic Gaussian/dense-IAF fixture already specified in
`docs/validation/terminal-gpu-repair.json`, with its supplied epsilon grid
1.1/1.3/1.5 at L=3/5. These are inherited, previously exercised hypotheses for
this same fixture, not a broad automatic search or learned-training claim.
Assess every verified member, use the campaign's 2,000 minimum warmup and
10,000 warmup/retained caps, retain the original acceptance and precision
requirements, and compare named model-coordinate quantities with the independent
Gaussian reference. Use fresh seed 2026091616 and a new source/output snapshot.

Reserve 2,400 summed GPU worker-seconds for the two-replication job. The current
ledger leaves over 32,000 GPU seconds uncommitted even after all queued repair
jobs are reserved. Start this final route check when fewer than four GPU workers
remain, preserving the existing total four-worker GPU limit. There is no rerun-until-pass:
missing members, posterior caps, reference errors or job caps remain findings.
Two replications establish mechanics and observed behavior only. No sensitivity,
calibration, training-quality or default-readiness claim follows. The skeptical
audit accepts this route expansion because its target and exact reference are
already available, the fixed payload needs no training, and it tests a separate
public preparation route within the original campaign scope and budget.

Command (from `source-dense-r1/`, with the existing memory-growth/thread settings):
`python -m bayesfilter.testing.inference_validation run suites/dense-gpu.json
--max-workers 1 --output <campaign-root>/dense-gpu-r1`.

## Confirmation and stopping checkpoint

The prepared32 experiment completed 224/224 independent fits and 32/32 datasets
in 34,494.12 summed CPU worker-seconds. All selected outputs passed their
declared posterior checks. The three predeclared rank tests detected no
discrepancy (parameter p=0.6385, bounded radius p=0.12, log likelihood p=0.5075;
threshold 0.05/3). This completes that separately scoped confirmation, with the
previously measured large-defect sensitivity and no ordinary-route or general
accuracy claim. No additional CPU numerical jobs will be launched.

GPU stopping produced one locally favorable but globally wrong mixture output
among eight replications: estimated x mean 4.956 versus exact 2, left-mode
probability zero versus exact 0.3000001147, modern R-hat 1.00690 and mean MCSE
0.03796. This is a posterior-promotion veto for that result, not a continuation
veto for the remaining independent numerical or transport tests. The global
quantity was measured by the validation assessor but not declared in the
controller's parameter-only precision policy. Preserve the counterexample and
add target-specific global quantities and start sensitivity to the next repair.
Do not alter the running experiment's checks or turn the quantity into a tuning
admission rule. The final result records all eight outcomes and their uncertainty.

The final dense-IAF check can share the device with the two remaining repair
workers. A trusted `nvidia-smi` check observed GPU 1 at 713/32,760 MiB and 1%
utilization before launch. This routine scheduling adjustment avoids leaving
available worker slots idle; it changes no target, settings, thresholds or
job budget. The live ledger records 77,475.83 CPU numerical seconds plus the
1,000-second overhead charge and 38,537.18 GPU numerical seconds, with 3,600 GPU
seconds reserved for the two remaining jobs. The additional 2,400 GPU seconds
remain well within the allowance. The frozen dense source is identical to the
repair and prepared32 source.

## Terminal status

All launched jobs have ended and no campaign worker remains. The final GPU
simplex subset returned both selected outputs while preserving all 44 verified
members; the dense-IAF check assessed all six verified members across two
replications. Preparation/search failures, posterior caps and the mixture
counterexample remain findings. No successful-only rerun replaced them.

Measured numerical use was 77,475.8265 CPU worker-seconds and 40,224.7381 GPU
worker-seconds. With the conservative 1,000 CPU-second test/report/build charge,
totals are 21.79884 CPU hours and 11.17354 GPU hours. Unused allowances are
7,924.1735 CPU seconds and 46,175.2619 GPU seconds; zero running or queued
reservations remain. The terminal audit checked 101 completed result hashes,
1,658 tensor checksums, seven frozen sources, 110 runtime manifests and 794
tuning records without finding inconsistent evidence. Current-checkout coverage
is deliberately conservative because unrelated Q20/training edits changed the
whole-package hash after freezing the experiments.

The post-run skeptical review rejects a general calibration or burn-in claim:
ordinary SBC is incomplete, local posterior checks missed a mixture mode, and
smaller actual-kernel defects had weak detection power. It accepts the complete
prepared-route rank experiment only at its stated scope and sensitivity. The
guide and plans now preserve that distinction. The next repair order is recorded
in the result; no additional numerical campaign or tuning-default change is
launched by this closeout.
