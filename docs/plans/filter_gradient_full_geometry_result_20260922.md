# Complete geometry execution: qualification and cost findings

The public `fit_low_rank_spd_quadratic_geometry` now calls one enclosing
TensorFlow/XLA numerical program. It owns center evaluation, active pilot,
design evaluation, finite-row partition, fitting, refinement, exact incumbent
selection and replay. Configuration/input validation, prepared random inputs
and formatting completed records remain outside that call. This result concerns
one geometry attempt; the iterative initializer and other E1--E6 obligations
remain open.

The numerical authority is original3582b4ac on identical realized inputs.
The approved versioned TensorFlow RNG remains the runtime stream. Legacy
seed-specific tests supply their original clouds and every finite-count
permutation at the test preparation boundary. No numerical tolerance is changed.
The previously reviewed ill-conditioning guards reject unusable pilot bases and
fits; the public pilot rejection stops before design evaluation.

The native signature contains center, scale, raw directions, design offsets and
permutation seed as tensors. Training/holdout branch tables include only counts
reachable under the original finite-sample rule. Scalar calls preserve the
original order; batched callbacks receive their exact active extents. Incumbent
indices exclude inactive storage and preserve strict earliest finite ties.
Center failure retains the legacy public recorded count0 while native attempted
accounting records its one callback.

## Qualification

| Runs | Evidence |
| --- | --- |
|02694--02696|Composition exposed missing raw-XlaSvd output shapes; attribution and explicit shape contracts repair both D3 smokes. No solver arithmetic changed.|
|02697--02703|81 CPU checks:72 full original records at D1/D3/D5 and9 shape, count, guard, seed and changing-input/HLO checks.|
|02704--02705|Six public endpoint checks and three original-cloud/incumbent checks, including120 samples/64 pilot directions.|
|02706--02712|87 GPU checks:72 full original records plus15 public/edge checks. Memory growth verified on GPU2 UUID `GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`.|
|02713|All15 CPU public/edge checks pass after restoring current public scalar-route labels; GPU metadata renewal remains pending.|
|02733--02736|All20 remaining public/parity CPU checks and102 policy checks pass in bounded fresh groups.|
|02737--02746|All33 initializer consumer cases pass in nine bounded CPU groups;102 policy checks renewed. Exact-input test-only cache preserves repeated legacy fits.|
|02727|Both larger public CPU cases pass: D4/sample260/pilot512 and D3/sample180/pilot96, including scalar/batch agreement and deterministic replay.|

The public design and pilot labels remain `tensorflow_scalar_row_loop` for
scalar targets. Original-reference comparison views normalize only the known
historical `scalar_value_and_score_loop` label; saved raw records retain their
actual metadata. The internal diagnostic pilot formatter retains its historical
label for existing comparison consumers. These are metadata changes only.

## Comparable cost evidence

Unmodified91928762e cannot serve as a numerically qualified comparator on the
chosen clouds.02714 exposes raw-XLA trust-eigen error in the improvement ratio
(4.063e-10).02719 additionally exposes raw-XLA pilot-eigen error in fit loss
(1.766e-9).02715/02720 isolate those errors by changing only the respective
eigensystem calls. The corrected prior and both new graph/XLA records agree with
the original at unchanged1e-10 comparisons. Preserve the failures.

`prior_refined` therefore means91928762e with its trust and pilot eigensystem
calls replaced by the shared qualified refined eigensystem. It remains a mixed
host/compiled implementation. It is not an unmodified before measurement. The
graph reference and XLA candidate use their declared solver implementations;
their comparison is not an identical-graph compiler ablation.

02721--02726 provide six fresh CPU processes, one per arm/extent. Exact source
and input hashes, full original records,20 synchronized warm calls, stage
RSS/PSS, graph/HLO sizes and cold phases are recorded. The fixtures end in
`holdout_fit_rejected` after completing fit/refinement/replay; they do not alone
qualify accepted-result formatting costs or whole iterative costs.

| Samples / pilot directions | Comparator | Cold total seconds | Warm median ms | Host peak RSS MiB |
| --- | --- | ---: | ---: | ---: |
|24 /6|Repaired prior|2.28|29.22|888.7|
|24 /6|Native XLA|6.23|5.58|1202.2|
|120 /64|Repaired prior|2.23|35.10|892.7|
|120 /64|Native graph reference|10.38|21.65|1217.5|
|120 /64|Native XLA|27.89|6.22|2314.6|

Use `artifacts/filter-gradient-repair-20260917/geometry-full-costs-cpu-02726-v2.json`
and its saved analyzer. The first analysis used `resource.ru_maxrss` and is
superseded: this counter already exceeds `/proc/self/status` VmHWM at preparation
in several workers. The v2 result retains both counters separately and uses
VmHWM for the process's resident peak. This discrepancy was also noted in the
earlier gap diagnostic; it must not be treated as a numerical-stage allocation.
These are maximum sampled resident readings, not certified exact peaks: the
campaign's [memory observation limits](filter_gradient_repair_memory_observability_20260918.md)
also record non-monotone VmHWM behavior on this host in a TensorFlow-free probe.
Small changes cannot be interpreted as allocator growth without corroboration.

Cold time and extra-host-memory triggers fire at both extents. At120 samples,
most XLA RSS growth occurs during the first call, after tracing. Graph sizes are
5457/17937 nodes at24/120 samples. Twenty warm calls add only0.12MiB RSS at120.
Count-dispatch compilation is a plausible cause of the increase; neither its
allocation ownership nor native executable eviction is isolated by these data.
All timings and memory differences are descriptive single-process observations.

02728/02729 pass two GPU numerical cost records, but both have shared-device
preflights. The matrix then stops on GPU contention before the XLA worker.
These timings are ineligible for matched GPU comparison. Clean GPU costs and
three-process terminal repeats remain required.

## Review and continuation

| Decision | Primary criterion | Veto / uncertainty | Next action | Not established |
| --- | --- | --- | --- | --- |
|Retain the enclosing geometry implementation|Original full records and public wiring pass CPU/GPU fixtures|Remaining public/consumer renewal and metadata GPU renewal pending|Finish registered suites and policy guard|Complete iterative or repository-wide execution closure|
|Investigate compile memory|Both extents exceed cold and host-memory triggers|Native allocation lifetime and count-branch contribution unisolated|E4 ownership and E6 count/signature attribution|Memory leak or leak freedom|
|Keep GPU cost comparison open|Two records numerically pass|Shared device and incomplete matrix|Fresh clean matched UUID processes|GPU performance ranking|
|Continue E2 iterative work|Numerical geometry dependency exists|Locator setup/selection and terminal mass decisions remain on host|Compose original locator/recentering/mass with complete records|HMC tuning authority, scientific admission or merge readiness|

Hard numerical vetoes are enforced at the unchanged comparisons. No statistical
performance ranking is supported. The observed warm reductions and cold-memory
increases are descriptive. Default-readiness remains unestablished; final source
qualification, accepted/public costs, repeats, consumer evidence and integration
are still needed.

Post-run review: the largest evidence weakness is whole-consumer coverage. A
compiled geometry step cannot certify an initializer whose locator, iteration
or mass decisions still run on the host. The strongest alternative explanation
for retained memory is bounded compiler/allocator caching; a signature-churn
test after coordinated Python release can distinguish it from unbounded growth.
An eigensystem-corrected comparator must remain clearly identified in every
cost summary. Main remains unmerged.

Checkpoint review: all declared bounded CPU public/initializer cases now pass.
Combined GPU consumers and scalar-label renewal remain pending due to contention.
Focused runtime/new-test Ruff and whitespace pass. The older geometry parity
fixture retains its two pre-existing C408 style warnings, and the runner retains
its pre-existing style warnings; no whole-tree Ruff-clean claim is made.
The native dependency plan is reviewed for original optimizer identity, reset
accounting and strict incumbent order. No terminal E2/merge claim is made.
