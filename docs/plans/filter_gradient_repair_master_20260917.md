# Complete filter and gradient execution repair

Status: executing on `repair/filter-gradient-xla-validation-20260918`; merge is gated.
Owner request: repair all findings in the September 17 audit, compare memory
and performance before/after, review this program, and merge only when tested.

September 19 refresh: source checkpoint `bbfaf742` is committed and pushed,
including compiled forecast shards and independently verified process comparisons.
The owner has authorized **another 48 GPU / 24 CPU process-hours**. The active
cumulative caps are **52 GPU / 32 CPU process-hours**, retaining every prior
charge. Through run 01130, charges are 14,705.260 GPU / 31,062.057 CPU seconds;
remaining allowances are 172,494.740 GPU / 84,137.943 CPU seconds. The earlier
16 GPU / 12 CPU proposal below is superseded, not an additional allocation.

The repair is incomplete. The static guard covers 176 sources with 1,157 exact
exceptions; it explicitly does not cover the whole repository. All F01--F20
terminal decisions remain open. Focused passes are checkpoint evidence;
terminal tests and comparisons must match the final source and harness.

Recovery review through 01130 confirms no active worker, the unchanged frozen
baseline, and the same cumulative budget (the extension is counted once).
The CPU score-cloud repair passes 11 checks; all 61 policy/controller checks
pass. Commit that tested checkpoint, then finish the pending GPU qualification
groups sequentially before changing the remaining preparation kernels. Keep
the 256 MiB host-memory investigation trigger and all numerical gates intact.

Resume in this order, using the same bounded runner and versioned artifacts:

1. Continue localizing the P59 36-dimensional assembly memory regression before
   a broad retry. Rank-one fitter sharing passes 15 focused and 37 existing
   fitter checks plus 12 adjacent-filter checks (01064--01066). Assembly 01067
   passes in 191.789 seconds with 9,939,348 KiB final process high-water RSS,
   versus candidate 01050's 354.963 seconds / 16,332,076 KiB and original
   baseline 01058's 74.134 seconds / 690,176 KiB. These single-process CPU
   observations remain repair diagnostics, not terminal performance evidence.
   The later stack localizes further graph construction in sequential
   transport callbacks; inspect their repeated coordinate-program construction.
   The first safe reduction is the reference grid: the owned `_axis_grid`
   implementation explicitly ignores its axis and uses the common [-1, 1]
   interval/configuration. Replace identical Case arms with that one grid;
   check existing transport values/pullbacks and graph growth before repeating
   the large diagnostic. Portable cache reuse remains a separate investigation
   because captured coefficients and graph context must retain their semantics.
   Recovery confirms all nine coordinate checks pass in 01069 after removing
   duplicate grid branches; the exact stale allowlist entry is removed and the
   guard passes. The matched assembly 01070 passes in 188.360 seconds at
   9,743,236 KiB peak host RSS. This modest descriptive change leaves the
   memory investigation open; avoid another broad retry without a new repair.
   The subsequent shared rank-one Legendre marginal passes all 17 coordinate,
   25 transport, six public pullback and 11 sequential CPU checks (01079--01082),
   plus all 59 policy/controller checks (01084). Assembly 01083 passes in
   151.950 seconds at 7,330,768 KiB (6.991 GiB) host peak. Four/eight-coordinate
   graphs shrink to 997/1,193 nodes. This remains well above the 0.658 GiB
   baseline; inspect remaining basis/mass and normalizer graph duplication.
   Shared owned-Legendre basis/mass graphs now pass 29 focused, 37 fitter,
   38 density, 25 transport, six public pullback and 11 sequential checks.
   Assembly 01094 passes in 112.504 seconds at 5,586,104 KiB (5.327 GiB) host
   peak. All 59 policy/controller checks pass in 01095. Record these as
   descriptive CPU improvements, with the original memory gap still open;
   proceed with the forecast-pool audit repair before another assembly retry.
   Deferred fitter pullbacks pass six GPU checks (01042); the
   coordinate repair passes nine CPU/GPU checks (01041/01043). Affected public
   pullback, sequential and transport suites pass 6/11/25 CPU checks
   (01046--01048). Their GPU checks remain pending during device contention.
2. Finish the reachable preparation/callback audit and explicit F01--F20
   dispositions. Preserve algorithms, total derivatives, thresholds, ordered
   operations and every seeded stream except the two approved initializers.
   Uniform log weights and the weighted target initializer now have stable XLA
   helpers; all 60 preparation checks pass on CPU (01045), with GPU pending.
   Ten additional preparation wrappers are now in the static guard.
   P72 fit/guard preparation passes 30 CPU checks (01057); separate compiled
   interpolation and exact duplicate decisions preserve realized-value order.
   GPU qualification passes all 81 guard/preparation checks in 01127; its
   earlier preparation checkpoint is `de363b43`.
   The additional support/line/spectrum gate repair passes all 81 CPU checks
   (01072), with 59 policy/controller checks in 01077. Its new measurement
   fixture passes CPU graph, XLA and both public source arms (01073--01076),
   with all 11 numerical summaries matching the original exactly. These are
   small diagnostic measurements; GPU and terminal repeats remain pending.
   The CPU forecast pool repair now executes a compiled whole shard and uses
   TensorFlow serialization. All 14 focused checks pass (01098); the three
   process-pool checks pass (01099), including exact scalar replay and uneven
   shards. Float32 scalar-factory compatibility is repaired and all 15 focused
   checks pass in 01101. The pool has 21 exact host-orchestration/reporting
   exceptions and no numerical row exemption. All 61 policy/controller cases
   pass (01121). Shard measurements 01103--01114 preserve baseline values,
   with exact XLA/public outputs and graph differences at most 6.107e-16.
   The public pool comparison first exposed child-source contamination (01115);
   source restoration and per-child verification repaired the harness.
   Runs 01116--01119 match exactly, but the five-row peak sum increases by
   285.984 MiB, triggering investigation. Final per-worker peak/signature
   snapshots are now required because a worker may not receive the last task;
   runs 01122--01125 now pass both pairs exactly with complete final snapshots.
   The five-row increase remains 279.852 MiB; each worker has both two-/three-row
   signatures, with one trace each. Investigate compiler/signature overhead
   before terminal memory acceptance.
   `cpu_xla_cloud` now has TensorFlow transport/result construction; all 11
   process and boundary checks pass in 01129. Its numerical XLA worker is
   unchanged. `quadratic_map_covariance` and `block_coordinate_center` have
   confirmed active preparation consumers and remain migration work.
3. Resolve the predator-prey residual (~1.578e-9 versus the unchanged 1e-10
   gate), core-affine/higher-rank slowdown, centered qualification, and host
   memory regression. Lazy pullbacks reduced measured host memory, but the
   four-date public endpoint still exceeded the original baseline by 362.6 MiB
   in a single-process diagnostic. That is an open investigation trigger.
4. Freeze repaired source and harness, run affected suites and all required
   paired three-process repeats, and review exact endpoint parity, compiled
   execution, host/device memory and warmed timing. Preserve invalid or stale
   evidence without admitting it to terminal comparisons.
5. Verify the terminal gate, integrate remote changes and retest affected
   code; merge and push only when every required gate passes.

Refresh review: the prior caps and pending-approval wording were stale; the
runner and this plan now agree. Source/harness churn invalidated earlier
comparisons, and repeated broad assembly retries did not identify the later
failure stage. Therefore complete source work before terminal repeats and use
bounded localization first. The pinned baseline, evidence contract, failure
criteria, hardware class, attempt limits and numerical tolerances remain valid.

Recovery after 01099: both approved GPUs are currently idle. Resume focused
GPU qualification sequentially through the driver while reviewing the forecast
measurement harness. Do not edit runtime or harness source while a numerical
worker runs. The remaining memory gap and terminal comparisons are still open;
none of the focused passes authorizes a merge.
Run 01100 passes all 29 basis/mass checks on GPU. Remaining affected GPU groups
still need qualification; the earlier 29 CPU cases do not cover those groups.

### Bounded fitting graph repair after 01058

The stack/RSS record in 01050 reaches 4.15 GiB during the initial fit and
6.39 GiB during the next fit. Inspection finds one full design/update graph
per axis, even when core shapes match. Share a single update/pullback program
for matching rank-one core shapes and static resource-gate outcomes; pass
their scheduled axis as a tensor. Higher-rank coordinates retain the prior
static selection after mixed-schema diagnostics 01063 showed rounding drift
in ill-conditioned histories. Preserve the original padded contraction,
schedule, solver, fixed-design derivative, rejection history and thresholds.
This is a mechanical repair of the existing weighted-ALS extension, not an
author TT-cross implementation. Paper Algorithm 2/(15)--(16), lines 693--725,
and author `models/full_sol.m:21--130` were re-inspected.

The skeptical review identifies risks in heterogeneous rank slicing, resource
rejections, discrete update order and captured pullback coefficients. First
run focused value/gradient parity against pinned pre-repair fitting, including
same-shape and heterogeneous cores, rejection and repeated sweeps. Check graph
growth before a single assembly retry. Existing numerical gates and the
900-second focused ceiling apply. Graph node counts and host peaks explain the
repair; terminal admission still needs current-source GPU checks and matched
three-process complete endpoint comparisons. No caches may be cleared only
for one measurement arm, and no timing or memory improvement excuses drift.

### Remaining P72 numerical gate boundary repair

Consumer inspection confirms `scripts/p73_density_aware_renewal_diagnostic.py`
uses `p72_line_probe_diagnostics` to select fitting data (lines 429--463).
Compile the complete line prediction/reduction callback, finite-cloud support
statistics, and heterogeneous singular-spectrum reductions. Keep scalar record
validation, ordered reason assembly and hashes at the host boundary. The
normalizer gate validates already materialized scalar fields and remains host
validation; its numerical normalizers are computed by the existing compiled
density path. No thresholds or fit-selection rule change.

This is a mechanical repair of local extension/admission code, not a new
source-faithfulness claim. Paper Algorithm 2/(15)--(16), lines 693--725, and
author `models/full_sol.m:21--130` were re-inspected. Risks include nonfinite
reduction behavior, empty clouds/spectra, strict threshold equality, index
rejection, ordered reasons, and nested callback compilation. Compare complete
public records against the frozen baseline, include these boundary cases,
check enclosing HLO and stable signatures, then run the existing P72 suite.
The same CPU/GPU qualification requirements, worker ceiling and cumulative
budget apply. This work cannot close assembly memory, terminal comparison,
scientific admission or the remaining preparation/callback audit.

The registered `source_guard_gates` fixture includes every numeric summary
from these three gates at 16/32 cloud points and four/eight recorded spectra.
Graph/XLA arms measure complete numerical statistics; public eager arms measure
both source versions including host provenance and record construction. Keep
the scopes distinct. Legacy graph/XLA failures stay visible; do not synthesize
baseline compiled timings. Public decisions, hashes, nonfinite rejections and
threshold boundaries are checked by the paired correctness suite. Qualify one
small CPU diagnostic before terminal three-process GPU repeats under the same
20-warm-call measurement contract.

### Bounded transport graph reduction after 01070

The large assembly uses equal-shaped rank-one Legendre cores. Its masked
marginal currently constructs a separate basis/mass/contraction Case graph
for every axis, despite the common algebra. Share that body for exactly this
static schema, passing each core and interval endpoints as tensors. Keep the
existing paired-core einsums, polynomial recurrence, multiplication order,
measure convention and all per-axis domain derivatives. Distinct basis
families, degrees or core shapes retain their existing heterogeneous path.
This reduces graph duplication without introducing a cross-context cache.

Review risks: tensor-selected bounds must retain every captured derivative;
nonconstant rank-one cores, unequal domains, both measures and graph/XLA
execution need pinned parity. CDF/bisection and veto rules stay unchanged.
Use focused complete coordinate and marginal pullback tests first, then the
existing public/sequential/transport checks and one unchanged assembly memory
diagnostic. Reject numerical drift at the original 1e-10 gate. The local
Legendre and grid-CDF extension is not an author-algorithm promotion.

## Question and scope

### Bounded basis graph reduction after 01083

The remaining basis-row and mass helpers build one Case arm for each axis,
including equal-degree owned Legendre bases. Share these exact-class schemas
with tensor interval endpoints and one native axis body; preserve the original
polynomial recurrence and multiplication order. Retain heterogeneous and custom
basis behavior, every query/domain derivative, axis selection/order and mass
measure. Do not enlarge global caches or mix graph/XLA contexts. This is a
mechanical repair of the local Legendre extension, not an author-basis claim;
paper Algorithm 2/(15)--(16), lines 693--725, and author
`models/full_sol.m:21--130` were re-inspected.

Skeptical review: shared basis arithmetic must not perturb ill-conditioned
fitter histories, and a missing derivative or a subclass override is a veto.
First compare complete values and query/interval gradients with checkpoint
`1da3170e`, including both measures, permuted/subset axes, higher ranks and
heterogeneous/custom fallback. Then run existing fitter, density and transport
checks before one unchanged assembly diagnostic under the same 900-second
ceiling. Graph counts and single-process RSS are explanatory; terminal paired
comparisons and GPU checks remain required. Existing 1e-10 gates and cumulative
budget apply; no retry without addressing the observed failure.

### CPU score-cloud broker repair

`cpu_xla_cloud` exports a persistent CPU process broker whose actual worker
already compiles the complete supplied B=1 value/score callback with a fixed
signature and XLA enabled. Its Python loops dispatch tasks, wait for futures
and pack completed results; they do not calculate scores. Remove NumPy from
input validation, transport and immutable result construction using TensorFlow
and standard-library types. Preserve exact task/row order, inherited CPU-only
bootstrap, worker persistence, heartbeat semantics, exception propagation and
the existing numerical callback. Keep TensorFlow imports lazy so spawned
children still configure their environment before framework import. This
scalar independent-score lane is ineligible for NeuTra training.

Skeptical review: a top-level TensorFlow import would break bootstrap isolation;
mutable inputs must become immutable result snapshots, and tensor conversion
must preserve binary64 values and noncontiguous/non-native-order buffers.
Exercise the real two-process quadratic value/score fixture, reversed rows,
invalid shapes/nonfinite inputs and worker failures before exact static
classification. No numerical algorithm, RNG or HMC tuner changes are included.
The source audit also finds real MacroFinance consumers of the quadratic MAP
and block-coordinate initializers; their diagnostic names do not exempt active
numerical decisions. Keep those distinct repairs open. The HMC interface and
capability registry were inspected; these are preparation helpers, not tuners.

### CPU forecast pool execution repair

The owned pool calls `ComplexityForecastWorker.evaluate` in a Python row loop
and uses NumPy for transport and validation. Move the complete shard recurrence
into a stable-signature TensorFlow/XLA program while preserving every row's
separate seed and existing terminal/process/observation draws. Keep ordered
process orchestration, request/hash checks, startup barriers and completed
tensor serialization on the host. Reuse the repository tensor byte helpers to
preserve the existing contiguous raw-byte identity hashes. Do not reseed a
whole shard from one row or replace per-row streams with one bulk stream.

This is external forecast generation on multicore CPU. Native tensor mapping
of independent forecasts is eligible here and must not be described as a
batch-native NeuTra training target. Preserve finite/covariance/variance vetoes
and row order. Review the actual owned callback, which currently materializes
status on the host; wrapping it blindly in `tf.map_fn` is invalid. Baseline
the committed scalar endpoint on identical rows/seeds, require exact replay
and identical input hashes, verify complete shard HLO and one trace per fixed
signature, then run the existing process-pool startup/replay tests. Include
uneven shards and invalid-result rejection. Qualified fixtures must measure
complete public pool overhead separately from numerical kernel time.

Registered measurements use `cpu_forecast_shard` and `cpu_forecast_pool`, both
explicit CPU generation lanes at q=1, two/five rows, two replications and ten
forecast dates. Seeds are (20260719, 70001+i), with identical rows in both
isolated source arms. The shard records graph/XLA numerical execution and
public host execution separately; the legacy scalar host boundary may reject
tracing, in which case its public execution is a parity reference only.
The process pool uses two persistent one-core workers and records full cold
startup, twenty warm calls, IPC, ordering/identity validation and output
conversion. Record each call's parent and worker RSS maxima; their sum is a
high-water sum, not simultaneous live memory. Preserve the 1e-10 parity gate,
exact replay, one-trace/HLO checks and the existing 256 MiB host-memory repair
trigger, also applied to the process peak sum. These diagnostic sizes establish
no scaling, training or posterior claim. Run one qualification pair per extent
before the existing final three-process repeats; each worker retains the
300-second ceiling and existing cumulative budget. No additional RNG exception.

Skeptical review: changing output transport from NumPy arrays to TensorFlow
tensors is the intended backend repair, but no owned consumer may depend on
mutable NumPy output. Only the owned complexity worker and its tests were
found in the current call-chain search; check its factory protocol explicitly.
Random stream drift, status loss, row/hash mismatch or unbounded retracing
are repair triggers and block acceptance. Use the existing bounded driver,
versioned artifacts and cumulative budget; no training or posterior claim.

Can every owned runtime route identified in findings F01–F20 preserve its
declared value/score semantics while removing Python numerical iteration,
nonreference NumPy, implicit pfor, and incomplete/default-off XLA boundaries?
Does the repaired complete calculation change trace/compile cost, host/device
memory, or steady execution time on the same inputs and hardware?

The source baseline is `3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf` (the prior
Kalman/UKF repair). Commit `39ad4f67` preserves the audit; its numerical source
is identical. No old LEDH result is reused. Baseline execution here is an
explicit engineering diagnostic, including any legacy implementation that
fails current policy. It grants no scientific or runtime admission.

The inventory and F01–F20 report define the initial scope. Follow actual
consumers, callbacks, preparation and imports into shared dependencies. Add
newly discovered reachable violations to the ledger rather than leaving them
outside the gate. Reference and archived implementations remain available as
such; relabeling an active runtime or deleting its tests cannot close a finding.
No algorithm substitution, changed score definition, hidden stop-gradient,
new reset rule, relaxed tolerance, or numerical retuning counts as a repair.

Owner-approved September 17 exception: migrate the two geometry initializers'
NumPy PCG64 probe-cloud RNG to the versioned TensorFlow stream
`geometry_tf_philox_cpu_xla_v1`. Seeds intentionally produce different clouds;
record the stream ID, seed, call order, shape and TensorFlow environment.
Numerical before/after comparisons must inject identical frozen clouds into
both versions. This does not authorize changing the sampling distribution,
thresholds, holdout roles or fit algorithm.

Scope confirmed by the owner on September 17: repair execution and block
unsupported canonical LEDH claims. The full canonical LEDH rebuild is outside
this campaign. Existing AD and finite-program manual-JVP scores retain their
explicit diagnostic semantics and cannot advertise training/HMC or canonical
admission. Compilation evidence does not remove that restriction.

## Completion ledger and order

Each finding must have source changes or a supported non-runtime classification,
an executed consumer/fixture check, relevant parity and compilation evidence,
and a recorded terminal decision. A passing subset never means complete.

| Phase | Findings | Work and required checks |
|---|---|---|
| 0 | all | Preserve baseline; inspect current callers; freeze comparison inputs; review plan and runner; establish resumable budget/logging. |
| 1 | F09–F14, F17, F20 | Native SQMC/initialization recurrences; TF quadrature and retained moments; SGQF preparation; TF admission/serialization; explicit analytical model derivatives or approved non-pfor model-local derivatives; tensor state simulators; remove indirect NumPy control. Check ordering, quadrature, moments, exact serialization and derivatives. |
| 2 | F01–F02, F07–F08 | Tensor time/particle/pseudo-time/parameter recurrences and batch-native algebra. Audit analytical score provenance through consumers. Preserve all Contract E moment/weight/transport terms. Existing finite-program AD/manual-JVP diagnostics cannot become canonical analytical scores by renaming. Canonical consumers must enforce the current rebuild contract. |
| 3 | F03–F06 | Pack heterogeneous TT core/checkpoint state as necessary; native forward/reverse time and ALS loops; preserve fitting branches, ranks, schedules and frozen proposals. Inspect paper and author-source anchors before any Zhao–Cui route behavior change; record mechanical execution refactors separately from algorithm changes. |
| 4 | F15–F19 | Batch-native CPU shards and TF serialization; XLA-default stable-signature factories; remove runtime NumPy from reachable geometry/identity/failure paths; isolate reference imports. Check all consumer signatures and role classification. Consult the HMC capability registry; no sampler/tuner algorithm change. |
| 5 | all | Repeat paired measurements, complete static/consumer guard, affected suites and integration tests; review changes and results; fresh remote integration check; merge only after all gates pass. |

Work within a phase may proceed independently when dependencies allow. Fixes
are committed in reviewable groups; artifacts record the exact source hashes
even before their source commit. Keep recovery status in the campaign ledger.

## Comparison contract

Use fresh processes for before/after × graph/XLA, same pinned fixture builder,
inputs, seed, shapes, dtype, outputs, device, TF environment, thread counts and
output lifetime. The baseline kernel comes from the pinned Git source, never
the edited checkout. Preserve actual failures: a baseline that cannot compile
has no invented XLA timing. Compare its existing valid execution separately.

Fixtures cover every repaired numerical family (complete endpoints whenever
available): Kalman/SRUKF, SGQF/model derivatives, SQMC, GenUT/Contract E,
particle/flow, TT forward/adjoint/APF, preparation and CPU score consumers.
Primitive evidence cannot substitute for complete endpoint evidence. Start
small, then use at least two horizon/parameter/particle extents for changed
recurrences and three fresh-process repeats for reported timing ratios. Use
FP64 parity authorities and relevant FP32/TF32 default execution checks.
TF32 tolerance must be justified before its fixture executes. Use fixed random
streams without retuning; obey the exact transport chunk rule.

Record preparation, trace, first execution (including compile), optimized HLO,
GraphDef node count, warmed execution, and output-copy time separately. Warm
timing synchronizes the returned tensors; serialization is outside timing.
At least 20 warm calls test fixed-shape allocation stability. Report /proc
host RSS/HWM and TF allocator current/peak; driver reservation is separate.
Inputs and output lifetimes match across arms. Check nested XLA boundaries so
the graph diagnostic is genuinely non-XLA. No Python callbacks are permitted
in a passing numerical graph. Check bounded tracing and graph growth.

Primary pass criteria: preserved values and analytic derivatives against
baseline plus independent identities/finite differences where meaningful;
finite outputs/status and branch parity; unchanged downstream consumer
semantics; policy guard clean for every admitted path; successful enclosing
GPU/XLA execution. Default FP64 comparison is atol=rtol=1e-10 unless a
documented existing numerical test requires a stricter or condition-aware
criterion. Discrete ordering and status outputs must match exactly.

Performance/memory ratios are explanatory, not mathematical validity. More
than 20% repeat-median warm-time regression, 2x device peak, 256 MiB additional
host memory, continuing fixed-shape growth, or horizon/parameter-dependent
trace unrolling triggers investigation before merge. A justified unavoidable
tradeoff must be recorded; unexplained regressions remain open. With three
process repeats, report descriptive medians/ranges, not statistical superiority.
The program does not certify posterior convergence, HMC readiness, LEDH
canonical scientific admission, or Zhao–Cui source-faithfulness from compilation.

## Environment, budget and stop conditions

Use the existing `/home/ubuntu/miniforge3/envs/tf-gpu` environment. No installs,
environment mutation, network research downloads or paid compute are included.
GPU2 is the initial device; check contention before launching. One GPU process
at a time; memory growth must be set/verified before any numerical import.
Two intra-op threads and one inter-op/OpenBLAS thread. CPU reference/test
processes explicitly hide GPUs. Test orchestration can use Python loops;
numerical implementations cannot use them to evade the policy.

The active total budget is 52 GPU process-hours and 32 CPU process-hours, with
at most 300 seconds per benchmark worker, 900 seconds per focused test group,
three unchanged-fixture retries, and fresh numbered artifacts for every attempt.
Focused tests may select a smaller 60, 120 or 300-second timeout through
`--test-timeout-seconds`; the default and ceiling remain 900 seconds. The
selected limit is both the pre-launch reservation and enforced worker timeout,
and unfinished attempts are charged that same limit. This permits bounded
checks near the cap without expanding cumulative compute authorization.
Elapsed failed attempts consume budget. The runner records commands, source,
environment, hardware, outcomes and wall time in ordinary JSON/log files under
`docs/plans/artifacts/filter-gradient-repair-20260917/`. No overwrite of prior
evidence. Crash recovery charges unfinished attempts their reserved timeout.
No posterior chains or learned-transport training campaigns are in this budget.

### Historical budget proposal, September 18 (superseded September 19)

The following proposal was not approved and is retained as planning history.
The September 19 owner authorization above adds 48 GPU / 24 CPU hours to the
original 4 GPU / 8 CPU caps instead. Its larger active totals are enforced by
the existing driver; prior accounting is unchanged.

The comparison audit at run 00834 has only 37 current pairs and 695 missing
pairs. Earlier measurements are preserved but many have stale harness or
shared-source hashes. GPU2 still has unrelated work; completing matched groups
on idle GPU3 requires both arms and every repeat on that same device.
Enumeration of the registered matrix finds 1,521 pending GPU jobs and 24 CPU
jobs. Using historical per-fixture/arm/mode/size median durations where known,
and 30 seconds for 708 jobs without a measured duration, estimates 33,764 GPU
and 720 CPU process-seconds. This is a planning estimate, not a runtime bound.
Passed test-group durations sum to 4,262 seconds; nine groups have no passing
duration, including the unresolved TP residual check.

The earlier proposal was **16 GPU process-hours and 12 CPU process-hours**,
inclusive of all time already charged. Through 00834 the charge was 8,676.105
GPU and 20,882.313 CPU seconds. No extra device class, package
change, posterior sampling, training campaign, algorithm change or relaxed
criterion is authorized by this proposal. One GPU worker at a time, the
300/900-second worker limits, three-repeat rule and contention checks remain.

Complete the remaining source/callback audit before the terminal repeat
matrix. Use only focused qualification during implementation, then freeze
source and harness for final repeats and affected suites. A new repair after
freezing invalidates its affected measurements and consumes the same total
budget. This addresses the avoidable evidence churn found in the recovery
review. A failed gate still blocks integration; budget authorization cannot
substitute for source coverage, numerical parity or final review.

September 18 isolation repair: validate in linked worktree
`/tmp/bayesfilter-filter-gradient-xla-validation-20260918` on branch
`repair/filter-gradient-xla-validation-20260918` when concurrent unrelated
edits occur in the primary checkout. Commit campaign changes separately from
those edits. All worktrees share the original artifact root, lock and cumulative
budget through Git's common directory. Use the same bounded driver in that
worktree; its absolute program path is the only additional approval prefix.
The target, fixtures, seeds, hardware, tolerances and promotion gates are unchanged.

September 18 contention repair: focused correctness tests may explicitly use
`--test-gpu-index 3` on the idle RTX 4090 while GPU2 has unrelated work. This
uses the same GPU hardware class, contention thresholds, process limit, and
cumulative budget. The option is restricted to tests and their matrix stage;
all before/after measurements remain pinned to GPU2. Test logs record the
visible device, TensorFlow version, TF32 state and verified memory-growth
policy. This is an infrastructure repair, not a change to the comparison
contract or authorization for additional compute.

September 18 sustained-contention amendment: new measurement groups may also
select GPU3 explicitly, preserving the RTX 4090 hardware class and original
compute budget. Within each fixture/size/mode, both source arms and all three
fresh-process repeats must use the same physical GPU. The driver must not
resume a GPU2 arm into a GPU3 pair, and the comparator must reject mixed-device
repeat aggregates. Existing GPU2 measurements remain usable only in complete
matched groups. This supersedes the temporary test-only restriction above;
GPU2 remains the default selection. The contention thresholds, memory-growth
policy, scientific fixtures, tolerances and stop conditions are unchanged.

Extend numerical preparation coverage to the complete core-affine loss/gradient,
additive and pair fitted initializers, seeded balanced/residual initialization,
and prefix-score training targets. Use fresh exact constant-density parents
and frozen tensor inputs, the original settings/seeds, two extents and three
repeats with 20 warm calls. No fitted historical parent is reused. Each result
includes every returned numerical field relevant to the endpoint. A baseline
that cannot trace retains its failed attempt and uses its valid eager reference
for parity, without inventing a baseline compilation time.

Generic stochastic TT coverage uses fresh fixed heterogeneous cores, four/eight
axes and four/eight rows. Compare the complete density objective/gradient and
one Adam update for both density fitting and square-root prefit. Every timed
update restores identical input parameters and optimizer slots in both arms;
reset time is included and no optimizer trajectory is treated as a frozen
input. Preserve all returned numerical terms, updated parameters, and optimizer
state. Use the existing loss, clipping, regularization, and seed semantics;
these fixtures establish execution parity, not a trained-model quality claim.

Centered-training coverage also includes the complete absolute-density objective
and gradient, all three Adam callbacks, complete batch/ratio target preparation,
and the existing shuffled prefix schedule. Use fresh exact parents, four/eight
sample rows and 32/64 schedule rows. Preserve every returned numerical field
and optimizer slot, restoring identical state before each update as above.
Stream checks require exact original Philox/Fisher-Yates permutations. Reuse
the existing dtype, comparison thresholds, three repeats and cumulative budget.
Keep baseline graph/XLA failures visible and compare to a valid eager reference
where necessary; no historical fitted parent is required for these checks.

Source-route boundary checks also compare the unchanged affine density,
stability shift, proposal correction, normalized weights, ESS and normalizer
calculation at four/eight samples. Preserve host rejection and keep invalid
inputs nonfinite inside enclosing XLA, where assertions can be discarded.
Use the same GPU, FP64 tolerance, frozen inputs and existing budget. These
helper checks cannot close retained-object/date-loop or transport coverage.

Complete source endpoint qualification additionally covers retained samples,
proposal/target densities, correction and normalized weights, ESS, normalizer
increments, and previous-marginal values/query gradients at four/eight rows.
Use the same fresh correlated two-axis TT and fixed affine frame in both arms,
with grid size 9, eight bisection steps, the existing 1e-12 CDF tolerances and
zero allowed floors. These small extents diagnose execution only. Check
captured tensor and variable pullbacks as well as explicit query derivatives;
an external tape must not silently lose coefficients captured by a callback.
Prepare immutable marginal metadata outside tracing, retaining every numerical
query in the compiled boundary. The remaining sequential date loop is a
separate open gate; endpoint qualification cannot close it.

Sequential qualification uses the existing frozen source-route replay at two
and four dates, including different row counts in the correctness fixtures.
Preserve every retained numerical field, previous-prefix density, callback
derivative and retained-object link. Numerical date evaluation uses native
TensorFlow control flow; public record/identity assembly stays on the host.
The fixed-TTSIRT preparation creates schema-only views, so basis contraction
and normalization remain inside the date graph. Existing contracted-density
metadata is constructed only for reporting after numerical execution; no owned
consumer reads that field. Distinct frozen callbacks have distinct static
branches, so measure and explain graph-size growth before making a compactness
claim. This is execution qualification of frozen inputs, not new fitting,
adaptive-filter, HMC-readiness or source-faithfulness evidence.

Sequential qualification at runs 00931--00940 exposed duplicated transport
graphs even when dates share the very same frozen transport. The bounded
repair separates transport dispatch from target/date dispatch: one TensorFlow
branch per distinct transport object and input shape, with native date
iteration and padded heterogeneous tensor schemas. Repeated use of one
transport must not copy its CDF graph for every date. Distinct fitted
transports and callbacks may still require distinct schemas; do not claim
constant graph size for that case. Preserve all outputs, callback/query/core
derivatives, and exact invalid-input behavior. Requalify the same two/four-date
fixture and retain the failed memory/graph observations as repair evidence.

The expanded captured-core derivative check also found nested basis pullback
captures and standalone TT TensorList-boundary failures. Preserve complete
derivatives with complete local recomputed pullbacks and by reusing that
boundary for public TT/density/transport calls. Pretrace the derivative using
graph differentiation, which remains valid if an outer initialization scope
pauses tape recording; bind inputs and resource captures in the forward
branch, and keep loop tapes internal. The shared boundary belongs in `ops/compiled_tensor_program_tf`
and must remain covered for explicit inputs, captured tensors and variable
resources. Pin the comparator's source, TT, density and transport modules
together; a legacy host loop importing candidate dependencies is not an
independent baseline. Recheck the source closure and artifact reload after
moving the helper. These are execution repairs with unchanged score meanings.
The boundary uses TensorFlow's concrete-function and branch-graph APIs; the
independent polynomial tests must cover explicit arguments, captured tensors,
resources and nested control flow in graph and XLA modes, including cold
construction inside an outer `tf.init_scope()`. Any TensorFlow
upgrade requires these checks again. These AD pullbacks preserve the existing
external-tape semantics and do not establish a canonical analytical LEDH score.

September 18 cold-graph repair: defer complete VJP graph construction until a
pullback is requested. This supersedes eager pretracing above while retaining
graph differentiation under the original primal/XLA context. Bind primal
resources before branch gradients and bind custom-gradient-only coefficients
through the enclosing function graph. A value-only call must not trace the
pullback; subsequent gradients must preserve explicit inputs, tensor/resource
captures, initialization-scope behavior, and nested Case/While semantics.
Test scalar and larger captures, repeated calls and resource updates before
repeating the same public sequence memory fixture. This changes construction
timing only, not the gradient definition or the CPU/GPU comparison contract.

September 19 fitter localization: the 36-dimensional preparation consumer
exceeds its bounded test and reaches 27.67 GiB observed RSS. Its timed stack
shows construction of per-core fitting update graphs. Defer the existing
accepted-update pullback graphs until requested, retaining the original
fixed-design derivative, accepted/rejected branches, solver, ranks, sweep
order and every threshold. Qualify value-only construction, heterogeneous
core/target pullbacks, rejection derivatives and the existing scalar consumer
against the prior source before retrying the full assembly. Do not extend
the 900-second ceiling or regard a passing tiny fit as large-case closure.
Duplicate forward environment construction remains a separate investigation.

September 19 transport localization: run 01038's 135-second stack reaches
retained-sample construction and the TTSIRT coordinate/marginal graph builder,
after both initial and step-one fits. Inspect the later snapshots before the
next retry. The current coordinate factory duplicates the complete paired-core
marginal and normalizer graph for each prefix/suffix. Repair by sharing a
fixed-shape marginal program with a tensor mask and a single coordinate-loop
body; preserve heterogeneous basis/domain dispatch as a static schema. Compute
the unchanged normalizer once for the complete transport call. Retain the
exact paired-core contraction order, forward/suffix endpoint interpolation,
trapezoid and bisection rules, all floors/status precedence, and full query,
core, mixture and captured-basis derivatives. No tolerance or fit change is
authorized. This is a mechanical execution repair of the existing grid-CDF
extension, not a new source-faithfulness claim. The paper Algorithm 2/(15)--(16)
and author `models/full_sol.m:33--38,76--130` anchor the surrounding route;
the local grid-CDF itself remains an extension.

Qualify prefix/suffix, heterogeneous core/basis and both reference measures
against the pinned previous implementation and independent density identities.
Check enclosing graph/XLA execution, preserved invalid statuses, external
pullbacks, and dimension-dependent graph size before the large consumer retry.
Terminal before/after memory evidence still uses the frozen campaign baseline
and three fresh-process paired repeats; localization snapshots cannot replace it.

The next reachable preparation repair must enclose prior sampling, transition
noise and model callbacks, deterministic weighted resampling, recentering,
local clipping, target evaluation and target-value construction. Its consumers
are `p59_author_sir_step_spec_assembly` and the P72 lower-gate diagnostic.
Compare the complete first-date and retained-prefix-date preparation results,
including indices, clipping decisions, frames, shifts and weights. Preserve
the existing Generator Philox stream (seeds 6301 and 6400+t), all settings and
the fixed resampling rule; the geometry RNG exception does not cover this
preparation. Paper section 3.2 Algorithm 2/(15)--(16), section 4.1's previous
marginal recursion, and author `models/full_sol.m:21--130` anchor the operation
ordering. Frozen replay is a fixed-HMC adaptation; deterministic resampling,
bounded local clipping and the grid-CDF fit remain repository extensions.
This is still an open execution gate, independent of passing replay tests.

September 19 follow-on preparation audit: P72 line probes feed coefficient
fitting in `scripts/p72_support_certified_lower_gate_diagnostic.py::_fit_p72_step`
and renewal selection in P73. Their diagnostic harness names do not exempt
the reachable numerical design construction. Replace the fraction loop and
host materialized duplicate-column decisions with fixed-signature tensor
programs that preserve endpoint selection, fraction-major ordering and first
exact duplicate retention. Interpolation and duplicate detection execute in
separate XLA calls: predicates must compare the realized coordinates rather
than compiler-recomputed expressions with different rounding. Return
fixed-size packed column indices and a count;
the host may resolve the public variable-length output schema, with selection
performed by a compiled gather. The original line-cloud construction detaches
inputs through host materialization; preserve and document that frozen-design
boundary. P72 set-weight normalization and fit/guard batch assembly must also
execute in XLA, retaining their validation, weights and audit exclusion. These
are repository extensions, not author-source changes. Check duplicate/degenerate
clouds, endpoint indices, frozen output semantics, invalid inputs and full
training-batch values/derivatives against the pinned baseline before admission.

For the large P59 completion blocker, the existing localization test may also
run with `--arm before --device CPU --test-timeout-seconds 900`. This uses the
frozen campaign baseline and the same current watchdog/test harness, with GPUs
hidden. Compare completion and final process high-water RSS descriptively to
01050; a single pair is localization evidence only and does not replace the
three-process GPU/public-endpoint timing matrix or establish memory acceptance.

Sequential timing review found that the original eager arm included public
result/identity assembly while the candidate graph/XLA arm contained only the
complete numerical kernel. Their equal numerical outputs do not justify an
end-to-end speed or peak-memory ratio. Record each timing scope and suppress
cross-scope ratios. The sequential matrix therefore additionally measures both
public endpoints in host-call mode, retaining their actual internal execution
defaults (candidate numerical XLA; historical eager baseline). Require those
matched public-endpoint pairs for timing comparisons. Graph/XLA kernel arms
remain compilation, graph-growth and allocation diagnostics, with the valid
baseline outputs as their parity authority. The public API fixture must reuse
the immutable callback/frame/transport schema when reference values change;
specification object identity alone must not force a new compilation. All
outputs, 20 warm calls, process repeats, hardware and existing budgets remain.

Stop the affected measurement on invalid comparison, corrupted artifacts,
numerical mismatch, uncontrolled allocation or GPU contention. Repair local
harness defects within the same scope/budget. A failing candidate prevents its
merge, not independent repairs or a bounded retry. Budget exhaustion requires
a recovery report and further authorization; never mark remaining work done.

## Permission allow list

Use one reusable tool-approval prefix for the bounded campaign driver:

```
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python /tmp/bayesfilter-filter-gradient-xla-validation-20260918/scripts/run_filter_repair_campaign.py
```

Its allowed actions are status, pause, a fixed test group, a registered measurement,
the sequential registered matrix, audit, comparison and gate verification.
The matrix has qualification, three-repeat and current-source test stages;
it resumes current evidence, stops on source changes or candidate failure,
checks the selected GPU's contention before each launch, and uses the same
cumulative budget. The active worktree uses `--test-gpu-index 3` for tests and
`--measurement-gpu-index 3` for measurement groups. Existing GPU2 evidence
remains preserved; compared arms and repeats must use the same device.
The pause action lets the active worker finish and stops before the next
worker. A subsequent explicit matrix command resumes the campaign.
It has no arbitrary shell/command
argument, package installation, network operation, deletion, Git merge or push.
This process allow list is distinct from the source policy exemptions: those
must name exact reference/reporting/schema functions with reviewed reasons;
there is no blanket module or numerical-loop exemption.

Existing Git approvals cover branch/commit/fetch/merge/push. User authorization
already covers execution and conditional merge. Request the driver approval
once when it is concrete and reviewed; reuse the exact prefix for all retries.
No broad Python, bash, or external-review-tool approval is needed. A platform
prompt can still be required; this plan cannot waive it.

## Skeptical review before execution

Review outcome: proceed with the above gates. The following defects in a naive
program were identified and addressed before launch:

1. Merely wrapping loops in XLA leaves unrolled graph growth: native tensor
   recurrence plus horizon/parameter graph checks are required.
2. Reusing the edited checkout for “before” contaminates the comparison: use
   the pinned source snapshot and record imported source hashes.
3. A faster different score is a wrong comparison: preserve semantics and
   test total derivatives and consumer wiring; AD cannot establish analytical
   LEDH admission.
4. A primitive-only benchmark misses full TT/GenUT compilation memory: require
   complete endpoint coverage before closing those findings.
5. Output copying, allocator reservation and import cost distort measurements:
   separate phases, synchronize identically, and retain both host/device data.
6. A hand-maintained pass flag could permit premature merge: the final gate
   must require all finding closures and passing, current-source test and
   comparison artifacts. Missing/skipped required checks fail the gate.
7. A permissive allow list could hide violations: only bounded orchestration
   is approved; source exceptions remain specific and cannot admit runtime debt.
8. Remote integration can invalidate prior tests: inspect the integrated tree
   and rerun affected gates before advancing main; no force push/reset.

This is the primary agent's plan review. No independent-agent review is claimed.
Terminal review must revisit these failure modes against executed evidence.
