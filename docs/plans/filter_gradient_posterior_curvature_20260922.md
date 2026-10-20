# Fixed-center posterior curvature execution repair

September 22 update: [the owner-directed rejection criterion](filter_gradient_rejected_dense_precision_decision_20260922.md)
supersedes the ill-conditioned discarded-matrix equivalence requirement below.
Runs 02622--02625 pass all 150 CPU/GPU posterior checks, with unchanged accepted
results and no downstream use after fit rejection. Policy 02626 passes 102
checks. Preserve all historical failures; the uninstalled allowance and
unfinished GPU reference are no longer acceptance requirements for this matrix.
Public integration and full campaign qualification remain open.

Continue the owner-authorized F18/F19 execution campaign from pushed `96e15e9d`.
Uniform public GPU consumers now pass (02403--02406); current-source costs
remain to be renewed after the shared COD repair documented below. The native
posterior GPU matrix is executing after a diagnostic resource-placement repair.
The public fixed-center regional curvature handoff still uses Python partition,
batch, replicate and pairwise consensus loops; the new native candidate is not
yet wired into that public endpoint.

The question is whether the complete fixed-center calculation can execute with
one stable TensorFlow/XLA program while preserving the original algorithm and
complete result. Baseline: the isolated `3582b4ac` dependency closure, including
its original dense graph least-squares implementation. Compare the candidate's
explicit complete graph reference and default XLA program against that closure.
No recentering, precision projection, damping, tolerance, optimizer or sampling
distribution change is authorized. Preserve the Philox seed-to-draw map using
the already qualified bit conversion and normal transform; compare original
box/ball/proposal draws directly before interpreting controller results.

Preserve center evaluation; all training partitions before all selection
partitions; fixed-size padding; support checks before target evaluation; charges
for failed attempted batches; ordered per-replicate dense fits and gates; all
pairwise generalized eigenvalue checks; the mean accepted precision; untouched
audit; factorization/reconstruction gates; and the separate normal proposal
audit. Rejection exposes no geometry. Host code may validate configuration and
format completed records, but cannot make numerical decisions during execution.
Use fixed capacities and `tf.while_loop`, not Python unrolling, pfor or row-wise
mapping of scalar targets. Center, pilot factor and seed are runtime operands.

The complete-record criterion is `atol=rtol=1e-10` with exact schemas, discrete
decisions, counts, partition order and shapes. The approved deficient-rank
`design_condition=inf` normalization is the only expected baseline diagnostic
change. The iterative factor-fit and huge-stress allowances do not apply here.
Require independent Gaussian covariance/orientation checks, nonlinear and support
vetoes, all attempted-batch accounting, malformed callback errors, changed-input
reuse, single trace and unchanged HLO/runtime operands on CPU/GPU. Python callback
side effects in old diagnostic tests must become TensorFlow resource recordings
before compiled consumers are tested; do not silently remove their assertions.

Before public wiring, qualify native full records and original random streams.
Then retain the public result type and serialized numerical schema with truthful
execution metadata. Check actual repository consumers and bounded cache/resource
lifetime. Compare original/graph/XLA complete public costs at D3/D5 using three
fresh processes per arm/device. Include complete payload materialization, cold
construction/tracing/compilation and warm calls, host RSS/VmHWM and GPU allocator
current/peak. Investigate >20% warm time, >2x cold/device memory or >256 MiB extra
host RSS without treating reservation as live tensors. Timing is descriptive;
no statistical superiority, broad leak freedom, HMC, posterior or DZ5 promotion
follows. A failed comparison blocks public admission and triggers local repair.

Skeptical review: the old dense XLA solver is not the original graph numerical
authority. Ordinary XLA random-double and normal kernels change seeded draws.
Graph tracing invokes Python callbacks, so Python counters do not measure target
execution. XLA can ignore TensorFlow assertions; represent finite/support gates
as explicit tensor statuses and preserve programming errors. Pairwise consensus
must check every pair and reciprocal, not only adjacent replicates. Partial
results must retain the same diagnostic fields as the original. Constant-folded
seed/configuration can hide retracing, and timing without payload/cold costs can
mislead. The checks above address these risks. Primary-agent review only.

Use the existing stable campaign driver, one numerical worker, focused 300-second
timeouts (900 only for a registered longer group), CPU references with GPUs
hidden, GPU3 idle preflight and verified growth. Keep runtime/tests/driver frozen
during workers. Existing cumulative caps remain 32 CPU/52 GPU process-hours;
through02364 charges are 48907.31598352069 CPU and46001.71196921611 GPU seconds.
Stop launches for contention, invalid provenance or budget exhaustion; preserve
all failed attempts in unique directories under the shared campaign artifact
root. No external consumer source/pin, package or environment changes. Broader
controller work, actual DZ5 transitions and F01--F20 terminal gates remain open;
main must not merge before they pass.

Run02365 preserves a seeded-stream failure: `stateless_fold_in` with its default
`auto_select` chooses a different stream inside XLA. Six stream cases fail before
cloud comparisons; the independent all-pairs consensus check passes. Bind
`alg="philox"` in the native seed folding, matching the original non-XLA route,
then repeat original seed/cloud/proposal comparisons before controller tests.

02366 passes all seven original stream/consensus checks.02367 passes the first
complete controller comparison.02368--02370 pass51 full D1/D3/D5 records, each
against original in both graph and XLA modes, including failed batches and both
holdout rejections.02371 passes the two complete three-replicate ball records,
callback counts and one trace, but its two HLO strings differ. Preserve and
inspect both HLOs and runtime operand counts before claiming stable compilation.
The first failure lacked the full compiler artifacts; the next diagnostic writes
them before asserting. No runtime source or numerical tolerance changed.

02372 localizes all20 differing HLO lines to Grappler's `zeros_N/_M` synthetic
node uniquifiers in dummy-source metadata. Both full HLOs have8537lines and all
six runtime operands (three numerical inputs plus three diagnostic resources).
Operations, constants, shapes and operand wiring are identical. The comparison
now normalizes only that exact metadata name pattern; all other characters
remain checked. Preserve raw HLO and all complete changed-input records. This
is a diagnostic correction, not runtime specialization repair or a relaxed
numerical criterion. Add independent geometry, singular-design, all-replicate
instability, malformed callback and single-entry ownership checks before costs.

02373 passes complete changed records, all runtime operands and the narrowly
normalized HLO comparison.02374 passes15 extra checks but preserves two rejected
design precision discrepancies and a diagnostic callback arity mistake. An
exact33x3 all-ones design reports a0.3783precision difference; the ill-conditioned
fixture reports2.91e-9 in one field. Both remain strict comparison failures.
Instrument the existing shared COD to expose its actual pivots/rank/threshold,
compare original graph precision and independent least squares, and preserve
complete records before choosing a repair. No shared runtime change or public
wiring is allowed before that attribution.02375 is a diagnostic-only vector
matmul shape error; repair it with the original transposed matvec. The ownership
fixture must use a two-output equality-testing callable, matching this API.

02376 finds native COD rank2 for the exact all-ones rank1 design. The second
pivot5.02e-15 exceeds the unchanged3.83e-15 threshold; original and independent
least squares use rank1. This is a real numerical rank error despite small
response residual. Inspect installed Eigen `Householder.h:114--120` and
`ColPivHouseholderQR.h:541--551`: the original first multiplies the essential
tail, then adds the leading row; the native update combines them in one dot.
Test that exact arithmetic separation in an isolated diagnostic copy before any
shared solver change. The other design has numerical rank2 in every arm;
its weak direction amplifies ordinary RHS/solve rounding. It remains a separate
strict comparison failure. No threshold or solver substitution is authorized.

02377 confirms that Eigen's separated tail-dot/leading-row update restores rank1
and the original minimum-norm solution in eager, graph and XLA. No threshold
changes.02378 preserves complete failing design records and passes16other checks,
including corrected cache/resource ownership. Install the localized shared COD
arithmetic correction as a candidate, add a direct repeated-column regression,
then run dynamic/active-row primal/pullback, dense/derivative and original factor
checks. All affected enclosing controllers and costs require renewed evidence;
earlier uniform costs remain historical measurements of96e15e9d. Keep the separate
ill-conditioned full-record failure open. Public posterior wiring remains off.

02379 passes34 native/dynamic/active-row primal and pullback checks;02380 repairs
the complete rank-one record at the unchanged tolerance.02381 passes20 dense and
posterior checks, but eight uniform comparisons use pre-public-wiring metadata
and fail exact key matching before numerical comparison. Update those assertions
to the already qualified public metadata comparator (all numerical fields kept),
then resume the same bounded matrix. This stale test is independent of the COD
change; preserve its failure. No execution metadata is simply dropped unchecked.

02382--02388 pass159 checks after metadata repair, including original factor
equivalence and72policy/controller cases.02389's independent100/160-digit SVD
agrees across precision settings. The rejected weak-direction matrix shows
original self-sensitivity: one-ULP input changes fail2--5precision entries at
1e-10, with differences up to3.29e-9. Current graph/XLA same-input differences
are2.91e-9/3.05e-10; relative response residuals remain about3e-16. This explains
the separate failure but does not waive it. Keep the original comparison as a
mandatory failing test. Continue independent complete-consumer renewal and native
costs while it remains open. Native costs use the previously specified D3/D5
original/graph/XLA boundaries,21calls per process and complete changed-input
records. Public wiring/costs still require its disposition and GPU qualification.

Recovery continuation: 02390--02402 completes the shared-COD CPU consumer
renewal with 276 passed and no failures, errors or skips. This includes original
D1/D3/D5 posterior records, partial diagnostics and input reuse, uniform public
records, quadratic/paired/batch consumers and original D3/D5 sequential lifecycle
records. Charged through 02402: CPU 50212.35936420217 and GPU
46001.71196921611 seconds. The predeclared caps remain unchanged.

Recovery review: the shared solver change requires current-source GPU and cost
evidence; the older checkpoint cannot close those checks. GPU 3 became idle,
so renew uniform public GPU tests first, then the native posterior GPU matrix
and its original streams. Preserve the unresolved strict ill-conditioned test
and its pending comparison decision. The CPU native cost matrix can follow
GPU qualification or run if contention returns; all numerical jobs remain
sequential. No additional scientific or numerical change is introduced.

02403--02406 pass all 132 public uniform/quadratic/paired/batch GPU checks.
02407 passes all seven original GPU seed/cloud/proposal and all-pairs checks.
02408 preserves 17 XLA fixture failures: TensorFlow pins int32 diagnostic
counters to CPU, while the controller and position recorder execute on GPU.
This is a test-resource placement failure before XLA numerical comparison.
Use int64 diagnostic counters explicitly on the recorder device, assert
colocation, run one Gaussian GPU smoke, then renew the same complete matrix.
Target values, scores, streams, thresholds and runtime source are unchanged.
The earlier COD diagnostic now loads 96e15e9d through FrozenCheckpoint and
requires each arithmetic replacement exactly once; old/current/trial provenance
can no longer silently collapse after the runtime repair.

02410--02414 pass seven GPU stream/consensus checks, 51 complete D1/D3/D5
records and the changing-input/HLO check after resource colocation.02415 passes
16 extra checks, but preserves two strict rejected-design failures. The existing
ill-conditioned precision discrepancy remains. A zero design newly reports
selection relative RMSE 1 in XLA versus 0 in original/graph, although every
precision is exactly zero and all arms reject. Preserve its complete records;
localize the center-vector versus batch-matrix arithmetic before changing source.
No diagnostic normalization or threshold change is authorized for this case.

Review also found that finite input scores can overflow on subtraction before
the least-squares fit. Add a bounded original-versus-native complete-record check
using a one-dimensional Laplace value with its analytical sign score, at ordinary
and extreme finite scales. Record graph/XLA exceptions as failures and retain
full original rejection details. This is a numerical-rejection regression, not
a changed target or scientific claim. Run this diagnostic before costs; all
predeclared budgets, vetoes and comparison criteria remain unchanged.

02416 passes both ordinary and extreme finite Laplace score CPU records;
no new runtime guard is required by those cases.02417 rejects the simple
vector-to-row syntax trial for the zero design.02418 captures unchanged complete
results and actual intermediates: raw center and batch scores are identical,
but XLA center projection differs by -5.551115123125783e-17 in one coordinate.
Graph projections are identical.02419 tests shared multiply/reduce and ordered
TensorFlow contraction trials; both restore every original zero-design field.

Install the shared multiply/reduce contraction as the posterior candidate.
It computes the unchanged sum score_i * factor_ij across the same axis for
center, training, selection and holdouts; no residual floor or special-case
zeroing is introduced. It preserves leading batch dimensions and has no Python
numerical loop. The explicit broadcast could cost memory in the graph reference
or at larger dimensions, so include its actual allocation and performance in
the planned native costs and later DZ5 scope. The sequential trial remains
diagnostic only. Freeze the old attribution factory to02419 so future runs
cannot call repaired arithmetic the original failure.

Before cost interpretation, renew the original CPU/GPU records, zero/rank-one
rejections, malformed callbacks, independent geometry, resource lifetime, stable
HLO and finite-score overflow checks. The ill-conditioned mandatory comparison
remains separate and unwaived. All other fields retain1e-10.

02421--02428 complete renewed projection qualification:78 GPU numerical
checks plus72 CPU policy/controller checks pass. This includes all51 complete
original dimensional records, streams, input/HLO reuse,17qualified extras and
both finite-score numerical-veto cases. The ill-conditioned comparison remains
mandatory and unwaived; it is not included in the passing subset. Native GPU
costs now execute six fresh processes atD3/D5 (original/graph/XLA),21calls each.
These first measurements explain allocations and compilation; public admission
and three-process terminal comparisons remain separate. Focused new-source
Ruff passes. Five unchanged pre-existing driver warnings were independently
confirmed against96e15e9d; no new driver lint warning was introduced.

02429--02434 pass all six native GPU cost processes with full initial/changed
original records. D3/D5 original warm medians are165.886/173.857ms, graph
47.220/55.001ms, XLA5.481/9.716ms. Total cold including factory and trace is
7.208/7.624s for XLA versus1.428/1.418s original. Observed extra host RSS is
141.57/140.45MiB; XLA allocator peaks48384/56832bytes versus about8.4MB.
Only the2x cold trigger fires. These single-process measurements are descriptive,
not timing superiority evidence. Analysis:posterior-native-gpu-costs-02434.json;
its exact analyzer and six-arm historical regression are archived at the shared
root. Graph count2733 and HLO about1.98MB are unchanged acrossD3/D5.

Attribution: XLA trace is1.09/1.11s and first execution6.12/6.51s; most host
allocation occurs there. Warm-call RSS grows16/24KiB over the measured19-call
interval, which is small but too short to establish a plateau. Add boundedD5
capacity checks at1,2and4replicates,3000alternating-input full calculations each,
one process per capacity/device,300-second timeout. Record before/build/trace/
first-call and sparse1000-call RSS/smaps/allocator observations, complete original
records, one trace, stable HLO and all three runtime operands. Preserve progress
if the parent deadline fires. This tests observed warm growth and graph capacity;
it cannot prove arbitrary target turnover, native executable eviction or general
leak freedom. No cleanup intervention or threshold change is allowed.

02435--02442 pass all150renewed CPU checks, including the same78numerical
cases and72policy/controller cases. Native CPU costs are now active; numerical
source is unchanged from the GPU cost processes.

02443--02448 pass six matched native CPU cost processes. OriginalD3/D5
warm medians99.073/96.710ms versus graph5.279/5.640ms andXLA1.080/1.617ms.
XLA total cold3.182/3.133s versus0.149/0.139s original; extra observed RSS
377.62/381.39MiB triggers the planned host-memory investigation. Combined
analysis:posterior-native-cpu-gpu-costs-02448.json. Runtime source and matched
inputs are identical across these12CPU/GPU processes. The bounded capacity
and3000-call CPU follow-up is now executing before interpreting the host cost.

02449 preserves a harness specification error before numerical execution:
the public configuration requires at least two replicates, so the proposed
one-replicate capacity is invalid. Correct the capacity diagnostic to2/4/8
replicates, keeping rows, batch, target, criteria and300-second ceiling fixed.
Each remains within the unchanged physical-row budget (largest planned728).
Do not weaken the minimum-replicate validation or treat this harness failure
as evidence about allocation. No runtime source changes.

02450--02455 pass all six3000-call capacity checks, each preserving complete
original records, one trace and stable HLO. Graph counts2733/2737/2737 and
near-flat final1000-call RSS growth4--12KiB (CPU) and4--8KiB (GPU) localize
the observed host increase to compilation/first execution. GPU current remains
5376bytes; per-scope allocator peaks stop growing. See the new
[checkpoint and review](filter_gradient_posterior_checkpoint_20260922.md).
Shared-COD GPU follow-up preflight then declined six busy samples before any
worker launch; preserve cod-tail-gpu-preflight-pause-02455.json and continue CPU.
02456 renews the strict ill-conditioned failure on current source: rejected
precision[0][1] differs by2.91209576e-9, with every record preserved. The
comparison proposal remains uninstalled. The native module docstring now
accurately states that completed reporting is separate; no numerical code changed.
