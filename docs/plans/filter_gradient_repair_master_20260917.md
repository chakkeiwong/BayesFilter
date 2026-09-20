# Complete filter and gradient execution repair

Status: executing on `repair/filter-gradient-xla-validation-20260918`; merge is gated.
Owner request: repair all findings in the September 17 audit, compare memory
and performance before/after, review this program, and merge only when tested.

September 20 refresh through run 01257: the complete block-score lifecycle
now executes numerical replicate/block/pair loops, qualification and coordinate
scaling in XLA. All 41 CPU/GPU checks and 121 affected GPU consumer checks pass.
Two-extent public and graph/XLA comparisons preserve outputs within 1.101e-13,
keep constant graph size and stable allocations, and trigger no new investigation.
All 61 policy/controller cases pass. The shared eigensystem comparison now uses
the already qualified binary64 residual refinement. Checkpoint `31a81b10`
contains the prior mass-construction repair; its precision graph/XLA timing
investigation remains open.

The owner has authorized **another 48 GPU / 24 CPU process-hours**, counted
once. Active cumulative caps are **52 GPU / 32 CPU process-hours**. Through
01257, charges are 18,323.631 GPU / 32,430.983 CPU seconds; remaining allowances
are 168,876.369 GPU / 82,769.017 CPU seconds. The earlier 16 GPU / 12 CPU
proposal below is superseded, not an additional allocation.

The repair is incomplete. The static guard covers 183 sources with 1,217 exact
exceptions; it explicitly does not cover the whole repository. All F01--F20
terminal decisions remain open. Focused passes are checkpoint evidence;
terminal tests and comparisons must match the final source and harness.

Recovery review through 01199 confirms no active worker, the unchanged frozen
baseline, and the same cumulative budget (the extension is counted once).
All six pending TT/preparation GPU groups pass 134 checks (01131--01136).
The quadratic initializer migration passes 33 CPU checks (01139). Its GPU
suite exposed the joint-locator int32 resource-placement failure; int64
accounting repairs that reproducer (01141), and all 26 joint-center CPU
checks pass (01142). Full GPU initializer/joint-center reruns now pass all
33/26 cases (01159/01160). All 61 policy/controller checks pass in 01161.
Keep the 256 MiB host-memory investigation trigger and
all numerical gates intact. Continue the remaining block-center and sequential
preparation repair; their numerical loops cannot receive host exemptions.
The TP continuation repair preserves all reported XLA projection fields
exactly (01163). Runs 01173/01174 now pass all 22 GPU/CPU cases, with unchanged
raw-residual and derivative tolerances. The original 1e-5 finite-difference
stencil failed equally in both GPU source arms (01168); a predeclared step
ladder identifies cancellation, and two converged fourth-order estimates now
provide the derivative diagnostic (see the detailed harness review below).
The ineffective first-factor
Cholesky trial in 01146 is reverted. GPU2/GPU3 became available, permitting
the focused GPU qualification below; every new run must recheck contention.
Block-center preparation and complete callbacks now pass all 43 CPU checks
(01148), including pinned public/private records and strict boundary decisions.
All 61 policy/controller checks pass in 01149. The enclosing ordered sweep and
sequential locator remain open; the guard excludes that numerical loop rather
than granting a host exception. With the original sequential dependency also
pinned, block-center checks pass on CPU/GPU (01155/01157). Native scalar-cloud,
orthogonal-frame and trust-region kernels pass 20 focused CPU/GPU checks
(01150/01156), and all 40 sequential consumer cases pass on CPU/GPU
(01152/01158). These five numerical groups now require GPU in the terminal
driver gate. Their focused runs do not freeze the broader source/harness.

Current execution queue, under the same runner, evidence contract and caps:

1. Finish sequential search/fit/selection and the enclosing block/quadratic
   numerical control. Include reachable exact-incumbent selection; host record
   assembly must not hide row-wise numerical eligibility checks. Keep external
   DZ5 callback compatibility explicit and qualify actual owned consumers.
   Include fixed-center replicate/family/stability/shrinkage selection. The complete
   block-score lifecycle and mass construction are now compiled and focused
   consumer-qualified; their final source-frozen evidence remains pending.
2. Retain the qualified TP continuation repair and converged derivative check;
   run terminal current-source TP tests and paired measurements after freezing.
3. Investigate remaining TT assembly and forecast-pool host-memory increases,
   core-affine/higher-rank warm-time regressions, and centered qualification.
4. Finish coverage/dispositions, then freeze source and harness for affected
   suites and required paired three-process comparisons. Review exact outputs,
   HLO, allocation stability, host/device peaks and warmed timings.
5. Integrate remote changes and retest; merge only after every terminal gate.

The following numbered entries preserve the earlier implementation sequence.
Run-local pending notes are superseded by the current queue and results above:

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

### Predator-prey residual localization after 01143

While GPU correctness qualification is held by contention, isolate the first
TP projection using the unchanged four-date/two-step fixture from 00645.
Compare the pinned and candidate teacher, continuation/feature matrix, scaled
chart, fitted weights and raw residual under identical graph/XLA contexts.
Use frozen realized teacher inputs as a second view to distinguish upstream
rounding from projection arithmetic. Preserve the original raw residual and
1e-10 gate; report intermediate differences without promoting local parity to
full-recursion correctness. No canonical LEDH claim, score substitution,
source/math change or tolerance relaxation. Use one bounded CPU diagnostic
group through the existing runner, then repair only the localized cause.

Runs 01144/01145 localize the mismatch: identical teacher particles/log weights
but 1.058e-16 continuation-feature differences produce 3.77e-10 to 8.25e-10
raw residual differences; supplying identical features makes every reported
projection field exactly equal in graph and XLA mode. The first-factor Cholesky
trial in 01146 leaves these errors unchanged (four passes, two failures); it is
reverted. That hypothesis did not explain the discrepancy. Preserve this
negative result and the unchanged raw-residual gate. Further localization must
distinguish continuation update arithmetic from feature normalization before
another runtime repair; local frozen-feature parity is not full-recursion parity.
Next expose the continuation and reference log likelihoods and their normalized
exponential on identical frozen teacher particles. Compare graph/XLA separately
at the same two-step continuation. These intermediate outputs may change XLA
fusion, so treat their errors as localization only; the original first-projection
and complete-recursion gates remain the acceptance authorities. No runtime
change is bundled into this diagnostic.

Run 01162 finds exact reference/max normalization but a 2.842e-14 XLA
continuation-log difference, producing the same 1.058e-16 feature difference
as the first-projection failure. Test a bounded execution change: peel one
initial recurrent Gaussian update before the native loop. The original first
update was already separate because its covariance starts at zero. This adds
one fixed body, independent of horizon, and preserves chronological arithmetic;
it lets the two-step case retain the original straight-line compiler context.
Dynamic shorter windows select their original state with tensor masks, avoiding
the previously unsupported conditional-gradient path. Re-run the unchanged
first-projection and full-recursion gates and graph-growth check; a local pass
alone is insufficient. Preserve longer-window and total-gradient semantics.
Revert an ineffective trial, and retain every failed diagnostic.
Run 01163 passes all eight breakdown checks, with exactly equal XLA
continuations and first-projection fields. Run 01164 passes ten full cases but
the two-step XLA gradient hits a zero-capacity loop-tape compiler error. Omit
the statically empty remainder loop when the fixed initial updates already
cover the window; it has no numerical iterations to execute. Requalify the
exact full-recursion failure before extending the window-length checks.
Run 01165 now passes that exact complete value/gradient/history case, including
the original raw-residual gate, finite differences and validity. Next exercise
dynamic zero-through-four-step windows against the original sliced recurrence,
including complete parameter gradients and one-trace compilation. Then qualify
the complete TP suite on GPU. The localized result closes neither wider
performance/memory work nor canonical LEDH admission.

Run 01166 passes both dynamic-window tests, including counts zero through four,
parameter gradients, HLO and one tracing signature. GPU run 01167 passes 20/21
cases; the complete four-date/two-step case has exact baseline/candidate values,
gradients and histories but fails the unchanged directional finite difference
by 3.22034652e-5 (relative 4.656e-7 versus the 1e-7 gate). Compare the original
positive/negative perturbed values and gradients in both source arms before
attributing this to recurrence compilation or conditioning. A baseline failure
would explain the comparator limitation, not waive the derivative gate.
Run 01168 confirms exactly equal perturbed values and central differences in
both GPU arms: the directional score is -69.1603179080489 and the original
central difference is -69.16028570458366. The same baseline gate fails. Record
a predeclared eight-step finite-difference ladder (1e-3, 3e-4, 1e-4, 3e-5,
1e-5, 3e-6, 1e-6, 3e-7) on both source arms to distinguish truncation from
roundoff amplification. It is explanatory only; neither a favorable step nor
baseline agreement waives the original test. Keep branch validity visible.

Run 01171 shows the same central differences in both arms at every step, with
all charts valid. Errors grow from roughly 1e-6 at 1e-3/3e-4 to 3.22e-5 at
1e-5 and 1.11e-3 at 3e-7. This identifies an unreliable subtraction stencil,
not evidence of a changed derivative. Repair this diagnostic with fixed central
stencils at 4e-3, 2e-3 and 1e-3 and two fourth-order Richardson estimates:
`(4 D(h/2) - D(h))/3`. Require the two estimates to agree and require **both**
to match the derivative at the unchanged atol=1e-8 / rtol=1e-7; preserve value,
gradient, raw-residual and branch-parity gates. This supersedes the single
1e-5 stencil as a pass criterion, retaining its values and failures as explicit
diagnostics. No candidate-dependent step selection or tolerance change.

Skeptical harness review: a single favorable step could conceal bias, so the
replacement checks convergence of two fixed estimates and validity of every
perturbed chart. Apply the same rule to all six existing CPU/GPU fixture cases,
not only the failing one. The formula cancels the central stencil's O(h^2)
term; lack of convergence remains a failure. Preserve the source baselines,
algorithms and numerical parameters. First rerun the exact failing GPU case,
then the full CPU/GPU suite before closing this focused blocker.

### Exact incumbent selection boundary

The reachable incumbent selector computes finite eligibility once per record
in Python. Move complete value/position/score eligibility and earliest-maximum
selection into one stable XLA program. Host iteration may pack record fields
and restore the chosen original object; it may not evaluate numerical gates.
Preserve first-record ties including signed zero, eligibility metadata,
nonfinite rejection, heterogeneous vector widths and empty vectors. Ragged
row structure is metadata, not permission to assume a common dimension.

Skeptical review: a padded representation must not make empty vectors invalid,
and masked nonfinite values must never defeat an eligible finite row. Compare
the pinned implementation and independent Python reference on those boundaries,
enclose the complete selector in HLO, check one trace across record counts, and
run actual joint/sequential/quadratic consumers. This is preparation only;
selection does not issue HMC tuning authority. Preserve existing source/RNG
contracts and use the same cumulative budget and runner. Outer numerical
lifecycles remain open after this bounded repair.

The first selector checks (01169/01170) pass, but review identifies a memory
risk: a single polymorphic TensorFlow trace can still create one XLA executable
per physical record count. Growing incumbent histories would compile too many
sizes. Pack records and coordinates into the next powers-of-two capacities,
mask unused records and make padded coordinates vacuously finite. This leaves
selection and returned object identity unchanged while reducing physical
signature counts to logarithmic growth. Test realized public input shapes
across capacity boundaries, in addition to one graph trace and all boundary
parity. Preserve zero-width rows and original first ties; do not call padding
a hard memory cap or constant-size compilation claim.

GPU measurements 01190--01193 preserve exact selector outputs but expose a
63.9x public warm-time regression: 0.610 ms before versus 38.968 ms after,
while the complete XLA numerical kernel takes 0.473 ms. The public record
builder slices and materializes scalar fields repeatedly. Bulk-unstack tensor
rows, serialize completed values/flags once, and avoid identity reshapes of
already-flat vectors. These operations only pack immutable records; numerical
eligibility remains inside XLA. Preserve object identity, buffer snapshots,
shape rejection and nonfinite behavior. Requalify the same public pair after
the focused selector tests. The 4.25/25.5 KiB device peaks exceed the existing
2x trigger; explain the realized allocation difference without waiving it.
Run 01194 passes all 15 GPU selector checks after bulk transport. Run 01195
preserves exact outputs and reduces the descriptive public median to 2.437 ms,
but still exceeds the 0.610 ms baseline; the device peak remains 25.5 KiB
versus 4.25 KiB. Keep both investigation triggers open. The complete score-fit
public pair (01186/01187) matches within 8.882e-16, with medians 19.008/2.302 ms
and no new trigger. Candidate graph/XLA kernels also pass parity (01188/01189),
with 916/922 graph nodes and no hidden XLA in the graph diagnostic. These are
single-process, small-extent observations; larger extents and final repeats
remain required, and cross-scope timing ratios are suppressed.

### Sequential symmetric score-fit compilation

The next preparation kernel encloses cloud generation, scalar/batched exact
evaluation, symmetric score design, unchanged training/holdout partition,
rank test, ridge-augmented complete orthogonal least squares, eigenvalue
projection, and fit diagnostics. Preserve the original coefficient ordering,
rank threshold, Philox stream, support rejection, status precedence and exact
cloud winner. Reuse the qualified native COD and binary64 XLA eigensolver;
do not replace least squares with normal equations or change the ridge.

Skeptical review: rank-deficient branches must retain their rejection and best
exact row without using a projected fallback; pairing and sample order are
part of the frozen method. Compare complete records against the pinned helper
on quadratic/nonquadratic targets, correlated/indefinite curvature, both holdout
partitions, insufficient support and nonfinite candidates. First use identical
frozen clouds to isolate the fit, then qualify original seeded end-to-end cloud
evaluation, callback count, HLO, one-trace reuse and actual sequential consumers
on CPU/GPU. Keep the original tolerance and use the same driver/budget. This
bounded kernel does not close the outer search/refinement lifecycle.

Run 01176 passes 13/14 focused cases; an external tape exposes the raw XLA
eigensolver's missing pullback before the wrapper can detach the fit result.
The original result crossed NumPy/host-materialization boundaries, so preserve
that frozen preparation contract inside the compiled return, not afterwards.
Also reduce the tall design to its thin-QR triangular factor before XlaSvd,
which computes full factors even when only singular values are consumed.
This retains singular-value rank testing and the original row-scaled threshold
without forming a large square observation-space factor or normal equations.
Recheck complete frozen/seeded records and the external-tape boundary.

Register two preparation comparisons: exact incumbent selection at 32/64
three-dimensional records, and complete symmetric score fitting at 16/32
cloud rows and two/four dimensions, with the original seed (2026, 919).
Measure matching complete public endpoints in host-call mode and numerical
graph/XLA programs separately. Public candidate calls retain internal XLA;
graph diagnostics explicitly disable it, including nested helpers. Record all
returned numerical fields and status/winner identity. Baseline tracing failures
remain evidence, with its valid public result as the parity authority; forbid
timing ratios between unequal scopes. Use the existing GPU, twenty warm calls,
two extents, three terminal repeats and memory/performance investigation gates.

Recovery call-chain audit also confirms `sequential_map_covariance` and
`quadratic_map_covariance` call `mass_matrix.covariance_from_precision`, whose
eigendecomposition, floor selection and inversion currently execute eagerly.
`structured_covariance_from_empirical` also loops over blocks numerically.
`posterior_local_initializer` reaches `fit_fixed_center_curvature`, whose
replicate, family, pair-stability and shrinkage loops remain host numerical
control. `block_score_geometry` has analogous replicate/block/stability loops.
The static guard excludes iteration for the latter modules and does not cover
mass_matrix; these are open F18/F19 dependencies, not source exemptions.
Carry their original decisions, reports, thresholds and HMC authority boundary
through the migration; removal of NumPy alone did not close XLA execution.

### Complete block-score lifecycle after 01238

Question: can the existing block score regression, replicate comparison,
consensus, selection/audit checks and coordinate scaling execute in a complete
GPU/XLA program with the same public records and first-failure behavior?
The comparator is commit `3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf`, including
its original `fixed_center_curvature.compare_precision_geometry` dependency;
loading that wrapper with the repaired dependency would be a contaminated
baseline. The current checkpoint is an additional localization comparator.
The owned public function has no discovered runtime caller beyond its export;
passing it does not establish an external consumer or HMC qualification.

Preserve symmetric least squares with the existing ridge, rank thresholds,
raw-SPD/condition rejection, declared block order, replicate order, pair order,
principal-angle snapping, all configuration thresholds and final status
precedence. No random stream or model changes. Native replicate/block/pair
loops carry fixed-size numerical records; host loops only reconstruct reports
and static block schemas. Specialize branches by distinct block widths, not
by replicate or block count. The graph diagnostic must have no nested XLA.
Preserve the dimension-one public rank exception when all fits succeed, and
earlier fit rejection when they do not. Preserve rejected report truncation.
These initializers do not compute a claim-bearing target gradient.

Skeptical review: a compiled block fit alone would miss the remaining Python
selection and stability arithmetic; compare complete endpoints. Verify rank,
nonfinite inputs, heterogeneous blocks, first failures after usable replicates,
selection versus stability versus audit precedence, strict caps, coordinate
scale rejection, independent exact-block identities, graph growth and HLO.
Pin both baseline modules. Use the unchanged FP64 `atol=rtol=1e-10` gate and
exact discrete decisions. Degenerate eigenspaces can rotate without changing
matrices; use separated spectra for orientation-sensitive comparison and keep
degenerate-spectrum outcomes explicit rather than relaxing tolerances.

Run focused CPU then GPU checks through the existing driver. Register matched
public and complete numerical fixtures at two block/replicate extents; retain
the baseline's actual tracing failures. Compare graph/XLA and public timings,
host/device peaks and at least 20 warm allocations under the existing 20%,
2x and 256 MiB investigation triggers. Tests reserve at most 900 seconds and
each measurement 300 seconds within the unchanged cumulative caps. Preserve
all results in the numbered campaign artifacts. No source/harness edits while
a numerical worker runs. A failed parity or threshold check triggers repair;
budget exhaustion, contention or invalid evidence stops the affected launch.
Passing focused tests does not close F01--F20, terminal repeats, external
callbacks, memory investigations, canonical LEDH work or merge readiness.

Runs 01239--01241 localize two issues without changing gates. The empty training
design has rank zero but XLA's zero-extent SVD cannot lower; a static zero-row
case now returns the original rank rejection. The new rotating-subspace fixture
shows a 2.6535e-6 degree angle discrepancy. Diagnostic 01241 finds default XLA
eigensystem residual 4.4172e-7 versus eager 4.4409e-15; the already qualified
mass eigensystem/refinement gives residual 5.9825e-16 and restores the angle
within 3.2e-14 degrees. Reuse that same-eigenproblem helper in the shared
precision comparison, including generalized eigenvalues. Graph-only diagnostics
explicitly use ordinary TensorFlow eigensystems. Qualify the existing
fixed-center consumers as well as block-score checks after this dependency edit.

### Complete mass construction dependency repair after 01199

Checkpoint `9a658d9f` is committed and pushed. Recovery verified the runner's
1,199 records and cumulative caps; the additional allocation is counted once.
Next compile mass construction reached by both initializers. Preserve symmetric
projection, jitter, eigenvalue floors, dense/diagonal inversion, structural
shrinkage, block ordering and every report/validation field. Block partitions
are fixed schemas; numerical repetition must use one native loop with branches
only for distinct block widths. Summary statistics and floor decisions execute
inside XLA. Host work validates configuration and serializes completed results.

Skeptical review: the existing floor is materialized and frozen before matrix
reconstruction, so migrating it must not introduce a derivative through floor
selection. Preserve derivatives through the matrices and the existing
TensorFlow eigensystem pullback; a raw XLA eigensolver without a derivative is
insufficient. Inspect TensorFlow 2.19.1's `SelfAdjointEigV2` pullback, retain its
gap treatment, and qualify first-order derivatives away from repeated spectra
against the pinned source and independent identities. Use the existing binary64
Jacobi precision, not TensorFlow's loose default XLA eigensolver tolerance.
Nonfinite/empty/no-positive-eigenvalue cases, conditioning clamps, signed
threshold equalities, asymmetric matrices and heterogeneous blocks must retain
their status and ordered metadata. Do not adjust floors or permit reflection.

The interface reference and capability registry were re-read. These are mass
preparation helpers, not tuners; their repair issues no HMC authority and runs
no posterior chains. Compare public records with baseline `3582b4ac`, verify
matrix/whitening identities, VJPs, complete HLO, no callbacks, one-trace reuse
and bounded block-graph growth. Then qualify existing mass and affected
initializer/locator consumers on CPU/GPU. Register matched public and tensor
measurements at two dimensions/block counts, preserve true baseline tracing
failures, and apply the existing 20-call/three-process terminal comparison
contract. No runtime/harness editing during workers; same driver, hardware,
900-second focused ceiling and cumulative budget. The known selector, TT and
forecast-pool regressions remain separate open investigations.

Run 01200 passes 55/57 CPU cases. The regularization VJP fixture's primal
differs by 2.0045e-10 in two entries at the unchanged 1e-10 gate; dense/diagonal
primal and pullback checks pass. Localize the eigensystem residual at the fixed
binary64 epsilon and its square before changing the eigensolver control. Both
use the same 100-iteration bound and matrix; compare residual and projected
matrix with the pinned eager source. This tests internal solver accuracy, not
a change to mass floors or acceptance thresholds. The other failure is the
shared pullback helper's Boolean output handling; encode the internal validity
flag as a numerical scalar, preserving the public rejection and all outputs.

Run 01201 localizes a 2.0421e-10 eigensystem residual, unchanged when the
Jacobi epsilon is squared. A smaller requested tolerance therefore does not
repair this backend result. Add bounded Jacobi refinement of the same symmetric
eigenproblem in the returned eigenbasis, using explicit off-diagonal residuals
and stable two-coordinate rotations. Stop at binary64 relative roundoff with
at most 100*n*n rotations; return sorted eigenpairs and retain the inspected
TensorFlow pullback. This improves the implementation of the existing spectral
operation, without changing its floors or target. Test residual, orthogonality,
reconstruction and derivatives for distinct/repeated/clustered, indefinite and
scaled spectra at dimensions 1, 2, 3, 6 and 12; preserve the failing raw-XLA
observation as explanatory output. The unchanged parity gate remains decisive.

### Active quadratic and block-center preparation migration

Consumer inspection confirms that the two public wrappers' outputs select real
initial centers/covariances. Their diagnostic names do not permit NumPy
numerical decisions. Migrate immutable arrays, coordinate transformations,
score summaries, centeredness, cycle/reversal checks and target callbacks to
TensorFlow with stable XLA signatures; restore the existing joint locator's
XLA default. Preserve coordinate scaling order, objective/score thresholds,
transaction ordering, failure statuses, row counts and callback records.
Use the pinned original wrappers for same-input checks of decisions/records,
independent quadratic identities, invalid inputs and existing consumer tests.
No new RNG change, fit algorithm, tuner, HMC chain or tolerance is authorized.

Skeptical call-chain review also finds numerical Python iteration in the
downstream `sequential_map_covariance` search, scalar cloud and trust-region
paths, and host control across successive geometry fits. These remain F18/F19
repair work; removing caller NumPy alone cannot close complete execution.
Classify schema/reporting separately and do not allowlist numerical loops.
First qualify the bounded wrapper kernels and real callers, then migrate the
remaining numerical control with unchanged fit/selection semantics. The
existing per-run limits, current cumulative budget, final paired comparisons
and memory triggers apply. Read the HMC interface and capability registry;
these helpers issue no tuning authority.

The next bounded step removes block-center NumPy and compiles its complete
target callbacks, embedding, cycle/reversal tests and score summaries. Compare
public/private records, exact row counts and ordered transaction decisions
against the pinned wrapper on identical inputs. Exercise strict threshold
edges, partial partitions, nonfinite callbacks, immutable snapshots and complete
callback HLO. A compiled callback must return a nonfinite rejection signal in
graph context; host replay retains its ValueError. The ordered sweep still
depends on the host-controlled sequential locator and remains explicitly open
until that dependency and the enclosing recurrence are migrated. No numerical
loop receives a reporting exemption, and no RNG or stopping criterion changes.

Next migrate the sequential helper's scalar-cloud iteration, orthogonal-frame
generation and trust-region bracket/bisection to stable native XLA programs.
Preserve the existing Philox words and Box-Muller ordering with the already
qualified compatibility primitive; no new random stream is permitted. Retain
the 80/80 trust-region limits, comparison signs, eigensolver precision and raw
predicted improvement. Use pinned eager helpers and independent SPD KKT checks,
multiple frame/count extents, scalar/batched score parity, HLO and one-trace
checks before existing sequential and block consumer suites. The outer search,
selection and fit lifecycle remains a distinct open repair. This is numerical
preparation, not a batch-native NeuTra training target or HMC tuner.

The external DZ5 wrapper also records some callback results with `.numpy()`
inside the callback. Such callbacks are not valid enclosing-XLA implementations;
record consumer compatibility explicitly rather than silently running an eager
fallback. The local repair tests must distinguish tensor-compatible callback
execution from host reporting. Whole-consumer qualification remains open when
an external adapter mixes these roles.

Run 01140 passes 32 initializer cases on GPU but exposes a real locator failure:
TensorFlow pins its int32 resource counters to CPU, so the GPU/XLA optimizer
cannot access them. Migrate only the accounting resources and matching cap to
int64 in both single/staged locators. This preserves bounded integer counts,
evaluation order and all numerical state. Qualify the actual locator first,
then the existing joint-center suite and affected initializer checks. This is
a portability repair, not a CPU fallback or optimizer change.

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
