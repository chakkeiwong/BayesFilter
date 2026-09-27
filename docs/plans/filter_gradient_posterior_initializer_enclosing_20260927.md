# Enclosing posterior-local initializer execution repair

The public single locator repairs one dependency of
`posterior_local_initializer.initialize_posterior_local_location_scale`.
The enclosing function still has numerical Python loops over movement attempts,
curvature attempts, partitions and candidate rows. The question is whether a
native controller can preserve this complete finite algorithm and its reports
without host numerical feedback. A passing locator or Gaussian smoke does not
answer that question.

The inspected source boundaries are the eligibility tracker at line 282,
candidate recorder inside the public function, movement recurrence at line 632,
curvature recurrence at line 758, row selection at line 840 and final physical
coordinate conversion at line 944. Recheck anchors after edits. Reuse the shared
joint-center, quadratic-geometry and fixed-center-curvature numerical programs;
do not copy or simplify them. Preserve the current versioned TensorFlow stream.
`GeometryTensorStream` hashes seed/call/role identities on the host and generates
clouds through CPU/XLA kernels. Seed identity preparation is static metadata;
cloud generation must remain compiled TensorFlow preparation, separate from
GPU target execution. Pre-generating bounded potential clouds must not consume
target rows or change target-call order.

Pin the complete repair-branch commit after single-locator qualification as
the immediate execution reference before implementation. Its full numerical
import closure, configurations, TF/TFP environment, RNG stream and input clouds
must be recorded. This reference incorporates already qualified numerical
repairs; it does not replace the campaign's oldest-original comparisons or
resolve the existing precision/reporting disagreements. Keep those gates open.

Execute three bounded steps in order:

1. Qualify native candidate-ledger and eligibility/accounting state, including
   exact earliest ties, nonfinite positions/values/scores, finite rejection
   sentinels, budget exhaustion and scalar/batch disagreement. Keep caller
   callbacks separate from completed reporting. Inspect existing selectors:
   `sequential_selection_tf.selection_numerics` does not test position
   finiteness and cannot silently replace the exact-incumbent rule.
2. Enclose the movement and curvature recurrences with fixed tensor state and
   TensorFlow loops. Use prepared clouds first to isolate execution arithmetic
   from random preparation. Preserve every movement row, candidate ledger row,
   partial curvature record, callback position/count, early exit and physical
   coordinate output. Reuse numerical authorities under one independent owner
   scope; serialize resource state and keep external derivatives frozen.
3. Qualify versioned CPU cloud preparation, integrate the public endpoint only
   after complete comparisons pass, and run actual caller assertions plus fresh
   original/graph/XLA cost arms at two dimensions. A source search must identify
   every internal consumer; tests must demonstrate the actual call chains.

Use healthy rotated Gaussian and nonlinear fixtures at D1/D3, movement and
curvature re-centering, insufficient successful fits, both attempt limits,
invalid initial/cloud rows, eligibility mismatch and evaluation-budget exits.
The primary criterion is complete same-mode reference records at existing
tolerances plus exact callback order and counts. Changed starts/scales/clouds
must reuse traces/HLO and replay exactly on the same backend. Keep graph/XLA
differences as separate witnesses. Prove actual XLA execution and no eager
fallback with a real compiler failure; test Python owner collection without
claiming native executable eviction.

Reserve 64 CPU and 44 GPU workers, 10,800 combined charged seconds, inside the
existing 56 CPU / 52 GPU-hour caps. Each focused worker normally has 300 seconds;
a documented compiler-capacity failure may have one 900-second retry. Use the
existing `scripts/run_filter_repair_campaign.py` command prefix, registered
groups and fresh `run-NNNNN` directories in the current artifact root. CPU
correctness precedes GPU qualification; sources remain frozen during matrices.
Apply the existing compute-GPU selector and verified growth policy. This is a
reservation within the campaign, not another CPU-hour extension.

Cold above 2x, warm above 1.2x, host RSS above 256 MiB extra or 2x, and GPU
allocator peak above 2x trigger attribution. Numerical failures veto performance
ranking. Source/environment/input drift invalidates a cohort; exhausting the
unit/global budget stops launches. A mismatch triggers localization against the
first differing callback or raw field, never a tolerance or rank-cutoff waiver.
Compiler errors must retain the defined caller error boundary without turning
partial counters into completed results.

Default audit and skeptical review: existing optimizer settings, rank choices,
clipping, radii and stopping thresholds are frozen for execution parity only;
they are not newly calibrated defaults. Native reductions can perturb strict
decisions, pre-generation can inadvertently change seed order, and moving
eligibility checks can change budget consumption. Complete records and ordered
callbacks are the earliest checks for those errors. A cheap Gaussian reference
and no-movement/initial-invalid controls can expose a wrong controller before
expensive qualification. Static seed metadata is not permission for Python
loops over numerical sampling or target evaluation. Host result formatting
must not retain numerical logarithms, norms or decisions. A bounded Python
owner alone cannot prove native memory bounds. No HMC, posterior coverage,
canonical LEDH, main-promotion or whole-repository claim follows from this unit.
Independent review has not yet been performed; the source audit above is the
current local pre-execution review, to be renewed at each material boundary.

First executable substep: the candidate ledger will call the existing
`_exact_incumbent._incumbent_selection` numerical body on the current incumbent
and each new candidate inside one TensorFlow loop. This preserves position,
score and value finiteness and earliest ties through the shared authority.
Return promotion flags, selected prefix indices, scaled-score norms and the
final incumbent as completed tensors. A fixed capacity bounds the loop and
ledger; a dynamic active count is an operand. Invalid active counts return an
explicit invalid-input status without selecting a candidate. Zero capacity is
an explicit configuration case. Derivatives remain frozen. This is an internal
dependency until the enclosing caller is qualified, not a second public route.

Compare all prefixes against the pinned exact-incumbent API and an independent
standard-library finite/maximum reference at D1/D3, including ties, signed
zeros, finite rejection sentinels with explicit ineligibility, nonfinite data,
empty/partial/full prefixes and changed inputs. Check one trace, stable HLO,
graph labels, no host callback and frozen gradients. Reserve six CPU / four GPU
workers and 1,800 seconds from this unit, not in addition to it. Register
`posterior_ledger_cpu` and `posterior_ledger_gpu` with the existing runner. Run
policy after the new source is guarded; add no numerical-loop or NumPy allowance.

The static consumer search across `bayesfilter`, `scripts`, `experiments` and
`docs/benchmarks` found no direct internal caller of the posterior initializer
beyond its definition/export. This does not classify dynamic or external users;
public integration must retain the existing exported-API consumer assertions.

The immediate pinned reference is `031692a0b` (public single-locator correctness
qualified; its uncontended GPU cost renewal remains open). The first dependency
uses the exact selector's numerical body in each loop step and changes no
eligibility or tie rule. CPU qualification proceeds while compute GPUs remain
shared. The current static scope has no direct internal initializer consumer
beyond its export; no external target readiness follows from this substep.

The candidate-ledger dependency passes ten CPU and ten GPU cases in
04327/04328. Tracker fixture 04329 fails before numerical execution because
loading the entire frozen initializer follows unrelated HMC imports into a
package unsupported by the strict reference loader. Preserve the failure and
failed harness snapshot. For this tracker-only question, execute the exact
frozen class excerpt instead: record full-file/class hashes and statically
verify every referenced global is TensorFlow or a named standard-library
builtin. This changes no tracker statement and imports no current project
numerics into the reference. The complete initializer later still requires
its complete frozen numerical import closure; this bounded extraction does not
qualify the whole initializer reference. Retry the same seven cases under the
existing first-substep reservation.

04330 reaches XLA but rejects the diagnostic callback recorder's dynamic
`Range(start,end)`: its endpoints depend on a resource counter. Repair the
recorder to create the fixed row-count range first and add the dynamic offset.
This records the identical positions in the identical order and changes no
tracker/target arithmetic. Preserve the failed harness snapshot and retry;
this localized harness failure remains inside the first-substep reservation.

Movement substep, after 04333: execute the exact frozen movement-loop and
candidate-recording statements from 031692a0b against prepared clouds. Verify
every Python file in that commit's BayesFilter tree against its current bytes
before using current imports as the reference dependency closure; fail on any
drift. New candidate modules are absent from the reference import path. Bind
only cloud preparation to supplied frozen arrays, preserving each seed and the
shared geometry program/result formatter. This isolates the recurrence while
retaining the complete existing geometry algorithm. Compare movement rows,
ledger rows, final status/incumbent, partial failures and ordered target calls.
Reference extraction and dependency hashes must be in artifacts; later complete
public integration still requires a full public-call comparison.

The internal movement owner accepts a finite exact incumbent and uses the
existing tracker resources supplied by its enclosing owner. It must not reset
already consumed target rows inside this stage. One native loop calls shared
geometry, replays its nomination, appends only a valid replay and uses the
shared exact candidate ledger. Read tracker mismatch before budget status,
preserving original precedence. Test healthy D1/D3 scalar/batch, nonlinear,
invalid cloud, budget and status mismatch before wider integration. Reserve
12 CPU / 10 GPU workers and 3,600 seconds from the enclosing unit. A numerical
discrepancy blocks public wiring and triggers first-field attribution.

04334 rejects the internal movement candidate during graph construction: the
history writer tries to scatter an empty D1/rank-zero basis tensor. The exact
reference completed. Preserve the failed candidate snapshot; skip the write
only for statically empty schema fields, whose history has no elements.
This does not alter any computed value or original comparison. Retry the D1
case before expanding the movement matrix.

D1 scalar 04335 and D3 batch 04336 pass complete movement rows, candidate
ledgers, tracker/physical callbacks and replay/HLO checks. Before broadening,
emit the candidate ledger's already-computed score norms in the movement output
so future public formatting needs no host numerical norm. Add explicit frozen
derivative assertions, then renew these two cases along with the four adverse
cases under one source freeze. The prior passing runtime/harness snapshots are
preserved. This is completion of the prepared numerical record boundary, not
an algorithm or tolerance change.

After the six-case CPU cohort, add a stationary Gaussian start to exercise the
early stop at exactly two successful movement fits; the existing D3 healthy
arithmetic fixture reaches the original movement-attempt limit and is not an
accepted initializer. Also exercise a real unsupported XLA target operation:
it must raise without eager execution and preserve tracker state for an owner
reset. Include these within the existing 12/10-worker movement reservation.
Keep the numerical runtime fixed if only test coverage or import formatting
changes, and archive the predecessor harness for the earlier cohort.

Curvature integration audit: `make_dense_initializer_cloud_program` already
preserves ordered partition evaluation, but its local winner summary lacks the
complete exact-candidate ledger and its validity contract differs from the
posterior evaluator. Reusing that helper requires explicit adapters and full
position eligibility checks through the shared exact authority. The complete
posterior path also records invalid-partition exits before any row-ledger
append and replays centers between curvature attempts. Preserve those boundaries
in the next substep; no dense-initializer shortcut may silently replace them.

Curvature substep after movement qualification: reuse the ordered cloud and
validated-fit programs, but ignore the cloud helper's local winner summary.
Append the replay and every row from each completed valid partition to the
shared exact ledger. If a partition fails, retain preceding ledger rows and
the original replayed center in the returned failure; no later partition may
execute. Select the complete exact incumbent before deciding whether to
re-center, preserving strict ties and finite-position eligibility. Fit only
after a complete cloud leaves the center unchanged. Preserve full fit records,
validation/fit exceptions, physical precision and covariance, marginal square
roots and logarithms as completed tensor outputs. Host formatting may decode
these decisions but must not perform numerical conversion or reselection.

Compare against exact 031692a0b recorder/curvature-loop statements, using the
same byte-verified numerical dependency closure as movement. Only cloud delivery
and completed result capture are adapted. Test stationary and moving D1/D3
targets, nonlinear or rejected fits, center-invalid, partial-cloud-invalid,
budget and eligibility failures, copied partition rows, and unchanged-input
replay. Compare all records and physical call order; stable HLO, trace reuse,
frozen derivatives and compiler-error behavior remain required. Reserve
16 CPU / 10 GPU workers and 4,800 seconds within the enclosing reservation;
use registered groups through the existing runner with 300-second ceilings.
Begin with a single stationary CPU case before expanding to the matrix.

Skeptical pre-execution review: evaluating all valid partitions before building
the ledger is equivalent here only because the original center is fixed for
that entire cloud and candidate recording invokes no target. An invalid
partition must not contribute any row even if individual earlier rows are
finite. The inherited fitter and exact-incumbent rule remain the numerical
authorities; no dense-route marginal formula may replace the original physical
covariance formula. Exceptions remain exceptions and are not renamed numerical
rejections. Successful prepared-cloud checks qualify this internal recurrence
only; public RNG, complete endpoint integration and costs remain separate.

Movement and prepared curvature qualification are complete through 04379; see
the September 27 movement and curvature-controller result notes. Curvature uses
factor_max=1 mechanics fixtures; the full public matrix must include the existing
default factor_max=2. No numerical method or tolerance changed.

Next qualify one compiled CPU preparation program for all potential movement
and curvature clouds. Reuse `geometry_random_tf._draw_kernel` and `_ball_kernel`
with the existing `geometry_tf_philox_cpu_xla_v1` stream. Host seed/call/role
hashing is configuration metadata only; every normal/ball draw and loop over
cloud generation must execute in the native program. Preserve the rank-zero
omission of the normal draw, subsequent permutation key, per-attempt movement
seed increments and curvature attempt/partition seeds exactly. CPU placement is
mandatory even when the enclosing target runs on GPU. Compare every generated
element and permutation key with the existing preparation helpers at D1/D3,
different seeds/radii and replay; require one trace and unchanged HLO as keys
change. Reserve four CPU and two GPU workers and 1,200 combined seconds inside
the enclosing reservation. One visible-GPU process may test CPU placement; it
is not a GPU random-stream substitution.

Pre-execution review: the dense initializer's separate RNG module uses a
different stream and cannot serve as this generator. Reusing the original
normal/ball authorities avoids that silent substitution. Pre-generating unused
potential clouds is permitted only because they consume no target evaluations
and their keys are derived independently from static configuration. Exact draw
comparison and the absence of any target-call interface are the early checks. Public
integration and original complete-endpoint comparisons remain required after
this preparation dependency passes.

The preparation module has three exact fixed-schema allowances, all confined
to `posterior_seed_keys`: one loop over configured movement seed roles and two
nested loops over configured curvature attempt/partition identities. They only
perform versioned seed hashing and metadata packing. No allowance covers a
random draw, sample row, target call, numerical decision or numerical recurrence.
The actual generator is fully guarded. This is the static seed-metadata boundary
already declared above, not permission for Python cloud-generation loops.

04380 catches a preparation implementation error before integration: movement
draws match, while curvature draws differ. The source `_sample_ball` explicitly
uses minimum_uniform=0.25 (posterior_local_initializer.py:1122); the new generator
incorrectly supplied the movement value 0.0 to both. Preserve the failed runtime
and test. Restore the curvature-only 0.25 argument and rerun exact equality;
change no RNG identity, seed, radius or comparison criterion. This is a localized
implementation repair inside the four-CPU/two-GPU preparation reservation.

Public-boundary source audit before wiring: `PosteriorLocalInitializerResult`
passes tensors through `numeric_tensor`; it does not explicitly freeze all
derivatives. `_build_result` performs physical scale/log operations outside the
native fitter, and `payload()` calls an eager `_eigen_summary`. Consequently the
earlier requirement to keep external derivatives frozen is not yet established
as parity with the current complete public endpoint. Internal stage checks do
not answer that question. Before replacing this endpoint, measure the actual
reference public derivatives with respect to supplied start/scale on an accepted
stationary Gaussian and an initial-invalid target. Record each returned field,
target counts, acceptance and derivative presence; use ordinary GradientTape
gradients, never pfor/Jacobian. Compare repeat payloads, including eigen summaries.
Also inspect the original 3582b4ac source to distinguish an intended frozen
boundary from partial derivatives introduced by earlier NumPy removal.

Reserve three CPU/two GPU workers and 900 combined seconds inside the enclosing
unit for this bounded audit; begin on CPU. This is explanatory reference evidence,
not new differentiability, numerical or performance admission. A partial
derivative must be reported as such. Preserve all existing criteria while
resolving the boundary; do not silently erase or advertise a new public gradient
contract. Native result reporting must include the eigen-summary computations,
not merely the already tested physical covariance and logarithm fields.

04384 exposes the reference boundary: an accepted-path call inside GradientTape
fails during geometry construction because XlaSelfAdjointEig has no gradient;
an initial-invalid return exposes identity gradients for the supplied center
and scale. The original 3582b4ac source converts inputs/results through NumPy,
disconnecting those derivatives. This is not evidence of a valid differentiable
initializer. Preserve the failed attempt. Extend the diagnostic to capture this
specific error, complete ordinary calls outside the tape, and compare a
stop-gradient input boundary against ordinary full payloads and physical target
counts. The prospective boundary restores the oldest-original disconnection
without changing values; it must not be advertised as a new analytical score.
Use the existing audit budget; no runtime implementation is changed by this
diagnostic.

04385 shows that stopping the supplied input gradients alone does not fix the
reference's construction error under an active tape; ordinary calls outside
the tape do complete. Preserve this narrower failed hypothesis. Capture both
errors, then test a diagnostic `tf.init_scope` boundary around the reference
call. That scopes construction away from an unrelated outer tape, preserving
the oldest-original nondifferentiable role. It is not permission to put eager
numerical work into the candidate: the new candidate must construct its owner
outside the tape and execute its numerical work in the enclosing XLA function.
This is the third CPU audit worker under the same reservation; success still
requires unchanged complete payloads and callback counts, with no new score claim.

04386/04387 pass the audit on CPU/GPU. Accepted and invalid ordinary calls match
the isolated reference boundary exactly, with 48 and one physical target rows
respectively. Full fields, including eigen summaries, replay exactly. The
accepted public call fails under an outer tape both with ordinary inputs and
with stopped inputs; isolating construction/execution in the diagnostic scope
restores the original nondifferentiable behavior. The candidate must isolate
owner construction from the caller tape and execute one native program with
frozen outputs. This repairs the accidental tape exposure introduced by the
earlier NumPy migration, not a mathematically valid public derivative.

The enclosing worker reservation is now 64 CPU / 44 GPU, with the combined
10,800-second and global 56/52-hour limits unchanged. The original worker count
underestimated separately needed dependency and boundary checks. This gives
room for full endpoint/default-factor comparisons and matched costs without
raising authorized compute. All previous attempts remain charged.

Full integration implementation review: one retained owner must share a single
eligibility tracker across initial replay, the bounded-chart locator, movement
and curvature. Construct it inside tf.init_scope, with configuration-sized
resources for the chart origin/units, then reset and assign those resources in
the compiled call. Starts/scales/clouds remain operands. Stage ledgers may start
from the previous exact incumbent, but completed reporting must omit the staged
proxy row and retain all actual earlier records and global ledger indices.
Preserve the public initial-invalid precedence, locator-best replay, tracker
precedence and partial-stage records. Preserve the explicit host-clock diagnostic
as non-default; no automatic eager fallback is allowed. Native summaries must
cover result payload eigenspectra as well as norms and logarithms.

For complete original comparisons, execute the full pinned 031692a0b public
module under an isolated module name. Verify every other pinned BayesFilter
Python file against current bytes before sharing imports. The replaced public
module must come entirely from Git, never partially from the candidate. Recheck
that no numerical dependency imports the replaced initializer; package export
metadata alone is not a numerical dependency. Start with a stationary D1 case
using the default factor_max=2, then expand before changing the public dispatch.
Owner construction must execute zero target calls. Compare full records and
ordered callbacks before any cost experiment; a discrepancy stops integration
and triggers first-field attribution, not a new tolerance or optimizer setting.

Recovery after 04389: the first complete pinned-module D1/default-factor_max=2
pilot passed on CPU. It preserved full payloads, physical callback order/counts,
changed scales, replay, one trace, stable HLO and frozen derivatives. Its exact
runtime, test, runner and policy sources are preserved under
posterior-enclosing-pilot-04389-source. Remote main 06590cb5a was merged without
conflicts as 035e19fdd; post-merge policy 04388 passed all 141 checks.

Broaden the complete-public comparison under the same reservation to D1/D3,
scalar/batch, nonlinear and displaced starts, initial/partial invalid rows,
evaluation-budget exhaustion and explicit status mismatch. Vary both starts
and scales on the same owner. The full-public reference and complete-record
criteria remain unchanged. Start with D3 stationary and stop the cohort on the
first discrepancy. Skeptical review: the initial pilot checked only changed
scales and could miss captured starts or changed strict stopping decisions;
the expanded operands and adverse cases address that gap before public wiring.

04390 fails during D3 tracing: tf.init_scope invokes TensorFlow record.stop_recording,
which suppresses forwardprop as well as the outer tape. The factor fitter then
receives a missing JVP. This is an implementation construction failure, not a
numerical mismatch. Preserve the complete failure and source snapshot. First
test tracing without that blanket recording suppression: all owner resources
are configuration constants and inputs enter only the frozen compiled program.
Require the same active-tape construction/invocation checks; if they fail,
localize the exact cross-tape edge before introducing another boundary.

04391 passes the complete D3 comparison under both an active construction tape
and an active invocation tape after removing blanket recording suppression.
Local forward derivatives work, full payloads/callbacks agree, and external
start/scale derivatives remain disconnected. Keep this runtime fixed while
running the remaining seven complete-reference cases as fresh-process shards.
The runner adds only the explicit shard list; the runtime docstring now describes
the verified boundary. No numerical source or criterion changes for this cohort.

04392--04394 pass D1 stationary, D1 displaced scalar and D3 nonlinear complete
comparisons. 04395 fails solely at the diagnostic formatter: its general-purpose
clean() emits the string "nan" whereas the public payload converts nonfinite
numbers to JSON null. Preserve this failure and replace that final formatting
step with the existing public _json_ready. Do not change the numerical source
or comparison tolerance. Run the four adverse shards, then a real unsupported
XLA-operation/owner-release check. This repairs a comparison harness, not target
eligibility. The nonlinear reference and candidate both reject terminal geometry.

04396--04400 pass all four adverse comparisons and the complete compiler/owner
boundary. The invalid third locator call is recovered by the original algorithm;
both implementations report invalid rows and retain the same accepted result.
It is not evidence of a curvature-partition-invalid full endpoint; that boundary
is independently covered in prepared curvature checks. The compiler test raises
on the actual unsupported operation, with no eager calls or physical target rows;
the Python owner/program/scope are collected. Native executable eviction remains
unproved. Qualify the nine complete checks on an available compute GPU with this
numerical source frozen before changing the public dispatch.

Public integration review while the GPU qualification runs: preserve the
original validation order, including an initially invalid target taking
precedence over dimension-dependent curvature row-count errors. Native owner
construction currently calls _cloud_row_counts before any target and would
change that behavior. The public integration must represent this static late
error as a deferred terminal status, compile only the valid prefix in that
configuration, and raise the same error at the completed reporting boundary
only if the original curvature stage would have been reached. Unused cloud
storage may use valid configuration-only extents; no curvature target or fit may
execute in that case. This is an error-precedence repair, not an eager fallback.

Use one retained public owner keyed by callback identity (including bound-method
receiver identity), dimension, device and validated configuration. Prepare
clouds on CPU using the qualified generator and execute the complete numerical
program on the requested device. Results remain frozen and resource invocation
serialized. Completed reporting traverses history for field names and JSON only;
all norms, logarithms, selection and eigen summaries come from native outputs.
Keep an explicit, labeled non-JIT wall-clock diagnostic for the existing opt-in
configuration. No compiler failure may select it automatically.

When the public module changes, full reference calls still execute its entire
031692a0b source. Verify every other existing BayesFilter Python dependency
byte-for-byte, and verify unchanged public-module helpers by AST identity for
stage references. Explicitly enumerate the replaced entrypoint/result/report
symbols; all numerical reference calls must stay on frozen or verified shared
code. New candidate modules are never reference dependencies. Test the exported
API against full frozen payloads and physical callbacks, including callback
identity, repeated changed operands, initial-invalid and late-configuration
precedence, compiler failure, explicit wall diagnostics and original API tests.

The configuration audit also identifies the inherited condition-number rule: the
posterior configuration permits positive values up to one, while the factor
configuration rejects them only after a completed valid curvature cloud. Preserve
that late exception, including partition-validation precedence, by a native
validation-only branch for that known static configuration failure. The numerical
fit never runs in that branch. Initial-invalid and valid-initial comparisons must
retain the original error text and physical callbacks for both this case and
invalid row extents. Other compiler errors continue to propagate, never fallback.

04411 passes the first actual exported D1 call chain, including a spy proving
three invocations of the same native owner and a veto on recomputed host eigen
summaries. 04412 passes all four late-row/factor-configuration checks, preserving
both initial-invalid precedence and all physical callbacks before the original
exceptions. The first reporting allowance list contained duplicate entries for
the same AST; the second lacked its occurrence count. Preserve both versions.
The corrected list uses the existing exact occurrences=2 schema for the two
identical completed-ledger loops. The guard passes with 276 sources / 1436 exact
allowances; all nine additions are completed-report traversal only.

Before the full public matrix, label the explicit host-wall route, report the
actual configured fit JIT mode, and inline the existing eigen-summary body under
the enclosing function's JIT setting. No eigen formula or default changes.
Extend full endpoint qualification to D5 with an explicitly configured two-factor
target: covariance D[diag(1-row_norm(L)^2)+LL^T]D, fixed documented fixture
loadings, analytic Gaussian scores, and the original factor_2 option. Require
both factor_2 replicate records to contain computed precision matrices. This
tests an identifiable family unavailable at D1/D3 and does not promote fitted
geometry. Compare every public field and callback at the unchanged tolerances;
a mismatch still stops the cohort. The fixture uses 50 movement-fit iterations
on both sides; other cases retain eight. CPU qualification precedes GPU.

The remaining public/cost worker partition expands the enclosing reservation to
96 CPU / 72 GPU workers while retaining the same 10,800 combined seconds and
global 56/52-hour caps. This gives room for separately executed public cases,
late exceptions, identifiable factors and repeated cost arms; it is not more
compute authorization. The 300-second worker limit and one documented
900-second compiler-capacity retry remain unchanged. Stop on any unit/global
budget exhaustion. These limits supersede only the earlier worker counts.

Cost-harness review (not yet launched): measure fresh-process prior/public graph/
public XLA arms at D1/D3, with three warm calls and changed-input calls per worker,
then three independently repeated interleaved cohorts. Time the complete exported
call and completed payload, including cold construction and CPU preparation.
Record RSS/HWM and mapping counts, TensorFlow allocator current/peak, and GPU
process sharing separately. Reset allocator peak before each call; do reference
comparisons and IR inspection after timed memory snapshots. Require identical
source/environment/fixture identities across each cohort and retain failed arms.

Use a fixed, identifiable one-factor Gaussian for these costs, so the condition
reports have a resolved numerical meaning. This deliberately does not resolve
historical ill-conditioned report gates. The immediate baseline is the whole
031692a0b public module with verified dependencies and the unchanged TensorFlow
random stream. It measures removal of host control since that checkpoint; it is
not the oldest-original campaign baseline. The old endpoint rebuilds callback
programs between calls, so no identical-graph compiler-ablation claim follows.

Compare complete numerical payloads at the existing tolerances and independently
check Gaussian location and covariance. Preserve jit_compile fields verbatim in
raw evidence, validate the measured owner's actual flag, and treat only those
boolean execution-setting fields separately in numerical comparison. In the
old public module the locator switch does not switch every fitting dependency;
the new graph arm is an explicit non-default diagnostic. Record nested XLA
function counts rather than calling it wholly uncompiled without inspection.
A graph numerical failure forbids graph speed ranking and triggers attribution;
it does not silently become a pass because the final covariance is close.
Default XLA and prior pairs still require their own complete-record comparison.

Primary cost triggers remain cold >2x, warm >1.2x, host RSS >256 MiB extra or >2x,
and GPU allocator peak >2x. Triggered arms require attribution rather than silent
acceptance. Short fixture timings are descriptive, not superiority or real-target
speed guarantees. A bounded owner and snapshots cannot prove native executable
cache eviction or broad capacity. Preserve the draft outside the frozen numerical
cohort until public correctness finishes and this contract is checked against the
actual harness and analyzer.

Recovery after 04422: all nine exported CPU fixtures and four late-validation
cases pass, including the identifiable D5 two-factor case. The new exported
initializer tests had accidentally occupied the already tracked curvature
public-test filename. After the matrix ended, preserved the new tests as
`test_filter_repair_posterior_initializer_public.py`, restored the entire prior
tracked curvature file byte-for-byte from 6bceb767e, and corrected only the new
runner registrations. Original curvature coverage is retained. The old matrix
source snapshot preserves the accidental overlap; no run source was edited.

Pre-execution boundary review: exercise the actual exported endpoint with a real
unsupported XLA string operation, verify zero physical/eager target calls,
bound-method reuse, receiver/configuration separation and Python owner/program/
scope collection. Separately require the explicit non-JIT wall-clock exception
and label; do not turn a compiler error into that path. Run the five original API
cases in fresh workers, then the exported GPU fixtures and boundary/API matrix.
These are correctness checks under the existing enclosing reservation. Restore
all original coverage before cost work. No tolerance or numerical algorithm is
changed by this recovery. Cache collection does not prove native memory eviction.

Cost draft review while public GPU sources are frozen: use factor_max=1 for
these D1/D3 one-factor Gaussian costs, so an unnecessary underidentified second
factor cannot make the timing fixture's report comparison meaningless. The D5
public qualification separately exercises factor_max=2 with a genuine two-factor
covariance; it is not a timing substitution. The original, graph and XLA cost
arms share all numerical settings and operand hashes. Preserve three complete
warm payloads and the measured costs before reference or compiler-IR inspection,
so a late diagnostic failure cannot erase the measurement. Compare every report
field at existing tolerance; only boolean jit_compile metadata may differ.

The independent analyzer rechecks full payloads, Gaussian means/covariances,
replay, compiled flags and absence of host callbacks. It verifies source,
environment, physical-device, growth and uncontended timing provenance. Failed
arms remain visible and produce no performance ratios. Tests inject a corrupted
condition number, independent mean, replay, compiler flag, source drift and
missing process repeat to check those vetoes. The three-process cohorts remain
descriptive; three warm calls per process do not meet or replace the master's
20-call terminal capacity gate. These immediate-checkpoint measurements time
completed public calls including payload reporting in every arm; this scope is
separate from the numerical-kernel timing required by the final master gate.

The cost manifest also records CPU model/affinity, host, Python, TensorFlow/TFP
versions, explicit XLA flags and host load observations. Compare fixed hardware
and affinity within the cohort. These factor_max=1 costs do not bound the
factor_max=2 default or D5 capacity; retain those limits in the results. The
parent campaign's terminal baseline, 20-call stability and actual-caller gates
remain required regardless of this smaller cost unit's outcome.

GPU 04432 reaches the 300-second complete-worker limit on the identifiable D5
fixture without a completed comparison. All preceding eight GPU fixtures pass.
Preserve 04432 as a capacity failure; captured output does not identify whether
compilation, reference execution or comparison consumed the deadline. Exercise
the plan's one 900-second capacity retry with identical sources/inputs, then
resume only the unexecuted late-validation group. A retry pass cannot erase the
300-second failure or prove which stage caused it. No tolerance or method change.

The prepared cost analyzer passes 14 synthetic adverse checks in an isolated
/tmp draft-review directory (0.06 pytest seconds; no TensorFlow/GPU import).
This covers failed-graph ratio suppression, full-field numerical corruption,
source drift and missing repeats. The frozen public GPU source tree was not
changed. The registered analyzer/policy check will bind the installed harness
to its campaign source identity before numerical cost launches.

Identical-source D5 GPU retry 04433 passes in 411.744036 seconds, retaining full
records, physical callback order/counts, changed-input reuse, replay and stable
HLO. The 300-second failure remains a real capacity observation. A read-only
snapshot at 6:06 elapsed observed 7,207,748 KiB host RSS and 618 MiB GPU process
reservation; it does not identify the responsible stage or TensorFlow allocator
peak. The snapshot is posterior-d5-capacity-observation-04433.json. Resume only
the final GPU validation group; do not rerun the already passing fixtures.

Public correctness is complete through 04434: 13 CPU and 13 GPU checks pass.
Restore the boundary diagnostic's physical counter, keeping the original
non-eager callback helper semantics. Install the reviewed cost-only harness and
analyzer now so boundary/API, policy and cost workers can share one source
closure. Numerical costs remain blocked until the original API/boundary checks
pass. The cost batches join the existing uncontended GPU preflight guard, with
all six GPU arm/dimension registrations covered by its rejection test. Internal
controller batch discovery excludes these new public/cost groups so earlier
internal batches do not silently expand. A module documentation correction
records the already qualified public dispatch; numerical implementation is
unchanged since 04434.

Original API 04440 exposes a graph-only empty-output failure on the D1 cloud-winner
fixture (the stage scalar is returned as shape [0], float32). Preserve the failure
and freeze numerical runtime. Costs are paused. Localize with direct native
outputs versus declared specs, isolated locator and movement stages, a fresh
process with Grappler disabled, and the unchanged XLA arm. Disabling Grappler is
only an explanatory diagnostic; it is not an approved global runtime repair.
Compare full records to 031692a0b when valid outputs exist. Reserve five CPU
workers at 300 seconds from the existing unit; inspect the first differing stage
before any implementation change. A non-XLA failure cannot be waived for costs.

Diagnostic 04441 fails during fixture construction: its existing independent
Gaussian helper requires NumPy arrays, but the new test passed lists. Restore
the original array inputs in this diagnostic-only fixture and preserve the
failed harness. This failure executes no candidate comparison.

04442 reproduces all native graph outputs as empty float32 tensors despite
nonempty declared schemas. 04443's movement isolation initially omitted explicit
None eligibility arguments; preserve the harness failure and correct its
configuration-only call. 04444 with Grappler disabled passes output schemas and
complete pinned records. This implicates a graph optimization/execution
interaction, not the report formatter. It does not justify disabling Grappler
repository-wide. Continue isolating the stage and individual optimization pass;
keep runtime source frozen and preserve the ordinary graph failure.

04445/04446 isolate healthy movement and zero-iteration locator outputs under
normal graph optimization. Next test function optimization alone, then preserve
one function boundary at a time with the documented tf.function _noinline
attribute in the diagnostic harness. These are topology-only hypotheses, not
algorithm alternatives. Complete-reference comparisons must pass before any
local execution annotation is installed. At most four additional 300-second
CPU workers under the unchanged enclosing/global time caps; no global optimizer
setting may be added to runtime as a shortcut.

04447 still fails with function optimization disabled, so inlining alone is
not established as the cause. Use the remaining selective-pass allocation for
dependency optimization, pruning, arithmetic optimization or constant folding,
stopping on the first distinguishing pass. Each change remains a fresh-process
diagnostic; no corresponding global runtime option is authorized for installation.

04448 passes complete records with only dependency optimization disabled. The
first discriminating pass is therefore TensorFlow dependency optimization on the
composed graph. Preserve normal options in runtime. Test a local no-inline
locator boundary as a topology-only repair candidate; if ineffective, inspect
the explicit dependencies at the enclosing locator/curvature boundaries. No
arithmetic, algorithm or target decision changes are permitted by this finding.

04449 retains the empty-output failure with a no-inline locator. Test the
explicit enclosing control edges next: replace the two blanket dependency lists
(location/curvature entire output histories) with each stage's completed scalar
status as a diagnostic source transformation. Preserve the exact original and
changed class sources and hashes in the diagnostic record. Numerical statements
and shared dependencies remain unchanged. This tests whether the enclosing
control topology causes dead outputs after dependency optimization; it does not
admit copied numerical implementations or permit dropping callback-order checks.

04450 still fails after reducing the enclosing status dependencies; reject that
change. Isolate the terminal curvature stage at the exact Gaussian center and
try its no-inline boundary (two bounded CPU diagnostics). A failure inside the
standalone curvature stage would supersede the earlier composition-only
hypothesis. Preserve this distinction; no runtime repair is installed yet.

Standalone curvature 04451 also returns empty tensors; the problem is therefore
inside that stage, not solely its enclosing posterior controller. 04452 exposed
an accidental late-bound diagnostic variable shadowing in the no-inline wrapper;
fix the harness, preserving its source. Continue with the validated fitter alone
(one bounded worker) before further topology experiments. Each narrowing step
must use the same prepared offsets and declared numerical settings.

Validated fitter 04453 reproduces the empty-output schema failure without target
callbacks. Next invoke its actual captured raw fit function with identical data
and captured constants, bypassing only partition validation. This single CPU
check separates the validated boundary from the underlying numerical fitter;
its outputs remain a diagnostic, not a new runtime route.

Raw fitter 04454 fails without validation or callbacks, localizing the issue to
its numerical graph. One final execution diagnostic retains TensorFlow's
functional If/While representation (the representation naturally retained by
XLA) before graph lowering. The private TensorFlow flag is confined to a fresh
test process and is not a proposed runtime global mutation. If it passes,
local graph construction boundaries need review before adopting any repair.

The short localization workers consume the existing time allocation. Repartition
the worker-count ceiling to 144 CPU / 96 GPU workers while preserving the 10,800
combined-second unit cap and global 56/52-hour caps. This permits smaller
root-cause shards and retains all failed charges; it adds no compute hours.

04455 preserves complete records with functional control flow retained. The next
candidate uses the public tf.experimental.function_executor_type context only
while constructing/tracing the owner, exiting before numerical calls. In TF
2.19.1, control_flow_util_v2.py lines 123--128 skips switch/merge lowering in that
context; eager/context.py lines 2903--2922 defines it as a public context manager
and lines 1507--1529 use thread-local options restored on exit. Verify restoration
and normal execution outside the context. This avoids a process-global optimizer
or private lowering-flag mutation. It is a local execution-policy candidate,
subject to complete records, callback ordering and original API regression.

04456 timed out at 300 seconds under the construction-only executor context;
reject that candidate without retry. The next local diagnostic annotates only
If/While operations built by the actual fixed-center fitting, selection and
stability authorities with TensorFlow's existing no-lowering attribute. It
uses a module-local TensorFlow delegate, retains exact numerical bodies and
records each marked graph operation. The TensorFlow package, global lowering
flag and optimizer settings remain unchanged. This tests a bounded construction
annotation before any such helper is installed in runtime.

Recovery through 04457: 04456 timed out and no worker remains active. 04457's
local fit annotations restore every declared output except the two terminal
eigen summaries, whose eight scalars still return empty float32 tensors. Their
conditionals belong to the enclosing owner rather than the three fit modules.
One additional 300-second diagnostic applies the same local annotation there.
The promotion criteria remain complete records and callback ordering; the
annotation is explanatory until installed and qualified with ordinary options.
No global optimizer change, tolerance change, or numerical replacement is proposed.
The unmodified Git dependency closure must remain the comparison authority if a
shared fitter changes. Review: this test discriminates the remaining dead branch
without changing numerical statements; schema-only success is insufficient.
All exploratory graph groups are now explicitly explanatory; original API
regression groups remain mandatory. Through 04457 the enclosing unit used
7,394.700262 seconds in 83 CPU and 46 GPU workers, leaving 3,405.299738 seconds
of its 10,800-second reservation. Global caps remain 56 CPU / 52 GPU hours.

04458 passes all declared output schemas and complete records with local
functional annotations in the fit modules and enclosing owner. Install explicit
functional-control constructors for non-XLA graph mode at those four boundaries;
XLA continues to select ordinary TensorFlow constructors. The shared helper
changes only the existing operation lowering attribute and fails during tracing
on an unexpected operation type. It does not change predicates, bodies, loop
bounds, TensorFlow optimizer options, the executor, or global module bindings.
Private-API compatibility is a material limitation, guarded by the regression.

Reference strengthening: the complete initializer and all numerical imports now
execute from Git 031692a0b in isolated module names. Add the empty runtime package
namespace to the existing strict frozen loader so its explicit device-policy
imports resolve. All frozen sources remain hashed; the three changed shared
fit modules must be present in that frozen closure, and the new helper must be
absent. Unchanged sources retain whole-tree byte checks. Stage references use the
same frozen namespace. This prevents a candidate-against-itself comparison.

Review: 04458's numerical statements were unchanged, but its module delegate
could affect shared comparison factories. Therefore it is explanatory evidence,
not final qualification. First run the exact cloud-winner failure against the
installed code and isolated reference with ordinary optimizer options, then
qualify ordered callbacks, original API and focused graph construction checks.
Cost cohorts remain blocked until these required checks pass. No extrapolation
from this small D1 fixture to native capacity or whole-program completion.

04459 fails before numerical execution: the frozen runtime package exports
stable_config_hash through its lazy registry, so an empty namespace is
insufficient. Execute its exact Git __init__ and redirect its import_module
binding into the frozen loader, preserving the original export map and numerical
closure. This is a harness repair; no source hash or numerical gate is waived.
The one new source-policy allowance is fixed-schema graph metadata traversal;
no numerical Python loop or NumPy exception is added.

04460 passes the installed D1 graph endpoint against the fully isolated frozen
reference, including complete records, exact callback positions/counts, changed
inputs, replay, one trace and frozen external gradients. Run the registered CPU
then GPU graph-qualification shards: the original cloud-winner and sentinel
regressions, D1/D3 stationary, moving and adverse complete-record cases,
compiler/cache/wall boundaries, plus representative D1/D3 default XLA renewal.
The two helper checks verify zero/nonzero loops, both branches, resource counts,
no global optimizer mutation and fail-closed representation changes. Source
snapshot: posterior-graph-repair-04461-source. Cost ratios remain prohibited on
any failed numerical arm; stop and localize rather than expanding tolerances.

04461--04464 pass the helper, original cloud/sentinel, and full D1 graph checks.
D3 graph 04465 fails full records. Its saved records identify a configuration
wiring defect: the frozen locator_config.jit_compile flag controls only the
locator; original movement and fixed-center fitting still call XLA-default
factories (031692a0b posterior_local_initializer.py:586--589,638--646,898--912;
fixed_center_curvature.py:356). The new owner had propagated the locator flag
into every stage, and even reported the factor fitter as non-JIT. The changed
input also differs in fitted matrices, so this is not merely a condition-number
report and cannot be waived as ill conditioning.

Correct the owner to preserve XLA for movement and curvature dependencies even
when the locator/enclosing call is the explicit graph exception. Use ordinary
TensorFlow conditionals in the owner for the first check. This is the original
public execution contract, not a new algorithm or hidden fallback; costs must
record the nested XLA functions. First rerun D3 graph and the original D1 failure.
If ordinary constructors now pass, discard the provisional private-attribute
repair from runtime and preserve it only as explanatory evidence. The reference
loader remains isolated and unchanged. Review: matched per-stage execution
settings are necessary for a fair comparison; simply ignoring condition-number
or jit_compile fields would hide the defect. No tolerance or algorithm change.

04466 restores every D3 numerical field and callback comparison at unchanged
tolerance across original, changed and replayed inputs. Its only six failures
are the two factor-fit jit_compile labels per invocation, still inferred from
the locator option. Emit the actual fit compilation setting from the native
curvature authority and report that completed scalar. Remove the provisional
functional-control helper, its active tests and allowance, and restore all three
shared fitting modules byte-for-byte. Their diagnostic implementation and tests
remain in posterior-graph-repair-04461-source. No private TensorFlow annotation,
global optimization change or additional policy allowance is in the final repair.
Rerun the registered CPU/GPU graph qualification with ordinary constructors.

Cost execution allocation after graph qualification: the original enclosing
unit retains its 10,800-second cap for correctness, failures and policy renewal.
Move the still-unlaunched matched cost cohort into a separate allocation of
10,800 combined charged seconds, at most 18 CPU plus 18 GPU workers and four
short analysis/policy workers. This consumes the same authorized 56 CPU / 52 GPU
hour campaign totals; it adds no hours. Each cost worker has 300 seconds, with
no automatic retry after numerical failure. Stop the cost unit on a failed
arm or source/environment drift, preserve measurements before comparison and
localize using fresh budgeted diagnostics. Document any later repair before
starting a new source-frozen cohort; never combine different sources in ratios.

The existing registered commands are run_filter_repair_campaign.py matrix
--stage tests --test-batch posterior_initializer_cost_cpu (and _gpu), with
--repeat 0, 1, 2, --test-timeout-seconds 300 and a preflight-qualified GPU index.
Within each repeat interleave prior/graph/XLA at D1 then D3. CPU comparisons
precede GPU. Run posterior_initializer_cost_analysis first. Use the installed
analyzer with explicit root/run bounds/devices and a new output filename, then
reopen/checksum the analysis and archive all underlying run records.

Skeptical review of cost design: the isolated prior module and candidate have
separate Python caches; reference execution and IR inspection occur after all
timed calls. Memory snapshots use bytes and separate RSS/HWM from allocator
current/peak. The graph control deliberately retains XLA dependencies under the
original public option and records their count; this is not a pure compiler
ablation. The Gaussian mean/covariance check is an independent veto in addition
to full record parity. Same seed, fixture, source, backend and visible UUID are
required across repeats. Sampled sharing cannot prove exclusivity; three process
repeats are descriptive, not statistical superiority. Actual DZ5 and native
20-call capacity remain separate gates. No numerical failure earns a speed ratio.

Residual scope: the final public graph option retains its original XLA
numerical dependencies. This resolves the exposed regression, not TensorFlow's
all-graph raw-fitter defect preserved in 04454. No all-non-XLA baseline claim
is permitted; keep that explanatory failure and its local-annotation candidate
available for a separate caller-driven repair if such a route is required.

Final correctness qualification through 04488 passes: 11 CPU and 11 GPU checks,
14 analyzer checks and 147 policy checks. Ordinary constructors preserve full
records and ordered callbacks after the original per-stage XLA settings are
restored. The final correctness unit used 9522.019909
combined seconds, within 10800; no worker remains. Archive and local terminal
review: filter_gradient_posterior_public_result_20260927.md. Continue the already
reserved matched cost cohort after committing the qualified source checkpoint.

Recovery after 04489: the qualified checkpoint was committed and pushed as
976c33552, including remote main 06590cb5a. The first CPU cost worker failed
before any timed public call because the environment imports TensorFlow
Probability without distribution metadata named tensorflow-probability. Preserve
04489 as a harness failure and charge its 5.878002 seconds. Record the imported
module's __version__ instead; no package installation or environment change.
The numerical runtime, inputs and gates remain unchanged.

Restart all accepted costs from a new frozen source snapshot. Repartition the
same 10,800-second cost allocation to 19 CPU / 18 GPU cost attempts plus four
short analysis/policy workers, counting failed 04489 among the CPU attempts.
The successful matrix still requires exactly 18 CPU and 18 GPU workers. The
time and global caps are unchanged. Analyzer checks already passed in 04477;
the reporting-only repair needs the first complete cost worker to verify the
actual environment field, followed by the existing numerical gates. Stop and
localize any subsequent failed arm before continuing the cohort.

Skeptical recovery review: module __version__ describes the numerical library
actually imported and avoids an unsupported packaging assumption. The repair
executes before timing and changes neither measured numerical work nor the
baseline. Preserve 04489 separately; do not count it as a successful cost arm
or combine its source fingerprint with the new cohort. Recheck GPU sharing
before GPU costs; CPU reference execution remains explicitly CPU-only.
