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
