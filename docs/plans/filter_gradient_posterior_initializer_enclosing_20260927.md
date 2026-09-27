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

Reserve 40 CPU and 24 GPU workers, 10,800 combined charged seconds, inside the
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
comparison and a counted target are the early discriminating checks. Public
integration and original complete-endpoint comparisons remain required after
this preparation dependency passes.
