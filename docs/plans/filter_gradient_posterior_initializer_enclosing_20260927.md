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
