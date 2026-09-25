# DZ5 graph replay localization

Continue from pushed `6ed86f8d8` under the existing 56 CPU / 52 GPU
process-hour caps. Through 03882, 34.491131 CPU hours remain. Reserve at most
24 CPU workers / 7200 charged seconds for this bounded diagnostic and any
localized repair checks. Use the stable campaign runner, one numerical worker
at a time, with runtime/scripts/tests frozen during each worker. GPU work,
MacroFinance edits, admission refresh, training and main merge are excluded.

Question: where does the first repeat-dependent analytical tangent arise in
the 96-observation CPU graph reference? Runs 03863/03864 preserve the failure;
03869's two-observation direct-filter diagnostic replays exactly. The full
default-XLA oracle already passes. A graph failure neither invalidates those
XLA results nor establishes repository-wide correctness.

Use the unchanged read-only `dz5-candidate-source-score-4c37f9f40-r1` snapshot,
truth-centered 185-row bank and its original 96 prepared observations. Start
with horizon 8, then 16/32 and, if needed, 64/96. Run direct shared-filter and
actual public target routes separately in fresh processes. A diagnostic fixture
proxy overrides only `likelihood_observations()` with an exact prefix of the
fully prepared data; it delegates all remaining attributes unchanged. Slicing
raw observations would incorrectly change the predecessor subtraction and is
forbidden. Public-route results include the original prior/admissibility logic;
direct results additionally expose initial and terminal moments and tangents.
Neither shortened route is a qualification of the full target.

Keep float64, TF32 disabled, op determinism enabled, two intra-op threads and
one inter-op thread, as in failed 03864. Three calls of one stable-signature
graph preserve full arrays, changed-element counts, maximum residuals, trace
count, graph identity, timings and RSS. Hash all loaded project sources before
and after. First-call arrays are the exact replay comparator; no approximate
equality or tolerance waiver is eligible. At the first differing prefix, stop
lengthening that arm and bisect the prefix or freeze its first differing
operator's operands. One-thread/operator-optimizer controls are explanatory
follow-ups only after reproducing the discrepancy. Record their configuration.

Replay differences are a repair trigger and veto qualification, while diagnostic
completion can pass even when a difference is found. Register these groups as
explanatory; a green diagnostic must never satisfy a numerical terminal gate.
Nonfinite/invalid results, provenance drift, timeout, or unexpected exceptions
stop that arm before interpretation. At most three localized retries per
unchanged fixture. Preserve all failed attempts and charge the runner's recorded
process time, including child execution. No unchanged full retry follows a
negative localization. The GPU graph finite-difference veto is a separate issue.

Before a runtime repair, identify the affected shared operation and compare
fixed operands to an independent reference. Repair must preserve the equations,
seeded streams, finite-value/status gates and original oracle tolerances. A
candidate needs focused regression, exact replay and renewed full actual target
oracle before this graph gap can close. Compiler and capacity observations are
explanatory; there is no performance ranking from these instrumented runs.

Skeptical review: instrumentation or shorter graphs may hide the defect, and
one successful prefix cannot prove determinism. Running the actual wrapper
alongside the direct filter distinguishes wrapper omission from shared-filter
behavior without editing external source. The fixed bank is an inherited
qualification fixture, not a posterior-wide sample. Thread count is an
execution hypothesis, not an assumed cause. This plan has adequate source,
baseline, stop and artifact boundaries for localization; no independent agent
review was used. Save numbered raw artifacts and a terminal result with the
remaining uncertainty and next discriminating action.

Execution through 03887: direct/shared and public-wrapper prefixes of 8 and 16
observations replay exactly across all three calls. The direct 32-observation
prefix also replays exactly, including initial/terminal means, factors and their
tangents. These successes do not prove a monotone first-failing horizon: graph
shape, fetch structure and allocation can also affect an intermittent defect.
Continue the matching public 32-observation diagnostic before choosing the next
localization. Full-horizon graph failures 03863/03864 remain unresolved.

Harness review while 03888 runs: the prefix diagnostic recreates its input
tensor per call, whereas 03864 reuses one `positions` tensor. The public prefix
also returns a dictionary rather than the original ordered tuple. These are
uncontrolled execution differences that can conceal buffer-lifetime/scheduling
effects. Preserve this cohort, finish 03888, then reuse one positions tensor
and verify it is bitwise unchanged after each call. Add an exact public-tuple
fetch control. Start with the reused-input horizon-8 public dictionary route to
isolate the input-lifetime change; do not infer that a larger horizon is needed
until this control is checked. A changed outcome triggers matched fresh/reused
input diagnostics before operator attribution. All changes are in the diagnostic
harness; the read-only target snapshot and numerical gates remain unchanged.

03889's reused-input public horizon-8 diagnostic and 03890's reused-input,
original-tuple horizon-32 diagnostic still replay exactly. Before lengthening
further, preserve the original oracle harness verbatim except the prepared-data
prefix and explanatory reporting: this also restores its explicit CPU placement,
retained first-result tensors, finite-difference reporting and RSS sampler.
Start at horizon 8, then extend adaptively. Prefix finite-difference and replay
failures are saved and reported, never converted into qualification; only the
diagnostic's completion is a test pass. Recheck input tensor equality. This
control tests whether the simplified harness concealed an execution condition;
it does not introduce a different numerical implementation or another allowance.

03892 reproduces the discrepancy at 64 observations in the original harness:
values are exactly equal, scores differ by at most 5.7980287238024175e-12,
and both independent prefix finite-difference gates pass (scaled errors
0.03750/0.06580). Input positions remain unchanged. This is a diagnostic
completion with a replay veto, not a qualified numerical run. Next shorten the
same harness to 48 observations; use shared-filter state/tangent records and
one-thread controls at a reproduced prefix before blaming an operator. Existing
simplified-prefix successes do not establish a rigorous lower bound on failure.

03893 reproduces at 48 observations: exact values, score difference at most
9.094947017729282e-13, both finite-difference checks passing. Freeze the mean,
factor and their complete analytical tangents after the first 47 observations
using the shared filter, then apply observation 48 twenty times to identical
frozen operands. Save every output before any assertion. Run two fresh children
sequentially at two and one intra-op threads; the second must read the exact
first child's checksummed inputs, without recomputing its initial state. Keep
explicit CPU placement, one trace and retained first outputs; verify that all
input tensors stay unchanged. Compare values, scores, means, factors, tangents
and validity. This freezes a later numerical state for execution diagnosis;
it does not qualify derivatives under changed parameters, a full trajectory,
or a numerical repair. Stop for invalid/nonfinite state, input mutation or
source mismatch. Replay differences trigger operand/operator capture, not a
tolerance change. The shared callbacks are time homogeneous and accept state
and process inputs; replacing only the initial moments/tangents preserves the
single update's equations. Primary-agent skeptical review: changed loop shape
may still conceal the failure, so an exact single-step replay cannot close the
full-prefix failure. This allocation remains inside the original 24/7200 cap.

03894: two-thread frozen step 48 replays exactly across 20 calls. The one-thread
child fails before computation because the read-only snapshot hides the sibling
output path. Preserve this harness failure. Copy the exact saved NPZ into the
second child's bound output directory and still require identical SHA-256, then
rerun the complete pair. This repairs input access, not numerical source or the
replay criterion. The full-prefix score mismatch remains unresolved.

If frozen step 48 remains exact, capture the shared recurrence's carried state
at every date of a 48-observation direct-filter execution. A diagnostic wrapper
around its existing `tf.while_loop` body adds TensorArray outputs; it calls the
unchanged body and returns its original state unchanged. Match only that named
SRUKF body, leaving nested model/CIR loops alone. Preserve three complete calls
and identify the first differing time/field and batch coordinates. This is
instrumentation of frozen source, not a copied numerical recurrence or runtime
edit. Instrumentation can affect allocation/scheduling, so an unreproduced
failure remains unresolved. In a reproduced case, use the previous saved state
as the next frozen-step input and trace the implicated shared operation. At most
three trajectory calls per worker, one trace, unchanged source hashes, finite
states and validity, no host callbacks. Full saved histories support an
independent comparator; summary-only assertions are insufficient.

03895 completes both 20-call thread controls with exact replay on the identical
saved step-48 inputs. No thread cause or single-step defect is established.
Proceed with the reviewed carried-history capture at 48 observations; the
full-prefix mismatch remains the repair trigger.

03896 localizes a reproduced difference: call 2 matches call 1 at every date;
call 3 first differs in `d_mean` at observation 35, then `d_factor` and score at
36. Means, factors, values and validity/conditioning diagnostics stay exact.
The complete histories are saved in `run-03896/prefix-replay-{0,1,2}.npz`.
Extract the shared exact state/tangents after observation 34 and replay the
single observation-35 update twenty times per thread setting. Verify the first
two trajectories' equality through date 34 before using the witness.

Expose the unchanged numerical body's intermediate tensors through a diagnostic
step wrapper: Python profiling captures symbolic local variables only during
tracing; all computation still calls the original body. Carry these extra
outputs through the one-step loop, without a host callback. Compare each
intermediate and save complete first and changed records; output instrumentation
is explanatory, not an eligible runtime path. The extra nested graph and fetch
lifetimes may conceal the defect, so failure to reproduce cannot close the
original mismatch. Reuse exact witness operands across one/two-thread children,
check input immutability, source closure and trace count, and remain within the
24-worker/7200-second allocation. A reproduced primitive difference is the
repair trigger; a mere agreement with the final score is not root-cause proof.

03897's frozen observation-35 update and all 36 exposed intermediates replay
exactly for 20 calls on each thread count. This does not identify an operator.
Return to the reproduced full recurrence and add four small tensor histories:
`d_predicted_mean`, `d_increment`, `inc_score`, and `d_innovation`. Capture
symbolic locals during the original body's tracing and append TensorArrays in
the same loop; do not insert an independently compiled step. The exact equation
is `new_d_mean = d_predicted_mean + d_increment`. This diagnoses which input to
the first differing mean tangent changes without copying the filter arithmetic.
Again, save three full calls and every discrepancy; no inference of correctness
from instrumentation that suppresses the failure.

03898 does not reproduce the mismatch in three instrumented trajectories. Keep
03896's first-difference witness; the extra fetches may affect optimizer/buffer
behavior. Next test the unchanged original 48-step harness with arithmetic
optimization disabled, and separately with one intra-op thread. These controls
are explicit graph-only diagnostics, not new runtime defaults. Inspect the
installed TensorFlow arithmetic/aggregate implementation for input-order or
buffer-forwarding behavior. A passing control only motivates a focused primitive
test; it cannot by itself prove causation or close the full-horizon gate.

03899's original 48-step graph harness passes exact replay and both independent
score checks when only arithmetic optimization is disabled. This is a successful
control, not a runtime repair. TensorFlow v2.19.1 source inspected from its public
tag gives a specific mechanism: `core/kernels/aggregate_ops.cc:53--73` forwards
an available input buffer and swaps its index into position zero; the expression
then sums that permuted order. `core/grappler/optimizers/arithmetic_optimizer.cc`
`AddOpsRewriteStage` collapses same-shaped add trees into AddN (lines 519--546,
601--644). The tag agrees with installed version 2.19.1; binary build provenance
is still a limitation, so use executable evidence too.

Collect the actual optimized public filter graph at prefix 8 with the original
harness, enabling only graph collection (no runtime tensor callbacks). Save the
pre/post graphs and AddN nodes/inputs. Independently evaluate a three-term
binary64 cancellation fixture through AddN, exposing different operands as
additional outputs to control buffer eligibility. Compare exact mathematical
sum and the declared ordered sum, and preserve all observed results. This is a
mechanism diagnostic; cancellation fixture failure does not prove that this
specific filter executes the same failure. Full target comparisons remain at
their unchanged tolerances. Extend to the exact implicated filter operation
only if optimized-node provenance supports it.

03900 confirms the installed primitive: the same three terms sum to 1 when no
extra operands or all operands are exposed, but to 0 when the first two are
exposed, across four exact replays each. The executed target's graph has 78 AddN
nodes before optimization and 92 after, including new add-tree rewrites in its
filter loop. This establishes buffer-sensitive AddN behavior and an applicable
optimizer mechanism. It does not yet identify the exact failing filter node.
The original wrapper also includes local autodiff/pfor in frozen MacroFinance
callbacks (`_metrics_and_derivatives`, line 74); these are external source debt,
not analytical BayesFilter recurrence or authority for a pfor allowance. No
MacroFinance edit is authorized by this unit and no whole-target policy claim
follows from the existing XLA numerical qualification.

Run the unchanged full 96-observation, 185-row oracle with the single explicit
graph-reference control `arithmetic_optimization=False`. Require both original
finite-difference tolerances, exact replay, finite/valid outputs, one trace,
input/snapshot/import identity and no host callbacks. Retain the old failing
default graph runs. This can qualify only this configured reference arm; it
does not repair the default graph behavior, change default XLA or close consumer
policy/admission gaps. A failure remains a veto and triggers operation-level
localization. At most one 900-second worker plus localized harness retry within
the remaining unit budget; save all original oracle/replay artifacts.
