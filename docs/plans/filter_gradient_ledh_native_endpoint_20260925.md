# LEDH native endpoint execution repair

Continue the authorized execution-only repair from `98435791c`, with the
owner's additional 24 CPU hours: total caps are now 56 CPU / 52 GPU hours.
At the opening checkpoint 03818, 75364.243198 CPU / 75275.068507 GPU seconds
are charged. Numerical method, data, reset semantics, analytical derivative,
admission criteria and historical-result prohibitions are unchanged.

Question: can the existing registered LEDH value and analytical-score endpoints
execute their complete numerical recurrences through stable TensorFlow/XLA
boundaries while preserving supplied operands, seeded input contracts, values,
scores and discrete diagnostics? A compiled callback alone cannot answer this.

Use up to 24 workers / 7200 charged seconds for this endpoint unit inside the
global caps, with one numerical worker at a time and the existing stable runner,
versioned raw output root, source freezes, device-selection and verified memory
growth. Initially allow only two 300-second seeded-input compatibility workers
(CPU explicit reference and GPU default); implementation/source inspection is
unmetered routine work. Extend the focused matrix within this unit only after
recording what the seed compatibility result permits. At most three localized
retries per unchanged fixture. Stop an arm at an unexplained numerical failure,
source drift, missing diagnostics or inability to establish memory growth.

The seed contract is separate from the approved versioned geometry stream.
The old value path uses NumPy SeedSequence to initialize TensorFlow Philox,
plus NumPy PCG64 uniforms for annealed systematic resampling. First inspect the
installed TensorFlow RNG source and public NumPy source/reference vectors. Check
whether the existing TensorFlow normal primitive yields the original stream
under XLA with identical key/counter. Compare integer seed/counter material
exactly, and report bitwise equality and maximum numeric differences for normals.
This is diagnostic compatibility evidence, not permission to change seeded
realizations. If native XLA differs beyond rounding, preserve frozen-cloud
qualification and prepare an explicit versioned-stream proposal; do not silently
promote it or borrow the geometry-stream approval.

Baseline: exact merged endpoint source `9d8202b77`; current dependencies are
recorded separately. No pre-August-21 LEDH result or obsolete chunk policy may
be used. Tests may use small arrays only with policy K=N. New sampling code
must remain TensorFlow, with integer seed encoding at the configuration boundary
and native loops for numerical mixing; no runtime NumPy or Python numerical
loop waiver. Historical code may be loaded solely in diagnostic tests.

The score path already has a native recurrence, but its public direct wrapper
is eager and direction callbacks can close over mutable Python cells. A cached
decorator that freezes those cells is invalid. Inspect actual consumers and
use an explicit program owner/model builder or fully threaded dynamic captures,
preserving the current analytical recursion. The value path additionally needs
native time/stage recurrences, finite-state branches, cumulative resampling and
fixed-capacity diagnostic buffers. String metadata belongs outside XLA; numerical
outputs and actual realized histories must retain their public contract.

Criteria: exact source/input identity; original full records at existing
tolerances; independent derivative/reference checks; changed operands and
exact replay; status/history agreement; no callbacks, numerical Python loops or
NumPy; stable explicit signatures; actual enclosing HLO. Any failure vetoes
that candidate and triggers bounded localization. Missing canonical batch
construction, unsupported annealing in the score implementation, or the
inherited predator--prey source mismatch cannot be repaired by substituting a
different numerical algorithm. Those claims remain blocked and separately
dispositioned under the agreed scope.

After numerical qualification, use matched fresh processes for original graph,
candidate graph and candidate XLA costs, recording cold/warm time, sampled host
RSS and TensorFlow allocator bytes separately. Combined-process correctness
runs cannot establish before/after memory attribution or executable eviction.
Define the exact paired fixture, repetitions and acceptance triggers before
launching that cost matrix. No scientific, HMC, training or main-promotion
claim follows from this execution unit.

Skeptical pre-run review: the RNG is an input dependency and must not be replaced
under an unrelated prior approval. Source loops and enclosing compilation are
different findings. Reusing the score executor for value would change flow-cap,
annealed and diagnostic behavior, so it is not presumed equivalent. Existing
NumPy-generated reference inputs are allowed only in diagnostic tests. The
first two workers answer seed compatibility without editing runtime behavior;
this is a valid bounded next action. Primary-agent review; no independent
reviewer or subagent used.

Seed localization 03819/03820 completes six diagnostic cases per CPU/GPU
backend: four seeds (including a >64-bit seed), three successive calls,
three shapes and both float64/float32. Raw full-integer Philox outputs and
generator counter advancement match exactly. Default XLA normal conversion
does not reproduce the existing normal realization: maximum absolute differences
reach 5.23, not rounding. Therefore do not use XLA StatelessRandomNormalV2 as
a drop-in replacement. Both diagnostic groups pass because they successfully
measure compatibility; **normal compatibility fails**.

Source inspection supplies a preservation repair: TensorFlow's installed
`include/xla/tsl/lib/random/random_distributions_utils.h` lines 33--94 and
`random_distributions.h` lines 693--715 define the original uint-to-uniform
mantissa construction, 1e-7 lower clamp and paired sin/cos Box--Muller transform.
`python/ops/stateful_random_ops.py` lines 552--556, 639--649 defines key/counter
extraction and 256 * element-count counter reservation. Port those exact
operations to TensorFlow over the already-matching raw words; compare normal
values at the predeclared 1e-12 float64 / 1e-5 float32 limits and exact
integer/seed/order state. Rounding is reported separately from changing the
random realization. This preserves the numerical input method, not a new
stream substitution. The actual host filter remains unchanged until this
candidate is qualified.

NumPy 2.1.3 (installed comparator) public source was read from tag `v2.1.3`:
`numpy/random/bit_generator.pyx` lines 58--65, 152--164, 341--374, 404--450
(four-word SeedSequence pool); `numpy/random/src/pcg64/pcg64.h` lines 97--101,
164--196, 282--287 and 327--336 (128-bit LCG, post-advance XSL-RR output,
seeding), and `pcg64.c` lines 148--159 (seed-word order). The fetched source
copies and license notices will be retained with the evidence. Implement only
the unspawned integer-seed path used by this endpoint, using uint32/uint64
tensor arithmetic and native loops. The public arbitrary nonnegative integer
seed is encoded to little-endian words at configuration time; this byte
encoding is not random-number generation.

Next run at most two 300-second compatibility workers plus localized retries
within the 24-worker unit. Require exact NumPy SeedSequence outputs, PCG64
initial/updated 128-bit states, raw words and float64 uniforms over small,
large and carry-boundary seeds. Verify native Box--Muller against the original
generator for the same shape/seed/call schedule, one trace, no callbacks and
changed operands/replay on CPU/GPU. Failures stop integration and trigger
source-localized repair. Independent source/reference checks are the gate;
matching only a new implementation to itself is insufficient. No full filter
or scientific claim follows. Primary-agent skeptical review: constructing the
exact random bits and transform avoids a scientifically unapproved seeded-data
migration, while retaining explicit rounding and downstream equivalence gates.

03821: all six native Box--Muller cases pass. The seven PCG/seed cases stop at
tracing because TensorFlow does not implement unary Neg for uint64. Replace
the rotation complement `(-r) & 63` by `(64-r) & 63`, identical for r in 0..63.
This is an unsigned-op compatibility repair; no random algorithm or tolerance
changes. Preserve the failed attempt and repeat the same CPU group.

03822/03823 pass all 13 compatibility checks on CPU/GPU. All seven integer
seeds reproduce SeedSequence outputs, PCG64 128-bit initial/final state, 32 raw
words and uniforms exactly. The native Box--Muller preserves seeded realizations
within the predeclared rounding limits, with exact counter schedules and replay.
The largest float64 absolute discrepancy is recorded in the raw result; this
does not admit the complete filter. The unsupported raw XLA normal shortcut is
excluded from runtime integration.

Next implement the full value recurrence over supplied normal clouds and
systematic uniforms, sharing all existing UKF, flow and Contract-E primitives.
Use native time/stage loops, finite-state branches and padded diagnostics with
an explicit completed-step count. Preserve the existing public history lengths
at the completed-result formatting boundary. First qualify the fixed-input
program before wiring random generation or the public convenience wrapper.
No alternative LEDH method or score is introduced. Test exact merged source
with only its RNG objects supplied frozen draws; callback settings and numerical
primitives remain the current shared ones. This isolates execution from random
rounding and preserves the original numerical recurrence.

Declare the callback configuration fixed when a compiled program owner is
constructed; runtime observations, supplied clouds and uniforms are dynamic
tensor operands. Test changed operands, exact replay, both annealed and composed
flow stages, the finite prior cap, dual-cap/trust-region options and early
nonfinite termination at N=8, d=2, T=1/3. Since the existing reset explicitly
casts to float32, compare float numerical fields at atol=rtol=1e-6 and require
exact validity, completed counts, NaN masks and history shapes. Independent
checks cover the one-step UKF mean/covariance and systematic cumulative/search
rule; these do not establish Monte Carlo accuracy or scientific admission.
Require enclosing HLO, no callbacks, one fixed signature and native loop bodies.
Begin with two 300-second CPU/GPU workers plus bounded localized retries. Keep
runtime/public integration and matched costs gated by those checks. Review:
fixed-cloud comparison is the right baseline for this mechanism, but cannot
stand in for seeded-public or analytical-score qualification. Declared float32
reset precision is part of the old program, not a new tolerance relaxation.

03824 reveals another shared dependency: `ledh_flow_perparticle_tf.py` still
unrolls its substeps in Python and calls MatrixDeterminant, unsupported by
GPU XLA. Seven enclosing tests fail at compile; the independent systematic
resampling and UKF checks pass. The group name ended in `cpu`, but the explicit
device option was omitted and the runner defaulted to GPU. Preserve/charge
this as GPU (38.052359 seconds), not CPU evidence; all later calls state device.
No numerical integration is admitted. Repair the shared flow using a native
substep loop and extend the existing native pivoted-elimination authority to
batched matrices. Retain the old determinant's product-then-log arithmetic via
a determinant entry point sharing that elimination; do not substitute the
sum-of-logs finite program. Freeze the old flow source as well as the old
public wrapper in the diagnostic comparator. Qualify nonsingular, pivoted,
singular and mixed-batch determinants independently, retain the existing
slogdet derivative checks, and compare complete flow outputs before repeating
the full value group. This is an implementation compatibility repair within
the same 24-worker/7200-second unit; methods and tolerances stay fixed.
Skeptical review: an outer XLA function is insufficient when an inner
dependency is unsupported or Python-unrolled. The frozen-flow comparator is
required to avoid testing both arms against the repaired implementation.

03825 CPU: 18 checks pass, including all standalone determinant/flow tests,
composed and annealed value cases and all nonfinite guards. The three-step
dual-cap/trust-region case fails the unchanged 1e-6 gate: final ESS differs
by 2.53058928e-5 (relative 5.95530411e-6). Preserve the veto. Before changing
runtime or thresholds, run one 300-second CPU localization: capture the old
reset's exact inputs, compare eager/graph/XLA reset outputs and existing
conditioning diagnostics, then substitute only the compiled reset into the
frozen full recurrence. Compare native graph separately. This diagnostic may
explain the failure but cannot admit it or relax the gate.

03826 localizes the discrepancy exactly to compiling the existing FP32 reset:
substituting only graph/XLA reset into the frozen recurrence reproduces native
graph/XLA outputs respectively (value exactly; ESS within 7e-15). The flow/time
repair is not the discrepancy source. The very first reset already reports
`marginal_valid=False` and `reset_valid=False`, while the old public
`program_valid=True` discards reset validity. The LM system condition proxy is
201, so do not call this severe ill-conditioning without stronger evidence.
Next one 300-second CPU precision diagnostic evaluates exactly those frozen
reset inputs in FP64 eager/graph/XLA and compares all three FP32 arms against
it. This does not authorize runtime dtype migration, numerical gate relaxation,
or relabeling the failed comparison as a pass. Preserve the reset diagnostics
gap for a direct additive observability repair; exposing flags is Class A.

03827 confirms precision sensitivity: FP64 eager/graph/XLA reset clouds agree
within 9.8e-15. The old FP32 eager first reset is 2.2765e-5 away from FP64;
graph is 5.7607e-6 and XLA 1.4092e-5 away. Thus the old eager result is not a
more accurate authority for this case. This explains the observed difference
but leaves the declared raw equivalence veto open. Do not globally increase
tolerances or silently switch the reset to FP64. Next run the unchanged
19-check GPU qualification (excluding these two explanatory diagnoses), and
retain explicit backend-specific qualification/failure. CPU diagnostics use
intentionally hidden GPUs. Remaining unit budget includes this 300-second run.

03828 GPU: 17 checks pass and two fail. Dual-trust value differs by 3.0702e-5
against the 1e-6 gate. The standalone FP64 flow differs by 1.7860e-8 against
its stricter 1e-12 primitive gate, despite the complete non-dual value cases
passing. Localize the latter before accepting the shared flow change: one
300-second GPU diagnostic over substeps 1/2/3/7, full flow outputs and explicit
Python-cast versus native-lambda fractions. No tolerance change or integration.

03829 identifies an actual coefficient bug in the proposed rewrite: GPU XLA
erases the double -> float -> double cast pair under excess precision. For
3 substeps, the candidate evaluates 1/3 as 0.3333333333333333 while the old
Python tf.cast produces 0.3333333432674408. Two substeps agree to 1e-16;
3/7 differ at ~2e-8. Fix the finite-program mismatch by explicitly rounding
the binary64 quotient significand to binary32 nearest-even with uint64
operations. Stage fractions are nonnegative and bounded by one with int32
denominators, so no subnormal/overflow case is needed. Reuse this helper in
both flow and annealing, and require exact fractions over denominators
1/2/3/7/24/4097/2147483647 before repeating CPU/GPU full tests. Existing gates
stay fixed. Add the already computed reset validity and LM condition to the
native output as reporting-only fields; preserve legacy program_valid and
emit a separate numerical_valid flag requiring every reset to pass. No
conditioning threshold is invented and no invalid reset is silently admitted.

03830/03831: each backend passes 25 checks; each retains the same single
dual-trust full-value veto (GPU value 1.68284e-5; CPU final ESS 2.53059e-5).
All standalone flow and exact-fraction checks now pass. Reporting fields are
compared against captured old reset flags; numerical_valid is separate from
the inherited, weaker program_valid. No public wrapper integration follows.
The flow-cost unit later changes only uint64 -> int64 rounding intermediates
to support non-XLA GPU AddN. Close this source version with CPU/GPU component
groups (38 checks each: exclude only the explicitly retained dual-trust gate
and explanatory localizations/costs), plus the policy group. Do not describe
these component passes as passing the complete endpoint gate. The guard adds
five sources (243 total) with unchanged 1346 exact allowances and no new
numerical-loop/NumPy exemption. These three final workers fit the original
24-worker/7200-second endpoint unit.

## Dynamic score-owner repair and skeptical review

Recovery through03858 found a flaw in the mechanics prototype: a model passed
before lazy tracing can capture a stale mutable direction. Replace this unused
prototype with a pure `model_builder(theta, direction)` evaluated inside the
owned graph; both inputs, initial clouds/covariances, their directional tangents,
noises and observations must be explicit tensors. Static configuration may not
stand in for these dynamic operands. Reject trace formatting and unsupported
annealing before execution. Preserve the shared hand-derived recursion.

Use a fresh CPU score-only group, then GPU if CPU qualifies, within the remaining
seven endpoint workers (17 of24 used through03858). Each has a300-second bound.
Reference: score wrapper/executor frozen at9d8202b77 with identical dynamic
operands and independent five-point value differences, steps1e-3 and5e-4.
Test direction changes/zero/replay, changed theta and initial-law tangents,
one signature and enclosing HLO/no callbacks. FP64 frozen-source value/score
agreement uses atol=rtol1e-9; finite differences use atol=rtol2e-6. Include both
no-reset derivative diagnostics (not likelihood estimates forT>1) and healthy
Contract-E finite-program checks, without scientific admission or retuning.
A failed derivative or reset diagnostic vetoes the affected candidate and
triggers bounded localization; do not relax gates. A compiled self-comparison
or one finite scalar does not establish derivative correctness. Model builder
purity is an explicit caller obligation; this factory cannot make arbitrary
external mutable Python configuration safe. Current actual consumers remain
unqualified until executable wiring checks, downstream numerics and costs pass.

The skeptical review rejects the mutable-capture baseline and the unsupported
construction-time freeze claim. Frozen-source and finite-difference authorities
answer different risks; replay alone cannot detect a wrong derivative. Preserve
all failed attempts. One final full policy worker follows settled runtime edits.
The full-value dual-reset veto, Younis KDM host recurrence, and endpoint memory
remain independent blockers. No algorithm substitution or admission is proposed.

03859 CPU passes eight score checks, including independent five-point derivatives
at both steps and changed directions with one XLA trace. A reporting defect used
an unset output environment variable, so successful JSON/HLO details were not
saved (JUnit retains the passes). Fix the path to the runner's JUnit directory;
repeat once onCPU for durable numerical residuals, then qualify GPU. No runtime
or numerical threshold changes. This uses workers19 and20 of24; policy remains
reserved after code settles.

03860CPU and03861GPU each pass eight score-only checks with saved JSON/HLO.
CPU maximum finite-difference error is4.14e-12; GPU residuals are independently
saved. Theta, direction, initial law and other operands change with one trace;
zero/mixed directions and replay pass. This qualifies the new owner interface
for the stated diagnostic fixtures only. Existing consumer migration/cost gates
remain open. The policy scope now adds only this owner factory (244 sources,
1346 existing allowances), leaving the rest of the score module's mixed trace
formatting/consumer audit explicitly open. Run the full policy group next;
this is worker21 of24. Four prior unmetered diagnostics are conservatively
charged120CPU seconds and excluded from qualification.
