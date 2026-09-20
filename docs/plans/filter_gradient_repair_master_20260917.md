# Complete filter and gradient execution repair

Status: executing on `repair/filter-gradient-xla-validation-20260918`; merge is gated.
Owner request: repair all findings in the September 17 audit, compare memory
and performance before/after, review this program, and merge only when tested.

Factor-status checkpoint through 01679: all 40 CPU and 40 GPU synthetic
precedence/boundary/enclosure cases pass (01675/01676), all 14 original/full
fitter CPU records pass (01677), all 223 affected GPU consumers pass (01678),
and all 67 policy/controller checks pass (01679). Invalid eigenvalue rejection
matches frozen f3f47f76 on both CPU/GPU after making positive comparisons
explicit; 01674's failed NaN reduction attempt remains preserved. No threshold,
optimizer, covariance arithmetic, derivative or random stream changed. The
public record now materializes a native tensor status and loading norms.

No worker is active. Charged GPU 31,795.749108093 / 187,200 and CPU
39,841.218343504 / 115,200 seconds. The partial policy guard covers 193 sources
with 1,276 exact exceptions. This small status dependency has no standalone
speed claim; its cost remains included in the forthcoming enclosing-factor and
terminal comparisons. Commit/push before installing the ordered-attempt draft.
Main and every terminal finding remain unmerged/open; the inherited structured
D5 graph/XLA comparison is still unresolved.

01675/01676 pass all 40 native-status cases on CPU/GPU, including mixed/all-
NaN eigenvalue rejection, original NaN condition/holdout comparisons and exact
nextafter boundary decisions. 01677 passes all 14 original full-fitter CPU
records. The 223-case GPU consumer group is active in 01678; preserve sources.

Next ordered-attempt draft is `/tmp/sequential_attempts_tf_draft.py`, with
independent recurrence checks drafted in `/tmp/test_sequential_attempts_draft.py`.
Neither is installed or tested. Its contract is unchanged first/second factor
order, no second fit after first acceptance, one exact scalar callback per
usable factor, last-proposal reporting when a later fit is unusable, best finite
exact-incumbent promotion independently of acceptance, unchanged radius and
stall updates, and ignored unused history rows. The second-fit callback must be
native and later qualified with the actual padded fitter; synthetic controls
alone cannot close that integration. No wrapper fallback to the host loop is
allowed. Review the draft against the original recurrence before running it.
Same focused CPU/GPU driver limits and remaining campaign budget apply.

01674 passes 36 and fails two synthetic all-NaN eigenvalue records: XLA
minimum can discard NaNs and yield +inf, unlike the frozen CPU rejection.
Replace min(eigenvalues)>0 with all(eigenvalues>0), which is equivalent on
finite inputs and rejects NaNs explicitly. Add mixed-NaN eigenvalues and
check frozen CPU/GPU behavior before claiming preserved rejection semantics.
No eigenvalue is repaired, no threshold changes, and valid fits are unchanged.
The failure is preserved; proceed with the bounded CPU/GPU decision checks.

Pushed proposal checkpoint: f3f47f76. Native factor-status implementation is
now the next bounded dependency. Tests will compare every materialized field
with f3f47f76 on synthetic healthy and conflicting-invalid fit records, including
nextafter condition/holdout thresholds, nonpositive/NaN eigenvalues, missing
Jacobian condition, both factor counts, and optimizer-failure precedence.
CPU/GPU tests must also enclose the helper with changed runtime inputs and
verify graph-reference equality. The unchanged original fitter and consumer
suites still gate integration. Numerical status only moves into XLA; the public
record stays host-side and has no numerical NumPy operations. No optimizer or
covariance formula changes. Skeptical review preserves NaN comparison semantics
instead of silently tightening the policy in an execution-only repair. Existing
300/900-second bounds and total budgets apply; no standalone speed claim is
needed for this small branch helper before enclosing escalation measurement.

Proposal checkpoint through 01673: all 22 complete focused/history cases pass
on each CPU/GPU; all 20 trust/preparation, 40 sequential, 43 block and 12 factor
GPU consumers pass, plus 67 policy/controller checks. Six fresh cost processes
pass: original/XLA outputs are exactly equal at D=3/5, graph/XLA maximum error
2.221e-16. Warm original/XLA medians are 3.875/1.820 and 4.126/1.685 ms;
XLA host peaks are 11.535/12.031 MiB lower, and GPU peaks rise from 15,104 to
21,504 bytes. No declared cost trigger fires. Graphs remain 264 nodes at both
sizes, with one trace and eight unchanged runtime operands; twenty warm calls
retain constant device allocations and at most 24 KiB host growth. The exact
analysis and `proposal-memory-comparison-01673.json` are preserved beside runs.
These are descriptive single-process dependency costs, not final repeats.

No worker is active. Charged through 01673: GPU 31,544.737419723 / 187,200 and
CPU 39,670.321654545 / 115,200 seconds; remaining GPU 155,655.262580277 and CPU
75,529.678345455. Runtime sources did not change during cost measurement. The
partial guard passes 192 sources / 1,276 exact exceptions, with no new exception.
Focused Ruff and whitespace checks pass. This checkpoint can be committed/pushed;
main remains unmerged. The structured D5 graph/XLA comparison remains open.

Next dependency is native factor-fit rejection precedence and numerical report
fields, enabling factor escalation to consume tensor status. A scratch draft
at `/tmp/filter_repair_factor_decisions_tf_draft.py` is not installed. Preserve
original invalid-domain precedence, finite/SPD/condition/rank/holdout/optimizer
status order, boundary comparisons, and missing-condition semantics, including
original NaN behavior; no numerical safeguard or threshold change is authorized.
Compare synthetic conflicting failures/nearest floats to frozen materialization,
then all original records/consumers before enclosing escalation. A status helper
alone will not complete the outer controller.

Earlier checkpoints follow.

Proposal qualification through 01667 passes: all 22 focused/full-history cases
on CPU and GPU against the completely pinned b3334646 dependency (01665/01666),
20 original trust/preparation, 40 sequential, 43 block and 12 factor GPU cases
(01658/01659/01662/01663), and 67 policy/controller checks (01667). No failed or
skipped cases. The six-arm `proposal_memory` matrix is active from 01668; keep
runtime/tests/driver frozen until it completes. The graph/XLA structured-fitter
blocker and all broader terminal work remain open. Main is unmerged.

01660/01661 pass all four complete public-history cases on CPU/GPU. Original
block and factor consumers pass all 43/12 GPU cases (01662/01663), and all 67
policy/controller tests pass (01664), including invalid-batch preflight. Final
baseline review pins b3334646's trust solver source too; its changed candidate
signature had retained identical default arithmetic, but the oracle should not
import that touched dependency. Rerun the 22 focused/public cases on CPU/GPU
against the fully pinned dependency, then the six cost arms. No other runtime
code changed after the consumer checks. Source guard: 192/1,276, partial scope.

Proposal cost contract, conditional on unchanged public-history and consumer
checks: six fresh GPU2 processes compare the complete b3334646 proposal
subprogram against candidate graph-reference and XLA at D=3/5. The original
trust solver remains XLA as in the frozen source; classify that arm as its
original mixed host/compiled route, not a non-XLA baseline. Candidate graph must
have no nested XLA functions. Use the identical deterministic nonlinear target,
center/score/scales/precision, radius .2 and original acceptance parameters.
Measure one cold and 20 synchronized warm calls with every output materialized,
separate construction/RSS/HWM and allocator snapshots, plus trace/HLO after
measurement. Compare all numerical fields at 1e-10 and every boolean exactly;
thresholds, 80/80 trust iterations, and target formula remain fixed. Memory
256MiB/2x and timing triggers retain their roles. This small dependency comparison
is descriptive and cannot replace final whole-controller/repeated evidence.
Skeptical review: duplicate after/XLA arms would measure the same program, so use
three genuinely distinct routes; freeze the original trust dependency as well
as its host wrapper before execution. Same 300-second bounds and total caps.

01658 passes all 20 original preparation/independent trust checks on GPU;
01659 passes all 40 sequential GPU consumers including the unchanged
transactional score-policy assertions. The new batch accidentally used
`factor` instead of registered `factor_geometry`; pause allowed 01659 to finish
before any unknown-group dispatch. Correct that name and validate every selected
batch group before launching any worker. A focused regression enforces this.
No numerical worker failed and no numerical tolerance changed. Continue the
remaining public-history CPU/GPU, block/factor GPU and policy groups without
repeating the unchanged completed consumers. The four pinned-history cases
must actually evaluate a proposal and compare every final/history field.

01656/01657 pass all 18 focused proposal cases on CPU/GPU, including exact
near-boundary decisions, ordered scalar callback counts, graph-reference parity,
and eight unchanged runtime HLO operands. Integrate the native proposal into
the existing public refinement loop, preserving every reported field and None
marker. The original transactional test's fixed .01 step / .00995 prediction
now enters at the native solve factory; its assertions and numerical fixture
are unchanged. Outer radius/incumbent and factor-escalation logic are still
open. Recheck all original trust, sequential, block and factor consumers plus
policy, then compare complete frozen/public histories before cost measurement.

Pushed execution checkpoint: b3334646. Next independent dependency is native
trust-region proposal and exact acceptance; the inherited fitter graph/XLA
blocker remains open and does not authorize a tolerance change. Before public
integration, compare the new stable-signature program with b3334646 across
D=1/3/5, boundary/interior steps, fractional/resolvable/disabled score gates,
nonfinite callbacks and nearest-representable score/ratio boundaries. Record
exact scalar callback count/order, fail without eager retry, and require all
eight numerical inputs in unchanged enclosing HLO after changed values. The
explicit graph reference uses tf.linalg.eigh; default XLA retains the existing
binary64 Jacobi eigensolver. No trust radius, thresholds or solver iterations
change. Existing independent trust-region tests remain required. Run CPU then
GPU under the existing 300-second focused limits; only then wire the public
proposal, rerun its full history/consumer tests and measure matching costs.
Skeptical review: scalar target fusion and score-boundary rounding could alter
acceptance, so decisions must match exactly; numerical field tolerance stays
1e-10. A helper alone does not close factor escalation or outer refinement.

Checkpoint review through 01655: all 66 policy/controller tests pass after
registering the cost diagnostics. The source guard passes 191 sources / 1,276
exact exceptions; no numerical-loop exception was added. New runtime/tests and
analysis scripts pass focused Ruff; whitespace checks pass. No numerical source
changed after the 441-case qualification. The inherited graph/XLA comparison
remains explicitly open. Commit/push preserves this execution checkpoint and
its unresolved terminal evidence; main stays unmerged.

Recovery through 01654: no numerical worker is active. The 441 focused
qualification cases remain passing. Default XLA before/after and standalone/
enclosed records pass at D=3 and D=5. The eight-arm comparison is preserved in
`structured-memory-comparison-01650.json`; D=5 graph/XLA fails 23 numeric fields
at the unchanged 1e-10 tolerance and is **not accepted**. 01651 crosses prepared
clouds with both fitter modes; 01652 proves the same eight numeric fitter gaps
on identical data in frozen f06fd505 and current compact source. Same-mode
before/after records pass. No tolerance, optimizer or method is changed.

The D=5 resource trigger is explained by two 33-branch compact-shape dispatches
(CPQR initializer and Jacobian QR), versus two five-branch dispatches at D=3.
Fixed-input public host overhead is 446.820 MiB, GPU peak 97,024/195,840 bytes,
and cold 13.883/30.541 seconds. Over six changing active sizes, 01653/01654
retain all records while old/new first-sequence times are 70.215/29.857 seconds
and host peaks roughly 2.79/1.97 GiB. New sizes cost about 11 seconds each in
original compact fitting and about 0.11 seconds in the repaired fixed-capacity
path after the first call. Second-sequence host growth stays below 0.1 MiB.
`structured-eligibility-cost-disposition-01654.json` and its exact analysis
script preserve inputs, source hashes, HLO branch counts and full comparisons.
This is a bounded, explained compilation tradeoff, subject to final repeats;
it does not establish a statistical ranking or arbitrary-capacity memory bound.

Budget through 01654: GPU 30,927.589910224 / 187,200 and CPU
39,575.574552332 / 115,200 seconds (remaining GPU 156,272.410089776 and CPU
75,624.425447668). The extension is counted once. Commit/push the execution
checkpoint with the graph/XLA blocker explicit; continue independent proposal/
outer-controller repair without calling the checkpoint terminal-qualified.
Main remains unmerged. No F01--F20 terminal disposition is closed.

Earlier checkpoint details follow.

Recovery through 01650: the eight cost workers pass individually, but the
independent complete-record comparison rejects D=5 graph/XLA: 23 fields exceed
the unchanged atol=rtol=1e-10 gate. Before/after XLA and public/enclosed XLA
pass at both dimensions. Preserve the failed comparison; do not call this
checkpoint cost-qualified. D=5 public host peak grows by 468,525,056 bytes,
device peak by 2.018x and cold time by 2.200x, so resource investigation also
remains open. Charged GPU 30,697.813371662 / 187,200 and CPU
39,575.574552332 / 115,200 seconds. No worker is active at recovery.

Next bounded diagnostic separates graph/XLA preparation from same-input fitting.
At the identical D=5/capacity32 measurement inputs, retain both complete prepared
records, cross both prepared inputs with graph/XLA fitters, and compare the
original compact fit using identical active tensors. Preserve all 200 optimizer
iterations, seeds, dimensions and tolerances. The artifact records every fit
field and first differences; instrumented results only locate the discrepancy
and cannot replace uninstrumented qualification. A 300-second GPU worker uses
the existing driver, growth/idle checks and cumulative budget. Skeptical review:
same seeds do not guarantee identical rounded QR clouds across compilers, and
optimizer sensitivity can amplify tiny preparation or factorization changes.
This crossed-input diagnostic distinguishes those explanations without changing
the required final record gate. No algorithm/optimizer repair is assumed yet.

01651 localizes both preparation and fitter effects. Same-mode compact/padded
records agree; graph/XLA also differ on identical prepared inputs. Verify the
same fields against frozen f06fd505 graph/XLA code before attributing the gap to
this repair. The failed total comparison remains mandatory and unwaived.

Resource follow-up: two fresh GPU2 processes compare original/current public
preparation and fitting at D=5/capacity32 over eligible-row counts 1,0,3,8,16,32,
then repeat that exact sequence. Repeated copies of the same nonzero eligible
row isolate shape/compilation effects while retaining original half-cloud
weighting and all 200 iterations. Compare every public field/count and input
hash at unchanged gates. Record first-use/reuse times and allocator/RSS/HWM at
each call. No fixture rejection is hidden; any changed record triggers repair.
The question is whether the upfront branch graph cost avoids repeated compact
shape compilation; stable warm memory alone cannot answer it. Run the bounded
`structured_cost_investigation` matrix at 300 seconds per worker, sequentially
under existing caps. These finite/single-process observations cannot establish
statistical speed ranking, unrestricted capacity, or terminal acceptance.
Skeptical review: repeated rows are an explanatory shape control, not additional
scientific coverage; all original qualification fixtures remain mandatory.

Prior qualification through 01642: all 441 expanded qualification cases pass, including
26 preparation/23 integration CPU and GPU, 223 GPU consumers, 40 sequential GPU,
14 original/full fitter CPU, and 66 policy/controller cases. No failed/skipped
JUnit cases in these groups. The reviewed eight-arm memory batch is active from
01643; runtime/test sources remain fixed. Charged through 01642: GPU
30465.160898087 / 187200 and CPU 39575.574552332 / 115200 seconds.
The compact-record adapter and non-power-of-two reuse are qualified. No main
merge or terminal F01--F20 acceptance follows from this dependency checkpoint.

01638 passes all 23 expanded GPU integration cases; 01639 passes all 223
GPU consumers, including the unchanged compact-array assertion. An independent
100-digit Decimal reconstruction in
`structured-position-rounding-disposition-01638.json` checks all 36 coordinates
from 01623: originals equal separately rounded multiply/add, while the old
candidate equals a single final rounding of the exact product-plus-add. Six
coordinates differ. This supports contraction as the mechanism without claiming
machine-code inspection; 01624's native two-stage repair restores all outputs.

01634 passes all five focused GPU boundary/three-row checks. The refreshed
qualification matrix is active through 01638: 01635 passes 26 preparation CPU
cases, 01636 passes 23 integration CPU cases, and 01637 passes all 26 preparation
GPU cases. Preserve sources until the current matrix and planned cost batch
finish. The compact public view is separated from fixed-capacity runtime data;
no pre-existing array-shape assertion was weakened.

Next dependency after qualification and cost: the factor proposal/escalation
and outer sequential controller still use host numerical decisions. A scratch
proposal-only draft at `/tmp/filter_repair_sequential_proposal_tf_draft.py` is
not installed or validated. Before using it, compare frozen trust-region steps,
exact callback order/counts, finite/acceptance/score-resolution boundaries,
radius changes and best-exact-incumbent promotion independently of model
acceptance; include nonpositive predicted improvement and optional score gates.
Use original shape/seed/tolerance/derivative rules and diagnose any changed
rounding. Keep scalar exact proposal callbacks distinct from batched training.
The full outer loop, terminal-fit attempts/budgets/history and block/quadratic
controllers remain separate required work; a compiled proposal alone does not
close them. Review found no target, tuning, or sampler-policy change in this
next preparation dependency.

01629/01631 pass 22 CPU/GPU preparation cases; 01630/01632 pass all 19
CPU/GPU integration cases. Consumer 01633 passes 222 and fails the unchanged
compact-record boundary assertion. Preserve its original compact arrays via
host record views while passing the raw fixed-capacity tensor record directly
to the compiled fitter; numerical decisions never use those views. This fixes
the interface, not the test. Add non-power-of-two reuse (three rows) to exercise
runtime 0.5/count weighting. Requalify full records and consumers. Benchmark
review also removes an unused candidate graph from the original public arm's
setup before any timing is collected; no baseline work may be inflated.

01628 passes enclosing HLO/runtime inputs, changed-occupancy parity, original
preparation pullbacks, frozen fitted derivatives, post-eviction use and graph/
resource release. Run `matrix --stage tests --test-batch structured` sequentially
with sources frozen: all preparation and full-fit CPU/GPU cases, 223 GPU
consumers, 14 full fitter CPU records, and policy/controller checks.

Structured cost contract, conditional on correctness: eight fresh GPU2 workers
compare f06fd505/current public preparation-plus-fit and current enclosing
graph-reference/XLA at D=3/capacity4 and D=5/capacity32. Use identical healthy
factor-covariance inputs, one eligible reused row, unchanged seeds and all 200
optimizer iterations. Record all public fields, evaluation counts, same input
and source hashes, host RSS/HWM and allocator bytes at construction and one
cold plus twenty complete warm calls. Inspect HLO/trace counts after timing.
Graph-reference is explicitly non-default. Before/after and graph/XLA records
must pass unchanged 1e-10 gates, discrete counts exactly; only the recorded JIT
mode is an expected field difference. Same-mode repeated results must be exact.
Investigate 256 MiB/2x memory or timing triggers and growing warm allocations.
Use `matrix --stage tests --test-batch structured_memory --test-timeout-seconds
300` under unchanged cumulative caps. These are descriptive checkpoint costs;
three-process terminal repeats and enclosing outer-controller comparisons
remain required. Review: frozen original methods, equal synchronization/output
lifetimes, no optimizer shortcuts or withheld failed arms; padding cost remains
in total overhead. No independent reviewer or statistical ranking is claimed.

01627 excludes the test tape as the retention cause. Installed TensorFlow
`python/ops/custom_gradient.py:487--524` registers a process-level gradient
closure that retains the traced result and custom pullback. The affine custom
gradient was traced inside the resource-owning consumer. Bind that small,
resource-free affine map in its own fixed-signature function under init_scope,
before consumer tracing; keep its numerical two-stage loop and exact pullback.
Remove the unhelpful identity-barrier trial. Require the same consumer graph
and resource to release, while original derivatives/records and HLO stay fixed.
This addresses graph ownership, not TensorFlow's separate executable cache.

01626 passes original preparation pullbacks, no-optimizer derivative, all
runtime HLO inputs and changed-occupancy reuse. The final consumer-graph release
check fails after an external tape was used. First release the test's tape
object and repeat; if retained, inspect native custom-gradient registration and
scope its identity/affine pullback graphs outside resource-owning consumers.
The lifetime gate remains mandatory; successful cache-eviction calls alone do
not prove resource release.

01624 restores exact preparation stages. 01625 passes all 18 complete/rejection
CPU cases, including healthy D=3/5 records, but returning differentiable
preparation under an external tape exposes XLA TensorList boundary support.
Use the exact pullback of positions = center + fresh*scale around the two native
rounding stages: dFresh=upstream*scale, dCenter=sum(upstream),
dScale=sum(upstream*fresh). This preserves the same mathematical derivative and
avoids retaining a loop history for a two-operation affine map. Qualify against
original preparation pullbacks, plus the geometry-only frozen derivative and
resource lifetime/HLO checks. No optimizer AD or algorithm derivative is changed.

01623 identifies position construction as the first discrepancy (5.55e-17).
At identical positions, inlined/standalone callback scores and score scaling
are exact. The original separately rounded multiply/add became contracted;
barriers alone did not preserve rounding. Test two native loop stages for the
same multiply and add, keeping arithmetic, derivative and fixed shape unchanged.
This trial must restore full records and retain runtime operands; do not add
backend flags or alter tolerances to accept fused arithmetic.

01622 retains the same two condition-number discrepancies; the first barrier
trial does not repair them. It also exposes missing AD registration on the raw
XLA barrier when differentiable preparation is returned. Wrap that mathematical
identity in an exact identity pullback; do not freeze previously differentiable
preparation. Next compare original/candidate positions, standalone versus
inlined callback scores at identical positions, and score standardization to
identify the first differing operation. Keep this instrumentation diagnostic.

01621 isolates the discrepancy to recomputed scores (maximum 2.22e-16):
padding the exact original data preserves all fit fields. Native preparation
had fused across the original host displacement/addition and target-output
boundaries. Introduce numerical no-op XLA barriers at those boundaries, then
rerun complete record gates without tolerance changes. Preserve frozen exact
winner outputs and the fit's post-validation derivative boundary. The frozen
D=5 fixture is an invalid-trial rejection, not a healthy fit; apply the already
qualified rejection-count semantics and add healthy one-/two-factor fixtures.

01620 passes seven CPU cases but complete integration is not qualified. The
new D=5 precision fixture also rejects in frozen source (invalid optimizer
trials), so its healthy-geometry assumption was false; retain rejection checks
using the already reviewed count semantics. Two D=3 Jacobian condition records
near 1e14 differ beyond the unchanged tolerance although fit status and optimizer
counts agree. Localize with original inputs padded directly versus recomputed
preparation, preserving full 200-iteration settings and all field gates. The
combined graph returns differentiable preparation beside frozen fitted geometry;
its tape returns a zero through those extra outputs. Check geometry-only output
separately and retain the original no-optimizer-derivative contract. No numerical
threshold or failed record is waived; memory qualification waits for resolution.

01618/01619 pass all 21 GPU/CPU preparation checks. Tensor gathers preserve
all seven runtime inputs and changed-data HLO; the callback counter repair
passes original ordered scalar/batched call counts. Integrate the padded fit
with an explicit native finite-input/active-weight gate and shared public-record
materializer. Keep the compact internal-schema adapter for existing prepared
callers; it still uses the existing XLA fitter. New preparation uses fixed
capacity and tensor active count. Test complete fresh/reused fit records,
nonfinite and dimension rejection, frozen derivatives, combined preparation/fit
HLO and resource lifetime. Do not claim the enclosing outer controller repaired.

01617 identifies specialized center/scale/radius operands at data-dependent
winner slicing. Use tensor gathers for that same index, as in the earlier
factor-input repair. Verify all seven inputs plus changed center, score, scale,
radius, seed and search data reuse unchanged HLO. The selected values and
finite/first-tie policy must remain identical to frozen compact preparation.

01616 ran on GPU (manual CLI defaults to GPU despite the group name). It
passes 18 preparation cases and exposes two harness errors plus a compiler
input issue: the int32 callback counter was placed on CPU; use int64 for the
GPU-compatible diagnostic counter. HLO has four operands instead of seven,
so input-specialization qualification fails even though changing reuse rows
preserves numerical results. Preserve HLO and identify missing operands before
integration; one trace alone is insufficient. CPU retries must explicitly pass
`--device CPU`. No runtime numerical tolerance or source algorithm changes.

September 21 continuation after pushed f06fd505: start structured preparation
qualification. The frozen compact oracle is f06fd505; it preserves the prior
cloud/target semantics while including the just-qualified domain guard. Compare
all preparation fields at capacities 4/32 and dimensions 3/5, zero/partial/full
reuse, scalar/batched callbacks, nonfinite rows and exact radius boundaries.
Explicit call counters check ordered scalar calls and one batched callback;
nontraceable callbacks must fail without eager retry. Changed eligibility must
reuse one trace with all seven data operands in unchanged HLO. Only then wire
the padded fitter and compare complete records. Skeptical review: a fixed-capacity
cloud alone could mask row reordering or altered fresh/reused weights; compare
the compact active prefix and every count, and test inactive sanitation. The
unchanged 300/900-second worker bounds and cumulative campaign caps apply.

September 21 qualified guard checkpoint through 01615: all 66 policy/controller,
38 combined GPU, 223 GPU consumer, two CPU and two GPU lifetime checks pass.
All 38 original CPU cases pass in three sequential fresh processes (14/18/6).
Eight fresh-process guard-cost arms pass at D=3/5 in graph-reference and XLA
modes. Every original same-mode field is exactly equal before/after; invalid
counts are zero on these healthy inputs. Graph/XLA complete records pass the
unchanged atol=rtol=1e-10 comparison. Analysis and its exact script are preserved
as `factor-guard-cost-comparison-01615.json` and
`analyze_filter_guard_memory_20260921.py` in the campaign artifact root.

Default XLA warm medians before/after are 22.090/21.847 ms (D=3) and
92.201/93.585 ms (D=5); added host peaks are 7,630,848/6,250,496 bytes and
GPU peaks are 80,896/84,224 and 100,352/102,912 bytes, respectively.
No guard-cost resource/timing trigger fires. Twenty warm calls show bounded
observed allocation fluctuations and one trace. These are single-process
mechanism/cost observations, not terminal repeats or performance rankings.
CPU executable retention across arbitrary distinct signatures remains an
explicit limitation, supported by the diagnosis below; no OS/package repair
or unlimited-process claim is made. No worker is active at this checkpoint.

Charged through 01615: GPU 29,038.68574661859 / 187,200 seconds; CPU
38,711.01263819063 / 115,200 seconds. Remaining: GPU 158,161.3142533814;
CPU 76,488.98736180937 seconds. Commit/push this qualified checkpoint, then
continue structured reuse preparation, enclosing controllers and the remaining
master gates. Main remains unmerged; no F01--F20 terminal disposition is closed.

Historical investigation sequence (preserved):

September 21 recovery through 01597: the native failure is LLVM executable
section allocation (`allocateMappedMemory: Cannot allocate memory`), before
any final-test numerical result. No worker is active. The host has about
200 GiB available and vm.max_map_count=65530 at recovery; these facts do not
alone establish the cause at the earlier failure. Next run the identical
38-case CPU sequence with one-second external /proc observations: mapping
count and anonymous executable mapping count, RSS/HWM, address space, threads,
system available/committed memory, process limits and cgroup memory limits.
Preserve observations in its numbered run directory even if the worker dies.
The comparator is 01595/01597's unmodified runtime/test sequence; the sampler
is explanatory and cannot qualify timing or waive a failing test. The run has
the existing 900-second ceiling and unchanged 32/52 CPU/GPU-hour caps. No OS
limit, package, numerical, optimizer or backend-default changes are authorized
by this diagnostic. A map count reaching the kernel limit at the same failure
would support map exhaustion; otherwise investigate the actual limiting
resource. Skeptical review: parent sampling survives a native worker crash,
avoids GDB overhead, and records resource limits rather than assuming that
ENOMEM means physical OOM. Required combined qualification remains blocking.

01598 repeats the failure after 276.738 seconds. Last live snapshot contains
65,021 mappings (limit 65,530), including 20,097 anonymous executable mappings,
with 194.8 GiB system available memory, unlimited address/data/RSS limits and
no cgroup memory cap/OOM events. The one-second sampler cannot capture the
instantaneous limit; this strongly supports executable-map exhaustion rather
than physical OOM. Next repeat all 38 cases, same order and process, releasing
the five factor/fitting/selection/stability factory caches and collecting Python
cycles only after each completed test module. Record maps immediately before
and after each release and external samples throughout. This tests whether
ordinary ownership release recovers compiler maps without an OS-limit change,
backend flag, numerical change or split process. Pass requires all original
tests plus demonstrable map release; otherwise the cleanup is not a repair.
This diagnostic has the unchanged 900-second bound. It says nothing about
runtime cache capacity adequacy; that remains a separate resource question.

01599 falsifies Python-cache cleanup as a repair: releasing all five factories
and collecting cycles changes neither 33,801 nor 64,785 mappings at the two
module boundaries; LLVM then fails again. Installed TensorFlow's
`include/tensorflow/compiler/jit/device_compilation_cache.h:62` explicitly says
the device cache owns executables, has no eviction policy, and grows without
bound. This matches the native stack, incremental mapping growth and failed
Python cleanup. Classify the combined CPU job as a compiler-resource stress
failure, preserved without waiver. Ordinary Python cache bounds do not bound
this backend cache. No system limit, compiler flag or package will be changed.

Qualification now requires the same 38 CPU cases in three fresh sequential
workers (14 full fitter, 18 padded, six domain), both CPU/GPU lifetime checks,
all 38 cases together in one default GPU/XLA process without cache cleanup,
and 223 actual GPU consumer checks. This is a bounded CPU reference profile
in response to a localized resource limit, not permission to omit a failed
numerical case. Long-lived CPU execution across arbitrarily many distinct
compiled signatures remains unsupported by this evidence. The full GPU test
and each CPU module are mandatory terminal jobs. No numerical/runtime source
changes follow from the allocation investigation. If any original case fails
under these profiles, repair it before interpreting cost. Run the fixed batch
with `matrix --stage tests --test-batch factor_guard`, default 900-second
per-worker ceiling; then `matrix --stage tests --test-batch factor_guard_memory
--test-timeout-seconds 300` for the eight already specified fresh GPU arms.
Both use the same approved driver prefix, idle preflight and verified growth.

Terminal-role review: `EXPLANATORY_TEST_GROUPS` in the driver lists exact
diagnostic jobs and their individual reasons. It covers only the factor-domain,
rejected solver, instrumented arithmetic and descriptive resource investigations
just reviewed against their test bodies and preserved outcomes. It does not
classify by name pattern, outcome, or the word localization. Current runtime,
complete-record, derivative, consumer and compiler-reuse gates remain required;
all new/unlisted jobs default to mandatory. Resource results still require the
ledger's explicit disposition and the independent terminal before/after matrix.
Regression tests verify this distinction, ensure new runtime failures stop the
matrix, and retain explicit execution of explanatory batches when requested.
This is the primary agent's review, with no independent review claimed.

Next numerical dependency after the guard checkpoint: enclose
`sequential_map_covariance._structured_factor_fit_data` and its reuse packing
with the qualified padded fitter. Generate the same two disjoint fresh clouds
with the unchanged `seed[1]+104729` holdout offset; preserve fresh callback row
order/counts, finite/nearby/nonzero reuse predicates, stable reused-row order,
and half-fresh/half-reused weights (all fresh when none are eligible). Carry the
active count as a tensor at fixed capacity; inactive NaNs must not enter a loss.
Keep strict finite exact-incumbent selection and first ties. Compare all original
preparation fields and complete fitter records to frozen compact execution at
zero/partial/full occupancy, including invalid reused rows and radius boundary
cases. Require one trace, runtime data operands and unchanged HLO when eligibility
changes. No new seeded stream, eligibility criterion, optimizer setting or
fitted-output derivative is permitted. GPU/CPU focused checks precede consumer
checks and matched fresh-process cost measurements at capacities 4/32. Outer
refinement, block sweeps and quadratic successive fits remain separate open
controllers; a compiled preparation helper cannot close them. The existing
bounded driver, fresh artifacts, 300/900-second ceilings and cumulative caps
apply. This dependency review found no sampler/tuner change; the HMC interface
and capability registry were consulted again during recovery.

Continuation through 01585: focused runtime guard checks pass on CPU/GPU
(01582/01583). Native fixed-center records now carry the rejected fit's anchors
and invalid-evaluation count. The new record check in 01584/01585 confirms the
same rejection status, empty geometry and all existing fields; its assumption
of equal new invalid counts across standalone/enclosing compilation is false
(216/225). These counts describe the continued, already-invalid optimizer
trajectory; neither route may expose its geometry. Check each count against its
own native result and compare all other rejection fields exactly. This changes
no pre-existing numerical gate or healthy-record criterion. Next qualify
compiler-input reuse, full original/healthy records, padded records and actual
consumers, then measure guard overhead. Main remains unmerged.

Guard resource diagnostic contract: compare frozen pre-guard 085baaaa and the
candidate in fresh GPU processes, separately for graph-reference and default
XLA, on healthy three-/five-dimensional one-/two-factor clouds. Use the same
deterministic inputs and full default optimizer settings, one cold call and 20
warm calls, with identical complete-record materialization and lifetimes. The
new invalid-evaluation count must be zero and is reported separately; compare
every pre-existing numerical field at the unchanged 1e-10 tolerance and discrete
counts exactly. Record input/source checksums, host RSS/HWM and allocator
current/peak before construction, after construction and after each call, then
inspect trace count and HLO after timing. Verify six tensor operands plus the
candidate's one int64 resource and changed-input reuse through the independent
qualification suite. Reserve at most 300 seconds per arm, sequentially under
the existing GPU idle/growth policy and cumulative caps. The ordinary 256 MiB,
2x memory and timing triggers remain investigation triggers. These eight
single-process observations explain guard cost, not terminal repeats or a
comparison against the original campaign baseline. A healthy-record mismatch
blocks this repair; no clip, tolerance, optimizer or RNG change is allowed.

01586 observes six/seven data operands plus one scalar int64 resource; the
compiler guard now checks exact types/shapes and capture identity while keeping
changed-data HLO equality. 01587 passes the six updated CPU domain checks.
Combined CPU qualification 01588 crashes inside TensorFlow execution of the
last enclosing-domain test after 37 completed checks (SIGSEGV, 274.048 seconds).
No passing suite claim follows. Before GPU qualification or resource timing,
replay the padded suite followed by that enclosing test in one CPU process to
isolate reuse/cache interactions. The 300-second ceiling and unchanged caps
apply. A process split cannot waive this crash; require a reproducer/cause or
record it as an unresolved blocker.

01589 passes all 19 padded-plus-enclosing-domain checks. The fixed-center
suite plus domain cases is the next independent call-history replay. Also
test a tiny enclosing fitter after evicting the 16-entry inner factory cache:
TensorFlow FuncGraph documents weak references to captured variables, so
resource lifetime is an explicit hypothesis. Inspect the enclosing variables
before and after collection, then require identical repeated outputs; CPU-only,
120-second bound, no numerical source change for this diagnostic. A reproduced
lifetime defect needs ownership tied to the enclosing callable, not an
unbounded global cache or a process-isolation workaround.

01590 passes all 19 fixed-center-plus-domain cases. The tiny 01591 lifetime
check fails deterministically: the enclosing graph's guard variable is deleted
after factory-cache eviction. Repair ownership at graph binding, retaining the
int64 resource in each enclosing FuncGraph's ordinary collection, never the
process default graph; keep the existing 16-entry factory limit. This Python
graph traversal is static ownership setup, not numerical iteration. Require
successful post-eviction calls, unchanged counts/values, one trace and release
after the enclosing graph is collected. Then repeat the original combined
CPU suite; the separate lifetime cause alone does not explain away its SIGSEGV.

01592 preserves the variable and post-eviction outputs but exposes the expected
two initial traces when TensorFlow creates a variable during first tracing.
01593 verifies stable one/two trace counts for prebound/lazy construction, but
the collection trial retains resources after deleting the consumer. Collections
are inherited by subsequently traced nested functions; replace them with a
namespaced owner attribute only on the outermost function graph. Require both
post-eviction survival and release of graph/resource after consumer deletion.
No global collection or numerical execution policy change is allowed.

Program review also found terminal-driver debt: `gate` and the tests matrix
currently require every registered explanatory trial to pass, including rejected
historical solver trials and deliberately failing full graph domain references.
Before terminal execution, classify those jobs explicitly as explanatory with
their preserved outcomes, while retaining current runtime, complete-record,
consumer and compiler-reuse tests as mandatory gates. Do not blanket-exclude
names containing "localization": several such groups are actual regression
checks. A reviewed exact classification plus controller regression is needed;
this is a harness repair, never permission to waive numerical failures.

01594 passes both resource-lifetime/release cases, but 01595 repeats the
SIGSEGV after 37 checks. Resource ownership is repaired independently; it has
not resolved the combined crash. Capture a native GDB backtrace of the same
CPU sequence under the existing 900-second full-suite ceiling and caps, with
network symbol fetching disabled and pytest's signal stack timer disabled to
let GDB observe the native fault. This is a debugging wrapper around the same
sequential worker, with no environment mutation or numerical change. Preserve
missing JUnit as failed evidence. GPU/performance work remains paused.

01596 was interrupted after 642 seconds because GDB spent about 97% CPU
loading shared-library symbols during every XLA compilation. The sampled stack
is compilation, not the reported crash. 01597 disables automatic shared-library
and Python symbol loading until the inferior stops, then loads TensorFlow symbols
for the native backtrace. Same numerical sequence and 900-second ceiling.

Current through 01577: compact-shape CPQR repairs the padded-fitter failure.
All 32 solver/derivative, 18 padded-fitter and 14 original/full fitter cases
pass independently on CPU and GPU. All 223 GPU consumer cases pass in 01575. The
32-spare representation costs 331 MiB extra host peak at the measured extent;
fresh graph/XLA controls explain branch graph/compiler overhead and show stable
warm allocations. Full graph runs reveal an existing loading-margin assertion
that XLA ignores, now a separate open correctness finding. Do not integrate
structured preparation or merge until its disposition and remaining gates are
complete. Detailed evidence and recovery are below and in the resume document.

Next after pushed `085baaaa`: localize domain-check semantics using the frozen
3582b4ac eager public fitter versus current graph and XLA on the exact failing
five-dimensional cloud. Preserve returned status, all diagnostics, covariance
and loadings; graph exceptions are observed failures, not timing evidence.
Also compare the public covariance constructor on healthy, exact-margin,
outside-margin and negative-scale inputs in all three modes. This tiny GPU
diagnostic has a 120-second bound, unchanged inputs/settings and cumulative
budget. It determines the required failure contract before a guard repair;
no assertions are bypassed, bounds relaxed or states clipped.

Skeptical review: an invalid optimizer trial can be followed by a valid final
state, so checking only final loadings does not reproduce the eager rejection
contract. A future XLA repair must propagate any forbidden evaluation to the
returned status and preserve all healthy original records and derivatives.
Do not change objective values or stopping rules simply to make parity pass.

01578 confirms the defect: original eager and current graph return
factor_optimizer_failed, while XLA continues and returns second_factor_unidentified
with finite matrices. The constructor's exact-margin, out-of-margin and negative
scale cases throw in eager/graph but return finite matrices in XLA. Test a
diagnostic-only covariance guard that returns a nonfinite tensor on the original
invalid domain, retaining the exact healthy value and pullback. Observe the
full trial's optimizer status rather than assuming poisoning one evaluation
records every invalid trial. No runtime change until healthy regression and
failure-propagation behavior are known. Same 120-second GPU bound and caps.

01579: the diagnostic constructor guard makes invalid direct calls nonfinite
and leaves healthy value/pullbacks exact. However the fitter still returns
second_factor_unidentified, confirming that poisoning one trial does not
preserve the original fail-on-any-invalid-evaluation contract. Do not install
that incomplete repair. Next observe every covariance evaluation with reset
int64 TensorFlow counters inside one XLA invocation, on the dimension-three
healthy fixture and dimension-five violating fixture. Preserve original
arithmetic and compare full instrumented versus uninstrumented records; repeat
each call to check reset/trace behavior. These resource counters are diagnostic
only pending evidence; they may not become a shared concurrent runtime state
by inference. Same 120-second GPU bound and cumulative budget.

01580's reset counters observe zero invalid evaluations out of 180 on the
healthy three-dimensional fit and 102 out of 400 on the violating five-dimensional
fit. Complete records remain exact and repeat resets/one trace pass. Before
using state for a runtime guard, check same-program concurrent invocations on
a healthy five-dimensional two-factor cloud and the violating cloud, plus
alternating sequential calls. The scientific target remains unchanged; this is
a tiny resource-isolation diagnostic, not parallel campaign workers. Runtime
int64 resource capture would be justified only if every invocation returns its
own reset count and records. Same 120-second GPU ceiling and caps.

01581 passes same-program healthy/violating concurrent and alternating calls:
counts reset per XLA invocation, complete records remain exact, one trace is
retained. Install an int64 invalid-evaluation counter in the compiled fitter
factory, reset at entry, observing the original covariance domain in each loss
and final evaluation. Preserve raw optimizer arithmetic; any recorded violation
must make public and fixed-center consumers return the original optimizer-failure
classification without exposing an admitted geometry. Graph reference keeps
immediate assertions. The public covariance constructor gains a nonfinite data
guard for invalid XLA calls while retaining eager/graph assertions. This is a
Class B failure guard, not clipping or a new numerical setting.

Before acceptance, test healthy exact values and pullbacks, direct invalid
inputs, failing public status, fixed-center rejection, alternating/concurrent
calls, original full records and compiler input reuse. Resource capture may add
one internal compiler operand; no data may become a specialized constant. A
healthy-record drift blocks this trial. The frozen diagnostic clones now read
085baaaa so future runtime changes cannot rewrite observed failure evidence.
Use 120/300-second focused checks and existing full-suite ceilings and caps.

Recovery after 01554: the enlarged CPU padded suite passes 13/18 cases, with
five complete-record failures at dimension five/two factors. The small- and
large-capacity partial occupancies differ in optimizer counts and covariance
(up to 5.478e-6). Passing standalone solver tests does not qualify the enclosing
fit. Preserve the uncommitted active-row implementation as an investigation;
do not integrate it into structured sequential preparation or allowlist it yet.

Next compare compact initialization, the pinned 7d08c68e shared-body diagnostic,
and the runtime active solver on identical frozen clouds at reused counts
0/1/4 and capacities 4/32. Observe initial raw state, normalized inputs and the
original objective/gradient; repeat the objective comparison with the compact
initial state supplied identically. This separates initializer changes from
shape-dependent loss arithmetic and from the runtime wrapper/sanitation.
Record every observation in a fresh numbered artifact through the sequential
CPU diagnostic driver (300-second bound), then inspect the first divergent
operation before proposing a repair. No formula, derivative, random stream,
optimizer setting, or tolerance changes. These are explanatory diagnostics,
not complete-record qualification. Skeptical review: observing intermediate
outputs can change fusion; any proposed execution-boundary repair must pass
the uninstrumented complete fitter on CPU and GPU and the original record gate.

Charges at recovery through 01554: GPU 26,948.367 seconds, CPU 35,200.426
seconds, under unchanged 52/32 process-hour caps. No worker is active.

01555: all four explanatory cases complete. Pinned shared-body and runtime
active initializer agree with each other but differ from compact by up to
1.599e-14 in the encoded state. At the same supplied state, gradients agree
exactly; a 2.168e-19 loss difference remains at the larger capacity. Localize
the compact/shared CPQR steps using identical predecessors. In particular,
the shared branch receives already-squared values whereas the compact reduction
can fuse multiplication and accumulation. Observe the first divergent norm,
reflector, product and update; compare a diagnostic-only branch that computes
squares inside its compact reduction. Capture optimized HLO. This is a bounded
120-second CPU explanatory diagnostic, not permission to change arithmetic or
qualify a fitter from intermediate observations.

01556 localizes the first discrepancy to the branch matrix-vector product,
not the squared norms: at identical predecessor state the RHS product differs
by 1.388e-17, followed by a matrix product difference of 6.939e-18. Moving
squaring into the branch has no effect. Optimized CPU HLO emits standalone dots
for compact operands and slice-dot fusions for the shared branch. Test an XLA
optimization barrier immediately after the branch's compact slices to preserve
the standalone dot boundary. Require exact same-predecessor step agreement,
then uninstrumented full-record CPU/GPU checks, without changing formulas or
adding a backend-specific branch. Same 120-second diagnostic limit and caps.

01557: the barrier is removed before CPU slice-dot fusion and does not repair
the discrepancy. Do not install it. Next bind the complete CPQR Householder
step to compact row shapes, while sharing the enclosing factorization loop,
rank decision, complete-orthogonal stage and pullback. This uses the original
formula/body inside each branch and avoids dispatching isolated sliced dot
operands. Compare the uninstrumented complete fitter at every failing CPU
occupancy plus full occupancy, six cases, within 300 seconds. Preserve branch
ordering, settings and exact discrete counts; a passing CPU trial still needs
GPU, solver/derivative, compiler-input and fresh-memory qualification.

01558: the single-step branch still changes four of six CPU complete records;
keep it diagnostic only. Widen the compact boundary to the CPQR loop while
sharing the size-independent complete-orthogonal stage, rank decision and
pullback. The loop's compact tensor arguments prevent sliced operands inside
its repeated products. This is the intermediate granularity between rejected
isolated-step sharing and the expensive full-COD branch. Repeat the same six
CPU records within 300 seconds; only a passing candidate proceeds to GPU and
fresh-memory measurement. Preserve failed attempts and unchanged tolerances.

01559 passes all six complete CPU records and exact optimizer counts with the
compact CPQR loop boundary. Install that execution structure, retaining the
two-tensor public API and the active-count entry's original full-rank pullback.
Preserve the rejected shared-dot source as a test-only historical reference;
later runtime edits must not silently rewrite the diagnostic experiments.
Run solver/pullback checks and the full padded suite on CPU/GPU before further
integration; then measure fresh compact/padded processes and input reuse.
The previous 01548/01549 pair is now analyzed in
`factor-shared-cod-memory-comparison-01549.json`: complete fields agree, warm
ratio 1.013, extra host peak 109,260,800 bytes and device peak ratio 1.663.
It describes the earlier diagnostic and cannot establish the new CPQR cost.

01560/01562 pass all 32 CPU/GPU solver tests; 01561/01563 pass all 18
CPU/GPU padded tests, including the larger capacity and exact optimizer records.
Fresh GPU XLA processes 01564/01565 preserve the records but show a new host
memory investigation trigger: peaks 1,545,408,512/1,892,716,544 bytes for
compact/32-spare (347,308,032 bytes extra), with cold calls 11.886/27.013 seconds.
Graphs contain 4,847/10,116 nodes and HLO 3,995,140/6,265,944 bytes. Device peaks
are 98,304/196,352 bytes. Do not call this an accepted cost yet.

Next measure four-spare XLA and compact/32-spare explicit graph-reference
processes with the same frozen cloud, cold plus 20 warm calls, output lifetimes
and memory stages. Retain complete numerical fields and compare within each
mode and across modes; a graph/XLA field discrepancy is evidence, not grounds
to drop a field. Register graph diagnostics separately and label their
non-default status. Use 300-second bounds for graph optimization and 120 for
the four-spare XLA arm, unchanged cumulative caps and GPU preflight. Inspect
static branch/node growth against post-cold allocation stability. The 256 MiB
rule is an investigation trigger, not permission to skip parity or accept a
general memory bound; record the actual cost and rejected lower-cost variants
before deciding whether this compact-shape representation can be integrated.

01566 four-spare XLA peak is 1,609,523,200 bytes (64,114,688 extra versus
compact), cold 13.538 seconds and warm 165.776 ms. The 0/4/32 capacities have
32/20/12 KiB post-cold host growth, consistent with compilation cost increasing
with static shape branches rather than continuing fixed-input allocation.
01567 compact graph reference fails the existing strict loading-row assertion:
an L-BFGS trial reaches 0.99999900000000019 against the frozen 0.999999 bound.
XLA logs that it ignores this assertion. Preserve the failure and report no
graph steady timing; do not disable the assertion, clip the state, alter the
optimizer or reinterpret it as a passing graph comparison. Complete the matched
padded graph arm to determine whether this is a capacity-specific failure.
Record the graph/XLA assertion-semantics gap separately from the active-row
repair; it remains an open consistency finding in the enclosing fitter.

01568 padded graph has the identical strict loading-margin failure as compact;
the failure is not introduced by padding. Retain both failed runs with no
steady graph timing. The diagnostic collector now records a numerical failure
artifact and an explicit expected-failure marker for this full graph arm;
XLA and all qualification tests still fail on any numerical exception.
For causal compiler-memory comparison add a separate four-iteration diagnostic
fixture at the same dimension/cloud/capacities, in fresh graph and XLA processes.
Four iterations are a predeclared measurement extent, not a change to the
runtime optimizer or its full-record gates. Record that extent in the artifacts
and never substitute its outputs for the failed complete graph comparison.
Use cold plus 20 warm calls, 120-second XLA / 300-second graph bounds and
unchanged cumulative caps. Require same-extent graph/XLA numerical comparisons.

The exact static row-shape binder is now allowlisted as fixed schema, supported
by 01560--01563; no Python numerical iteration is exempted. The memory cost and
graph assertion-semantics issue remain separate open integration findings.

01569--01572 four-iteration controls pass every field comparison within and
across graph/XLA modes at the unchanged 1e-10 gate. XLA compact/padded warm
medians are 30.899/30.736 ms versus graph 87.864/100.031 ms. GPU peaks are
96/192 KiB for XLA versus 8.12/8.25 MiB for graph. Padded-minus-compact host
peak is 169,861,120 bytes in graph mode and 349,372,416 bytes in XLA, showing
both graph branch overhead and additional compiler cost. Full-iteration XLA
records also agree and warm overhead is 2.02%; peak-device overhead is exactly
2x (196,608 versus 98,304 bytes), including the largest sampled temporary.
Analysis: `factor-cpqr-capacity-memory-comparison-01572.json`.

The extra static branch memory is explained at this extent, with 12--32 KiB
post-cold growth across 20 warm calls; this is no general allocation bound.
The cheaper isolated-operation and per-step candidates failed full records,
so compact CPQR is the current parity-preserving candidate. Retain the 331 MiB
host / 96 KiB device cost explicitly for terminal review and three-process
repeats; do not silently waive it or claim all memory investigations complete.
Run the original full fitter and downstream regression groups after this solver
change before checkpointing. Structured preparation remains unwired while the
newly observed graph/XLA assertion-semantics defect is investigated.

Continuation from pushed `7d08c68e`: test a diagnostic-only COD clone that shares
its CPQR/COD loops and dispatches only the active row shapes for column norms,
reflector norm and transposed matrix-vector products. Keep every pivot, rank
threshold, Householder update, projection, initializer, loss and optimizer rule.
The earlier full-solver shape clone 01536 is the explanatory parity/cost
comparator; compact current runtime remains the numerical authority. At the
same five-dimensional 11-active/42-capacity fixture, record complete public
fields/counts, cold time, host/device allocations and graph/HLO sizes. Reject
the clone as a runtime proposal if it loses parity. A passing primal diagnostic
does not qualify derivatives or the public COD API; those require explicit
checks before runtime migration. This trial is confined to the frozen
initializer, whose external derivative was already absent. Use the existing
300-second GPU ceiling, memory-growth/idle controls and cumulative caps.
Skeptical review: compiler fusion can cross branch boundaries; compare complete
records, not only solver residuals. Same-process RSS is explanatory and cannot
replace fresh-process resource qualification. No numerical tolerances or
algorithmic direction change is authorized, and no runtime source is modified.

Run 01546 stops before numerical execution because the diagnostic rewrite's
return anchor matches both full-rank and COD functions. Narrow the checked
anchor to the COD return before the appended helper, preserving the separate
full-rank routine. Retry this harness-only repair under the same ceiling and
budget; the failed run is not parity or memory evidence.

Run 01547 preserves every complete record/count using the shared-body clone.
Its graph has 6,392 nodes / 4,615,457 HLO bytes, versus compact 4,839 nodes /
3,994,045 bytes. Cold calls are 20.497/11.975 seconds; same-process memory is
still insufficient to classify the overhead. Measure compact/shared clones in
two fresh GPU processes on these exact frozen inputs, one cold and 20 warm calls,
with full-field comparisons and stage-wise host/device memory. Use 120-second
per-worker bounds and the unchanged 256 MiB investigation trigger. A passing
pair only nominates runtime migration; CPU/GPU occupancy and derivative checks
still precede any integration. The original full-COD-clone resource trigger is
not waived by this narrower design's descriptive timing.

Fresh processes 01548/01549 show compact/shared cold calls 11.828/20.912 seconds,
warm medians 167.136/169.254 ms, and about 104.199 MiB added host peak. Post-cold
host growth is 24/16 KiB. The shared-body candidate is below the unchanged
256 MiB investigation trigger at this extent. Install the optional active-row
representation in the shared COD routine, preserving its full-rank pullback;
inactive rows and their gradients must be zero, and invalid active counts must
not produce an accepted fit. Keep ordinary two-argument behavior unchanged.
Test CPU/GPU compact/padded values, rank thresholds, same-active-row gradients,
finite differences and HLO reuse, then factor records across occupancies and
the 32-row capacity. Runtime integration remains gated until those checks and
fresh-process costs pass. Pin the explanatory source clones to 7d08c68e so later
runtime changes cannot silently rewrite the comparison. No derivative removal
or rank-tolerance change is authorized.

Run 01550 passes 20/32 CPU cases including all six new active-row checks, but
direct `tf.function(complete_orthogonal_lstsq)` calls fail: TensorFlow expands
the optional None into a custom-gradient tensor operand. Keep the public
dispatcher plain and supply two tensor-only custom-gradient signatures to one
shared implementation body. This restores ordinary direct tracing and retains
the active-count gradient as None. Rerun the entire focused CPU group before
GPU or fitter qualification; no numerical formula or baseline is changed.

Run 01551 passes 31/32; compiler-IR inspection still requires the new optional
None argument. Preserve the existing public two-tensor signature exactly and
add a three-tensor active-row entry to the same implementation body instead.
This avoids changing downstream compiler-introspection calls. Both entries
retain custom pullbacks; the new active-row tests use the explicit entry.

Runs 01552/01553 pass all 32 CPU/GPU solver cases, including compact/active
values and pullbacks, inactive-NaN sanitation/zero gradients, rank changes,
invalid-count rejection, finite-difference derivatives and HLO reuse. Expand
the padded full-record suite to both four and 32 spare rows before accepting
the active initializer. Preserve zero/partial/full small-capacity occupancy
cases and the existing one/two-factor settings. Use 300 seconds CPU and the
existing 900-second GPU ceiling for the enlarged suite (the prior 12-case GPU
suite takes 152.6 seconds; six new factor cases include more shape branches).
This is an attempt reservation within the same cumulative cap, not new compute.

September 20 current checkpoint through 01545, based on pushed `375e7257`:
weight normalization and weighted-loss arithmetic now remain runtime operations
inside enclosing XLA calls. One optimization barrier on input weights repairs
all full CPU record failures without changing formulas, optimizer settings,
thresholds, derivatives or random streams. Full fitter suites pass all 14 CPU
and all 14 GPU cases (01521/01522), including the original 3582b4ac record gate.
All 223 affected GPU consumers pass (01541).

Small-capacity padded fitting also passes all 12 CPU/GPU cases (01542/01530).
Active Jacobian entries had been identical; zero padding changed QR rounding
near the rank threshold. A fixed-shape QR branch selected inside XLA preserves
the original compact factorization. Six/seven runtime HLO inputs and changed-data
executable reuse remain checked. The one new exact allowlist entry covers only
static QR shape binding; the numerical work stays inside TensorFlow/XLA.

The larger default 32-row reuse capacity remains blocked. Runs 01531--01533
show stable warm allocation and about 63 MiB added host peak, but optimizer
counts/records differ. Same-state gradients agree (01535), localizing the
problem to initializer shape rounding. A diagnostic-only compact COD dispatch
restores all records (01536), but its 32.055-second cold call and approximately
847 MiB added same-process RSS trigger further investigation. That COD clone
is not installed. Padded fitting is not wired into sequential preparation.
Next reduce duplicated COD compilation by sharing its control/body and binding
only row-dependent operations, or establish its actual bounded cost with fresh
processes. Preserve every field and tolerance.

Fresh changed-data memory comparison 01537--01540 confirms the compact repair:
checkpoint/candidate XLA median new-cloud calls are 6.403 seconds/9.911 ms;
post-cold host growth is 335,212,544/57,344 bytes. Cold XLA calls are 8.745/8.546
seconds; final GPU current allocations are identical at 10,496 bytes. All
measured public fields pass unchanged 1e-10 comparisons, including graph/XLA.
Artifacts: `factor-compilation-memory-comparison-01540.json` and
`factor-capacity-comparison-01533.json`. One process per arm gives descriptive
mechanism evidence; terminal source-frozen three-process comparisons remain.

All 63 policy/controller checks pass (01545). The partial guard covers 189
sources / 1,274 exact exceptions. Inventory 01544 finds 2,876 working-tree
Python files, 2,875 parsed and one unchanged external legacy parse error.
Focused changed-runtime/new-test Ruff and whitespace checks pass.

Charges through 01545: 26,812.952 GPU / 35,037.820 CPU process-seconds, leaving
160,387.048 GPU / 80,162.180 CPU seconds under the unchanged 52/32-hour caps.
No numerical worker is active at this checkpoint. Continue with the existing
runner prefix, sequential numerical jobs and fresh versioned artifact paths.

Structured reuse and outer sequential/block/quadratic control, external DZ5
callbacks, prior memory/timing investigations, all F01--F20 terminal dispositions,
final frozen-source tests/repeats, remote integration/retest and terminal review
remain open. Main stays unmerged. Canonical LEDH rebuilding remains excluded.

The older checkpoints below are historical context.




September 20 continuation through 01421, following pushed checkpoint `d91a1268`:
ordered scalar multistart L-BFGS, endpoint replay, eligible exact replay and
selection now execute in one native XLA program. Original optimizer settings,
smooth-box transform, scalar authority, first ties and target-call order/counts
are preserved. Unsupported host callbacks fail without an eager retry. The
batched locator and outer refinement remain explicitly open.

All 14 focused CPU/GPU locator cases pass (01401/01402), as do 40 sequential,
43 block-center and 12 factor GPU consumer cases (01403--01405), and 63 policy/
controller cases (01420). Current two-/four-start measurements 01408--01419 have
exactly equal outputs. Warm public times are 482.254/5.516 and 961.428/9.151 ms
before/after; graph/XLA times are 51.017/3.884 and 103.620/7.403 ms. Graphs remain
2,181 nodes at both extents; warm device allocation is constant and late host
growth is at most 12,288 bytes. No new trigger fires. Public host peaks rise by
143--146 MiB and cold calls are slower; these costs remain in the analysis.
`sequential-scalar-locator-diagnostic-01419.json` is single-process checkpoint
evidence, not terminal acceptance or a statistically supported timing ranking.

Through 01421 charged time is 23,288.629 GPU / 33,454.492 CPU seconds, leaving
163,911.371 GPU / 81,745.508 CPU seconds under the unchanged caps. No worker is
active at this checkpoint. The partial guard covers 188 sources / 1,273 exact
exceptions. Inventory 01421 finds 2,863 working-tree Python files, 2,862 parsed
and one unchanged external legacy error (2,859 tracked plus four new files).
Focused new-file Ruff and whitespace pass. Remote refs were fetched; main was
an ancestor of `d91a1268`, with no remote divergence to resolve at that point.

Next localize the existing CPU fitter initializer rounding, then continue
batched/outer sequential, block and quadratic control. The explicit bounded
initializer diagnostic and locator measurement contracts are in the master.
External DZ5 callbacks, CPU original-record parity and previous memory/timing
investigations remain open. No F01--F20 terminal disposition is closed. Source-
frozen repeats, full integration/retest and terminal review still gate merge.

Earlier recovery checkpoints follow for historical context.

September 20 continuation through 01399, on fixed-fitting checkpoint `8f334b96`:
initial exact replay and seeded search selection now execute in enclosing native
XLA programs. Eligibility, first-maximum ties, row order, seeded draws and gather
pullbacks are preserved. Optional movement reporting is repaired; its legacy
zero-cloud fixture now enters at the native TensorFlow boundary. All 20 focused
CPU/GPU selection checks, 40 sequential GPU, 43 block-center GPU, 12 factor GPU
and 63 policy/controller cases pass (01344/01346--01348/01373/01398).

Corrected matched comparisons 01374--01397 pass at two extents. Public warm
replay is 4.676/0.408 and 9.087/0.410 ms before/after; search selection is
11.250/0.702 and 16.815/0.800 ms. Maximum compared error is 4.441e-16. Graph
sizes remain 93 replay / 346 search nodes at both extents, with constant warm
device allocation and late candidate host growth at most 12,288 bytes. No new
memory/time trigger fires. These are single-process checkpoint observations;
final three-process evidence remains required. Earlier 01349--01372 measurements
are superseded because the extracted baseline did extra tensor packing.

Current charged time: 22,702.050 GPU / 33,316.018 CPU seconds, leaving
164,497.950 GPU / 81,883.982 CPU seconds under the unchanged 52/32 process-hour
caps. No worker is active at this checkpoint. The guard covers 187 sources /
1,273 exact exceptions and remains partial. Inventory 01399 discovers 2,859
working-tree Python files (2,858 parsed and one unchanged external legacy error;
its tracked-file count of 2,857 excludes the two new files).

Next compile ordered scalar locator starts, then batched locator and outer
refinement/block/quadratic control, under the reviewed plan below. Include the
uncovered `batched_quadratic_center.py` chunk/round loops and non-XLA fit in that
audit. External DZ5 callbacks, CPU fitter parity (01325), and all existing
memory/timing investigations stay open. All F01--F20 terminal dispositions remain
open; no merge, HMC admission or whole-repository compliance is established.

Earlier recovery checkpoints follow for historical context.

September 20 continuation through 01339, based on pushed selector checkpoint `22094f76`.
The complete fixed-center replicate fitter, conditional factor escalation,
selection and audit now share a compiled program. Default XLA eigenpair
accuracy needed repair before original-source parity passed. The current-source two-extent
measurement matrix through 01338 passes numerical gates; public warmed times
are 13.533/0.127 and 27.529/0.268 seconds before/after. Added public host peaks
of about 0.47--0.48 GiB remain an open investigation. Native graph/XLA nodes
stay at 5,341/7,395 at both extents and warm candidate allocations are stable.
These are single-process checkpoint observations, not final acceptance.

Late review restored the original frozen preparation derivative boundary:
01323 reproduced external-tape failures; 01324 passes both focused repairs.
Full CPU run 01325 passes 13 cases but fails original complete-record parity
in two sensitive diagnostic fields. Localization 01326/01327 traces this to
initializer rounding amplified by a saturated factor state. Keep this gate
open and preserve the original tolerance. All 14 current GPU fitter checks (01330), 223 affected GPU consumers (01331),
and 63 policy/controller cases (01332) pass. Details are in the execution record.

The owner-authorized cumulative caps remain **52 GPU / 32 CPU process-hours**.
The extra 48 GPU / 24 CPU hours are counted once. Driver accounting includes
all prior attempts, including failed/interrupted runs and supplemental charges.
Through 01339, charges are 21,725.543 GPU / 33,230.249 CPU seconds; remaining
allowances are 165,474.457 GPU / 81,969.751 CPU seconds. No broader compute or
scientific scope has been requested. No numerical worker is active at this checkpoint.

The repair is incomplete. The partial guard covers 186 sources / 1,273 exact
exceptions; all F01--F20 terminal dispositions remain open. Syntax inventory
01339 discovers 2,857 working-tree Python files, with 2,856 parsed and one
unchanged external legacy parse error. These counts are search evidence. Sequential/block/quadratic
control, external callbacks, and existing memory/timing investigations remain
open. Main stays gated until source-frozen tests and three-process comparisons,
remote integration, retesting and terminal review pass.

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

September 20 continuation after 01276: first enclose fixed-center family
selection, consensus, ordered shrinkage and mean selection error in one native
TensorFlow program. Then move replicate fitting/escalation into the enclosing
lifecycle. This dependency order keeps optimizer failures separate from selector
parity. The exact baseline remains 3582b4ac; current thresholds, family priority,
weight tie ordering, report order, optional caps and audit separation are fixed.
Static schema packing/report reconstruction may stay on the host; numerical
eligibility and selection may not. Existing private monkeypatch selector tests
will use real score clouds that produce their intended choices; no Python
callback escape hatch will be introduced to preserve test-only injection.

Skeptical pre-run review: mean reductions, asymmetric-but-tolerated matrices,
incomplete families, first-error ordering and NaN comparison semantics can
change branch decisions even when a final precision is close. Compare complete
records against the pinned selector, exercise these cases and equal-error weight
ties, and inspect enclosing HLO plus graph growth at two replicate/weight
extents. Explicit graph mode is diagnostic only. Run focused CPU checks before
GPU checks and affected consumers, with the existing 300-second worker ceiling
(900 only if justified by measured compilation). Register matched numerical and
public measurement scopes; failed baseline compilation remains evidence, not a
zero-time comparator. Three-process final comparisons wait for source freeze.
Any changed decision or tolerance failure is a repair trigger; resource or
artifact corruption stops the affected attempt. This is preparation-helper
qualification, not tuning authority, HMC admission, or terminal F01--F20 closure.
The original baseline is an execution comparator, not a new scientific default.
Artifacts use the next unique campaign run directories under the unchanged caps.

The selector checkpoint is `22094f76`. Next enclose dense and structured fits
across replicates, the one-factor stability/escalation decision, selector call,
and the audit/inverse/final-status step in one stable program. Preserve the
existing dense kernel and factor optimizer, initial state, stopping tolerances,
failure priority and diagnostic fields. Dimensionally unidentified factor
families remain absent fits; a skipped second-factor family must remain absent
from reports. Keep audit computation behind the selected-candidate branch.
The full wrapper retains partition/schema validation and report construction.

Pre-run review identifies optimizer rounding/history drift and accidental
execution of the skipped factor/audit branch as the main new risks. First compare
the complete numerical fit records with the pre-enclosure checkpoint under the
same XLA mode, plus the original pinned public endpoint on deterministic clouds.
Exercise dimension rejection, raw-SPD/projection gates, incomplete stability
caps, factor escalation, blocked selection, audit veto and unchanged selection
under changed audit data. Existing Python fit-injection tests will be recast at
the native tensor boundary or with real clouds; no production host fallback is
authorized. Exceptions raised by backend compilation remain failures, never
silent eager execution. Use focused 300-second checks and the existing larger
ceiling only after observing compilation cost; record every failure. Compare
two replicate extents before continuing. No numerical tolerances, seeded draws,
optimization method, scientific target or compute cap changes are authorized.

Measurement ceiling refinement after 01303: the original full fitter takes
294.135 process-seconds at two replicates, with a 13.533-second warm median.
Its four-replicate arm cannot reasonably fit the prior 300-second ceiling.
Use 900 seconds for `fixed_fitting` measurements in both source arms and at both
extents, retaining the same cumulative accounting, attempt budget, inputs,
twenty warm calls and comparison criteria. Other measurements keep 300 seconds.
Run 01302 was manually interrupted at 164.283 seconds and remains invalid; it
was not a demonstrated hang or resource failure. Run 01303 is its valid retry.

The 01303/01304 original-versus-compiled comparison fails: fitted precision
differs by up to 7.319e-8 and dense projection burden by 1.002e-8; selected family
and status agree. Current-checkpoint enclosure parity did not cover this older
dependency drift. Localize weighted dense initialization, correlation
eigensystems, encoding and identical-state loss/gradient arithmetic before
another complete measurement. Default XLA eigensystem convergence is a concrete
suspect given 01241's prior residual finding. Test the existing repository
refined eigensystem on the same frozen inputs as a diagnostic; preserve the
eigenproblem, floor, optimizer and thresholds. A repair is justified only by
same-input residual/output evidence, then full CPU/GPU consumer checks.
The 496.688 MiB added host peak is also an investigation trigger, independent of
parity. Performance promotion remains blocked; do not spend the larger baseline
repeat until these focused issues have been characterized.

Bounded memory repair hypothesis: `_make_factor_program` supplies a Python
value/gradient callback to TFP L-BFGS, which traces its complete loss separately
inside multiple line-search branches. Share a stable-signature TensorFlow
value/gradient function within that program. Preserve its captured training
tensors, analytic objective, autodiff engine and optimizer configuration.
Skeptical risk is a changed captured-input derivative or optimizer rounding;
the complete original-record and same-mode pre-enclosure checks must still pass.
Compare graph nodes, staged host peaks, cold time and warm allocation with 01306
on the identical fixture, plus before/after public timing after qualification.
This is explanatory memory localization, not authority to accept the 550 MiB
increase or to change any numerical tolerance. Reject the sharing change if
record parity fails, and retain the localized evidence for the next repair.

Late enclosure review also checks the frozen preparation derivative boundary.
The original public fixed-center/factor fit records were materialized through
NumPy; a caller could not differentiate the fitted geometry through those
records. Returning native tensors now may accidentally expose optimizer/eigen
backpropagation. After the active matrix completes, test external-tape use
against the original disconnected result. If it is exposed or raises an
unsupported gradient error, freeze the completed native fit outputs inside the
compiled function. This preserves internal loss/gradient computation and values,
not a new derivative algorithm. Recheck full records, affected consumers and
matched current-source measurements; retain the pre-fix evidence separately.
No gradient of a filtering target or an explicitly differentiable covariance
constructor may be removed. This is a local derivative-contract repair under
the existing plan and caps.

1. Finish sequential search/fit/selection and the enclosing block/quadratic
   numerical control. Include reachable exact-incumbent selection; host record
   assembly must not hide row-wise numerical eligibility checks. Keep external
   DZ5 callback compatibility explicit and qualify actual owned consumers.
   Include fixed-center replicate fitting/escalation and audit control; its complete
   family/shrinkage selector is now compiled. The complete
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
Historical discovery at 01199 (superseded by the mass and stability results
below): `structured_covariance_from_empirical` also loops over blocks numerically.
`posterior_local_initializer` reaches `fit_fixed_center_curvature`, whose
replicate, family, pair-stability and shrinkage loops remain host numerical
control. `block_score_geometry` has analogous replicate/block/stability loops.
The static guard excludes iteration for the latter modules and does not cover
mass_matrix; these are open F18/F19 dependencies, not source exemptions.
Carry their original decisions, reports, thresholds and HMC authority boundary
through the migration; removal of NumPy alone did not close XLA execution.

### Fixed-center family stability after 01257

Bounded fixed-center step, after checkpoint `3ab315b5`: compile the
existing `_family_stability` all-replicate qualification and pair loop. This
is a dependency of family selection, not closure of the enclosing fixed-center
fitter. Keep its all-or-nothing usable-replicate rule, pair order, original
replicate identifiers, optional-cap `None` reports and pass semantics. In
particular, missing caps do not become failed caps. Rank, finite/symmetric
matrix validation and non-SPD generalized-eigenvalue absence must retain the
original exception/report behavior. Reuse `_precision_geometry_kernel`; do
not substitute a new metric or another eigensolver.

Skeptical review: testing only positive-definite accepted fits would miss
optional caps, missing matrices, unusable fits and invalid comparisons.
Pin the original fixed-center module and its comparison function, then compare
complete family records for those cases and rotated separated eigenspaces.
Use one native pair body, fixed-size reports and host schema reconstruction;
host packing must not filter numerical eligibility. Check HLO, no callbacks,
no hidden XLA in graph diagnostics, and graph-size independence from replicate
count. Add a matched two-extent complete family-stability fixture, preserving
actual baseline trace failure and the existing error/memory/time gates.
Use the same driver/caps; no source edits during workers. Full family fitting,
shrinkage selection and external/HMC qualification remain explicitly pending.

Results: all 39 CPU/GPU checks pass (01274/01260) after preserving the
constructor rejection for invalid principal-angle/rank field combinations.
All 121 existing GPU consumers pass (01261). Two/four-replicate public timings
are 5.388/2.928 and 27.713/9.144 ms (before/after); all compared values match
within 2.221e-15, with no new trigger. Graph/XLA node counts are 355/1,231
at both extents. See `fixed-stability-diagnostic-01273.json`; terminal repeats
remain pending. The 61 policy checks pass (01275), and 01276 refreshes the
syntax-only inventory. No numerical worker remains running.

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


### CPU full-record sensitivity investigation after 01325

Run 01325 passes 13 cases but fails original full-record CPU parity in the
prediction-Jacobian condition number (3.908e-9 absolute, 1.131e-10 relative)
and a principal angle (4.494e-10 degrees). No tolerance changes are authorized.
The focused diagnostic 01326 finds identical optimizer iteration/evaluation
counts, rank five of six, fitted precision error 4.728e-11, and raw factor
parameter error 1.994e-4. Evaluating the original Jacobian diagnostic at the
current raw state agrees with the new diagnostic to 3e-14. Thus direct SVD
accuracy does not explain this discrepancy; a nearly saturated factor state
amplifies trajectory rounding into derived fields. Compare identical original
initial states and shared/unshared loss graph arithmetic before changing the
implementation. Use one bounded CPU localization and preserve complete report
parity as the gate. The existing optimizer and stopping criteria remain frozen;
rank deficiency is an explanatory diagnostic, not authority to drop fields or
relax their comparison. GPU qualification remains separate.


### Next sequential dependency: exact replay and search-cloud selection

After the fitter checkpoint is measured and preserved, compile the sequential
locator's initial exact replay, finite eligibility, and stable first-maximum
selection, plus its analogous search-cloud selection. These are required
numerical dependencies of the remaining enclosing optimizer/search lifecycle.
Keep initial candidates in their original order and replay them with the scalar
authority, as before; batched search uses the already supplied batch callback.
Preserve target-row accounting and first-maximum ties, finite value/score rules,
all-nonfinite rejection, and the selected diagnostic candidate on budget failure.
The inherited selector did not independently reject a nonfinite position when
its supplied value/score were finite; do not silently add a new eligibility rule.

Skeptical review: selecting endpoints before exact scalar replay could change
call counts or choose a surrogate value. Reusing the generic exact-incumbent
selector could add position-finiteness semantics absent here. Preserve this
specific selector's rules, compare frozen original records and pathological
eligibility cases, and test an actual sequential consumer. Public result schemas
and callback summaries stay at host reporting boundaries. Numerical gradients
through a selected position/score remain gather derivatives; do not impose a new
stop-gradient on those channels. Selection masks/counts remain nondifferentiable.

Use focused CPU/GPU parity, all existing sequential consumers and enclosing
HLO/trace-size checks, then register matched public and numerical graph/XLA
measurements at two candidate extents. The original eager implementation is
only the execution comparator. Host loops over optimizer starts and refinement
attempts remain open, with no numerical exemptions granted. No new scientific
claim, algorithmic objective, seeded stream, threshold or compute allocation.


Sequential selection measurement contract: register `sequential_replay`
(four/eight candidate rows) and `sequential_search_selection` (eight/sixteen
seeded search rows), both dimension two with a fixed quadratic target. The
before executable extracts the unchanged original endpoint's replay/finite/sort
block using the pinned module's scalar/cloud helpers; it is explicitly a legacy
Python diagnostic comparator. It does not claim to time the enclosing optimizer.
The after public helper and complete tensor graph/XLA programs implement that
same bounded block. Exact wrapper/consumer tests constrain extraction risk.
Report output parity, cold/warm time, graph size, host/device peaks, allocator
stability, and imported-source/harness identity at the existing thresholds.
Keep pure graph/XLA and host-wrapper comparisons separate. The existing 300-second
measurement ceiling and cumulative caps apply. No terminal performance claim
follows from these single-process qualification pairs.

### Next enclosing dependency: ordered scalar locator starts

After preserving the replay/search checkpoint, enclose the sequential locator's
scalar multistart L-BFGS lifecycle, endpoint replay, candidate replay and selection
in one stable TensorFlow program. Preserve start order, smooth-box transform,
TFP optimizer settings and stopping condition, exact scalar target authority,
all endpoint/initial replay calls, first-maximum selection and evaluation counts.
Use native loops with a fixed candidate mask so invalid endpoints are skipped
without a dynamic host-selected shape or extra target calls. Host loops may only
reconstruct the existing locator report. The batched locator and outer refinement
remain separate open dependencies; no claim of complete locator compilation yet.

Skeptical review: sharing the loss graph can alter floating-point trajectories;
masked replay can accidentally evaluate ineligible endpoints or reorder calls;
wrapping TFP can expose captured-input derivatives or unsupported callbacks.
Compare complete records with the pre-enclosure source and original optimizer
settings, test finite/nonfinite starts, mixed eligible endpoints, zero starts,
ties, budget rejection, exact target-call order/counts, and enclosing HLO/graph
growth at two start counts. Preserve all numerical tolerances. Compilation or
backend exceptions must surface as failures; do not silently retry eager or
remove a start after an unsupported callback fails tracing. Python callbacks
that require host materialization remain migration debt, not eligible targets.
CPU diagnostics precede GPU checks, affected consumers and matched memory/time
qualification. Use the existing run driver, 300-second focused ceiling, unique
outputs and cumulative caps. An observed compilation cost may justify the already
reviewed 900-second ceiling, not a new allocation or numerical relaxation.

Recovery call-chain review also confirms that
`inference/batched_quadratic_center.py` retains numerical cloud-chunk/fit-round
Python loops and an explicit non-XLA fit. Its integration scripts call the public
initializer directly; its diagnostic nonclaims do not repair that execution debt.
Include this file explicitly in the remaining quadratic lifecycle audit before
F18/F19 closure. Existing partial-guard success does not cover it.

Scalar-locator measurement contract: compare the original public endpoint with
the new public endpoint under its unchanged budget-rejection boundary, before
any refinement. Use two/four starts, dimension two, a fixed SPD quadratic target,
four optimizer iterations and seven line-search iterations; retain the original
box radius, tolerance and stopping rule. These reduced iteration counts are
bounded diagnostic settings, not proposed defaults or optimization-quality
evidence. Return the selected point, exact evaluation count and every optimizer
diagnostic. Also compare the new complete graph versus XLA program at the same
settings. Preserve original tracing failures, fixed inputs and source/harness
identities; report public and numerical timing separately. Dedicated fixture
`sequential_scalar_locator` uses the existing 300-second ceiling and two extents.
Gate numerical equality, actual target-call order/accounting, no Python callbacks,
bounded graphs and original consumer behavior before interpreting memory/time.
The three-process terminal requirement and all earlier open findings remain.

Bounded follow-up for the still-open CPU fitter comparison: after the scalar
locator checkpoint, inspect the weighted initializer's least-squares rounding
separately from its eigensystem and inverse. On the existing frozen failed
fixture, compare original Eigen COD, native COD and one residual-correction
solve for the same least-squares problem. A high-precision independent Decimal
solve may assess rounding on this tiny full-rank diagnostic only; it must not
enter runtime or define a new target. If residual correction is tested, keep
the same rank rule, projection, optimizer, stopping tolerances and initial
parameterization. Assess initializer error and original complete fitted records;
closer baseline agreement alone does not establish greater numerical accuracy.
This is explanatory localization under the existing 300-second CPU limit. Do
not add a runtime correction unless it reduces a demonstrated arithmetic error
and then passes the unchanged full CPU/GPU records and consumers. No input-
specific initializer, tolerance relaxation or field omission is authorized.

September 20 pre-run review at pushed scalar checkpoint `dcfaa15d`: register
`fixed_fitting_initializer` as a CPU-only explanatory diagnostic, with the
existing two frozen 9-by-3 training replicas. Compare Eigen COD, native COD in
graph/XLA and one residual correction in graph/XLA against independent Decimal
normal equations at 60/90 digits. Require agreement below 1e-50 between Decimal
precisions and a singular-value ratio above 0.01 before using that reference.
The high-precision normal equations are diagnostic only. Preserve full original,
native and corrected fit records, including all existing 1e-10 comparisons;
record their failures rather than weakening them. Each injected solver clears
the enclosing caches so an old graph cannot masquerade as a different arm.
The diagnostic pass means reference validity and finite execution, not CPU
parity closure or authorization to install a runtime correction. No source
changes while the sequential driver runs; use the 300-second ceiling.

Run 01422 validates the independent reference and improves solve accuracy with
one correction, but full-record parity worsens slightly. Keep the correction
diagnostic-only and the CPU gate open; preserve all original thresholds.

### Batched locator enclosure and progress delivery

Enclose the existing batched TFP L-BFGS, batched endpoint check, scalar exact
candidate replay and stable selection. Preserve the exact optimizer settings,
start order, scalar authority, invalid-endpoint exclusion and conservative row
accounting. No batch-to-scalar optimizer substitution is allowed.

The existing objective emits Python progress with host materialization on every
call. Inspecting its consumers confirms these events report values, scaled
scores, transformed gradients and standardized positions. Retain every event
in order by collecting numeric rows in device variables during the compiled
call and delivering them afterwards. Label deferred delivery explicitly. This
cannot preserve real-time callback cancellation; document that timing boundary.
External DZ5 callbacks additionally mutate host diagnostic state and remain
ineligible until migrated; deferred events must not imply those counters are
qualified. Callbacks that inspect tensors on the host must still fail tracing.

Use no telemetry resources when no progress callback is requested. For progress,
use a per-program lock, reset storage inside each invocation, capture immutable
result tensors and synchronize before releasing the lock. Bound optional trace
storage to at most 16 MiB and 4,096 events, with at least one event. The maximum
line-search iteration setting is not an evaluation-count bound: TFP's internal
Hager-Zhang bisection has a separate loop. Therefore report total calls and an
explicit overflow flag; reject incomplete telemetry before emitting any saved
objective event. Never overwrite earlier events or silently truncate. This is
an observation-storage limit, not a changed optimization stopping rule. Limit
the optional-program cache to four entries and measure the extra host/device
cost. No numerical target is re-evaluated to reconstruct telemetry.

Skeptical review: compare full original records and event arrays on quadratic,
nonlinear, nonfinite and tied targets with both stopping rules; use resource
counters to verify original callback order/counts. Check repeated/concurrent
invocations, overflow rejection, unsupported callbacks, two start counts,
HLO/no Python callbacks, graph-mode diagnostics without hidden XLA and downstream
consumers. Preserve frozen derivatives and unchanged numerical tolerances.
CPU focused diagnostics precede GPU checks and matched public/numerical
measurements, both with and without telemetry. Use the existing sequential
driver, 300/900-second limits and cumulative budget. This review authorizes no
runtime least-squares correction or whole-initializer compliance claim.

Batched-locator measurements use separate fixtures/worker so the already-frozen
scalar measurement harness remains intact. Use the same two/four starts, SPD
quadratic target, dimension two, four optimizer iterations and seven line-search
iterations as the scalar diagnostic; preserve all other original settings.
Measure public before/after with and without progress delivery, and current
graph/XLA numerical scopes separately. Progress numerical scope includes reset
and every trace write; public scope includes delivery of every objective row.
Return common point/accounting/optimizer/endpoint fields, total observed calls
and overflow status. Full trace values are checked against the pinned original
by focused tests; no target is replayed for measurement. Preserve original
tracing failures, 20 warm calls, two extents and the 300-second ceiling. These
single-process diagnostics do not replace source-frozen terminal repeats.

Late checkpoint review adds the explicit original-public derivative-boundary
check (`locator_frozen`) for scalar, batched and buffered localization: watch
both starts and scale in an external tape, compare full records with frozen
3582b4ac and require disconnected returned geometry. Numerical value/score
parity alone does not check whether wrapping the optimizer accidentally asks
XLA to compile its derivative. Preserve the existing public boundary; no new
optimizer differentiation is authorized. Use the 300-second focused ceiling.

### Initializer arithmetic isolation after 01422

The residual-correction diagnostic improves the linear solve but fails the
unchanged complete-record gate. Next compare the symmetric projection,
covariance inversion, initial factor state and encoding on identical inputs,
using both existing frozen training replicas. Capture each intermediate from
the original 3582b4ac initializer and the candidate XLA initializer; report both
end-to-end differences and single-operation differences after supplying the
same predecessor. This distinguishes propagated rounding from a faulty local
operation. Preserve the original eigenvalue floor, correlation extraction,
anchor/sign rules and encoding bounds.

Use independent 60/90-digit Decimal matrix inversion only on these tiny
well-conditioned matrices, with reference agreement below 1e-50. Report inverse
residuals, symmetry and eigensystem residuals; compare initial encoded states
before attempting another full optimizer run. Record complete numerical arrays
and the actual command/environment through the existing CPU-only driver and a
new numbered artifact. The 300-second ceiling and cumulative caps apply.

Skeptical review: replacing only the initial vector can diagnose sensitivity
but cannot qualify a runtime fix. A match to the rounded baseline is not an
independent accuracy criterion. Do not install a projection, inverse, solver or
encoding change until a demonstrated arithmetic defect justifies it and the
unchanged full records and consumers pass. No tolerance relaxation, input-
specific initializer, changed optimizer setting or omitted field is allowed.

Optional progress memory investigation remains separate. At two/four starts,
the 4,096-event buffer contains 327,680/655,360 numerical bytes before allocator
rounding and temporary/output copies. Original public progress streamed events
to the host, so retaining them on device necessarily changes that allocation.
The measured 1,023,488/2,097,408-byte public peaks trigger the existing 2x rule;
graph and XLA versions with identical buffers have similar peaks. Before
accepting this cost, measure the same fixed inputs with 128 and 4,096 event
capacities, verify identical saved events and optimizer records, and inspect
allocator current/peak values after construction, cold execution and repeated
calls. Clear neither arm's compiler caches selectively. This is a CPU/GPU
resource diagnostic using the normal sequential driver, 300-second ceiling,
verified GPU growth and cumulative budget. It cannot replace final repeats or
establish a process-wide memory cap; the four-entry cache bounds retained
programs, while caller-held snapshots have their own lifetimes.

### Structured fit fixed-capacity dependency

Before enclosing structured sequential refinement, qualify a fixed-capacity
representation for its variable eligible reused rows. Preserve the original
fresh/reused order, the 1/2 fresh plus 1/2 reused weighting when reuse is nonempty,
uniform fresh weights otherwise, and disjoint holdout data. Inactive rows must
have zero offsets, zero responses and zero weights and must not affect input
eligibility, fitting, score errors or active-row diagnostics. The fresh block
always has at least twice the parameter dimension, so adding zero rows cannot
change the original COD epsilon*min(rows, columns) threshold. The prediction
Jacobian rank threshold must still use the active equation count, not capacity.

First extend only the internal fitter signature with explicit active-row
metadata and test padded versus compact fits under identical existing settings.
Current compact TensorFlow fits isolate representation changes; the frozen
3582b4ac full-record gate remains separate and unchanged. Compare every public
fit field, optimizer iteration/evaluation counts, rejection and identification
decisions at zero, partial and full reused-row occupancy, and inspect enclosing
HLO/trace counts as the active count changes. Include weighted-loss and active
Jacobian-rank reference checks. No additional runtime solver, new RNG stream,
ridge, optimizer setting or comparison tolerance is introduced.

Skeptical review: padded zeros can alter floating-point reduction order, while
using capacity in rank thresholds changes the actual numerical decision.
Sanitize only inactive rows, never a nonfinite eligible observation. A new
complete-record discrepancy blocks this representation and triggers localization
before wiring it into sequential preparation. Internal unidentifiable raw
optimizer coordinates are explanatory; the original complete public fields and
discrete optimization counts remain the comparison authority. Use focused CPU
then GPU checks, the sequential driver, 300/900-second bounds and cumulative
caps. Only after these checks pass, integrate preparation and fit consumption;
host-side row-count selection must not be admitted as an XLA replacement.

September 20 arithmetic localization after 01498: the full-occupancy inputs,
initial raw vectors and first objective values match exactly, but the initial
gradients differ by 2.168404344971009e-19. The same diagnostic frozen initial
vector still gives different optimizer counts and fitted records. Before any
further runtime trial, capture normalized weights, offsets, responses, anchors,
loss intermediates and their pullbacks from each initializer observation, and
archive unoptimized/optimized HLO beside the new numbered CPU run. Instrumented
outputs can alter fusion, so compare the independently observed gradient with
the original objective gradient and preserve any discrepancy. These arrays and
compiler artifacts explain arithmetic only; a passing diagnostic does not close
the padded-fit or original-record gates. Use the unchanged 300-second limit,
sequential driver and cumulative budget; install no arithmetic correction based
only on agreement with a rounded baseline.

Run 01500 reveals an additional compiler-specialization risk: the observed
two-factor HLO has only the two holdout arrays as runtime parameters; training
data and initial-state dependencies are embedded constants. Before changing the
runtime, inspect the actual four-iteration compact fitter HLO in both one- and
two-factor modes. Expected input arity is six. Record parameter lines and graph
trace counts separately: one TensorFlow trace does not prove one XLA executable.
This is a bounded CPU compiler diagnostic, not a performance comparison. If
confirmed, replace data-dependent scalar slicing in differentiated code with
TensorFlow gathers, preserving every formula, then rerun specialization and
complete-record checks. An arithmetic or public-record discrepancy remains a
repair trigger; no changed optimizer settings or tolerances are authorized.

After the gather trial, 01507 exposes one GPU padded condition-report mismatch
despite identical optimizer states. The padded representation stays unintegrated.
CPU consumer suite 01508 exposes two additional pre-enclosure two-factor field
differences and retains the original CPU-record failure. Test a single XLA
optimization barrier on normalized weights to isolate constant folding of
uniform enclosing-loop weights from dynamic standalone weights. This is an
execution-only hypothesis based on the 01500 gradient HLO; it changes no formula
or setting. Run only the two new failing cases first under the 300-second CPU
limit, and remove the barrier if it fails. A passing local check still requires
full CPU/GPU records, input-specialization guards and consumer qualification.

### Changed-data compilation memory diagnostic

The static TensorFlow trace guard missed XLA value specialization. Compare
checkpoint 1e9afd2c with the indexing repair in fresh GPU processes, separately
in XLA and explicit graph-reference mode. Pin both the factor and COD source in
the checkpoint arm; use the same current unchanged eigensystem dependency.
Use the existing deterministic dimension-three, ten-row cloud, one factor and
four L-BFGS iterations. Record one cold call, three identical warm calls, four
distinct but same-shape training clouds (each first call and immediate repeat),
then ten repetitions of the last cloud. No RNG, optimizer or tolerance changes.

Record complete public numerical fields, synchronized call-plus-materialization
times, `/proc` RSS/high-water, allocator current/peak, one TensorFlow trace and
the first/last HLO parameter arity/hash. Inspect HLO only after timed memory
sampling, since inspection can compile. Candidate XLA must retain six runtime
parameters and identical HLO across changed inputs. Numerical discrepancies are
reported against unchanged 1e-10 gates; this experiment only explains resource
behavior and cannot waive the existing complete-record failures. Single-process
timings are descriptive; terminal repeats still compare original 3582b4ac.
Use one fresh process per arm/mode through the sequential driver, verified GPU
growth/idle preflight and a 300-second ceiling each. The question is whether
data specialization causes changed-input recompilation and host growth; unchanged
warm timings alone cannot answer it. Do not describe a bounded cache or these
tiny extents as a general memory cap or whole-program qualification.

September 20 checkpoint accounting through 01577: 27,876.007 GPU /
35,838.542 CPU process-seconds, leaving 159,323.993 GPU / 79,361.458 CPU
seconds under unchanged 52/32-hour caps. Inventory 01577 finds 2,879 working
Python files, 2,878 parsed and the one unchanged external legacy error. Focused
Ruff and whitespace checks pass. No worker is active; main remains unmerged.
