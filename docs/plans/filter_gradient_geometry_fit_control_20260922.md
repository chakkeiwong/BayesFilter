# Enclosing quadratic fit and result decisions

Continue the authorized F18/F19 repair on the repair branch within the unchanged
32 CPU / 52 GPU process-hour caps. GPU3 keeps its two idle preflights and verified
memory growth. CPU runs hide GPUs. One numerical worker executes at a time and
runtime, tests and driver remain frozen throughout each worker or matrix.

Question: can the low-rank geometry initializer enclose its constrained
score-difference fit, covariance, training/holdout diagnostics, center refinement,
exact incumbent update/replay and final acceptance in one native XLA program?
The inputs are the already evaluated and partitioned design and frozen pilot
basis. Pilot generation/evaluation, data-dependent partition preparation and the
outer iterative wrapper remain separate open work; this is not yet a compiled
whole initializer.

The authority is isolated original 3582b4ac. Supply identical frozen pilot basis,
design cloud and training/holdout partition to the original public initializer
and compare its complete result suffix with the candidate. Freezing those inputs
is an independent reference comparison, not a replacement RNG or pilot algorithm.
Retain original source hashes and complete original payloads. Compare every fit,
holdout, spectral, proposal, replay and incumbent numerical field at unchanged
1e-10 absolute/relative tolerance; decisions, shapes, statuses, reason order,
evaluation counts and provenance are exact. Artifact hashes remain tied to each
arm's actual floating-point payload and are not a numerical equality metric.

Reuse the existing score-curvature least-squares kernel without rank-threshold,
floor, clipping, condition-cap or acceptance changes. Graph-reference execution
uses the existing non-XLA SVD switch. Reuse the refined eigensystem already
qualified for the proposal, rather than the known inaccurate raw XLA eigensystem.
Inputs, callback resources and incumbent records must remain runtime operands
under a stable explicit signature; default JIT stays true. Host construction may
inspect fixed tensor schemas; numerical decisions and repetition stay native.

Qualify D1/D3/D5, holdout enabled/disabled/rejected, indefinite raw curvature,
singular finite designs, finite numerical extremes, nonfinite fitted values,
constrained/unconstrained proposals, finite but rejected exact proposals, replay
failures and exact incumbent ties. Test changed inputs/resources, one trace,
enclosing HLO and absence of host callbacks. Preserve every failed attempt.
Register bounded 120/300-second focused groups in the existing driver, maximum
three attempts per unchanged job. Fresh original/graph/XLA D3/D5 costs include
the same reporting work and retain the 20% warm, 2x cold/device and 256 MiB host
investigation triggers. No public wiring before default-GPU and consumer checks.

Skeptical review: padded rows can change least-squares thresholds and reduction
order, so this first suffix has exact static train/holdout extents with no padding
or dynamic-shape claim. A replay mismatch must not rewrite the original incumbent
or silently become a new fit veto. The original evaluates a proposal and replay
even if the holdout later rejects the geometry; preserve that order and charge.
A finite proposal can improve the exact incumbent even when it fails the center
refinement gate. Scalar callbacks here are endpoint evaluations, not training.
The callback adapter rejects unsafe XLA assertions and Python callback ops; its
construction/conversion-error boundary is separately qualified in 02499/02500.
Check original records before drawing any speed or memory conclusion.

Promotion vetoes are any record/count/source mismatch, unsupported callback,
unexplained memory/performance regression or missing GPU/consumer evidence.
Contention, invalid provenance or exhausted budgets stop the affected launch.
Failures trigger a localized repair under the existing contract, not tolerance
relaxation. No public, HMC, posterior, scientific or complete-repair claim follows
from a CPU suffix pass. No external source/pin, environment or package changes.
This is primary-agent review; no independent review is claimed.

02509 passes the first original D3 Gaussian suffix comparison in graph and XLA,
including the complete fit report, covariance, proposal, incumbent replay and
all callback counts. Proceed to the predeclared D1/D3/D5 failure cases and changed
input/enclosing-program checks in separately bounded workers.

02510--02513 pass all 35 planned CPU checks (11 full original suffixes at each
dimension plus failed-fit/no-call and changed-input/enclosing-XLA checks).
The next cost comparison extracts the exact original function AST from the
`fit = _fit_constrained_quadratic(...)` assignment through its return. Only the
function signature changes to supply already completed prefix data; the body
hash and original source closure are archived. Both arms build the full result
class, diagnostics, provenance hash and array payload. Identical prefix data
and exact candidate records are prepared outside both timed suffixes. Compare
the extracted baseline with the unchanged full original initializer as a
harness gate, including every field except each arm's floating-point payload
hash. A proxy timing of the full original initializer would include omitted
pilot/design work and is explicitly disallowed.

The candidate formatter reuses the already computed spectral diagnostics. The
existing public result method recomputes eigenvalues eagerly; leave that method
unchanged while testing the internal complete-schema formatter. No numerical
computation is moved into the formatter. Qualify full payload equality, not
merely the result suffix, in every cost arm before interpreting its timing.

02514--02519 pass all six cost processes, including complete initial/changed
payloads and the extracted-baseline versus full-original comparison. Before
retaining the implementation, inspect resource lifetime: a cached spectral
function first traced inside a target-owning graph may keep that graph alive
through TensorFlow's custom-gradient registry. Add a D4 resource-change/release
check in a fresh worker, without clearing caches after the call. Add exact
constant-target ties at D1/D3/D5 to verify earliest-center provenance. A lifetime
failure triggers an isolated tracing repair and renewed affected checks/costs;
do not infer general native executable release from Python weak references.

02520 passes all six changed-input, failed-fit, exact-tie and ownership checks;
the first cached spectral call does not retain this controller's callback,
resource or graph in the tested fresh process. No lifetime repair is warranted.

Costs in 02514--02519 trigger the unchanged cold and additional-host-memory
investigations: D3/D5 original warm medians are 5.408/6.935 ms versus XLA
6.332/7.518 ms; XLA cold is 2.155/2.300 seconds and additional observed RSS is
305.3/312.0 MiB. No >20% warm trigger fires in these single-process descriptive
observations. Most RSS increase occurs at first execution. Archive:
`geometry-fit-cpu-costs-02519.json`, produced by the prior checked analyzer with
one filename/role option added in `analyze_geometry_fit_costs_20260922.py`.

Next separate synchronized native-call, result construction and full-payload
times, without equating the original call (which already includes its result
construction) to a candidate raw kernel. Repeat the six matched CPU arms with
3,000 alternating full-result calls each, observing allocation every 1,000 calls.
This discriminates persistent warm allocation from one-time compilation and
reporting costs; the same full-payload and original-source checks remain gates.
No runtime optimization or tolerance change is justified by the current timing
alone. Preserve the first measurements and record any continuing growth before
considering public integration.

02521--02526 pass every complete payload comparison and all 3,000 additional
calls per arm. XLA graphs have 1,663 nodes at both D3 and D5; graph-reference
programs have 1,019. XLA RSS rises about 508/520 KiB after the timed calls, with
zero growth in the last 1,000-call interval at both dimensions. The observed
306.5/313.4 MiB additional host footprint arises mainly on first execution;
bounded fixed-signature reuse does not show continuing growth here. The original
and graph arms also reach stable late observations. Compiler/runtime overhead is
consistent with these phases, but exact attribution and native executable
eviction remain unproved. The cold/host triggers remain recorded, not waived.

Warm medians are 5.418/7.014 ms original, 6.408/7.494 ms graph, and 6.147/7.457 ms
XLA. Native calls account for roughly 1.19/1.30 ms of the XLA totals; result
construction and full payload work dominate the remaining time. The original
call includes result construction, so it is not a raw-kernel comparator.
`geometry-fit-components-growth-02526.json` retains phase/memory observations and
checksums. Its payload-plus-overhead time is total minus call minus result time:
the first raw payload timer inadvertently included memory instrumentation. Total
elapsed and allocation observations are unaffected; the timer boundary is fixed
for future runs. `geometry-fit-cpu-costs-02526.json` preserves complete costs.

Before future public wiring, add a one-entry callback-identity cache over only
numerically relevant settings and tensor extents. Seed/reporting-only changes
must not retrace this seed-free suffix; changed numerical settings must rebind.
Test replacement and old Python graph/callback release. This is ownership and
construction repair, not an arithmetic optimization; no native eviction claim.

02527 passes callback-identity, relevant-setting cache binding and old graph/
callback release. The original lifetime test deleted a local captured variable,
which could empty its closure cell and make resource release too easy. A helper
factory now retains the variable through the callback's own closure; the stricter
02528 check still passes. Focused Ruff passes all touched numerical/test files.
02529 passes all 72 policy/controller checks. Audit 02530 passes with 2,979
working Python files, 2,978 parsed and one unchanged vendor-reference error.
The guard is explicitly partial: 210 sources and 1,306 exact exceptions. New
fit numerics have no numerical-loop exemptions; formatting exceptions only copy
completed fields and the callback scanner has exact host-validation exceptions.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain internal fit controller | Complete original suffixes, full payloads, callback counts, ties, changing operands/resources and Python release pass on CPU | GPU and complete initializer/consumer evidence pending | Dynamic design partition, pilot lifecycle and GPU costs | Enclose prefix/preparation and qualify GPU when idle | No public switch, complete repair, posterior/HMC claim or merge |
| Preserve cold/RSS investigation | Most extra RSS appears at first execution; bounded 3,000-call intervals become stable | 256 MiB/cold triggers remain recorded | Native executable retention across arbitrary targets/extents | Include in final repeated whole-endpoint costs and ownership checks | No general leak-free claim or terminal memory acceptance |

Primary-agent review: the isolated original AST body plus full original payload
comparison addresses omitted-baseline work. The strongest remaining alternative
explanation for apparent numerical completeness is excluded pilot/partition
control, explicitly outside this suffix. GPU3 remains busy (8,675 MiB and near
100% utilization); its idle gate was not waived. No numerical worker is active.
Charges through 02530: 51,724.668491946984 CPU / 48,325.1067211973 GPU seconds.
No external source/pin, package, environment or HMC tuning authority changed.
