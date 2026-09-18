# Filter execution repair record

Work continues on `repair/filter-gradient-xla-20260917`. No merge is authorized
by the current evidence. All findings remain open until their complete route
tests, policy checks and paired measurements are recorded.

The owner confirmed execution repair plus blocking unsupported canonical LEDH
claims. A full canonical algorithm rebuild is excluded. GenUT's shared NeuTra
training/HMC capability and its obsolete admitted factory now fail closed.
Finite-program AD and manual JVPs remain diagnostic and keep their scalar.

## Focused execution and repairs

- Run 00009: 45 preparation/consumer tests pass. It covers TF quadrature,
  Hilbert ordering, polynomial bases, retained moments, numeric identity bytes,
  predator-prey derivatives and state consumers. Earlier run 00008 exposed
  TensorArray clear-after-read and an incorrect new SIR reference fourth stage;
  the simulator's existing half-step fourth stage was preserved.
- Runs 00006 and 00011: native GenUT loops exposed AD/XLA zero-capacity
  TensorList failures. Skipping the statically configured zero-iteration
  correction preserves the original no-op. Run 00013 passes the public
  graph/XLA score test. Reverse AD differentiates the same independent-row
  scalar program, without pfor or canonical-score claims. Active corrections,
  multi-date cases and baseline comparisons are still required.
- Run 00010: replacing LGSSM parameter unrolling by native mapped directions
  restores the original operation layout. Fourteen tests pass. Three strict AD
  rounding checks fail on newly created September fixtures: 1 and 2 ULPs in
  float64 and 3 ULPs in float32. Run 00012 executes exactly these tests against
  pinned, unchanged baseline source and reproduces all three discrepancies.

## Review of replacement LGSSM fixtures

The old tests depend on missing July JSON fixtures. Those files also cannot
serve as current LEDH evidence under the August 21 invalidation. Fresh, fixed
September mechanics inputs were supplied without consulting old results.
Their AD bit-equality thresholds do not transfer automatically from different
inputs. The unchanged September baseline exhibits exactly the same errors as
the repaired native-direction implementation (runs 00012 and 00010).

The three replacement-fixture assertions now forbid exceeding those observed
baseline errors (1/2/3 ULPs), while normalization and other zero-ULP assertions
remain unchanged. This is an explicit test-contract correction, not a numerical
repair or evidence that the full algorithm is canonical. Closing F02 additionally
requires before/after output parity and enclosing XLA/memory checks. A worse
candidate cannot be excused by further increasing these bounds.

The original four vectorized derivative edits changed reduction layout and
initially produced larger discrepancies (run 00007). They were replaced with
native parameter loops; no numerical tolerance was changed to accept those
larger errors.

## Source scope note

The new predator-prey derivative follows the existing local standard RK4
program. The inspected author predator step uses a half-step fourth stage;
that difference predates this execution repair and was not changed. No
source-faithfulness or scientific admission follows from this mechanical
derivative repair.

## Remaining work

Recovery verified run 00026 (14 CPU-pool tests), run 00027 (26 joint-center
and exact-incumbent tests), and run 00028 (CPU-target checks) passed. These
are focused evidence, not closure of the affected findings.

Run 00029 passed all six actual-SV checks (FD, batch permutation, diagnostic
trace API, analytical XLA parity). Run 00030 passed 14 TT contraction/ALS
checks, including sweep-count-independent graph size. Run 00032 passed both
complete generic branch-axis value parity checks at the unchanged 1e-12
relative threshold. These need final-source confirmation after dependent edits.

The first complete actual-SV FP64 GPU/XLA pair (after run 00031, baseline
run 00033, T=4/B=2) has identical input hashes and exactly identical outputs.
Nodes: 10390 -> 2425; cold execution including compile: 20.77 -> 6.98 s;
warm median: 14.86 -> 32.06 ms; peak TF allocation: 8,527,872 -> 16,916,480
bytes; peak host HWM: 1,444,306,944 -> 1,255,276,544 bytes. This is one
process per arm, not a final repeated comparison. Warm slowdown triggers
investigation; the near-2x device increase also requires explanation. Native
control-flow overhead and checkpoint lifetimes are candidate mechanisms.
Do not close F05 or claim a performance improvement from this pair.
The measurement worker now uses TensorFlow's public device synchronization
barrier before starting the output-copy timer. Earlier pairs synchronized via
the output copies, so their end-to-end comparisons remain descriptive but their
copy-time field also includes any outstanding device work. Final comparisons
must use the corrected worker in both arms.

The generic adjoint candidate now uses the existing XLA scaled CholeskyQR2
backend for the same augmented ridge objective because the original
`lstsq(fast=False)` cannot compile under XLA. No ridge or threshold was changed.
The original scalar and analytical-score tests, condition veto checks, and
paired complete-endpoint evidence must pass before this candidate is accepted.

The actual-SV TT refactor preserves the existing `extension_or_invention`
classification in the July 31 fixed-branch admission plan. Inspected anchors:
Zhao--Cui paper Algorithm 2 (local text lines 693--725), Algorithm 3
(890--918), SV example (1994--2038), and author
`eg2_sv/mainscript.m` configuration/solve plus `models/sv/{setup,transition,
like,priorpdf}.m`. The author uses rank-adaptive functional TT construction;
the local fixed two-axis ALS route is not that implementation. This change
only replaces the existing dates, fixed `(0,1,1,0)` sweeps and manual
directional replay by native tensor recurrences. It makes no new source
faithfulness or admission claim. The August 10 local closed derivative
derivation supplies the same-program formula, including scale/ridge terms.

Runs 00015–00016 found that the now-compiled batched CPU target fails its
existing FP64 square-root reconstruction threshold (roughly 4e-9 versus
1e-10). The TensorFlow generic XLA eigensolver convergence tolerance is a
candidate cause. The repair candidate uses the same XLA symmetric Jacobi
operation with FP64 machine epsilon and at most 100 sweeps. It changes no
model, covariance ridge, acceptance threshold or score definition. Require
eigensystem/reconstruction and gradient checks plus complete scalar/batch
target parity before accepting it; preserve any unsuccessful evidence.

Complete the CPU pool and SGQF consumer tests, TT/adjoint and particle/flow
recurrences, reachable inference NumPy migration, stable-signature and source
policy guards, complete measurement fixtures and repeated paired comparisons.
The campaign gate must reject missing, skipped or stale-source evidence.

## September 17 continuation: compact checkpoints and graph failure

Runs 00039 and 00040 pass the TT contraction/analytical-direction checks and
all six complete actual-SV checks after compact checkpoint reconstruction.
Run 00041 executes that implementation on GPU2/XLA. Device peak drops to
4,333,568 bytes (baseline run 00033: 8,527,872; prior candidate run 00037:
16,916,480). Trace is 0.79 s, cold execution 4.18 s, warm median 19.66 ms,
and host HWM 1,253,552,128 bytes. These remain single-process diagnostics;
the warm regression relative to the old 14.86 ms baseline still requires
repeated matched-worker measurements and an explicit terminal assessment.

Run 00038 exposed five dense LEDH graph failures returning empty float32
tensors where traced outputs were nonempty float64 tensors. CPU-only
localization reproduced the failure with AutoGraph disabled and transport
removed. Disabling Grappler arithmetic, constant, or dependency optimization
restored the expected values. Isolating the unchanged flow calculation in a
fixed-signature function with `_noinline=True` also restored them while leaving
the optimizer enabled. Plain function encapsulation without that attribute did
not work. Run 00042 checks all existing dense LEDH tests with this local
function boundary; enclosing XLA and score compilation remain separate gates.
That temporary function-boundary workaround failed reverse differentiation in
run 00042 and was removed. Hoisting the invariant process/observation
covariance stabilization outside the time loop fixes both value and derivative
execution, with normal optimizer settings. Run 00045 passes all 30 dense LEDH
tests, including two new complete value/score XLA checks. No tolerance,
covariance regularizer, score, or transport branch was changed.

The new bootstrap regression group exposed a reference-helper unpacking typo
in run 00043 (fixed), then two actual seeded-resampling mismatches under XLA in
run 00044. All graph-mode cases and the no-resampling XLA case pass. This is
an unresolved random-stream compatibility issue; the mismatch is not excused
as numerical noise or hidden by loosening output tolerances.

Run 00054 had one float32 Contract-E assertion at 2 ULP for the aggregate
score. The pinned unchanged source in run 00055 reproduced the same failure
and value, so the assertion was corrected to the observed baseline bound of 2
ULP. The per-batch score remains within the existing 3-ULP baseline bound.

## Recovery through run 00092

Run 00088 passed all 28 Algorithm-1 and determinant checks. The original
detached epsilon selection and report derivatives remain detached. Observation
eigensystems are hoisted; pseudo-time histories use fixed tensor buffers.
The explicit non-XLA Algorithm-1 reference mode requires a local no-inline
coefficient boundary; the XLA path keeps coefficients inline because saved
transpose dimensions otherwise lose their static shape. This is not canonical
algorithm or analytical-score admission.

Run 00089 passed all seven annealed transport checks. Conditional VJPs now
recompute only the selected branch, preserving tensor and variable captures.
Fixed active-row padding uses dense selection to avoid sparse-gradient shape
changes. Failed attempts 00073--00086 remain preserved.

Run 00090 passed 20/24 complete OT and structural checks; all XLA cases passed.
The four non-XLA LEDH gradient failures came from nested covariance
factorizations. Hoisting invariant prior/observation factors while retaining
both original stabilization operations repaired the failure. Run 00091 passed
all 24 cases at the original 1e-10 thresholds. The temporary graph-reference
context/no-inline workaround was then removed. Run 00092 passed six captured
gradient checks, including both branches, mixed captures and variable updates.
Final-source endpoint confirmation remains required.

Unrecorded direct CPU diagnostics before run 00090 are conservatively charged
30 process-minutes in supplemental-compute-0001.json. This is a recovery budget
estimate, not a measured runtime; the driver includes it in remaining budget.

The adapted and Gaussian TT routes are being moved to complete native date
recurrences. Runs 00093--00094 identified graph-time object validation and an
XLA-unsupported log determinant. The adapted triangular map uses its unchanged
sum of log diagonals for the determinant. Full parity and consumer checks are
still pending; these findings remain open.

## Native mapped TT and measurement guard continuation

Run 00095 passed all 13 mapped-TT parity/veto checks: horizons 1 and 4,
graph/XLA, Gaussian reference and Student-t floor. Run 00096 passed all 13
Gaussian consumer checks, including the existing Student-t lane parity,
retained-capture bit identity, frozen target RMS/Gram identities and snapshot
serialization. The reference tests now freeze their independent NumPy Kalman
hints before invoking runtime; three active frozen-artifact benchmark consumers
use the shared tensor-indexed hint reader. Other legacy hint factories still
need consumer review. Snapshot construction consumes completed histories.

The existing adapted and Gaussian programs remain local extensions. Inspected
again: paper Algorithm 2, equations (15)--(16), local extracted lines 693--725;
Section 5.2, lines 1581--1594; author
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:64` through
the sequential fit/normalizer update; and
`deep-tensor.dev/src/@TTSIRT/marginalise.m:25` through its final defensive
normalizer at line 85. The author uses adaptive TTSIRT fitting and QR-based
marginalization. The local frozen ALS/Gram and C2 row law remain existing
extensions, not new source-faithfulness evidence. The mechanical refactor
preserves their formulas, floor clamp and detached floor adaptation.

Run 00097 passed the campaign/source guard group, including new rejection tests
for stale source, output structure/type mismatch, numerical/resource failures
misclassified as compilation failures, and regression thresholds. The comparison
runner now requires both registered extents (except the four fixed audit
fixtures), three repeats, current imported sources, matching fixture/hardware,
20 warm calls, and genuine XLA/HLO. It records baseline compilation failures and
may compare a valid explicit graph/host reference separately; no XLA timing is
invented. Eager host reference runs are explicitly labeled and may contain the
baseline's nested step compilation. Regression investigation acceptance is not
yet implemented: flagged differences currently keep the gate closed.

Run 00098 passed the new complete Gaussian TT CPU/XLA measurement smoke.
Run 00099 passed on idle GPU2: 6,285 GraphDef nodes, trace 2.64 s, first execution
7.46 s, median of 20 warm calls 14.35 ms, TF peak 4,013,568 bytes and zero warm
current-allocation range. This is a worker/endpoint smoke, not a before/after
performance claim. Its worker subsequently changed to add more fixtures, so it
is intentionally ineligible for the final matched matrix.

## Scalar adjacent TT continuation

Run 00100 completed all 24 endpoint checks. Run 00101 passed the 13 mapped TT
checks after the observability-only finiteness correction. The structural
factory now has a cache identity regression test; its final rerun is pending.

The scalar adjacent TT route is being moved to one native date/ALS endpoint.
Source anchors were re-inspected: paper Algorithm 2 equations (15)/(16), local
text lines 695-720, and author `models/full_sol.m:73-125`. Its fixed ALS fit
remains the existing `extension_or_invention`. No source-faithfulness promotion
is sought. Reports and per-update hashes are constructed from completed tensor
histories, and the original t=1 warm-start rule is retained even when t=0 fits
an adjacent state.

Review rejected substituting the other TT route's CholeskyQR2 backend: the
scalar route allows condition vetoes through 1e16, beyond that backend's Gram
conditioning range. A plain full-rank QR candidate passed initial primitive
tests in run 00102, but code review found that Eigen's original complete
orthogonal decomposition truncates numerical rank. The replacement now uses
native column-pivoted Householder QR and a complete orthogonal decomposition
with Eigen's epsilon*column-count pivot threshold. Anchors are TensorFlow's
bundled `matrix_solve_ls_op_impl.h:146-160`, Eigen
`ColPivHouseholderQR.h:375`, and
`CompleteOrthogonalDecomposition.h:455-570`. Solver identity is recorded as
`tensorflow_native_complete_orthogonal_decomposition`; it never claims to be
the old `lstsq(fast=False)` implementation.

Run 00105 passed 20 primitive tests, including rank-truncated and rank-deficient
baseline parity, graph/XLA execution, and high-condition full-rank diagnostics.
Runs 00104/00106 exposed a TensorFlow conditional-gradient capture assertion.
Run 00107 passed the graph value/score baseline comparisons and rejection
tests, but XLA differentiated detached SVD diagnostics; those outputs are now
detached as they were at the original host boundary. Run 00108 then exposed
variant FakeParam gradient state under conditional ALS. Runs 00109-00111
preserve failures while localizing eager preparation captures in the branch
pullback. No threshold or scalar has been relaxed; F05 remains open pending
complete XLA, final-source parity and measurement evidence.

Runs 00112--00114 localized the remaining TensorFlow FakeParam variant state
to reporting and fixed basis calculations. Residual/pre-update snapshots are
now detached explicitly, preserving their original reporting-only role. Fixed
basis tables are prepared outside the parameter tape; the native core/date
recurrences remain differentiable. Run 00115 passed the focused XLA score.
Run 00116 exposed a zero-iteration date-gradient buffer at T=2 and cancellation
from forming basis outer products before the square-root core contraction.
The T=2 static shape case avoids the empty loop; retained marginal evaluation
again contracts each basis/core before the outer product. Run 00117 passed all
11 scalar checks at the original thresholds. Run 00119 passed all 12 checks,
including T=3/T=6 baseline value/score parity, identical graph node counts and
one trace per fixed shape. No marginal floor or tolerance was changed.

Run 00118 passed 21 QR/COD tests, including the actual COD pullback and the
public fit backend dispatch/metadata consistency check. The fixed solver API
now accepts and actually executes the native backend it reports. Additional
rectangular/multiple-response coverage was added while migrating geometry.

Run 00120 passed the factor-correlation geometry suite after removing direct
and indirect NumPy, native tensor encode/decode, replacing implicit pfor with
a parameter-direction TensorFlow loop, and adding a bounded-cache stable XLA
boundary around preparation, L-BFGS and fit diagnostics. This preserves the
original objective, anchors, thresholds and fit rejection rules. HMC tuning
interface and its public capability registry were re-inspected; no HMC tuner
or sampler mathematics changes are included. GPU and matched baseline evidence
remain pending. All F01--F20 remain open; no merge is authorized by these
subset checks alone.

## Recovery and auxiliary consumers

Runs 00122--00132 extended rectangular COD coverage, confirmed APF execution,
and exercised complete retained-moment, particle, Algorithm-1 and scalar-TT
endpoints. The first particle observation-gradient smokes failed XLA because
nested map functions created dynamic TensorLists. Fixed dense-buffer row loops
and batched covariance stabilization repaired those paths; runs 00126/00129
passed. Run 00131 executed the complete scalar-TT value/score on GPU2. These
are endpoint smokes, not a performance comparison or canonical LEDH admission.

Run 00132 passed 28 Algorithm-1/solver checks. Run 00133 passed all 25 OT and
structural endpoint checks after correcting the cache test's callback fixture.
Run 00134 passed the campaign guard checks, including matching exact measurement
sets for documented tradeoff reviews and refusing to waive ongoing memory
growth. No investigation has been accepted by a review artifact yet.

Fixed-center curvature now uses immutable TensorFlow outputs and compiled
dense fitting, matrix comparison and shrinkage kernels. Host validation checks
raw buffer layouts before tensor conversion, retaining exact overlapping-view
rejection, disjoint interleaved views, and signed-zero-normalized copied-row
rejection. The interval helper describes storage only; it does not compute
numerical arrays. Existing thresholds, selection order and audit-only veto are
unchanged. Run 00135 passed 120 checks but exposed the dense solver's missing
dynamic-row support; run 00136 passed all 121 fixed-center, buffer and downstream
curvature checks after repairing it.

Block-score geometry no longer imports NumPy or uses its epsilon helper.
Packed symmetric score design and unpacking are tensor operations shared with
the sequential initializer. A stable XLA factory encloses each declared block
fit, preserving the rank and SPD vetoes. Run 00137 passed its existing suite.
The heterogeneous block/replicate reporting and initializer controller are
still host orchestration; these checks do not certify a whole HMC tuner.

COD now supports one dynamic-row signature across underdetermined and tall
systems. Run 00138 exposed XLA compiling an inactive pullback branch with a
rectangular triangular solve; shape-valid padding fixes that inactive branch
without changing the selected full-rank formula. Run 00139 passed 25 COD checks,
including one-trace dynamic-row primal/FD-gradient checks in graph and XLA.
The wider TT suites must be rerun against the final shared solver.

Run 00140 passed the sequential initializer suite after replacing its direct
and indirect NumPy dependence with TensorFlow and standard-library scalar
validation. Its optimizer-wide runtime qualification remains outside F18's
auxiliary migration scope. Two other initializer modules still depend on
NumPy PCG64 probe-cloud streams; exact preservation versus an explicitly
versioned TensorFlow stream remains unresolved. No stream was silently changed.

Run 00141 passed the complete GenUT CPU/XLA measurement smoke. Run 00142 tests
the Contract E builder. Baseline builder smokes, the final matched GPU matrix,
source exemptions and stable-signature guards remain pending. F01--F20 remain
open, and the branch must not merge while those gates are incomplete.

## Geometry stream decision and recovery

Run 00142 passed the complete Contract E CPU/XLA smoke. Run 00143 preserved a
baseline GenUT tracing failure (ForwardAccumulator/ensure_shape TraceType).
Runs 00144--00146 developed a provisional exact PCG64 compatibility stream;
the last passed nine diagnostic checks. The owner then explicitly selected a
versioned TensorFlow stream. The provisional source and tests are removed;
their run records remain historical engineering evidence. Read-only upstream
NumPy source inspection during the provisional work did not change packages.

The replacement `geometry_tf_philox_cpu_xla_v1` fixes Philox, CPU/XLA placement,
FP64 draws, SHA-256 seed/call/domain mapping and per-call shape. The approved
change alters seeded draws, so frozen common clouds are required for numerical
baseline comparisons. No distribution, fit threshold or holdout role changes.
Skeptical review identified accidental comparison of different random inputs
as the principal risk; stream metadata and frozen-cloud regression checks
address it. Reproducibility binds the recorded TensorFlow version.

Run 00147 passed 13 quadratic checks and failed three. Two failures require
restoring Python scalar reporting types. The other expected exact hashes
between differently shaped scalar/batch XLA reductions despite already passing
1e-12 numerical comparisons. Exact artifact hashing is retained; reproducible
hashes are now checked within each route and scalar/batch numerical tolerances
remain unchanged. This is not a waiver of numerical parity.

## Geometry and signature recovery through run 00161

Run 00148 preserved the missing XLA StatelessShuffle kernel. Fisher-Yates with
unbiased integer rejection replaced that primitive; run 00149 passed all four
new-stream tests. Run 00151 passed the initializer suite, and run 00155 passed
five checks including the default XLA locator. Runs 00150/00152 exposed both
old seed-specific fixture assumptions and FP64 eig/SVD residuals up to 1.5e-7.
Explicit XLA Jacobi precision and shape restoration repaired those residuals;
run 00154 passed 22 quadratic tests, including complete pinned-baseline parity
with identical frozen clouds, rank deficiency, and budget/eligibility checks.
The original NumPy rank cutoff and numerical tolerances were preserved.

Run 00158 passed 30 signature/Kalman/moment-teacher checks after fixing a new
fixture's missing time axes. Run 00159 passed the Algorithm-1/solver group,
including its new observation-gradient case. Further review found that the
signature wrapper rounded Python lists to float32 before float64 entry points.
Endpoint dtype declarations now preserve list precision and index/mask types.
Run 00160 passed 31 checks; one new test mishandled IndexedSlices returned by a
gather gradient. Its assertion now materializes that gradient as a tensor.

The old Contract E primal/JVP date recurrences and scaled TT forward replay
are independent reference helpers, not the runtime factory implementations.
Their documentation now makes that role explicit. Consumer discovery found
one obsolete July canonical-selection script still using the primal helper;
new launches are disabled under the August 21 invalidation. Historical source
and artifacts remain preserved. The active fused recurrence's small parameter
basis comprehension was also replaced by a tensor diagonal construction.

The active Student-t defensive-floor configuration selector still had an
80-step Python numerical bisection. Its unchanged interval, formula, iteration
count and selection criterion are being migrated to one native XLA loop.
This is a mechanical repair to an existing local extension, with no new
Zhao-Cui source-faithfulness claim. Focused scalar-baseline tests are required.

## Frozen moment-hint consumers and reset checks

Run 00161 passed all 32 signature checks. Run 00162 passed the Student-t
selector/reference checks without changing its interval, formula or threshold.
Runs 00163--00165 isolated two moment-hint defects: tracing an empty T=1
history update, and XLA eigenvectors producing GH weights with about 1e-9
error despite accurate eigenvalues. A static T=1 branch avoids the invalid
update. Weights now use the equivalent normalized Hermite recurrence
`w_i=1/(n*p_{n-1}(x_i)^2)` at the same nodes. Tightening the eigen epsilon
alone did not fix the vectors; that failed trial remains recorded.

Run 00166 passed all 12 GH/moment checks at unchanged tolerances, including
orders 2/9/17 and n=1/2 full histories. Run 00167 passed 19 checks, adding
native Kalman joint hints on identical legacy PCG fixtures and a complete
adapted TT consumer. Compiled consumers now index frozen TF histories rather
than executing stateful NumPy callbacks. Fixture generation and independent
oracle code remain explicitly diagnostic. The LGSSM runtime density callback
also no longer executes NumPy factorization.

Run 00168 attempted the old phase-3 reset suite and found its July certificate
files absent; those historical LEDH results are also ineligible under the
August invalidation. They were not restored, regenerated or used as baselines.
The campaign reset group now uses fresh independent cloud fixtures and checks
the primitive value, covariance identities, all five analytical pullbacks,
JVP/VJP duality, finite differences and enclosing XLA. Run 00169 passed these
checks. This is primitive engineering evidence, not canonical LEDH admission.
The stable-signature wrapper also restores its truthful `_jit_compile=True`
introspection attribute for compatibility.

No finding is closed by these focused runs. Final source guards, repeated
matched GPU measurements and the full current-source test gate remain pending.

## Public fitter and TT algebra recovery

Run 00170 passed the compiled Gaussian consumer group. Runs 00171--00172
identified stale solver metadata and a missing test import in the public ALS
migration. Run 00174 passed the new frozen-reference/cache checks after core,
point, weight and holdout values became explicit signature inputs. Runs 00173
and 00175 each passed all 39 TT algebra/analytical derivative checks. Full
current-source fitter and consumer groups remain required. The source guard
is under construction; unreviewed entries remain violations, not exceptions.

The public squared-density migration preserves its paired contraction and
33-point suffix trapezoid rule. Before implementation, paper Eq. (13),
Proposition 2 and conditional construction were reread at
`.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539`
and author `@TTSIRT/marginalise.m:25-85` in the vendored source. The paired
normalizer/marginal operation retains its existing paper relationship; the
finite-grid conditional remains an existing `extension_or_invention`, not the
paper's algebraic CDF construction. This is an execution refactor with no new
source-faithfulness claim. Frozen original-source numerical tests check both
measures, heterogeneous degrees, empty/full marginals and every conditional
axis; enclosing HLO and bounded graph size check compilation.

## Recovery through run 00191

Runs 00176--00178 repaired and passed the squared-density group. Run 00179
passed the public fitter group. The TT map group failed in 00180 and passed
in 00181; simulation retries 00182--00185 ended in a pass. Runs 00186 and
00187 passed preparation and SGQF. TTSIRT runs 00188/00189 passed; broader
00190 exposed CPU XLA's unsupported determinant operation in affine maps.
The existing native slogdet primitive repairs it. Run 00191 passed 22 checks,
including affine roundtrip/determinant and empty suffix transport shapes.
Earlier failures remain recorded and consume budget. The three unfinished
Gaussian benchmark callers now obtain callbacks via PreparedMomentHints.callbacks.

Scalar retained-grid repair was reviewed against paper Algorithm 2/(15)--(16)
and author models/full_sol.m:73--125. It preserves the existing
extension_or_invention classification, one-axis weighted ALS, raw propagation
weights, max-shift derivative, and explicit model parameter-score methods.
All parameter columns share a matrix solve; no AD replacement is permitted.
The legacy value moments are unnormalized sums while derivative moment reports
use quotients. This discrepancy is preserved, not silently numerically repaired.
Pinned-source parity, finite differences, bounded graphs, and veto checks are
required before this endpoint can close any finding. F01--F20 remain open.

## Scalar, panel and source-guard recovery through run 00197

Run 00192 failed during scalar test collection; 00193 passed 12 checks and
00194 passed the expanded 15-check group. Run 00195 passed all nine retained
panel checks. Run 00196 passed the 15 scalar checks again after runtime model
binding and report refactors. Run 00197 passed TTSIRT with the heterogeneous
constant/correlated/constant transition schema regression included.

Panel checks cover exact transformed and KSC mixture endpoints, heterogeneous
gamma/beta/sigma, pinned-source values and reordered analytical scores at
1e-10, independent finite differences at 2e-7, coordinate report semantics,
changed-input signature reuse, HLO and width-independent graph size. These
remain local retained-grid extensions, not Zhao-Cui source-faithfulness or
batch-native NeuTra training evidence.

The exact source guard now participates in both the policy tests and terminal
merge gate. Its policy JSON is part of current-source provenance. Negative
tests exercise new/changed numerical loops, NumPy, host callbacks, disabled
JIT defaults and stale exceptions. Review exposed missing checks for explicit
non-XLA tf.function calls and bare decorators; those checks are now included.
Fifteen individually reviewed graph boundaries were classified: thirteen
private enclosed graphs (with named enclosing endpoints and compilation tests)
and two explicit Algorithm-1 graph-reference paths. None exempts numerical
Python iteration. The guard covers 136 sources and 740 exact exceptions;
selected scopes in mixed modules remain a qualification limit.

The stronger guard found a remaining non-XLA start-bank selector. After reading
the HMC tuning interface and capability registry, only its compilation default
was changed; chronological greedy eligibility, endpoints, thresholds and
public tuner authority remain unchanged. A new focused group includes the
existing frozen oracle and boundary tests plus HLO/signature checks. Full
current-source tests, remaining classifications and matched repeated GPU
measurements are still required. No F01--F20 closure or merge is claimed.

## Recovery through run 00201

Run 00198 passed the start-bank group, including the frozen oracle and enclosing
XLA checks. Run 00199 passed all 25 campaign/policy checks. Run 00200 passed
the TT algebra group. Run 00201 passed the first squared-density GPU/XLA
measurement at size 1. This single measurement is qualification, not a repeated
performance result.

Recovery verified 201 recorded attempts and no active numerical worker. Charged
time is 12014.481 CPU seconds and 273.560 GPU seconds, leaving 16785.519 CPU
seconds and 14126.440 GPU seconds under the original budget. GPU2 was idle
(18 MiB driver reservation, zero utilization) before run 00202. The eight newly
registered numerical fixtures still need baseline, graph and XLA qualification
and matched repeats. All findings remain open.

## Recovery through run 00214

Runs 00202--00208 passed candidate GPU/XLA qualification for TTSIRT preparation,
the SV/SIR/predator-prey simulators, scalar/panel retained filters and panel KSC.
Runs 00209--00210 passed candidate GPU graph qualification for squared density
and TTSIRT preparation. Run 00211 failed on the simulator's GPU uint64 AddN;
signed int64 counter addition between bitcasts preserves modular arithmetic
and passed the graph retry in 00212. Run 00214 passed the seeded-simulator
regression group. Run 00213 recorded a pinned-baseline XLA tracing failure:
Generator.from_seed creates a variable inside repeated tracing. That failure
has no timing comparison; the valid eager baseline must be measured separately.

Recovery verified 213 attempts: 12014.481 CPU seconds and 414.857 GPU seconds.
The source guard still has incomplete scopes in mixed filtering modules.
Direct exact-transformed SGQF and other active SV component recurrences are
being added to its coverage as they are repaired, with pinned-source parity,
analytical finite differences and enclosing compilation checks. A selected-scope
guard pass cannot close the repository-wide audit. No finding is closed.

## Recovery through run 00216

Run 00215 passed all six direct exact-transformed SGQF checks, including pinned
value/score/history parity, every analytical score column against finite
differences, stable signatures, HLO and bounded graphs. Run 00216 passed the
campaign/policy group. These are focused checks; subsequent source edits mean
they do not satisfy the terminal current-source gate. The augmented-noise
SGQF wrapper and remaining component recurrences are now being migrated,
preserving their distinct value and diagnostic/analytical score definitions.

## Recovery through run 00219

Run 00217 passed all nine direct/augmented SGQF checks. Expanded mixture tests
in 00218 exposed a fixture keyword error and a float32 literal in the float64
derivative branch; both were repaired. Run 00219 passed 24 checks and failed
five graph-reference cases. Every tested XLA mixture value/score, derivative
history, metadata, finite difference and bounded-graph check passed. The
one-coordinate graph cases return malformed tensors (empty float32 outputs
instead of float64 histories); this is unresolved and is not excused by the
passing XLA cases. Optimizer localization is the next bounded diagnostic.

At recovery, 219 attempts consumed 12171.107 CPU seconds and 414.857 GPU
seconds of the original 28800/14400-second budgets. F01--F20 remain open.

## Scalar graph optimizer repair through run 00226

Runs 00220--00225 localized the malformed scalar graph outputs. Optimizer
options merge rather than replace, so independent probes explicitly reset all
five investigated options. Disabling arithmetic, constant or dependency
optimization restored the Kalman outputs; disabling function or loop
optimization did not. Fixed output buffers and hoisted mixture constants were
insufficient. Removing redundant 1x1 transposes, while preserving the original
add-and-scale symmetrization arithmetic, repaired Kalman and the shared
sigma-point/SGQF recurrences with normal optimizer settings. No global optimizer
disable or no-inline boundary is retained.

Run 00226 passed all 37 SV checks. Temporary monkeypatch/optimizer localization
tests were then removed; default-optimizer graph/XLA endpoint parity remains.
The bounded-graph test now compares two date/component extents within each
scalar/matrix specialization, since removing scalar transposes legitimately
changes the graph size across those two static shapes. It also verifies finite
differences, changed-input signature reuse and HLO. Numerical tolerances are
unchanged. The shared symmetrization change still requires affected Kalman and
SGQF consumer confirmation.

The remaining dense scalar and Gaussian-to-TT loops in `filtering.py` are
numerical work, not artifact-only loops. Their native migration preserves the
existing local quadrature/ALS extension. Reinspected anchors are paper
Algorithm 2/(15)--(16), extracted lines 693--725, and author
`source/models/full_sol.m:73--125`. The author uses adaptive TTIRT/TTSIRT;
this repair makes no new source-faithfulness claim. Public endpoint parity,
unchanged vetoes and bounded enclosing compilation are required.

## Recovery through run 00232

Run 00227 passed 27 filtering-wrapper checks. Runs 00228 and 00229 each passed
17 Kalman checks and failed the same SVD-CUT graph-autodiff Hessian check on
the candidate and pinned baseline respectively. Values and first derivatives
agree, but the graph Hessian is NaN. This pre-existing failure remains a repair
trigger. Run 00230 passed 29 filtering-wrapper checks after restoring the
Gaussian TT normalizer veto. Run 00231 passed 24 squared-density/chunk checks,
including partial-block arithmetic, pinned-source parity and enclosing HLO.
Run 00232 passed all 26 SGQF value, score, integration and consumer checks.

Recovery status: 232 attempts, 12811.444 CPU seconds and 414.857 GPU seconds
charged against the original 28800/14400-second budgets. The next Kalman
diagnostic isolates nondifferentiable reporting norms from recurrence-state
cotangents; no likelihood, point-placement or covariance derivative is stopped.
All F01--F20 findings remain open pending terminal current-source evidence.

## Recovery through run 00235

Runs 00233 and 00234 passed all 18 Kalman checks. Reporting-only zero residual
norms contaminated the graph Hessian through undefined norm derivatives.
Stopping those diagnostic residual cotangents repaired the Hessian; likelihood,
covariance and sigma-point derivatives remain live. Run 00234 additionally
checked the Hessian against a centered finite difference of the compiled score
without relaxing tolerance. Run 00235 passed 30 filtering-wrapper checks,
including bitwise pinned-source parity of heterogeneous default TT cores.

The full mixed-module audit distinguishes completed result assembly, fixed
schemas and host validation from numerical recurrence. Remaining all-axes
multistate retained-grid loops are explicitly historical under AGENTS.md.
The old leaderboard helper `_zhao_cui_predator_prey_tt_value_score` has no
callers; the active dispatch rejects the retained-grid route. Existing HMC
route-policy regression tests are added to the campaign wrapper group.
The exact-node guard is being extended to both complete mixed modules and
their new native implementations. This classification grants no scientific
or HMC admission to a reference route.

Through 00235 the original budget has consumed 12965.982 CPU seconds and
414.857 GPU seconds. All findings remain open for terminal evidence.

## Recovery through run 00237

Run 00236 passed the campaign/policy group. Run 00237 passed all 29 retained SV
tests after removing temporary optimizer-localization probes. The exact-node
guard covers 139 sources and 818 exceptions, including the complete mixed
filtering and SV modules. The expanded wrapper group is running as 00238.

Benchmark review still finds missing complete SV, dense/Gaussian wrapper and
frozen-cloud geometry coverage. These fixtures must be added before freezing
the harness for repeated measurements. A graph diagnostic for the mixture UKF
must disable its nested recurrence compilation explicitly; an outer non-XLA
wrapper alone is insufficient. This is a diagnostic switch with XLA still the
default. Full current-source tests and matched measurements remain required;
no finding closure or integration is authorized by these focused passes.

## Recovery through run 00254

Run 00238 passed 40 wrapper and route-policy checks; 00239 and 00241 passed
the campaign/policy group. Run 00240 qualified the direct SV score on GPU/XLA.
Runs 00242--00250 completed its two-size qualification: the baseline's host
`.numpy()` validation prevents graph/XLA tracing, its eager reference passes,
and both candidate modes preserve the values, score and histories. Runs
00251--00254 record the same baseline limitation for augmented SGQF; its eager
reference and candidate graph mode agree. The matrix then stopped before
another launch on GPU2's 7% utilization reading (18 MiB reservation). No
contention threshold was relaxed.

The endpoint fixtures and sequential matrix now cover the remaining SV,
dense/Gaussian filter and frozen-cloud quadratic-fit calculations. Review found
that the mixture UKF's diagnostic switch stopped at an inner recurrence. The
switch now propagates through both value and principal-square-root score
calculations, preserves XLA defaults, and labels non-XLA score metadata as a
reference exception. A regression checks pinned-source parity and the complete
graph for nested XLA. The value UKF fixture is added alongside its score fixture.
These changes require fresh qualification and final current-harness repeats.
The quadratic fixture covers the numerical fit, not the whole host-controlled
initializer. All findings remain open.

## Recovery through run 00258

Run 00255 passed 31 SV checks and failed two UKF value cases because the
intermediate likelihood wrapper omitted the diagnostic JIT keyword. The
wrapper now propagates it; run 00256 passed all 33 cases, including pinned
value/score/history parity and full-graph inspection for nested compilation.

Runs 00257--00258 preserved the direct-SV baseline graph failure and successful
eager reference under the updated fixture harness. The matrix again stopped
on GPU2's 7% utilization sample after its own worker exited, with reservation
back at 18 MiB. The driver now waits for two consecutive idle samples, with
at most six samples and ten seconds of intervening waits. The original
100 MiB / 5% thresholds remain unchanged; persistent contention still vetoes
launch. Preflight samples are recorded in each subsequent matrix run. Focused
tests check stale utilization recovery and both original veto thresholds.

## Recovery through run 00284

Run 00259 passed the campaign/policy group. Runs 00260--00277 qualified
direct-SV and augmented-SGQF at both extents: baseline graph/XLA attempts fail
on host validation, its eager execution passes, and candidate graph/XLA
values, scores and histories match. Runs 00278--00282 qualified the first
mixture-SGQF extent with maximum absolute discrepancy 2.22e-16. Run 00283
preserves the second-extent baseline graph failure; run 00284 hit the unchanged
300-second limit in the eager baseline, with repeated component-filter tracing
and compilation. No timing or parity result is inferred from that timeout.

The mixture-only timing fixture is now a scalar panel at T=1 and T=2, keeping
the same frozen components, model, quadrature, numerical tolerances and twenty
warm calls. This bounds the legacy repeated-compilation cost. Wider-panel
correctness remains covered by the SV tests; no wider-panel performance claim
is authorized. All timing repeats must use the revised harness in both arms.

Review found that the test gate accepted skipped tests when pytest returned
zero. It now requires readable, nonempty JUnit results without skipped, failed
or errored cases. The GPU categorical stream comparison has a dedicated GPU
group and contention preflight; CPU tests no longer silently skip that check.
Three static-name defects in touched files were also repaired: the attempt04
TensorFlow import, frozen-APF reporting variable and Lane-B ProductBasis type.
These do not change numerical algorithms. All findings remain open pending
complete current-source checks and repeated measurements.

## Recovery through run 00343

Run 00285 passed the campaign/policy checks. Run 00286 qualified the larger
quadratic fit on GPU/XLA with complete HLO, one trace, no callbacks and stable
warm allocator current. Runs 00287--00336 qualified both extents of direct SV,
augmented SGQF, mixture SGQF value/score and mixture Kalman. The legacy graph
and XLA attempts fail on host validation; their eager references pass and
candidate graph/XLA outputs meet the unchanged FP64 parity criterion. Runs
00337--00341 qualified the first CUT4 extent; 00342 preserves the baseline
graph failure and 00343 completed its eager reference.

The launcher stopped between workers on its source-change guard when the
existing target-failure consumer tests were added to the required inventory.
Review found a Gaussian-TT benchmark output indexing defect: packed core
history has shape [dates, core_count, packed_width]. The fixture now reshapes
the first core to the legacy [dates, 1, 7, 1] output. This changes harness
provenance and requires fresh pairs; no old pair is silently reused. Direct
GPU jobs now receive the same contention preflight as matrix jobs, and parity
logs identify the actual eager reference separately from failed graph attempts.

Through 00343 the campaign has consumed 13361.483 CPU seconds and 1788.036 GPU
seconds of the original 28800/14400-second budgets. All findings remain open.

## Recovery through run 00391

Runs 00344--00363 qualified CUT4, dense/Gaussian filtering and the frozen-cloud
quadratic fit. Gaussian-TT graph/XLA outputs matched the eager baseline within
4.44e-16; the quadratic fit matched within 9.77e-15. The latter remains a fit
benchmark, not whole-initializer qualification. Baseline symbolic-tracing
failures remain recorded. Runs 00364--00388 passed rectangular, factor,
covariance, Sinkhorn, SQMC and baseline DNS checks. Candidate DNS graph mode
failed in 00389 because its quadrature helper forced nested XLA. The diagnostic
JIT option now reaches DNS and retained-moment quadrature; defaults remain XLA.

Further review confirmed non-XLA TT core/row preparation outside the compiled
recurrences. Compiled preparation now preserves the original Philox words,
BoxMullerDouble floor/output order, Sobol points, seed salts and Christoffel
transformation. No random-stream migration is applied to these TT algorithms.
Runs 00390/00391 passed independent draw/row/weight/status parity, complete HLO,
stable signatures and date-independent graph size. The full adapted/Gaussian
TT consumer group is running as 00392. Before this mechanical repair, paper
Algorithm 2/(15)--(16), extracted lines 693--725, and author
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:73--125`
were reinspected. Existing fixed-row ALS remains a local extension; this
repair introduces no source-faithfulness claim.

Runner review also found that KeyboardInterrupt could abandon a live worker
and its attempt record. Interrupted workers are now terminated and charged
their observed duration with exit 130. The harness changes invalidate previous
measurement pairs for final comparison; fresh current-harness pairs are
required. All F01--F20 findings remain open.

## Preparation review through run 00430

Run 00392 passed all 16 mapped-TT consumer checks; 00393 passed the policy and
runner checks, and 00394 passed the primitive/retained-moment group including
the non-XLA boundary regressions. Fresh qualification 00395--00428 preserved
Kalman/SRUKF, Sinkhorn, SQMC, DNS and the first retained-moment graph outputs.
DNS's corrected graph/XLA comparisons matched within 1.25e-16. The matrix was
interrupted during GPU preflight after 00428, with no active worker lost.

The wider preparation review found eager basis recurrences in public TT-fit,
fixed marginal and scalar-retained preparation. They now compile by default,
with point values as signature inputs and native inlining inside an enclosing
graph. The exact guard covers 140 sources and 815 schema/host/reference/graph
exceptions. Run 00429 exposed missing required MeasureConvention fields in a
new test fixture; 00430 passed all 14 preparation checks after that test-only
repair. Existing numerical tolerances, ranks, row laws and fitting objectives
are unchanged. Affected complete scalar/fitter checks and the remaining full
matrix are still required; no finding is closed.

## Recovered scalar reference through run 00433

Run 00431 completed with seven passes and five failures in the old scalar
adjacent-TT oracle: that test imported the frozen filter module but used the
current fitter and density dependencies. Its eager gradient tape then crossed
the newly compiled preparation boundary. This mixed implementation was not
the pinned baseline and cannot be used to judge numerical parity.

The scalar reference now runs in a separate CPU process with the complete
baseline Python package and its original fixture builder extracted from Git.
It uses the same locally installed Sylvester library as the current process.
Run 00432 preserved an import failure due to the absent untracked library in
the snapshot; the explicit library path repaired that harness defect. Run
00433 passed all 12 scalar checks, including independent baseline values and
scores, transition-before-first-observation behavior, condition vetoes,
signature reuse and horizon-independent graph size at the original tolerance.
No numerical implementation or tolerance was changed by this repair.

Four nested-graph exception records now name the actual
`native_fixed_tt_fit.program` enclosing endpoint instead of its obsolete
`evaluate` name. Retained scalar, fitter and policy checks precede resuming
the matched measurement matrix. All F01--F20 remain open until the full
current-source gate passes.

Runs 00434--00436 passed 15 scalar-retained, 37 fitter and 38 campaign/policy
checks. Runs 00437--00450 qualified retained moments at both extents and SGQF
derivatives at the first extent plus the larger graph calculation. SGQF
baseline XLA failed on its original TensorListReserve; its valid graph and
eager comparisons are preserved. The repaired XLA derivative matched within
4.44e-15 at the first extent; the larger graph difference was 8.88e-15.

Review found that the matrix used eager as the reference after every baseline
XLA failure even when the corresponding graph arm had passed. The comparison
report already preferred graph in this case. The controller now applies the
same order, preserving the failed XLA attempt separately. Run 00452's redundant
eager worker was interrupted and charged 157.342 seconds; no result is claimed.
Run 00453 passed all 40 campaign/policy checks, including both branches of
reference selection. The numerical harness is unchanged, so existing matched
measurements retain their provenance eligibility.

Runs 00454--00472 completed both SGQF derivative, joint-target and GenUT
extents in graph/XLA. GenUT's pinned symbolic-tracing failures remain explicit;
its repaired outputs agree with the isolated eager baseline to 7.22e-16.
Run 00473 was rejected by the import-provenance guard: the frozen snapshot
omitted the two `experiments` parent package markers, so Python selected the
live regular package. Its log is retained, but it supplies no comparison.

Snapshot preparation now includes both parent `__init__.py` files from the
same pinned commit. Existing snapshots receive only those missing markers,
with their hashes added to the manifest; numerical baseline files are neither
changed nor replaced. Run 00474 passed all 41 policy/runner tests, including
new-snapshot creation, old-snapshot recovery and import isolation with the live
checkout on the search path. The benchmark harness itself is unchanged.

## Static quadrature regression repair through run 00504

Runs 00475--00502 passed Contract E, generic TT and adapted TT qualification
at both extents, with the baseline's symbolic failures recorded. Maximum
observed absolute discrepancies were 8.88e-16, 3.02e-14 and 3.39e-13,
respectively, within the unchanged comparison criterion. Run 00503 preserves
the Gaussian-TT baseline graph failure; the matrix then stopped at its
source-change guard, between workers.

An early inspection of the first-repeat metrics identified roughly 16x warm
DNS slowdown and 4--6x retained-moment slowdown. The repaired quadrature helper
was recalculating its parameter-independent Jacobi eigenproblem on every
kernel call; the original NumPy rule had been evaluated only during tracing.
The helper now caches bounded order/dtype/JIT-specific tensor constants after
compiled TensorFlow preparation, lifted out of enclosing graphs. No dynamic
model, fit or score input is cached, and no numerical tolerance is changed.
Static preparation can occur during the first consumer trace and its cost is
included in that trace timing. Run 00504 passed all 24 primitive/moment checks,
including preparation HLO, numerical derivatives, and reuse across separate
consumer graphs. Fresh DNS/retained measurements and downstream checks remain
required before accepting the performance repair.

The bounded driver also now supports `pause` through the existing approval
prefix. It preserves the active worker and checks the request before launching
another one; an explicit matrix restart clears the consumed request. This
avoids terminating numerical work merely to inspect or repair a later stage.

Run 00505 passed all 42 policy/runner checks, including the pause boundary.
Runs 00506--00509 preserve DNS parity at both sizes. The first-size XLA warm
median is 0.4185 ms versus the pinned baseline's 0.4161 ms; this removes the
earlier roughly 16x slowdown. Runs 00510--00513 preserve retained-moment
parity. Their XLA warm times improve from the earlier 4--6x slowdown to about
2x the tiny unrolled baseline, but that remaining regression is still open for
three-repeat investigation. All eight new candidate runs have zero warm
allocator-current range. These are single-process qualification measurements,
not the final repeated comparison or finding closure.

## Compatibility recovery through run 00523

Run 00514 passed 14 preparation checks. Runs 00515--00516 requalified the
first joint-target extent; the matrix then stopped on GPU2 contention from
an unrelated MacroFinance process. Runs 00517--00519 passed 40 filtering,
33 SV/SGQF and 13 start-bank tests. Run 00520 failed three of 32 signature
checks because the mechanics-reference moment teacher still calls
`SquaredTTDensity._defensive_marginal_values`, removed in the execution refactor.

The helper is restored through the compiled density dispatcher. Its operation
shares the existing marginal defensive term and returns before evaluating an
unneeded TT normalizer. Full-axis, reference-measure, Lebesgue-volume, empty-axis
and unsupported-custom-density behavior is checked against the pinned source.
This restores the existing local extension; it adds no source-faithfulness
claim. Paper Proposition 2/(14), local text lines 594--626, and author
`third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25`
through its final defensive normalizer were reread before the repair.
The explicitly diagnostic moment-teacher reference retains its reference role.

Runs 00521--00523 passed 32 signature, 31 squared-density and 42 policy/runner
checks. No policy exception or comparison tolerance was added. GPU2 was idle
on recovery; its contention gate remains active. Retained-moment optimized HLO
still contains basis-evaluation loops inside the axis recurrence; this is a
candidate explanation for the remaining timing regression, not a resolution.
The full current-source tests and repeated comparisons remain pending.

## Complete public boundaries through run 00529

Runs 00524--00526 passed preparation, filtering-wrapper and SV/SGQF checks.
The matrix was paused between workers for two additional review findings:
the public squared-TT normalized methods performed their last exponential or
division eagerly, and the public retained-moment helper compiled quadrature
but relied on its caller to compile the complete moments. Both are execution
gaps even when an enclosing benchmark function compiles successfully.

Normalized squared-TT results now complete inside the existing dispatcher.
Retained moments now use a bounded, explicit-signature cache, with core values,
suffix Gram matrix, defensive weight and normalizer passed as live tensors.
The moment formulas and quadrature rule are unchanged. Four existing exact
shape-packing exceptions moved with the numerical body; four new exact entries
cover shape inspection, TensorSpecs and object/tensor packing in its wrapper.
No numerical recurrence is exempted. Run 00527 passed 26 primitive/moment
checks, including dense-quadrature parity after changing every numerical
input family. Run 00528 passed 33 squared-density checks, including complete
normalized-result HLO and unchanged frozen-source parity.

Fresh remote fetch found no change to `origin/main`. Unrelated worktree edits
to HMC preparation/tuning were observed and preserved; they are not campaign
repairs. The matrix source guard remains active.

Run 00529 exposed five harness-test failures caused by controller unit tests
reading the live campaign's pause request. The affected tests now use temporary
output roots; run 00530 passed all 42 checks with the real campaign still paused.
Runs 00531--00533 passed preparation, filtering-wrapper and SV/SGQF checks before
another deliberate pause for analytical-gradient review.

## Analytical adjoint and provenance review

Review found a missing pullback for the squared-TT relative Gram floor, also
present in the pinned source. If `A = G + rho * trace(G)/r * I`, its pullback is
`G_bar = A_bar + rho * trace(A_bar)/r * I`. The forward program uses this
Gram-dependent floor, but the reverse program retained only `A_bar` from the
Cholesky pullback. This is a missing term of the declared total derivative of
the existing scalar, not a proposal to replace the score definition or filter.

Run 00534 retained two passing original full-path FD tests and demonstrated
two new failures. At the explicit diagnostic floor 0.1, the analytical score
was -0.6143184409037679 versus same-value-program FD 5.946229346998066. With fixed
defensive weight zero, its square-root pullback divided by zero even though
that amplitude is constant in the differentiated model parameters. The repair
adds the trace-floor pullback and a zero-safe division for that constant-zero
branch. No value formula, default control or tolerance changes. Run 00535 is
the verification attempt; paired default-control benchmark parity remains
required independently.

The comparison review also found that imported-source hashes were checked
against current files but not against the worker launch snapshot. A source
edit during a worker could therefore be attributed to already-loaded code.
Candidate comparisons now require both matches. New launch snapshots include
the two parent experiment package markers; old snapshots may use only their
identical pinned baseline marker hashes. Focused tests cover both mid-worker
changes and this bounded marker recovery. This changes controller validation,
not the measurement harness, fixtures or numerical comparison tolerances.

Run 00535 passed all four full-path adjoint FD checks, including both new
regressions, at the original 1e-6 FD criterion. Run 00536 passed 28 primitive
and retained-moment tests after batching independent Legendre evaluations.
This performance repair preserves each axis's original Gauss-Legendre order
and normalized-polynomial recurrence, including heterogeneous degrees; padded
weights are zero and inactive basis columns are masked. The complete moment
calculation uses one shared polynomial recurrence followed by tensor axis
contractions, with the existing branch implementation retained for other basis
families. No model parameter or derivative-bearing numerical input is cached.
Six exact fixed-schema entries cover rule binding, packing and type/shape
inspection; no numerical recurrence is exempted. Repeated GPU timing, memory
and default-control adjoint parity are still pending.

Run 00537 passed all 43 policy/runner checks, including measurement-launch
provenance. Runs 00538--00540 passed preparation, filtering wrappers and SV/SGQF.
The source guard then stopped the matrix because concurrent unrelated work
modified `bayesfilter/inference/hmc_preparation.py`. Those edits, and related
`hmc_kernel_tuning.py` edits, are preserved outside the campaign checkpoint.
Validation moves to a linked worktree to freeze its source. The driver derives
the artifact root from Git's common directory, so this isolation cannot reset
the budget, split the campaign lock or overwrite earlier evidence. No final
finding closure or merge decision has been issued.

## Isolated validation recovery on September 18

Campaign checkpoint `4e0d02b3` is checked out on
`repair/filter-gradient-xla-validation-20260918` in the linked validation
worktree. The original checkout's concurrent HMC preparation/tuning edits are
excluded and preserved. The local Sylvester op library was copied unchanged
(SHA-256 `cc2ad31f8eab27bb90449b7e77eae0c6aac34ab7ac4dfe4662510a0eb96a7022`).
Run 00541 passed all 44 policy/controller checks. Runs 00542--00551 passed
preparation, filtering wrappers, SV/SGQF, start-bank, signatures, Contract E
reset, Student-t selection, moment hints, QR and scalar TT groups. The matrix
then paused at a worker boundary for a controller provenance repair.

The comparison controller previously required the candidate's measured source
directory to equal the current checkout, discarding valid measurements when
validation moved to a linked worktree. Reuse now requires the measured directory
to match its recorded launch directory and share the current Git common
directory/campaign root. Every imported source must still match both its launch
snapshot and the current validation source. Different repositories, changed
code, mismatched launch roots and stale harnesses remain ineligible. This
localized repair changes no fixture, numerical kernel, tolerance or budget.

Runs 00552/00553 passed all 45 policy/controller checks. Qualification reused
unchanged Kalman/SRUKF and DNS evidence and refreshed the affected routes.
Runs 00554--00572 preserved Sinkhorn, SQMC, retained moments, SGQF derivatives,
joint-target and completed GenUT comparisons. The matrix paused after 00572.

Retained-moment runs 00561/00563 measured 0.5870/0.7424 ms warm XLA medians
against baseline 0.4302/0.5374 ms, respectively. Both had zero warm allocator
current range and errors below 8e-16. Optimized HLO still contained the fixed
basis-polynomial recurrence inside each moment call. The immutable Legendre
moment matrices now use a separate zero-input compiled preparation lifted out
of the complete moment graph. All numerical density inputs remain live. The
existing graph/XLA mixed-degree and changed-input parity test also requires
the prepared matrix capture. No new policy exception is required; repeated
measurements remain necessary before a performance conclusion.

## Additional execution closure found on September 18

The broader scan found omitted latent-SIR parameter derivatives and initial-RQMC
time recursion. Both now use native TensorFlow operations/control flow. The
SIR public tensor entries also use bounded fixed-signature XLA specializations.
Runs 00589--00591 exposed fixture defects, not accepted comparisons: the SIR
fixture used an invalid finite transport/reset configuration, and the RQMC
FP64 fixture did not supply dtype-consistent observation callbacks. The corrected
SIR fixture uses the existing reduced-model scale and 20/100 transport settings;
the RQMC reference arm explicitly binds the unchanged linear observation matrix
at FP64. These are new fixture qualification choices, not runtime retuning.
Run 00592 passed all 11 checks (174.90 process seconds): T=2/4 graph and XLA
value/score/history parity at 1e-10, HLO, no callbacks, and Cholesky/inverse
directional checks against finite differences. Pinned function bodies here are
local refactor oracles; isolated source-process timing remains outstanding.

This scan also found unguarded Contract E--TP recursions, moment-teacher
preparation and score directions, TT proposal/source-route operations, and
predictive helpers. Several are explicitly independent references or host
metadata, but classification is incomplete and the current 142-source guard
does not establish repository-wide compliance. F02/F04/F09/F11/F17/F19 remain
open for this wider closure. No merge or policy-completion claim is authorized
by the passing subset. In particular, the Contract E--TP helper contained
another Python-unrolled Kalman likelihood and backward information recursions;
their native-control-flow repair is now under focused validation. Run 00593
failed at test import before any numerical check; the corrected import is
retried within the unchanged campaign budget.

Run 00594 passed all five Kalman/backward-information checks, including
derivative finite differences, enclosing HLO and constant graph-node counts.
Run 00595 passed the policy/controller group after registering the additional
isolated measurement harness. Runs 00596--00609 passed paired SIR T=2/4
graph/XLA and RQMC T=2 graph/XLA plus T=4 graph qualification with exact
before/after values. Before run 00610, the driver stopped at its GPU2 contention
check: the MacroFinance dense-affine validation worker occupied the device.
No worker was killed or GPU contention limit changed. CPU/source work continues.
Cumulative charged time through 00609 is 15,978.406 CPU and 4,886.611 GPU seconds
against 28,800/14,400 seconds. Timing repeats and remaining additional fixtures
are still pending.

Further review identified `tf.experimental.numpy.finfo` in the Contract E--TP
chart diagnostics. Exact IEEE epsilon constants replace that indirect NumPy
call; the source detector now recognizes TensorFlow's NumPy namespace and its
import aliases. This repair preserves the existing FP16/FP32/FP64 epsilons and
does not add an unsupported dtype or change a chart threshold.

Run 00610 passed 14 checks and failed four multi-date XLA comparisons at the
chart condition-number diagnostic: XLA versus eager SVD differed by up to
approximately 1.1e-9 relative. All graph comparisons passed. The test had
compared XLA to an eager baseline, violating the campaign's matched-execution
contract for the diagnostic. The repair compares baseline and candidate in the
same graph/XLA mode at the unchanged 1e-10 threshold, and separately retains
eager-reference objective and gradient checks at that threshold. Fresh charts
are built once by a diagnostic LP and held fixed across all arms and finite
differences; no historical chart or LEDH result is reused. No mathematical
gate or runtime chart tolerance is relaxed.

Run 00611 passed all 18 repaired LGSSM Contract E--TP checks in 143.916 process
seconds. Predictive, progressive-score, full-continuation and finite-lookahead
recursions preserve their complete same-mode histories and scores at 1e-10;
objective/score also match the eager authority and directional finite
differences. The bounded core's graph-node count is independent of horizon.

The next repair batches UKF Gaussian projection across basis axes and replaces
seeded TT channel scatter loops with tensor masks. The original deterministic
channel rule, quadrature order, Gaussian projection and measure are retained.
This initializer remains `extension_or_invention`; the Zhao-Cui paper Algorithm
2/(15)--(16), text lines 693--717, and author `models/full_sol.m:64`--125 were
reinspected. Those anchors describe sequential approximation and coordinate
preparation, not this UKF initializer. No new source-faithfulness claim is made.
Legendre axes use one batched polynomial recurrence; other basis implementations
use fixed-schema dispatch inside a native axis map. Fresh tests compare mixed
degrees/domains, both measures, live frame changes and exact channel values.

Run 00612 found a tracing error in packing basis-domain tensors with
`tf.constant`. The repair packs the immutable bounds with TensorFlow before
tracing. Run 00613 passed all 24 focused and existing UKF-initializer checks in
9.636 process seconds. Its default
projection and channel programs emit HLO and have fixed signatures; the
Legendre projection graph does not grow with dimension. Other basis dispatch
and the paired isolated GPU benchmark remain pending.

SIR runs 00596--00599 localize the initial memory flag: baseline graph/XLA host
peaks were 1,578,401,792/1,861,824,512 bytes; candidate peaks were
1,579,151,360/1,862,483,968. Prepared/traced RSS is similar, and the difference
appears during the first compiled execution. Candidate versus baseline XLA
peak differs by only 659,456 bytes in this process pair. Candidate XLA GPU
peak is 91,648 bytes versus graph 557,312; warm current bytes stay exactly
23,552 across all 20 XLA calls, with only 12,288 bytes RSS movement. This
supports first-execution compilation/runtime overhead rather than a new
fixed-shape device leak; it does not identify each host allocation. Three
process repeats are still required before accepting a memory/performance
tradeoff or reporting a stable ratio.

Consumer tracing of `linear/types.py` found only the independent NumPy Kalman
filter and derivative modules plus lazy public exports; TensorFlow filters use
`types_tf.py`. Its module documentation now makes that reference-only role
explicit. No runtime implementation was reclassified to excuse a violation.

The full UKF numerical initializer now encloses covariance stabilization,
frame formation, projection and channel construction in one program. Run
00614 passed 27 checks and caught one raw eigenvector-sign parity failure.
TensorFlow eager and XLA `SelfAdjointEigV2` return opposite eigenvector signs
on the same separated-eigenvalue fixture. The algorithm specifies no sign
convention. This must not be silently repaired with an invented sign policy.
Review of all repository consumers found no use of this returned frame as a
reference-coordinate map: the projection uses only row squared norms, and
downstream density training consumes the resulting cores. The explicit
reference-coordinate inputs are unchanged. Validation therefore requires
unchanged eigenvalues/covariance, sign-aligned frame equality and frame Gram
equality, each at 1e-10; all TT coefficients/cores retain ordinary elementwise
1e-10 parity. Raw eigenvector/frame bitwise compatibility across execution
backends is not claimed. This documented decomposition-gauge check does not
apply to SRUKF sigma-point factors, where orientation can change the target.

Run 00615 passed all 28 UKF initializer checks, including heterogeneous
Legendre/Lagrange dispatch and the complete default compiled initializer.
The exact policy guard covers 146 reviewed sources; this remains a subset,
not evidence of repository-wide compliance.

Run 00616 failed four new moment-teacher checks before a numerical conclusion:
the comparison helper applied `is_finite` to integer preparation fields and
the controls fixture omitted eight required fields. The two polynomial checks
passed. The harness now compares integer fields exactly and supplies the
existing integration fixture's controls. This retry retains the algorithm,
inputs and FP64 tolerances. Recovery found no active campaign worker; charged
time is 16,245.435 CPU and 4,886.611 GPU seconds before the retry.

Run 00617 passed the six corrected moment-teacher checks (34.090 process
seconds). The next preparation/freezing checks retain FP64 atol=rtol=1e-10.
For the existing FP32-only nonlinear teacher, the predeclared tolerance is
1e-6 for preparation/random draws and 1e-5 for complete same-XLA moment
directions, with exact integer/Boolean branch parity. These allow FP32
libm/contraction rounding, remain stricter than its existing 1e-2/2e-2
finite-difference score checks, and are not a TF32 claim. Random uniform bits
must match exactly. The stream conversion follows installed TSL
`random_distributions_utils.h:33` and `:72`; it does not migrate the TT stream.
The preparation uses the original power-specific quadrature orders and
first-valid-row selection. Scale-shift loops return explicit validity and
convergence status; their public boundaries preserve the original errors.

Run 00618 passed all 18 teacher execution checks in 112.998 process seconds.
Run 00619 passed 15 existing numerical/consumer checks and failed three route
identity checks: the reset's bounded signature dispatcher was rejected because
the factory recognized only direct TensorFlow wrapper types. This is a real
cross-module compatibility gap in the earlier signature repair. The factory
now recognizes only registered repository dispatchers, binds their immutable
signature settings and factory source, and still rejects copied attributes,
substituted callables and changed JIT settings. The dependency closure records
each actual wrapper. This does not grant canonical LEDH admission.

Run 00620 passed 31 identity checks and exposed another source-verification
defect in all three teacher identities: `compile()` inherited the identity
module's future-annotations flag when checking a dependency that did not
declare it. Source recompilation now uses `dont_inherit=True`; a regression
check binds a dependency without that future import. Run 00621 passed all 35
identity checks in 27.879 process seconds. Exact registered-symbol, wrapper,
source-change and non-admission checks remain enforced. The guard now covers
149 reviewed sources with 870 exact schema/validation/reference exceptions.

Run 00622 passed all 19 final teacher execution checks, including unchanged
preparation/freezing graph size at T=2/4. Run 00623 passed all 32 signature,
Kalman covariance-derivative, correlated-Kalman and padded teacher checks.
The teacher consumer numerical checks passed in 00619; its three identity
failures are repaired and pass in 00621. Paired teacher/initializer/TP
measurements and wider uncovered-route repairs remain open.

Recovery at commit 6119c247: run 00624 passed the policy/controller group.
Runs 00625--00628 passed fresh candidate SIR T=2/4 graph/XLA qualification,
with exact baseline parity. The additional matrix then stopped at its GPU2
contention check; no threshold was relaxed and no other worker was stopped.
The campaign was explicitly paused before further edits. Charged time through
00628 is 16,790.173 CPU / 5,093.223 GPU seconds.

Run 00629 passed all 18 Gaussian-Hermite proposal checks in 26.013 process
seconds. Polynomial and TT-axis/inverse-CDF recurrences now use tensor control
flow with padded heterogeneous ranks. Default sampling encloses the full
calculation in XLA with a bounded fixed-signature cache. Same-mode graph/XLA
values and branch outputs match the pinned implementation at 1e-10; existing
independent quadrature checks through degree six retain their stricter gates.
Polynomial graph size is independent of degree. The proposal remains an
extension/diagnostic, and full preparation, random-input generation and paired
GPU measurements are not yet closed by this result.

The next SSL-LSTM repair replaces forecast draw/time loops and synthetic-data
time iteration with tensor recurrences. Prediction is not a NeuTra training
route. The existing CPU Philox stream is retained using the reviewed FP64 word
conversion; geometry is still the only authorized RNG migration. Fixture and
observation parity use the unchanged 1e-10 criterion, and complete predictions
are compared with identical terminal states and innovations. No posterior or
HMC readiness claim follows from these execution checks.

Recovery on September 18 resumed the isolated validation branch at 6119c247.
Runs 00630/00631 passed 58/59 forecast checks after repairing one hardcoded
test-only repository path. Run 00632 caught an XLA dynamic-slice defect in
chunked forecasting; fixed-size gathers repaired it. Run 00633 passed all 62
forecast checks, including chunk remainders and exact parameter embedding.

Fresh predator-prey TP charts were constructed with an independent diagnostic
LP and frozen before value/gradient comparisons. No historical LEDH result was
reused. Runs 00634--00638 exposed host model validation during tracing,
unsupported gradient FakeParam operations in conditionals, and dynamic reshape
arguments generated by the gradient of tf.repeat inside the date loop. Fixed
model setup is lifted out of tracing; first/terminal steps are specialized;
the numerical date and continuation recurrences remain native tensor loops.
An equivalent fixed index gather preserves particle/weight order and removes
the repeat-gradient compilation defect.

Run 00639 passed 10 checks; both multidate gradient programs now compile.
One XLA T=4/lookahead=2 comparison still fails: raw near-zero feature residuals
differ by at most 1.578e-9, above the fixed 1e-10 gate. This is unresolved;
compilation success does not close its parity requirement. The test now reports
all output differences and chart conditioning to localize that failure.

Forecast innovation generation now uses CPU/XLA and the existing explicit
Philox FP64 word/Box-Muller conversion. Root/family/arm seed folding is unchanged.
Calibration chain/chunk numerical iteration is tensor control flow; host loops
only construct metadata signatures. Forecast caches are bounded to 16 shapes.
Run 00640 passed 66 checks including calibration parity, but two new innovation
oracle tests passed the wrong module's config class to the pinned implementation.
That fixture is corrected without changing seeds, inputs or tolerances.
All findings remain open pending remaining source coverage, strict TP parity,
isolated measurements, current-source suites and terminal review.

Recovery after run 00663: the forecast matrix was explicitly paused and its
worker drained before source edits. Runs 00641--00644 passed 68 predictive,
20 Hermite proposal, 9 complexity-target, and the policy/controller checks.
Run 00645 retained the TP raw-residual mismatch (10 passed, 1 failed).
Runs 00646--00663 passed fresh-process SSL forecast comparisons, with exact
before/after outputs in both graph and XLA modes. The second-size repeats
remain incomplete. Charged time is 18,209.294 CPU / 5,238.892 GPU seconds.

The TP comparison now executes objective, gradient, directional finite
difference, validity and HLO assertions before the raw residual comparison;
their results cannot be inferred from printed gradients. The raw-residual
1e-10 gate remains unchanged while the proposed scaled-residual criterion
awaits an explicit owner decision.

Source preparation review re-inspected paper Algorithm 2/(15)--(16), local
text lines 693--725, author models/full_sol.m:64--125 and computeL.m:24--48.
Weighted recentering and quantile stretching preserve the existing computeL
adaptation; deterministic reference points and seeded TT channels remain
repository extensions, not source-faithful author initializers. Tensorizing
those calculations does not change that classification or their thresholds.

Run 00664 passed all 11 SIR/RQMC regression checks; run 00665 passed the
policy/controller group. Source-preparation run 00666 passed 42 checks and
failed one error-message assertion: the test expected lowercase `nonfinite`
but the preserved enum is `NONFINITE_VALUE`. This fixture typo is repaired.
The checks include exact seeded core values, heterogeneous rank activity
decisions, baseline recentering with discarded nonfinite columns, compiled
reference designs, and a native SIR coordinate time recurrence. No numerical
tolerance changed. The guard covers 155 sources with 929 exact exceptions;
eight reviewed preparation functions are guarded in source_route.py, not the
whole module. Its sequential object construction and other unreviewed routes
remain open.

Two direct CPU pytest diagnostics were mistakenly launched outside the
controller. They completed and are conservatively charged 600 process seconds
in supplemental-compute-0002.json, but cannot close campaign test gates.
Their TP result confirmed value, gradient, finite-difference, validity and HLO
assertions pass before the unchanged raw-residual failure. Current-source
structured reruns remain required. Run 00665's source digest predates an
import-only formatter change and is likewise not a final current-source gate.

Run 00667 passed all 44 source-preparation and consumer checks. Run 00668
established that the installed TensorFlow 2.19.1 StatelessRandomGammaV3 op has
no XLA CPU kernel. Matching upstream CPU/GPU implementation source was read
and preserved as tensorflow-v2.19.1-stateless-gamma-cpu.cc and
tensorflow-v2.19.1-stateless-gamma-gpu.cu.cc in the campaign artifact root.
Their SHA-256 values are
ed2a629d653176bfd882897dea3da040265bd38c9bb47a19e960656801db9fdf and
c0316bb283199241d2d0c4643fb35fc1dab968baad2a2df7c25f9199bf7f6b48.
This was read-only implementation inspection, with no package/environment
change. No external paper or research dataset was acquired.

The native sampler preserves the original 256-block-per-output Philox
schedule, reversed normal/uniform buffers, rejection rule and scalar-alpha
semantics. Run 00669 passed six positive-seed gamma checks. Run 00670 passed
25 Hermite checks but caught three negative-seed gamma mismatches. The XLA
StatelessRandomGetKeyCounter operation had changed negative int32 seed bits;
using TensorFlow's explicit seed scrambling repaired that incompatibility.
Run 00671 passed 12 checks spanning positive/negative seeds, 24/256 draws and
six concentrations. No seed stream was migrated, and uniform bits remain
subject to exact equality checks.

Run 00672 passed all 11 transformed-Student proposal tests, including the
existing 8,192-sample covariance check and complete seeded graph/XLA parity
for negative and 64-bit seeds. Default preparation, conditional mean, density
and seeded transformation use bounded XLA specializations. Hermite random
preparation and GPU stream parity still require the next structured reruns;
the raw TP residual mismatch remains unresolved.

Run 00673 passed the GPU gamma/Hermite/Student group in 80.989 process
seconds, including negative and 64-bit seeds. Runs 00674--00679 completed
the remaining SSL forecast repeats with exact before/after outputs. All
three fresh-process repeats at both fixture sizes now pass in graph and XLA
modes. The recovered controller has finished and was explicitly paused
before further edits. Charged time through 00679 is 19,091.296 CPU /
5,386.604 GPU seconds. The raw TP residual gate is unchanged; no approval
for residual rescaling has been received.

Runs 00680/00682 exposed forced nested XLA in the complexity forecast graph
diagnostic; the baseline eager reference passed in 00681. The candidate
terminal filter now inherits the target's explicit compilation setting, and
the forecast cache key includes that setting. Public forecast compilation
still defaults on. A regression checks the complete graph diagnostic has no
nested XLA and preserves the compiled outputs. The unchanged measurement
harness and frozen inputs are retained for the retry.

Run 00683 passed all 69 forecast checks, including the graph boundary
regression; run 00684 passed the complete Hermite CPU group after the
negative/int64 seed repair. Five isolated measurement fixtures now cover
weighted recentering, complete gamma draws, seeded Student proposals, the
UKF moments-to-cores calculation, and full fixed-TT teacher moments and
derivatives. They use the same FP64 1e-10 gate, two extents, 20 synchronized
warm calls, and existing three-repeat/compute limits. The teacher uses an
explicit frozen sinusoidal design and exact degree-one polynomial integrals
so numerical preparation cannot change its input hash between source arms.
UKF comparisons include projected coefficients, cores, eigenvalues and frame
covariance; arbitrary eigenvector signs are excluded from elementwise parity.
The baseline APIs and their host-validation/tracing failures are retained.
These fixtures do not close source-route sequential preparation or TP parity.

The completed SSL comparison triggered the declared performance investigation:
candidate XLA warm medians were 0.750/1.027 ms versus baseline 0.598/0.700 ms
for 2/4 draws (25%/47% slower). Graph nodes fell from 2,387/4,747 to 476;
candidate host peaks were 954 MiB versus 1,007/1,089 MiB, with no continuing
device allocation growth. Inspection found a serial native loop over draws
enclosing the time loop. The next repair batches draws and replications in
one time recurrence using the same gate/transition/emission equations and
parameter embedding. Distinct-draw parity and isolation are checked before
remeasuring; no tolerance, seed, law or data change is proposed.

Runs 00686/00688 passed complexity forecast graph/XLA at size 1 (maximum
errors 2.22e-16/0). Run 00689 retained the baseline graph-mode nested-XLA
failure, 00690 passed its eager reference, 00691 passed candidate graph parity
at size 2, and 00692 passed baseline XLA. The controller was then paused and
drained before the SSL batching edit. Commit 7e4f399a preserves the prior
implementation, tests, incomplete fixtures and explicit TP blocker.

Runs 00693 and 00694 both passed the 71-check forecast group, including distinct
draws and isolation. The second run was redundant and is charged in full; it
adds no independent evidence. Two unused scalar-forecast imports were removed
after both workers exited. The new SSL measurements remain required.

Austria-SIR review found another four-substep Python recurrence in
_batch_native_transition_mean_and_jacobian. Before its mechanical refactor,
the paper's SIR section (local text 2282--2310) and author
models/sir_austria/sir_step.mlx, extracted document lines 7--18, were
reinspected alongside odefun.mlx:19--24. The paper specifies 0.02 observation
intervals and 0.005 integration steps; the author source uses a half-step
fourth stage. The refactor preserves that existing variant and analytical
tangent propagation, with native tensor substeps and an XLA-default bounded
complete-data endpoint. This is a mechanical fixed-HMC execution adaptation;
it does not resolve the module's other TT/optimizer/prefix preparation loops.
Pinned complete-data score parity and independent transition directional
finite differences are required before its measurements.

Run 00695 passed four new Austria recurrence/derivative checks but its existing
consumer rejected the sealed Lane-B state hash. The same consumer passed on
the isolated pinned baseline in 00696, so this is an actual compatibility gap
from the earlier general simulator compilation repair. General simulation
parity at 1e-12 did not establish byte identity of frozen scientific inputs.
The repair preserves the four existing dataset hashes and seed 81120 by
reproducing the baseline once in a fresh, recorded process and storing that
exact immutable dataset as repository input. The zero-argument sealed-data
API will load those tensors; the general simulator remains compiled and
continues to preserve the original stream and numerical tolerance. No new
data, RNG migration, hash update, target change or tolerance change is allowed.

Run 00697 reproduced and exported the baseline sealed dataset after all four
original hashes passed. sealed_sir_dataset_tf.py stores those FP64 literals
and returns them through one fixed XLA signature; the loader retains every
original hash and clipping check. This is a fixed scientific input asset,
not an alternative simulation algorithm. Its Python module hash participates
in the existing imported-source provenance and stale-evidence checks.

Run 00698 passed the four new Austria checks and all sealed input hashes, but
failed the existing scalar-authority complete-data score gate (atol 6e-13).
Recovery drained the remote fetch and confirmed no numerical worker was active.
The next diagnostic separates transition-mean, analytical-tangent and score
rounding while retaining that stricter gate. The original comparison contract,
dataset, seeds, budget and source anchors remain unchanged. This is a repair
trigger, not evidence to relax parity or promote the candidate.

Run 00699 localized the mismatch: transition-mean differences of 5.68e-14
were amplified to 2.60e-12 in the complete-data score; tangent differences
were at most 7.11e-15. A boundary on only the final RK increment (00700) was
insufficient. The repaired state recurrence retains each scalar arithmetic
boundary through the identity nextafter(x,x), with the exact identity
pullback. It remains batch-native TensorFlow/XLA, with unchanged equations,
substeps and tolerances. Run 00701 passed all six checks, including the
original strict consumer. Explicit ordinary-gradient preservation and GPU
checks follow before any promotion. Failed attempts remain in the budget.

Run 00702 passed GPU values, explicit scores and strict scalar parity, but the
new ordinary-gradient diagnostic placed its tape outside the compiled loop,
triggering TensorList crossing the XLA boundary. Moving the whole value and
gradient calculation inside that diagnostic's compiled signature repaired the
harness. Run 00703 passed all six GPU checks, including the identity pullback.
The rounding boundary changes no score definition and does not introduce a
stop-gradient. Existing module lint debt (import ordering, obsolete noqa and
unsorted exports) was observed separately; the new test/data files pass lint.

Run 00704 passed both sealed-data checks and 00705 passed all 49 controller /
policy checks. The guard remains a 158-source reviewed subset with 929 exact
exceptions. The new complete Austria preparation fixture compares every
returned tensor from three parameter rows and 8/16 samples; its outputs include
proposal points, all densities and analytical complete-data scores.

Runs 00706--00717 completed the three-repeat SSL rerun at both extents, with
exact output parity in graph and XLA modes. The XLA repeat-median warm times
are 0.606/0.606 ms versus baseline 0.598/0.700 ms for 2/4 draws. The previous
25%/47% regressions are removed (now about +1.4%/-13.4%, descriptive only).
Graph nodes are 293 at both extents, versus 2,387/4,747. Host peaks are about
951 MiB versus 1,007/1,089 MiB; device peaks are 16,640/23,040 bytes versus
16,640/26,112. These are allocator measurements, not driver reservation.
Source-route and Austria TT execution debt and the raw TP residual gate remain
open; this successful repair does not establish campaign completion.

The complexity matrix was paused after 00724 to repair shared Austria TT
contractions. Runs 00718/00723 preserve the baseline's invalid nested-XLA graph
diagnostic; 00719/00724 passed its eager reference. Runs 00720--00722 passed
candidate graph and both XLA arms at size 1. Those measurements remain
reusable only if their imported dependencies are unchanged.

Pre-edit TT classification: centered external-parameter density remains the
existing `extension_or_invention`; execution refactoring cannot close its
source-faithfulness gap. Paper section 3.1, equations (13)--(14), local text
532--642, and author @TTFun/eval_reference.m, int_reference.m and
models/tensordot/ttdot.m were inspected for the amplitude/product/integral
contractions. The next helper pads fixed heterogeneous core shapes and moves
the original axis products to native tensor loops, batching components and
component pairs. No ranks, measures, fit, regularization, sample law or
normalization changes are included. Pinned values, cross masses, prefix
marginals, ordinary gradients and enclosing HLO are required before consumer
use. A compiled backward recomputation keeps TensorLists within XLA when
an eager diagnostic tape calls the public compiled primitive.

Runs 00725--00727 exposed three distinct compiler boundaries: heterogeneous
basis branches exporting variant loop state, external tapes trying to export
compiled contraction state, and a zero-iteration Legendre loop with no XLA
adjoint storage. The primitive recomputes local basis pullbacks inside branches,
uses existing analytical basis derivatives where available, and supplies an
explicit compiled full pullback at its public boundary. The internal detached
forward invocation is fully restored by that pullback; no input derivative is
omitted. Constant/linear Legendre polynomials now have explicit tensor formulas,
preserving exact values while removing their empty loop. This shared source
repair invalidates earlier imported-basis measurements; the campaign's source
checks must force those reruns. The next suite includes actual centered
Lagrange bases as well as heterogeneous Legendre degrees, masses and ranks.

Run 00728 passed all 12 centered primitive checks on CPU; 00729 passed the
complete freshly issued centered-child consumer. The parent is an exact
constant-density test fixture issued against current source, not a historical
trained artifact. Pinned values, manual theta scores and ordinary derivatives
passed unchanged 1e-10 gates. No missing historical tensor asset or stale
identity was bypassed. Recovery confirmed charged time of 19,703.127 CPU /
5,966.385 GPU seconds, with the measurement matrix paused and no worker active.

The trainer now batches parent/component mass and prefix contractions and
reduces regularization over its packed position. The next GPU check covers
the complete absolute-density, point/global/prefix score loss and every core
gradient against the pinned implementation, as well as enclosing HLO. This is
execution parity of the existing extension; no optimizer, loss weight, rank,
dataset, tolerance or source-faithfulness status changes.

Run 00730 passed all 14 centered-TT checks on GPU (204.344 process seconds),
including the full composite training loss and every core gradient. Complete
immutable-child public endpoints now use bounded XLA signatures and a complete
compiled input pullback; the next check also differentiates the returned
analytical score and query coordinates. The quadratic solver's numerical
iteration is moved to one XLA while loop. Its existing convergence, curvature
failure, iteration count and trace schedule are compared to the pinned solver,
with independent diagonal solutions and a nonfinite-action veto. Host trace
serialization occurs only after the numerical solve has finished. The six
existing compiled training callback factories now use bounded explicit tensor
signatures. Unreviewed initializer and optimizer scopes remain open.

Run 00731 passed six finite quadratic-solver cases on CPU. Run 00732 passed
the complete public centered-child CPU endpoint, including gradients of scores
and query coordinates across its default XLA boundary. Run 00733 passed two
GPU optimizer updates against the pinned callback and verified a single stable
signature. New isolated fixtures cover complete centered-child values/scores at
residual ranks 2/3 and 4/8 query rows, and complete quadratic solves at 4/8
coordinates. Parent density, residual cores, equations and thresholds are fixed;
core hashes enter the matched fixture identity. These use the existing two-size,
three-repeat, 20-warm-call contract and unchanged total budget. No historical
centered training result is used. Baseline host/tracing failures remain evidence.

Run 00734 passed the complete centered-child and quadratic-solver GPU group
(seven checks, 85.103 seconds). The default complete child calls retain host
finite-output checks because XLA may discard assertions inside compiled graphs.
All new helper, test and measurement files pass lint. The exact policy guard
covers 161 source files / 964 reviewed exceptions; its scope statement still
explicitly excludes unreviewed initializer and source-route execution debt.

Checkpoint e67b8375 preserves the preceding repairs. Run 00735 passed all
49 policy/controller checks. Runs 00736--00742 began complete centered-child
measurements. Size 1 graph/XLA and size 2 graph matched within 3.56e-15;
size 2 XLA candidate had not launched when the matrix was paused and drained.
At size 1, XLA nodes fell 70,311 to 9,909 and host peak fell 2,527 to 1,665 MiB,
but warm time rose 1.72 to 3.26 ms. Graph warm time also rose 36.7 to 140.4 ms.
These single-process observations trigger repair, not a reported timing ratio.
Fixed-shape allocator growth was zero; XLA device peaks were 141,824/159,744 bytes.

Inspection found a redundant heterogeneous-basis branch per axis although the
Lane-B factory explicitly builds a replicated ProductBasisSpec. The next repair
shares that immutable basis and evaluates all coordinates in one batched call;
heterogeneous bases retain their existing native branch evaluation. TT products,
core ranks, feature definitions, queries and gradients are unchanged. The full
primitive/consumer/gradient suite must pass before remeasuring the same fixture.

Run 00743 passed all 15 GPU centered primitive, consumer, gradient and optimizer
checks after shared-basis batching. The subsequent matrix launch was vetoed by
GPU2 contention (523 MiB, observed utilization up to 97%); it created no new
numerical run. The unrelated process was left intact. CPU repair work continues.

The existing additive and adjacent-pair TT initializers now encode their same
finite-state coefficient layout as batched tensors. Fixed boundary ranks are
unpacked only after encoding; there are no Python basis/channel scatter loops.
The parent-plus-additive and disjoint-pair construction remains the existing
extension, with no change to its ranks or fit equations. Exact core equality,
complete coefficient pullbacks and multi-component parity against the pinned
scatter construction are required. Other initializer contractions remain open.

Run 00744 caught a float32 literal in the FP64 pair-encoding Select operation.
An explicit FP64 zero repaired the dtype error; 00745 passed all nine checks,
including the pre-existing strict matrix-free solve test and exact core and
pullback parity at basis widths 3/5. Charged time through 00745 is 19,799.661 CPU
/ 6,842.117 GPU seconds. A request for an owner decision on the TP residual
comparison is pending; its raw 1e-10 gate has not been changed.

Initializer feature operators now compute their original forward/backward TT
messages with two tensor scans, then batch all single-axis and adjacent-pair
integrals. The pair construction retains the all-ones coefficient function on
unselected axes, so it does not assume a partition of unity when contracting
an arbitrary basis. Complete compiled pullbacks cover parent cores and query
coordinates. Rank-2, 36-axis prefix/global comparisons and independent directional
finite differences precede integration measurements. No fit law, regularizer,
ridge, feature ordering, rank, or sampling stream changes are included.

Run 00746 passed both new 36-axis initializer-operator tests on CPU (165.566
seconds), including pinned integrals and core/query finite differences. This
does not establish the historical artifact-dependent integration suite, whose
required tensor assets remain absent. The complete fitted initializer and its
coefficient pullback are being checked separately in 00747. Routine lint repairs
in the touched training module removed one newly unused import and its existing
import/export formatting warnings; no numerical formulas changed in that cleanup.

Call-chain review found that the independent-reference exemption on
estimate_t1_prefix_scores is too broad: run_zhao_cui_austria_sir_parameter_density_t1.py
uses it to build fit targets (for example lines 1733--1745 before this repair).
It is therefore runtime preparation debt and must be tensorized; an independent
comparison use cannot exempt these actual training consumers. Its non-pfor
Jacobian choice was repaired previously, but its point iteration and enclosing
XLA boundary still need repair. No all-path policy compliance claim is made.

Run 00747 passed all four centered-initializer CPU checks (228.898 seconds),
including both complete fitted initializers, every returned field, coefficient
pullbacks and HLO. Run 00748 passed both compiled prefix-training-target checks
(15.750 seconds), preserving the original Philox stream at positive and negative
seeds and testing batch isolation. These checks supersede the broad reference
exemption on the actual prefix training-target consumer. GPU evidence and
complete before/after measurements remain required.

Recovery confirmed 20,209.875 CPU / 6,842.117 GPU process-seconds charged, no
active numerical worker, and continuing unrelated GPU2 contention. The next
bounded CPU check covers the previously untested balanced, seeded residual and
connected-channel initializers. Review caught and removed an unintended equal-
basis-width restriction; first/last random shapes and the original seed schedule
are preserved. Distinct shape branches replace one branch per axis. Invalid
first-axis width-one seeded channels fail before compilation, rather than
silently dropping an out-of-bounds update. The work remains an execution repair
of the existing extension, with no algorithm or RNG migration. The raw TP
residual gate remains unchanged pending the already requested owner decision.

Run 00749 passed all eight seeded-initializer CPU checks (23.217 seconds):
mixed basis widths, ranks 1/3, positive/negative seeds, residual initialization,
connected-channel values and input gradients, HLO, and stable tracing. Recovery
collected run 00750, which passed all four complete centered-initializer CPU
checks (228.229 seconds). Runs 00751--00753 then passed the two prefix-target
checks, nine centered-solver checks, and 49 policy/controller checks. Total
charged time through 00753 is 20,493.437 CPU / 6,842.117 GPU process-seconds.

The checkpoint review inspected the new initializer contractions, heterogeneous
seeded-core shapes, complete coefficient pullbacks, exact policy exceptions,
and actual prefix-target training consumers. Numerical values, feature order,
fit controls and existing Philox draws remain covered by pinned comparisons;
no new RNG migration or scientific admission is claimed. The passing static
guard covers 163 source files and 994 exact exceptions, not all reachable
runtime paths. Remaining optimizer/core-affine and source-route recurrences,
fresh GPU checks, and complete measurements keep F01--F20 and merge open.

Checkpoint 841dc969 preserves the initializer/prefix repair. Run 00754 passed
52 policy/controller checks after adding the test-only GPU3 selector; tests
verify the selected device preflight and prohibit using the option to change
paired measurement hardware. Run 00755 passed five seeded-initializer GPU checks
but failed three exact derivative-identity assertions. Candidate and pinned
gradients matched exactly; both differed from `2*x` by at most 2.78e-17 because
the diagnostic objective used `x**2` (Pow). Replacing this test-only objective
with `tf.square` gives its direct `2*x` derivative without weakening any
tolerance. Numerical implementation, random streams, and the raw TP 1e-10 gate
are unchanged. The failure and retry are charged to the original GPU budget.

Runs 00756--00758 passed the eight seeded-core GPU checks, four complete
initializer GPU checks, and two prefix-target GPU checks (27.627, 310.262 and
21.605 seconds). All used idle GPU3 with verified memory growth and recorded
TF32/device provenance. These are correctness results, not paired GPU2 timings.
Remote fetch found origin/main still at 3582b4ac; checkpoint 841dc969 is seven
commits ahead, with no incoming commits. Integration remains conditional.

The next bounded repair addresses the existing core-affine product-rule block
encoding and coordinate masks. It preserves the current extension's exact
parent/tangent blocks, heterogeneous ranks and widths, parameter order, and
ordinary training-loss gradients. Classification remains extension_or_invention;
the author source does not establish this parameterized tangent construction.
The underlying amplitude/mass products were rechecked against paper section 3.1,
equations (13)--(14), local text lines 539--651, and author
`third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTFun/int_reference.m:22--29`
and `source/models/tensordot/ttdot.m:20--23`. Tests must compare original blocks,
round-trip inversion, parent-block rejection, complete loss/gradient consumers,
stable HLO signatures and directional finite differences. No rank, loss,
regularization, RNG or source-faithfulness change is proposed.

Run 00759 passed all 15 centered primitive, public value/score, input-gradient,
training-loss and optimizer-consumer GPU checks (191.239 seconds). Total
charge through 00759 is 20,501.218 CPU / 7,421.328 GPU process-seconds. GPU3
correctness evidence is fresh for this checkpoint; GPU2 measurements remain
pending. New numerical helper/tests pass Ruff and whitespace checks. The wider
lint invocation found existing import/export and artifact-exception style debt
in the two Lane-B modules; no blanket repository lint success is claimed.

Run 00760 passed the refreshed 52 controller/policy checks. The core-affine
repair now uses one tensor-index encoding/decoding program across every core
and parameter component; Python only describes and packs the fixed tensor
schema. Run 00761 passed all six heterogeneous block, exact inverse/mask,
corruption-veto and bounded-graph checks, plus the complete loss/gradient and
directional comparisons before a final inspection-helper error. The test called
the bounded dispatcher's graph accessor without its required position argument.
The retry supplies that argument; no numerical source or tolerance changed.

Run 00762 passed seven core-affine CPU checks. After adding a graph-reference
purity check and preserving the enclosing compilation boundary, 00763 and
00764 passed all eight core-affine GPU and CPU checks. Run 00765 passed the
expanded controller/policy checks, including same-device comparison resumption
and repeat aggregation. The guard covers 164 reviewed sources and 1,016 exact
exceptions; this is not repository-wide closure.

Runs 00766–00773 qualified the complete core-affine point/global/prefix loss
and training gradient at four and eight query rows on GPU3. All four matched
graph/XLA pairs passed at the unchanged tolerance; maximum absolute discrepancy
was 4.45e-16. Both extents traced once, had no numerical callbacks, and used
10,355 candidate graph nodes versus 44,990 baseline nodes. The explicit graph
arms had no nested XLA. Verified memory growth remained enabled.

These are qualification observations, not three-repeat timing claims. At four
rows the XLA trace/first-call times were 3.083/7.346 seconds versus
12.506/37.534 seconds; host high-water memory was 1,433 versus 2,113 MiB.
The candidate warm median was 5.775 ms versus 2.042 ms, and at eight rows it
was 5.355 ms versus 1.992 ms. Graph warm latency also increased. This triggers
the predeclared performance investigation; reduced compilation cost does not
close it. Device allocator peaks were 195.25/227 KiB versus 186.5/206.25 KiB,
and live allocator bytes stayed constant across all 20 warm calls.

Recovery charge through 00773: 20,577.311 CPU and 7,795.484 GPU process-seconds.
Next checks cover the recent fitted-initializer boundary repair, then complete
initializer measurements. Review of the native TT contractions identifies
sequential small tensor loops as a plausible warm-latency mechanism; repeat
evidence and an explicit investigation are still required before acceptance.
Generic stochastic TT density training remains numerical execution debt:
core evaluation/mass products, penalties, seeded preparation, and optimizer
steps must preserve the existing extension's loss, gradients, RNG stream and
metadata while moving into stable compiled boundaries. The paper's section
3.1, equations (13)–(14), and author int_reference.m:22–29 / ttdot.m:20–23
were re-inspected for the underlying amplitude/mass algebra. No source-route,
fit-method, L1 selection, or scientific admission claim changes.

Run 00774 passed all four initializer CPU reference checks, including full
coefficient pullbacks and the new graph-reference purity assertion. An earlier
GPU3 launch was vetoed before worker allocation because an unrelated process
held 417 MiB; it produced no measurement. GPU follow-up and initializer timing
remain required. Focused lint passes for the new helper, fixtures and tests;
the campaign driver retains previously existing import/dict style findings.

Checkpoint f048c5c9 preserves the core-affine repair. Runs 00775 and 00776
passed 48 stochastic TT checks each on CPU and GPU3, including all existing
P75/P76 checks and fresh baseline/finite-difference checks. Core contractions,
penalties, seeded core initialization, complete objectives and two optimizer
steps preserve the existing extension and seeded stream. Compiled updates
carry a finite/domain status through XLA and reject invalid updates before
assignment. Both objectives preserve all input/core pullbacks at their public
compiled boundaries. Source guard now covers 166 sources / 1,054 exact
exceptions; only schema handling and validation were exempted.

Run 00777 passed four Austria training/calibration GPU checks. The batch-native
training factory now has bounded explicit signatures; penalties are tensor
reductions, and the original one-core normalizer rescale is a cached XLA
calculation. Fresh two-step baseline parity, HLO, stable tracing, target mass,
and invalid-calibration no-mutation checks passed. Artifact source closure now
also includes these native execution dependencies. Numerical measurements for
these changes remain pending, so no finding is closed.

Run 00778 reproduced the unchanged stochastic trainer's graph failure: its
regularization uses a Python boolean test of a symbolic tensor. The controller
initially stopped because TensorFlow's error puts backticks around `tf.Tensor`.
The classifier now recognizes that exact trace-time error only; a regression
check rejects the same text under another exception type or execution phase.
The frozen baseline is unchanged. Its valid eager path supplies parity, and
its unavailable graph/XLA time remains explicitly unavailable.

Recovery collected 00779--00808: complete stochastic density loss/gradient,
density Adam update and square-root prefit update qualified at both extents
against the valid eager baseline. Baseline graph/XLA host-operation failures
remain recorded. Candidate maximum discrepancies were below 1.8e-15, with exact
prefit-update parity. These are qualification observations, not repeat results.
Charge through 00808 is 20,872.611 CPU / 8,140.058 GPU process-seconds.

Review found the public train_step and prefit_step still used their old update
body when called inside tf.function. Its assertions can be discarded by XLA,
so this path could mutate parameters on an invalid calculation even though
the standalone compiled wrapper rejected it. Both paths now call the same
guarded tensor update. Rejected updates preserve parameters and optimizer slots
and emit a nonfinite gradient norm when XLA removes the assertion. The Austria
training callback uses the same assignment guard, preserving its existing loss.
This is a validity repair; no clipping threshold or accepted calculation changes.
Enclosing-XLA rejection, unchanged valid two-step updates, and fresh artifact
reload/source-closure checks precede new timing repeats. Algebra anchors remain
paper section 3.1 equations (13)--(14) and the cited author integration/product
recurrences; the stochastic training construction remains extension_or_invention.

Runs 00809 and 00810 passed 50 stochastic and six Austria consumer GPU checks
(70.800 / 54.780 process-seconds), including invalid enclosing-XLA no-mutation,
valid two-step optimizer parity, fresh artifact reload and tensor tamper rejection.
Run 00811 passed the policy/controller checks. The exact guard covers 166 sources
and 1,048 exceptions with no stale exceptions; removing duplicate update bodies
removed six host-schema exceptions. New helpers, fixtures and tests pass Ruff.
The earlier qualification measurements precede this assignment-guard repair;
fresh current-source qualification and repeats remain required. No F01--F20
closure, merge or push is claimed by this checkpoint.
