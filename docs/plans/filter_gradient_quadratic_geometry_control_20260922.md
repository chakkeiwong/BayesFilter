# Quadratic geometry control continuation

Continue F18/F19 from pushed `813eed67` within the existing 32 CPU / 52 GPU
process-hour campaign. The posterior rejected-design comparison remains
unwaived, and GPU3 contention currently prevents its remaining qualification.
This work is independent: enclose the existing low-rank quadratic initializer's
center proposal, exact evaluation, acceptance tests and incumbent replay in
stable TensorFlow/XLA functions. The full geometry and iterative wrapper remain
open until their enclosing control is also repaired and qualified.

Preserve the constrained trust-ball solver and unconstrained SPD solve, all
thresholds, callback ordering/counts, finite-status decisions, exact incumbent
identity, and complete returned diagnostics. No optimizer, random stream,
distribution, numerical rank rule or score changes. CPU work is an explicit
reference lane with GPUs hidden; the default candidate is GPU/XLA. Leave the
public helpers unchanged until candidate full-record, GPU and cost checks pass.

Compare the isolated original `3582b4ac` source closure and the frozen TensorFlow
checkpoint `813eed67`. The original remains the numerical authority; the newer
checkpoint localizes execution changes and cannot replace the original. Require
`atol=rtol=1e-10` for all numerical fields and exact schemas, decisions, callback
counts, reason order and shapes. Cover interior/boundary, zero-step, invalid
precision, target nonfinite and unchanged acceptance boundaries; preserve every
failed record. Test six runtime inputs, changed target resources, single tracing,
enclosing XLA HLO and callback identity/release. No Python callback, pfor or NumPy
may enter the numerical controller. Host formatting occurs after calculation.

Skeptical review: a traced callback is not an execution count; use colocated
int64 TensorFlow resources. Constrained solver rejection must skip the target.
XLA assertions do not provide finite-status gates. Host exceptions and target
programming errors need explicit compatibility testing before public wiring.
Scalar-target evaluation here is one proposed point, not a batch training route.
Repeated partial helper timings cannot qualify the whole initializer. Fresh
original/graph/XLA measurements at D3/D5 include full reporting and cold costs;
retain the campaign's warm, compile and allocation investigation thresholds.
Full public consumers and final repeated whole-endpoint costs remain mandatory.

Use the bounded campaign driver and unique numbered artifacts, one numerical
worker, unchanged GPU3 idle gate and verified growth. Stop the affected launch
on contention, budget exhaustion, invalid provenance or numerical failure, then
continue an independent repair where justified. Runtime/tests/driver stay frozen
during workers. No external source/pin or environment changes and no main merge.
This is primary-agent review; no independent review is claimed.

02481 preserves an original-versus-checkpoint failure before candidate promotion:
the frozen TensorFlow checkpoint and both enclosing modes agree with each other,
but differ from the original interior D3 proposal by about 1e-9 in position and
2.34e-9 in refined score norm. Every arm calls the target once and accepts. This
is inherited numerical debt, not permission to replace the original authority.
Compare the raw XLA eigensystem with the already qualified refined eigensystem
on that same matrix; retain eigen residuals, orthogonality and independently
solved positions before changing the solver dependency. The original remains
the unchanged 1e-10 gate. A frozen checkpoint known to fail it is explanatory
only for this defect, and must remain in every full-record artifact.

02482 attributes the inherited error to raw XLA eigenvectors: eigen residual
1.62424e-8 and independent solve error 1.28795e-9 despite good orthogonality.
The existing refined eigensystem reduces these to 3.25463e-17 and 2.77556e-17.
The internal controller now supplies that eigensystem to the unchanged trust
bracketing/bisection body; its explicit non-XLA graph reference uses TensorFlow
`eigh`. A private optional eigensystem argument keeps the existing public
helper's arithmetic unchanged during qualification. No threshold, root solve or
acceptance criterion changes. Public repair of the inherited raw-eigen error
also remains required; the checkpoint's incorrect record is preserved for
attribution and cannot be a numerical authority.

02483 passes the repaired D3 smoke.02484 passes 25 checks and preserves 12
failures: the planned determinant error-status operation is unsupported by
CPU XLA; graph-mode D1 returns malformed empty outputs when `eigh` is nested
under the input-validity branch; and HLO comparison sees only dummy-source
`zeros/_N` metadata identifiers.02485 confirms native `Lu` is also unsupported,
while the standalone D1 trust body and enclosing XLA controller produce correct
typed results. All raw evidence remains archived.

The retry uses a native partial-pivot elimination solely for the original exact
zero-pivot solve-error status, without a numerical-rank tolerance. The actual
unconstrained solution still uses the original `tf.linalg.solve`. The constrained
eigensystem is evaluated before the validity conditional on the same matrix
(or an identity for invalid inputs); target evaluation remains conditional.
This avoids nesting the graph eigensystem under the validity branch, with no
change to usable-input arithmetic. The HLO comparator normalizes only the
observed dummy-source constant name suffix, retaining every operation, operand,
shape and constant and preserving both raw HLOs. Full original records and
changed-input checks remain mandatory;02486 is the bounded CPU retry.

02486 passes 29 checks: the native exact-zero pivot status fixes the unsupported
unconstrained path. Moving only the eigensystem outside the condition does not
fix D1 graph outputs. Removing the redundant validity conditional around the
whole trust calculation, using identity/zero safe inputs only for invalid
cases, fixes the scalar output shape without executing its target.02487 passes
that D1 case and six runtime-operand/HLO checks after moving configuration
constants inside tracing.02488 passes all 48 CPU cases, including D1/D3/D5 records,
strict nextafter acceptance boundaries, singular/duplicate/permuted/tiny direct
solves, zero calls on rejected steps, exact replay, resource changes and cache
release. This isolates a nested-control interaction; it is not a general
TensorFlow backend diagnosis. Public numerical helper behavior is still unchanged.

02489--02494 are the planned six fresh-process CPU cost arms at D3/D5, each with
21 calls plus a changed-input call and complete original record comparisons.
The archived analyzer is the earlier controller-cost analyzer with one added
filename/role selection only; original/source/input provenance, device and
sample/repeat checks remain unchanged. GPU qualification remains blocked by
GPU3 occupancy under the unchanged idle threshold.

## CPU costs and checkpoint review

All six CPU processes02489--02494 pass complete original initial and changed
records. These include construction, tracing, the complete calculation and
report materialization;20 warm calls follow the first call in each process.
They are single-process descriptive dependency measurements.

| Dimension | Original warm ms | Graph warm ms | XLA warm ms | Original / XLA cold s | Extra observed XLA RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.432 | 0.964 | 0.752 | 0.00262 / 0.802 | 214.3 |
| 5 | 1.386 | 0.949 | 0.802 | 0.00249 / 0.847 | 217.2 |

The cold trigger fires; no warm-time or additional-host-RSS trigger fires.
XLA construction/trace/first execution take 0.089/0.138/0.575 seconds at D3 and
0.088/0.133/0.626 seconds at D5. RSS rises mainly at first execution: approximately
580.5 to 788.6 MiB and 580.9 to 791.4 MiB. Graph size is 689 nodes at both dimensions.
The20 warm calls add 4/16 KiB; this short observation cannot establish a plateau
or absence of a leak. Arbitrary target turnover/native executable eviction,
larger dimensions, GPU costs and whole-initializer costs remain open.
Exact analysis: `geometry-control-cpu-costs-02494.json`, with its archived
`analyze_geometry_control_costs_20260922.py`.

02495 passes all 22 existing geometry/reference checks with the public wrapper
unchanged.02496 passes 72 policy/controller checks. New numerical control has no
Python-loop exemption; two new exact exceptions format completed records only.
Focused Ruff and whitespace pass. The guard is still partial at 208 sources and
1,300 exceptions, not a repository-wide compliance result.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain internal candidate and original error evidence | 48 CPU record/boundary/cache checks and 6 cost processes pass;22 existing checks pass | GPU and public callback-error compatibility remain unqualified | Resource/runtime behavior on the default GPU and complete public consumer chain | Qualify GPU, failure compatibility and public integration; then enclose remaining geometry/iterative control | No complete filtering repair, public admission, posterior/HMC claim or merge |
| Record the inherited trust eigenvector defect | Same-input raw eigen residual explains position/score errors; refined solve matches independent reference | Original strict criterion preserved | Broader spectral and downstream public behavior | Repair public dependency only with its full qualification | No tolerance waiver or changed trust/acceptance algorithm |

Primary-agent review: exact-zero pivot detection must not acquire a hidden rank
tolerance; malformed callback behavior must not be inferred from healthy cases.
The public wrappers deliberately retain their previous behavior while these
boundaries are qualified. The strongest alternative explanation for speed gains
is omitted host work; the measured scope includes final reporting but remains a
small dependency, not the whole initializer. The weakest evidence is default-GPU
coverage, currently prevented by GPU3 occupancy. No source pin, package,
environment or external consumer is changed.

Audit 02497 passes: 2,973 working Python files, 2,972 parsed and one unchanged
external-reference parse error. Charges through 02497 are 51,226.57090895319
CPU and 48,325.1067211973 GPU seconds. No numerical worker is active; the
32 CPU / 52 GPU process-hour caps are unchanged.

## Callback boundary continuation from 0ecf4775

The checkpoint is committed and pushed. Remote main has advanced to c7adbda7;
it remains separate until qualification. GPU3 is still busy. Continue the
unchanged CPU reference lane and preserve original 3582b4ac and checkpoint
0ecf4775 records before repairing callback compatibility. At most three
120-second attempts per unchanged job remain enforced by the campaign driver.

Question: can the native center proposal and replay preserve rejection records
for callback construction/conversion errors without a Python numerical callback
or an eager retry? Test exceptions, wrong return arity, nonscalar values,
incompatible dtypes, flattened valid scores, and malformed score sizes. Use full
original records and resource counters where they can measure execution.
Original malformed score broadcasting is outside the documented same-dimension
score contract; any new fail-closed shape guard must be recorded explicitly,
not counted as original numerical parity. No callback may be evaluated eagerly
to qualify it. Construction errors may become NaN/status outputs in the native
adapter, preserving the original invalid-evaluation boundary.

Skeptical review also identifies dynamic Assert/CheckNumerics operations: XLA
may discard their runtime vetoes. Such callbacks must be rejected at native
program construction until they express validity through numerical outputs;
merely catching Python tracing errors does not solve that problem. Inspect the
whole traced callback graph, including nested functions, and reject host Python
callback operations as well. This is configuration validation, not a numerical
loop exemption. Preserve a direct assertion reproducer as explanatory evidence.
Do not generalize this limited admission check to arbitrary callback correctness
or claim full public exception compatibility. No public wiring, tolerance,
optimizer, RNG, HMC authority or external consumer changes occur in this step.

02498 preserves ten callback-compatibility failures and two passing flattened
score cases. The repair catches construction/conversion errors inside the
native adapter and emits the original invalid-evaluation values; it does not
catch arbitrary runtime/compiler errors or retry the callback eagerly. A
configuration-only scan rejects Python callback operations and (for XLA)
Assert/CheckNumerics throughout the traced function closure. Wrong score sizes
are explicitly rejected under the documented same-dimension score contract;
the original's scalar broadcasting is recorded outside parity. Resource counters
verify that adapter construction and skipped numerical branches perform no
target evaluation. Renew full healthy records and graph ownership after this
boundary change, then renew costs on the resulting implementation.

02499 passes all 21 callback/shape/admission/resource checks. 02500 renews all
48 original-record, nextafter, runtime-input and graph-release checks after the
adapter change. Renew the six CPU cost arms and add 3,000 alternating
initial/changed-input calls per arm at D3/D5, observing memory every 1,000 calls.
Keep the 21 timed calls, original authority and existing investigation triggers.
This answers the previously unmeasured warm-reuse question; it does not test
arbitrary callback turnover or prove a general allocation bound. Run one
numerical worker at a time, at most 120 seconds each, with sources frozen.

02501--02506 pass all six renewed costs and 3,000 additional alternating calls
per arm. D3/D5 original versus XLA warm medians are 1.398/0.772 ms and
1.352/0.801 ms; XLA cold is 0.812/0.884 seconds and extra observed RSS is
213.8/216.8 MiB. XLA has 692 nodes at both extents. Its final 1,000-call intervals
add 0/4 KiB; bounded reuse shows no large continuing growth here. The cold
trigger remains explicit. Analysis: `geometry-control-cpu-costs-02506.json`.

02507 catches an invalid allow-list role label (`configuration`). Correcting
those two exact metadata-scan exceptions to the existing `host_validation` role
repairs the policy schema; 02508 passes all 72 policy checks. No numerical loop
was exempted. Public helper integration and GPU qualification remain pending.
