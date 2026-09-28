# Follow through the nonlinear score-study direction consumers

Read-only inspection during Gaussian direction qualification found the same
execution-policy violation in `score_study/nonlinear_adapter.py`: line72 loops
over six EKF/UKF directions in Python; lines95–96 do so for nonlinear LEDH,
SGQF and mixture-covariance proposals. These remain open consumer work even
after Gaussian qualification. The actual consumers call the factories in
`score_study/nonlinear_tf.py`; compiling each individual direction does not
compile the complete returned score.

After the bounded Gaussian correctness/cost unit, reuse its shared
`make_direction_kernel` authority in these configuration-specific factories.
Do not copy filter equations, introduce autodiff scores, or change the nonlinear
model, covariance providers, reference-grid authority, diagnostics, admission
rules or finite-scalar interpretation. Inspect the actual factory output shapes
and total initialization derivatives first. Expose dynamic theta, observations,
directions and base noises through one fixed enclosing XLA signature; preserve
explicit non-JIT debug behavior. Keep every existing auxiliary field and the
original optional diagnostic-failure payload ordering. Native flags plus host
enforcement must retain value-invariance and invalidity rejection.

Before numerical execution, freeze a fresh nonlinear fixture with complete
arrays, c/b nonlinear controls, Contract E/dual-cap controls and exact K=N in
the plan, independently of historical LEDH test fixtures. Compare the same
current directional authority evaluated by test-only Python loops. Start at
d1,o1,N8,T2,float64; require full nested parity at1e-9 and five-point derivatives
at two predeclared steps with absolute2e-6 gate on the same finite scalar.
Test EKF, UKF, LEDH, SGQF and mixture covariance, optional diagnostics, changed
inputs, one trace, enclosing HLO and actual evaluate_nonlinear wiring on CPU
reference and trusted non-display GPU. All unexpected numerical/status failures
stop that case for localization; do not tune the fixture or relax the gate.

Initial allocation:16 workers/3600 CPU and3600 GPU process-seconds inside the
remaining global caps, one numerical worker at a time,300-second initial
timeouts. Activate/charge this allocation only after the current Gaussian
allocation closes. Use the stable registered campaign runner and unique raw
artifact directories. A source-bound terminal readback and matched fresh-owner
cost screen are required before accepting this consumer repair; reserve the
cost cohort explicitly after correctness. No whole-program completion follows
from these five consumers. No HMC/training, canonical admission, package/system/
cache mutation, live MacroFinance edits, subagents or main merge.

Skeptical review: shared assembly code reduces implementation duplication but
does not transfer Gaussian numerical evidence to nonlinear models. Compiled
assertions may be ignored, and optional diagnostics may lose static dimensions;
neither justifies dropping checks or fields. Finite-difference comparison must
differentiate the existing approximate finite program, not an unrelated exact
nonlinear likelihood. The independent grid reference remains a reporting
authority, not a substitute gradient. Self-review passes for source preparation;
freeze exact operands and inspect the complete call chain before execution.

Execution preparation at d9a534584: the fresh exact fixture is
`tests/fixtures/filter_repair_nonlinear_directions_20260929.json`, generated once
by a test-only standard-library normal stream with seed9292027. It freezes
d=o=1,N8,T2,float64; theta [.57,-.77,-.53,.83,.18,-.23]; observations
[[.24],[-.12]]; c=.17,b=.09; changed theta/data; every initial/process/reset
array; SGQF level2; within_fraction .6; and all positive Contract E/dual-cap
controls. Parameter directions are identity6 and .7 times its one-row roll.
Five-point steps are2e-4 and1e-4, with unchanged absolute2e-6 gate. Test all
eight combinations: EKF, UKF, and each LEDH/SGQF/mixture provider with and
without optional diagnostics. Reserve20 workers (16 actual CPU/GPU cases,
one terminal readback and three localized retries), keeping3600 CPU/3600 GPU
process-second limits inside the unchanged global caps.

The exact nonlinear route was inspected: evaluate_nonlinear calls
nonlinear_tf.make_moment_filter or make_ledh_kernel, then the shared analytical
quadrature/LEDH executor. Moment filters return value, directional score and
minimum covariance; the LEDH family returns optional compact diagnostics.
The moment filter's six tangent contractions are another numerical Python
comprehension and will be replaced with explicit tensor operations. The
reported minimum covariance reduction is post-run diagnostic reporting;
actual covariance usability remains enforced inside each compiled filter.

Actual endpoint tests replace only noise/data generation with the frozen
arrays and use the already configured worker runtime. They retain the real
CPU/XLA reference-grid computations:257 points/radius8, the existing coarse
129-point check and expanded385-point/radius12 check, tolerance1e-7, tail1e-9.
This reference is not an exact nonlinear likelihood or a substitute derivative
target. Keep all existing refinement gates; do not replace a failed reference
with a stub. Verify actual host rejection with a nonfinite observation for
moment filters and a nonfinite reset design for particle consumers. The test
may inject the former only into the proposal owner after healthy reference
checking, so it isolates rejection in the filter rather than an earlier grid
veto. Preserve diagnostic failure payloads. No Gaussian numerical source is
changed; the existing shared direction assembly is reused unchanged.

Skeptical preparation review: d1 cannot establish multidimensional filtering
quality, but it matches this scalar nonlinear consumer's actual contract. The
reference checks can veto endpoint execution, while finite differences compare
each approximate program to its own value. Both distinctions are explicit.
Unknown shapes must fail closed and be bound from the fixed configuration;
no dropped payload or scalar fallback is allowed. Self-review passes for the
declared20-worker unit; no new scientific/admission claim or independent review.

04783 EKF and04784 UKF pass CPU qualification including actual grid refinement;
04785 nonlinear LEDH returns invalid at the first unchanged fixture evaluation.
Pause this case and run one bounded localization worker on the identical
arrays/controls. Compare original directional and enclosing outputs/status,
request the existing compact diagnostics, and preserve all nonfinite/rejection
sentinels. Diagnostic completion cannot qualify a healthy finite score. Do not
tune controls, alter the fixture, relax tolerance or proceed to cost claims for
this rejected case. The worker is charged to the declared20-worker allocation.

04786 localizes the rejection to the unchanged original program: all six
directional values are -inf with zero score sentinels, and the enclosing
diagnostic path matches the original complete outputs including invalidity.
Both higher-moment corrections are valid; covariance minima remain positive
(prediction .3965/.3276, update .1994/.1684). At the second reset the maximum
column-weight residual is2.188536452e-4. Because transport rows sum to one and
weights sum to one, its column residual sums to zero up to roundoff, so total
variation is at least that maximum. The actual reset gate at
ledh_canonical_score_tf.py:569–580 requires column TV<=1e-4. This is insufficient
balancing under the frozen four/four iteration controls, not established
ill-conditioning or an enclosing-XLA discrepancy. No controls or gates change.

Record the failed T2 fixture as an intended-refusal regression after verifying
the exact returned TV from existing reset transport/weights. A diagnostic can
temporarily observe the existing shared executor result at trace time, adding
the TV/program-valid fields to the test output only; it must delegate the actual
computation unchanged and cannot replace numerical validity. Verify original
and enclosing rejection and the real endpoint exception on CPU/GPU. This
qualifies refusal only; the healthy T2 derivative/cost criterion stays open.
Classify each remaining provider on the same fixture before attempting its
healthy-score check. A rejected provider must be recorded, not repeatedly
launched as an expected healthy case. Continue the independent EKF/UKF GPU
qualification and unaffected healthy providers. Fresh scope-specific offline
calibration with disjoint calibration/untouched-check arrays is required before
a new healthy nonlinear LEDH T2 scope is nominated; it will have its own reviewed
fixture/controls and cannot erase or tune on this preserved failed fixture.

Reserve24 workers (same3600/3600 process-second caps) to finish classification
and refusal checks: three exact-TV localization workers, six provider/diagnostic
cases on each device, the independent EKF/UKF checks and terminal readback.
For a healthy provider the original derivative, changed-input and reference-grid
criteria remain mandatory. For a rejected provider require exact original
sentinel/full-output parity, real host rejection and enclosing HLO; explicitly
label it refusal-only and keep its healthy T2 derivative/cost gate open. This
is not a waiver that converts a refused fixture into valid numerical evidence.

04795 preserves a refusal-test harness error: the newly added branch applied
bitwise equality to every floating diagnostic, although the original contract
uses1e-9 for those fields. SGQF's coordinate-cap displacement differs by
5.55e-17; rejection/value/score sentinels and validity agree. Apply the original
1e-9 finite-field rule consistently to all refusal variants, require exact
nonfinite locations and discrete fields, and keep -inf/zero score sentinels
exact. Record maximum finite-field error. This corrects an unintended stronger
harness assertion; it does not change any original numerical tolerance or
permit a rejected fixture to count as healthy. Preserve04795 and retry it.
