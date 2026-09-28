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
