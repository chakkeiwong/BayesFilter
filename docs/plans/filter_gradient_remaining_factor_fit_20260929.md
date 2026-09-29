# Remaining unselected factor fits and isotropic reporting

Completed diagnostic allocation04923--04930 after the stable-angle repair
through04922; the committed baseline is bebb2591a. See the matching result note.
Optimizer first-divergence and isotropic reporting obligations remain open.
The canonical NeuTra architecture and owner-deferred iAPF/KDM are unchanged.

Question: which actual optimizer/matrix differences explain the4534 remaining
full fitted-record leaves, and which reported geometry is well-defined and
usable? The selected precision/covariance/center/audit agree at existing bounds;
unselected fits3/4/5 account for4477 leaves, with57 selection/stability leaves.
Eight factor2 angle leaves reflect different matrices, now independently
verified on each backend. Do not repeat the repaired acos/SVD trial.

Start with saved04919/04920 payloads, raw records and identical04591 operands.
Bind each differing field to its consumer and eligibility decision. CPU fit3
is a holdout-rejected factor1 fit ending at200 iterations; fit4 is a
holdout-rejected factor2 fit with optimizer failure at iteration7; fit5 passes
its fit gate but is unselected and unconverged after200 iterations. Preserve
these distinctions. Neither holdout rejection nor nonconvergence by itself
proves severe matrix ill-conditioning.

Trace `factor_correlation_geometry.py` and its native callers from initial
parameters through the shared objective/analytical gradient, L-BFGS termination,
final covariance/precision, status and selection. Reuse saved input and
counter evidence first. If new instrumentation is needed, require exact
ordinary-program output/decision reproduction before using its intermediate
trace. Locate the first same-operand discrepancy, then compare objective,
gradient, solve residuals and stationarity to independent references. Do not
run unchanged full optimizers repeatedly hoping for matching terminal records.

Separately audit isotropic or rank-cut repeated eigenspaces. A requested
subspace cutting an eigenvalue multiplicity is non-unique; report that fact
explicitly rather than mistaking an arbitrary eigenvector basis for a unique
geometric angle. Any runtime refusal/reporting change needs known-good no-fire
checks and preserved public error precedence. No new numerical ridge, rank
convention, convergence threshold, iteration cap or usability rule is authorized
merely by this diagnostic plan.

Pass criterion: explain and repair an actual reachable numerical/status defect
with unchanged accepted-result bounds, derivative authority and selected
geometry; invalid systems must refuse explicitly. Complete exported records
and discrete decisions must be checked. Accurate descriptions of different
unconverged fits are not automatically implementation defects, and their
diagnostic status cannot be used to admit an unqualified runtime.

Initial allocation: at most8 serialized workers/1800 CPU and1200 GPU
seconds under the remaining global budget; start with saved-evidence/source
inspection and the smallest discriminating diagnostic. CPU reference and
trusted GPU/XLA with verified growth remain distinct. Version every artifact
under the existing campaign root. Source drift, invalid reference, missing
ordinary-program witness or exhausted budget stops the affected experiment.
Failed candidates can trigger a bounded repair; they do not relax acceptance.

Skeptical review: the main risks are assuming all4500-plus leaves represent
independent defects, demanding exact optimizer trajectories on different
floating backends, or waiving a selected/status defect because some fits are
unselected. Grouping by causal source and tracing actual consumers addresses
these risks. Current evidence does not establish optimizer convergence,
isotropic reporting correctness, whole-program closure or terminal costs.

Execution review, 2026-09-29: the factor optimizer's objective gradient uses
the existing TensorFlow Probability value-and-gradient authority. This is a
geometry-fitting derivative, not a filter analytical-score implementation;
its role must remain explicit. Start with a source-bound saved-record reader,
then recover missing optimizer states using the existing factory. A standalone
or instrumented call must reproduce the corresponding ordinary saved fit before
its internal state can explain that fit. Initial-state and same-operand
value/gradient checks precede any new optimizer intervention. Compare to an
independent high-precision objective/derivative calculation; no optimizer or
ridge change is proposed. Covariance condition, prediction-Jacobian condition,
gradient norm and solver termination have different meanings and are reported
separately. Their values are explanatory, not new acceptance thresholds.

Existing iteration/tolerance settings are a frozen comparator, not demonstrated
convergence. Saved selected geometry and all existing status/bound checks are
the preservation criterion. Source/hash drift, unusable reference, or missing
ordinary-output reproduction vetoes causal interpretation of a probe. A
repeated eigenspace is checked with known diagonal/rotated matrices and reported
as non-unique when the selected rank cuts the multiplicity; any reporting guard
needs separated-spectrum no-fire checks. This review finds no need for another
full trajectory replay before the bounded probes. Raw artifacts use fresh
run directories under the existing campaign root; the result note is
filter_gradient_remaining_factor_fit_result_20260929.md. No performance or
posterior-quality ranking follows from these diagnostic runs.
