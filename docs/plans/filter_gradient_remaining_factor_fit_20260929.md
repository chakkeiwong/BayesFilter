# Remaining unselected factor fits and isotropic reporting

Prepared continuation after the stable-angle repair through04921. Activate a
new bounded allocation before numerical work; no worker is launched by this
document. Freeze the committed shared-angle repair as the source baseline.
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

Suggested initial allocation: at most8 serialized workers/1800 CPU and1200 GPU
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
