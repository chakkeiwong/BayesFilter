# R-to-TensorFlow diagnostic comparison

This is the next independent diagnostic under the owner's continuous-execution
directive. It must not bypass the unresolved exact-paper-replication gap.
The current TensorFlow `execute_iapf` consumer accepts only d=o=1 and uses
bounded density/relative-shape fitting; its model fixture differs from the
paper's dense transition matrix. The existing log-quadratic TF comparator
fits a full precision matrix, whereas the validated R alternative is diagonal.
Consequently an end-to-end same-method comparison is not yet available.

Question: do shared, mathematically identical Gaussian twisting and fitting
operations agree across the actual R and TensorFlow implementations? The
primary criterion is FP64 agreement at 1e-10 relative/absolute tolerance on a
deterministic correlated two-dimensional fixture and diagonal fitting cloud.
Compare normalization integrals, mixture probabilities, both proposal branches
under supplied common noise, and both profiled density and relative-shape
objectives with analytical gradients. For density loss, use the same fixed
normalization by setting R's log_density_scale=-d*log(2*pi)/2; map R
log-variance derivatives to TF log-standard-deviation derivatives by factor 2.
No optimizer result or whole filter equivalence is claimed.

Trace `execute_iapf` -> `make_density_recursive_fit_kernel` ->
`bounded_density_fit` -> `_density_profile` and
`execute_iapf` -> `make_fitted_twist_kernel` -> `normalizer`/
`twisted_transition`. Add executable wiring checks and a real endpoint call
that must reject d=o=2 under the current scalar restriction. This prevents
primitive agreement from being mistaken for general consumer capability.

Use R to export JSON-free CSV fixtures and expected values, then use actual
TensorFlow functions inside a stable-signature tf.function. This is a tiny
CPU FP64 independent-reference diagnostic; intentionally hide GPU devices,
disable JIT explicitly, and record that this is not default GPU/XLA evidence.
Use no NumPy, autodiff or pfor. Model parameters are algebraic fixtures, not
tuned defaults. The proposed relative loss is a separately labelled objective.

Invalid/nonfinite output, wrong consumer wiring, unexpected endpoint behavior,
or parity failure veto the corresponding implementation claim and trigger
localized investigation. Relative errors, runtime and source differences are
explanatory, not scientific promotion criteria. Heuristic ranking and stochastic
uncertainty are N/A for algebraic parity. No full-paper, filter accuracy,
gradient-score, LEDH, KDM, HMC or production-readiness conclusion follows.

Budget: transfer 180 summed worker seconds from the small-dimension completion
plan only after its three datasets and diagnostics close, provided its
remaining recorded allowance is at least 180. Maximum three attempts, 60
seconds each, including fixture export, framework import and diagnostics.
Prepare and self-review now; do not launch until the source allowance is
released. Remaining source allowance stays available for eligible repairs.
Versioned output root:
`docs/plans/artifacts/iapf-r-tf-component-parity-20260921-01`.
Preserve command, source snapshots, hashes, commit, environment, CPU/JIT
exception, deterministic inputs, outputs, timing, result and checkpoint.

Skeptical pre-execution audit: PASS. The two loss normalizations and parameter
coordinates are reconciled algebraically before comparing. The fixture has
non-diagonal covariances and exercises both mixture branches, avoiding a
scalar-only false reassurance. Common deterministic inputs remove RNG
differences. Direct calls and wiring checks distinguish reusable primitives
from the presently restricted endpoint. Do not silently port the full model,
change a fitting family or claim that more repetitions settle source identity.
