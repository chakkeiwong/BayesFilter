# Refuse unresolved principal subspaces

Completed through04935: all five workers passed within this allocation.
See filter_gradient_subspace_identifiability_result_20260929.md.

Question: can a comparison correctly refuse a requested principal subspace
whose boundary eigenvalues are unresolved, while preserving all well-separated
geometry and original validation precedence? Baseline0bfae62f2 follows the
qualified angle and saved-factor diagnostics through04930. Canonical NeuTra,
deferred iAPF/KDM, optimizer settings and all matrix/angle accuracy bounds stay
unchanged.

For the largest r eigenvectors of a symmetric matrix, a unique selected
subspace requires a gap between eigenvalues d-r-1 and d-r. A rank cutting a
repeated eigenvalue does not specify a unique subspace, even when the covariance
is perfectly well-conditioned. An arbitrary eigensolver basis must not yield
an apparently meaningful passing principal-angle check. Full rank has no cut.

The proposed shared kernel guard flags a cut as numerically unresolved when
the computed gap is no larger than twice a conservative spectral uncertainty
estimate. That estimate is
||AV-V Lambda||_F + ||A||_F ||V^T V-I||_F + gamma_d ||A||_F,
where gamma_d=d*epsilon/(1-d*epsilon) and epsilon is binary64 machine precision.
The residual bounds the spectral eigenvalue error for an orthogonal basis;
the orthogonality and roundoff terms make the test conservative for a computed
basis. Twice the estimate covers uncertainty on both sides of the cut. This
is a numerical-resolution refusal, not a proof that every refused binary64
matrix has exactly repeated eigenvalues, or an interval-arithmetic certificate.
There is no tuned eigengap cap or user-overridable bypass.

The guard changes only the eligibility of an unresolved angle. It does not
change matrices, ranks, eigenvectors, finite accepted angles, optimizer settings,
or regularization. Keep the seven-value internal kernel signature. Undefined
active angle slots are nonfinite internal status values; public comparison and
stability wrappers must raise an explicit subspace-resolution error before
serialization. The native stability/initializer path must carry a distinct
error code, without overriding earlier shape/symmetry/rank errors or silently
substituting another rank. Incomplete replicate groups keep their existing
precedence. This is a Class B validity/reporting repair under AGENTS.md.

The initial budget is at most10 serialized workers,1800 CPU and1800 GPU seconds,
starting after04930 within the existing global caps. Fresh output directories
use the campaign runner's subspace_identifiability groups. CPU is explicit
reference; GPU uses a currently available nondisplay device with trusted access,
TF32 disabled for FP64 references and verified memory growth. No training/HMC,
package changes, live MacroFinance edits or algorithm substitution is involved.

Pass criteria: identical/isotropic and rotated repeated-spectrum controls
refuse partial rank; full rank and a selected subspace containing the whole
repeated cluster remain valid. Separated spectra, scale changes, known rotation
angles and original error precedence must pass on CPU/GPU, graph/XLA. Complete
saved04919/04920 fitted records must retain their existing statuses, selection
and all finite values under the original bounds. Verify the enclosing public
initializer refuses an unresolved required subspace. Run focused regressions
and the unchanged source-policy guard. New runtime loops, NumPy, host callbacks,
pfor or JIT-default changes veto adoption.

Skeptical review: the main risk is confusing matrix conditioning with subspace
identifiability, or calling a near tie exactly degenerate. The error message
must say that the selected subspace is not numerically resolved. Scale-aware
residual checks and separated-spectrum no-fire controls test false rejection.
The guard is conservative diagnostic resolution, not an optimal condition
estimator. Baseline accepted angles are mathematical comparators only where
their subspaces are resolved. A false rejection, changed healthy decision,
source drift, unavailable reference, missing ordinary-output witness or exhausted
allocation stops adoption; investigate with the smallest diagnostic. Wall times
are descriptive here; matched guard costs remain part of terminal acceptance.
Result: filter_gradient_subspace_identifiability_result_20260929.md.
