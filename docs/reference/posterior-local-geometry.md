# Fixed-center posterior curvature refinement

`bayesfilter.inference.refine_posterior_local_curvature` is an opt-in second
stage after localization. It keeps your center fixed, estimates regional
quadratic score curvature, validates that curvature on fresh points, and returns
a lower-Cholesky **position** factor. It does not modify the supplied tensors or
any existing guide. Adopt the returned factor explicitly only after acceptance.
This API does not build an HMC mass artifact or certify a MAP, posterior
covariance, whitening, or sampler convergence.

## Minimal TensorFlow example

Call outside `tf.function`, using float64 center and a positive-diagonal lower
Cholesky pilot factor in the same parameter chart as the target. The target must
return the log density and its matching raw-coordinate analytical gradient.
Eligibility is required: a finite rejection sentinel is not a valid target row.
An always-true eligibility function is valid only for unrestricted targets.

```python
import tensorflow as tf
from bayesfilter.inference import (
    PosteriorCurvatureRefinementConfig,
    refine_posterior_local_curvature,
)

precision = tf.constant([[3.0, 0.6], [0.6, 1.5]], tf.float64)
location = tf.constant([0.7, -0.4], tf.float64)

@tf.function(input_signature=[tf.TensorSpec([64, 2], tf.float64)])
def value_and_score(theta):
    delta = theta - location
    scores = -delta @ precision
    return 0.5 * tf.reduce_sum(delta * scores, axis=1), scores

def eligible(theta):
    return tf.ones(tf.shape(theta)[0], tf.bool)

result = refine_posterior_local_curvature(
    value_and_score,
    center=tf.constant([0.6, -0.3], tf.float64),
    pilot_factor=tf.constant([[1.2, 0.0], [0.2, 0.8]], tf.float64),
    batched_eligibility_fn=eligible,
    config=PosteriorCurvatureRefinementConfig(
        seed=20260908,
        lineage={"target": "two-dimensional-gaussian", "chart": "raw"},
    ),
)
if not result.accepted:
    raise RuntimeError(f"curvature rejected: {result.status}")
guide_center = result.center
guide_position_factor = result.refined_factor
```

Configure TensorFlow devices before its import/initialization in an application.
GPU applications must follow BayesFilter's memory-growth policy; the documented
validation for this helper is deliberately CPU-only, not GPU qualification.

## Mathematical conventions

For row positions `theta = c + z @ F.T`, pull back scores with
`scores_z = scores_theta @ F`. Fit the local model

```text
g_z(z) ≈ g_z(0) - K z                   (column notation)
response = g_z(0)[None, :] - scores_z
coefficient = least_squares(z, response)
K = (coefficient + coefficient.T) / 2
Sigma_theta = F K^-1 F.T
Lnew Lnew.T = Sigma_theta
```

The exact nonzero center score supplies the fixed intercept. No probe can move
the center, even if it has a higher log density. The fit uses unrestricted score
least squares followed by symmetrizing, preserving the existing dense fitter's
method; it is **not** the exact symmetry-constrained least-squares optimum for
arbitrary nonquadratic scores. For a quadratic log density and full-rank design,
it recovers the exact precision in exact arithmetic, regardless of pilot scale
or center offset. For nonquadratic targets it is a regional approximation.

`result.precision_z` is **pilot-coordinate** K, not the inverse of
`result.refined_covariance`. The raw covariance is built without explicit
inversion: solve `chol(K) A = F.T`, form `Sigma_theta = A.T @ A`, and take its
lower Cholesky. Position covariance is the inverse of raw negative-log
curvature; do not interchange it with a momentum-mass convention.

## Sampling and acceptance

Default fitting points have independent Uniform(-1,1) pilot coordinates. This
means a half-width of one per decorrelated coordinate and coordinate SD
`1/sqrt(3)`, **not** a Euclidean radius-one ball or a joint 68% probability region.
For a Euclidean local envelope, set
`fit_design="uniform_ball"`: `coordinate_half_width` is then the Euclidean
radius in pilot coordinates and every fit point satisfies `||z||_2 <= radius`.
The returned factor is not multiplied by a box width or by a shrinking local
radius. The independent proposal audit always uses standard normal draws in
refined coordinates. A successful small-ball fit does not establish that the
Gaussian initializer has acceptable support and score behavior; that separate
veto remains necessary before the helper returns usable geometry.

The sequence is: center → independent training/selection partitions for each
replicate → freeze their mean precision → untouched audit → factor construction
→ refined-factor Gaussian proposal check. Every replicate must pass full-rank,
raw SPD, conditioning, and selection residual checks. All pairwise generalized
precision eigenvalues must lie in `[1/1.5, 1.5]` by default, including overall
scale changes. Audit and proposal may only veto: no refit, clipping, fallback,
resampling invalid rows, width reduction, or automatic recentering follows.

The relative score residual is the Frobenius norm of the prediction error
divided by the norm of the center-subtracted observed score. Computation is
scaled before squaring to avoid needless overflow/underflow. Zero response and
zero prediction have residual zero, but flat curvature still fails raw SPD.
The defaults below are engineering hypotheses, not universal statistical tests.

| Setting | Default |
| --- | --- |
| Replicates / rows per partition | 2 / `max(32, 4D)` |
| Fixed target batch size / row budget | 64 / 10000 physical attempted rows |
| Pilot coordinate half-width | 1.0 |
| Fit design | `uniform_box` |
| Proposal distribution (fixed) | `standard_normal` |
| Maximum design / precision condition number | 1e6 / 1e10 |
| Selection / untouched audit relative RMSE cap | 0.20 / 0.20 |
| Refined Gaussian proposal relative RMSE cap | 0.35 |
| Maximum pairwise generalized-eigenvalue spread | 1.5 |
| Factor reconstruction absolute / relative tolerance | 1e-12 / 1e-10 |

`center_score_refined_l2 = ||Lnew.T g_theta(c)||` uses the unshrunk refined factor.
It is diagnostic only: the caller's proposed `<=0.5` localization policy is not
an extra center optimizer or an acceptance gate here. Accurate curvature alone
does not make an arbitrarily off-center guide suitable for training.

## Budget and failure behavior

For `N` logical rows per partition, `B` batch size, and `R` replicates, preflight
requires `B + (2R + 2) * B * ceil(N/B)` physical rows. A center occupies one
padded batch. Each final partial batch repeats its last valid row; duplicates
are checked but excluded from fits/statistics. With `N=33, B=8, R=2`, a complete
run uses 199 logical rows and 248 physical rows, not 199 physical target calls.
An over-budget configuration raises before any callback.

Diagnostics distinguish physical/logical/padded **attempts**, eligibility
batches, target callback attempts and rows, and completed logical rows. An
ineligible batch counts toward attempted rows but never reaches the target.
A target batch returning nonfinite output counts as a target attempt but not a
completed batch. Seeds and roles for each attempted partition are recorded;
later unattempted partitions do not appear. Callbacks always see shape `[B,D]`.

Success status is `eligible_for_local_position_factor`. Rejection statuses:

- Support/target: `ineligible_target_row`, `nonfinite_position`,
  `nonfinite_target_value`, `nonfinite_target_score`, `nonfinite_transformed_score`.
- Fit/stability: `curvature_fit_rejected`, `curvature_fit_numerical_failure`,
  `replicate_instability`, `replicate_numerical_failure`,
  `nonfinite_consensus_precision`.
- Holdouts/factor: `audit_rejected`, `factorization_failed`,
  `factor_reconstruction_failed`, `nonfinite_refined_center_score`,
  `refined_proposal_rejected`.

Every rejection has `refined_factor`, `refined_covariance`, and `precision_z`
set to `None`. Inspect `failure_partition` and replicate metrics for the reason.
Programming mistakes (wrong dtype/shape, invalid config, callback exception)
raise instead of being disguised as target rejection. `result.payload()` is
strict-JSON compatible: nonfinite diagnostic numbers become null, rejection
status remains explicit, and unsupported lineage objects are not stringified.

An explicit bounded retry may be appropriate for a poor pilot or an unsuitable
center, but it belongs to a separate caller decision. This helper never changes
the target or silently substitutes narrower geometry to pass its checks.

## Validation and limits

`tests/test_posterior_curvature_refinement.py` covers numerical/contract gates;
`tests/test_posterior_curvature_refinement_regressions.py` covers legacy parity,
affine units, lazy imports, and reproducibility. The independent executable
`scripts/run_posterior_curvature_refinement_integration.py` uses only analytical
synthetic targets (1D, correlated/ill-conditioned/142D Gaussians, quartic, banana,
mixture saddle, flat direction, bounded sentinel) and verifies fixture scores
against autodiff. It does not import MacroFinance or use its data.

The dense TensorFlow kernel has graph/eager parity coverage. The one-shot
orchestrator deliberately runs eagerly; it is not differentiable through its
host decisions and rejects tracing the entire public function. A fixed-shape
compiled target is supported. Host-XLA Gaussian target evidence is not evidence
for another target, full-refiner XLA, GPU scalability, or full-chain TFP/XLA.
The unit-ball design defines one unit of distance in the supplied pilot
metric. It is not the unique multivariate meaning of a standard deviation,
not a typical Gaussian shell, and not a posterior probability statement. It
does not alter the fixed-center least-squares equations. See
`../plans/bayesfilter-posterior-curvature-refinement-plan-2026-09-08.md` and
the same-prefix result memo for the exact command manifest and final evidence.
