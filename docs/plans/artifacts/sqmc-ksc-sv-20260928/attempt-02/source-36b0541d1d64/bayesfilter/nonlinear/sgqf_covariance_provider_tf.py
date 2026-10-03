"""Batched fixed-SGQF moments for the shared LEDH proposal lifecycle.

Uses repository sparse-grid nodes and the shared analytical quadrature moment
implementation. Signed weights integrate moments; they never sample ancestors.
No covariance repair is applied: nonpositive covariance is a validity failure.
"""
import tensorflow as tf

from bayesfilter.nonlinear.fixed_sgqf_tf import (
    tf_fixed_sgqf_cloud, tf_fixed_sgqf_level2_axis_cloud,
)
from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
    quadrature_predict_with_parameter_tangent,
    quadrature_update_with_parameter_tangent,
)


class SGQFCovarianceProvider:
    def __init__(self, dimension, level, dtype_name="float64"):
        # Build only once before graph tracing. The exact level-two constructor
        # avoids the general merge search's high-dimensional neighbor explosion.
        self.cloud = (tf_fixed_sgqf_level2_axis_cloud(dimension) if level == 2 else
                      tf_fixed_sgqf_cloud(dimension, level, merge_tolerance=0., zero_weight_tolerance=0.))
        self.point_rule = (tf.cast(self.cloud.points, dtype_name), tf.cast(self.cloud.weights, dtype_name))
        weights = self.point_rule[1]
        self.metadata = {"provider": "fixed_sgqf", "level": level,
                         "point_count": self.cloud.point_count,
                         "negative_weight_count": int(tf.reduce_sum(tf.cast(weights < 0, tf.int32)).numpy()),
                         "signed_weights_used_for_sampling": False,
                         "covariance_ridge": 0., "ridge_rationale": "retain quadrature moments; reject invalid covariance",
                         "shared_kernel": "ledh_canonical_score_stages_tf.quadrature_*_with_parameter_tangent"}

    @staticmethod
    def _checked(result):
        mean, covariance, dmean, dcovariance, shared_valid = result
        eigenvalues = tf.linalg.eigvalsh(covariance)
        scale = tf.reduce_max(tf.abs(eigenvalues), axis=-1)
        eps = tf.cast(2**-23 if covariance.dtype == tf.float32 else 2**-52, covariance.dtype)
        margin = eps * tf.cast(tf.shape(covariance)[-1], covariance.dtype) * scale
        valid = tf.reduce_all(tf.reduce_min(eigenvalues, axis=-1) > margin)
        valid &= tf.reduce_all(tf.math.is_finite(eigenvalues))
        valid &= tf.reduce_all(shared_valid)
        # This numerical return guard survives XLA's removal of Assert ops.
        moments = tuple(tf.where(valid, x, tf.cast(float("nan"), x.dtype)) for x in result[:4])
        return (*moments, shared_valid & valid)

    def predict(self, *args, **kwargs):
        return self._checked(quadrature_predict_with_parameter_tangent(
            *args, **kwargs, point_rule=self.point_rule, jitter=0.))

    def update(self, *args, **kwargs):
        return self._checked(quadrature_update_with_parameter_tangent(
            *args, **kwargs, point_rule=self.point_rule, jitter=0.))
