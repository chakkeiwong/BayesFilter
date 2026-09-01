"""Does the full-rank route's telemetry still expose a violated pivot under XLA?"""
import numpy as np, tensorflow as tf
from bayesfilter.nonlinear.factor_srukf_tf import (
    TFFactorSRUKFDerivatives, TFFactorSRUKFModel, tf_factor_srukf_value_and_score,
)
b, pdim = 1, 1
# Nearly rank-deficient carried factor: second state coordinate variance ~1e-28
model = TFFactorSRUKFModel(
    tf.constant([[0.0, 0.0]], tf.float64),
    tf.constant([[[1.0, 0.0], [0.0, 1.0e-14]]], tf.float64),
    tf.constant([[[1.0e-14, 0.0], [0.0, 1.0e-14]]], tf.float64),
    tf.constant([[[0.5]]], tf.float64),
    lambda x, q: x + q,
    lambda x: x[..., :1],
)
derivatives = TFFactorSRUKFDerivatives(
    tf.zeros([b, pdim, 2], tf.float64), tf.zeros([b, pdim, 2, 2], tf.float64),
    tf.zeros([b, pdim, 2, 2], tf.float64), tf.zeros([b, pdim, 1, 1], tf.float64),
    lambda x, q: tf.broadcast_to(tf.eye(2, dtype=tf.float64), [b, tf.shape(x)[1], 2, 2]),
    lambda x, q: tf.broadcast_to(tf.eye(2, dtype=tf.float64), [b, tf.shape(x)[1], 2, 2]),
    lambda x, q: tf.zeros([b, pdim, tf.shape(x)[1], 2], tf.float64),
    lambda x: tf.broadcast_to(tf.constant([[[1.0, 0.0]]], tf.float64), [b, tf.shape(x)[1], 1, 2]),
    lambda x: tf.zeros([b, pdim, tf.shape(x)[1], 1], tf.float64),
)
obs = tf.constant([[[0.1]]], tf.float64)
res = tf_factor_srukf_value_and_score(obs, model, derivatives, jit_compile=True)
print("XLA default run completed with rel pivot:", float(res.diagnostics["relative_qr_pivot"][0]))
print("value:", float(res.log_likelihood[0]), "score:", res.score.numpy().ravel())
print("below 1e-12 default floor:", float(res.diagnostics["relative_qr_pivot"][0]) < 1e-12)
print("value finite:", np.isfinite(res.log_likelihood.numpy()).all())
