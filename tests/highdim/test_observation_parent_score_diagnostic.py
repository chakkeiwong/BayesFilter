"""Independent density/analytical-derivative checks for diagnostic proposals."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import math
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, log, logsumexp
from bayesfilter.highdim.covariance_proposal_native_models_tf import NativeModels
from bayesfilter.highdim.covariance_proposal_score_batch_tf import gaussian_mixture
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.testing.observation_parent_score_diagnostic_tf import conditional_geometry


def test_weighted_conditional_density_and_total_derivative_equal_bayes_identity():
    dtype = tf.float64
    spec = NonlinearSQMCSpec('sir_d18')
    models = NativeModels(spec, tf.constant([[.1, -.2, -.7], [0., 0., 0.]], dtype))
    random = lambda shape, seed: tf.random.stateless_normal(shape, [281028, seed], dtype=dtype)
    means = Jet(spec.initial_mean(dtype)[None, None]+3*random([2, 32, 18], 1),
                random([3, 2, 32, 18], 2))
    observation = tf.reduce_mean(models.observation(means).v, [0, 1])+random([9], 3)
    cm, diag, _, log_alpha, log_z = conditional_geometry(models, means, observation)
    x = Jet(cm.v+random([2, 32, 18], 4), random([3, 2, 32, 18], 5))
    residual = x[:, :, None]-cm[:, None]
    component = -.5*(18*math.log(2*math.pi)+log(diag).sum(-1)[:, None, None]
                     +(residual*residual/diag[:, None, None]).sum(-1))
    direct = logsumexp(component+log_alpha[:, None], -1)
    covariance = models.c(tf.eye(18, batch_shape=[2, 1], dtype=dtype))
    prior, valid = gaussian_mixture(x, means, covariance, shared_covariance=True)
    identity = models.observation_density(x, observation)+prior-log_z[:, None]
    tf.debugging.assert_equal(valid, [True, True])
    tf.debugging.assert_near(tf.reduce_sum(tf.exp(log_alpha.v), 1), tf.ones([2], dtype), atol=1e-12)
    tf.debugging.assert_near(direct.v, identity.v, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(direct.t, identity.t, atol=1e-9, rtol=1e-9)
