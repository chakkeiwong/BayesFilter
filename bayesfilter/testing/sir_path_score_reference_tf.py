"""Independent-reference analytical complete-data scores; never a filter runtime."""
import math
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_native_models_tf import NativeModels
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec


def make_path_value_score(horizon, chunk=1000):
    spec=NonlinearSQMCSpec('sir_d18'); dtype=tf.float64
    @tf.function(input_signature=[tf.TensorSpec([3],dtype),
        tf.TensorSpec([chunk,horizon+1,18],dtype),tf.TensorSpec([horizon,9],dtype)],
        jit_compile=True,autograph=False)
    def kernel(theta, paths, observations):
        models=NativeModels(spec,theta[None])
        before=models.c(tf.reshape(paths[:,:-1],[1,chunk*horizon,18]))
        current=models.c(tf.reshape(paths[:,1:],[1,chunk*horizon,18]))
        residual=current-models.transition(before)
        trans=-.5*((residual*residual).sum(-1)+18*math.log(2*math.pi))
        y=tf.reshape(tf.broadcast_to(observations[None],[chunk,horizon,9]),[chunk*horizon,9])
        obs=models.observation_density(current,y)
        terms=(trans+obs).reshape([chunk,horizon]).sum(-1)
        initial=-.5*(18*math.log(2*math.pi)+tf.reduce_sum(tf.square(paths[:,0]-spec.initial_mean(dtype)),1))
        return terms.v+initial,tf.transpose(terms.t)
    return kernel
