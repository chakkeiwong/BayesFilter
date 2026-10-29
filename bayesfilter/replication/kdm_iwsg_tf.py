"""Batched Gaussian KDM/IWSG primitive from Younis–Sudderth eqs. (14)–(15).

This validates the resampling primitive only; it is not a trained full filter.
Explicit epsilon choices reproduce the declared author numerical conventions;
zero epsilon defines the paper expression. No implicit default is selected.
"""
import math
import tensorflow as tf


class GaussianMixtureIWSG:
    def __init__(self, *, batch_size, particles, samples, dimensions,
                 density_weight_epsilon, injection_epsilon, jit_compile=True):
        if min(batch_size,particles,samples,dimensions)<1 or min(density_weight_epsilon,injection_epsilon)<0:
            raise ValueError('Invalid mixture configuration')
        self.density_epsilon=float(density_weight_epsilon)
        self.injection_epsilon=float(injection_epsilon)
        b,n,s,d=batch_size,particles,samples,dimensions
        self.signature=[tf.TensorSpec([b,n,d],tf.float64),tf.TensorSpec([b,n],tf.float64),
                        tf.TensorSpec([b,d],tf.float64),tf.TensorSpec([b,s,d],tf.float64)]
        self.evaluate=tf.function(self._evaluate,input_signature=self.signature,jit_compile=jit_compile,autograph=False)
        self.score_check=tf.function(self._score_check,input_signature=self.signature,jit_compile=jit_compile,autograph=False)

    def _density(self,locations,weights,bandwidth,samples):
        positive=weights+self.density_epsilon
        log_weights=tf.math.log(positive)-tf.math.log(tf.reduce_sum(positive,axis=-1,keepdims=True))
        residual=(samples[:,:,None,:]-locations[:,None,:,:])/bandwidth[:,None,None,:]
        component=tf.reduce_sum(-.5*residual*residual-tf.math.log(bandwidth[:,None,None,:])-.5*math.log(2*math.pi),axis=-1)
        return tf.reduce_logsumexp(component+log_weights[:,None,:],axis=-1)

    def _evaluate(self,locations,weights,bandwidth,samples):
        # Sample and denominator are fixed while the numerator differentiates.
        samples=tf.stop_gradient(samples)
        log_density=self._density(locations,weights,bandwidth,samples)
        iwsg=tf.exp(log_density-tf.stop_gradient(log_density))+self.injection_epsilon
        valid=tf.logical_and(tf.reduce_all(bandwidth>0),tf.reduce_all(weights>=0))
        valid=tf.logical_and(valid,tf.reduce_all(tf.math.is_finite(log_density)))
        return dict(log_density=log_density,iwsg=iwsg,valid=valid)

    def _score_check(self,locations,weights,bandwidth,samples):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([locations,weights,bandwidth,samples])
            result=self._evaluate(locations,weights,bandwidth,samples)
            fixed=tf.stop_gradient(samples)
            integrand=tf.reduce_sum(fixed*fixed+.3*fixed,axis=-1)
            raw=tf.reduce_sum(result['iwsg']*integrand)
            normalized=result['iwsg']/tf.reduce_sum(result['iwsg'],axis=1,keepdims=True)
            normalized_loss=tf.reduce_sum(normalized*integrand)
        raw_gradient=tape.gradient(raw,[locations,weights,bandwidth])
        normalized_gradient=tape.gradient(normalized_loss,[locations,weights,bandwidth])
        sample_gradient=tape.gradient(raw,samples)
        return dict(**result,raw_loss=raw,normalized_loss=normalized_loss,raw_gradient=raw_gradient,
                    normalized_gradient=normalized_gradient,sample_gradient_stopped=tf.constant(sample_gradient is None))
