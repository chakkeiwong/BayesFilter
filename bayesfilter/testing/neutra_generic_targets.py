"""Parameterized exact benchmark densities and a separate truth evaluator.

Benchmark-only TensorFlow reference work. Learners receive DensityTarget,
which exposes dimension, signature and density/score callables only. Parameters,
component counts, references and responsibility features stay in the evaluator.
This is an accidental-leakage boundary, not a hostile-code security boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import tensorflow as tf

F64 = tf.float64


@dataclass(frozen=True)
class DensityTarget:
    parameter_dim: int
    signature: str
    log_prob_kernel: object
    log_prob: object
    value_score: object
    hessian: object

    def value_score_status(self, x, beta):
        value, score, valid = self.value_score(x)
        return value, score, {'bridge_valid': valid & tf.equal(beta, tf.constant(1., F64))}


from bayesfilter.testing.neutra_target_specifications import (
    fixed_specification, random_mixture_specification,
)


class ExactTargetEvaluator:
    def __init__(self, specification, *, jit_compile=True):
        self.specification = json.loads(json.dumps(specification,allow_nan=False))
        spec = self.specification
        self.dimension = len(spec['mean']) if spec['kind']=='gaussian' else spec['dimension']
        if spec['kind'] not in ('gaussian','isotropic_mixture') or self.dimension < 1:
            raise ValueError('unsupported benchmark')
        if spec['kind']=='gaussian':
            self.mean = tf.constant(spec['mean'],F64)
            self.chol = tf.linalg.cholesky(tf.constant(spec['covariance'],F64))
            tf.debugging.assert_all_finite(self.chol,'invalid covariance')
        else:
            self.centers = tf.constant(spec['centers'],F64)
            self.variances = tf.constant(spec['variances'],F64)
            self.weights = tf.constant(spec['weights'],F64)
            if self.centers.shape != (len(spec['weights']),self.dimension):
                raise ValueError('component shapes disagree')
            tf.debugging.assert_positive(self.variances)
            tf.debugging.assert_positive(self.weights)
            tf.debugging.assert_near(tf.reduce_sum(self.weights),tf.constant(1.,F64),atol=1e-12)
            if spec.get('warp_curvature',0.) and self.dimension != 2:
                raise ValueError('this shear fixture is two dimensional')
        signature = hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        signature_spec = [tf.TensorSpec([None,self.dimension],F64)]
        def value_score(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                value = self._density(x)
            score = tape.gradient(value,x)
            return value,score,tf.math.is_finite(value)&tf.reduce_all(tf.math.is_finite(score),1)
        def hessian(x):
            with tf.GradientTape(persistent=True) as outer:
                outer.watch(x)
                _,score,_ = value_score(x)
                components = [tf.reduce_sum(score[:,j]) for j in range(self.dimension)]
            return tf.stack([outer.gradient(v,x) for v in components],1)
        wrap = lambda f:tf.function(f,input_signature=signature_spec,jit_compile=jit_compile,autograph=False)
        self.target = DensityTarget(self.dimension,signature,self._density,wrap(self._density),
                                    wrap(value_score),wrap(hessian))
        self.feature_program = wrap(self._features)

    def unwarp(self, x):
        curvature = self.specification.get('warp_curvature',0.)
        if not curvature: return x
        return tf.stack((x[:,0],x[:,1]-curvature*(x[:,0]**2-self.specification['warp_center'])),1)

    def _component_logs(self, x):
        x = self.unwarp(x)
        distance = tf.reduce_sum((x[:,None,:]-self.centers[None,:,:])**2,2)
        return -.5*(distance/self.variances + self.dimension*tf.math.log(2*math.pi*self.variances))+tf.math.log(self.weights)

    def _density(self, x):
        if self.specification['kind']=='gaussian':
            standardized = tf.transpose(tf.linalg.triangular_solve(self.chol,tf.transpose(x-self.mean)))
            return -.5*(tf.reduce_sum(standardized**2,1)+self.dimension*math.log(2*math.pi))-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.chol)))
        return tf.reduce_logsumexp(self._component_logs(x),1)

    def sample(self, count, seed):
        """Exact reference generation; callers keep it in the CPU evaluator lane."""
        z = tf.random.stateless_normal([count,self.dimension],seed,dtype=F64)
        if self.specification['kind']=='gaussian':
            return tf.matmul(z,self.chol,transpose_b=True)+self.mean
        labels = tf.random.stateless_categorical(tf.math.log(self.weights)[None,:],count,
            tf.random.experimental.stateless_fold_in(seed,1))[0]
        x = z*tf.sqrt(tf.gather(self.variances,labels))[:,None]+tf.gather(self.centers,labels)
        curvature = self.specification.get('warp_curvature',0.)
        if curvature:
            x = tf.stack((x[:,0],x[:,1]+curvature*(x[:,0]**2-self.specification['warp_center'])),1)
        return x

    def _features(self, x):
        if self.specification['kind']=='gaussian':
            z = tf.transpose(tf.linalg.triangular_solve(self.chol,tf.transpose(x-self.mean)))
            return tf.concat((z,z*z,tf.cast(tf.abs(z)>2.,F64)),1)
        responsibilities = tf.nn.softmax(self._component_logs(x))
        delta = self.unwarp(x)[:,None,:]-self.centers[None,:,:]
        standardized = delta/tf.sqrt(self.variances)[None,:,None]
        first = responsibilities[:,:,None]*standardized
        second = responsibilities[:,:,None,None]*standardized[:,:,:,None]*standardized[:,:,None,:]
        radial = tf.cast(tf.reduce_sum(standardized**2,2)>4*self.dimension,F64)*responsibilities
        return tf.concat((responsibilities,tf.reshape(first,[tf.shape(x)[0],-1]),
                          tf.reshape(second,[tf.shape(x)[0],-1]),radial),1)

    def summary(self, x, log_weights):
        weights = tf.nn.softmax(log_weights)
        values = self.feature_program(x)
        return {'feature_mean':tf.reduce_sum(weights[:,None]*values,0).numpy().tolist(),
                'weight_ess':float((1/tf.reduce_sum(weights**2)).numpy()),
                'finite':bool(tf.reduce_all(tf.math.is_finite(values)).numpy()),
                'role':'explanatory_single_population_no_uncertainty_or_promotion'}
