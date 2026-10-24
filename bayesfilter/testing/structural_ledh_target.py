"""Structural campaign targets. Explicit finite LEDH approximation for Model B.

No canonical admission is implied. All filter derivatives are analytical; the
shared frozen-position-force runner and sequential HMC controller own sampling.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.testing.structural_ledh_models_tf import StructuralModelSpec, make_model_a_kalman


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def ready(x):
    if hasattr(x, 'numpy'):
        return x.numpy().tolist()
    if isinstance(x, dict):
        return {str(k):ready(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)):
        return [ready(v) for v in x]
    return x


class StructuralTarget:
    def __init__(self, dataset, design, controls, batch_size=4, jit_compile=True, dtype=tf.float32, reset_precision_policy='uniform'):
        self.spec = StructuralModelSpec(dataset['model'])
        self.p = self.spec.parameter_count
        self.horizon = len(dataset['observations'])
        self.batch_size, self.dtype = batch_size, dtype
        self.controls = ProposalControls(**controls)
        self.observations = tf.constant(dataset['observations'], dtype)
        self.design = {k:tf.constant(design[k],dtype) for k in ('initial','innovations','uniforms','beta')}
        self.n = len(design['initial'])
        self.program = make_batched_score_filter(self.spec,self.n,self.horizon,self.controls,
            dtype=dtype,jit_compile=jit_compile,batch_size=batch_size,
            fixed_draw_design=(self.design['beta'],self.design['uniforms']),reset_precision_policy=reset_precision_policy)
        identity=dict(schema='structural_ledh_target_v1',dataset=dataset,
            design=design,controls=asdict(self.controls),dtype=dtype.name,
            prior=ready(self.spec.prior_parameters(dtype)),reset='contract_e_chol_v1',
            route='conditional_innovation_completion_v1')
        if reset_precision_policy!='uniform':
            identity['reset_precision_policy']=reset_precision_policy
        self.target_signature = digest(identity)
        self.scope = self.target_signature
        self.mean,self.scale = self.spec.prior_parameters(dtype)
        signature=[tf.TensorSpec([batch_size,self.p],dtype)]

        @tf.function(input_signature=signature,jit_compile=jit_compile,autograph=False)
        def likelihood(q):
            result=self.program(q,self.observations,self.design['initial'],self.design['innovations'])
            valid=result['valid'] & (result['steps_completed']==self.horizon)
            return result['value'],result['score'],valid
        self.likelihood=likelihood

        @tf.function(input_signature=signature,jit_compile=jit_compile,autograph=False)
        def value_score(q):
            value,score,valid=likelihood(q)
            prior,prior_score=self.log_prior(q)
            v,s=value+prior,score+prior_score
            return v,s,valid & tf.math.is_finite(v) & tf.reduce_all(tf.math.is_finite(s),-1)
        self.value_score=value_score

    def log_prior(self,q):
        mean,scale=self.spec.prior_parameters(q.dtype)
        residual=(q-mean)/scale
        value=-.5*tf.reduce_sum(residual**2+math.log(2*math.pi)+2*tf.math.log(scale),-1)
        return value,-residual/scale

    def hmc_components(self,neutra_payload=None):
        from bayesfilter.inference.neural_force_hmc import (
            FrozenPositionOnlyForce,FrozenTargetPotential,PositionOnlyForceEvaluation,
            TargetPotentialEvaluation,bind_neural_force_hmc_tuning_runner)
        p,batch=self.p,self.batch_size
        kalman=(make_model_a_kalman(batch,self.horizon,tf.float64,True)
                if self.spec.name=='model_a' else None)

        @tf.function(input_signature=[tf.TensorSpec([None,p],tf.float64)],jit_compile=True,autograph=False)
        def evaluate(q):
            count=tf.shape(q)[0]
            groups=(count+batch-1)//batch
            padded=tf.concat([q,tf.repeat(q[-1:],groups*batch-count,axis=0)],axis=0)
            values=tf.TensorArray(tf.float64,size=groups,element_shape=[batch])
            scores=tf.TensorArray(tf.float64,size=groups,element_shape=[batch,p])
            statuses=tf.TensorArray(tf.bool,size=groups,element_shape=[batch])
            def step(i,values,scores,statuses):
                row=tf.slice(padded,[i*batch,0],[batch,p])
                v,s,valid=self.value_score(tf.cast(row,self.dtype))
                return i+1,values.write(i,-tf.cast(v,tf.float64)),scores.write(i,-tf.cast(s,tf.float64)),statuses.write(i,valid)
            _,values,scores,statuses=tf.while_loop(lambda i,*_:i<groups,step,
                (0,values,scores,statuses),parallel_iterations=1)
            return values.concat()[:count],scores.concat()[:count],statuses.concat()[:count]

        @tf.function(input_signature=[tf.TensorSpec([None,p],tf.float64)],jit_compile=True,autograph=False)
        def force(q):
            _,s,good=evaluate(q)
            return PositionOnlyForceEvaluation(s,good)

        @tf.function(input_signature=[tf.TensorSpec([None,p],tf.float64)],jit_compile=True,autograph=False)
        def endpoint(q):
            if kalman is not None:
                # Independent exact endpoint: never evaluate LEDH here.
                count=tf.shape(q)[0];groups=(count+batch-1)//batch
                padded=tf.concat([q,tf.repeat(q[-1:],groups*batch-count,axis=0)],axis=0)
                values=tf.TensorArray(tf.float64,size=groups,element_shape=[batch])
                def step(i,values):
                    row=tf.slice(padded,[i*batch,0],[batch,p])
                    exact=kalman(row,tf.cast(self.observations,tf.float64))
                    prior,_=self.log_prior(row)
                    return i+1,values.write(i,-exact['value']-prior)
                _,values=tf.while_loop(lambda i,*_:i<groups,step,(0,values),parallel_iterations=1)
                v=values.concat()[:count]
                good=tf.math.is_finite(v)
            else:
                v,_,good=evaluate(q)
            return TargetPotentialEvaluation(v,dict(value_valid=good,force_valid=good,
                numerical_failure=~good,outside_declared_support=tf.zeros_like(good)))
        scope=digest(dict(target=self.scope,endpoint='kalman' if kalman is not None else 'finite_ledh'))
        frozen_force=FrozenPositionOnlyForce(force,scope+':analytical_fp32',semantics='deterministic_position_only_proposal_field')
        frozen_target=FrozenTargetPotential(endpoint,scope+':endpoint')
        transport=None
        if neutra_payload is not None:
            from bayesfilter.testing.surrogate_hmc_sir_neutra import frozen_neutra_bindings
            # Artifact parsing/finite assertions are host work. The returned
            # pure numerical functions are compiled with the GPU HMC caller.
            with tf.device('/CPU:0'):
                parts=frozen_neutra_bindings(neutra_payload,target_signature=self.target_signature,
                    raw_force=frozen_force,raw_target=frozen_target,target_scope=scope,
                    model_parameter_transform=self.spec.physical,
                    model_parameter_identity=scope+':physical_parameters_v1')
            binding,transport=parts['binding'],parts['transport']
            frozen_force,frozen_target=parts['force'],parts['target']
            scope=scope+':'+parts['artifact'].manifest.transport_hash
        else:
            binding=bind_neural_force_hmc_tuning_runner(force=frozen_force,target=frozen_target,target_scope=scope,
                model_parameter_transform=self.spec.physical,
                model_parameter_identity=scope+':physical_parameters_v1')
        class Adapter:
            target_scope=scope
            def adapter_signature(self):
                return scope+':'+binding.binding_hash
        return dict(adapter=Adapter(),binding=binding,scope=scope,transport=transport,
                    force=frozen_force,target=frozen_target)
