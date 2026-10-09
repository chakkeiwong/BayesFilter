"""TensorFlow adaptation of AFT/CRAFT stage mechanics for benchmark research.

Source: google-deepmind/annealed_flow_transport @ 09688408bac3588583e719b054d03ae6d05c1c4f,
aft.py, craft.py and flow_transport.py (Apache-2.0, Copyright 2020 DeepMind
Technologies Limited). Stage maps use the existing configured IAF authority.
Source ordering, disjoint AFT populations and between-pass CRAFT updates are
explicit; numerical equivalence does not certify finite-population accuracy.
"""
from __future__ import annotations

import math
import time

import tensorflow as tf

from bayesfilter.inference.neutra_transport import NeuTraTransport
from bayesfilter.inference.neutra_warm_start_tf import (
    F64, MALAProgram, effective_sample_size, systematic_indices, transport_increment, valid_log_weights, CandidateFailure)


class FlowTransportSMC:
    def __init__(self,target,proposal,transport_config,*,particles,stages,
                 inner_updates,passes,learning_rate,gradient_clip,mutation_steps,
                 step_size,resampling_fraction,jit_compile=True,inline_target=True):
        self.target,self.proposal=target,proposal
        if min(particles,stages,inner_updates,passes,mutation_steps) < 1 or particles < 2:
            raise ValueError('flow-SMC requires positive work limits and at least two particles')
        if not 0 < resampling_fraction <= 1 or not math.isfinite(step_size) or step_size <= 0:
            raise ValueError('invalid flow-SMC mutation/resampling settings')
        self.particles,self.stages,self.inner_updates,self.passes=particles,stages,inner_updates,passes
        self.step_size,self.resampling_fraction=step_size,resampling_fraction
        self.flow=NeuTraTransport(transport_config)
        self.variables=self.flow.trainable_variables
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                               beta_1=.9,beta_2=.999,epsilon=1e-8)
        self.optimizer.build(self.variables)
        self.initial=self.snapshot()
        self.gradient_clip=tf.constant(gradient_clip,F64)

        # Compose the pure density in this graph: on the measured TF GPU
        # route, differentiating a nested compiled target incurred ~1.5 s
        # after each optimizer mutation despite one TF trace. The controlled
        # composition diagnostic preserves gradients/states to FP64 error.
        target_density=target.log_prob_kernel if inline_target else target.log_prob
        def bridge(x,beta):
            return (1-beta)*proposal.log_prob(x)+beta*target_density(x)
        self.bridge=bridge
        self.mutation=MALAProgram(bridge,target.parameter_dim,steps=mutation_steps,jit_compile=jit_compile)

        def evaluate(x,lw,previous,beta):
            x,lw=tf.stop_gradient(x),tf.stop_gradient(lw)
            with tf.GradientTape() as tape:
                y,inc=transport_increment(lambda z:bridge(z,previous),lambda z:bridge(z,beta),self.flow,x)
                weights=tf.nn.softmax(lw)
                loss=-tf.reduce_sum(weights*tf.where(weights>0,inc,tf.zeros_like(inc)))
            gradients=tuple(tape.gradient(loss,self.variables))
            norm=tf.linalg.global_norm(gradients)
            return loss,gradients,y,inc,norm

        signature=[tf.TensorSpec([particles,target.parameter_dim],F64),
                   tf.TensorSpec([particles],F64),tf.TensorSpec([],F64),tf.TensorSpec([],F64)]
        self.evaluate=tf.function(evaluate,input_signature=signature,jit_compile=jit_compile,autograph=False)

        def weights_only(x,lw,previous,beta):
            y,inc=transport_increment(lambda z:bridge(z,previous),lambda z:bridge(z,beta),self.flow,x)
            weights=tf.nn.softmax(lw)
            loss=-tf.reduce_sum(weights*tf.where(weights>0,inc,tf.zeros_like(inc)))
            return loss,y,inc
        self.weights_only=tf.function(weights_only,input_signature=signature,jit_compile=jit_compile,autograph=False)

        def weight_and_resample(x,lw,previous,beta,seed):
            _,y,inc=weights_only(x,lw,previous,beta)
            updated=lw+inc;increment=tf.reduce_logsumexp(updated)
            lw=tf.nn.log_softmax(updated);ess=effective_sample_size(lw)
            def resample():
                indices=systematic_indices(lw,tf.random.stateless_uniform([],seed,dtype=F64),particles)
                return tf.gather(y,indices),tf.fill([particles],tf.constant(-math.log(particles),F64))
            y,lw=tf.cond(ess<resampling_fraction*particles,resample,lambda:(y,lw))
            return y,lw,increment,ess
        self.weight_and_resample=tf.function(weight_and_resample,
            input_signature=signature+[tf.TensorSpec([2],tf.int32)],jit_compile=jit_compile,autograph=False)

        def apply(*grads):
            clipped,norm=tf.clip_by_global_norm(grads,self.gradient_clip)
            finite=tf.math.is_finite(norm)
            def update():
                self.optimizer.apply_gradients(zip(clipped,self.variables))
                return tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(v)) for v in self.variables]))
            return tf.cond(finite,update,lambda:tf.constant(False))
        self.apply=tf.function(apply,input_signature=[tf.TensorSpec(v.shape,v.dtype) for v in self.variables],
                               jit_compile=jit_compile,autograph=False)

    def snapshot(self):
        return tuple(tf.identity(v) for v in (*self.variables,*self.optimizer.variables))

    def restore(self,state):
        for v,x in zip((*self.variables,*self.optimizer.variables),state,strict=True):v.assign(x)

    def initial_population(self,seed):
        return self.proposal.sample(self.particles,seed),tf.fill([self.particles],
            tf.constant(-math.log(self.particles),F64)),tf.constant(0.,F64)

    def advance(self,population,previous,beta,seed):
        x,lw,log_z=population
        y,lw,increment,ess=self.weight_and_resample(x,lw,previous,beta,seed)
        log_z+=increment
        y,_,acc,bad=self.mutation.run(y,beta,tf.constant(self.step_size,F64),
            tf.random.experimental.stateless_fold_in(seed,1))
        if not bool(tf.reduce_all(tf.math.is_finite(y)).numpy()) or not bool(valid_log_weights(lw).numpy()):
            raise CandidateFailure('invalid flow-SMC state or weights')
        return (tf.stop_gradient(y),tf.stop_gradient(lw),log_z),{
            'beta':float(beta.numpy()),'ess_before_resampling':float(ess.numpy()),
            'mutation_acceptance':float(acc.numpy()),'invalid_proposals':int(bad.numpy()),
            'log_normalizer':float(log_z.numpy())}

    def run_aft(self,seed,*,callback=None):
        populations=[self.initial_population(tf.random.experimental.stateless_fold_in(seed,j)) for j in range(3)]
        records=[]
        for stage in range(1,self.stages+1):
            previous,beta=tf.constant((stage-1)/self.stages,F64),tf.constant(stage/self.stages,F64)
            self.restore(self.initial)
            best=None; best_value=math.inf
            for iteration in range(self.inner_updates):
                if iteration==0 and callback:callback({'stage':stage,'operation':'evaluate_gradient','iteration':iteration})
                started=time.monotonic()
                loss,gradients,_,_,norm=self.evaluate(*populations[0][:2],previous,beta)
                if iteration==0 and callback:callback({'stage':stage,'operation':'evaluate_validation',
                    'gradient_seconds':time.monotonic()-started,'gradient_traces':self.evaluate.experimental_get_tracing_count()})
                validation=self.weights_only(*populations[1][:2],previous,beta)[0]
                value=float(validation.numpy())
                # Author evaluates/retains best parameters BEFORE applying this update.
                if math.isfinite(value) and value<best_value:
                    best_value,best=value,self.snapshot()
                if iteration==0 and callback:callback({'stage':stage,'operation':'optimizer_update',
                    'validation_traces':self.weights_only.experimental_get_tracing_count()})
                if not bool(self.apply(*gradients).numpy()):
                    raise CandidateFailure('nonfinite AFT optimizer update')
            if best is None:raise CandidateFailure('no finite AFT validation candidate')
            self.restore(best)
            for j in range(3):
                if callback:callback({'stage':stage,'operation':'advance_population','population':j})
                populations[j],row=self.advance(populations[j],previous,beta,
                    tf.random.experimental.stateless_fold_in(seed,100*stage+j))
            row.update(stage=stage,validation_free_energy=best_value,
                       gradient_norm=float(norm.numpy()),evaluation_population='disjoint_test')
            records.append(row)
            if callback:callback(row)
        x,lw,log_z=populations[2]
        return {'complete':True,'particles':x,'log_weights':lw,'log_normalizer':log_z,'stages':records,
                'method':'aft','separate_train_validation_test':True}

    def run_craft(self,seed,*,callback=None):
        states=[self.initial for _ in range(self.stages)]
        records=[]
        for outer in range(self.passes):
            population=self.initial_population(tf.random.experimental.stateless_fold_in(seed,outer))
            gradients=[]
            for stage in range(1,self.stages+1):
                previous,beta=tf.constant((stage-1)/self.stages,F64),tf.constant(stage/self.stages,F64)
                self.restore(states[stage-1])
                loss,grad,_,_,_=self.evaluate(*population[:2],previous,beta)
                gradients.append(grad)
                population,row=self.advance(population,previous,beta,
                    tf.random.experimental.stateless_fold_in(seed,1000*outer+stage+100))
            # No map changed while the pass that generated these gradients ran.
            for stage,grad in enumerate(gradients):
                self.restore(states[stage])
                if not bool(self.apply(*grad).numpy()):raise CandidateFailure('nonfinite CRAFT optimizer update')
                states[stage]=self.snapshot()
            records.append({'pass':outer,'training_log_normalizer':float(population[2].numpy()),
                            'updates_after_complete_pass':True})
            if callback:callback(records[-1])
        population=self.initial_population(tf.random.experimental.stateless_fold_in(seed,1000000))
        evaluation=[]
        for stage in range(1,self.stages+1):
            self.restore(states[stage-1])
            population,row=self.advance(population,tf.constant((stage-1)/self.stages,F64),
                tf.constant(stage/self.stages,F64),tf.random.experimental.stateless_fold_in(seed,1000000+stage))
            evaluation.append(row)
        x,lw,log_z=population
        return {'complete':True,'particles':x,'log_weights':lw,'log_normalizer':log_z,
                'stages':evaluation,'training_passes':records,'method':'craft',
                'evaluation_maps_frozen':True,'independent_evaluation_population':True}
