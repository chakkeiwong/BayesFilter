"""Batched kernels for source-mapped NeuTra warm-start research.

Global MH/MALA follow flonaco sampling.py at 6b9286b4e581 (MIT):
Copyright (c) 2021 Marylou Gabrié, Grant Rotskoff and Eric Vanden-Eijnden.
The retained source licence is .localresources/flonaco-author-20260929/upstream/LICENSE.
This TF adaptation uses explicit random arrays and log acceptance ratios.
It supplies no independent transport architecture or numerical defaults.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_probability as tfp

F64 = tf.float64


class CandidateFailure(RuntimeError):
    """A bounded method/setting failed; other planned candidates remain valid."""


def value_score(log_density, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = log_density(x)
    score = tape.gradient(value, x)
    valid = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score), axis=1)
    return value, score, valid


def independence_step(log_density, log_proposal, x, proposed, uniform):
    ratio = log_density(proposed) - log_density(x) + log_proposal(x) - log_proposal(proposed)
    finite = tf.math.is_finite(ratio) & tf.reduce_all(tf.math.is_finite(proposed), axis=1)
    accepted = finite & (tf.math.log(uniform) < tf.minimum(ratio, 0.))
    return tf.where(accepted[:, None], proposed, x), accepted, ratio, finite


def mala_step(target_value_score, x, gaussian_noise, uniform, step_size):
    """flonaco convention y=x+dt*score(x)+sqrt(2*dt)*noise, beta=1."""
    dt = tf.cast(step_size, F64)
    old_value, old_score, old_valid = target_value_score(x)
    proposal = x + dt*old_score + tf.sqrt(2*dt)*gaussian_noise
    new_value, new_score, new_valid = target_value_score(proposal)
    reverse = tf.reduce_sum(tf.square(x-proposal-dt*new_score), axis=1)/(4*dt)
    forward = tf.reduce_sum(tf.square(proposal-x-dt*old_score), axis=1)/(4*dt)
    ratio = new_value-old_value-reverse+forward
    valid = old_valid & new_valid & tf.math.is_finite(ratio)
    accepted = valid & (tf.math.log(uniform) < tf.minimum(ratio, 0.))
    return tf.where(accepted[:, None], proposal, x), accepted, ratio, valid


def normalized_weights(log_weights):
    return tf.nn.log_softmax(log_weights)


def valid_log_weights(log_weights):
    """Zero weights (-inf) are valid; NaN, +inf and empty support are not."""
    return (tf.reduce_any(tf.math.is_finite(log_weights)) &
            tf.reduce_all(tf.math.is_finite(log_weights) | (log_weights == -math.inf)))


def effective_sample_size(log_weights):
    weights = normalized_weights(log_weights)
    return tf.exp(-tf.reduce_logsumexp(2*weights))


def systematic_indices(log_weights, uniform, count):
    cdf = tf.cumsum(tf.exp(normalized_weights(log_weights)))
    positions = (tf.cast(tf.range(count), F64)+tf.cast(uniform, F64))/tf.cast(count, F64)
    return tf.minimum(tf.searchsorted(cdf, positions, side='right'), tf.size(log_weights)-1)


def next_temperature(log_weights, log_ratio, beta, cess_fraction, *, iterations=40):
    """Weighted CESS/N, distinct from ordinary ESS of cumulative weights."""
    lw = normalized_weights(log_weights)
    threshold = tf.math.log(tf.cast(cess_fraction, F64))

    def admissible(b):
        # At zero increment an excluded point contributes its old weight, not
        # NaN from 0 * -inf. Excluded rows never re-enter through 0 * +inf.
        delta = tf.where((b == beta) | ~tf.math.is_finite(lw),
                         tf.zeros_like(log_ratio), (b-beta)*log_ratio)
        return 2*tf.reduce_logsumexp(lw+delta)-tf.reduce_logsumexp(lw+2*delta) >= threshold

    def body(i, lo, hi):
        mid = (lo+hi)/2
        ok = admissible(mid)
        return i+1, tf.where(ok, mid, lo), tf.where(ok, hi, mid)

    _, lo, _ = tf.while_loop(lambda i, *_: i < iterations, body,
                            (tf.constant(0), beta, tf.constant(1., F64)))
    return tf.where(admissible(tf.constant(1., F64)), tf.constant(1., F64), lo)


def transport_increment(log_previous, log_next, transport, x):
    """Negative of AFT get_delta: correct orientation and Jacobian sign."""
    y, ld = transport.forward_and_logdet(x)
    return y, log_next(y)+ld-log_previous(x)


class MALAProgram:
    def __init__(self, log_density_by_beta, dimension, *, steps, jit_compile=True):
        self.log_density_by_beta = log_density_by_beta
        self.steps = steps

        def program(x, beta, dt, seed):
            trace = tf.TensorArray(F64, size=steps+1, clear_after_read=False).write(0, x)
            def body(i, current, trace, accepted, invalid):
                step_seed = tf.random.experimental.stateless_fold_in(seed, i)
                noise = tf.random.stateless_normal(tf.shape(current), step_seed, dtype=F64)
                uniforms = tf.random.stateless_uniform([tf.shape(current)[0]],
                    tf.random.experimental.stateless_fold_in(step_seed, 1), dtype=F64)
                current, acc, _, valid = mala_step(
                    lambda y: value_score(lambda z: log_density_by_beta(z,beta), y),
                    current, noise, uniforms, dt)
                return (i+1, current, trace.write(i+1,current),
                        accepted+tf.reduce_sum(tf.cast(acc,F64)),
                        invalid+tf.reduce_sum(tf.cast(~valid,tf.int32)))
            _, last, trace, accepted, invalid = tf.while_loop(
                lambda i,*_: i < steps, body,
                (tf.constant(0),x,trace,tf.constant(0.,F64),tf.constant(0)))
            return last, trace.stack(), accepted/(steps*tf.cast(tf.shape(x)[0],F64)), invalid

        self.run = tf.function(program, input_signature=[
            tf.TensorSpec([None,dimension],F64), tf.TensorSpec([],F64),
            tf.TensorSpec([],F64), tf.TensorSpec([2],tf.int32)],
            jit_compile=jit_compile, autograph=False)


class GabrieProgram:
    """Pinned global-then-MALA sampling block; optimization stays outside it."""
    def __init__(self, transport, target, *, walkers, steps, jit_compile=True):
        dimension = transport.parameter_dim

        def program(x, dt, seed):
            trace = tf.TensorArray(F64, size=steps)
            def body(i,current,trace,global_acc,local_acc,invalid):
                key = tf.random.experimental.stateless_fold_in(seed,i)
                proposed,_ = transport.forward_and_logdet(tf.random.stateless_normal(
                    [walkers,dimension],key,dtype=F64))
                current,acc,_,valid = independence_step(target.log_prob, transport.log_prob,
                    current, proposed,tf.random.stateless_uniform([walkers],
                    tf.random.experimental.stateless_fold_in(key,1),dtype=F64))
                current,acc_local,_,valid_local = mala_step(target.value_score,current,
                    tf.random.stateless_normal([walkers,dimension],
                        tf.random.experimental.stateless_fold_in(key,2),dtype=F64),
                    tf.random.stateless_uniform([walkers],
                        tf.random.experimental.stateless_fold_in(key,3),dtype=F64),dt)
                return (i+1,current,trace.write(i,current),
                    global_acc+tf.reduce_mean(tf.cast(acc,F64)),
                    local_acc+tf.reduce_mean(tf.cast(acc_local,F64)),
                    invalid+tf.reduce_sum(tf.cast(~valid|~valid_local,tf.int32)))
            _,last,trace,ga,la,bad = tf.while_loop(lambda i,*_:i<steps,body,
                (tf.constant(0),x,trace,tf.constant(0.,F64),tf.constant(0.,F64),tf.constant(0)))
            return last,tf.reshape(trace.stack(),[walkers*steps,dimension]),ga/steps,la/steps,bad

        self.run = tf.function(program,input_signature=[tf.TensorSpec([walkers,dimension],F64),
            tf.TensorSpec([],F64),tf.TensorSpec([2],tf.int32)],jit_compile=jit_compile,autograph=False)


@dataclass(frozen=True)
class SMCConfig:
    particles: int
    mutation_steps: int
    max_stages: int
    cess_fraction: float
    resampling_fraction: float
    step_size: float
    waste_free: bool = False
    jit_compile: bool = True
    step_size_schedule: tuple[tuple[float,float], ...] = ()
    temperature_schedule: tuple[float, ...] = ()
    use_resampling: bool = True

    def __post_init__(self):
        if self.particles < 2 or self.mutation_steps < 1 or self.max_stages < 1:
            raise ValueError('SMC requires at least two particles and positive work limits')
        if not 0 < self.cess_fraction < 1 or not 0 < self.resampling_fraction <= 1:
            raise ValueError('SMC overlap and resampling fractions outside their domains')
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError('SMC mutation step must be positive and finite')
        if self.waste_free and (self.mutation_steps < 2 or self.particles % self.mutation_steps):
            raise ValueError('waste-free trajectories require at least two states and must divide particles')
        if type(self.use_resampling) is not bool:
            raise ValueError('use_resampling must be a boolean')
        if not self.use_resampling and not self.temperature_schedule:
            raise ValueError('AIS requires an explicit frozen temperature schedule')
        if self.temperature_schedule:
            schedule = self.temperature_schedule
            if (len(schedule) < 2 or schedule[0] != 0. or schedule[-1] != 1.
                    or any(not math.isfinite(b) for b in schedule)
                    or any(a >= b for a,b in zip(schedule, schedule[1:]))):
                raise ValueError('frozen temperatures must increase strictly from zero to one')
            if len(schedule)-1 > self.max_stages or self.waste_free:
                raise ValueError('frozen schedule exceeds stage budget or requests unsupported waste-free retention')
        if self.step_size_schedule:
            betas=[float(b) for b,_ in self.step_size_schedule]
            if betas[0]!=0. or betas!=sorted(set(betas)) or betas[-1]>1.:
                raise ValueError('SMC step schedule must start at zero and increase within [0,1]')
            if any(not math.isfinite(dt) or dt<=0 for _,dt in self.step_size_schedule):
                raise ValueError('SMC scheduled steps must be positive and finite')


class AnnealedSMC:
    """Resample/move at previous temperature, then weight; particles source order.

Waste-free retention follows MCMCSequenceWF: ancestor plus P-1 transitions,
with fewer resampled ancestors and trajectory-major concatenation. Adaptation
and artifact bookkeeping are host orchestration; kernels remain batched TF.
"""
    def __init__(self,target,proposal,config):
        self.target,self.proposal,self.config = target,proposal,config
        if config.waste_free and config.particles % config.mutation_steps:
            raise ValueError("particle count must divide into complete retained trajectories")
        def bridge(x,beta):
            return (1-beta)*proposal.log_prob(x)+beta*target.log_prob(x)
        self.bridge = bridge
        self.mutation = MALAProgram(bridge,target.parameter_dim,
            steps=config.mutation_steps-1 if config.waste_free else config.mutation_steps,
            jit_compile=config.jit_compile)
        self.temperature = tf.function(next_temperature, input_signature=[
            tf.TensorSpec([config.particles],F64),tf.TensorSpec([config.particles],F64),
            tf.TensorSpec([],F64),tf.TensorSpec([],F64)],jit_compile=config.jit_compile)

        def weight_stage(x,lw,beta):
            ratio=target.log_prob(x)-proposal.log_prob(x)
            next_beta=self.temperature(lw,ratio,beta,tf.constant(config.cess_fraction,F64))
            raw=lw+(next_beta-beta)*ratio
            return next_beta,tf.nn.log_softmax(raw),tf.reduce_logsumexp(raw)
        self.weight_stage=tf.function(weight_stage,input_signature=[
            tf.TensorSpec([config.particles,target.parameter_dim],F64),
            tf.TensorSpec([config.particles],F64),tf.TensorSpec([],F64)],
            jit_compile=config.jit_compile,autograph=False)
        count=config.particles//config.mutation_steps if config.waste_free else config.particles
        self.resample=tf.function(lambda lw,u:systematic_indices(lw,u,count),
            input_signature=[tf.TensorSpec([config.particles],F64),tf.TensorSpec([],F64)],
            jit_compile=config.jit_compile,autograph=False)
        if config.temperature_schedule:
            self._build_fixed_program()

    def _build_fixed_program(self):
        """Neal AIS / resample-move SMC: weight, optional resample, mutate.

        Both fixed routes mutate at EVERY stage. Disabling resampling in the
        old adaptive route merely weighted an unmoving proposal population.
        All stage and mutation loops here have one traced TensorFlow body.
        """
        cfg = self.config
        betas = tf.constant(cfg.temperature_schedule, F64)
        steps = tf.constant([next((dt for lo,dt in reversed(cfg.step_size_schedule)
                                  if b >= lo), cfg.step_size)
                             for b in cfg.temperature_schedule[1:]], F64)
        stages = len(cfg.temperature_schedule)-1

        def program(initial, seed):
            log_weights = tf.fill([cfg.particles], tf.constant(-math.log(cfg.particles), F64))
            history = tf.TensorArray(F64, size=stages)
            def body(i, x, lw, log_z, roots, history, valid):
                key = tf.random.experimental.stateless_fold_in(seed, i)
                ratio = self.target.log_prob(x)-self.proposal.log_prob(x)
                raw = lw+(betas[i+1]-betas[i])*ratio
                increment = tf.reduce_logsumexp(raw)
                lw = raw-increment
                ess = effective_sample_size(lw)
                resample = tf.constant(cfg.use_resampling) & (ess < cfg.resampling_fraction*cfg.particles)
                def do_resample():
                    indices = systematic_indices(lw,tf.random.stateless_uniform([],key,dtype=F64),cfg.particles)
                    return tf.gather(x,indices), tf.fill([cfg.particles],tf.constant(-math.log(cfg.particles),F64)), tf.gather(roots,indices)
                x,lw,roots = tf.cond(resample,do_resample,lambda:(x,lw,roots))
                x,_,acceptance,invalid = self.mutation.run(x,betas[i+1],steps[i],
                    tf.random.experimental.stateless_fold_in(key,1))
                x = tf.ensure_shape(x,[cfg.particles,self.target.parameter_dim])
                state_ok = valid_log_weights(lw) & tf.reduce_all(tf.math.is_finite(x))
                state_ok &= tf.reduce_all(tf.math.is_finite(self.target.log_prob(x)))
                valid &= state_ok & tf.math.is_finite(increment) & tf.equal(invalid,0)
                row = tf.stack((betas[i+1],ess,tf.cast(resample,F64),acceptance,
                                tf.cast(invalid,F64),log_z+increment))
                return i+1,x,lw,log_z+increment,roots,history.write(i,row),valid
            _,x,lw,log_z,roots,history,valid = tf.while_loop(
                lambda i,*_:i<stages,body,(tf.constant(0),initial,log_weights,
                tf.constant(0.,F64),tf.range(cfg.particles),history,tf.constant(True)))
            return x,lw,log_z,roots,history.stack(),valid
        self.fixed_program = tf.function(program,input_signature=[
            tf.TensorSpec([cfg.particles,self.target.parameter_dim],F64),
            tf.TensorSpec([2],tf.int32)],jit_compile=cfg.jit_compile,autograph=False)

    def _run_fixed(self, seed, callback):
        initial = self.proposal.sample(self.config.particles,seed)
        x,lw,log_z,roots,history,valid = self.fixed_program(initial,
            tf.random.experimental.stateless_fold_in(seed,271))
        rows = []
        for i,values in enumerate(history.numpy().tolist()):
            row = dict(zip(('beta','ess','resampled','mutation_acceptance',
                            'invalid_proposals','log_normalizer'),values,strict=True))
            row.update(stage=i,resampled=bool(row['resampled']),
                       mutation_steps=self.config.mutation_steps,
                       mutation_beta=row['beta'],mutation_at_every_stage=True)
            rows.append(row)
            if callback is not None: callback(row)
        return {'complete':bool(valid.numpy()),'reason':None if bool(valid.numpy()) else 'invalid_fixed_stage',
                'particles':x,'log_weights':lw,'log_normalizer':log_z,'roots':roots,
                'stages':rows,'method':'smc' if self.config.use_resampling else 'ais',
                'frozen_schedule':list(self.config.temperature_schedule)}

    def run(self,seed,*,callback=None):
        cfg=self.config
        if cfg.temperature_schedule:
            return self._run_fixed(seed,callback)
        x=self.proposal.sample(cfg.particles,seed)
        lw=tf.fill([cfg.particles],tf.constant(-math.log(cfg.particles),F64))
        roots=tf.range(cfg.particles)
        beta=tf.constant(0.,F64)
        log_z=tf.constant(0.,F64)
        rows=[]
        for stage in range(cfg.max_stages):
            key=tf.random.experimental.stateless_fold_in(seed,stage+1)
            ess=effective_sample_size(lw)
            resample=bool((ess < cfg.resampling_fraction*cfg.particles).numpy())
            accept,invalid=tf.constant(0.,F64),tf.constant(0)
            dt=next((step for lower,step in reversed(cfg.step_size_schedule)
                     if float(beta.numpy())>=lower),cfg.step_size)
            displacement=tf.zeros([self.target.parameter_dim],F64)
            # Diagnostic only: the weighted population immediately before any
            # resampling/mutation. A collapsed scale remains explicitly zero.
            weights=tf.exp(normalized_weights(lw))
            center=tf.reduce_sum(weights[:,None]*x,axis=0)
            population_variance=tf.reduce_sum(weights[:,None]*(x-center)**2,axis=0)
            if resample:
                n=cfg.particles//cfg.mutation_steps if cfg.waste_free else cfg.particles
                idx=self.resample(lw,tf.random.stateless_uniform([],key,dtype=F64))
                x=tf.gather(x,idx)
                before=tf.identity(x)
                roots=tf.gather(roots,idx)
                x,trace,accept,invalid=self.mutation.run(x,beta,tf.constant(dt,F64),
                    tf.random.experimental.stateless_fold_in(key,1))
                displacement=tf.reduce_mean((x-before)**2,0)
                if cfg.waste_free:
                    x=tf.reshape(trace,[cfg.particles,self.target.parameter_dim])
                    roots=tf.tile(roots,[cfg.mutation_steps])
                lw=tf.fill([cfg.particles],tf.constant(-math.log(cfg.particles),F64))
            next_beta,lw,increment=self.weight_stage(x,lw,beta)
            log_z+=increment
            valid=bool(valid_log_weights(lw).numpy())
            rows.append({"stage":stage,"beta":float(next_beta.numpy()),
                "ess":float(effective_sample_size(lw).numpy()),"resampled":resample,
                "mutation_acceptance":float(accept.numpy()),"invalid_proposals":int(invalid.numpy()),
                "mutation_beta":float(beta.numpy()),"mutation_step_size":dt,
                "mutation_steps":cfg.mutation_steps if resample else 0,
                "mutation_mean_squared_displacement":displacement.numpy().tolist(),
                "mutation_population_variance":population_variance.numpy().tolist(),
                "unique_roots":int(tf.size(tf.unique(roots).y).numpy()),
                "log_normalizer":float(log_z.numpy())})
            if callback is not None: callback(rows[-1])
            if not valid or float(next_beta.numpy()) <= float(beta.numpy()):
                return {"complete":False,"reason":"invalid_weights_or_no_temperature_progress",
                        "particles":x,"log_weights":lw,"stages":rows}
            beta=next_beta
            if float(beta.numpy()) == 1.:
                return {"complete":True,"particles":x,"log_weights":lw,"roots":roots,
                        "log_normalizer":log_z,"stages":rows}
        return {"complete":False,"reason":"temperature_stage_cap","particles":x,
                "log_weights":lw,"stages":rows}


def discover_modes(target, starts, *, max_iterations, score_tolerance, merge_distance,
                   curvature_tolerance, jit_compile=False):
    """JAMS-style preparation; optimizer graph is a documented CPU reference lane."""
    def objective(x):
        v,g,_=target.value_score(x)
        return -v,-g
    program=tf.function(lambda x: tfp.optimizer.lbfgs_minimize(objective,x,
        tolerance=score_tolerance,max_iterations=max_iterations,parallel_iterations=1),
        input_signature=[tf.TensorSpec(starts.shape,F64)],jit_compile=jit_compile)
    result=program(starts)
    points=result.position
    values,scores,valid=target.value_score(points)
    hessians=-target.hessian(points)
    hessians=(hessians+tf.linalg.matrix_transpose(hessians))/2
    eigenvalues=tf.linalg.eigvalsh(hessians)
    stationary=tf.reduce_max(tf.abs(scores),axis=1)<=score_tolerance*10
    positive=tf.reduce_min(eigenvalues,axis=1)>curvature_tolerance
    accepted=valid & stationary & positive & ~result.failed
    representatives=[]
    records=[]
    for i in range(int(starts.shape[0])):
        group=None
        if bool(accepted[i].numpy()):
            for j in representatives:
                delta=points[i]-points[j]
                dist=tf.einsum('i,ij,j->',delta,(hessians[i]+hessians[j])/2,delta)
                if bool((dist < merge_distance).numpy()):
                    group=j
                    break
            if group is None:
                representatives.append(i)
                group=i
        records.append({"start":starts[i].numpy().tolist(),"endpoint":points[i].numpy().tolist(),
            "log_target":float(values[i].numpy()),"score_max":float(tf.reduce_max(tf.abs(scores[i])).numpy()),
            "hessian_eigenvalues":eigenvalues[i].numpy().tolist(),"accepted_mode":bool(accepted[i].numpy()),
            "optimizer_failed":bool(result.failed[i].numpy()),"representative_index":group})
    return {"representatives":tf.gather(points,representatives),"records":records,
            "evaluations":int(result.num_objective_evaluations.numpy()),
            "mode_count":len(representatives),"exhaustive_discovery":False}
