"""FP64 source-reference Lorenz refinement controller.

Independent repetitions share batch-native numerical kernels and retain
individual stopping decisions. All time, RK4, fitting and refinement loops
are native TensorFlow graphs. This is not a NeuTra training route or a new
production precision default.
"""
import math
import tensorflow as tf
from bayesfilter.replication.multivariate_controlled_smc_tf import (
    MultivariateGaussianControlledSMC, lorenz_rk4)

D = tf.float64
SOURCE_APF_COUNTS = {1: 1382, 2: 2867, 3: 4505, 4: 6144}


def apf_particles_for_refinements(iterations):
    if iterations not in SOURCE_APF_COUNTS:
        raise ValueError("Source APF count requires one through four refinements")
    return SOURCE_APF_COUNTS[iterations]


def needs_refinement(iteration, maximum, ess, particles):
    """Scalar boundary authority retained for callers and source checks."""
    return (iteration < maximum) & ((iteration == 0) |
        (tf.reduce_min(ess)*tf.constant(100., D)/tf.cast(particles,D) <= 90.))


def active_refinements(iterations, maximum, ess, particles):
    """One stop decision per repetition; ess has shape [time, repetition]."""
    return (iterations < maximum) & ((iterations == 0) |
        (tf.reduce_min(ess, axis=0)*tf.constant(100., D)/tf.cast(particles,D) <= 90.))


class LorenzRefinement:
    """Controlled-SMC implementation with an optional diagnostic guard candidate.

    ``first_failure_guards=True`` is experimental: full healthy coefficient
    parity failed in attempts132/133. Local guard tests do not admit this
    candidate as an equivalent numerical or production route.
    """
    def __init__(self, *, forcing, particles=512, observations=101,
                 replicas=1, jit_compile=True, precision_policy="none",precision_floor=None,precision_eigen_tolerance=None,precision_eigen_refinement=False,precision_stable_integral=False,precision_activation_margin=None,first_failure_guards=False):
        self.h, self.n, self.d, self.p, self.b = observations, particles, 8, 6, replicas
        self.r = 1e-4
        self.forcing = tf.constant(forcing,D)
        self.engine = MultivariateGaussianControlledSMC(
            observations=observations,particles=particles,replicas=replicas,
            dimension=self.d,observation_dimension=self.p,initial_mean=[0.]*self.d,
            initial_covariance=tf.eye(self.d,dtype=D)*.01,
            transition_covariance=tf.eye(self.d,dtype=D)*.001,
            transition_mean=lambda x:lorenz_rk4(x,self.forcing),
            observation_log_density=lambda y,x:-.5/self.r*tf.reduce_sum((y-x[...,:self.p])**2,-1),
            solver="centered_qr",jit_compile=jit_compile,precision_policy=precision_policy,
            precision_floor=precision_floor,precision_eigen_tolerance=precision_eigen_tolerance,precision_eigen_refinement=precision_eigen_refinement,precision_stable_integral=precision_stable_integral,precision_activation_margin=precision_activation_margin,first_failure_guards=first_failure_guards)
        self.run = tf.function(self._run_guarded if first_failure_guards else self._run,input_signature=[
            tf.TensorSpec([observations,replicas,self.p],D),
            tf.TensorSpec([5,observations,replicas,particles,self.d],D),
            tf.TensorSpec([5,observations,replicas],D),tf.TensorSpec([],tf.int32)],
            jit_compile=jit_compile,autograph=False)

    def initial_policy(self, y):
        a = tf.broadcast_to(tf.linalg.diag(tf.constant([.5/self.r]*self.p+
                [0.]*(self.d-self.p),D)),[self.h,self.b,self.d,self.d])
        b = -tf.pad(y,[[0,0],[0,0],[0,self.d-self.p]])/self.r
        c = .5/self.r*tf.reduce_sum(y*y,-1)
        return a,b,c

    def _run_guarded(self, y, normal, uniform, maximum):
        policy = self.initial_policy(y)
        current = self.engine.forward(*policy,y,normal[0],uniform[0])
        valid = current["valid"] & (maximum>=0) & (maximum<=4)
        names = ("x","ancestors","log_incremental_weight","logZ","ess","precision_margin")
        fields = dict(A=policy[0],b=policy[1],c=policy[2],**{k:current[k] for k in names})
        history = {k:tf.repeat(v[None],5,axis=0) for k,v in fields.items()}
        gram, rss = tf.ones([5,self.b],D), tf.zeros([5,self.b],D)
        counts = tf.zeros([self.b],tf.int32)
        safety=dict(minimum_qr_rcond=tf.ones([5,self.b],D),protection_count=tf.zeros([5,self.b],D),
            minimum_fraction=tf.ones([5,self.b],D),
            minimum_raw_margin=tf.ones([5,self.b],D),
            maximum_relative_change=tf.zeros([5,self.b],D),
            maximum_eigen_residual=tf.zeros([5,self.b],D),maximum_eigen_orthogonality=tf.zeros([5,self.b],D),maximum_eigen_rotations=tf.zeros([5,self.b],D))

        empty_fit=self.engine._empty_fit(*policy)
        fit_history={k:tf.repeat(v[None],4,axis=0) for k,v in empty_fit.items()}
        attempted=tf.zeros([4,self.b],tf.bool)
        accepted=tf.zeros([4,self.b],tf.bool)
        failure=dict(refit=tf.where(valid,-1,0),time=current['first_failure_time'],
            codes=current['failure_codes'],mean=current['failure_mean'],covariance=current['failure_covariance'],
            proposed_A=policy[0],proposed_b=policy[1],proposed_c=policy[2])

        def select(active, new, old):
            # The repetition axis is1 for all policy and forward arrays.
            shape=[1,self.b]+[1]*(new.shape.rank-2)
            return tf.where(tf.reshape(active,shape),new,old)

        def body(i,counts,policy,current,valid,history,gram,rss,safety,fit_history,attempted,accepted,failure):
            active = active_refinements(counts,maximum,current["ess"],self.n)
            # Inactive work slots duplicate an active system, so discarded
            # work cannot cause the shared fail-closed validity guard to fire.
            first = tf.argmax(tf.cast(active,tf.int32),output_type=tf.int32)
            work_indices = tf.where(active,tf.range(self.b),first)
            work = lambda value:tf.gather(value,work_indices,axis=1)
            fitted = self.engine.fit(*(work(p) for p in policy),
                work(current["x"]),work(current["log_incremental_weight"]))
            proposed = tuple(fitted[k] for k in ("A","b","c"))
            forward = tf.cond(fitted["valid"],
                lambda:self.engine.forward(*proposed,work(y),work(normal[i+1]),work(uniform[i+1])),
                lambda:current)
            good = valid & fitted["valid"] & forward["valid"]
            accepting=active&good
            updated = tuple(select(accepting,new,old) for new,old in zip(proposed,policy))
            updated_current = dict(current)
            updated_current.update({k:select(accepting,forward[k],current[k]) for k in names})
            updated_current["valid"] = good
            fields = dict(A=updated[0],b=updated[1],c=updated[2],
                          **{k:updated_current[k] for k in names})
            index=tf.reshape(i+1,[1,1])
            saved={k:tf.tensor_scatter_nd_update(history[k],index,
                select(active,v,history[k][i+1])[None]) for k,v in fields.items()}
            gram_value=tf.where(active,tf.reduce_min(fitted["gram_eigenvalue_ratio"],axis=0),gram[i+1])
            rss_value=tf.where(active,tf.reduce_max(fitted["residual_sum_squares"],axis=0),rss[i+1])
            gram=tf.tensor_scatter_nd_update(gram,index,gram_value[None])
            rss=tf.tensor_scatter_nd_update(rss,index,rss_value[None])
            diagnostic=dict(minimum_qr_rcond=tf.reduce_min(fitted['qr_reciprocal_condition_lower_bound'],axis=0),
                protection_count=tf.reduce_sum(fitted["protection_applied"],axis=0),
                minimum_fraction=tf.reduce_min(fitted["protection_fraction"],axis=0),
                minimum_raw_margin=tf.reduce_min(fitted["raw_precision_margin"],axis=0),
                maximum_relative_change=tf.reduce_max(fitted["relative_precision_change"],axis=0),
                maximum_eigen_residual=tf.reduce_max(fitted["eigen_residual"],axis=0),
                maximum_eigen_orthogonality=tf.reduce_max(fitted["eigen_orthogonality"],axis=0),
                maximum_eigen_rotations=tf.reduce_max(fitted["eigen_rotations"],axis=0))
            safety={k:tf.tensor_scatter_nd_update(safety[k],index,
                tf.where(active,v,safety[k][i+1])[None]) for k,v in diagnostic.items()}
            fit_history={k:tf.tensor_scatter_nd_update(fit_history[k],tf.reshape(i,[1,1]),v[None])
                         for k,v in fitted.items()}
            attempted=tf.tensor_scatter_nd_update(attempted,tf.reshape(i,[1,1]),active[None])
            accepted=tf.tensor_scatter_nd_update(accepted,tf.reshape(i,[1,1]),accepting[None])
            fit_codes=tf.where(fitted['failure_rows'],fitted['failure_code'],0)
            observed=dict(refit=i+1,time=tf.where(fitted['valid'],forward['first_failure_time'],fitted['first_failure_time']),
                codes=tf.where(fitted['valid'],forward['failure_codes'],fit_codes),
                mean=forward['failure_mean'],covariance=forward['failure_covariance'],
                proposed_A=proposed[0],proposed_b=proposed[1],proposed_c=proposed[2])
            failure={k:tf.where(~good,v,failure[k]) for k,v in observed.items()}
            return (i+1,counts+tf.cast(accepting,tf.int32),updated,updated_current,good,saved,gram,rss,safety,
                    fit_history,attempted,accepted,failure)

        i,counts,policy,current,valid,history,gram,rss,safety,fit_history,attempted,accepted,failure = tf.while_loop(
            lambda i,counts,p,c,v,*_:v & tf.reduce_any(
                active_refinements(counts,maximum,c["ess"],self.n)),
            body,(tf.constant(0),counts,policy,current,valid,history,gram,rss,safety,fit_history,attempted,accepted,failure),
            parallel_iterations=1)
        density_constant = tf.constant(.5*self.p*math.log(2*math.pi*self.r),D)
        normalized = history["logZ"]-density_constant*tf.cast(
            tf.range(1,self.h+1)[None,:,None],D)
        executed=tf.range(5)[:,None]<=counts[None,:]
        return dict(history=history,iterations=counts[0] if self.b==1 else counts,
            fit_history=fit_history,attempted_refits=attempted,accepted_refits=accepted,first_failure=failure,
            valid=valid,executed=executed[:,0] if self.b==1 else executed,
            minimum_fit_gram_ratio=gram[:,0] if self.b==1 else gram,
            maximum_fit_rss=rss[:,0] if self.b==1 else rss,
            normalized_logZ=normalized,safety={k:v[:,0] if self.b==1 else v for k,v in safety.items()})

    def _run(self, y, normal, uniform, maximum):
        policy = self.initial_policy(y)
        current = self.engine.forward(*policy,y,normal[0],uniform[0])
        valid = current["valid"] & (maximum>=0) & (maximum<=4)
        names = ("x","ancestors","log_incremental_weight","logZ","ess","precision_margin")
        fields = dict(A=policy[0],b=policy[1],c=policy[2],**{k:current[k] for k in names})
        history = {k:tf.repeat(v[None],5,axis=0) for k,v in fields.items()}
        gram, rss = tf.ones([5,self.b],D), tf.zeros([5,self.b],D)
        counts = tf.zeros([self.b],tf.int32)
        safety=dict(minimum_qr_rcond=tf.ones([5,self.b],D),protection_count=tf.zeros([5,self.b],D),
            minimum_fraction=tf.ones([5,self.b],D),
            minimum_raw_margin=tf.ones([5,self.b],D),
            maximum_relative_change=tf.zeros([5,self.b],D),
            maximum_eigen_residual=tf.zeros([5,self.b],D),maximum_eigen_orthogonality=tf.zeros([5,self.b],D),maximum_eigen_rotations=tf.zeros([5,self.b],D))

        def select(active, new, old):
            # The repetition axis is1 for all policy and forward arrays.
            shape=[1,self.b]+[1]*(new.shape.rank-2)
            return tf.where(tf.reshape(active,shape),new,old)

        def body(i,counts,policy,current,valid,history,gram,rss,safety):
            active = active_refinements(counts,maximum,current["ess"],self.n)
            # Inactive work slots duplicate an active system, so discarded
            # work cannot cause the shared fail-closed validity guard to fire.
            first = tf.argmax(tf.cast(active,tf.int32),output_type=tf.int32)
            work_indices = tf.where(active,tf.range(self.b),first)
            work = lambda value:tf.gather(value,work_indices,axis=1)
            fitted = self.engine.fit(*(work(p) for p in policy),
                work(current["x"]),work(current["log_incremental_weight"]))
            proposed = tuple(fitted[k] for k in ("A","b","c"))
            forward = tf.cond(fitted["valid"],
                lambda:self.engine.forward(*proposed,work(y),work(normal[i+1]),work(uniform[i+1])),
                lambda:current)
            good = valid & fitted["valid"] & forward["valid"]
            accepting=active&good
            updated = tuple(select(accepting,new,old) for new,old in zip(proposed,policy))
            updated_current = dict(current)
            updated_current.update({k:select(accepting,forward[k],current[k]) for k in names})
            updated_current["valid"] = good
            fields = dict(A=updated[0],b=updated[1],c=updated[2],
                          **{k:updated_current[k] for k in names})
            index=tf.reshape(i+1,[1,1])
            saved={k:tf.tensor_scatter_nd_update(history[k],index,
                select(active,v,history[k][i+1])[None]) for k,v in fields.items()}
            gram_value=tf.where(active,tf.reduce_min(fitted["gram_eigenvalue_ratio"],axis=0),gram[i+1])
            rss_value=tf.where(active,tf.reduce_max(fitted["residual_sum_squares"],axis=0),rss[i+1])
            gram=tf.tensor_scatter_nd_update(gram,index,gram_value[None])
            rss=tf.tensor_scatter_nd_update(rss,index,rss_value[None])
            diagnostic=dict(minimum_qr_rcond=tf.reduce_min(fitted['qr_reciprocal_condition_lower_bound'],axis=0),
                protection_count=tf.reduce_sum(fitted["protection_applied"],axis=0),
                minimum_fraction=tf.reduce_min(fitted["protection_fraction"],axis=0),
                minimum_raw_margin=tf.reduce_min(fitted["raw_precision_margin"],axis=0),
                maximum_relative_change=tf.reduce_max(fitted["relative_precision_change"],axis=0),
                maximum_eigen_residual=tf.reduce_max(fitted["eigen_residual"],axis=0),
                maximum_eigen_orthogonality=tf.reduce_max(fitted["eigen_orthogonality"],axis=0),
                maximum_eigen_rotations=tf.reduce_max(fitted["eigen_rotations"],axis=0))
            safety={k:tf.tensor_scatter_nd_update(safety[k],index,
                tf.where(active,v,safety[k][i+1])[None]) for k,v in diagnostic.items()}
            return i+1,counts+tf.cast(accepting,tf.int32),updated,updated_current,good,saved,gram,rss,safety

        i,counts,policy,current,valid,history,gram,rss,safety = tf.while_loop(
            lambda i,counts,p,c,v,*_:v & tf.reduce_any(
                active_refinements(counts,maximum,c["ess"],self.n)),
            body,(tf.constant(0),counts,policy,current,valid,history,gram,rss,safety),
            parallel_iterations=1)
        density_constant = tf.constant(.5*self.p*math.log(2*math.pi*self.r),D)
        normalized = history["logZ"]-density_constant*tf.cast(
            tf.range(1,self.h+1)[None,:,None],D)
        executed=tf.range(5)[:,None]<=counts[None,:]
        return dict(attempted_iterations=i,rejected_refit=(~valid)&(i>0),history=history,iterations=counts[0] if self.b==1 else counts,
            valid=valid,executed=executed[:,0] if self.b==1 else executed,
            minimum_fit_gram_ratio=gram[:,0] if self.b==1 else gram,
            maximum_fit_rss=rss[:,0] if self.b==1 else rss,
            normalized_logZ=normalized,safety={k:v[:,0] if self.b==1 else v for k,v in safety.items()})
