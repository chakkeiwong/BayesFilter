"""Default-XLA C2 frozen-branch preparation."""
from types import SimpleNamespace
import tensorflow as tf
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import _evaluate_core
from bayesfilter.ops.stateless_random_tf import philox_normal_float64, philox_uniform_float64

D=tf.float64


def make_branch_preparation(model,horizon,count,sample_step,operand_specs,diagnostic_specs,
                            jit_compile=True):
    """Enclose compatible sampling and original exact-prefix weight feedback."""
    dimension=model.state_dim()
    @tf.function(input_signature=(tf.TensorSpec([horizon,dimension],D),
        tf.TensorSpec([2],D),tf.TensorSpec([],tf.int64),operand_specs),
        jit_compile=jit_compile,autograph=False)
    def prepare(observed,theta,seed,operands):
        def words(offset):
            return tf.stack([seed,tf.cast(offset,tf.int64)])
        covariance,_=model.stationary_covariance_and_derivative(theta)
        chol=tf.linalg.cholesky(covariance)
        initial=tf.einsum('ij,nj->ni',chol,philox_normal_float64([count,dimension],words(1001)))
        initial_log_q=model.initial_log_density(theta,initial)
        states=tf.tensor_scatter_nd_update(tf.zeros([horizon,count,dimension],D),[[0]],initial[None])
        ancestors=tf.zeros([horizon-1,count],tf.int32)
        auxiliary=tf.zeros([horizon-1,count],D)
        log_q=tf.zeros([horizon-1,count],D)
        initial_mass=tf.fill([count],-tf.math.log(tf.cast(count,D)))
        transition_mass=tf.fill([horizon-1,count],-tf.math.log(tf.cast(count,D)))

        def prefix(time,states,ancestors,auxiliary,log_q):
            branch=SimpleNamespace(dtype=D,particle_count=count,time_steps=horizon,
                observations=observed,states=states,initial_log_proposal_density=initial_log_q,
                ancestors=ancestors,auxiliary_log_probabilities=auxiliary,
                transition_log_proposal_density=log_q,initial_log_base_mass=initial_mass,
                transition_log_base_mass=transition_mass)
            return _evaluate_core(model,branch,theta,evaluation_steps=time)['final_log_weights']
        weights=prefix(tf.constant(1),states,ancestors,auxiliary,log_q)
        status=tf.where(tf.reduce_all(tf.math.is_finite(observed[0])),0,1)
        status=tf.where((status==0)&~tf.reduce_all(tf.math.is_finite(initial)),2,status)
        status=tf.where((status==0)&~tf.reduce_all(tf.math.is_finite(initial_log_q)),3,status)
        diagnostics=tf.nest.map_structure(lambda spec:tf.zeros([horizon-1,*spec.shape],spec.dtype),diagnostic_specs)

        # A one-observation branch has no transition buffers to update.
        # TensorFlow rejects even tracing a scatter into their empty extent.
        if horizon == 1:
            return {'states':states,'ancestors':ancestors,'auxiliary_log_probabilities':auxiliary,
                    'transition_log_proposal_density':log_q,'initial_log_proposal_density':initial_log_q,
                    'diagnostics':diagnostics,'status':status,'failed_time':tf.constant(0)}

        def step(time,states,ancestors,auxiliary,log_q,weights,diagnostics,status,failed_time):
            auxiliary_row=tf.identity(weights)
            uniforms=philox_uniform_float64([count],words(2000+17*time))
            cdf=tf.math.cumsum(tf.exp(auxiliary_row))
            cdf=tf.concat([cdf[:-1],tf.ones([1],D)],0)
            ancestor=tf.searchsorted(cdf,uniforms,side='right',out_type=tf.int32)
            parents=tf.gather(states[time-1],ancestor)
            points,density,row=sample_step(time,parents,words(3000+31*time),theta,operands)
            points=tf.ensure_shape(points,[count,dimension])
            density=tf.ensure_shape(density,[count])
            status=tf.where(tf.reduce_all(tf.math.is_finite(points))&tf.reduce_all(tf.math.is_finite(density)),0,4)
            if 'time_index_valid' in row:
                status=tf.where(~row['time_index_valid'],9,status)
            if 'cdf_bracket_valid' in row:
                status=tf.where((status==0)&~row['cdf_bracket_valid'],5,status)
            if 'finite' in row:
                status=tf.where((status==0)&~row['finite'],6,status)
            status=tf.where((status==0)&~tf.reduce_all(tf.math.is_finite(observed[time])),1,status)
            status=tf.where((status==0)&~tf.reduce_all(tf.math.is_finite(auxiliary_row)),7,status)
            status=tf.where((status==0)&(tf.abs(tf.reduce_logsumexp(auxiliary_row))>1e-10),8,status)
            states=tf.tensor_scatter_nd_update(states,tf.reshape(time,[1,1]),points[None])
            index=tf.reshape(time-1,[1,1])
            ancestors=tf.tensor_scatter_nd_update(ancestors,index,ancestor[None])
            auxiliary=tf.tensor_scatter_nd_update(auxiliary,index,auxiliary_row[None])
            log_q=tf.tensor_scatter_nd_update(log_q,index,density[None])
            diagnostics=tf.nest.map_structure(lambda history,value:tf.tensor_scatter_nd_update(history,index,value[None]),diagnostics,row)
            weights=prefix(time+1,states,ancestors,auxiliary,log_q)
            return time+1,states,ancestors,auxiliary,log_q,weights,diagnostics,status,tf.where(status==0,failed_time,time)

        result=tf.while_loop(lambda time,*args:(time<horizon)&(args[-2]==0),step,
            (tf.constant(1),states,ancestors,auxiliary,log_q,weights,diagnostics,status,tf.constant(0)),
            maximum_iterations=horizon-1,parallel_iterations=1)
        return {'states':result[1],'ancestors':result[2],'auxiliary_log_probabilities':result[3],
                'transition_log_proposal_density':result[4],'initial_log_proposal_density':initial_log_q,
                'diagnostics':result[6],'status':result[7],'failed_time':result[8]}
    return prepare


def bootstrap_step(model,count):
    dimension=model.state_dim()
    def sample(time,parents,seed,theta,operands):
        del time,operands
        transition=model.transition_matrix(theta)
        sigma=tf.constant(float(model.sigma),D)
        normal=philox_normal_float64([count,dimension],seed)
        states=tf.linalg.matmul(parents,transition,transpose_b=True)+sigma*normal
        log_q=model.transition_log_density(theta,parents,states,1)
        return states,log_q,{'finite':tf.reduce_all(tf.math.is_finite(states))}
    return sample


def gaussian_step(count, dimension):
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import _gaussian_transform_core

    def sample(time, parents, seed, theta, operands):
        del parents, theta
        means, cholesky, indices = operands
        normal = philox_normal_float64([count, dimension], seed)
        result = _gaussian_transform_core(normal, means[time-1], cholesky[time-1])
        return result["physical_points"], result["physical_log_density"], {
            "finite": result["finite"], "time_index_valid": indices[time-1] == time}
    return sample
