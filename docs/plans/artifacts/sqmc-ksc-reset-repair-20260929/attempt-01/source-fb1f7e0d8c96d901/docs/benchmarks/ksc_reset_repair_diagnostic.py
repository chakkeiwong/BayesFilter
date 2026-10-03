"""Stage reductions of the full analytical executor; diagnostic only."""
import tensorflow as tf
from ksc_reset_mechanism_diagnostic import predictive_functional

def trace_kernel(spec, route, controls, n, horizon, design_kind, *, stages):
    from bayesfilter.highdim.sqmc_campaign_tf import numerical_settings,route_settings,reset_design
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    settings=dict(numerical_settings(controls),**route_settings(route))
    design=reset_design(n,spec.dimension,tf.float64,design_kind)
    signature=[tf.TensorSpec([2],tf.float64),tf.TensorSpec([2],tf.float64),
               tf.TensorSpec([n,1],tf.float64),tf.TensorSpec([horizon,n,1],tf.float64),
               tf.TensorSpec([horizon,n],tf.float64),tf.TensorSpec([horizon,1],tf.float64)]
    @tf.function(input_signature=signature,jit_compile=True,autograph=False)
    def compute(theta,direction,initial,noise,uniforms,obs):
        model,_=spec.model(theta,direction)
        states,covs,ds,dc=spec.initial_cloud(theta,initial,direction)
        value,score,trace=canonical_value_and_analytical_score(model,theta,states,covs,noise,obs,
            with_score=True,return_trace=True,initial_state_tangent=ds,initial_covariance_tangent=dc,
            reset_design=design,process_ancestor_uniforms=uniforms,**settings)
        def stack(key): return tf.stack([row[key] for row in trace])
        if not stages: return value,score[0],stack('ancestor_indices')
        result=dict(value=value,score=score[0],valid=trace[-1]['program_valid'])
        clouds={'source':(stack('children')[...,0],stack('d_children')[...,0],
                          stack('posterior_logits'),stack('d_posterior_logits')),
                'final':(stack('states_after_reset')[...,0],stack('d_states_after_reset')[...,0],
                         stack('outgoing_log_weights'),stack('d_outgoing_log_weights'))}
        for stage in ('raw_reset','pre_cap','post_cap'):
            key='higher_moment_stage_'+stage
            clouds[stage]=(stack(key)[...,0],stack(key+'_tangent')[...,0,0],
                            stack('outgoing_log_weights'),stack('d_outgoing_log_weights'))
        for stage,(x,dx,logw,dlogw) in clouds.items():
            w=tf.nn.softmax(logw,axis=-1)
            mean=tf.reduce_sum(w*x,axis=-1)
            centered=x-mean[:,None]
            var=tf.reduce_sum(w*centered**2,axis=-1)
            skew=tf.reduce_sum(w*centered**3,axis=-1)/var**1.5
            kurt=tf.reduce_sum(w*centered**4,axis=-1)/var**2
            result[stage+'_moments']=tf.stack([mean,var,skew,kurt],axis=-1)
            result[stage+'_max_standardized']=tf.reduce_max(tf.abs(centered)/tf.sqrt(var[:,None]),axis=-1)
            result[stage+'_within_parity_variance_ratio']=(tf.math.reduce_variance(x[:,::2],axis=-1)+tf.math.reduce_variance(x[:,1::2],axis=-1))/(2.*var)
            pv,ps=predictive_functional(theta,direction,x[:-1],dx[:-1],logw[:-1],dlogw[:-1],obs[1:,0])
            result[stage+'_next_log_prediction']=pv
            result[stage+'_next_predictive_score']=ps
        children=clouds['source'][0]
        # Mean of all squared pair distances is twice the uniform variance.
        result['transport_mean_cost']=2.*tf.math.reduce_variance(children,axis=-1)
        result['transport_cost_scale']=tf.maximum(result['transport_mean_cost'],tf.cast(1e-3,tf.float64))
        result['transport_effective_epsilon']=result['transport_cost_scale']*controls['reset_epsilon']
        for key in ('fraction_coordinatewise_cap_active','minimum_coordinatewise_cap_derivative',
                    'maximum_coordinatewise_pre_cap_absolute','maximum_coordinatewise_post_cap_absolute',
                    'maximum_diagonal_scaled_system_condition','maximum_diagonal_pre_cap_particle_rms',
                    'maximum_diagonal_post_cap_particle_rms','skew_residual','kurtosis_residual'):
            result[key]=stack('higher_moment_'+key)
        return result
    return compute
