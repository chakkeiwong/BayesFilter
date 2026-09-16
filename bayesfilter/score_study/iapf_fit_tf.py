"""Bounded diagonal density-scale fit of GJL (2017), equation (15).

The bounds and projected local optimizer are explicit adaptations. This is
not an unrestricted/global argmin. The objective remains density least
squares; normalization below is constant with respect to fit parameters.
"""
from functools import lru_cache
import math
import tensorflow as tf
from .gaussian_tf import parameterized_model, gaussian_log_density_and_tangent
from .fitted_twist_tf import normalizer
from .conditional_means_tf import conditional_mean, validate_curves


def profile_density_loss(z, target, parameters):
    """Profile scale and return loss, analytical gradient and diagnostics.

    z is cloud-standardized; target has fixed maximum one. The density omits
    its fixed (2*pi)^(-d/2)/prod(cloud_sd) factor, preserving the minimizers.
    """
    loss,gradient,log_scale,shape_error,_=_density_profile(z,tf.math.log(target),parameters)
    return loss,gradient,tf.exp(log_scale),shape_error


def _density_profile(z, log_target, parameters):
    """Same density objective, with scale/shape evaluated before underflow.

    All factors of the parameter-dependent scaling cancel analytically.
    The loss and its gradient retain exp(2*log_amplitude); dropping that
    factor would change the objective and is not done here.
    """
    d=z.shape[-1]
    mean,log_sd=parameters[:d],parameters[d:]
    standardized=(z-mean)*tf.exp(-log_sd)
    log_density=-tf.reduce_sum(log_sd)-.5*tf.reduce_sum(standardized**2,axis=1)
    log_amplitude=tf.reduce_max(log_density)
    log_relative_density=log_density-log_amplitude
    density=tf.exp(log_relative_density)
    target=tf.exp(log_target)
    log_relative_scale=(tf.reduce_logsumexp(log_relative_density+log_target)-
                        tf.reduce_logsumexp(2*log_target))
    scale=tf.exp(log_relative_scale)
    residual=density-scale*target
    squared_amplitude=tf.exp(2*log_amplitude)
    loss=squared_amplitude*tf.reduce_mean(residual**2)
    mean_gradient=2*squared_amplitude*tf.reduce_mean((residual*density)[:,None]*standardized*tf.exp(-log_sd),axis=0)
    log_sd_gradient=2*squared_amplitude*tf.reduce_mean((residual*density)[:,None]*(standardized**2-1),axis=0)
    shape_error=tf.reduce_sum(residual**2)/tf.reduce_sum(density**2)
    return loss,tf.concat([mean_gradient,log_sd_gradient],0),log_amplitude+log_relative_scale,shape_error,log_amplitude


def bounded_density_fit(points, log_targets, *, mean_bound, sd_lower, sd_upper,
                        max_steps, max_backtracks, tolerance, floor_ratio):
    """Projected backtracking fit with explicit finite/convergence status."""
    dtype=points.dtype; d=points.shape[-1]
    cloud_mean=tf.reduce_mean(points,axis=0)
    cloud_sd=tf.sqrt(tf.reduce_mean((points-cloud_mean)**2,axis=0))
    log_target_max=tf.reduce_max(log_targets)
    log_target=log_targets-log_target_max
    z=(points-cloud_mean)/cloud_sd
    lower=tf.concat([tf.fill([d],tf.cast(-mean_bound,dtype)),
                     tf.fill([d],tf.cast(math.log(sd_lower),dtype))],0)
    upper=tf.concat([tf.fill([d],tf.cast(mean_bound,dtype)),
                     tf.fill([d],tf.cast(math.log(sd_upper),dtype))],0)
    parameters=tf.clip_by_value(tf.zeros([2*d],dtype),lower,upper)
    finite_input=(tf.reduce_all(tf.math.is_finite(z)) &
                  tf.reduce_all(tf.math.is_finite(log_targets)) &
                  tf.reduce_all(cloud_sd>0))
    initial_loss,initial_gradient,_,_,_=_density_profile(z,log_target,parameters)
    tol=tf.cast(tolerance,dtype)
    # In an active iteration some |gradient_j| exceeds tolerance. This
    # geometry bound lets that component traverse the full parameter box;
    # an absolute cap (formerly 64) can stall when density amplitudes shrink.
    step_limit=tf.reduce_max(upper-lower)/tol

    def projected_norm(par,gradient):
        return tf.reduce_max(tf.abs(par-tf.clip_by_value(par-gradient,lower,upper)))

    def body(iteration,par,loss,gradient,step,healthy):
        def search(j,step,trial,trial_loss,accepted):
            trial=tf.clip_by_value(par-step*gradient,lower,upper)
            trial_loss=_density_profile(z,log_target,trial)[0]
            # The projected displacement, not -step*||g||^2, enters Armijo.
            threshold=loss+tf.cast(1e-4,dtype)*tf.reduce_sum(gradient*(trial-par))
            accepted=tf.math.is_finite(trial_loss) & (trial_loss<=threshold)
            return j+1,tf.where(accepted,step,step*.5),trial,trial_loss,accepted
        _,used_step,trial,trial_loss,accepted=tf.while_loop(
            lambda j,step,trial,trial_loss,accepted:(j<max_backtracks)&~accepted,
            search,(0,step,par,loss,tf.constant(False)),parallel_iterations=1)
        par=tf.where(accepted,trial,par)
        loss,gradient,_,_,_=_density_profile(z,log_target,par)
        healthy=healthy & accepted & tf.reduce_all(tf.math.is_finite(gradient))
        return iteration+1,par,loss,gradient,tf.minimum(used_step*2,step_limit),healthy

    iteration,parameters,loss,gradient,_,healthy=tf.while_loop(
        lambda iteration,par,loss,gradient,step,healthy:
            (iteration<max_steps)&healthy&(projected_norm(par,gradient)>tol),
        body,(0,parameters,initial_loss,initial_gradient,tf.cast(1.,dtype),finite_input),
        parallel_iterations=1)
    loss,gradient,log_scale,shape_error,log_amplitude=_density_profile(z,log_target,parameters)
    projected_gradient=projected_norm(parameters,gradient)
    converged=healthy & (projected_gradient<=tol)
    center=cloud_mean+cloud_sd*parameters[:d]
    sd=cloud_sd*tf.exp(parameters[d:])
    covariance=tf.linalg.diag(sd**2)
    log_constant=-.5*tf.cast(d,dtype)*tf.math.log(2*tf.acos(tf.cast(-1.,dtype)))-tf.reduce_sum(tf.math.log(cloud_sd))
    log_lambda=log_scale+log_constant-log_target_max
    log_floor=tf.math.log(tf.cast(floor_ratio,dtype))+log_constant-tf.reduce_sum(parameters[d:])
    eps=tf.cast(2**-23 if dtype==tf.float32 else 2**-52,dtype)
    boundary=tf.reduce_any(tf.minimum(parameters-lower,upper-parameters)<=32*eps*(1+tf.abs(parameters)))
    valid=(healthy & tf.math.is_finite(loss) & tf.math.is_finite(shape_error) &
           tf.math.is_finite(log_lambda) & tf.reduce_all(tf.math.is_finite(sd)) & tf.reduce_all(sd>0))
    return center,covariance,log_floor,{
        "valid":valid,"converged":converged,"scaled_loss":loss,
        "normalized_shape_residual":shape_error,"log_lambda":log_lambda,
        "boundary_active":boundary,"iterations":iteration,
        "projected_gradient":projected_gradient,"cloud_sd":cloud_sd,
        "fitted_sd":sd,"density_log_constant":log_constant,
        "density_log_amplitude":log_amplitude,
        "objective_underflow":(loss==0)&(shape_error>0)}


@lru_cache(maxsize=16)
def make_density_recursive_fit_kernel(d,o,N,T,mean_bound,sd_lower,sd_upper,
                                      max_steps,max_backtracks,tolerance,floor_ratio,
                                      dtype_name="float64",jit_compile=True,
                                      transition_curve=0.,observation_curve=0.):
    validate_curves(d,o,transition_curve,observation_curve)
    values=(mean_bound,sd_lower,sd_upper,tolerance,floor_ratio)
    if not all(math.isfinite(x) and x>0 for x in values) or not sd_lower<1<sd_upper:
        raise ValueError("positive finite controls and sd_lower < 1 < sd_upper required")
    if not 1<=max_steps<=10000 or not 1<=max_backtracks<=50 or N<=d:
        raise ValueError("invalid density-fit iteration/point budget")
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,o],dtype),
                                  tf.TensorSpec([T,N,d],dtype)],jit_compile=jit_compile)
    def fit(theta,observations,clouds):
        A,_,H,_,_,_,_,_,Q,_,R,_=parameterized_model(theta,d,o)
        centers=tf.TensorArray(dtype,size=T);covariances=tf.TensorArray(dtype,size=T)
        floors=tf.TensorArray(dtype,size=T);diagnostics=tf.TensorArray(dtype,size=T,element_shape=[10])
        def step(t,center,V,log_floor,valid,converged,centers,covariances,floors,diagnostics):
            x=clouds[t]
            lg,_=gaussian_log_density_and_tangent(observations[t]-conditional_mean(x,H,observation_curve,quadratic=True),
                tf.zeros([6,N,o],dtype),R,tf.zeros([6,1,o,o],dtype))
            lf=tf.cond(t+1<T,lambda:normalizer(conditional_mean(x,A,transition_curve),tf.zeros([6,N,d],dtype),
                Q,tf.zeros([6,d,d],dtype),center,V,log_floor)[0],lambda:tf.zeros([N],dtype))
            center,V,log_floor,info=bounded_density_fit(x,lg+lf,mean_bound=mean_bound,
                sd_lower=sd_lower,sd_upper=sd_upper,max_steps=max_steps,
                max_backtracks=max_backtracks,tolerance=tolerance,floor_ratio=floor_ratio)
            summary=tf.stack([info["scaled_loss"],info["normalized_shape_residual"],info["log_lambda"],
                tf.cast(info["boundary_active"],dtype),tf.cast(info["iterations"],dtype),
                info["projected_gradient"],tf.reduce_min(info["fitted_sd"]),tf.reduce_max(info["fitted_sd"]),
                info["density_log_amplitude"],tf.cast(info["objective_underflow"],dtype)])
            return (t-1,center,V,log_floor,valid&info["valid"],converged&info["converged"],
                    centers.write(t,center),covariances.write(t,V),floors.write(t,log_floor),diagnostics.write(t,summary))
        out=tf.while_loop(lambda t,*_:t>=0,step,(T-1,tf.zeros([d],dtype),tf.eye(d,dtype=dtype),
            tf.zeros([],dtype),tf.constant(True),tf.constant(True),centers,covariances,floors,diagnostics),parallel_iterations=1)
        return out[6].stack(),out[7].stack(),out[8].stack(),out[4],out[5],out[9].stack()
    return fit
