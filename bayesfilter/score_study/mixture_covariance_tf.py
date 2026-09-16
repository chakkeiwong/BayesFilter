"""Persistent Gaussian-mixture assumed-density covariance provider for LEDH.

This is a local KDM covariance candidate, not a reproduction of Younis' learned
filter. Components are conditioned and reweighted at every observation. Their
weighted within/between moments drive the existing LEDH flow. LEDH's reset
continues to act on its particle cloud; this independent filter retains its own
component identities. All initial, weight and covariance tangents are total.
"""
from functools import lru_cache
import tensorflow as tf
from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
    quadrature_predict_with_parameter_tangent, quadrature_update_with_parameter_tangent)
from bayesfilter.highdim.ledh_canonical_score_tf import _value_and_analytical_score_impl
from .gaussian_tf import parameterized_model, chol_tangent
from .canonical_adapter_tf import gaussian_direction_inputs, make_canonical_kernel


def mixture_moments(means, covariances, dmeans, dcovariances, logits, dlogits):
    weights = tf.nn.softmax(logits)
    dweights = weights * (dlogits - tf.reduce_sum(weights*dlogits))
    mean = tf.einsum("n,ni->i",weights,means)
    dmean = tf.einsum("n,ni->i",dweights,means)+tf.einsum("n,ni->i",weights,dmeans)
    centered,dcentered = means-mean,dmeans-dmean
    spread = covariances+tf.einsum("ni,nj->nij",centered,centered)
    dspread = dcovariances+tf.einsum("ni,nj->nij",dcentered,centered)+tf.einsum("ni,nj->nij",centered,dcentered)
    covariance = tf.einsum("n,nij->ij",weights,spread)
    dcovariance = tf.einsum("n,nij->ij",dweights,spread)+tf.einsum("n,nij->ij",weights,dspread)
    return mean,covariance,dmean,dcovariance


def mixture_schedule(model,theta,observations,means,covariances,dmeans,dcovariances):
    """Fixed-count Gaussian mixture: predict, condition, reweight, persist."""
    dtype = theta.dtype
    count,dimension = means.shape
    horizon = observations.shape[0]
    logits = tf.fill([count],-tf.math.log(tf.cast(count,dtype)))
    dlogits = tf.zeros_like(logits)
    arrays = tuple(tf.TensorArray(dtype,size=horizon) for _ in range(14))
    def step(t,m,P,dm,dP,l,dl,*buffers):
        pred = quadrature_predict_with_parameter_tangent(m,P,dm,dP,
            lambda x:model.transition_mean_fn(theta,x),
            lambda x,dx:model.transition_mean_tangent_fn(theta,x,dx),model.process_covariance,
            d_process_noise_covariance=model.process_covariance_tangent_fn(theta),jitter=0.)
        aggregate_prediction = mixture_moments(*pred,l,dl)
        update = quadrature_update_with_parameter_tangent(*pred,model.observation_fn,
            model.observation_tangent_fn,model.observation_covariance,observations[t],
            d_observation_covariance=model.observation_covariance_tangent_fn(theta),
            jitter=0.,return_evidence=True)
        m,P,dm,dP,logg,dlogg = update
        eigenvalues = tf.linalg.eigvalsh(P)
        eps = tf.cast(2**-23 if dtype == tf.float32 else 2**-52,dtype)
        margin = eps*tf.cast(dimension,dtype)*tf.reduce_max(tf.abs(eigenvalues),-1)
        valid = tf.reduce_all(tf.reduce_min(eigenvalues,-1)>margin)
        valid &= tf.reduce_all(tf.math.is_finite(eigenvalues))
        m,P,dm,dP,logg,dlogg = (tf.where(valid,x,tf.cast(float("nan"),dtype))
                               for x in (m,P,dm,dP,logg,dlogg))
        raw,draw = l+logg,dl+dlogg
        normalization = tf.reduce_logsumexp(raw)
        weights = tf.nn.softmax(raw)
        l,dl = raw-normalization,draw-tf.reduce_sum(weights*draw)
        aggregate_update = mixture_moments(m,P,dm,dP,l,dl)
        values = (*aggregate_prediction,*aggregate_update,m,P,dm,dP,l,dl)
        return (t+1,m,P,dm,dP,l,dl,*(a.write(t,v) for a,v in zip(buffers,values)))
    result = tf.while_loop(lambda t,*_:t<horizon,step,
        (0,means,covariances,dmeans,dcovariances,logits,dlogits,*arrays),parallel_iterations=1)
    names = [prefix+name for prefix in ("predicted_","post_")
             for name in ("means","covariances","d_means","d_covariances")]
    names += ["component_means","component_covariances","d_component_means",
              "d_component_covariances","component_log_weights","d_component_log_weights"]
    return dict(zip(names,[a.stack() for a in result[7:]]))


def gaussian_mixture_initial(theta,direction,d,o,within_fraction):
    """Positive 2d-point mixture matching the Gaussian initial first two moments."""
    _,_,_,_,m,dm,P,dP,*_ = parameterized_model(theta,d,o)
    dm,dP = tf.tensordot(direction,dm,1),tf.tensordot(direction,dP,1)
    L = tf.linalg.cholesky(P); dL = chol_tangent(L,dP)
    design = tf.concat([tf.eye(d,dtype=theta.dtype),-tf.eye(d,dtype=theta.dtype)],0)
    design *= tf.sqrt(tf.cast(d*(1-within_fraction),theta.dtype))
    means = m+tf.einsum("ij,nj->ni",L,design)
    dmeans = dm+tf.einsum("ij,nj->ni",dL,design)
    return means,tf.broadcast_to(within_fraction*P,[2*d,d,d]),dmeans,tf.broadcast_to(within_fraction*dP,[2*d,d,d])


@lru_cache(maxsize=16)
def make_mixture_covariance_kernel(d,o,N,T,controls_tuple,within_fraction,dtype_name="float64",jit_compile=True):
    if not 0 < within_fraction <= 1: raise ValueError("within fraction must be in (0,1]")
    make_canonical_kernel(d,o,N,T,controls_tuple,dtype_name,jit_compile)
    dtype = tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([6],dtype),
        tf.TensorSpec([T,o],dtype),tf.TensorSpec([N,d],dtype),tf.TensorSpec([T,N,d],dtype),
        tf.TensorSpec([N,d],dtype)],jit_compile=jit_compile)
    def kernel(theta,direction,observations,initial_noise,noise,design):
        model,initial,dinitial,P,dP,*_ = gaussian_direction_inputs(theta,direction,initial_noise,d,o)
        schedule = mixture_schedule(model,theta,observations,
            *gaussian_mixture_initial(theta,direction,d,o,within_fraction))
        value,score,trace = _value_and_analytical_score_impl(model,theta,initial,
            tf.broadcast_to(P,[N,d,d]),noise,observations,with_score=True,return_trace=True,
            initial_state_tangent=dinitial,initial_covariance_tangent=tf.broadcast_to(dP,[N,d,d]),
            reset_design=design,moment_schedule=schedule,**dict(controls_tuple))
        return value,score[0],schedule,tuple(x["predicted_covariances"] for x in trace)
    return kernel
