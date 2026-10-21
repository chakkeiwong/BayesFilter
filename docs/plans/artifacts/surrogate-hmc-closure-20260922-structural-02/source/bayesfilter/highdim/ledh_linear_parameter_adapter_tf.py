"""Analytical stationary linear-Gaussian parameters for the shared LEDH engine.

No likelihood or flow algorithm is reimplemented here. Initial particles use
fixed standard-normal base draws; P0 is also the separate UKF design covariance,
as in ledh_canonical_filter_tf. These two roles have distinct derivatives.
"""
from __future__ import annotations
import tensorflow as tf
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import PerPointScoreModel
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky, all_finite


def stationary_covariance_and_tangent(f,q,d_f,d_q):
    """Solve the discrete Lyapunov equation after a sufficient stability check.

    ||F||2 < 1 implies rho(F) < 1. This is deliberately conservative for
    nonnormal stable matrices; such matrices are rejected, never silently
    accepted from a small equation residual. All planned LG families satisfy
    this certificate throughout their declared prior.
    """
    f=tf.convert_to_tensor(f); dtype=f.dtype; n=int(f.shape[0])
    q=tf.cast(q,dtype); d_f=tf.cast(d_f,dtype); d_q=tf.cast(d_q,dtype)
    finite=all_finite(f,q)
    safe_f=tf.where(finite,f,tf.zeros_like(f))
    gram=tf.matmul(safe_f,safe_f,transpose_a=True)
    norm_squared=tf.reduce_max(tf.linalg.eigvalsh(gram))
    stable=finite & (norm_squared < tf.constant(1.,dtype))
    q_valid,_=safe_cholesky(q,"stationary_process")
    allowed=stable & q_valid
    kron=tf.reshape(tf.einsum("ij,kl->ikjl",safe_f,safe_f),[n*n,n*n])
    operator=tf.eye(n*n,dtype=dtype)-kron
    operator=tf.where(allowed,operator,tf.eye(n*n,dtype=dtype))
    rhs=tf.where(allowed,q,tf.eye(n,dtype=dtype))
    p=tf.reshape(tf.linalg.solve(operator,tf.reshape(rhs,[n*n,1])),[n,n])
    p=(p+tf.transpose(p))*tf.constant(.5,dtype)
    p_valid,_=safe_cholesky(p,"stationary_covariance")
    residual=p-f@p@tf.transpose(f)-q
    residual_scale=tf.maximum(tf.reduce_max(tf.abs(p))+tf.reduce_max(tf.abs(q)),
                              tf.constant(1e-30,dtype))
    relative_residual=tf.reduce_max(tf.abs(residual))/residual_scale
    eps=2.**(-23 if dtype==tf.float32 else -52)
    valid=allowed & p_valid & (relative_residual <= tf.constant(64*n*eps,dtype))
    tangent_input_valid=all_finite(d_f,d_q)
    safe_df=tf.where(tangent_input_valid,d_f,tf.zeros_like(d_f))
    safe_dq=tf.where(tangent_input_valid,d_q,tf.zeros_like(d_q))
    d_rhs=safe_df@p@tf.transpose(safe_f)+safe_f@p@tf.transpose(safe_df)+safe_dq
    dp=tf.reshape(tf.linalg.solve(operator,tf.reshape(d_rhs,[n*n,1])),[n,n])
    dp=(dp+tf.transpose(dp))*tf.constant(.5,dtype)
    # Nonfinite placeholders are confined to the model-to-core boundary.
    # The canonical core records failure before any factorization/solve.
    nan=tf.constant(float("nan"),dtype)
    return tf.where(valid,p,nan),tf.where(valid & tangent_input_valid,dp,nan),{
        "value_valid":valid,"force_valid":valid & tangent_input_valid & all_finite(dp),
        "stability_norm_squared":norm_squared,"relative_equation_residual":relative_residual,
        "positive_definite":p_valid & allowed}


def linear_gaussian_parameter_model(base_transition,base_q,h,r,*,infer_process_scale):
    """F=a*A, Q=exp(2*b)*Qbase (or fixed Q); all derivatives are analytical."""
    a=tf.convert_to_tensor(base_transition);dtype=a.dtype;d=int(a.shape[0])
    base_q=tf.convert_to_tensor(base_q,dtype);h=tf.convert_to_tensor(h,dtype)
    r=tf.convert_to_tensor(r,dtype);m=int(h.shape[0])
    def q_of(theta):
        return base_q*(tf.exp(2*theta[1]) if infer_process_scale else tf.constant(1.,dtype))
    def dq_of(theta,direction):
        return 2*q_of(theta)*direction[1] if infer_process_scale else tf.zeros_like(base_q)
    def stationary(theta,direction):
        return stationary_covariance_and_tangent(theta[0]*a,q_of(theta),direction[0]*a,dq_of(theta,direction))
    def initial(theta):
        p,_,_=stationary(theta,tf.zeros_like(theta))
        return tf.zeros([d],dtype),p
    def dinitial(theta,direction):
        _,dp,_=stationary(theta,direction)
        return tf.zeros([d],dtype),dp
    def transition(theta,x):
        return theta[:,:1]*tf.linalg.matmul(x,a,transpose_b=True)
    def dtransition(theta,x,dx,dtheta):
        return (theta[:,:1]*tf.linalg.matmul(dx,a,transpose_b=True)+
                dtheta[:,:1]*tf.linalg.matmul(x,a,transpose_b=True))
    return PerPointScoreModel(
        transition,dtransition,
        lambda x:tf.linalg.matmul(x,h,transpose_b=True),
        lambda x:tf.broadcast_to(h,[tf.shape(x)[0],m,d]),
        lambda x,dx:tf.linalg.matmul(dx,h,transpose_b=True),base_q,r,
        process_covariance_fn=q_of,process_covariance_tangent_fn=dq_of,
        initial_distribution_fn=initial,initial_distribution_tangent_fn=dinitial)
