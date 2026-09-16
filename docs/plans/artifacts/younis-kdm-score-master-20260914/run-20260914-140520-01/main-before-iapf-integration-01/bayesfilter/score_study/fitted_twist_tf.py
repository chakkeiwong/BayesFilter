"""Frozen Gaussian-plus-floor psi-APF and offline recursive fitting.

Uses Guarniero--Johansen--Lee equations (5)--(6), Algorithm 3 and the
independent final run of Algorithm 4. Log-quadratic least squares and fixed
iteration count are explicit local adaptations, not their equation (15) fit
or adaptive stopping rule. Fitted psi is frozen when differentiating theta.
"""
from functools import lru_cache
import tensorflow as tf
from .gaussian_tf import (parameterized_model,chol_tangent,gaussian_log_density_and_tangent,symmetric)


def gaussian_floor_log(x,dx,center,covariance,log_floor):
    lg,dlg=gaussian_log_density_and_tangent(x-center,dx,covariance,
        tf.zeros([6,1,x.shape[-1],x.shape[-1]],x.dtype))
    total=tf.reduce_logsumexp(tf.stack([lg,tf.fill(tf.shape(lg),log_floor)],-1),-1)
    return total,tf.exp(lg-total)[None,:]*dlg


def normalizer(mean,dmean,Q,dQ,center,V,log_floor):
    lg,dlg=gaussian_log_density_and_tangent(center-mean,-dmean,Q+V,dQ[:,None,:,:])
    total=tf.reduce_logsumexp(tf.stack([lg,tf.fill(tf.shape(lg),log_floor)],-1),-1)
    probability=tf.exp(lg-total)
    return total,probability[None,:]*dlg,probability


def twisted_transition(mean,dmean,Q,dQ,center,V,probability,noise,uniform):
    S=Q+V; L=tf.linalg.cholesky(S)
    K=tf.transpose(tf.linalg.cholesky_solve(L,tf.transpose(Q)))
    dK=tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(L,
        tf.linalg.matrix_transpose(dQ-K@dQ)))
    C=symmetric(Q-K@Q)
    dC=symmetric(dQ-dK@Q-K@dQ)
    LC=tf.linalg.cholesky(C); LQ=tf.linalg.cholesky(Q)
    residual=center-mean
    adapted=mean+tf.einsum("ij,nj->ni",K,residual)+tf.einsum("ij,nj->ni",LC,noise)
    dadapted=dmean+tf.einsum("pij,nj->pni",dK,residual)-tf.einsum("ij,pnj->pni",K,dmean)
    dadapted+=tf.einsum("pij,nj->pni",chol_tangent(LC,dC),noise)
    prior=mean+tf.einsum("ij,nj->ni",LQ,noise)
    dprior=dmean+tf.einsum("pij,nj->pni",chol_tangent(LQ,dQ),noise)
    choose=uniform<probability
    return tf.where(choose[:,None],adapted,prior),tf.where(choose[None,:,None],dadapted,dprior)


@lru_cache(maxsize=12)
def make_fitted_twist_kernel(d,o,N,T,dtype_name="float64",jit_compile=True):
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,o],dtype),
        tf.TensorSpec([N,d],dtype),tf.TensorSpec([T,N,d],dtype),tf.TensorSpec([T+1,N],dtype),
        tf.TensorSpec([T,N],dtype),tf.TensorSpec([T,d],dtype),tf.TensorSpec([T,d,d],dtype),
        tf.TensorSpec([T],dtype)],jit_compile=jit_compile)
    def kernel(theta,observations,initial_noise,noise,uniforms,mixture_uniforms,centers,covariances,log_floors):
        A,dA,H,dH,m,dm,P,dP,Q,dQ,R,dR=parameterized_model(theta,d,o)
        L=tf.linalg.cholesky(P)
        x=m+tf.einsum("ij,nj->ni",L,initial_noise)
        dx=dm[:,None,:]+tf.einsum("pij,nj->pni",chol_tangent(L,dP),initial_noise)
        def predict(x,dx):
            return tf.einsum("ij,nj->ni",A,x),tf.einsum("pij,nj->pni",dA,x)+tf.einsum("ij,pnj->pni",A,dx)
        def future(x,dx,t):
            return normalizer(*predict(x,dx),Q,dQ,centers[t],covariances[t],log_floors[t])
        def resample(x,dx,logw,dlogw,u):
            w=tf.nn.softmax(logw)
            indices=tf.searchsorted(tf.cumsum(w),u,side="right")
            indices=tf.minimum(indices,N-1)
            return tf.gather(x,indices),tf.gather(dx,indices,axis=1),tf.reduce_logsumexp(logw)-tf.math.log(tf.cast(N,dtype)),tf.reduce_sum(w[None,:]*dlogw,-1)
        initial_log,initial_dlog,_=future(x,dx,0)
        x,dx,total,dtotal=resample(x,dx,initial_log,initial_dlog,uniforms[0])
        arrays=tf.TensorArray(dtype,size=T,element_shape=[N,d])
        def step(t,x,dx,total,dtotal,clouds):
            mean,dmean=predict(x,dx)
            _,_,probability=normalizer(mean,dmean,Q,dQ,centers[t],covariances[t],log_floors[t])
            x,dx=twisted_transition(mean,dmean,Q,dQ,centers[t],covariances[t],probability,noise[t],mixture_uniforms[t])
            clouds=clouds.write(t,x)
            residual=observations[t]-tf.einsum("ij,nj->ni",H,x)
            dr=-tf.einsum("pij,nj->pni",dH,x)-tf.einsum("ij,pnj->pni",H,dx)
            lg,dlg=gaussian_log_density_and_tangent(residual,dr,R,dR[:,None,:,:])
            lp,dlp=gaussian_floor_log(x,dx,centers[t],covariances[t],log_floors[t])
            lf,dlf=tf.cond(t+1<T,lambda:future(x,dx,t+1)[:2],
                lambda:(tf.zeros([N],dtype),tf.zeros([6,N],dtype)))
            x,dx,term,dterm=resample(x,dx,lg+lf-lp,dlg+dlf-dlp,uniforms[t+1])
            return t+1,x,dx,total+term,dtotal+dterm,clouds
        result=tf.while_loop(lambda t,*_:t<T,step,(0,x,dx,total,dtotal,arrays),parallel_iterations=1)
        return result[3],result[4],result[5].stack()
    return kernel


def quadratic_features(points):
    d=points.shape[-1]
    columns=[tf.ones_like(points[:,0])]+list(tf.unstack(points,axis=1))
    columns += [-.5*points[:,i]**2 for i in range(d)]
    columns += [-points[:,i]*points[:,j] for i in range(d) for j in range(i+1,d)]
    return tf.stack(columns,-1)


def fit_log_quadratic(points,targets,floor_ratio):
    """Full-rank QR fit; reject nonpositive precision instead of clipping it."""
    d=points.shape[-1];dtype=points.dtype
    features=quadratic_features(points)
    scale=tf.maximum(tf.sqrt(tf.reduce_mean(features**2,0)),tf.cast(1.,dtype))
    scaled=features/scale
    singular=tf.linalg.svd(scaled,compute_uv=False)
    eps=tf.cast(2**-23 if dtype==tf.float32 else 2**-52,dtype)
    rank_valid=tf.reduce_min(singular)>eps*tf.cast(max(scaled.shape),dtype)*singular[0]
    q,r=tf.linalg.qr(scaled,full_matrices=False)
    coefficient=tf.linalg.triangular_solve(r,tf.transpose(q)@targets[:,None],lower=False)[:,0]/scale
    precision=tf.linalg.diag(coefficient[1+d:1+2*d])
    pairs=[(i,j) for i in range(d) for j in range(i+1,d)]
    if pairs:
        indices=pairs+[(j,i) for i,j in pairs]
        values=tf.concat([coefficient[1+2*d:],coefficient[1+2*d:]],0)
        precision=tf.tensor_scatter_nd_add(precision,indices,values)
    eigenvalues=tf.linalg.eigvalsh(precision)
    valid=rank_valid & (tf.reduce_min(eigenvalues)>eps*tf.cast(d,dtype)*tf.reduce_max(tf.abs(eigenvalues)))
    factor=tf.linalg.cholesky(precision)
    covariance=tf.linalg.cholesky_solve(factor,tf.eye(d,dtype=dtype))
    center=tf.linalg.cholesky_solve(factor,coefficient[1:1+d,None])[:,0]
    log_peak=tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))-.5*tf.cast(d,dtype)*tf.math.log(2*tf.acos(tf.cast(-1.,dtype)))
    log_floor=tf.math.log(tf.cast(floor_ratio,dtype))+log_peak
    rmse=tf.sqrt(tf.reduce_mean((features@coefficient[:,None]-targets[:,None])**2))
    return tuple(tf.where(valid,x,tf.cast(float("nan"),dtype)) for x in (center,covariance,log_floor)),(valid,rmse,singular)


@lru_cache(maxsize=12)
def make_recursive_fit_kernel(d,o,N,T,floor_ratio,dtype_name="float64",jit_compile=True):
    if not 0<floor_ratio<1: raise ValueError("declare a positive floor ratio below one")
    feature_count=1+2*d+d*(d-1)//2
    if N<feature_count: raise ValueError("too few fit points for full quadratic")
    dtype=tf.as_dtype(dtype_name)
    @tf.function(input_signature=[tf.TensorSpec([6],dtype),tf.TensorSpec([T,o],dtype),
        tf.TensorSpec([T,N,d],dtype)],jit_compile=jit_compile)
    def fit(theta,observations,clouds):
        A,_,H,_,_,_,_,_,Q,_,R,_=parameterized_model(theta,d,o)
        centers=tf.TensorArray(dtype,size=T); covariances=tf.TensorArray(dtype,size=T)
        floors=tf.TensorArray(dtype,size=T); errors=tf.TensorArray(dtype,size=T)
        def step(t,center,V,log_floor,valid,centers,covariances,floors,errors):
            x=clouds[t]
            residual=observations[t]-tf.einsum("ij,nj->ni",H,x)
            lg,_=gaussian_log_density_and_tangent(residual,tf.zeros([6,N,o],dtype),R,tf.zeros([6,1,o,o],dtype))
            lf=tf.cond(t+1<T,lambda:normalizer(tf.einsum("ij,nj->ni",A,x),tf.zeros([6,N,d],dtype),
                Q,tf.zeros([6,d,d],dtype),center,V,log_floor)[0],lambda:tf.zeros([N],dtype))
            (center,V,log_floor),(ok,error,_)=fit_log_quadratic(x,lg+lf,floor_ratio)
            return t-1,center,V,log_floor,valid&ok,centers.write(t,center),covariances.write(t,V),floors.write(t,log_floor),errors.write(t,error)
        out=tf.while_loop(lambda t,*_:t>=0,step,(T-1,tf.zeros([d],dtype),tf.eye(d,dtype=dtype),
            tf.zeros([],dtype),tf.constant(True),centers,covariances,floors,errors),parallel_iterations=1)
        return out[5].stack(),out[6].stack(),out[7].stack(),out[4],out[8].stack()
    return fit
