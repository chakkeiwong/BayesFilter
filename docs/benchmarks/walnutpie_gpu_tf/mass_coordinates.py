"""Fixed dense mass basis around the shared diagonal TF/XLA authority.

For supplied momentum covariance M=L L', y=L' q and p=L r preserve the
canonical one-form p' dq=r' dy. Identity mass in y therefore implements M in
q. Diagonal adaptation in y yields M_eff=L diag(m_y) L'. This is fixed-basis
adaptation, not unrestricted dense covariance adaptation.

All transformations and model gradients remain TensorFlow/XLA operations.
The basis must be held fixed across continuation; checkpoints carry it.
"""
from typing import NamedTuple
from .api import make_supplied_mass_functions, state_spec
from .core import Kernel
from .r2p import R2PKernel


def physical_uturn(tf, bq, bp, fq, fp, inv, L):
    """Paper Definition 10 in physical q, expressed in fixed y coordinates.

    p' M_eff^-1 delta_q = r' diag(inv) L^-1 L'^-1 delta_y.
    Simply applying the ordinary y-coordinate U-turn changes the algorithm.
    """
    delta=tf.linalg.triangular_solve(L,(fq-bq)[...,None],lower=True,adjoint=True)
    delta=tf.linalg.triangular_solve(L,delta,lower=True)[...,0]*inv
    return (tf.reduce_sum(bp*delta,-1)<0)|(tf.reduce_sum(fp*delta,-1)<0)


class DenseState(NamedTuple):
    sampler: object
    mass_cholesky: object
    valid: object


class DenseRun(NamedTuple):
    state: object
    draws: object
    log_density: object
    depth: object
    evaluations: object
    macros: object
    failed: object
    reverse_failed: object


class DenseFit(NamedTuple):
    initial: object
    warmup: object
    sampling: object
    valid: object


class DenseFunctions(NamedTuple):
    initialize: object
    warmup: object
    sample: object
    fit: object
    physical_positions: object
    physical_mass: object


def validate_mass(tf, mass, config, *, dense=False):
    """Host validation and shared/per-chain broadcasting before initialization.

    Returns [C,D] for diagonals or [C,D,D] for full symmetric positive definite
    matrices. This optional host check gives readable errors; the compiled
    initializer separately propagates invalid inputs in its valid flag.
    """
    x = tf.convert_to_tensor(mass, dtype=tf.float64)
    shape = [config.chains, config.dimension, config.dimension] if dense else [config.chains, config.dimension]
    shared = shape[1:]
    if x.shape.as_list() == shared:
        x = tf.broadcast_to(x, shape)
    elif x.shape.as_list() != shape:
        raise ValueError('mass must have shape '+str(shared)+' or '+str(shape))
    if not bool(tf.reduce_all(tf.math.is_finite(x)).numpy()):
        raise ValueError('mass must be finite')
    if dense:
        scale = tf.maximum(tf.reduce_max(tf.abs(x), axis=[-2,-1]), tf.constant(1.,tf.float64))
        symmetric = tf.reduce_max(tf.abs(x-tf.linalg.matrix_transpose(x)), axis=[-2,-1]) <= 1e-12*scale
        if not bool(tf.reduce_all(symmetric).numpy()):
            raise ValueError('dense mass must be symmetric')
        if not bool(tf.reduce_all(tf.linalg.eigvalsh(x)>0).numpy()):
            raise ValueError('dense mass must be positive definite')
    elif not bool(tf.reduce_all(x>0).numpy()):
        raise ValueError('diagonal mass must be positive')
    return x


def make_dense_mass_functions(tf, model, config, model_signature=(), *, adapt_step=True):
    """initialize/fit(q,seeds,M,epsilon,*data), M has shape [C,D,D].

    The compiled API expects a symmetric SPD matrix. For readable exceptions,
    call validate_mass(...,dense=True) before it. Bad input is also marked
    invalid by the compiled initializer. DenseState stores the immutable basis
    so a continuation cannot silently reinterpret the existing target cache.
    Returned draws/log_density are in physical coordinates; state.sampler is
    in y coordinates. Use physical_positions and physical_mass to inspect it.
    """
    c=config
    qs=tf.TensorSpec([c.chains,c.dimension],tf.float64)
    ms=tf.TensorSpec([c.chains,c.dimension,c.dimension],tf.float64)
    ss=tf.TensorSpec([c.chains,2],tf.int32)
    es=tf.TensorSpec([c.chains],tf.float64)
    vs=tf.TensorSpec([c.chains],tf.bool)
    st=DenseState(state_spec(tf,c),ms,vs)
    def from_y(y,L):
        return tf.linalg.triangular_solve(L,y[...,None],lower=True,adjoint=True)[...,0]
    def transformed(y,L,*data):
        return model(from_y(y,L),*data)-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)),-1)
    def factory(tf,bound,c,values):
        L=values[0]
        base=R2PKernel if c.variant=='r2p' else Kernel
        class PhysicalKernel(base):
            def uturn(self,old,new,forward,inv):
                return physical_uturn(tf,self.choose(forward,old.bq,new.bq),
                    self.choose(forward,old.bp,new.bp),self.choose(forward,new.fq,old.fq),
                    self.choose(forward,new.fp,old.fp),inv,L)
        return PhysicalKernel(tf,bound,c)
    inner=make_supplied_mass_functions(tf,transformed,c,(ms,*model_signature),
        adapt_step=adapt_step,_kernel_factory=factory)
    @tf.function(input_signature=[qs,ss,ms,es,*model_signature],jit_compile=True,autograph=False)
    def initialize(q,seeds,mass,epsilon,*data):
        finite=tf.reduce_all(tf.math.is_finite(mass),[-2,-1])
        scale=tf.maximum(tf.reduce_max(tf.abs(mass),[-2,-1]),tf.constant(1.,tf.float64))
        symmetric=tf.reduce_max(tf.abs(mass-tf.linalg.matrix_transpose(mass)),[-2,-1])<=1e-12*scale
        safe=tf.where((finite&symmetric)[:,None,None],mass,tf.eye(c.dimension,batch_shape=[c.chains],dtype=tf.float64))
        L=tf.linalg.cholesky(safe)
        good=finite&symmetric&tf.reduce_all(tf.math.is_finite(L),[-2,-1])&tf.reduce_all(tf.linalg.diag_part(L)>0,-1)
        L=tf.where(good[:,None,None],L,tf.eye(c.dimension,batch_shape=[c.chains],dtype=tf.float64))
        y=tf.linalg.matvec(L,q,transpose_a=True)
        s=inner.initialize(y,seeds,tf.ones_like(q),epsilon,L,*data)
        return DenseState(s._replace(valid=s.valid&good),L,good)
    def convert(r,state):
        L=state.mass_cholesky
        # Batched triangular solve: [C,D,N] RHS for all retained draws.
        draws=tf.transpose(tf.linalg.triangular_solve(L,tf.transpose(r.draws,[1,2,0]),
            lower=True,adjoint=True),[2,0,1])
        lp=r.log_density+tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)),-1)[None,:]
        s=DenseState(r.state._replace(valid=r.state.valid&state.valid),L,state.valid)
        return DenseRun(s,draws,lp,r.depth,r.evaluations,r.macros,r.failed,r.reverse_failed)
    @tf.function(input_signature=[st,*model_signature],jit_compile=True,autograph=False)
    def warmup(state,*data):
        return convert(inner.warmup(state.sampler,state.mass_cholesky,*data),state)
    @tf.function(input_signature=[st,*model_signature],jit_compile=True,autograph=False)
    def sample(state,*data):
        return convert(inner.sample(state.sampler,state.mass_cholesky,*data),state)
    @tf.function(input_signature=[qs,ss,ms,es,*model_signature],jit_compile=True,autograph=False)
    def fit(q,seeds,mass,epsilon,*data):
        initial=initialize(q,seeds,mass,epsilon,*data)
        w=warmup(initial,*data);r=sample(w.state,*data)
        return DenseFit(initial,w,r,r.state.sampler.valid&initial.valid)
    @tf.function(input_signature=[st],jit_compile=True,autograph=False)
    def physical_positions(state):return from_y(state.sampler.q,state.mass_cholesky)
    @tf.function(input_signature=[st],jit_compile=True,autograph=False)
    def physical_mass(state):
        L=state.mass_cholesky
        return tf.matmul(L/state.sampler.inverse_mass[:,None,:],L,transpose_b=True)
    return DenseFunctions(initialize,warmup,sample,fit,physical_positions,physical_mass)
