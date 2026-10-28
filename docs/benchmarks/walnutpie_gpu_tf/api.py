"""Stable-signature XLA lifecycle for row-independent TensorFlow targets.

Caller supplies an unconstrained FP64 log density including Jacobian terms.
Fixed-count numerical adaptation is upstream Walnutpie; its external wall-clock
publication/convergence controller is deliberately outside this API.
"""
from typing import NamedTuple
from .core import Kernel, Adam


class State(NamedTuple):
    q: object
    lp: object
    g: object
    epsilon: object
    inverse_mass: object
    minimum: object
    seed: object
    adam: object
    iteration: object
    weight: object
    draw_mean: object
    score_mean: object
    draw_m2: object
    score_m2: object
    macro_total: object
    macro_count: object
    valid: object


class Run(NamedTuple):
    state: object
    draws: object
    log_density: object
    depth: object
    evaluations: object
    macros: object
    failed: object
    reverse_failed: object


class Fit(NamedTuple):
    initial: object
    warmup: object
    sampling: object
    valid: object


class Functions(NamedTuple):
    initialize: object
    warmup: object
    sample: object
    fit: object
    value_gradient: object


def initial_state(k, q, seed):
    tf, c = k.tf, k.cfg
    lp, g = k.value_gradient(q)
    mass = (1-c.mass_smoothing)*tf.abs(g)+c.mass_smoothing
    inv = 1./mass
    eps, seed, valid, _ = k.initial_step(q, lp, g, inv, seed)
    valid &= tf.reduce_all(seed >= 0, -1)
    # Each chain owns a distinct Philox key; duplicate keys correlate streams.
    equal_keys = seed[:, 0, None] == seed[None, :, 0]
    valid &= tf.reduce_sum(tf.cast(equal_keys, tf.int32), axis=1) == 1
    z = tf.zeros_like(q)
    minimum = max(c.min_micro_steps, int(2./c.target_macro_steps+.5))
    return State(q, lp, g, eps, inv, tf.fill([k.c], minimum), seed,
        k.adam_init(eps), tf.constant(0), k.f(c.mass_init_count), z, z,
        c.mass_init_count*inv, c.mass_init_count*mass,
        tf.fill([k.c], k.f(2.)), tf.fill([k.c], k.f(1.)), valid)


def run(k, state, n, adapt):
    tf, c = k.tf, k.cfg
    specs = [(tf.float64, [k.c,k.d]), (tf.float64, [k.c]),
             (tf.int32, [k.c]), (tf.int32, [k.c]), (tf.int32, [k.c]),
             (tf.bool, [k.c]), (tf.bool, [k.c])]
    arrays = tuple(tf.TensorArray(dtype, size=n, element_shape=shape, clear_after_read=False)
                   for dtype, shape in specs)
    def body(i, s, arrays):
        t = k.transition(s.q,s.lp,s.g,s.epsilon,s.inverse_mass,s.minimum,s.seed,s.adam,adapt=adapt)
        valid = s.valid & k.health(t.q,t.lp,t.g) & tf.reduce_all(t.seed >= 0,-1)
        s = s._replace(q=t.q,lp=t.lp,g=t.g,seed=t.seed,adam=t.adam,valid=valid)
        if adapt:
            discount = 1.-1./(c.mass_init_count+tf.cast(s.iteration, tf.float64))
            weight = discount*s.weight+1.
            dq, dg = t.q-s.draw_mean, t.g-s.score_mean
            qm, gm = s.draw_mean+dq/weight, s.score_mean+dg/weight
            q2 = discount*s.draw_m2+dq*(t.q-qm)
            g2 = discount*s.score_m2+dg*(t.g-gm)
            inv = tf.sqrt(q2/g2)
            total = s.macro_total+tf.pow(k.f(2.),tf.cast(t.depth,tf.float64))
            count = s.macro_count+1.
            minimum = tf.maximum(c.min_micro_steps,tf.cast(tf.floor(total/count/c.target_macro_steps+.5),tf.int32))
            eps = tf.exp(t.adam.theta)
            valid &= tf.math.is_finite(eps) & (eps > 0) & tf.reduce_all(tf.math.is_finite(inv) & (inv > 0),-1)
            s = s._replace(epsilon=eps,inverse_mass=inv,minimum=minimum,iteration=s.iteration+1,
                weight=weight,draw_mean=qm,score_mean=gm,draw_m2=q2,score_m2=g2,
                macro_total=total,macro_count=count,valid=valid)
        vals = (t.q,t.lp,t.depth,t.evaluations,t.macros,t.failed,t.reverse_failed)
        return i+1,s,tuple(a.write(i,x) for a,x in zip(arrays,vals))
    _, state, arrays = tf.while_loop(lambda i,s,a: i<n, body, (tf.constant(0),state,arrays),parallel_iterations=1)
    return Run(state,*(a.stack() for a in arrays))


def state_spec(tf,c):
    v=tf.TensorSpec([c.chains],tf.float64); q=tf.TensorSpec([c.chains,c.dimension],tf.float64)
    return State(q,v,q,v,q,tf.TensorSpec([c.chains],tf.int32),
        tf.TensorSpec([c.chains,2],tf.int32),Adam(*(v for _ in range(6))),
        tf.TensorSpec([],tf.int32),tf.TensorSpec([],tf.float64),q,q,q,q,v,v,
        tf.TensorSpec([c.chains],tf.bool))


def make_functions(tf, model, config, model_signature=()):
    """Build complete initialize/warmup/sample/fit XLA graphs.

    model(q,*data)->[C] must preserve chain independence. This cannot be inferred
    automatically: test row isolation for each model. Data stay runtime operands.
    Check result.valid at a host boundary. Checkpoint State preserves all moments,
    RNG counters and caches; sample does not change tuning. Adaptation after a
    target/data change requires reinitialization. RNG is TF Philox, not C++ RNG.
    """
    c=config; data=list(model_signature)
    qs=tf.TensorSpec([c.chains,c.dimension],tf.float64)
    ss=tf.TensorSpec([c.chains,2],tf.int32); st=state_spec(tf,c)
    def kernel(values): return Kernel(tf,lambda q:model(q,*values),c)
    @tf.function(input_signature=[qs,ss,*data],jit_compile=True,autograph=False)
    def initialize(q,seed,*values): return initial_state(kernel(values),q,seed)
    @tf.function(input_signature=[st,*data],jit_compile=True,autograph=False)
    def warmup(state,*values): return run(kernel(values),state,c.warmup,True)
    @tf.function(input_signature=[st,*data],jit_compile=True,autograph=False)
    def sample(state,*values): return run(kernel(values),state,c.draws,False)
    @tf.function(input_signature=[qs,ss,*data],jit_compile=True,autograph=False)
    def fit(q,seed,*values):
        k=kernel(values); s=initial_state(k,q,seed)
        w=run(k,s,c.warmup,True); r=run(k,w.state,c.draws,False)
        return Fit(s,w,r,r.state.valid)
    @tf.function(input_signature=[qs,*data],jit_compile=True,autograph=False)
    def value_gradient(q,*values): return kernel(values).value_gradient(q)
    return Functions(initialize,warmup,sample,fit,value_gradient)
