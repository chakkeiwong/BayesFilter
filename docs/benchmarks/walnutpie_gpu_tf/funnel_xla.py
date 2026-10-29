"""Fixed-parameter R2P funnel execution. Numerical runtime is TensorFlow/XLA.

Author quantile tuning is an external independent-reference input. This module
does not implement Appendix C adaptation and never imports its NumPy code.
"""
from .api import initial_state, make_kernel, run, state_spec
from .core import Config
from .worker import model_for


def functions(tf, dimension, step, delta, chunk=64):
    cfg=Config(chains=4,dimension=dimension,warmup=0,draws=chunk,max_depth=12,
        variant='r2p',r2p_max_exponent=10,initial_step=step,max_error=delta)
    k=make_kernel(tf,model_for(tf,2),cfg)
    @tf.function(input_signature=[tf.TensorSpec([4,dimension],tf.float64),
                                  tf.TensorSpec([4,2],tf.int32)],
                 jit_compile=True,autograph=False)
    def initialize(q,seed):
        return initial_state(k,q,seed,mass=tf.ones_like(q),
                             epsilon=tf.fill([4],k.f(step)),adapt_step=False)
    @tf.function(input_signature=[state_spec(tf,cfg)],jit_compile=True,autograph=False)
    def sample(state):return run(k,state,chunk,False)
    return initialize,sample,cfg
