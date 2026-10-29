"""XLA equivalents of pinned C++ InitConfigBuilder choices (unconstrained TF)."""
from dataclasses import dataclass
import math
from typing import NamedTuple
from .api import initial_state,make_kernel

@dataclass(frozen=True)
class Initialization:
    positions: str = 'supplied'
    scale: float = 1.
    mass: str = 'score'
    average_masses: bool = False
    adapt_step: bool = True

    def __post_init__(self):
        if self.positions not in ('supplied','zero','normal'):raise ValueError('invalid positions')
        if self.mass not in ('supplied','unit','score'):raise ValueError('invalid mass')
        if not math.isfinite(self.scale) or self.scale<=0:raise ValueError('scale must be positive')

class Initializers(NamedTuple):
    initialize: object
    random_positions: object

def make_initializers(tf,model,config,initialization=Initialization(),model_signature=(), *, _kernel_factory=None):
    """Inputs are positions [C,D], keys [C,2], mass [C,D], steps [C], *data.

    Supplied operands are ignored only for explicitly selected auto modes.
    Steps always seed the upstream search, or are used directly with adapt_step=False.
    None init_search_limit enables uncapped upstream search; caller owns timeout.
    """
    c=config;policy=initialization
    qspec=tf.TensorSpec([c.chains,c.dimension],tf.float64)
    sspec=tf.TensorSpec([c.chains,2],tf.int32)
    espec=tf.TensorSpec([c.chains],tf.float64)
    @tf.function(input_signature=[sspec],jit_compile=True,autograph=False)
    def random_positions(seed):
        k=make_kernel(tf,lambda q:-tf.reduce_sum(q*q,-1),c)
        q,next_seed=k.random(seed,k.active(),normal=True)
        return q*policy.scale,next_seed
    @tf.function(input_signature=[qspec,sspec,qspec,espec,*model_signature],jit_compile=True,autograph=False)
    def initialize(q,seed,mass,epsilon,*data):
        bound=lambda x:model(x,*data)
        k=make_kernel(tf,bound,c) if _kernel_factory is None else _kernel_factory(tf,bound,c,data)
        input_keys_valid=tf.reduce_all(seed>=0,-1)
        if policy.positions=='normal':
            q,seed=random_positions(seed)
        elif policy.positions=='zero':q=tf.zeros_like(q)
        if policy.mass=='unit':mass=tf.ones_like(q)
        elif policy.mass=='score':
            _,g=k.value_gradient(q)
            mass=(1-c.mass_smoothing)*tf.abs(g)+c.mass_smoothing
        if policy.average_masses:
            mass=tf.broadcast_to(tf.exp(tf.reduce_mean(tf.math.log(mass),0))[None,:],q.shape)
        state=initial_state(k,q,seed,mass,epsilon,policy.adapt_step)
        return state._replace(valid=state.valid & input_keys_valid)
    return Initializers(initialize,random_positions)
