"""Standalone complete-fit bridge, XLA-compatible with CPU sampler execution.

Not a TFP TransitionKernel; not differentiable; no Python callbacks in graph.
Caller configures GPU visibility and memory growth before importing TensorFlow.
"""
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf

BUILD = Path(__file__).resolve().parents[3] / ".localresources/native_sampler_bridge"

@dataclass(frozen=True)
class FitConfig:
    engine: str
    kind: int
    dimension: int
    chains: int
    warmup: int
    draws: int
    depth: int
    halvings: int = 5

    def __post_init__(self):
        if self.engine not in ("stan", "walnutpie"):
            raise ValueError("engine must be stan or walnutpie")
        if self.kind not in range(4):
            raise ValueError("kind must be 0..3")
        if not (1 <= self.dimension <= 4096 and 1 <= self.chains <= 64):
            raise ValueError("invalid dimension or chains")
        if self.warmup < 1 or self.draws < 1 or self.warmup+self.draws > 10000000:
            raise ValueError("invalid iterations")
        if self.chains*(self.warmup+self.draws)*(self.dimension+7) > 100000000:
            raise ValueError("output exceeds bridge bound")
        if not (1 <= self.depth <= 15 and 0 <= self.halvings <= 10):
            raise ValueError("invalid trajectory bounds")
        if self.kind == 2 and self.dimension % 2:
            raise ValueError("banana requires even dimension")
        if self.kind == 3 and self.dimension < 2:
            raise ValueError("funnel requires at least two dimensions")

    def native_config(self):
        return [int(self.engine == "walnutpie"), self.kind, self.dimension,
                self.chains, self.warmup, self.draws, self.depth, self.halvings]

def compiled_fit(config: FitConfig, device="/CPU:0"):
    """Return a stable-signature full fit. Explicit controls and seeds are inputs.

    Returns rows[C,W+S,D+7] and metadata[C,D+4]; rows[:,:W] are warmup.
    controls=[bend,initial_step_size,target_accept,walnutpie_energy_bound].
    Seeds are per-chain uint32 values carried in int64 tensors.
    GPU device executes D2H/native CPU/H2D inside the XLA custom call.
    """
    op = tf.load_op_library(str(BUILD / "_native_sampler_ops.so"))
    d,c = config.dimension,config.chains
    @tf.function(input_signature=[tf.TensorSpec([4],tf.float64),
        tf.TensorSpec([c,d],tf.float64),tf.TensorSpec([d,d],tf.float64),
        tf.TensorSpec([c],tf.int64)],jit_compile=True,autograph=False)
    def fit(controls,starts,inverse_transform,seeds):
        with tf.device(device):
            return op.bf_native_sampler(controls,starts,inverse_transform,seeds,
                engine=int(config.engine=="walnutpie"),kind=config.kind,
                warmup=config.warmup,draws=config.draws,depth=config.depth,halvings=config.halvings)
    return fit
