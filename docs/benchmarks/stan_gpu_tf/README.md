# Stan's Euclidean NUTS algorithm with TensorFlow model inputs

This standalone implementation evaluates a TensorFlow log density and its
position gradient inside the same XLA computation as the Stan-style sampler.
Initialization, multinomial NUTS tree construction, leapfrog integration and
windowed adaptation reuse `../stan_nuts.py`. Native Stan is needed for the
independent tests and comparator, but is not called by the sampler.

The supported algorithm is the pinned Stan 2.37 Euclidean multinomial NUTS
transition with unit, diagonal or dense inverse metric. The implementation uses
TF Philox streams, rather than Stan MIXMAX/Boost streams. It is an algorithm
port, not the entire Stan compiler, constrained-parameter language or Math
library. The caller supplies unconstrained parameters and includes any
transform Jacobians in the density.

## A model with data supplied at execution time

Set GPU memory growth before importing TensorFlow. When using the GPU
command-buffer execution option, also set XLA_FLAGS before importing the
framework; see the execution discussion below.

```python
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    assert tf.config.experimental.get_memory_growth(gpu)

from docs.benchmarks.stan_gpu_tf import Config, make_functions

chains, dimension, observations = 4, 8, 512

def log_prob(q, design, response):
    residual = tf.matmul(q, design, transpose_b=True) - response
    return (-.5 * tf.reduce_sum(q*q, axis=-1)
            -.5 * tf.reduce_sum(residual*residual, axis=-1))

config = Config(dimension, chains=chains, warmup=1000, draws=2000)
functions = make_functions(tf, log_prob, config, [
    tf.TensorSpec([observations, dimension], tf.float64),
    tf.TensorSpec([observations], tf.float64)])

with tf.device('/GPU:0'):
    q = tf.zeros([chains, dimension], tf.float64)
    present = tf.ones_like(q, tf.bool)
    epsilon = tf.ones([chains], tf.float64)
    inverse_metric = tf.ones_like(q)
    seeds = tf.constant([[101, 0], [202, 0], [303, 0], [404, 0]], tf.int32)
    # design and response are caller-provided float64 tensors of these shapes.
    result = functions.fit(q, present, epsilon, inverse_metric, seeds,
                           design, response)

assert bool(tf.reduce_all(result.valid))  # Host reporting boundary.
posterior_draws = result.sampling.draws  # [draws, chains, dimension]
warmup_draws = result.warmup.draws       # Archive separately; discard for inference.
```

`fit` has a stable signature and `jit_compile=True`. It can be called from a
larger enclosing compiled TF function. Its model data remain tensor operands;
changing values at the same shapes does not require retracing. A different
shape or static configuration needs a new function. All model operations and
their position derivatives must have compatible XLA lowerings. This is not a
promise that arbitrary TF operators, Python callbacks or mutable resource
updates can compile.

Each returned density value must depend on only its corresponding row of `q`.
The derivative of the sum of row densities is then the collection of independent
chain gradients. Batch normalization across chains, cross-chain reductions,
random likelihood evaluations and target mutation violate this contract. Pass
frozen model parameters as tensors, and use deterministic model evaluation.
No pfor or sample-wise model map is used.

## State and continuation

The separate `initialize`, `warmup`, `sample`, and `value_gradient` functions
also have explicit compiled signatures. The legacy direct-port positional
signatures remain valid; model tensors are appended as additional arguments.
For retained chunk continuation:

```python
next_chunk = functions.sample(
    result.sampling.position, result.warmup.step_size,
    result.warmup.inverse_metric, result.sampling.seed, design, response)
```

Reuse the exact model data, metric, step size and final random state. Changed
data define a different target and require a fresh fit before combining draws.
Seeds are `[chain, (key, counter)]` int32 tensors. Keys must be distinct and
counters nonnegative. Static per-chain RNG expansion preserves the original
port's independent streams and masking; graph size grows linearly with chain
count. `rng_layout='map'` selects the original serial RNG-loop baseline.

Raw compiled functions return validity flags because XLA may not enforce TF
assertions. Check those flags before using results. Invalid input values may
produce invalid flagged outputs; they are not admitted posterior draws. Warmup
and sample telemetry retain the direct-port column definitions described in
`../../reference/nuts-direct-stan.md`. Diagnostics and serialization belong
outside the numerical graph. No derivative through the discrete sampler is
provided.

## GPU control flow, fusion and performance

Ordinary GPU XLA compilation can still execute dynamic loop predicates through
small device-to-host transfers. The initial profile in this campaign observed
450 transfers of one byte during a 16-transition regression sample. The target,
gradients and state arrays remained GPU computations.

The master exposes `--command-buffer`, which sets:

```text
XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL
```

This is a pinned-backend execution option, not a portable TensorFlow API. Its
validation requires native transition parity and full-fit replay. TensorFlow
2.20 disables command buffers during profiling because of a known upstream
CUPTI/graph memory-corruption issue; those profiles describe its fallback
executor. Do not interpret their predicate transfers as normal command-buffer
execution, or force-enable the unsafe profiling option. Non-profiled runtime
logs can establish command-buffer use, but do not by themselves prove absence
of every host/device copy. Consult the
[campaign result](../../plans/bayesfilter-stan-gpu-tf-result-2026-10-28.md)
for the measured status on TF 2.20 and this CUDA installation.
The library does not silently change process-wide XLA flags after import.

The optimized HLO includes the model and its gradient inside leapfrog. XLA
forms multiple fusion kernels and may call GPU linear-algebra libraries. A
complete NUTS fit is not one CUDA kernel. Fusion counts alone establish neither
speed nor sampler validity. Record compilation time separately and compare
complete tuning plus retained execution with matched targets and diagnostics.

## Reproduce the reviewed program

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/stan_gpu_tf/master.py \
  --output docs/benchmarks/artifacts/stan-gpu-tf-20261028/new-run \
  --device both --gpu 0 --command-buffer
```

Use a fresh output directory. GPU execution requires the trusted tool context.
The master enforces the campaign's aggregate worker-time/attempt budget from
sibling manifests, captures sources, and runs native-oracle tests, CPU/GPU
integration, timing and post-run diagnostics. A new serious campaign needs a
new reviewed plan/budget; do not delete old evidence to restart accounting.

Plan: `../../plans/bayesfilter-stan-gpu-tf-master-2026-10-28.md`.
Results: `../../plans/bayesfilter-stan-gpu-tf-result-2026-10-28.md`.
Tracked measurements: [evidence-2026-10-28.json](evidence-2026-10-28.json).
The completed campaign used 11 attempts and 1990.90 worker seconds. Its
remaining allowance is not a new full benchmark budget; the example command
requires a fresh authorized campaign budget to rerun the complete program.
The independent source comparison covers the selected Stan algorithm and
tested options; it does not certify all Stan features or production readiness.
