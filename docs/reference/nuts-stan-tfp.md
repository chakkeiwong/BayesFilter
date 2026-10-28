# Native TFP NUTS with Stan-style warm-up

[`StanNoUTurnSampler`](../benchmarks/stan_nuts_tfp.py) now uses **stock
`tfp.mcmc.NoUTurnSampler`** for its transition. It is a TFP TransitionKernel
that supports `sample_chain`, retains native target/gradient caches, and uses
TFP bijectors for a frozen Euclidean inverse mass. Its retained sampling loop
executes inside XLA. The existing Stan-style warm-up also uses stock TFP.

This replaces the October27 wrapper around our custom Stan tree. The class
name is retained for existing callers, but its implementation identity is
`native_tfp_v2`. It does not reproduce native Stan's exact tree traversal or
RNG. The [direct Stan port](nuts-direct-stan.md) remains available separately
for source-fidelity work. This is still the standalone benchmark API, not a
change to public HMC defaults.

## A complete compiled lifecycle

The target maps independent FP64 rows `[chains, dimension]` to FP64 values
`[chains]`. Targets own constraints/transforms/Jacobians. This tiny CPU example
illustrates the API; its counts are not a convergence prescription.

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Explicit CPU reference.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow_probability as tfp
from docs.benchmarks.stan_nuts_warmup import Config, freeze
from docs.benchmarks.stan_nuts_tfp import make_functions, require_valid

def log_prob(q):
    return -.5 * tf.reduce_sum(q*q, axis=-1)

config = Config(2, chains=2, warmup=160, draws=32, depth=6,
                initial_buffer=10, terminal_buffer=10, window=20)
warmup, sample = make_functions(tf, tfp, log_prob, config)
warmed = require_valid(warmup(tf.zeros([2, 2], tf.float64),
                              tf.constant([101, 202], tf.int32)))
frozen = freeze(warmed)
result = require_valid(sample(*frozen))
# Exact continuation under the fixed kernel; retain warm-up separately.
continuation = require_valid(sample(result.position, frozen.step_size,
                                    frozen.metric_factor, result.seed))
```

The factory reuses the audited Stan-style controller: initial epsilon search,
windowed covariance, regularization, dual averaging, boundary restarts and
final smoothing. The adaptation algorithm is in one shared module; it is not
reimplemented in the TFP adapter. Warm-up and retained sampling each have
stable signatures and `jit_compile=True`. Functions accept the final metric
and step sizes as tensors, so tuning does not require retracing.

`Config` follows the existing TFP adaptation contract: diagonal/dense mass,
positive warm-up/draw counts, and explicit fitting windows for short warm-up.
Warm-up samples are archived separately and excluded from posterior summaries.
`num_burnin_steps` in TFP itself merely discards transitions; it is not a
replacement for this adaptation.

## Using the kernel with sample_chain

For user-controlled sampling, build a frozen kernel after adaptation:

```python
from docs.benchmarks.stan_nuts_tfp import StanNoUTurnSampler
kernel = StanNoUTurnSampler.from_warmup(log_prob, warmed, config)

@tf.function(input_signature=[tf.TensorSpec([2, 2], tf.float64)],
             autograph=False, jit_compile=True)
def sample_tfp(q):
    return tfp.mcmc.sample_chain(
        32, q, kernel=kernel, trace_fn=lambda _, kr: kr.valid,
        return_final_kernel_results=True, parallel_iterations=1)

result = sample_tfp(warmed.position)
require_valid(result.final_kernel_results)
```

To continue exactly, pass the last state and its matching final kernel results
to the next compiled `sample_chain` through `previous_kernel_results`, without
a new chain seed. Keep states and results together: TFP's transformed kernel
reuses its transformed position and gradient cache. Replacing the position
requires bootstrapping, as does changing the metric or target.

You can also construct a kernel with `step_size` (scalar or `[chains]`),
`inverse_metric`, `metric`, and `seed_state`. Unit mass uses bare TFP NUTS.
Diagonal mass uses an elementwise scale; dense mass uses `ScaleMatvecTriL`.
For inverse mass C and q=A z with A A^T=C, the transformed unit-mass
Hamiltonian gives the intended physical dynamics. TFP includes the constant
log-Jacobian. No matrix multiply is needed for each diagonal leapfrog.

## Migration from the October27 interface

- `seed_state` is one int32 `[2]` stateless pair for TFP's batched randomness.
  Direct-port `[chains,2]` key/counter states are rejected. TFP generates
  independent row draws; equal explicit pairs reproduce a run.
- A seedless `one_step` splits and advances the adapter's carried pair. A new
  explicit pair overrides it for that transition. TFP `sample_chain(seed=...)`
  supplies fresh transition seeds; use continuation results and omit its seed
  to continue exactly across chunks. Integer seeds are rejected by direct
  `one_step`; prefer tensor pairs in compiled code.
- `from_warmup` accepts `stan_nuts_warmup.WarmupResult`. Direct-port warm-up
  results carry different RNG and metric schemas and are rejected explicitly.
- Results contain `inner_results`, next `seed`, `seed_initialized` and
  cumulative `valid`. Use `kernel.native_results(results)` for TFP acceptance,
  leapfrogs, divergence and depth-cap flags. Target and gradient caches are
  in **transformed coordinates**, including the constant density Jacobian.
  Native TFP reports a depth-cap flag, not exact tree depth.
- Nonzero step-size jitter is not supported by this adapter. The inherited
  benchmark uses zero. Maximum depth is capped at20 to bound TFP's static
  instruction storage; this is an engineering limit, not a tuning default.

The wrapper intentionally does not claim arbitrary nested-state, sharded or
external adaptation-wrapper compatibility. FP64 independent rows remain the
validated shape contract. Compile the whole repeated loop; the class methods
are graph-building operations, not separate eager sampling drivers.

XLA numerical checks return validity flags. Call `require_valid` at the host
boundary after each chunk. Correctly rejected nonfinite proposals leave a
valid current state and count as zero acceptance; their divergence flags are
reported. Finite state/cache, convergence, precision and reference checks are
separate requirements. `is_calibrated=True` is a kernel property, not a
posterior convergence certificate.

## Comparison

The new [benchmark](../benchmarks/benchmark_native_tfp_stan.py) calls this
module's compiled factory and prices the actual `sample_chain` route against
CmdStan, including warm-up. The [plan](../plans/bayesfilter-stan-tfp-native-2026-10-28.md)
fixes target definitions, seeds, posterior screens, timing boundaries and
compute budget. Results belong to that execution, not to the prior direct
wrapper's timings. Compilation and host serialization are reported separately.
Native Stan is compiled C++, not an XLA route.
