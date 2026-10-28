# Direct Stan Euclidean NUTS in TensorFlow

[stan_nuts.py](../benchmarks/stan_nuts.py) translates the Stan 2.37.0
Euclidean multinomial NUTS transition and adaptation into TensorFlow. It
supports unit, diagonal and dense inverse metrics. Initialization, warm-up
and retained sampling each execute inside a stable-signature XLA function.
CmdStan is needed for the independent tests and comparisons, not at runtime.
This remains the standalone research route authorized by the owner; the
repository's public HMC tuning APIs are separate.

The source-level comparison and executed performance study are recorded in
the [execution result](../plans/bayesfilter-full-stan-nuts-port-result-2026-10-27.md).
The [source coverage matrix](nuts-stan-coverage.md) distinguishes native-oracle
checks, inspected source, local invariants and deliberate API differences.
The older [adaptation guide](nuts-tuning.md) describes the TFP comparator and
the shared dual-averaging arithmetic.
The [TFP implementation](nuts-stan-tfp.md) now uses stock TFP NUTS with
Stan-style warm-up. It replaced the earlier interface to this direct tree on
October 28. Use this direct module for audited Stan tree behavior; the native
TFP implementation has its own transition and separate performance evidence.

## A generic target

Supply a deterministic differentiable TensorFlow log density mapping FP64
positions of shape `[chains, dimension]` to FP64 values of shape `[chains]`.
Rows must be independent. The target owns parameter constraints, transforms,
Jacobians and constants that depend on position. The sampler checks shape,
dtype and retained numerical health; it cannot prove that a user-supplied
density or gradient describes the intended posterior.

This CPU reference example shows the complete lifecycle. For GPU execution,
set visibility before import and use the repository's memory-growth helper
before creating tensors. The example's counts and starting points illustrate
the API, rather than establish convergence for a new model.

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Explicit CPU reference.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from docs.benchmarks.stan_nuts import (
    Config, FrozenState, make_functions, freeze, require_valid,
)

def log_prob(q):
    scales = tf.constant([1., 3.], tf.float64)
    return -.5 * tf.reduce_sum(tf.square(q / scales), axis=-1)

cfg = Config(2, chains=4, warmup=1000, draws=2000, metric='diag')
initialize, warmup, sample = make_functions(tf, log_prob, cfg)
seed = tf.constant([[71, 0], [72, 0], [73, 0], [74, 0]], tf.int32)
q, seed, valid, attempts = initialize(
    tf.zeros([4, 2], tf.float64), tf.zeros([4, 2], tf.bool), seed,
)
if not bool(tf.reduce_all(valid)):
    raise RuntimeError('Initialization failed')
epsilon = tf.fill([4], tf.constant(cfg.initial_epsilon, tf.float64))
inverse_metric = tf.ones([4, 2], tf.float64)
warmed = require_valid(warmup(q, epsilon, inverse_metric, seed))
frozen = freeze(warmed)
retained = require_valid(sample(*frozen))
continued = FrozenState(retained.position, frozen.step_size,
                        frozen.inverse_metric, retained.seed)
next_chunk = require_valid(sample(*continued))
```

Save `warmed.draws` separately and exclude them from posterior estimates.
`require_valid` is a host boundary check: raw XLA results carry per-chain
validity because TensorFlow assertions alone may be omitted by XLA. Keep
the same target, metric and epsilon when continuing retained sampling.
Continuation preserves per-chain random state. With thinning, use chunk lengths
divisible by `thin` to preserve the concatenated save pattern. Same-backend chunk continuation
is tested; CPU/GPU arithmetic need not generate identical long trajectories.
Each state row is a stream key followed by a nonnegative counter. Use distinct
keys across chains and independent fits; merely offsetting counters under a
shared key creates overlapping streams and is rejected within a batch.

## Options and metric convention

The supplied metric is the inverse mass (C), with kinetic energy

\[
K(p)=\tfrac12 p^T C p,\qquad \dot q=Cp,\qquad p\sim N(0,C^{-1}).
\]

For unit/diagonal modes, supply shape `[chains, dimension]`; unit mode
requires ones. Dense mode takes `[chains, dimension, dimension]` and requires
a symmetric positive-definite matrix. Dense momentum uses the cached
Cholesky factor (C=LL^T) and (p=L^{-T}z). Diagonal mode uses vector
arithmetic and storage, including its variance estimator.

| Option | Meaning |
| --- | --- |
| `adapt=True` | Initial epsilon search, dual averaging and windowed metric updates; unit mode adapts epsilon only |
| `adapt=False` | Use supplied metric and epsilon throughout warm-up and sampling |
| `warmup=0` | No warm-up transitions; an enabled initial epsilon search still executes |
| `draws` | Number of sampling transitions before thinning |
| `thin` | Save zero-based iterations 0, thin, 2 thin, … independently in each phase |
| `jitter` | Per-transition epsilon multiplier (1+j(2u-1)), with (0\le j<1) |
| `depth` | Maximum binary-tree depth; counters support depths up to 30 |
| `max_delta` | Stan's strict energy-error threshold inside tree construction |
| `init_radius` | Uniform unconstrained random initialization within ±radius; zero requests zeros |

The initialization `present` mask marks supplied coordinates. Fully supplied
or radius-zero initialization gets one attempt; partial/random initialization
gets up to 100, following Stan's service limits. This is a flat unconstrained
coordinate interface. It does not implement Stan's named constrained-variable
initialization or model transforms. Numerical target failures must be expressed
by tensors; arbitrary Python exceptions cannot execute inside XLA.

Zero-dimensional targets return constant empty positions as an explicitly
degenerate local evaluation convention. Its synthetic telemetry is not a
claim of reproducing Stan's zero-parameter service/CSV behavior.

## Adapting and diagnosing a new model

Start with Stan's inherited settings: delta .8, gamma .05, kappa .75,
t0 10, initial buffer 75, terminal buffer 50 and first window 25. They are
starting hypotheses. Choose a warm-up budget appropriate to the target,
inspect adaptation and then assess fresh frozen sampling with multiple chains,
modern R-hat, bulk/tail ESS, Monte Carlo precision and available reference
information. High acceptance alone does not establish convergence.

For fewer than 20 warm-up transitions there is no metric adaptation.
For a larger budget that cannot fit the buffers, the pinned 2.37 source sets
shortened stage lengths but returns without restarting its window counter.
Consequently a fresh adapter performs no metric updates in that fallback.
The direct port reproduces that observed release behavior and tests it against
actual headers. Supply fitting windows when metric adaptation is intended.
Zero-length buffers/windows follow the source's counter behavior as well.

Each metric update shrinks the window covariance as

\[
C=\frac{n}{n+5}S+\frac{5}{n+5}10^{-3}I.
\]

For a singleton window, Stan keeps the previous metric in place of (S).
After an update the sampler searches for epsilon again, resets dual averaging
and discards the covariance accumulator. Final sampling uses the smoothed
epsilon and frozen metric. Warm-up telemetry records the unsmoothed next
epsilon; read `warmed.step_size` for the final frozen value.

The initial search brackets acceptance .8, independently of delta. It follows
Stan's repeated first probe, fresh momenta and strict comparisons. The finite
2048-probe loop bound exceeds the binary exponent range relevant to the source's
1e7 step-size cap. XLA subnormal flushing can change the number of probes at
underflow; such a failed search is not a valid warm-up result.

Sampling telemetry columns are acceptance probability, position-change
indicator, evaluated leapfrogs, depth-cap indicator, divergence, cumulative
retained-state health, tree depth, selected-state energy and realized epsilon.
Warm-up appends next nominal epsilon, metric-update indicator, cumulative
search probes and cumulative validity. Work and acceptance include evaluated
leaves in rejected subtrees, as in Stan. The divergence threshold stops a tree;
its reporting flag alone is not an external epsilon-tuning rejection rule.

## Fidelity and performance

The tree traversal retains both Stan proposal rules: subtree merges use
ordinary normalized multinomial weights, while outer doubling uses the
subtree-to-existing-weight ratio capped at one. It applies all three momentum
U-turn criteria at both levels. The binary-carry stack traverses leaves and
merges in the same order as the recursive source, with workspace proportional
to dimension times maximum depth. It does not build an exponentially large
TensorFlow graph.

The independent oracle instantiates unmodified Stan `base_nuts` and
`expl_leapfrog` headers with controlled target, momentum and uniform inputs.
Tests check proposals, cache values, work, depth, divergence, acceptance,
energy, jitter and random consumption. Adaptation is replayed through native
adapters using actual positions and acceptance statistics from this port.
These checks establish the tested source behavior, not posterior correctness
for every target a caller might supply.

Runtime randomness is TensorFlow stateless Philox with independent per-chain
state. Stan's MIXMAX generator and Boost distribution transforms are not
ported; identical integer seeds therefore do not mean identical chains.
The Stan language/compiler, distribution library and generated quantities
remain separate. This is a transfer of the declared Euclidean sampler,
rather than all of Stan.

Use [benchmark_full_stan.py](../benchmarks/benchmark_full_stan.py) through the
bounded [attempt launcher](../benchmarks/run_stan_port_attempt.py). It supports
CPU scalar, CPU batch and GPU batch execution. Select a layout from separate
calibration, then freeze it for the comparison. Report compilation, warm-up,
sampling, serialization and diagnostics separately. Native Stan includes
process startup and CSV output in its process-wall result. The fixed-HMC arm
uses the public standard tuner and reports every verified candidate; aggregate
cost of assessing all candidates is distinct from a single member's cost.
The harness's `--warmup` controls NUTS. Fixed HMC uses the public preparation
schedule and acceptance policy; its recorded protocol gives the actual counts.
Its compilation and archive costs also differ from the reusable NUTS timing
boundary, so consult the result's cost definitions before comparing totals.

The October 27 comparison selected CPU scalar chains for these fixtures.
The direct port passed the declared posterior screens on six of seven targets,
as did TFP and native Stan; all failed on the centered funnel. It was slower
than TFP on the smaller targets. On the larger regression its total tuning and
sampling time was lower descriptively, because warm-up was cheaper; retained
sampling alone was still slower. Native Stan had lower observed elapsed costs.
See the result for timing boundaries, replication limits and the fixed-HMC
comparison. Source fidelity alone is not evidence of faster execution.
