# NUTS tuning for the standalone XLA benchmark

This guide covers the owner-authorized October 27, 2026 experiment. The
subsequent [direct Stan port guide](nuts-direct-stan.md) describes the complete
Euclidean transition in TensorFlow, its expanded options and native-header
validation. The text below documents the earlier TFP-based comparator.
Its restrictions should not be confused with those of the direct port.
The
implementation is in [stan_nuts_warmup.py](../benchmarks/stan_nuts_warmup.py),
with comparison orchestration in
[benchmark_stan_transplant.py](../benchmarks/benchmark_stan_transplant.py).
It is a standalone diagnostic route; retired public NUTS APIs remain retired.
The [plan and review](../plans/bayesfilter-stan-nuts-transplant-2026-10-27.md)
define the evidence and compute limits. See the
[results](../plans/bayesfilter-stan-nuts-transplant-result-2026-10-27.md)
for the executed comparison and limitations.

## Reusing the Python code

The adaptation is implemented in Python with TensorFlow operations. CmdStan
is needed for the independent comparison tests, not to run the TensorFlow
implementation. TensorFlow and TFP supply native compiled numerical kernels;
"Python implementation" does not mean interpreted Python inside each leapfrog
step. The enclosing warm-up and retained-sampling functions compile with XLA.

The module accepts arbitrary fixed, differentiable log densities with a
specific interface: float64 positions of shape [chains, dimension] produce
float64 log densities of shape [chains]. Each row must depend only on the
corresponding position row. The callable must remain deterministic and fixed
during warm-up and sampling. The caller supplies unconstrained coordinates,
parameter transforms and any required Jacobian terms. Shape, dtype and
disconnected gradients are checked; the implementation cannot automatically
prove row independence or that the supplied density is the intended posterior.

From the repository root, the following is a CPU reference example. Set device
visibility before importing TensorFlow. Serious GPU use must configure and
verify the repository memory-growth policy before creating tensors.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Explicit CPU reference.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import tensorflow_probability as tfp
from docs.benchmarks.stan_nuts_warmup import (
    Config, FrozenState, make_functions, warmup_checked, freeze, sample_checked,
)

def log_prob(q):
    scale = tf.constant([1., 3.], tf.float64)
    return -0.5 * tf.reduce_sum(tf.square(q / scale), axis=-1)

config = Config(dimension=2, chains=4, warmup=1000, draws=2000, metric="diag")
warmup, sample = make_functions(tf, tfp, log_prob, config)
initial = tf.constant([[.1, -1.], [-.5, 2.], [.8, .3], [-1., -.4]], tf.float64)
warmed = warmup_checked(warmup, initial, tf.constant([49, 10], tf.int32))
state = freeze(warmed)
retained = sample_checked(sample, state)

# Keep warmed.draws separately; use retained.draws for posterior diagnostics.
next_state = FrozenState(
    retained.position, state.step_size, state.metric_factor, retained.seed,
)
next_chunk = sample_checked(sample, next_state)
```

The example's counts and starts illustrate the interface; they are not
evidence of adequate warm-up or convergence for a new target. Each compiled
function has a fixed signature. Construct new functions when the target or
configuration changes. Keep the same target, metric and step size when
resuming retained sampling. Dense-coordinate round trips can introduce
floating-point differences at a chunk boundary; no bitwise trajectory
invariance across arbitrary layouts or chunking is promised.

WarmupResult contains position, step_size, metric_factor, seed, draws,
telemetry, valid and completion_token. SampleResult contains position, seed,
draws, telemetry, valid and completion_token. The raw functions return a
per-chain valid flag instead of relying on assertions that XLA may omit.
The checked helpers materialize only that status at the host boundary and
raise on failure. The benchmark also checks both warm-up and sampling status.
A failed result is debugging evidence, not a posterior sample set.

Warm-up telemetry columns are acceptance probability, selected-state indicator,
leapfrog count, tree-depth saturation, divergence flag, current state/cache
health, next unsmoothed step size, metric-update flag, cumulative search probes
and cumulative validity. The final frozen step size is warmed.step_size; the
last telemetry step size precedes final smoothing. Sampling has the first six
columns, with cumulative validity in the sixth. A rejected nonfinite proposal
can have zero acceptance and a divergence flag while preserving a valid
retained state.

The reusable arithmetic routines are stan_schedule, dual_update,
welford_update, regularized_metric, reasonable_step_size and leapfrog_probe.
They can be composed inside other TensorFlow functions. The supplied
make_functions controller specifically wraps TFP NUTS; it is not a general
adapter for arbitrary transition-kernel APIs.

## What is adapted

Each chain has its own step size, dual-averaging state and window-local
covariance estimate. The initial metric is identity. Initial step size is a
hypothesis used by a one-leapfrog acceptance search, which brackets 0.8 by
doubling or halving with refreshed momentum. This 0.8 bracket is separate
from the configurable dual-averaging target acceptance.

Warm-up has three stages. The initial fast buffer adapts step size; expanding
slow windows adapt the metric and step size; the terminal fast buffer adapts
step size in the final metric. The source defaults are 75 initial transitions,
25 in the first slow window and 50 terminal transitions. With 1000 warm-up
transitions, zero-based metric updates are 99, 149, 249, 449 and 949.
All metric updates use only their own window. Earlier windows are discarded
from the covariance accumulator, while every warm-up draw is saved as evidence.

For n samples with sample covariance S, the installed inverse metric is

    C = n/(n+5) S + (5/(n+5)) 0.001 I.

For n greater than one, S is the unbiased streaming sample covariance. For
a one-sample window, Stan leaves S equal to the preceding inverse metric
before applying the same shrinkage; the transplant follows this behavior.
For a diagonal metric, use the diagonal of this expression. The constants
come from the pinned Stan 2.37.0 source. This regularization is an adaptation
rule, not a posterior covariance accuracy guarantee. After an update, rerun
the step-size search in the new metric and reset dual averaging with
mu=log(10 epsilon). Finish warm-up by freezing the smoothed log-step-size
estimate. During retained sampling, NUTS continues selecting trajectory
length at each transition; adaptation of metric and step size has stopped.

The standard dual-averaging settings are delta=0.8, gamma=0.05, kappa=0.75 and
t0=10. They are source-inherited starting hypotheses. Increasing delta usually
produces smaller steps and more leapfrogs; it must not be treated as a universal
speed improvement. Maximum depth=10 and zero step-size jitter are explicit
benchmark settings. The implementation does not tune depth using final results.

## Mapping the metric to TensorFlow

Let C=A A^T be the learned inverse metric, with A lower triangular. The code
runs unit-mass TFP NUTS in z coordinates, where q=A z. If p_z is standard
normal, physical momentum is p_q=A^-T p_z. Hence its kinetic energy is
p_q^T C p_q/2 and its velocity is C p_q. This is the same constant-metric
Hamiltonian in different coordinates. A constant log-determinant does not
change the fixed-metric transition or its acceptance ratios, so the target
closure evaluates log p(A z) without that constant.

At each metric boundary, the physical position q is preserved, z is
recomputed by triangular solve, and the transformed gradient/target cache
is rebuilt. Reusing a gradient cache from the old coordinates would be wrong.
The tests check dense orientation against Stan's actual leapfrog integrator,
incoming and outgoing value/gradient caches, separate per-chain estimates,
reproducibility, every window flag, final freezing and chunked sampling.

## Source matches and deliberate differences

The isolated installation is .localresources/cmdstan/cmdstan-2.37.0. The release
tarball hash and installation logs are preserved under
docs/benchmarks/artifacts/stan-transplant-20261027/install-01/. The
[subsequent code audit](../plans/bayesfilter-stan-adaptation-code-audit-result-2026-10-27.md)
rebuilds the oracle from the current C++ test driver and the unmodified Stan
headers. It checks the actual step-search control flow, streaming covariance
and singleton windows, and replays actual warm-up positions and acceptance
statistics through native adapters. Initial/boundary epsilon searches are
tested independently; supplied search outcomes in the replay isolate adapter
ordering from different random-number generators.

The initial dual-averaging shrinkage center follows the Stan service order:
it uses the supplied epsilon before the first reasonable-step search. At
metric boundaries it uses the new searched epsilon. NUTS adaptation consumes
the average trajectory Metropolis probability, not the frequency with which
the selected position changes. TFP drops the acceptance contribution of a
divergent leaf while retaining its leapfrog in the denominator. Stan adds
the leaf's probability; at the configured threshold of 1000, that omitted
probability underflows to zero in FP64, so this distinction does not alter
the adaptation statistic for finite energies at this threshold. This
equivalence must be reassessed if the threshold changes. Their complete
transitions and random streams differ.

This is Stan-style adaptation around stock TFP NUTS, not a translation of
Stan's C++ tree builder. Tree expansion, U-turn checks, multinomial selection,
random-number generation and batch scheduling can affect paths and work.
The benchmark's scalar layout uses independent compiled chains. The batched
layout retains separate adaptation state but may execute masked dense work
while other chains are still active.

For warm-up below 20 transitions there is no metric adaptation, following
Stan's documented tiny-budget behavior. For larger warm-up, the configured
buffers and first window must fit. The transplant explicitly rejects an
undersized schedule rather than claiming to reproduce the release's
short-budget fallback. Supply fitting buffers for mechanics tests. The step
search has a 2048-iteration safety bound after the initial direction probe;
exhaustion or an invalid epsilon marks execution invalid. The bound exceeds
the number of doublings/halvings across the FP64 range below Stan's 1e7 cap.
An input epsilon already above 1e7 skips the search, matching Stan; crossing
that bound during a search fails. XLA may flush subnormal numbers earlier
than native Stan, so underflow failures need not have identical probe counts.

The supported scope is positive warm-up with positive fitting windows,
identity initial metric, a shared supplied starting epsilon, FP64, and adapted
diagonal or dense Euclidean metrics. This does not implement Stan's unit-metric
mode, supplied initial metrics, adaptation-disabled/zero-warm-up entry point,
zero-length buffers, automatic short-budget rescaling, step-size jitter,
parameter constraints/initialization or its NUTS tree builder. Unsupported
configuration choices are rejected or absent from the interface. Frozen
sampling is available separately. Diagonal adaptation currently retains dense
matrix storage and factorization, so it does not have Stan's O(d) diagonal
storage or arithmetic. This is a performance limitation, not a different
diagonal metric.

All repeated numerical work is inside stable-signature XLA functions:
initial epsilon search, gradients, warm-up, Welford updates, metric
factorization, adaptation resets and retained sampling. HLO and graph checks
cover the enclosing functions. Host orchestration, native subprocess launch,
serialization and posterior diagnostics are outside these timed numerical
calls. No pfor, Python callbacks or eager sampler fallback is used.

## Running the comparison

Use the existing tfgpu environment; no additional Python package is required.
Build CmdStan in the isolated directory with make -j4 build, then build the
model using its absolute path as the make target. The executable model is
docs/benchmarks/stan_targets. Native Stan is an external C++ reference and is
not presented as an XLA implementation.

Example CPU benchmark command after the checks and layout calibration:

    /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
      docs/benchmarks/benchmark_stan_transplant.py \
      --device cpu --layout scalar --mode compare \
      --warmup 1000 --draws 2000 --depth 10 \
      --seeds 20261027,20261028,20261029 \
      --output docs/benchmarks/artifacts/stan-transplant-20261027/comparison-NEW

Always use a fresh output directory and the measured layout selected by the
current calibration. CPU runs hide GPU before importing TensorFlow. GPU runs
require trusted execution and verified memory growth before initialization.
The benchmark records dtype FP64, TF32 off, placement, memory policy, source
hashes, command, seeds, warm-up, samples, telemetry and costs. Each native
chain saves warm-up and writes 17-digit CSV output.

The native and TensorFlow target densities and gradients are checked at
identical points before comparisons. Common posterior checks use original
coordinates and analytic standard-normal reference coordinates, including
second moments. Samples from tuning and discarded compilation executions
never enter posterior results. Starts drawn from the reference distribution
make this an intentionally favorable initialization study; it does not
establish robust initialization from arbitrary points.

## Interpreting results

Show numerical and posterior checks beside timing. Report rejected failures,
tree saturation and native divergence counts, but do not use native energy
flags alone to choose a smaller epsilon or reject a tuning candidate. A bad
retained state/cache invalidates execution. Failure to explore the centered
funnel is evidence about that configured sampler and budget, not proof that
the NUTS direction is invalid.

Warm-up plus sampling time includes initial and boundary searches in the
TensorFlow function. Native process wall time additionally includes startup
and CSV output; its printed warm-up time omits initial epsilon search. Both
numbers should be shown, together with TensorFlow serialization and common
diagnostic costs. First-call times include compilation and a discarded full
execution; they are not pure compiler time. Build and process setup costs
belong in cold-start accounting and must not disappear from the report.

This campaign checks a fixed retained budget. It does not measure a sequential
time-to-precision stopping rule. With three repetitions, timing differences
and ESS rates remain limited evidence; posterior failures forbid a valid
efficiency ranking. A broader HMC comparison must use the existing
[fixed-HMC guide](hmc-tuning-interface.md), charge all candidate preparation,
measurement, verification and separate selection, and freeze choices before
final evaluation. The earlier finite-grid HMC pilot does not meet that protocol.
