# Walnutpie for TensorFlow models on GPU XLA

This standalone package ports deterministic Walnutpie commit
5710b0334ba69cf0caa35fc02a2338b61d756953 into batch-native TensorFlow.
Initialization, macro refinement, reverse checks, tree building, adaptation and
retained sampling execute inside explicitly signed XLA functions. A caller's
TensorFlow model and its position gradient execute in the same computation.
There is no CPU sampler custom call. XLA may schedule GPU loops from the host;
GPU execution does not imply one fused CUDA kernel or zero scheduling traffic.

This is a TensorFlow implementation compiled to CUDA by XLA, not a modification
that makes arbitrary upstream C++ or Stan-language programs CUDA compilable.
It is not the paper's R2P variant. The existing public inference policy is
unchanged by this standalone owner-authorized study.

## Model and state interface

Set memory growth before importing TensorFlow, configure and verify growth on
all visible GPUs before initializing a device, then build functions:

~~~python
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    assert tf.config.experimental.get_memory_growth(gpu)
from docs.benchmarks.walnutpie_gpu_tf import Config, make_functions

def log_density(q, location):
    return -.5 * tf.reduce_sum((q-location)**2, axis=-1)

config = Config(chains=4, dimension=10)
functions = make_functions(tf, log_density, config,
    [tf.TensorSpec([10], tf.float64)])
with tf.device('/GPU:0'):
    q = tf.ones([4, 10], tf.float64) * .2
    seeds = tf.constant([[101, 0], [102, 0], [103, 0], [104, 0]], tf.int32)
    fit = functions.fit(q, seeds, tf.zeros([10], tf.float64))
assert bool(tf.reduce_all(fit.valid).numpy())
retained = fit.sampling.draws  # [draws, chains, dimension]; excludes warmup
~~~

The model must return FP64 [chains] log densities from FP64 [chains, dimension]
positions, with independent rows and XLA-supported forward/gradient operations.
Include constraint-transform log Jacobians. Runtime model data are explicit
operands, not frozen Python captures. External Python callbacks are unsupported.
No derivative through the accept/reject algorithm is promised. Validate each
model's row isolation and mathematical target separately.

Functions also expose initialize(q,seeds,*data), warmup(state,*data),
sample(state,*data), and value_gradient(q,*data). State contains caches, seeds,
Adam and discounted moments. Sample continuation freezes tuning; warmup
continuation retains the adaptation iteration. Different data require fresh
initialization and adaptation. Seeds use distinct nonnegative int32 keys and
counters; Philox does not reproduce C++ mt19937 streams.

The supplied position is required. The upstream automatic random-start and
wall-clock/multichain stopping controllers are outside this fixed-count API.
Initialization's otherwise-unbounded step search has a 64-update safety limit;
exhaustion or nonfinite initial state makes valid false. This is a documented
engineering extension. Callers must check valid before using results. Rejected
macro/reverse proposals are telemetry, not invalid retained states.

## Source fidelity

- walnuts.hpp: endpoint absolute energy threshold; deterministic step halving;
  coarser reverse rejection; displacement U-turns; Barker internal selection;
  Metropolis outer selection before combined-turn stopping.
- adam.hpp: update once per base macro attempt, including failed proposals.
- adaptive_walnuts.hpp and online_moments.hpp: discounted draw/score moments,
  geometric inverse-mass estimate, and positive lround minimum-step adaptation.
- config.hpp and util.hpp: absolute-score initial mass, fixed-momentum initial
  step search, and pinned numerical defaults.

Binary-carry tree storage replaces recursion while preserving depth-first
random consumption and selection. Runtime arrays have O(chains × dimension ×
depth) tree storage. Batched lanes execute until the slowest active lane stops;
reported logical gradient evaluations exclude work on masked inactive lanes.

## Reproduce

For compatible Linux x86-64 systems, a tracked 243.2 KiB C++ executable is
available in [prebuilt/](prebuilt/README.md), with exact sources, checksums,
licenses and platform requirements. Pass it with `--oracle` to a worker or to
`master.py --phase validate` / `--phase benchmark` to skip the native build.
The original master's `build`, `qualify` and `all` phases rebuild the supplied
oracle path; do not point those phases at the tracked binary. This does not
cache TensorFlow model XLA compilation.

The master enforces 3600 cumulative worker seconds and 16 launches across sibling
run directories, snapshots source/header hashes, records failed attempts, and
uses fresh output directories. GPU commands require the repository's trusted
execution permissions. CPU references intentionally hide GPUs.

~~~bash
python -m docs.benchmarks.walnutpie_gpu_tf.master \
  --phase all --device both --gpu 1 --command-buffer \
  --output docs/benchmarks/artifacts/walnutpie-gpu-tf-20261028/new-run
~~~

The oracle compiles unmodified upstream headers with -O3. Test-only Random
specialization supplies shared deterministic draws, allowing branch and
adaptation comparison independently of RNG-library differences. Native timing
uses ordinary mt19937_64 and analytic model gradients, including serial and
one-thread-per-chain execution. First-call/compilation time is separate from
seven synchronized steady-state repetitions. Do not convert short-run timing
or convergence summaries into sampler-efficiency superiority claims.

Master plan and result note are in docs/plans with the prefix
bayesfilter-walnutpie-gpu-tf. The result note, not compile success, establishes
which numerical checks passed and whether any measured speed benefit exists.


## Execution-layout experiments

The default remains fieldwise. Config(execution_layout="packed") stores the
same twelve span fields in one contiguous stack. Config(execution_layout="flat")
uses that stack and one leaf/carry scheduler instead of a nested carry loop.
Both share the numerical macro, reverse, selection, RNG and adaptation code.
They are optional execution layouts, not different tuning protocols. Packed
passed native-oracle GPU qualification and reduced paired complete-fit time by
about 3–7% on five measured workloads. Flat passed correctness checks but was
slower on Gaussian and banana, so it is retained only as an experimental arm.
The original default remains unchanged. See
../../plans/bayesfilter-walnutpie-optimization-result-2026-10-29.md for the
measured decision, uncertainty intervals and limitations.

The optimization campaign uses optimization_master.py with a separate bounded
budget and source snapshots. The layout worker compares equal-target, equal-seed
complete fits with command buffers on for every arm; compilation is separate.
It uses four balanced AB/BA repetitions in each of seven seed blocks and releases
retained output objects between timed calls. Reuse Functions objects for fixed
signatures and keep runtime model operands on the chosen device. Retaining many
fit results can change allocation/command-buffer behavior; measure the intended
consumer pattern explicitly when integrating an application.
