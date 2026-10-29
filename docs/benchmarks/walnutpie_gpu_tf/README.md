# Walnutpie for TensorFlow models on GPU XLA

This standalone package ports deterministic Walnutpie commit
5710b0334ba69cf0caa35fc02a2338b61d756953 into batch-native TensorFlow, and offers
the WALNUTS randomized two-point (R2P) transition as an explicit alternative.
Initialization, macro refinement, reverse checks, tree building, adaptation and
retained sampling execute inside explicitly signed XLA functions. A caller's
TensorFlow model and its position gradient execute in the same computation.
There is no CPU sampler custom call. XLA may schedule GPU loops from the host;
GPU execution does not imply one fused CUDA kernel or zero scheduling traffic.

This is a TensorFlow implementation compiled to CUDA by XLA, not a modification
that makes arbitrary upstream C++ or Stan-language programs CUDA compilable.
The existing public inference policy is unchanged by this standalone
owner-authorized study. Deterministic Walnutpie remains the package default.

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

For a frozen NeuTra metric, set Config(adapt_mass=False) and initialize with
Initialization(mass='unit') or explicit supplied masses. Warmup then preserves
that metric while adapting the base step and minimum macro work. The existing
default adapt_mass=True retains the pinned WalnutPie metric adaptation.
The target_accept statistic is exp(-abs(deltaH)) at base macro attempts; it
is not the endpoint Metropolis acceptance fraction of fixed-length HMC.

The fixed-count API takes supplied positions. The additional initialization and
run-control APIs below support random starts and adaptive stopping. The initial
step search retains its 64-update safety limit by default; explicitly set
init_search_limit=None for uncapped source behavior. Exhaustion, invalid step or
mass, or nonfinite initial state makes valid false. An uncapped search has no
wall-clock check inside XLA. Callers must check valid before using results.
Rejected macro/reverse proposals are telemetry, not invalid retained states.

## Initialization and run control

Initialization selects supplied, zero or independent normal positions (with a
positive scale), supplied/unit/absolute-score masses, optional geometric mass
averaging across chains, and supplied steps or the pinned fixed-momentum step
search. Supplied masses have shape [chains, dimension] and contain diagonal
mass entries, not inverse masses. Steps have shape [chains]. Every input has a
stable explicit signature; inactive placeholder operands are ignored only when
the chosen mode replaces them. These are unconstrained positions; they do not
implement Stan's constrained parameter initialization.

Using the model, q and seeds from the example above:

~~~python
from docs.benchmarks.walnutpie_gpu_tf import (
    Initialization, make_initializers, RunControl, make_controllers, drive_chunks)

signature = [tf.TensorSpec([10], tf.float64)]
initializer = make_initializers(tf, log_density, config,
    Initialization(positions='normal', scale=1., mass='score',
                   average_masses=True, adapt_step=True), signature)
control = make_controllers(tf, log_density, config, RunControl(), signature)
with tf.device('/GPU:0'):
    data = tf.zeros([10], tf.float64)
    state = initializer.initialize(q, seeds, tf.ones_like(q),
                                  tf.fill([4], tf.constant(.1, tf.float64)), data)
    assert bool(tf.reduce_all(state.valid).numpy())
    warm = drive_chunks(control.warmup_chunk, control.begin(state), data)
    assert warm['reason'] in ('criterion', 'maximum')
    # begin resets the phase's count and log-density moments, not sampler tuning.
    sample = drive_chunks(control.sample_chunk,
                          control.begin(warm['state'].sampler), data)
assert sample['reason'] in ('criterion', 'maximum')
retained = tf.concat([tf.boolean_mask(x.draws, x.mask)
                      for x in sample['chunks']], axis=0)
~~~

Warmup and sampling chunks are both returned; only sampling draws belong in
posterior calculations. Full compiled control.warmup/control.sample are also
available. Fixed-capacity outputs have a mask; padded zeros are not draws.
ControlState preserves all sampler state, count, and online log-density moments.
Resume a cancelled phase by passing its ControlState to the same chunk function;
do not call begin on that checkpoint. Targets/data/configuration must remain
unchanged within a phase. Invalid termination requires checking the sampler
valid flag and rejecting that run, rather than treating its mask as certification.

Warmup stops on the upstream relative step/mass agreement criteria. Sampling
uses the exact upstream legacy statistic
sqrt(1 + sample_variance(chain_logp_means)/mean(chain_logp_sample_variances)).
This log-density-only statistic does not establish parameter convergence or
posterior accuracy. Stop reasons are criterion, maximum, invalid, deadline or
cancelled; reaching maximum is not convergence. Geometric means are used in the
warmup statistic but final tuning remains per-chain, matching the pinned C++
controller's behavior of not applying its pooled return value to samplers.

Numerical statistics and stopping decisions are inside XLA. drive_chunks handles
optional deadline_seconds, cancelled() and on_chunk(result, count) on the host,
between completed chunks. The deadline includes compilation and can overrun by
one chunk; it is not a hard timeout or an interrupt inside a device graph.
Chains advance synchronously, so stop counts and scheduling are not identical
to C++ asynchronous workers. Callbacks must not mutate sampler/data state.

## R2P transition

Select Config(..., variant='r2p'). The same make_functions, initialization and
run-control interfaces apply. Forward refinement chooses exponent c; random
selection uses 2**c microsteps with probability 2/3, or 2**(c+1) with probability
1/3. Per-macro jitter defaults to +/-20%. Reverse probabilities enter cumulative
log weights at both tree endpoints. Zero reverse support gives exact zero weight
while retaining the endpoint for geometry. Physical energy stays separate from
these probability corrections. The selector uses the optimized endpoint energy
criterion of paper Remark 13 and checks intermediate finiteness.

The pinned author critical search falls back to maxC even when the threshold
was not met; the finer branch can use maxC+1. R2PKernel.r2p_macro returns
threshold_met, forward/backward exponents, chosen exponent and work counts for
diagnostics. A cap hit is not proof that the energy threshold was achieved.

R2P with positive warmup uses a local combination with Walnutpie Adam/metric
adaptation, always holding minimum microsteps at one. This is not the paper's
Appendix C quantile tuning. To use independently supplied fixed tuning, choose
warmup=0 and Initialization(mass='supplied', adapt_step=False), then call sample
on the initialized state. No claim of reproducing the paper's long funnel
experiments or of faster sampling follows from these engineering tests.

R2P sources are the JMLR 27 paper 25-1452, sections 3.1–3.4 and Appendix A, and
author repository bob-carpenter/walnuts at
895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a. The author MIT notice is preserved in
[WALNUTS-R2P-LICENSE](WALNUTS-R2P-LICENSE). The author macro integrator is tested
unmodified. Its whole-tree code appears to omit a cumulative correction in the
backward two-leaf branch (WALNUTSpy/WALNUTS.py, lines 453–462). Our tree includes
that correction, following equations (13)–(14), with an independent recursive
paper-equation reference and a negative-control test. This is not bitwise
whole-chain equivalence with the author Python implementation.

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

Initialization/control/R2P qualification has its own bounded gap_master driver
(3600 cumulative worker seconds, at most 18 launches). Use a fresh output path:

~~~bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
python -m docs.benchmarks.walnutpie_gpu_tf.gap_master \
  --phase validate --device gpu --gpu 1 \
  --output docs/benchmarks/artifacts/walnutpie-gaps-20261029/new-run
~~~

Other phases are tests (CPU-only) and deterministic (the saved 528-case C++
reference). CPU validation requires CUDA_VISIBLE_DEVICES=-1. The validate phase
needs the locally pinned C++ headers, Eigen, author Python source and paper;
missing sources fail instead of silently selecting another baseline. The exact
source paths and acquisition pins are in
[the gap-closure plan](../../plans/bayesfilter-walnutpie-gap-closure-2026-10-29.md);
[the result note](../../plans/bayesfilter-walnutpie-gap-closure-result-2026-10-29.md)
records the final qualified revisions.


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
# Author funnel reproduction

The copied author experiments, source hashes and MIT license are in
author_reference/. Their funnel10 has **11 total dimensions**; the older
local benchmark used 10. The bounded funnel_master driver runs the author's
quantile tuning as an independent CPU reference and transfers frozen h/delta
into the fully XLA-compiled R2P sampling kernel with identity mass. Appendix C
tuning itself is not yet a native TensorFlow implementation.

See docs/plans/bayesfilter-walnuts-author-funnel-2026-10-29.md for the reviewed
comparison, budget and diagnostics. Run the module
docs.benchmarks.walnutpie_gpu_tf.funnel_master with --phase tune --device cpu
--engine author-original --dimension 11 and a fresh --output directory.
Sampling uses --phase sample --engine tf --device gpu --gpu 1, an explicit
--tuning pointing at tuning.json, and a fresh --output directory. Reference
engines are author-original and author-corrected, on CPU. The corrected arm
inserts the cumulative weight required at the second backward leaf.

GPU launches require trusted execution; the driver configures memory growth
before framework imports. Timing includes a result-tensor device barrier.
The copied million-iteration scripts are archival reference files, not the
bounded launch entry points.

Completed comparison: docs/plans/bayesfilter-walnuts-author-funnel-result-2026-10-29.md.
Machine-readable summary: author-funnel-evidence-20261029.json. At 8192 retained
draws per chain, both author and TF R2P explore the neck, but all compared arms
fail the combined R-hat/ESS screen. The exact v=-30 transient remains incomplete.

## Supplied initial mass: fixed or adaptive

`make_supplied_mass_functions` provides complete initialize/warmup/sample/fit
graphs with a caller-supplied diagonal mass. Mass means momentum covariance
**M**, not inverse mass: the checkpoint stores `inverse_mass = 1 / M`.

```python
from docs.benchmarks.walnutpie_gpu_tf import (
    Config, make_supplied_mass_functions, validate_mass,
)

cfg = Config(chains=4, dimension=11, adapt_mass=False)
functions = make_supplied_mass_functions(tf, model, cfg)
mass = validate_mass(tf, [1.0] * 11, cfg)  # shared diagonal -> [4, 11]
state = functions.initialize(q, seeds, mass, epsilon)
warm = functions.warmup(state)
draws = functions.sample(warm.state)
```

Set `adapt_mass=True` to initialize the existing upstream position/score
moment adaptation from the supplied mass. Set it to `False` to keep that mass
unchanged through warmup; step-size adaptation still runs. Sampling always
freezes both mass and step size. `adapt_step=False` only skips the *initial*
step search; it does not disable step adaptation during warmup. Every caller
must check the returned validity flag before using results. For runtime model
data, pass `model_signature` to the factory and the corresponding `*model_data`
after the usual operands in each call.

`make_dense_mass_functions` accepts a shared or chain-specific symmetric
positive-definite matrix after `validate_mass(..., dense=True)`. Its compiled
initialize/fit operands are `(q, seeds, mass[C,D,D], epsilon[C], *model_data)`.
It uses fixed coordinates `y=L.T @ q` for `M=L @ L.T`, with the model, gradient,
Jacobian and physical-coordinate U-turn calculation inside XLA. Returned
draws and log densities use physical coordinates. The checkpoint contains
the basis, latent sampler state and validation flags. `physical_positions`
and `physical_mass` inspect the state in the original coordinates.

For dense input, `adapt_mass=True` adapts diagonal masses in that fixed basis:
`M_effective = L @ diag(m_internal) @ L.T`. It **does not learn an unrestricted
dense covariance matrix**. With `adapt_mass=False`, the physical mass remains
the supplied matrix. The private kernel extension changes only the U-turn
calculation; all integrator/tree/adaptation mechanics share the existing code.

Plans and validation: `docs/plans/bayesfilter-walnutpie-supplied-mass-2026-10-29.md`,
`tests/test_walnutpie_supplied_mass.py`. This standalone API does not change
BayesFilter's public inference defaults.

## Experimental CPU-tree / GPU-integration execution

The standalone `hybrid.make_hybrid` prototype pauses between integration leaves:
CPU/XLA manages tree merging, selection and stopping; GPU/XLA executes the
existing integration leaf, including model gradients and local refinement.
Model data remain resident on GPU. Python dispatches compiled functions and
copies packed buffers; it does not evaluate the target or numerical tree.
This is separately compiled heterogeneous execution, not one fused XLA graph.
It supports fixed-kernel diagnostic sampling and is not exported as a default.

`hybrid_benchmark` compares it with complete CPU/XLA and GPU/XLA execution on
a synthetic nonlinear neural regression. The reviewed experiment is
`docs/plans/bayesfilter-walnutpie-hybrid-large-model-2026-10-29.md`; replay tests
are in `tests/test_walnutpie_hybrid.py`. Results and hardware-specific limits
are recorded under `docs/benchmarks/artifacts/walnutpie-hybrid-large-20261029-01/`.
Use the complete measured workload to choose placement: the cheap funnel's CPU
timings do not establish an advantage for CPU tree management on costly models.
