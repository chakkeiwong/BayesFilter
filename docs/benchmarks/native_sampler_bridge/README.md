# Native Stan and Walnutpie in TensorFlow/XLA

This standalone integration runs the actual native C++ samplers from an XLA
custom call. CPU XLA calls the native library directly. GPU XLA copies inputs
to host memory on the supplied CUDA stream, runs the CPU sampler, and copies
the results back to device memory. The graph is XLA compiled; the sampler
itself is ordinary compiled C++ running on CPU.

The implementation is intentionally a complete-fit interface: each call owns
initialization, warmup, sampling, RNG and output buffers. There is no Python
callback, subprocess or CSV writing in the sampling path. Native CmdStan and
the independent Walnutpie executable are used only by the validation harness.

The reviewed program is in
`docs/plans/bayesfilter-native-stan-walnutpie-master-2026-10-28.md`; completed
measurements and terminal review are in
`docs/plans/bayesfilter-native-stan-walnutpie-result-2026-10-28.md`.

## Supported interface

`wrapper.compiled_fit(FitConfig(...), device)` returns a stable-signature
`tf.function(jit_compile=True)`. All numerical inputs use FP64 except seeds.

| Input | Shape | Meaning |
|---|---|---|
| controls | `[4]` | banana bend, initial step, target acceptance, Walnutpie Hamiltonian-error bound |
| starts | `[C,D]` | fully supplied initial positions |
| inverse_transform | `[D,D]` | row-major transform for the Gaussian target; supply identity for other targets |
| seeds | `[C]`, int64 | per-chain seeds in the uint32 range |

`FitConfig` selects engine (`stan` or `walnutpie`), target kind, dimension,
chains, warmup, retained draws, trajectory depth and Walnutpie halvings.
There is one compiled function per static configuration. Supplying another
seed or initial state does not retrace that function.

| Target kind | Compiled model |
|---|---|
| 0 | standard multivariate Gaussian |
| 1 | Gaussian with log density `-.5 * ||inverse_transform @ q||^2` |
| 2 | product banana, with an even number of coordinates |
| 3 | centered funnel: `v ~ Normal(0,3)`, `x | v ~ Normal(0,exp(v/2))` |

The Gaussian transform must be nonsingular for a proper target. The bridge
checks shape and finiteness; it does not prove that supplied model data define
a normalized distribution. Both engines use the same generated
`docs/benchmarks/stan_targets.stan` model and Stan gradients. Regression kind 4
from that model is not exposed by this initial bridge.

Outputs are `rows[C,W+S,D+7]` and `metadata[C,D+4]`.
`rows[:, :W, :]` is warmup and must be discarded from posterior estimates.
Retained parameter draws are `rows[:, W:, 7:]`.

| Row columns | Meaning |
|---|---|
| 0 | unnormalized log density |
| 1 | Stan acceptance statistic; unavailable for Walnutpie |
| 2 | native step size; during Walnutpie warmup this is its reported adaptive step |
| 3–6 | Stan tree depth, leapfrog count, divergence, energy; unavailable for Walnutpie |
| 7 onward | model parameters |

Metadata contains final step size, D diagonal inverse-mass entries, minimum
microsteps, gradient-call count and caught target exceptions. The last three
fields are exported for Walnutpie only. Unavailable diagnostics are **NaN**,
never fabricated zero counts. The Walnutpie gradient count includes
initialization and warmup as well as retained sampling.

## Native adaptation and reproducibility

Stan calls `hmc_nuts_diag_e_adapt` from pinned CmdStan 2.37.0. Its native
dual averaging and window adaptation are retained: gamma .05, kappa .75,
t0 10, windows 75/50/25, no step jitter and no thinning. Delta is an input.
Its native short-warmup adjustment remains active. Initialization radius 2
preserves CmdStan's RNG consumption even when all starts are supplied.

Walnutpie is pinned to commit
`5710b0334ba69cf0caa35fc02a2338b61d756953` of
`https://github.com/flatironinstitute/walnutpie`. We use its unchanged
gradient-based initial mass, initial step search, `AdaptiveWalnuts` and
`WalnutsSampler`. Initialization uses a separate `seed_seq{seed,2}` RNG;
transitions use `seed_seq{seed,chain_index+1}`. This is an explicit bridge
seed convention, not a claim that every Python Walnutpie CLI seed maps to the
same stream: its wrapper also adjusts seeds using its `id` and chain count.

The upstream high-level controller's variable stopping and parallel threads
are replaced by fixed-count, sequential calls to its documented sampler
classes. The same fixed-count configuration matches the upstream high-level
Gaussian oracle to floating-point roundoff. Walnutpie's native warmup is
different from Stan's; it is not described as a Stan adaptation transplant.
Upstream builder defaults are pinned baselines, not target-specific optimal
tuning. In particular, the source uses smoothing for initial mass; its
warmup mass estimator does not use the similarly named smoothing field.

Calls are serialized within the native library to protect Stan's autodiff
arena. Repeating a call with the same inputs reproduces the same fit. There
are no mutable persistent sampler handles. Change the seeds for new fits.
XLA may reuse identical pure calls; implicit stateful RNG behavior is not
part of this interface.

## Build and run

Validated environment: Linux x86-64, g++ 11.4, Python 3.13, TensorFlow 2.20.0,
installed CUDA runtime 12 and RTX 4080 SUPER. Build outputs live under
`.localresources/native_sampler_bridge`; no environment-wide installation
is performed. `build.py` records compiler commands and dependency hashes.

Required local source roots:

```text
.localresources/cmdstan/cmdstan-2.37.0
.localresources/walnutpie-5710b0334ba69cf0caa35fc02a2338b61d756953
```

The downloaded Walnutpie archive is preserved alongside its extracted source
with SHA-256
`5170a0bd5883c666f810dec14fdffbd2d403ea75b12712a3144598d4b881754f`.
Upstream licenses remain with their sources. The build requires C++20 for
Walnutpie and uses a separate C++17 TensorFlow translation unit. The current
combined build links CUDA even when subsequently used for CPU-only fits.

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/native_sampler_bridge/build.py

TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/native_sampler_bridge/master.py \
  --skip-build --device both --gpu 0 \
  --output docs/benchmarks/artifacts/native-stan-walnutpie-20261028/my-new-run
```

GPU runs require the repository's trusted execution permissions. The master
sets device visibility before subprocess imports and verifies memory growth.
It rejects an existing output directory and a stale native build. The master
records source snapshots, command lines, logs, timings and exit codes; each
validation process also saves HLO, binary draws, metadata and a manifest.

The program runs contract tests, native oracles, CPU smoke, GPU graph
composition where requested, the overhead benchmark and post-run diagnostics.
`--device cpu` is an explicit CPU reference mode. Its default is GPU.
The sampling budget is 20 minutes; per-process timeouts and the small fixed
run list bound this program below that budget.

Minimal use after configuring the environment and GPU memory growth:

```python
from docs.benchmarks.native_sampler_bridge.wrapper import FitConfig, compiled_fit
import tensorflow as tf

cfg = FitConfig(engine="stan", kind=0, dimension=10, chains=4,
                warmup=500, draws=500, depth=10)
fit = compiled_fit(cfg, device="/GPU:0")
with tf.device("/GPU:0"):
    rows, metadata = fit(
        tf.constant([0.3, 1.0, 0.8, 0.5], tf.float64),
        tf.zeros([4, 10], tf.float64),
        tf.eye(10, dtype=tf.float64),
        tf.constant([101, 202, 303, 404], tf.int64),
    )
    retained = rows[:, cfg.warmup:, 7:]
```

Those counts illustrate the engineering interface; they do not establish
posterior convergence. Use engine `walnutpie`, depth 5 and initial step .1
to reproduce its benchmark configuration, including native step-size search.

## Limits

The measured bridge reuses these compiled benchmark models. It does not yet
accept arbitrary TensorFlow target functions or arbitrary Stan model modules.
A new compiled model requires an explicit data/initialization/output adapter
and target-gradient checks. Copying GPU buffers does not supply a gradient
bridge between TensorFlow targets and Stan autodiff.

This interface is not a TFP `TransitionKernel`, is not differentiable, and
does not implement exact checkpoint continuation, dense metrics, asynchronous
cancellation or parallel chains. Every invocation performs a fresh full fit.
GPU execution blocks the supplied stream until the CPU fit finishes; no
device-wide barrier is used. CUDA graph capture is not supported or tested.

The custom-call status handling uses TensorFlow's installed internal XLA
header. It passed actual runtime-error propagation checks in this pinned
build, but it is not a stable cross-version ABI. Rebuild and rerun the master
after changing TensorFlow, Stan, Walnutpie, compiler or CUDA versions.

Transport fidelity is established for the tested configurations. Mathematical
correctness of all upstream algorithms, universal posterior convergence,
optimal tuning and broad performance superiority are separate questions.
