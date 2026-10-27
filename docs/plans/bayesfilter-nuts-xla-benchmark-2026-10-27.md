# Standalone NUTS XLA benchmark

## Scope and migration

The owner's 2026-10-27 request reopens NUTS for a simple speed benchmark, with
fully XLA-compiled numerical execution on CPU or GPU chosen by measurement.
This is a narrow exception to the October 10 exclusion: the new standalone
`docs/benchmarks/benchmark_nuts_xla.py` may execute stock TFP NUTS. Retired
entry points, q20, the modern-HMC campaign, public tuners and defaults remain
unchanged. This benchmark issues no canonical tuning/admission artifact.

## Research intent and evidence contract

Question: for simple known targets, how much NUTS time comes from compilation,
adaptation and retained sampling; which XLA device is economical for each
dimension/chain batch? Candidate mechanism: whole-chain compilation and batched
chains, initially using the installed unmodified TFP implementation. Comparator:
the same stock TFP NUTS, target, dtype, seeds, shape, adaptation and tree cap on
the other device. Stock TFP fixed-L HMC is an explicitly untuned contextual
baseline, not evidence of the best achievable HMC performance.

Expected failure: tree control and small vector arithmetic may cost more than
they save on GPU; larger chain batches may be limited by their longest trees.
Primary engineering pass: the *enclosing* warm-up and retained-chain functions
compile with `jit_compile=True`, stable static signatures, saved optimized HLO,
verified requested device and finite retained state/log-density/score.
Performance is measured with a device synchronization token, fresh run seeds,
equal output transfer policy, and repeated warm calls. First call is separately
labeled compilation plus execution; it is not a pure compilation measurement.

Device nomination uses median retained wall time for identical workload shapes;
end-to-end cost includes first call and adaptation. ESS/sec and moment error are
reported alongside raw throughput, never silently substituted for correctness.
A short pilot can nominate a device but cannot establish a statistically
supported sampler ranking or convergence. Confirmed speed claims need separate
selection and validation runs and uncertainty on paired workload measurements.

| Diagnostic | Role | Consequence |
| --- | --- | --- |
| Failed full-chain XLA, wrong device, callback, nonfinite retained cache | Continuation veto for that cell | Repair implementation; do not time a fallback |
| Rank/folded split R-hat, bulk/tail ESS, exact-reference moment error | Promotion veto / descriptive pilot diagnostics | Do not claim posterior qualification from a short run |
| Proposal rejection, native divergence, energy flags, tree cap hits | Explanatory | Preserve counts; never drop rejected draws or force epsilon direction from energy flags |
| CUDA/resource/serialization failure | Repair trigger | New attempt within remaining budget |
| Budget exhaustion | Continuation veto for launches | Preserve partial results and report missing cells |

No NeuTra map, filtering likelihood, default policy, NUTS algorithm change or
production sampler promotion is part of this study. Failure of one target or
device rejects that cell, not the NUTS research direction.

## Targets and exact checks

Use an isotropic Gaussian as the control. For each independent banana pair,
let `z1,z2 ~ N(0,1)`, `x=z1`, `y=z2+b*(z1^2-1)`. The triangular map has
Jacobian determinant one, so log density up to a constant is
`-.5*(x^2 + (y-b*(x^2-1))^2)`. Thus means are zero, `Var(x)=1`,
`Var(y)=1+2*b^2`, `Cov(x,y)=0`, and `E[x^2*y]=2*b`. The inverse coordinates
must have independent standard-normal moments. These identities follow from
`E[z^2]=1`, `E[z^3]=0`, `E[z^4]=3` and independence. Product pairs provide
a higher-dimensional batching probe without introducing an unknown posterior.
The repository already uses this banana definition in
`bayesfilter/testing/inference_validation/targets.py`; b=.5 is its inherited
fixture, not a universal difficult-target setting.

Next targets, only after this harness is valid: rotated anisotropic Gaussian
(conditioning), centered funnel (variable curvature), then a small exact
Kalman-likelihood target from the repository. Banana alone cannot represent
expensive filters or certify those model paths as XLA-compatible.

## Defaults and assumptions audit

| Choice | Provenance/status | Risk and early check |
| --- | --- | --- |
| TensorFlow/TFP | Repository backend; installed versions recorded | Inspect installed NUTS source and compile full chain |
| float64, TF32 disabled | Convenience reference choice | Hardware-sensitive cost; no extrapolation to FP32 |
| b=.5 | Existing banana fixture | Mild curvature; exact inverse/moment test |
| Dimensions 2 then 32; chains 4 then 32 | Convenience shape probes | Small set cannot locate all CPU/GPU crossovers |
| NUTS max depth 6, leapfrog unroll 1 | Bounded pilot hypothesis, stock unroll | Record cap hits; no silent depth reduction for speed |
| epsilon .25, target acceptance .8 | Warm-start hypotheses | Record dual-averaged epsilon and all acceptance counts |
| HMC L=8 | Untuned contextual baseline | No NUTS-vs-optimized-HMC superiority claim |
| Warm-up 128, retained 256, repeats 3 | Small pilot convenience counts | Insufficient for posterior/default promotion; report diagnostics |
| Seeds 20261027 + replicate index | Reproducibility convention | No reuse of one chain as independent replications |
| CPU threads 4 | Bounded convenience resource choice | Record affinity/load; not optimal CPU tuning |
| Max 20 minutes / 8 worker launches / 180 s per worker | Conservative task budget | All failed attempts consume budget; stop rather than expand |

## XLA and source audit

Installed source authority:
`/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow_probability/python/mcmc/nuts.py`.
Its constructor documents `unrolled_leapfrog_steps` as a trajectory-length
multiplier; changing it is *not* a semantics-free compiler optimization.
`one_step` uses tensor while loops, a maximum over active chain lifetimes and
static tree instructions. Preserve multinomial selection, generalized U-turn
checking, RNG splitting and tree validity. Do not implement a custom tree in
this first benchmark. Third-party constructor setup is host-side; all repeated
target, gradient, integrator, tree, RNG, adaptation and sampling work must be
inside the enclosing compiled functions. Python CLI/I/O/timing and untimed
post-run diagnostics are explicitly outside XLA. No pfor or Python callbacks.

Before timing, disable soft placement, save optimized HLO of warm-up and sample
functions, inspect graph library for PyFunc/EagerPyFunc, execute once and verify
output device. Configure GPU growth before import/initialization; CPU workers
hide GPUs before import. GPU workers use trusted permissions and an idle GPU.

## Skeptical pre-execution audit

The old May benchmark used only two chains, a small Gaussian, a fixed epsilon
and fixed HMC length; it cannot determine this machine's best device or compare
optimized statistical efficiency. Reusing its conclusion would be a wrong
baseline. It remains historical. The new device comparison holds target, dtype
and workload fixed, reports actual trajectory work and separates compile cost.
GPU idle status must be checked; other jobs are not stopped. Adaptation uses
acceptance evidence and does not reject a candidate merely for safely rejected
proposal errors. Short-chain ESS is descriptive. Audit passes for bounded
engineering pilots with these limits, not a broad sampler study.

## Execution and preservation

Interpreter: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`.
Versioned output root: `docs/benchmarks/artifacts/nuts-xla-20261027/`. Each
worker requires a fresh directory. Preserve command, git HEAD, hashes of the
benchmark/dependency source, dirty-source status, versions, device/memory
policy, seeds, exact shapes, elapsed wall times, HLO, sample tensors and JSON.
Use CLI `--help` for exact arguments once implemented. Run focused CPU tests,
then paired CPU/GPU pilots at the small shape; expand to the batched shape only
if the remaining budget allows. Record actual commands and decisions in
`bayesfilter-nuts-xla-benchmark-result-2026-10-27.md`.
