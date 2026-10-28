# Implementation review: TF models inside Stan GPU/XLA sampling

This is a skeptical self-review, not an independent reviewer endorsement.
The reviewed plan is `bayesfilter-stan-gpu-tf-master-2026-10-28.md`.

## Source and implementation

The direct port's tree, stopping criteria, progressive multinomial selection,
leapfrog and adaptation were inspected against installed Stan 2.37 sources.
In particular, the initial dual-averaging center is set before initial step-size
search in Stan services, and reset after metric updates; the port preserves
that sequence. This task does not duplicate those numerical implementations.

The shared factory now constructs its kernel while tracing each lifecycle
function, binding model data from explicit TensorSpecs. Existing callers retain
their positional signatures. The injected kernel changes only RNG graph
construction; the same stateless Philox operations receive the same per-chain
keys/counters and the same stopped-chain masks. Supplied oracle tapes retain the
original behavior. No NumPy runtime, pfor, Python callback, native custom call
or CPU model evaluation is introduced.

The full fit calls compiled initialization, warmup and retained sampling from
one enclosing explicit-signature function. XLA sees the model and position
gradient. Independent chain rows and deterministic targets remain caller
obligations; arbitrary row-coupled or unsupported TF functions are not valid.
Invalid numerical inputs produce explicit invalid flags, which the host must
check. Differentiating through discrete acceptance decisions is not supported.

## Findings and repairs

| Finding | Resolution |
| --- | --- |
| Earlier factory captured target data at construction | Explicit numeric/boolean model inputs are appended to every lifecycle signature |
| Per-chain RNG used a runtime serial map | Optional static graph expansion preserves primitive random streams; compare compiler cost and full fits |
| Initial plan could confuse XLA compatibility with one-kernel fusion | Inspect HLO, retain fusion counts, explicitly allow multiple kernels/GPU library calls |
| A successful GPU output does not prove device-only model evaluation | Native-sampler calls/PyFunc rejected; target gradient must appear inside loop HLO |
| TF profiling changes command-buffer execution | Pinned TF 2.20 source confirms fallback; report observed fallback copies separately from normal execution |
| No-copy profiler assertion would reject valid command buffers | Retire that invalid diagnostic; preserve failed runs and use non-profiled runtime logs without claiming zero copies |
| Long CPU adaptive trajectories differ between RNG graph layouts | Preserve differences; require common-state transition checks and avoid equal-work speed claims there |
| First master scoped budget to one invocation | Aggregate worker wall and attempts across sibling manifests; preserve prior charges |
| Worker snapshots initially omitted imported numerical helpers | Snapshot helper/diagnostic dependencies and installed-source anchors in final run |
| Native subprocess startup/CSV differs from TF numerical timing | Report that scope explicitly; add in-process sequential Stan baseline for supported targets |

## Verification already observed

132 CPU tests passed, including unmodified-native transition, initialization,
step-size search and adaptation checks plus new model/RNG/continuation tests.
CPU and GPU validation each passed 182 native transition cases with exact
discrete choices. Full-fit TF regression input mutation changed draws without
retracing; analytical gradients matched; enclosing producer/consumer composition
agreed within FP64 tolerance. Terminal measurements and unresolved limitations
belong in the result note after the campaign finishes.

## Terminal review

All four benchmark targets completed on CPU/GPU, including a clean final GPU
command-buffer run. The final controlled CPU and GPU comparisons each passed
96 common-state transitions exactly. GPU retained draws also matched exactly
across map/static layouts and ordinary/command-buffer execution. Non-profiled
runtime logs confirm command-buffer use. The 36 saved-draw diagnostic cases
are preserved; Gaussian, banana and regression GPU cases passed the declared
screen, while every centered-funnel case failed. CPU static banana and
CPU/native regression also failed parts of the screen. No posterior-efficiency
ranking is accepted.

The campaign used 11 attempts and 1990.90 of 2400 worker seconds. The result
and compact evidence record preserve commands, source snapshots/hashes,
failed attempts, timings, diagnostic details and numerical limits. The
unexplained first command-buffer worker failure remains a robustness limitation.
Long CPU trajectory differences remain documented; their first perturbations
were roundoff-sized but the full causal explanation is not proved. CPU/GPU
fixture generation also differs by small roundoff and is disclosed.

Review conclusion: the supported standalone TF/GPU integration passes its
engineering checks. Static RNG expansion and command buffers have favorable
descriptive GPU timings but increase compilation cost, and native Stan remains
faster on the small fixtures. Do not promote a CPU default, a universal speed
claim, full Stan-language compatibility, zero host transfers or posterior
validity from this evidence. See the
[terminal result](bayesfilter-stan-gpu-tf-result-2026-10-28.md).
