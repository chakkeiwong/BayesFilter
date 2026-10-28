# Native Stan and Walnutpie XLA bridge: completed integration study

The native C++ bridge works on CPU XLA and GPU XLA with host staging. The
final master program completed successfully in 27.12 seconds. For both
engines, every tested native/CPU-XLA/GPU-XLA fit has byte-identical draws and
adapted metadata. A GPU producer/custom-call/consumer test also agrees exactly
with the native calculation. The sampler continues to run on CPU inside the
GPU call.

Plan: `docs/plans/bayesfilter-native-stan-walnutpie-master-2026-10-28.md`.
Final evidence: `docs/benchmarks/artifacts/native-stan-walnutpie-20261028/master-02/`.
Entrypoint: `docs/benchmarks/native_sampler_bridge/master.py`.
Usage and API: `docs/benchmarks/native_sampler_bridge/README.md`.

## What was executed

The final invocation was:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true timeout 600 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/native_sampler_bridge/master.py \
  --skip-build --device both --gpu 0 \
  --output docs/benchmarks/artifacts/native-stan-walnutpie-20261028/master-02
```

GPU processes used trusted elevated execution on GPU 0, an RTX 4080 SUPER,
and verified memory growth before initialization. CPU processes explicitly
hid GPUs. TensorFlow 2.20.0 compiled the enclosing graphs. Stan 2.37.0 and
Walnutpie commit 5710b0334ba69cf0caa35fc02a2338b61d756953 supplied native
FP64 sampling. TensorFlow's TF32 flag was recorded as true; native FP64 CPU
sampler calculations do not use TF32.

All four chains run sequentially in the same process. Initial positions are
supplied, seeds are 101/202/303/404, and each chain has 500 warmup plus 500
retained transitions. The three benchmark targets each have dimension 10.
Stan uses depth 10 and delta .8; Walnutpie uses upstream initialization and
adaptation with five trajectory doublings, five halvings and error bound .5.
The question is overhead relative to the same native engine/configuration,
not an optimally tuned cross-algorithm comparison.

## Validation

- Four analytic target value/gradient checks have maximum absolute error zero
  at the tested point: Gaussian, nontrivial transformed Gaussian, banana and
  centered funnel. These are focused construction checks, not exhaustive
  numerical analysis.
- The supplied-start Stan service exactly matches the native CmdStan
  executable for 100 warmup and 100 retained Gaussian draws, including all
  recorded sampler columns.
- The separate Walnutpie executable uses its upstream high-level API and an
  independent analytic Gaussian gradient. Fixed-count draws and adapted
  parameters agree with the bridge to 6.67e-15 maximum absolute error.
- Eight CPU smoke fits cover both engines and all four target families.
  Six benchmark fits per execution route preserve native bytes, replay
  deterministically, change with seeds, retain finite states and trace once
  per static configuration. CPU/GPU output hashes agree across processes.
- Six invalid-input cases per route propagate TensorFlow runtime errors.
  Four contract tests cover static rejection, native multithreaded callers,
  compiled consumers/dynamic seeds and invalid native input. All pass.
- GPU composition tests for both engines exercise dynamic preceding GPU
  arithmetic, non-diagonal input layout, the host-staged call and subsequent
  GPU transpose/scaling. Results match native exactly for two seed vectors.
  Optimized HLO contains `BfNativeSamplerV1`; outputs are on GPU:0.

GPU benchmark peak TensorFlow allocator usage was 551,168 bytes. Peak process
RSS was 432,128 KiB in the CPU benchmark and 732,980 KiB in the GPU benchmark,
including framework/compiler overhead. These measurements describe these small
fits, not general model memory requirements.

## Timing results

Milliseconds per complete four-chain fit, including adaptation and output
materialization; medians of three alternating-order measurements. Compilation
is excluded from these steady-state times. Each process also measures its own
direct-native baseline, so CPU and GPU native columns are shown separately.

| Engine / target | Native in CPU process | CPU XLA | Native in GPU process | GPU XLA | GPU/native ratio |
|---|---:|---:|---:|---:|---:|
| Stan / Gaussian10 | 38.118 | 37.057 | 36.092 | 36.921 | 1.0230 |
| Stan / banana10 | 55.537 | 55.916 | 55.638 | 56.683 | 1.0188 |
| Stan / centered funnel10 | 188.782 | 189.019 | 190.519 | 191.331 | 1.0043 |
| Walnutpie / Gaussian10 | 18.954 | 19.367 | 18.790 | 19.618 | 1.0441 |
| Walnutpie / banana10 | 30.772 | 31.257 | 30.486 | 31.173 | 1.0225 |
| Walnutpie / centered funnel10 | 36.233 | 36.650 | 36.249 | 37.099 | 1.0235 |

GPU staging adds approximately 0.69–1.05 ms in this final run, or 0.43–4.41%
relative to its matched native baseline. CPU overhead is near the timing
noise level for these fits. The negative apparent overhead for Gaussian Stan
is measurement noise, not a wrapper speedup. Earlier retained executions
show some scheduling variation, including a larger funnel timing difference.
No confidence interval or statistically supported performance ranking is
claimed from three repetitions.

First-call times, including graph tracing/compilation and a complete fit, were
26.66–197.97 ms on CPU and 112.27–295.61 ms on GPU. These are combined
first-call costs, not isolated compiler-only times. Build compilation costs
and compiler commands are in the build manifests.

The direct-native baseline includes Python/ctypes dispatch and allocation.
The XLA route includes TensorFlow dispatch, allocation and synchronized output
materialization; GPU times include staging in both directions. Python list
conversion and output comparison occur outside timed intervals. There is no
claim about multicore scaling or about TFP-versus-native speed under these
different short workloads.

## Sampler diagnostics

The following diagnostics are explanatory. They were computed from retained
draws only, using the repository's rank-normalized split/folded R-hat and
bulk/tail ESS routines in an explicit CPU post-run reporting exception.
CPU and GPU draws are identical, so the same diagnostics apply to both.

| Engine / target | Maximum R-hat | Minimum bulk ESS | Minimum tail ESS | Stan divergences |
|---|---:|---:|---:|---:|
| Stan / Gaussian10 | 1.0070 | 3835.9 | 1178.5 | 0 |
| Stan / banana10 | 1.0060 | 1583.5 | 1115.4 | 0 |
| Stan / centered funnel10 | 1.1681 | 20.7 | 35.1 | 3 |
| Walnutpie / Gaussian10 | 1.0073 | 2696.1 | 1175.6 | unavailable |
| Walnutpie / banana10 | 1.0071 | 850.7 | 728.4 | unavailable |
| Walnutpie / centered funnel10 | 1.4260 | 8.6 | 24.8 | unavailable |

Bulk ESS can exceed the draw count for antithetic chains; the values above
are the diagnostic routine's actual outputs. No artificial cap is applied.

For the centered funnel, the exact marginal probability of v below -3 is
Phi(-1), approximately 15.87%. Stan visited this region in 5.3% of retained
draws; Walnutpie visited it in 14.75%. Walnutpie's neck visits are descriptively
encouraging, but its maximum R-hat is 1.426 and minimum bulk ESS only 8.6.
Both short runs have poor mixing. The faster Walnutpie time is not evidence
of better converged-posterior efficiency. Missing Walnutpie divergence
telemetry does not mean zero divergences.

## Terminal review and decisions

The final review checked native source use, initialization, RNG identity,
warmup exclusion, static layouts, stream synchronization, errors, replay,
timing boundaries, manifests and comparison fairness. It is a self-review,
not an independent certification of the upstream sampler mathematics.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Accept standalone CPU bridge | Exact native arrays and native oracles pass | No unresolved transport/target-construction veto in tests | Other models and TF versions untested | Reuse for these compiled models | Public/default readiness |
| Accept standalone GPU staging | Exact arrays and compiled producer/consumer pass | Growth, HLO, placement, errors pass | Blocking host work and larger transfers | Use when integration with a GPU graph is needed | GPU-native sampling or a GPU speedup |
| Retain Walnutpie as a research candidate | Native source and oracle checks pass | No observed bridge defect; telemetry incomplete | Funnel mixing and target-specific tuning | Design a separate replicated posterior-efficiency study if requested | Superiority over Stan |
| Reject funnel posterior promotion from this run | Poor R-hat/ESS in both engines | Stan divergences; missing Walnutpie diagnostic fields | Short chains, one seed set and tuning baselines | Longer independently seeded validation with a declared criterion | Rejection of the bridge or local-adaptation research direction |

| Inference status | Finding |
|---|---|
| Hard veto screen | All bridge checks pass; funnel results cannot support convergence claims |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, overhead, neck visitation, ESS and R-hat |
| Default-readiness | Not evaluated; standalone exception only |
| Next evidence needed | Target-specific tuning, repeated seed sets, posterior agreement and uncertainty-aware efficiency comparison |

Strongest alternative explanation for apparent performance differences is
unequal trajectory work/tuning across engines and ordinary timing noise across
processes. A target-gradient mismatch, native replay disagreement or broken
GPU composition would overturn bridge acceptance. The weakest evidence for
sampler quality is the short centered-funnel run. Its failure is a sampling
diagnostic failure, not a transport failure, because wrapped and native arrays
are identical. It does not reject the target or the locally adaptive method.

## Repair and provenance record

The first build needed a generated-model namespace correction. The first
CmdStan comparison (`cpu-smoke-01`) failed because radius-zero initialization
skipped native RNG draws. Restoring CmdStan radius 2 repaired exact replay.
The next source review found missing Walnutpie gradient-based mass and
initial step search; adding the actual upstream methods repaired this omission.
Earlier smokes remain transport evidence for their recorded implementation.
No performance claim relies on them.

Timing review removed asymmetric Python list conversion before the benchmark.
The first master run passed; terminal review then added the GPU composition
test and the final master run passed. These repairs kept model, comparison,
hardware and evidence criteria unchanged. All run directories were preserved.
The campaign used twelve validation/benchmark worker launches; tiny contract
and GPU-composition tests are additional engineering checks. Recorded worker
time remains far below the 20-minute sampling budget.
The final inventory contains exactly twelve validation/benchmark manifests.
Eleven completed processes record 51.81 seconds in total. The first failed
CPU smoke did not record an end time; charging its entire 180-second timeout
gives a conservative total below 232 seconds, leaving over 968 seconds of
the sampling budget unused. The tiny contract/composition checks do not
support performance or posterior claims.

Each master directory preserves source snapshots and command/exit manifests.
The repository HEAD recorded at execution is
`e38348bee2b321d695181ec74e7486a611f67df4`; the implementation is uncommitted,
so source snapshots and SHA-256 hashes identify the actual executed code.
Unrelated worktree changes were preserved.
Final read-back verified both benchmark result sets, all binary-output
hashes, twelve runtime error cases, final master exit codes and exact identity
of the executed source snapshots with the current implementation. Python
syntax checks and the targeted whitespace check passed. No new git commit
was created by this task.

Remaining scope limits: only the exposed compiled benchmark targets and
diagonal metrics; sequential chains; no arbitrary TensorFlow target callback;
no exact checkpoint continuation, differentiable sampler, TFP TransitionKernel
or CUDA graph capture. Internal XLA status ABI requires the pinned build and
revalidation after upgrades. Upstream mathematical correctness was not
independently proved by this integration study.
