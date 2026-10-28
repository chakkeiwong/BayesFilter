# Stan algorithm for GPU TensorFlow models: execution result

The master program was reviewed and executed. The standalone implementation
accepts runtime TensorFlow model tensors and evaluates the model and its
position gradient inside the same enclosing GPU/XLA computation as
initialization, Stan windowed adaptation and Euclidean multinomial NUTS.
The tested numerical path calls no CPU sampler or Python model callback.
It compiles to multiple GPU kernels and library operations; it is not a
single fused CUDA kernel or a port of the entire Stan language and Math library.

The prior native bridge was committed as 90617dce2. The implementation is
[stan_gpu_tf](../benchmarks/stan_gpu_tf/README.md), the executable coordinator is
[master.py](../benchmarks/stan_gpu_tf/master.py), and the
[review](bayesfilter-stan-gpu-tf-review-2026-10-28.md) records the audit and repairs.
This result closes the bounded
[master plan](bayesfilter-stan-gpu-tf-master-2026-10-28.md); it does not promote a
repository default or authorize a larger sampler campaign.

## Validity before performance

Four chains each used 500 warmup and 500 retained transitions, FP64, diagonal
metrics, maximum depth 10, target acceptance .8, and the pinned Stan adaptation
defaults. These are short engineering runs with one seed, not convergence
guarantees. Modern R-hat, bulk/tail ESS, reference moments and MCSE were assessed
after execution. The inherited screen requires R-hat <=1.01, bulk/tail ESS
>=400, standardized MCSE <=.05, reference agreement within
max(4 MCSE, .05 reference SD), finite retained states/cache and movement in every
chain. Any retained divergence also vetoes posterior-efficiency ranking here.

| Target / execution | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Reference moments | Complete screen |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Gaussian10, GPU static | 1.00874 | 3102.1 | 1350.0 | 0 | Pass | Pass |
| Gaussian10, CmdStan | 1.00723 | 3498.8 | 1351.4 | 0 | Pass | Pass |
| Banana10, GPU static | 1.00485 | 1937.8 | 1368.8 | 0 | Pass | Pass |
| Banana10, CmdStan | 1.00726 | 1700.2 | 1236.2 | 0 | Pass | Pass |
| Centered funnel10, GPU static | 1.15579 | 20.4 | 22.8 | 1 | Pass | Fail |
| Centered funnel10, CmdStan | 1.04801 | 51.4 | 145.3 | 10 | Fail | Fail |
| Regression512x32, GPU static | 1.00903 | 3267.4 | 1248.7 | 0 | Pass | Pass |
| Regression512x32, CmdStan | 1.01173 | 3170.8 | 1204.8 | 0 | Pass | Fail |

All tested GPU layouts and command-buffer settings produced exactly the same
retained draws, so their diagnostics coincide. CPU map and static Gaussian
passed. CPU map banana passed, but CPU static banana failed reference-moment
agreement. Both CPU regression layouts narrowly failed the R-hat screen. Every
funnel configuration failed. These outcomes prevent posterior-efficiency
ranking; they do not invalidate the independently checked transition algorithm.

For the centered funnel, the observed fraction with v<-3 was .049 for the GPU
run and zero for this CmdStan run, compared with the exact marginal probability
about .1587. CPU map/static fractions were .0015/.068. These particular short,
centrally initialized runs did not adequately explore the funnel. This does
not establish that Stan generally fails this target. Longer target-specific
tuning and fresh sampling would be needed for a converged comparison, under
a new bounded campaign.

## Execution times

Times below are seconds for initialization, all warmup and all retained
transitions across four chains. TF columns show medians of three alternating-
order repetitions of the same seed, excluding compilation and draw serialization
and including scalar completion synchronization. The command-buffer run has
no verbose runtime tracing. These are descriptive wall times, not statistical
performance or posterior-efficiency rankings.

| Target | Original GPU map | GPU static | GPU static + command buffers | CmdStan CPU, 4 parallel processes |
| --- | ---: | ---: | ---: | ---: |
| Gaussian10 | 1.4970 | 0.8253 | 0.5085 | 0.04475 |
| Banana10, bend .3 | 1.7993 | 0.9914 | 0.6264 | 0.04865 |
| Centered funnel10, failed screen | 11.2922 | 6.2759 | 4.1072 | 0.06666 |
| Regression512x32 | 2.0499 | 1.3710 | 1.0615 | 0.14471 |

Static RNG graph expansion plus command buffers reduced observed GPU time by
factors 2.94, 2.87, 2.75 and 1.93 respectively. Within the command-buffer run,
static/map time ratios imply factors 1.75, 1.73, 1.65 and 1.41. Retained draws
and final RNG states matched exactly between the GPU layouts. Native Stan is
still substantially faster on these small targets. The regression fixture is
a well-conditioned analytic Gaussian with repeated matrix evaluation; it is
not evidence about expensive neural-network or filtering likelihoods.

CmdStan timings include subprocess initialization and CSV output and come
from one four-process run. They use a different RNG and therefore different
adaptive trajectories and work. An additional native in-process comparison
avoids process/CSV costs but runs four chains sequentially:

| Target | CPU map TF | CPU static TF | Native Stan in-process CPU |
| --- | ---: | ---: | ---: |
| Gaussian10 | 0.07511 | 0.08851 | 0.03664 |
| Banana10 | 0.08778 | 0.10440 | 0.05266 |
| Centered funnel10, failed screen | 0.44209 | 0.44705 | 0.19993 |
| Regression512x32 | 0.52802 | 0.54600 | Not exposed by this bridge |

The in-process native scope includes ctypes dispatch, allocation, initialization,
adaptation and retained sampling. Its outputs were not retained for posterior
assessment; the CmdStan diagnostics above must not be attributed to that run.
The GPU static/command-buffer times are descriptively 13.9, 11.9 and 20.5 times
the in-process native times on the first three fixtures. These are hardware,
layout and work differences as well as implementation differences. No evidence
here supports a general GPU crossover point or an ESS-per-second advantage.

Static expansion increases compilation cost. On the clean command-buffer GPU
run, first-call times including execution were 23.96, 25.39, 28.82 and 26.84
seconds for static, versus 9.61, 10.73, 17.03 and 11.16 seconds for map. On CPU,
static first calls were 10.04-11.70 seconds versus 4.48-5.56 seconds for map.
The optimization is optional and remains inappropriate to call universally
faster, especially for short one-off fits or CPU execution.

## Engineering evidence and source scope

- 132 tests passed, including tests built against unmodified installed Stan
  headers for initialization, step-size search, windowed adaptation, metric
  updates, tree transitions, stopping and proposal selection. The existing
  [coverage matrix](../reference/nuts-stan-coverage.md) identifies source anchors.
- CPU and GPU each passed 182 native transition cases with exact discrete
  choices. Maximum scaled residuals were 1.57e-15 and 2.78e-16 respectively.
  The tape file SHA-256 is
  ffa0822815dfcaf69e6bd7b69c8b9c84bbaf742dfeb3c51798b9189d11a143af.
- Uniform/normal Philox outputs matched exactly for 1, 4 and 16 chains. The
  final CPU and GPU controlled comparisons each checked 96 transitions from
  common states/steps/seeds; every floating value and discrete choice matched.
- The runtime-input regression changed its draws when input data changed,
  without retracing. Analytical position gradients matched exactly. Calling
  the full fit within an enclosing TF producer/consumer function agreed to
  1.11e-16; full-fit replay was exact. Unit/diagonal/dense interface and
  continuation tests passed.
- The qualified full-fit GPU HLO contained the TF model and position gradient
  inside the loops, 229 fusion instructions, 11 while instructions, and only
  the permitted __cublas$gemm custom call. There was one trace and GPU output
  placement, with no PyFunc, native sampler call or host model callback.

The implementation supports deterministic row-independent unconstrained FP64
TF models with stable shapes and XLA-compatible forward and gradient operators.
The caller owns constrained-parameter transforms and log Jacobians, passes
frozen model parameters/data as tensors, and checks returned validity flags.
The random generator is TF Philox, not Stan MIXMAX. Differentiation through the
discrete sampler, every TF operator, the entire Stan language, and universal
CPU/GPU bitwise trajectory identity are outside the checked scope.

Long CPU adaptive fits differed between RNG layouts despite exact primitive RNG
and common-state checks. Their first changed warmup draws occurred at iterations
148, 85 and 53 (zero-based) for Gaussian, banana and funnel, with maximum
differences 1.28e-15, 1.11e-16 and 1.78e-15. Regression was identical. This is
consistent with arithmetic-rounding amplification, but does not prove the cause
of every later adaptive branch. CPU banana/funnel final seeds differed; their
whole-fit timing comparisons therefore need not measure identical work.

CPU/GPU starts differed by at most 1.11e-16. Separately generated regression
designs differed by at most 4.44e-16, while other model data matched. This small
fixture mismatch prevents a bitwise cross-device input claim. Within-backend
layout comparisons used identical inputs; native Stan consumed the CPU fixture.

## Command buffers and profiling

The initial profiler captured 450 one-byte device-to-host loop-predicate copies
during 16 regression transitions. Numerical model/state work remained in GPU
HLO. The explicit command-buffer option includes WHILE and CONDITIONAL lowering
alongside GPU fusion/library calls. Non-profiled logs in qualification-01
record execution of command buffers on device 0.

TF 2.20's pinned command_buffer_thunk.cc, lines 224-231, disables command
buffers while its profiler is active because of an upstream CUPTI/CUDA-graph
memory-corruption issue. The local source copy and URL/hash are preserved in
.localresources/stan-gpu-tf-20261028/. Consequently those profiles observe the
fallback executor, not normal command-buffer execution. The unsafe profiling
override was not enabled. Normal command-buffer use is established; complete
absence of host transfers during normal execution remains unproved. Neither
HLO fusion counts nor GPU output placement alone proves that stronger claim.

The clean GPU benchmark recorded memory growth enabled before initialization,
TF32 off, GPU 0 RTX 4080 SUPER, TensorFlow 2.20.0, TFP 0.25.0 and Python 3.13.13.
TensorFlow allocator peak was 5,271,552 bytes and process maximum host RSS was
2,965,332 KiB. The longer qualification process peaked at 5,494,528 allocator
bytes and 4,279,640 KiB host RSS. Allocator accounting does not include every
compiler/library/driver allocation; these values are not total device-memory
requirements. CPU-only workers intentionally hid GPUs.

## Attempts, artifacts and closure

The campaign used 11 worker attempts and 1990.90 worker wall seconds of the
reviewed limits of 12 attempts and 2400 seconds. Source inspection, reporting
and edits are outside the worker compute cap. No larger experiment was launched.

| Phase | Worker seconds | Outcome |
| --- | ---: | --- |
| tests-01 | 214.43 | 132 tests passed |
| validation-01 CPU / GPU | 49.95 / 120.04 | Native-oracle and TF integration passed |
| validation-command-buffer-01 | 116.25 | Exit 1 without Python error; cause unresolved, preserved |
| command-buffer-probe-01 | 3.12 | Small conditional-loop smoke passed |
| validation-command-buffer-02 | 116.85 | Numerical checks passed; incorrect profiler no-copy assertion failed |
| benchmark-01 CPU / GPU | 127.67 / 372.78 | Four fixtures each completed |
| qualification-01 GPU | 514.91 | Controlled checks, HLO, runtime logs, all fixtures passed engineering checks |
| benchmark-command-buffer-01 GPU | 333.43 | Clean timing without verbose runtime logging |
| diagnostics-01 CPU | 21.47 | 36 saved-draw assessments and controlled/replay checks completed |

Each phase has a manifest, actual commands, source snapshot and logs under
docs/benchmarks/artifacts/stan-gpu-tf-20261028/. Draws, telemetry, HLO, profiles
and native CSVs remain there. The tracked
[evidence summary](../benchmarks/stan_gpu_tf/evidence-2026-10-28.json) preserves
timings, diagnostics, source hashes, command/budget records and external source
provenance without committing large binaries or HLO. Workers ran from commit
90617dce2 plus their preserved source snapshots. Terminal documentation was
updated afterward; numerical implementation files match the final GPU snapshot.

The profiler assertion was repaired after source inspection; no numerical
acceptance rule was relaxed. The first failed worker remains unexplained and
limits robustness claims despite subsequent successful qualification. The
posterior failures reject those short-chain candidates, not the research
direction. Required engineering phases and the planned diagnostic repair are
complete; a longer convergence or realistic-model performance campaign would
need a separately bounded plan.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep standalone GPU/TF implementation | Native-oracle and enclosing XLA checks pass | No observed numerical/host-model fallback defect | Finite source/test coverage | Use documented optional API | All Stan/TF features or production readiness |
| Keep static RNG and command buffers as explicit options | Identical GPU draws and lower observed timing | No GPU replay mismatch | Compilation cost, hardware and backend dependence | Test realistic expensive TF models under a new plan | Universal speedup or zero host traffic |
| No posterior-efficiency ranking | Required screens fail for funnel and some CPU/native arms | Convergence/reference/divergence vetoes active | Short chains, central starts and one seed | Fresh target-specific tuning and longer validation | Stan or GPU method intrinsically fails funnel |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | All engineering checks passed after diagnostic repair; funnel posterior candidates fail; CPU static banana and CPU/native regression also fail declared screens |
| Statistically supported ranking | None; three timing repeats and one sampler seed do not establish it |
| Descriptive-only differences | Timings, 1.93-2.94 GPU time factors, ESS/R-hat, divergences and neck visitation |
| Default-readiness | Not established; standalone optional implementation only |
| Next evidence needed | Replicated realistic-model timing with uncertainty, converged fresh samples, memory/compile scaling, and compatible normal-execution transfer observation |

Post-run red team: the strongest alternative performance explanation is that
these small FP64 models mainly measure control/launch overhead and a consumer
GPU's execution characteristics, not expensive-target throughput. Different
native RNG trajectories also change work. Larger models and independently
replicated timings could overturn the observed ratios. The weakest scientific
evidence is the one-seed short funnel comparison; it cannot rank either sampler.
The strongest engineering evidence is exact common-variate native transition
agreement plus actual model/gradient HLO and exact GPU replay across execution
options. Preserve that distinction in future reports.
