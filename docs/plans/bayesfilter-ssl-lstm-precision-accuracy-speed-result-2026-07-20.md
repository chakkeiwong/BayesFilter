# SSL-LSTM Precision Accuracy And Speed Result

Date: 2026-07-20  
Status: `MIXED_ACCURATE_NOT_FASTER_ALL_FLOAT32_VETOED`

## Outcome

The current selected SSL-LSTM principal-root UKF target is genuinely all
float64; enabling TensorFlow's TF32 policy does not change float64 matrix
operations. Two lower-precision policies were implemented in an isolated
experimental mirror and compared with the current all-float64 direct-JVP
target at `q in {5,10,20}`.

- `mixed_lstm32_filter64` passes the prospective value, score, finiteness, and
  covariance-branch gates. It reduces warm TensorFlow allocator use by about
  32--56 percent, but it does not produce a meaningful end-to-end warm speedup.
- `all_float32_tf32` cannot preserve the strict principal-root recursion. A
  pure attempt encountered a nonpositive covariance eigenvalue already at
  q=5. The timed arm therefore used an explicit `1e-6` eigenvalue floor; the
  maximum number of placement eigenvalues floored at one filtering step was 6,
  15, and 29 at q=5, q=10, and q=20 respectively. It is rejected for the
  unchanged target even where timing is faster.

The production FP64 implementation was not changed.

## Decision Table

| Decision field | Result |
| --- | --- |
| Decision | Keep all-float64 as the admitted executable target. Retain the isolated mixed implementation as an experimental memory candidate; do not promote it for speed. Reject all-float32/TF32 for the current principal-root HMC target. |
| Primary criterion | Mixed: pass at all q. All-float32: fail due to branch changes and accuracy failures. |
| Veto diagnostics | Mixed raised no floor, nonfinite, device, XLA, or resource veto. All-float32 requires covariance flooring and fails at least one value or scaled-score threshold at every q. |
| Main uncertainty | GPU 0 background utilization contaminated most timing pairs. Only clean paired timing is interpreted, and it still shows no useful mixed-precision acceleration. |
| Next justified action | Profile the all-float64 versus mixed local LSTM/JVP kernels. If memory, rather than runtime, blocks a future rung, mixed precision is a viable optional candidate subject to downstream HMC validation. |
| Not concluded | No posterior correctness, HMC convergence, NeuTra improvement, statistically supported speed ranking, scientific superiority, or dtype-default promotion. |

## Accuracy Results

Errors are maxima over five fixed parameter points and two fresh-process GPU
repetitions against the all-float64 experimental reference.

| q | Mixed value abs | Mixed score abs | Mixed scaled score | Mixed branch | FP32 value abs | FP32 score abs | FP32 scaled score | FP32 max placement floors |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 5 | `8.23e-7` | `2.07e-6` | `1.08e-7` | unchanged | `5.86e-3` | `1.28e-2` | `8.97e-4` | 6 |
| 10 | `1.67e-7` | `9.52e-7` | `4.39e-7` | unchanged | `6.33e-4` | `5.73e-3` | `3.35e-3` | 15 |
| 20 | `9.49e-7` | `1.28e-5` | `2.09e-6` | unchanged | `8.69e-4` | `1.03e-2` | `9.89e-3` | 29 |

The mixed limits were value `2e-5`, absolute score `2e-4`, and scaled score
`2e-4`; every rung passes by a wide margin. The all-float32 limits were value
`2e-3`, absolute score `2e-2`, and scaled score `2e-3`. q=5 fails the value
limit; q=10 and q=20 fail the scaled-score limit; all rungs also fail the
unchanged-floor-branch requirement.

Before GPU execution, a CPU-hidden five-point screen found the same structure:
mixed precision passed through q=20, while pure all-float32 aborted at q=5 on
an active/nonpositive placement eigenvalue. The timed FP32 arm's explicit
floor is therefore a visible changed-target diagnostic, not a hidden repair.

## Speed And Memory Results

Warm timing is descriptive. Ratios below one favor the candidate.

| q | Mixed median ratio | Clean mixed ratio | FP32 median ratio | Clean FP32 ratio | FP64 peak | Mixed peak | FP32 peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | `1.027` | `0.936` | `0.519` | `0.780` | 2.000 MiB | 0.875 MiB | 0.770 MiB |
| 10 | `1.160` | none | `1.201` | none | 3.576 MiB | 2.416 MiB | 2.000 MiB |
| 20 | `1.075` | `1.019` | `0.709` | `0.782`, `0.636` | 12.647 MiB | 8.111 MiB | 6.631 MiB |

The mixed arm is accurate but its only clean q=20 comparison is 1.9 percent
slower than FP64. At q=5 it is 6.4 percent faster in the clean pair, too small
and too low-dimensional to justify a production change. Mixed casting also
adds graph work and keeps the dominant covariance/root algebra in FP64, so
the absence of a meaningful end-to-end speedup is consistent with Amdahl's
law. Its reliable benefit is warm allocator reduction.

The regularized all-float32 arm is descriptively 22--36 percent faster in the
two clean q=20 comparisons and uses about half the warm allocator memory. That
speed cannot override the numerical veto: the arm implements a repeatedly
floored covariance recursion rather than the current strict-SPD target.

Cold compile/first-call time was approximately 38--45 seconds per worker for
all arms, with no systematic lower-precision advantage. Host high-water memory
was about 1.8--2.0 GiB for all arms.

## Comparator Validation

The experimental engine was isolated so no shared production dtype contract
or other active lane was changed. Its all-float64 arm was checked against both
the current target and the earlier production GPU/XLA JVP artifacts. Across
q=`5,10,20` and the same points, the latter comparison had maximum value error
`4.39e-7` and maximum score error `2.65e-7`. This is negligible relative to the
prospective precision gates and reflects graph-grouping roundoff around the
dense principal root, not a different model or derivative.

## Verification

Focused CPU-hidden tests cover dtype-policy semantics, FP64 target
reproduction, finite mixed/FP32 execution, visible FP32 floor activation,
CPU-XLA compilation, the bounded benchmark schedule, and summary veto logic.

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest \
  tests/test_ssl_lstm_precision_experiment_tf.py \
  tests/test_ssl_lstm_precision_gpu_xla_benchmark.py -q
```

Result: 7 tests passed.

Static checks:

- `compileall`: pass for the experimental engine, benchmark, and tests.
- `git diff --check`: pass.
- 18 of 18 fresh GPU/XLA worker artifacts completed without nonfinite, device,
  timeout, allocator, or host-memory failure.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `3250e0cb708eef7f8cbeafb62b2fd27741e3554f` plus documented dirty working-tree changes |
| Branch | `main` |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/benchmark_ssl_lstm_precision_gpu_xla_2026_07_20.py --mode supervisor --physical-gpu 0 --authorize-gpu-benchmark` |
| Environment | Python 3.13.13; TensorFlow 2.20.0; TensorFlow Probability 0.25.0; conda env `tfgpu` |
| Device | trusted physical GPU 0; GPU 1 left to the other active lane |
| Dtypes | reference FP64/FP64; mixed FP32 LSTM plus FP64 filter; aggressive FP32/FP32 with TF32-eligible matmuls |
| XLA | enabled for every GPU worker |
| TF32 policy | enabled and recorded separately from storage dtype |
| Seeds | N/A; deterministic target, observations, points, and sigma rule |
| Data version | in-repository deterministic SSL-LSTM q-ladder fixtures |
| Wall time | 875.55 seconds for 18 fresh workers |
| Plan | `docs/plans/bayesfilter-ssl-lstm-precision-accuracy-speed-plan-2026-07-20.md` |
| Result | this file |
| Output | `docs/plans/artifacts/ssl-lstm-precision-accuracy-speed-2026-07-20/gpu-xla/summary.json` plus per-worker JSON/log files |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Isolated implementation, FP64 comparator reproduction, CPU XLA, 18 GPU XLA workers, dtype manifests, and resource checks pass. |
| Numerical validity | Mixed passes prospective value/score/branch gates. All-float32 fails strict-SPD branch preservation and accuracy gates. |
| Scientific interpretation | No posterior, sampler, model-adequacy, or scientific conclusion was tested. |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Mixed passes. All-float32 is vetoed by repeated floor activation plus accuracy failures. |
| Statistically supported ranking | None. Two repetitions and contaminated GPU load do not support a stochastic performance ranking. |
| Descriptive-only differences | Warm times, ratios, compile times, allocator peaks, and host RSS. |
| Default readiness | Neither lower-precision policy is ready as a new default. |
| Next evidence needed | Kernel-level profile for the accurate mixed arm; downstream HMC validation only if mixed becomes necessary for memory or a later fused implementation demonstrates material speed. |

## Post-Run Red Team

The strongest alternative explanation for the mixed timing is inefficient cast
boundaries rather than an inherent limitation of mixed precision. That is
plausible and motivates profiling, but does not alter the current observation
that this implementation is not faster end to end. A fused implementation
with a large clean q=20 reduction would overturn the speed conclusion.

The strongest alternative explanation for the all-float32 failure is that the
covariance update, not the LSTM itself, needs a numerically stronger square-root
form. That leaves FP32 LSTM computation viable and is consistent with the mixed
pass. It does not rescue the tested all-float32 principal-root recursion.

The weakest evidence is timing because GPU 0 background utilization varied.
The numerical comparison is deterministic and repeated exactly; the mixed
accuracy pass and all-float32 floor/accuracy veto do not depend on timing.
