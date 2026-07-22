# LGSSM NeuTra Training Topology Performance Benchmark Result

Date: 2026-07-14

Plan:
`docs/plans/bayesfilter-lgssm-neutra-training-topology-performance-benchmark-plan-2026-07-14.md`

## Verdict

**The graph-native computation is descriptively slightly faster on the two
matched rungs, but a material speedup is not established.**

At five steps, the synchronized warm graph-native run took `49.995 s` versus
`53.464 s` for the host-stepped comparator: `1.069x`, or `6.49%` less time. At
20 steps, it took `200.507 s` versus `202.957 s`: `1.012x`, or `1.21%` less
time. The latter is the more informative steady rung and is too small to
distinguish confidently from run-to-run system variation with one repetition.

The claimed target was the speed effect of changing only outer training control
topology. The quantity actually compared was the same exact-target reverse-KL
dense-IAF update, seeds, optimizer state, and diagnostics under a Python loop
over an XLA-compiled step versus one XLA-compiled `tf.while_loop`. Final state
and diagnostic maximum absolute differences were `0.0` at both rungs.

This result does not provide a measured 100-step time. It also does not support
transport-quality, posterior-correctness, HMC, recipe-ranking, broad GPU, or
default-readiness claims.

## Matched Results

| Steps | Mode | Cold compile + execution | Warm execution | Warm seconds/step |
| ---: | --- | ---: | ---: | ---: |
| 5 | Host-stepped | `60.524 s` | `53.464 s` | `10.693` |
| 5 | Graph-native | `53.282 s` | `49.995 s` | `9.999` |
| 20 | Host-stepped | `208.743 s` | `202.957 s` | `10.148` |
| 20 | Graph-native | `195.396 s` | `200.507 s` | `10.025` |

| Rung | Warm host/graph ratio | Graph time reduction | State parity | Diagnostic parity |
| ---: | ---: | ---: | ---: | ---: |
| 5 steps | `1.0694x` | `6.488%` | max abs `0.0` | max abs `0.0` |
| 20 steps | `1.0122x` | `1.207%` | max abs `0.0` | max abs `0.0` |

Cold timing also favored graph-native on both rungs, but cold compilation is
descriptive only and was not the primary criterion. Each cell used one cold and
one warm repetition in a separate process on the same RTX 4080 SUPER with
TensorFlow `2.19.1`, XLA JIT enabled, TF32 enabled, and `float64` computation.

## Evidence

- Five-step comparison:
  `docs/benchmarks/artifacts/lgssm_neutra_training_topology_benchmark_2026_07_14/steps_005/comparison.json`
  (`sha256:55d3c810d124b41171ba70fdf245f8f72b07bce9396e2d5431688d670b73ad75`)
- Twenty-step comparison:
  `docs/benchmarks/artifacts/lgssm_neutra_training_topology_benchmark_2026_07_14/steps_020/comparison.json`
  (`sha256:3f1a09231250435aaa7b178fefa29a655f38ede3a45bf46dae0c3b2e96f32548`)
- Benchmark harness:
  `docs/benchmarks/benchmark_lgssm_neutra_training_topology.py`

All steps had finite target values, valid pre-regularized target status, zero
nonvalid status count, and zero floor count. Cold-versus-warm parity within each
cell and host-versus-graph parity at each rung were exact for serialized state
and selected diagnostics.

## Attempt Ledger

| Attempt | Classification | Result |
| --- | --- | --- |
| Host 5, attempt 1 | Harness preflight failure before timing | Shared exact target contains internal `While`; absence of all `While` operations was an invalid outer-topology test. No artifact emitted. |
| Host 5, attempt 2 | Harness artifact failure after computation | Relative output path was compared with absolute repository root. No timing promoted and no artifact emitted. |
| Host 5, attempt 3 | Passed | Valid synchronized cell artifact. |
| Graph 5 | Passed | Valid synchronized cell artifact. |
| Host 20 | Passed | Valid synchronized cell artifact. |
| Graph 20 | Passed | Valid synchronized cell artifact. |

The plan allowed at most six benchmark processes. That budget was consumed, so
the conditional 100-step rung was not launched. Both repairs were local harness
repairs and changed neither the scientific target nor the compared computation.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep the graph-native training topology | Directionally passed on 5 and 20 synchronized warm rungs | No parity, target, finite-value, GPU, or XLA veto fired | Only one warm repetition; 20-step advantage is `1.21%` | Continue graph-native for its no-host-loop invariant; do not justify it by a large speed claim | No material or broad speedup |
| Do not publish a 100-step time from this benchmark | Not measured | Campaign process budget exhausted | Long-rung scaling remains unmeasured | Measure the actual planned training run or authorize a separate repeated 100-step performance campaign if precision matters | No defensible 100-step wall time |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Passed: exact cross-mode state and diagnostics, valid target telemetry, finite values, GPU/XLA execution |
| Statistically supported ranking | No; one warm repetition per cell and no uncertainty interval |
| Descriptive-only differences | Graph-native was `6.49%` faster at 5 steps and `1.21%` faster at 20 steps |
| Default-readiness | Not applicable; graph-native is retained for the implementation invariant, not promoted by performance evidence |
| Next evidence needed | Repeated randomized-order 100-step cells with uncertainty if a precise performance or runtime claim is required |

## Post-Run Red Team

The strongest alternative explanation is transient system-load or thermal
variation: a `1.21%` difference can easily arise without a real topology
advantage. The exact parity evidence establishes that the compared updates were
the same, not that the timing estimate is precise. A repeated randomized-order
benchmark whose interval includes no improvement would overturn the speed
interpretation while leaving the graph-native correctness and policy benefits
intact. The weakest evidence is performance uncertainty; no repeated timing
distribution was collected.

