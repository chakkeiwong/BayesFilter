# Hybrid Walnutpie large-model feasibility

Synthetic nonlinear regression; four chains, FP64, fixed kernel, two transitions per trial. This is an execution experiment, not a posterior convergence study.

Observations: 16384; hidden width: 128. Numerical components are XLA compiled; hybrid uses explicit host dispatch between CPU tree and GPU integration.

| Comparison (lower favors numerator) | Paired ratio | 95% interval |
|---|---:|---|
| hybrid_over_gpu | 1.1381 | [1.1225352098867232, 1.1539203474607242] |
| hybrid_over_cpu | 0.1132 | [0.11180653162147786, 0.11466159138053329] |
| gpu_over_cpu | 0.0995 | [0.09891009025975694, 0.10006189848761259] |

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Do not promote hybrid | Paired transition timings | Matched fields/RNG/work and finite caches passed | One synthetic target, sharing, small timing sample | Validate another target or inspect transfer overhead | Posterior quality or universal backend preference |

## Inference status

| Item | Finding |
|---|---|
| Hard veto screen | Engineering replay passed; no posterior screen claimed |
| Statistically supported ranking | Only paired intervals excluding one, conditional on this target/session |
| Descriptive-only differences | Work, memory, compilation, transfer counts |
| Default readiness | None |
| Next evidence | Independent target and hardware/session replication |

## Result review

Alternative explanations include workload selection, short trajectories, CPU sharing and device-specific launch behavior. Another target, chain count, CPU threading or isolated session could overturn the result. The candidate includes Python dispatch; it does not bound an optimized C++ dispatcher.

Extended review, transfer diagnostics and CPU-thread sensitivity: docs/plans/bayesfilter-walnutpie-hybrid-large-model-result-2026-10-29.md. Machine-readable terminal review: review.json.
