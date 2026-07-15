# Kalman QR CPU Analytical Scaling Diagnostic Plan

Date: 2026-07-14

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `AUTHORIZED_AFTER_SKEPTICAL_AUDIT`

## Question And Evidence Contract

Determine whether the surprising CPU analytical QR timings are caused by an
incorrect benchmark boundary, an accidental batch cross-product, TensorFlow
thread configuration, or poor CPU/XLA scaling of the forward-sensitivity
kernel.

- Baseline: the completed lattice's exact CPU/XLA analytical records at
  `D=20`, `P=150`, `T=120`, `B={1,16}`, float32.
- Counterfactuals: the same batch-native analytical function with CPU affinity
  restricted to logical CPU IDs 0-15 and
  `(intra,inter)=(1,1),(16,16),(16,1)`. Use the existing July 9 scalar-row
  analytical artifact for the implementation/reference comparison; do not
  recompile its statically duplicated large `B=16` graph after the focused
  attempt exceeds the five-minute diagnostic bound.
- Primary diagnostic: synchronized warm-call ratios and batch-native versus
  scalar-row time at `B=16`, conditional on numerical parity.
- Hard vetoes: non-finite output, analytical reference mismatch, XLA failure,
  timeout, or inability to establish the requested thread/affinity settings.
- Explanatory only: all timings, CPU utilization, and thread differences.
- Nonclaims: no speed ranking, optimal thread count, exclusive-host result,
  universal CPU/XLA conclusion, or default-policy change.
- Result artifact:
  `docs/plans/bayesfilter-kalman-qr-cpu-analytical-scaling-diagnostic-result-2026-07-14.md`.

## Skeptical Audit

The original CPU schedules correctly isolate tracing from five synchronized
warm calls, but they do not measure physical-core scaling: both intra-op and
inter-op pools were set to the same limit, processes were not affinity-pinned,
the host is dual-socket NUMA, and foreign CPU workloads were allowed. The
diagnostic fixes affinity and compares within one process, so process-to-process
load differences are reduced. It remains a debugging diagnostic rather than
promotion evidence because the host is not exclusive and there are no
independent replications.

## Commands And Stops

Run `docs/benchmarks/run_kalman_qr_cpu_analytical_scaling_diagnostic_2026_07_14.py` under
`CUDA_VISIBLE_DEVICES=-1`, `taskset`, and the three declared thread settings.
Use XLA and two synchronized measured calls after one warm call. Stop on a hard
veto. Otherwise record the ratios, code-path diagnosis, remaining uncertainty,
and the minimum benchmark repair needed before interpreting CPU scaling.
