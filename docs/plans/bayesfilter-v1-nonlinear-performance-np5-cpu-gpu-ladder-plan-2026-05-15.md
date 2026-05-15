# BayesFilter V1 Nonlinear Performance NP5 CPU/GPU Ladder Plan

## Date

2026-05-15

## Governing Master Program

This plan executes Phase NP5 in:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

## Purpose

Produce trusted, shape-specific CPU/GPU comparisons for nonlinear paths whose
benchmark and XLA support metadata are already documented.

## Entry Gate

NP5 may start only after:

- NP1 benchmark rows exist for the shapes being compared;
- NP4 identifies XLA support cells or the benchmark artifact records
  equivalent support-cell entries;
- branch diagnostics and parity status exist for compared cells.

## Evidence Contract

Question:

- For which exact tested shapes does CPU graph, CPU XLA, GPU graph, or GPU XLA
  win after compile/warmup separation?

Baseline:

- NP1 and NP4 CPU rows.

Primary criterion:

- Every row supports only a shape-specific timing statement with identical
  model equations, shapes, dtype, and branch status across CPU/GPU.

Veto diagnostics:

- GPU probes are missing;
- actual GPU benchmark commands are not trusted or escalated;
- nontrusted sandbox GPU rows are used for comparison;
- CPU/GPU rows use different shapes, dtypes, or branch status;
- tiny rows become broad speedup claims.

Explanatory diagnostics only:

- first-call timing;
- GPU visibility;
- memory notes;
- single-shape speedup.

What will not be concluded:

- broad GPU speedup;
- GPU HMC performance;
- production deployment policy;
- default backend policy.

Artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-YYYY-MM-DD.md
docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-*.json
docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-*.md
```

## Trusted Execution Policy

GPU/CUDA/NVIDIA commands must run under one of:

- `escalated_sandbox`;
- `trusted_external`.

Rows from `nontrusted_sandbox` GPU-visible commands are sandbox diagnostics
only and cannot support CPU/GPU comparison.

Required trusted pre-probes:

```bash
nvidia-smi
python -c "import json, tensorflow as tf; print(tf.__version__); print(json.dumps([(d.name, d.device_type) for d in tf.config.list_physical_devices()]))"
```

The benchmark commands themselves must also be trusted or escalated.

The NP5 result must record trusted pre-probe provenance before interpreting
any GPU-visible row.  If either trusted pre-probe or trusted benchmark
execution is unavailable, NP5 must stop with a structured blocker result rather
than reuse nontrusted GPU diagnostics.

The NP5 comparator matrix must be restricted to cells that are supported by
NP4 or carry an equivalent focused support-cell entry.  For this run, the
default admissible cells are static-shape TensorFlow value paths only.  Score
paths remain out of scope unless a separate NP4-grade score XLA support cell is
added first.

Each benchmark row must include:

- comparator id;
- model, backend, value/score path, dtype, static shape, horizon, point count,
  branch status, parity status, and tolerance;
- execution mode: graph or XLA;
- requested device and actual device;
- trust label: `escalated_sandbox`, `trusted_external`, or
  `nontrusted_sandbox`;
- compile/warmup policy, first-call time, steady-state time, and memory notes
  when available;
- exact command, environment, artifact path, and non-implication text.

Rows may support only exact tested-shape statements.  The result must not rank
CPU/GPU rows unless the model equations, dtype, shape, branch status, parity
status, mode, and support-cell id match.

## Required Comparisons

Where supported:

- CPU graph;
- CPU XLA;
- GPU graph;
- GPU XLA.

For the first NP5 run, a narrow admissible ladder may compare Model B
`return_filtered=False` value rows for cubature, UKF, and CUT4 at a fixed
static horizon, because those cells vary backend/point count and are covered
by the NP4 value support boundary.  Broader Model C, `return_filtered=True`,
score, dynamic-horizon, or batched-panel claims require additional matched
support and benchmark rows.

Rows must vary at least one of:

- horizon;
- backend/point count;
- batched parameter points;
- batched independent panels.

## Continuation Rule

Continue to NP6 after recording shape-specific CPU/GPU status and non-claims.
NP5 cannot unlock HMC convergence, broad GPU speedup, or default backend
changes by itself.
