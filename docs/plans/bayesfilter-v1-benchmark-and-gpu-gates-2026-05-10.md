# Plan: BayesFilter v1 Benchmark And GPU Gates

## Date

2026-05-10

## Purpose

This note defines benchmark and GPU gates for BayesFilter v1.  It prevents
performance claims from outrunning reproducible shape, compile-time, memory,
and device evidence.

## CPU Benchmark Harness Requirements

Every benchmark artifact should record:

- backend name;
- dtype;
- number of time steps;
- state dimension;
- observation dimension;
- stochastic integration rank if nonlinear;
- sigma-point or CUT point count;
- eager versus compiled mode;
- compile time;
- steady-state run time;
- memory if available;
- TensorFlow and TensorFlow Probability versions;
- CPU device metadata where practical.

Execution artifacts now available:

```text
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.md
docs/benchmarks/bayesfilter-v1-filter-benchmark-medium-2026-05-11.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-medium-2026-05-11.md
```

The artifacts are CPU-only and record process-level RSS snapshots.  RSS deltas
are diagnostic metadata, not isolated per-backend allocation profiles.

## Candidate Benchmark Groups

Linear:

```text
linear_qr_value
linear_qr_score_hessian
linear_svd_value
```

Structural nonlinear:

```text
svd_cubature_value
svd_ukf_value
svd_cut4_value
svd_cut4_score_hessian_smooth_branch
```

## CPU Shape Ladder

Small shapes:

- state dimension: 2--4;
- observation dimension: 1--4;
- time steps: 8--32;
- stochastic rank: 1--3.

Medium shapes:

- state dimension: 10--30;
- observation dimension: 5--50;
- time steps: 24--100;
- stochastic rank: 3--8.

Large shapes are not v1 default evidence until memory and runtime policies are
settled.

## GPU/XLA-GPU Gate

GPU commands must use escalated permissions on this machine.  Required first
probes:

```bash
nvidia-smi
python -c "import json, tensorflow as tf; print(tf.__version__); print(json.dumps([(d.name, d.device_type) for d in tf.config.list_physical_devices()]))"
```

Policy:

- non-escalated GPU failures are sandbox evidence only;
- CPU/GPU benchmarks must use the same model shapes and dtype;
- XLA-GPU unsupported operations must be reported as such, not hidden;
- GPU results cannot be generalized beyond the tested backend/shape pair.

## Output Artifacts

Planned artifacts:

```text
docs/benchmarks/bayesfilter-v1-filter-benchmark-YYYY-MM-DD.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-YYYY-MM-DD.md
```

## Veto Diagnostics

Do not make a performance claim if:

- compile time is omitted;
- point count is omitted;
- CPU and GPU use different shapes;
- device probes are missing;
- GPU commands were non-escalated;
- a benchmark mixes client-specific target claims with generic fixtures.
