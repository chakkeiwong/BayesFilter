# BayesFilter V1 Nonlinear Performance NP0 Inventory Baseline Plan

## Date

2026-05-15

## Governing Master Program

This plan executes Phase NP0 in:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

## Purpose

Build an explicit current-surface matrix for all nonlinear filter paths before
any benchmark, optimization, XLA, or GPU evidence is interpreted.

This is an evidence-indexing phase.  It must not edit production code, run
benchmarks, or make new performance claims.

## Scope

In scope:

- TensorFlow nonlinear value paths in `bayesfilter/nonlinear/sigma_points_tf.py`
  and `bayesfilter/nonlinear/svd_cut_tf.py`.
- TensorFlow analytic score paths in
  `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`.
- Public helper exports from `bayesfilter/nonlinear/__init__.py`, including
  rule/placement constructors, classified separately as helper surfaces rather
  than value, score, benchmark, or default-policy candidates.
- NumPy reference paths in `bayesfilter/filters/sigma_points.py` and
  `bayesfilter/filters/particles.py`.
- Current nonlinear tests, benchmark scripts, and benchmark artifacts.

Out of scope:

- new timing runs;
- code changes;
- GPU/CUDA probing;
- production default policy;
- nonlinear HMC or Hessian readiness claims.

## Entry Gate

NP0 may start after the master program and supervisor audit exist and have
converged:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-supervisor-audit-2026-05-15.md
docs/plans/bayesfilter-v1-nonlinear-performance-subplans-supervisor-audit-2026-05-15.md
```

## Evidence Contract

Question:

- What nonlinear filter surfaces exist now, and which are production
  TensorFlow candidates versus testing helpers or NumPy references?

Baseline:

- Repository code and artifacts at NP0 execution time.

Primary criterion:

- Every public nonlinear surface is classified with no unknown support cells.

Veto diagnostics:

- NumPy reference filters are labeled as XLA/GPU-ready.
- Dense one-step projection diagnostics are treated as exact nonlinear
  likelihood evidence.
- A prior tiny GPU/XLA row is treated as broad performance evidence.
- Current score branches are described without branch requirements.

Explanatory diagnostics only:

- prior runtime artifacts;
- prior branch-grid summaries;
- prior dense projection rows for Models B-C.

What will not be concluded:

- new performance ranking;
- new XLA support;
- GPU speedup;
- default backend policy;
- exact nonlinear likelihood quality for Models B-C.

Artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-result-YYYY-MM-DD.md
```

## Required Matrix

Each row must include:

- function/backend;
- value, score, testing helper, or reference;
- implementation file;
- public export status;
- public/helper classification rationale, including any explicitly excluded
  helper-level export and why it is not a filter candidate;
- execution support: eager, graph, XLA, GPU;
- static shape requirements;
- `return_filtered` support and known compiled-mode implications for value
  paths;
- branch requirements;
- per-backend and per-model branch classification for score paths, including
  smooth-branch, structural fixed-support, blocked, or not applicable status;
- result container;
- diagnostics emitted;
- current tests;
- current benchmark coverage;
- current artifact references;
- optimization candidates;
- authoritative non-claims for the cell.

The result must explicitly enumerate all public exports from
`bayesfilter/nonlinear/__init__.py`.  If a public export is a helper rather
than a filter surface, the row may mark performance benchmarking as not
applicable, but it must still have no unknown support cells.  The NP0 result
must contain enough static-shape, branch, artifact, and non-claim metadata to
seed the NP1 benchmark row schema and the NP4 XLA support matrix without
reinterpretation.

## Execution Steps

1. Inspect nonlinear implementation modules and exports.
2. Inspect current tests and benchmark harnesses.
3. Inspect existing nonlinear benchmark/result artifacts.
4. Populate the current-surface matrix.
5. Mark unsupported or deferred cells explicitly.
6. Write the NP0 result artifact.

## Continuation Rule

Continue to NP1 only if NP0 has no unknown surface classification cells and
the result explicitly separates production TensorFlow paths from NumPy
reference paths.
