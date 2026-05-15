# BayesFilter V1 Nonlinear Performance NP4 XLA JIT Gate Plan

## Date

2026-05-15

## Governing Master Program

This plan executes Phase NP4 in:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

## Purpose

Convert XLA support from benchmark-only evidence into a focused regression and
claim-boundary matrix for supported nonlinear paths.

## Entry Gate

NP4 may start after NP1 benchmark baselines exist.  If NP2 or NP3 changed code,
NP4 must run after those changes and include the affected cells.

## Evidence Contract

Question:

- Which nonlinear paths compile and run under `tf.function(jit_compile=True)`
  for fixed static shapes, and where is XLA explicitly unsupported or not
  claimed?

Baseline:

- Existing graph parity tests, NP1 baselines, and accepted NP2/NP3 changes.

Primary criterion:

- Each supported XLA cell has eager/graph/XLA parity and a support-matrix row
  stating shape and retracing boundaries.

Veto diagnostics:

- XLA failure is hidden by graph fallback;
- dynamic-shape support is claimed from static-shape evidence;
- `return_filtered` support is not separated;
- unsupported cells lack non-claim text;
- CPU XLA and GPU XLA are conflated.

Explanatory diagnostics only:

- compile time;
- first-call timing;
- GPU XLA success on one small shape.

What will not be concluded:

- GPU speedup;
- dynamic-shape support beyond matrix rows;
- production default policy;
- HMC readiness.

Artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-YYYY-MM-DD.md
```

## Required Support Matrix

Each matrix row must include:

- model/backend/path;
- value or score;
- static shape;
- dynamic horizon status: supported, unsupported, or not claimed;
- `return_filtered=False` status;
- `return_filtered=True` status;
- eager/graph/XLA parity tolerance;
- concrete-function/retracing boundary;
- CPU XLA status;
- GPU XLA status if trusted GPU evidence exists;
- unsupported ops or control-flow notes;
- authoritative non-claim text.

CPU and GPU XLA status values must be one of:

- `supported`;
- `tested_unsupported`;
- `untested`;
- `skipped_no_trusted_device`;
- `not_claimed`.

NP5 may use only `supported` cells for CPU/GPU performance comparison.

The NP4 result must not infer support from the older timing harness alone.
Support rows must be grounded in focused parity/compile checks or marked with
the status vocabulary above.

Value-path rows must separate `return_filtered=False` from
`return_filtered=True`.  A value cell may be marked `supported` only if the
focused CPU-XLA check validates the log-likelihood and, when
`return_filtered=True`, filtered means and covariances against the eager
baseline for the same static shape.

Score rows must separate smooth-branch cells from structural fixed-support
cells.  A score cell may be marked `supported` only if a focused CPU-XLA check
validates value and score parity against eager execution on a certified branch.
If no such focused check is added, score CPU-XLA cells must be marked
`untested` or `not_claimed`.

Dynamic-horizon support must be `not_claimed` unless an explicit dynamic-shape
test exists.  Current static Python loops over observation length are
fixed-shape evidence only.

GPU XLA cells must be `skipped_no_trusted_device` when NP4 is run CPU-only.
They must not be inferred from CPU-XLA success.

## Required Tests

Minimum focused tests:

- cubature value XLA parity;
- UKF value XLA parity;
- CUT4 value XLA parity;
- value XLA parity for `return_filtered=False` and `return_filtered=True`
  separately, including filtered means/covariances when returned;
- certified analytic score XLA parity where feasible;
- no retracing for same static shape;
- explicit skip/xfail only for documented unsupported cells.

## Continuation Rule

Continue to NP5 only for CPU/GPU benchmark cells that are listed as supported
in the NP4 matrix or whose benchmark artifact records an equivalent
support-cell entry.
