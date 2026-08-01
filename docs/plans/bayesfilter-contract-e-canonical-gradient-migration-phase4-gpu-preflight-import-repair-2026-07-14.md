# Phase 4 GPU Preflight Import Repair

Date: 2026-07-14

Status: `REPAIRED_BEFORE_NUMERICAL_OUTPUT`

The first shell launch of the Phase 4 GPU preflight exited in approximately
three seconds while importing the harness:

```text
ModuleNotFoundError: No module named 'bayesfilter'
```

Running a script by its `docs/benchmarks/...py` path placed that directory,
rather than the repository root, first on `sys.path`. The error occurred before
`main()`, before fixture construction, and before any compiled Contract E
forward result. It is a harness-entry/environment defect, not candidate, GPU,
numerical, or scientific evidence. It is classified as preflight attempt 0 and
does not consume either of the subplan's two allowed full production-shape
attempts.

The repair inserts the resolved repository root into `sys.path` before local
package imports. It does not change the frozen shape, fixture, dtype, TF32/XLA
settings, transport steps, chunks, ridge, hard vetoes, or nonclaims. A
CPU-hidden import/argument smoke and syntax check must pass before the unchanged
trusted-GPU launch is retried.
