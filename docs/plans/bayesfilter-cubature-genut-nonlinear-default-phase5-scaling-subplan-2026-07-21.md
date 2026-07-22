# Phase 5 Scaling Subplan: Cubature/GenUT Candidate

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_GPU_XLA_RESOURCE_SCALING_ONLY`

> **Correction, 2026-07-22:** This resource ladder used arbitrary transformed
> inputs rather than an SV DGP.  It is engineering-only and scientifically
> irrelevant to SV performance.

## Research Question

After the valid tiny GPU/XLA smoke, is the generic candidate finite
value/recursive-score program computationally feasible as particle count and
horizon increase, and does it remain finite with bounded allocator growth?

## Scope

- Candidate exact transformed-SV target only; no NAWM inference and no new
  model adapter is invented in this subplan.
- TensorFlow float32 with TF32 enabled on the trusted GPU/XLA route.
- Fixed stateless inputs and the same candidate controls for all cells.
- Scalar state (`d=1`) because the current target adapter is scalar. This is a
  scaling diagnostic, not high-dimensional evidence.
- Ladder: `N in {12, 24, 48, 96}`, `T in {2, 10, 50}`.
- Fresh artifact root:
  `docs/benchmarks/artifacts/cubature_genut_gpu_xla_scaling_20260721/attempt01/`.

## Baseline And Pass Criteria

The baseline is the accepted tiny GPU/XLA smoke configuration, not Contract E
and not an exact nonlinear filtering oracle. Each cell must have finite value,
finite score, finite reset diagnostics, verified GPU placement, verified XLA,
and allocator telemetry. Record compile/warm execution separately when the
harness supports it.

## Vetoes And Nonclaims

Stop the ladder on nonfinite output, memory-policy failure, silent CPU fallback,
or graph/allocator failure. Runtime and memory are explanatory diagnostics;
they do not establish numerical superiority. Passing this ladder does not
establish FP64 agreement, full-horizon accuracy, model-scope tuning, or
leaderboard/default readiness.

## Skeptical Pre-Mortem

The ladder could pass while hiding a target mismatch or a score derivative
error because it reuses one exact-SV adapter and does not compare Contract E.
It could fail from the current static Python loop or XLA graph growth rather
than from the cubature mathematics. The earliest discriminator is the same
fixed-input finite-score/central-FD check at the failing cell, followed by
allocator and concrete-function inspection.

## Execution

Use a fresh versioned artifact directory. Run with trusted GPU permissions and
`TF_FORCE_GPU_ALLOW_GROWTH=true`. Preserve all failed cells and record the
exact command, environment, git commit, device, controls, seeds, wall time,
and artifact paths.
