# BayesFilter V1 Nonlinear Performance NP1 Benchmark Harness Plan

## Date

2026-05-15

## Governing Master Program

This plan executes Phase NP1 in:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

## Purpose

Upgrade nonlinear benchmark harnesses so value and score performance can be
measured across predeclared shapes without changing production semantics or
promoting proxy diagnostics.

## Entry Gate

NP1 may start only after NP0 identifies the exact supported production
TensorFlow value and score paths and any reference-only paths.

## Evidence Contract

Question:

- How do current nonlinear paths scale with horizon, dimensions, point count,
  execution mode, device, and `return_filtered` status?

Baseline:

- Existing nonlinear CPU and GPU/XLA benchmark harnesses and the NP0 matrix.

Primary criterion:

- Benchmark artifacts record enough metadata to support shape-specific timing
  statements and reproduce the benchmark command.

Veto diagnostics:

- benchmark code changes production semantics;
- compile and warmup time are mixed into steady-state timing;
- point count, dtype, device, branch status, or parity status is missing;
- score timing is reported without branch status;
- CPU/GPU rows use different shapes or equations;
- proxy diagnostics become promotion criteria.

Explanatory diagnostics only:

- process RSS deltas;
- first-call wall time;
- dense one-step projection errors;
- tiny GPU-visible rows.

What will not be concluded:

- broad GPU speedup;
- default backend choice;
- exact nonlinear likelihood quality;
- HMC or Hessian readiness.

Artifact:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-YYYY-MM-DD.md
docs/benchmarks/bayesfilter-v1-nonlinear-performance-*.json
docs/benchmarks/bayesfilter-v1-nonlinear-performance-*.md
```

## Predeclared Shape Ladder

Tiny:

- `T`: 2, 3, 8;
- state dimension: existing Model A-C dimensions;
- innovation dimension: existing Model A-C dimensions;
- observation dimension: existing Model A-C dimensions;
- paths: value timing rows and certified-score timing rows, with branch
  precheck metadata kept distinct from timed score execution.

Small:

- `T`: 8, 16, 32;
- state dimension: 2-4;
- innovation dimension: 1-3;
- observation dimension: 1-4;
- paths: value timing rows and certified-score timing rows, with branch
  precheck metadata kept distinct from timed score execution.

Medium candidate:

- `T`: 24, 64, 100;
- state dimension: 10-30;
- observation dimension: 5-50;
- innovation dimension: 3-8;
- run only after tiny/small memory and runtime guardrails pass.

CUT4 guardrail:

- CUT4 rows must record augmented dimension and point count before execution.
- Default NP1 CUT4 rows are capped at `point_count <= 512`.
- Medium CUT4 rows are excluded unless a phase result note explicitly narrows
  the shape so the point-count cap holds or a separate plan approves a larger
  cap with memory stop conditions.

Default dtype:

- `tf.float64`, unless a separate dtype subplan is approved.

## Required Row Metadata

Every row must record:

- model and backend;
- value or score path;
- dtype;
- `T`;
- state, innovation, observation, and parameter dimensions;
- point count and polynomial degree;
- `return_filtered` status;
- execution mode: eager, graph, or XLA;
- requested and actual device;
- CPU/GPU trust label when GPU-visible;
- whether GPU devices were intentionally hidden for CPU-only rows;
- compile/warmup policy;
- first-call and steady-state timing;
- memory metadata if available;
- branch status;
- parity status and tolerance;
- command;
- environment;
- output artifact path;
- non-implication text.

The harness code must emit these as explicit row fields, not only as prose in
the result note.  Required schema fields include `row_role`, `path`, `mode`,
`requested_device`, `actual_device`, `device_trust_label`,
`gpu_intentionally_hidden`, `compile_warmup_policy`, `parity_status`,
`parity_tolerance`, `branch_precheck_id`, `branch_precheck_status`,
`command`, `environment_id`, `artifact_path`, `skip_category`, `skip_reason`,
and `non_implication_text`.

Allowed `row_role` values are:

- `value_timing`;
- `score_timing`;
- `branch_precheck`;
- `skipped`.

Score timing rows must reference a distinct `branch_precheck` row through
`branch_precheck_id`.  Branch precheck rows certify branch status only; they
are not timing rows and cannot substitute for score timing.

The benchmark manifest is mandatory and must record the exact command line,
Python, TensorFlow, and TensorFlow Probability versions where importable, git
commit or explicit dirty-worktree status, CPU/GPU visibility policy, output
artifact paths, and the governing NP1 plan/result paths.  The row schema must
include an `environment_id` that points to this manifest entry.

CPU-only rows must record both requested and actual device fields even when GPU
devices are intentionally hidden.  For CPU-only NP1 execution,
`requested_device` must be `cpu`, `gpu_intentionally_hidden` must be true, and
the manifest must show `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import.

Rows whose CUT4 point count exceeds the default cap must be emitted as
`row_role=skipped` with `skip_category=cut4_point_cap` and a concrete
`skip_reason`; they must not be silently omitted or collapsed into a generic
status string.

## Execution Steps

1. Write a phase result note or template-derived run plan before any ladder.
2. Add or update benchmark harness code only if needed.
3. Run CPU-only smoke rows first with GPU hidden.
4. Run the predeclared tiny/small CPU rows, including value timing and
   certified-score timing where branch prechecks pass.
5. Record failed/skipped rows as diagnostic evidence, not as missing data.
6. Save JSON/Markdown artifacts.
7. Write the NP1 result note and continuation decision.

## Continuation Rule

Continue to NP2 only if value-path benchmark rows are reproducible, split
`return_filtered=False` from `return_filtered=True`, define the parity
comparator and tolerance for each row, and carry shape, branch, timing,
manifest, artifact, skip, and non-claim metadata.

Continue to NP3 only if certified score-path timing rows carry branch and
parity metadata and each score timing row links to a separate branch-precheck
row through `branch_precheck_id`.
