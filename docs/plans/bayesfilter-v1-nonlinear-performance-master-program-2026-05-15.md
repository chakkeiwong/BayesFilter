# BayesFilter V1 Nonlinear Performance Master Program

## Date

2026-05-15

## Purpose

This master program governs the workflow for speeding up BayesFilter's current
nonlinear filters without letting performance claims outrun correctness,
compiled-behavior, or hardware evidence.

The program covers:

- benchmark design for all current nonlinear filter surfaces;
- implementation optimization;
- XLA JIT compilation gates;
- matched CPU/GPU benchmarking under the local trusted-GPU policy;
- claim consolidation after evidence is recorded.

## Scope

Production TensorFlow nonlinear surfaces:

- `bayesfilter/nonlinear/sigma_points_tf.py`
  - `tf_svd_sigma_point_log_likelihood`
  - `tf_svd_sigma_point_log_likelihood_with_rule`
  - `tf_svd_sigma_point_filter`
- `bayesfilter/nonlinear/svd_cut_tf.py`
  - `tf_svd_cut4_log_likelihood`
  - `tf_svd_cut4_filter`
- `bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py`
  - `tf_svd_cubature_score`
  - `tf_svd_ukf_score`
  - `tf_svd_cut4_score`
  - `tf_svd_sigma_point_score_with_rule`

Reference-only nonlinear surfaces:

- `bayesfilter/filters/sigma_points.py`
  - `StructuralSVDSigmaPointFilter`
- `bayesfilter/filters/particles.py`
  - `particle_filter_log_likelihood`

Benchmark and diagnostic harnesses:

- `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py`
- `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_gpu_xla.py`
- `docs/benchmarks/benchmark_bayesfilter_v1_model_bc_testing.py`
- `docs/benchmarks/benchmark_bayesfilter_v1_model_bc_gpu_xla.py`

Out of scope unless a later subplan explicitly opens them:

- changing mathematical filter semantics;
- promoting dense one-step projection diagnostics to exact nonlinear
  likelihood evidence;
- changing default backends based on one tiny benchmark;
- client-repo switch-over work;
- nonlinear Hessian implementation without a named consumer;
- GPU HMC performance claims.

## Current Implementation Facts

The TensorFlow value filters currently:

- place sigma points through `tf.linalg.eigh`;
- loop over a static observation horizon with Python `range`;
- rebuild augmented block covariance matrices per time step;
- compute value diagnostics such as eigen gaps, floor counts, support
  residuals, and PSD projection residuals;
- optionally collect filtered means and covariances in Python lists;
- return rich result containers from high-level wrappers.

The analytic score filters additionally:

- propagate parameter derivatives through the sigma-point placement;
- require smooth or structural fixed-support branch conditions;
- loop over parameter dimension for Kalman-gain derivatives;
- return score-only derivatives while Hessian status remains deferred.

CUT4-G has point count:

```text
2 * augmented_dim + 2**augmented_dim
```

so it must be benchmarked under explicit dimension limits.

The NumPy reference filters are useful for semantics and tests, but they are
not XLA/GPU candidates without a separate TensorFlow rewrite.

## Master Evidence Contract

Question:

- Which current nonlinear filter paths can be made faster for fixed model
  shapes while preserving value, score, branch, and compiled-mode diagnostics?

Baselines:

- Current eager TensorFlow value and score implementations.
- Current graph-mode TensorFlow implementations.
- Existing CPU nonlinear benchmark artifacts.
- Existing CPU/GPU/XLA diagnostic artifacts.

Primary promotion criteria:

- A candidate optimization may be accepted only if it preserves relevant
  eager/graph/XLA parity and improves steady-state runtime or memory for at
  least one predeclared shape class without making another required shape
  materially worse.
- A production default may change only after small and medium shape ladders
  pass correctness, branch, compile, and performance gates.

Veto diagnostics:

- value parity fails beyond predeclared tolerance;
- score parity or finite-difference/reference evidence fails for score paths;
- branch diagnostics fail or are missing;
- XLA compilation fails for a path claimed to be XLA-ready;
- CPU/GPU benchmark rows use different shapes, dtypes, or model equations;
- GPU/CUDA evidence is collected without trusted or escalated execution;
- compile time, warmup policy, point count, or shape metadata is missing;
- a diagnostic-only metric is treated as production correctness evidence;
- one tiny-shape result is generalized to broad speedup.

Explanatory diagnostics only:

- dense one-step Gaussian projection errors for nonlinear Models B-C;
- process RSS deltas;
- first-call wall time after separating compile/warmup;
- GPU-visible success on a single small shape;
- branch-grid timing when not paired with value/score validation.

What will not be concluded from this program alone:

- exact nonlinear likelihood correctness for Models B-C;
- HMC convergence or production sampler readiness;
- GPU speedup across all shapes;
- nonlinear Hessian readiness;
- suitability for external client switch-over.

Required artifacts:

- phase plan or result notes under `docs/plans`;
- benchmark JSON/Markdown artifacts under `docs/benchmarks`;
- reset-memo updates after meaningful implementation or evidence changes;
- source-map updates for new governing artifacts.

Required run manifest for every meaningful benchmark, compile gate, or
optimization result:

- git commit or explicit dirty-worktree status;
- exact command;
- environment or conda environment;
- Python, TensorFlow, and TensorFlow Probability versions where applicable;
- CPU/GPU status and device visibility;
- whether GPU devices were intentionally hidden;
- dtype;
- model, backend, shape, horizon, parameter dimension, and point count;
- random seeds or `N/A`;
- warmup policy and wall time;
- output artifact paths;
- governing phase plan;
- result file;
- derivation or proof-obligation artifact path when an optimization changes
  algebraic value formulas or analytic derivative structure.

NP1-NP5 are research-engineering phases.  Before any long run, sweep, ladder,
or default-policy-informing diagnostic, the executing agent must either create
a phase-specific plan/result note from the repo templates under
`docs/plans/templates` or explicitly fill the equivalent fields in the phase
result before the command is run.  A master-program paragraph is not sufficient
authorization for a ladder, sweep, GPU benchmark, or optimization promotion.

## Phase Table

| Phase | Name | Status | Output |
| --- | --- | --- | --- |
| NP0 | inventory and baseline reconciliation | planned | current-surface matrix |
| NP1 | benchmark harness upgrade | planned | shape-ladder benchmark scripts/artifacts |
| NP2 | value fast-path optimization | planned | value-path patch and result |
| NP3 | score fast-path optimization | planned | score-path patch and result |
| NP4 | XLA JIT gate | planned | focused XLA compile/parity tests |
| NP5 | CPU/GPU ladder | planned | trusted CPU/GPU benchmark artifacts |
| NP6 | reference-path decision | planned | NumPy-reference rewrite/defer decision |
| NP7 | consolidation and default-policy gate | planned | final performance decision table |

## Phase Subplans

```text
NP0: docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-baseline-plan-2026-05-15.md
NP1: docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-plan-2026-05-15.md
NP2: docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-plan-2026-05-15.md
NP3: docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-plan-2026-05-15.md
NP4: docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-jit-gate-plan-2026-05-15.md
NP5: docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-plan-2026-05-15.md
NP6: docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-plan-2026-05-15.md
NP7: docs/plans/bayesfilter-v1-nonlinear-performance-np7-consolidation-default-policy-plan-2026-05-15.md
Supervisor audit: docs/plans/bayesfilter-v1-nonlinear-performance-subplans-supervisor-audit-2026-05-15.md
```

## NP0: Inventory And Baseline Reconciliation

Purpose:

- Build an explicit matrix of all nonlinear filters, execution modes, result
  containers, diagnostics, known benchmarks, and current limitations.

Evidence contract:

- Question: what exactly exists now, and which paths are production candidates?
- Baseline: code and benchmark state as of this plan date.
- Primary criterion: every public nonlinear surface is classified as
  production TF, testing helper, or NumPy reference.
- Veto: particle/reference filters are mislabeled as XLA-ready.
- Artifact: `docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-result-YYYY-MM-DD.md`.

Required matrix columns:

- backend/function;
- value or score;
- implementation file;
- execution mode support: eager, graph, XLA, GPU;
- branch requirements;
- result type;
- diagnostics emitted;
- current tests;
- current benchmark coverage;
- optimization candidates;
- non-claims.

## NP1: Benchmark Harness Upgrade

Purpose:

- Extend the current nonlinear benchmark harnesses so they can answer
  performance questions for value and score paths across a predeclared shape
  ladder.

Evidence contract:

- Question: how do current nonlinear paths scale with horizon, state dimension,
  innovation dimension, observation dimension, parameter dimension, point count,
  execution mode, and device?
- Baseline: existing `benchmark_bayesfilter_v1_nonlinear_filters.py` and
  `benchmark_bayesfilter_v1_nonlinear_gpu_xla.py`.
- Primary criterion: benchmark artifacts record enough metadata to support
  shape-specific timing statements.
- Veto: benchmark code changes production semantics or omits compile/warmup
  separation.
- Artifact: benchmark JSON/Markdown plus
  `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-YYYY-MM-DD.md`.

Required benchmark dimensions:

- models: Model A affine oracle, Model B nonlinear accumulation, Model C
  autonomous nonlinear growth;
- backends: SVD cubature, SVD-UKF, SVD-CUT4;
- paths: value, analytic score where certified;
- modes: eager, graph, XLA;
- devices: CPU, GPU where trusted;
- shapes: at least tiny and small; medium only after memory guardrails pass;
- `return_filtered`: false and true for value paths.

Required row metadata:

- model name and parameterization;
- dtype;
- `T`;
- state, innovation, observation, and parameter dimensions;
- point count and polynomial degree;
- execution mode;
- requested and actual device;
- compile/warmup time;
- steady-state timing policy;
- memory metadata if available;
- branch status;
- parity status;
- output artifact paths.

## NP2: Value Fast-Path Optimization

Purpose:

- Optimize value-only nonlinear TensorFlow filters while preserving current
  high-level wrappers and diagnostics.

Candidate changes:

- add a tensor-only log-likelihood fast path that avoids result containers and
  string diagnostics;
- add `diagnostics_level = "none" | "basic" | "full"` where compatible with
  existing wrappers;
- precompute sigma-point rules outside compiled functions;
- avoid full augmented eigendecomposition when block factors can be reused;
- avoid forming full innovation precision when only vector and cross-covariance
  solves are needed;
- use `cross_covariance @ kalman_gain.T`-style covariance updates where
  equivalent under the implemented covariance law;
- replace Python list collection for filtered states with a compiled-friendly
  storage strategy when `return_filtered=True`;
- test `tf.while_loop` or `tf.scan` variants for large horizons after static
  unrolled parity is preserved.

Mathematical pre-gate:

- Any value-path algebraic rewrite must first be written as a local derivation
  note in project notation.
- The note must state the implemented covariance law, the floored/eigen
  covariance being used, and the exact equality being relied on.
- For the covariance-update rewrite, the note must prove the equivalence
  between the original update and the proposed update under the same
  implemented innovation covariance; if the equivalence holds only under
  symmetry, positive definiteness, fixed flooring, or exact solve conditions,
  those conditions must become executable assertions or explicit branch
  preconditions.
- If the derivation is not written or the conditions cannot be checked, the
  rewrite stays out of production and may only be benchmarked as an
  experimental branch with no promotion claim.
- The NP2 result note must record the derivation artifact path in its run
  manifest before any optimized value-path benchmark is used as promotion
  evidence.

Evidence contract:

- Question: which value-path changes improve runtime without changing
  log-likelihood or filtered-state outputs?
- Baseline: NP1 value benchmark rows before optimization.
- Primary criterion: parity passes and steady-state timing improves on a
  predeclared shape class.
- Veto: diagnostics disappear from high-level wrappers, branch metadata changes
  meaning, or XLA support regresses.
- Artifact: patch, tests, benchmark artifact, and
  `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-YYYY-MM-DD.md`.

## NP3: Score Fast-Path Optimization

Purpose:

- Optimize analytic score paths without weakening branch-certification
  contracts.

Candidate changes:

- vectorize the parameter-axis loop for `d_kalman_gain`;
- reuse eigensystem solves across derivative terms;
- add a score-only fast path that avoids diagnostics not needed by HMC or
  optimization callers;
- separate branch-checking mode from steady-state timing mode while preserving
  a required pre-run branch gate;
- benchmark parameter dimension scaling separately from state dimension
  scaling.

Mathematical pre-gate:

- Any score-path transformation must first name the derivative expression it
  preserves and either cite the existing derivation artifact or add a local
  proof obligation in project notation.
- The proof obligation must include the parameter-axis tensor shapes and the
  equality between the old scalar-parameter loop and the proposed batched solve
  or vectorized expression.
- Branch-checking mode may be separated from steady-state timing only if the
  timing row records the exact branch precheck artifact used for that row.
- Parity against the old implementation is necessary but not sufficient for
  promotion when the transformation changes analytic derivative structure.
- The NP3 result note must record the derivative proof-obligation or source
  derivation artifact path in its run manifest before any optimized score-path
  benchmark is used as promotion evidence.

Evidence contract:

- Question: can analytic score runtime be reduced while preserving certified
  branch behavior and score parity?
- Baseline: NP1 score benchmark rows before optimization.
- Primary criterion: score parity and branch diagnostics pass, and a
  predeclared score shape class improves.
- Veto: assertions are removed instead of relocated behind an explicit branch
  precheck; Model C structural fixed-support behavior is weakened.
- Artifact: patch, tests, benchmark artifact, and
  `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-YYYY-MM-DD.md`.

## NP4: XLA JIT Gate

Purpose:

- Convert XLA support from benchmark-only evidence into focused regression
  gates for supported nonlinear paths.

Evidence contract:

- Question: which nonlinear paths compile and run under
  `tf.function(jit_compile=True)` for fixed static shapes?
- Baseline: graph-mode parity tests and NP1/NP2 benchmark evidence.
- Primary criterion: XLA compiled result matches eager/graph result for each
  supported value or score cell.
- Veto: XLA failure is hidden by falling back to graph mode; dynamic shapes are
  reported as supported when only static shapes compile.
- Artifact: focused tests and
  `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-YYYY-MM-DD.md`.

Minimum tests:

- cubature value XLA parity;
- UKF value XLA parity;
- CUT4 value XLA parity;
- analytic score XLA parity for certified branches;
- no retracing for same static shape;
- clear skip or xfail only for explicitly unsupported paths.

Required support matrix:

- model/backend/path;
- static shape: supported or unsupported;
- dynamic horizon: supported, unsupported, or not claimed;
- `return_filtered=False` status;
- `return_filtered=True` status;
- eager/graph/XLA parity tolerance;
- concrete-function/retracing boundary;
- known unsupported TensorFlow ops or control-flow patterns;
- whether CPU XLA and GPU XLA both passed or only one passed;
- authoritative non-claim text for unsupported cells.

NP4 must produce this matrix as the claim boundary used by NP5 and NP7.  A
benchmark row may not treat a cell as XLA-supported unless it appears as
supported in the NP4 matrix or the benchmark result itself records an
equivalent support-cell entry.

## NP5: CPU/GPU Ladder

Purpose:

- Produce trusted CPU/GPU comparisons for the supported nonlinear paths.

Evidence contract:

- Question: for which exact tested shapes does CPU graph, CPU XLA, GPU graph,
  or GPU XLA win after compile/warmup is separated?
- Baseline: NP1 and NP4 CPU rows.
- Primary criterion: shape-specific CPU/GPU timing statements with identical
  model equations, shapes, dtypes, and branch status.
- Veto: GPU probe is missing, non-escalated GPU failure is treated as hardware
  failure, or tiny rows become broad speedup claims.
- Artifact: benchmark JSON/Markdown plus
  `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-YYYY-MM-DD.md`.

Required trusted pre-probes:

```bash
nvidia-smi
python -c "import json, tensorflow as tf; print(tf.__version__); print(json.dumps([(d.name, d.device_type) for d in tf.config.list_physical_devices()]))"
```

All GPU/CUDA/NVIDIA commands must use escalated or trusted permissions on this
machine.  This requirement applies to the benchmark commands themselves, not
only to the pre-probes.

For artifacts in this repo:

- `escalated_sandbox` means the command was run by Codex with escalated sandbox
  permissions;
- `trusted_external` means the command was run outside the Codex sandbox in an
  explicitly trusted context;
- `nontrusted_sandbox` means the command was run without trusted GPU access and
  can only provide sandbox diagnostics.

A CPU/GPU artifact must record one of these labels for each GPU-visible
command.  GPU-visible rows produced by `nontrusted_sandbox` commands cannot
support CPU/GPU comparison.

## NP6: Reference-Path Decision

Purpose:

- Decide whether the NumPy reference sigma-point and particle filters should
  remain reference-only or receive TensorFlow performance backends.

Evidence contract:

- Question: is a TensorFlow rewrite justified for either reference path?
- Baseline: NP0 inventory and user-facing needs.
- Primary criterion: a rewrite is allowed only if a concrete downstream use
  needs compiled/GPU behavior.
- Veto: reference semantics are replaced by an unvalidated fast path.
- Artifact:
  `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-YYYY-MM-DD.md`.

Decision branches:

- defer and document reference-only status;
- implement a new TensorFlow bootstrap particle backend under separate plan;
- implement a TensorFlow reference sigma-point backend only if it differs from
  existing production paths in a needed way.

## NP7: Consolidation And Default-Policy Gate

Purpose:

- Decide which optimizations are accepted, which are optional, and whether any
  default behavior changes.

Evidence contract:

- Question: what performance policy is justified by NP0-NP6?
- Baseline: original implementation and benchmark artifacts.
- Primary criterion: every accepted change has correctness, branch, compile,
  and shape-specific timing evidence.
- Veto: optional paths are promoted to defaults without medium-shape evidence
  or without score/value parity.
- Artifact:
  `docs/plans/bayesfilter-v1-nonlinear-performance-final-summary-YYYY-MM-DD.md`.

Required decision table columns:

- decision;
- affected function/backend;
- correctness status;
- branch status;
- XLA status;
- CPU/GPU status;
- primary timing result;
- memory status;
- default-policy result;
- uncertainty;
- next justified action;
- what is not concluded.

The final summary must keep three ledgers separate:

- engineering correctness: API compatibility, eager/graph/XLA parity,
  container behavior, tests;
- numerical or sampler validity: value/score branch conditions,
  finite-difference/reference evidence, floor/eigen diagnostics, Monte Carlo
  uncertainty when applicable;
- performance evidence: compile time, steady-state runtime, memory, CPU/GPU
  shape-specific comparisons.

Evidence from one ledger cannot promote another ledger without the required
checks for that ledger.

## Execution Rules

Each phase follows:

```text
phase-specific plan or evidence contract
skeptical audit
implementation or benchmark
targeted tests
benchmark artifact
result note
reset-memo update if state changed
continuation decision
```

Long or research-decision-making runs require a phase-specific plan or result
note before execution.  Quick import, compile, shape, or smoke checks may be
recorded directly in the relevant result note if they are not used as
promotion evidence.

Phase-specific notes should use the repo templates in `docs/plans/templates`
when practical.  If a template is not used, the note must still include the
question, comparator, primary criterion, veto diagnostics, explanatory
diagnostics, non-claims, run manifest, artifact path, interpretation, and next
step.

## Implementation Guardrails

- Preserve existing high-level APIs unless a compatibility subplan approves a
  change.
- Keep rich diagnostics available even if fast paths skip them.
- Do not remove branch assertions from score paths without replacing them with
  an explicit precheck and documented contract.
- Keep dtype fixed when comparing CPU and GPU.
- Keep CUT4 dimension ladders bounded because of exponential point growth.
- Separate compile time from steady-state runtime.
- Use result tensors inside compiled benchmark functions; materialize only
  outside the timed compiled call when needed.
- Do not use benchmark code as a hidden production dependency.
- Preserve unrelated worktree changes.

## Initial Optimization Hypotheses

H1:
- For tiny horizons and small augmented dimensions, graph mode will often beat
  XLA/GPU because compile and launch overhead dominate.

H2:
- For longer horizons, a compiled fast path that avoids rich diagnostics should
  outperform current high-level wrappers.

H3:
- CUT4 may improve nonlinear moment projection on some fixtures, but its
  exponential point count will dominate medium augmented dimensions.

H4:
- Score-path runtime will become parameter-dimension sensitive unless
  parameter-axis solves are vectorized.

H5:
- GPU becomes plausible only for larger batched, horizon, or point-axis
  workloads; tiny unbatched FP64 rows are diagnostic, not decisive.

## Supervisor Review Requirement

This master program must be critically reviewed before execution.  The review
loop is:

1. Codex drafts or revises the plan.
2. Claude Code reviews critically through `scripts/claude_worker.sh`.
3. Codex audits the review rather than accepting it automatically.
4. Codex revises the plan when the review identifies material blockers.
5. Codex explains accepted and rejected changes back to Claude Code.
6. Stop when Claude returns `ACCEPT` or after 5 review loops.

Claude Code must be instructed not to edit files, run tests, run benchmarks,
or run GPU/HMC commands during this planning review.

## Non-Claims

This plan does not claim:

- any new timing result;
- any new GPU speedup;
- any new XLA production guarantee;
- any correctness change;
- any nonlinear HMC readiness improvement;
- any default backend change.
