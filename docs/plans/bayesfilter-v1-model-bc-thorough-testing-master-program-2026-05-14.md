# BayesFilter V1 Model B/C Thorough Testing Master Program

## Date

2026-05-14

## Scope

This master program defines the next BayesFilter-local V1 testing campaign for
the nonlinear Models B and C:

- Model B: smooth nonlinear accumulation;
- Model C: autonomous nonlinear growth with structural fixed-support score
  handling for the default zero-phase-variance law.

It builds on the completed V1 master execution:

```text
docs/plans/bayesfilter-v1-master-program-execution-summary-2026-05-14.md
```

This is a control document for future execution.  It does not execute tests by
itself.

Plan-review artifacts are allowed to update this master program, add BC0-BC8
subplans, register source-map provenance, and record supervisor audits.  They
must not update the V1 reset memo or claim new numerical evidence.  Reset-memo
updates occur during phase execution only.

## Lane Boundaries

Stay inside the BayesFilter V1 lane:

- edit BayesFilter-local tests, testing helpers, benchmark scripts, and V1 plan
  artifacts only;
- do not edit MacroFinance, DSGE, Chapter 18b, structural SVD/SGU plans, or
  the shared monograph reset memo;
- update only the V1 reset memo during execution:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Production implementation must remain TensorFlow/TensorFlow Probability.  NumPy
is allowed only in tests, offline oracles, and artifact readers.

GPU/CUDA/XLA-GPU commands require escalated permissions.  CPU-only validation
must set `CUDA_VISIBLE_DEVICES=-1`.

## Current Evidence Baseline

Already passed:

- Model B and default Model C finite value tests for SVD cubature, SVD-UKF, and
  SVD-CUT4;
- analytic score tests for Model B on the smooth branch;
- analytic score tests for default Model C only through the structural
  fixed-support branch with `allow_fixed_null_support=True`;
- benchmark rows with score/value branch metadata;
- tiny CPU HMC smoke for Model B + SVD-CUT4 only;
- scoped GPU/XLA diagnostic for one tiny Model B + SVD-CUT4 shape.

Still not certified:

- HMC convergence for any nonlinear model/filter;
- HMC smoke for Model C or for Model B with cubature/UKF;
- exact full nonlinear likelihood for Models B-C;
- nonlinear Hessians;
- broad GPU speedup;
- client integration.

## Program Goal

Turn Models B and C from first-rung evidence fixtures into thorough validation
fixtures for nonlinear filtering and score-driven inference.

The program should answer:

1. Do SVD cubature, SVD-UKF, and SVD-CUT4 remain stable across wider parameter,
   horizon, and observation-noise boxes for Models B-C?
2. Do analytic scores remain correct against finite differences across those
   boxes?
3. Does the default Model C structural fixed-support branch behave consistently
   and transparently?
4. Which filters are viable HMC targets, and at what claim scope?
5. Is GPU/XLA useful only after scaling horizon, point count, or batch axes?
6. Do we need exact nonlinear references or nonlinear Hessians for any named
   claim?

## Claim Vocabulary

Use these labels consistently:

- `certified`: default or focused tests pass at the stated scope;
- `diagnostic`: useful evidence that is not a public readiness claim;
- `blocked`: a named branch/gate prevents a claim;
- `deferred`: deliberately not implemented or not run because no consumer or
  claim requires it.

Never promote:

- dense one-step Gaussian projection to exact full nonlinear likelihood;
- tiny HMC smoke to convergence;
- one GPU/XLA row to broad speedup;
- testing-only autodiff Hessian oracles to production API.

## Phase Matrix

| Phase | Gap Closed | Primary Output |
| --- | --- | --- |
| BC0 | Baseline reconciliation | current B/C evidence matrix |
| BC1 | Wider value/score branch boxes | branch-grid artifacts for B/C/filter cells |
| BC2 | Score accuracy stress tests | finite-difference score residual tables |
| BC3 | Horizon/noise robustness | value/score stability over horizon/noise ladders |
| BC4 | Approximation-quality references | stronger reference decision/artifact |
| BC5 | HMC readiness ladder | opt-in HMC diagnostics by target |
| BC6 | GPU/XLA scaling ladder | escalated shape/batch timing diagnostics |
| BC7 | Hessian consumer decision | consumer-gated Hessian decision |
| BC8 | Consolidation and release-candidate gate | final matrix, docs, reset memo, commit |

Phase dependency note:
- BC0-BC4 are sequential.
- BC5 and BC6 are independent downstream diagnostics once their own entry
  gates pass.  BC6 does not depend on HMC classification from BC5; it depends
  on stable BC1/BC3 shapes and escalated GPU permissions.
- BC7 and BC8 run after BC5/BC6 are completed, blocked, or explicitly
  deferred.

Subplans:
- BC0: `docs/plans/bayesfilter-v1-model-bc-bc0-baseline-reconciliation-plan-2026-05-14.md`
- BC1: `docs/plans/bayesfilter-v1-model-bc-bc1-wider-branch-boxes-plan-2026-05-14.md`
- BC2: `docs/plans/bayesfilter-v1-model-bc-bc2-score-accuracy-stress-plan-2026-05-14.md`
- BC3: `docs/plans/bayesfilter-v1-model-bc-bc3-horizon-noise-robustness-plan-2026-05-14.md`
- BC4: `docs/plans/bayesfilter-v1-model-bc-bc4-reference-decision-plan-2026-05-14.md`
- BC5: `docs/plans/bayesfilter-v1-model-bc-bc5-hmc-ladder-plan-2026-05-14.md`
- BC6: `docs/plans/bayesfilter-v1-model-bc-bc6-gpu-xla-scaling-plan-2026-05-14.md`
- BC7: `docs/plans/bayesfilter-v1-model-bc-bc7-hessian-consumer-decision-plan-2026-05-14.md`
- BC8: `docs/plans/bayesfilter-v1-model-bc-bc8-consolidation-release-gate-plan-2026-05-14.md`

Supervisor review:
- `docs/plans/bayesfilter-v1-model-bc-subplans-supervisor-audit-2026-05-14.md`
- `docs/plans/bayesfilter-v1-model-bc-supervisor-audit-evaluation-2026-05-15.md`

## Phase BC0: Baseline Reconciliation

Purpose:
- convert existing B/C evidence into one current matrix.

Actions:
- read P1-P8 result artifacts and current nonlinear tests;
- list Model B and Model C rows for SVD cubature, SVD-UKF, and SVD-CUT4;
- record value status, score status, branch label, HMC status, GPU status,
  reference status, and Hessian status;
- verify that default Model C rows use structural fixed support with
  `allow_fixed_null_support=True`.

Primary gate:
- no B/C/filter cell has unknown status.

Veto diagnostics:
- Model C score claim omits structural fixed-support branch;
- dense one-step reference is described as exact full likelihood.

Artifacts:
- `docs/plans/bayesfilter-v1-model-bc-bc0-baseline-matrix-result-YYYY-MM-DD.md`.

## Phase BC1: Wider Value/Score Branch Boxes

Purpose:
- test whether current value/score claims survive beyond the tiny P2/P3 boxes.

Actions:
- define bounded parameter boxes for Model B:
  \[
    \rho\in[0.55,0.85],\quad
    \sigma\in[0.15,0.40],\quad
    \beta\in[0.45,1.10].
  \]
- define bounded parameter boxes for Model C:
  \[
    \sigma_u\in[0.60,1.40],\quad
    \sigma_y\in[0.60,1.40],\quad
    P_{0,x}\in[0.10,0.50].
  \]
- run deterministic grid and seeded random boxes for all three filters;
- record branch counts, active floors, weak gaps, nonfinite rows,
  deterministic residuals, support residuals, structural-null residuals, and
  failure labels.

Primary gate:
- each model/filter has either a stable box with passing diagnostics or a
  narrowed box with explicit blocked labels.

Stop and narrowing rules:
- predeclare deterministic grid rows and seeded random rows before execution;
- allow at most one planned narrowed-box proposal per model/filter after the
  full box is evaluated;
- the narrowed box must retain a named scientific/use-case rationale and must
  not be selected solely to remove failing rows;
- if the narrowed box still fails a veto diagnostic, classify the cell as
  `blocked` and stop rather than narrowing again.

Veto diagnostics:
- active floors are hidden by regularization;
- default Model C is evaluated without `allow_fixed_null_support=True`;
- failures are aggregated without row-level labels.

Artifacts:
- branch diagnostic test updates if needed;
- JSON/Markdown branch artifacts under `docs/benchmarks` or `docs/plans`;
- BC1 result file.

## Phase BC2: Score Accuracy Stress Tests

Purpose:
- quantify analytic score accuracy for Models B-C across the stable BC1 boxes.

Actions:
- compare analytic score to centered finite differences for each model/filter;
- sweep step sizes to detect cancellation or truncation;
- include compiled/eager parity for score paths where practical;
- record max absolute residual, relative residual, finite-difference step, and
  branch diagnostics for each row.

Primary gate:
- analytic score residuals are below model/filter-specific tolerances on the
  declared stable boxes, or rows are blocked with exact failure labels.

Tolerance rule:
- set absolute and relative tolerances before execution from the P1/P2
  finite-difference baseline, machine precision, parameter scale, and finite
  difference step ladder;
- record the tolerance table in the BC2 result before interpreting pass/fail;
- maximum absolute residual and maximum relative residual are pass/fail
  metrics; step-sensitivity plots and per-parameter residual distributions are
  explanatory unless predeclared as veto diagnostics.

Veto diagnostics:
- finite-difference residuals are computed across active branch changes;
- compiled/eager mismatch is ignored;
- score status is promoted when only value status passed.

Artifacts:
- score stress tests;
- score residual tables;
- BC2 result file.

## Phase BC3: Horizon And Noise Robustness

Purpose:
- test whether B/C filter behavior remains stable as observations become more
  demanding.

Actions:
- create synthetic panels for horizons \(T\in\{3,8,16,32\}\);
- test observation-noise scales around the default and low-noise stress cases;
- record finite log likelihood, score finiteness, branch diagnostics, and
  runtime;
- separate deterministic panels from seeded stochastic panels.

Primary gate:
- each model/filter has a documented robustness envelope or a declared blocker.

Stop rules:
- run the planned horizon ladder \(T\in\{3,8,16,32\}\) and predeclared noise
  scales once per deterministic/seeded panel family;
- if low-noise rows fail, report the largest passing envelope and exact
  blocker labels instead of repeatedly changing the ladder;
- do not add new stochastic seeds after seeing failures unless the result is
  explicitly labeled exploratory and not used for promotion.

Veto diagnostics:
- stochastic panels omit seeds;
- low-noise failures are interpreted as global model failure without branch
  diagnostics;
- runtime regressions are treated as correctness failures.

Artifacts:
- synthetic panel helpers if needed;
- robustness benchmark artifacts;
- BC3 result file.

## Phase BC4: Approximation-quality References

Purpose:
- decide whether Models B-C need stronger references than dense one-step
  projection for the claims being made.

Actions:
- start as a decision phase;
- if a current claim requires stronger evidence, choose one bounded reference:
  dense low-dimensional multi-step quadrature or seeded high-particle SMC;
- label exact, deterministic approximation, and Monte Carlo evidence
  separately;
- keep reference dependencies out of production.

Primary gate:
- either a stronger reference artifact is added with proper labels, or exact
  nonlinear references remain explicitly deferred because no current claim
  needs them.

Required decision table:
- claim;
- current comparator/reference basis;
- comparator type: exact, deterministic approximation, Monte Carlo, or
  diagnostic-only;
- why the comparator is sufficient for that claim;
- what remains out of scope.

Veto diagnostics:
- Monte Carlo reference has no seed/particle metadata;
- dense projection is called exact full likelihood;
- reference dependency enters production imports.

Artifacts:
- BC4 decision/result file;
- optional reference artifact/test only if justified.

## Phase BC5: HMC Readiness Ladder

Purpose:
- move beyond tiny smoke only where target gates pass.

Entry gate:
- BC1 and BC2 pass for the target model/filter.

Target order:
1. Model B + SVD-CUT4, because tiny smoke already passed;
2. Model B + SVD cubature and Model B + SVD-UKF;
3. Model C only after structural fixed-support branch diagnostics and score
   residuals pass on the intended target box.

Actions:
- define prior and target parameter box for each target;
- run finite value/score and compiled/eager parity before sampling;
- run opt-in CPU HMC ladders with multiple seeds;
- report warmup/draw counts, chains, step size, acceptance rate, divergence
  count, R-hat, ESS, MCSE/SD where available, finite-gradient failures, and
  runtime;
- save JSON summaries and NPZ chains for audit.

Primary gate:
- HMC target can be classified as `diagnostic`, `blocked`, or `candidate`
  without claiming convergence unless convergence diagnostics, posterior
  recovery, and the predeclared target-specific acceptance criteria pass.

Veto diagnostics:
- HMC starts before score/branch gates pass;
- tiny smoke is promoted to convergence;
- Model C is sampled without structural fixed-support score diagnostics;
- GPU HMC is run without escalated permissions.

Artifacts:
- opt-in HMC tests;
- HMC benchmark artifacts;
- BC5 result file.

## Phase BC6: GPU/XLA Scaling Ladder

Purpose:
- test whether point-axis, horizon, or batch scaling makes GPU/XLA useful.

Entry gate:
- BC1 stable boxes and BC3 horizon/noise envelopes exist for the shapes tested.
- BC6 does not require BC5 HMC classification.

Actions:
- run only with escalated GPU permissions;
- compare CPU graph, CPU XLA, GPU graph, and GPU XLA;
- test larger horizons, batched parameter points, and, if feasible, batched
  independent panels;
- record device visibility, shape, point count, warmup policy, first-call and
  steady-call times, and memory notes.

Primary gate:
- each row makes a shape-specific performance statement only.

Veto diagnostics:
- non-escalated GPU failure is treated as CUDA failure;
- one tiny shape becomes a broad speedup claim;
- benchmark changes production behavior.

Artifacts:
- GPU/XLA benchmark artifacts;
- BC6 result file.

## Phase BC7: Hessian Consumer Decision

Purpose:
- decide whether any Model B/C testing phase has named a nonlinear Hessian
  consumer.

Actions:
- review BC1-BC6;
- name any consumer: Newton/trust-region optimization, Laplace approximation,
  observed information, Riemannian HMC, or curvature diagnostics;
- if no consumer exists, keep Hessians deferred;
- if a consumer exists, write a separate implementation plan with tensor shape,
  memory, branch, and second-derivative provider contracts.

Primary gate:
- Hessian status is explicit and non-promotional.

Veto diagnostics:
- Hessian code starts before a consumer is named;
- testing-only autodiff oracle is exposed as production API;
- second-order SVD branch issues are ignored.

Artifacts:
- BC7 result file;
- optional Hessian implementation subplan only if justified.

## Phase BC8: Consolidation And Release-candidate Gate

Purpose:
- consolidate Model B/C evidence and decide what can be claimed.

Actions:
- write final B/C evidence matrix;
- update V1 reset memo;
- update source map for new artifacts;
- run fast public API, focused nonlinear suite, focused HMC opt-in if claims
  require it, and full default CPU;
- commit only V1-lane files.

Primary gate:
- final result states B/C value, score, reference, HMC, GPU, and Hessian status
  for each filter.

Veto diagnostics:
- claims exceed artifacts;
- out-of-lane files are dirty or staged;
- default CI depends on GPU, HMC, or external projects.

Artifacts:
- final B/C testing summary;
- reset memo update;
- source-map update;
- commit.

## Default Validation Commands

Fast public API:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Focused nonlinear B/C:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

Full default CPU:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Opt-in HMC diagnostics:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_nonlinear_model_b_readiness_tf.py \
  -p no:cacheprovider
```

## Stop Conditions

Stop and ask for direction if:

- a primary gate fails;
- any veto diagnostic fires;
- execution requires editing MacroFinance, DSGE, Chapter 18b, structural plans,
  or the shared monograph reset memo;
- a phase needs a new mathematical or product decision not stated here;
- GPU/CUDA commands are needed but escalation is unavailable;
- tests fail in a way that cannot be resolved within the phase scope.

## Recommended Execution Prompt

Use this prompt when ready to execute:

```text
Execute the BayesFilter V1 Model B/C thorough testing master program from
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md.
Treat it as the controlling roadmap for this phase.  Execute BC0-BC4 in order;
execute BC5 and BC6 only when their independent entry gates pass; then execute
BC7 and BC8 after BC5/BC6 are completed, blocked, or explicitly deferred.  Use
plan, audit, execute, test, audit result, tidy, reset-memo update, result
artifact, commit, and continuation decision for each phase.  Continue
automatically only when the primary gate passes and no veto diagnostic fires.
Stay inside the BayesFilter V1 lane and update only
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md.
```

If the execution starts from the planning-review commit, first verify that the
working tree is clean and that the reset memo has not already been edited by
another lane.
