# NeuTra Algorithm Full-Validation Plan (2026-08-17)

## Objective

Establish whether the current NeuTra training plus shared sequential HMC
procedure is reliable across increasing target difficulty, without confusing a
good training loss, a good proposal, or a short-chain diagnostic with a valid
posterior result.

This plan is target-independent. It covers analytic and synthetic controls
first, then geometry stress tests, then application targets. It does not alter
target implementations or promote one model's hyperparameters as another
model's default.

## Test Ladder

### 1. Harness and Positive Controls

Run the existing TensorFlow/XLA batch-native contract tests and compile checks.
Use an analytically known standard Gaussian and a non-identity correlated
Gaussian as transport/HMC controls. Verify:

- target value and analytic gradient against an independent reference;
- exact affine transport reconstruction and log-determinant identity;
- finite batched loss, gradients, and optimizer updates;
- sequential HMC movement, warm-up readiness, modern R-hat, bulk/tail ESS, and
  energy-error vetoes;
- recovered means/covariances and reference-law output tests.

These are harness gates. A positive-control failure blocks downstream
interpretation rather than being counted as a model failure.

### 2. Multimodal Controls

Use source-bound two-component and three-component Gaussian mixtures with known
component weights, means, and covariances. For each mixture:

- train at least two independently seeded transports under the reviewed
  target-specific protocol;
- measure mode occupancy against the known weights, not only aggregate moments;
- run the shared sequential HMC controller with `L >= 2`;
- report per-mode occupancy, cross-mode transitions, rank-normalized split and
  folded R-hat, bulk/tail ESS, energy errors, and finite/status vetoes;
- compare predictive/output distributions from posterior draws against exact
  mixture simulations using predeclared, dependence-aware diagnostics.

The three-component case is the primary multimodal generalization test. Passing
the two-component case is necessary but not sufficient.

### 3. Geometry Stress Controls

Run the varying-Hessian Gaussian surrogate, Banana, and reverse funnel. Use
separate tuning scopes and a small capacity/budget ladder per target. For each:

- retain a naive affine baseline and the tuned NeuTra candidate;
- record training trajectory, heldout objective, transport Jacobian/finite
  checks, and proposal-support diagnostics;
- perform sequential HMC only after support and handoff gates pass;
- compare output-law behavior at posterior mean and across retained posterior
  draws where the scientific question requires uncertainty propagation;
- classify failures as target/value implementation, training/capacity,
  support, tuning, or sampler failures.

Banana's existing feature decomposition is a diagnostic input, not a promotion
criterion. Reverse-funnel results require an explicit capacity and learning-rate
schedule record.

### 4. Application Targets

Complete the KSC-UKF handoff and rerun sequential HMC only from a complete,
hash-bound broad-grid artifact. Repeat the German support repair only under a
new evidence contract; the current failed support audit remains holdout
evidence. For each application target, require:

- source/data/reference hashes;
- target-specific tuning artifact and frozen controls;
- GPU memory-growth and XLA provenance;
- at least four chains under the shared sequential controller;
- warm-up readiness, retained-sample convergence, ESS, energy, movement, and
  downstream output-law checks.

### 5. Final Downstream Confirmation

Only after the controls and stress targets pass their own gates, run the final
learned scientific model as an untouched confirmation target. Do not tune on
its claim data. Compare the predictive distribution generated at a declared
posterior summary or by a declared posterior-draw protocol against the true
data-generating parameter output distribution. This is the scientific endpoint;
transport loss and HMC diagnostics alone are not sufficient.

## Promotion and Veto Rules

| Evidence | Role | Rule |
|---|---|---|
| Nonfinite target/transport/gradient, invalid hash, crash | Hard veto | Stop the lane and repair the implementation or artifact binding. |
| Missing complete handoff or stale tuning artifact | Continuation veto | Do not infer settings or launch HMC. |
| Proposal ESS below predeclared threshold | HMC veto | Preserve training evidence; do not interpret HMC metrics. |
| R-hat/ESS/energy/movement failure | Posterior veto | Candidate is not promoted; diagnose before retuning. |
| Acceptance rate, loss, runtime, isolated MMD | Explanatory | Never promote a candidate by these alone. |
| Output-law agreement with uncertainty and independent controls | Promotion evidence | Required for a scientific validity claim, still target-specific. |

No single diagnostic establishes correctness. Promotion requires all applicable
hard gates plus target-specific output-law evidence.

## Replication Policy

- Use at least four independent training/HMC seeds for each serious target.
- Keep calibration, selection, audit, and claim data disjoint.
- Use paired seeds and common budgets when comparing candidates.
- Report uncertainty or MCSE for stochastic comparisons; do not rank candidates
  from one seed or short chains.
- Keep `L >= 2`; no NUTS route is used.

## Pre-Mortem

| Failure that could look like success | Discriminating check |
|---|---|
| Low training loss but one-mode trapping | Known mixture mode occupancy and cross-mode transitions |
| R-hat near one from identical chains | Independent seeds, movement checks, ESS, and output-law comparison |
| Good aggregate moments hiding tail error | Per-mode, tail, quantile, and predictive-distribution checks |
| Proposal support passes globally but fails locally | Median/minimum batch ESS and normalized-weight diagnostics |
| Benchmark speed masks invalid XLA/device mode | Manifested memory growth, device, dtype, and XLA verification |

## Artifacts and Execution

Each rung gets a fresh artifact root with a manifest containing the exact plan,
commit, command, environment, device, memory policy, XLA mode, seeds, target
signature, hashes, wall time, and result status. Each terminal result includes
a decision table, inference-status table, strongest alternative explanation,
and next action.

The execution order is: harness controls, two-component mixture, three-
component mixture, varying-Hessian/banana/reverse-funnel, KSC/German
applications, and only then the final learned-model confirmation. A failure at
an earlier rung is a continuation veto for promotion claims, not evidence that
later models are impossible.

## Skeptical Review

The plan survives review because it uses known-law controls before application
targets, separates training/proposal/sampler/output-law evidence, requires
target-specific tuning, forbids inferred handoffs, preserves disjoint audits,
and states what each metric cannot prove. The main residual risk is compute
budget: if four-seed retained HMC is not affordable for a target, that target
must remain diagnostic rather than being promoted on a shorter run.
