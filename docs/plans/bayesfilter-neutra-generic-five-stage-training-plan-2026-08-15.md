# Generic NeuTra five-stage training plan (2026-08-15)

## Research intent ledger

| Field | Predeclared statement |
|---|---|
| Main question | Can the successful reverse-funnel continuation idea be expressed as a target-agnostic Python training controller, and does the same structural procedure remain viable on independent known-law targets? |
| Candidate | A callback-driven TensorFlow/XLA five-stage controller with named variable groups: affine shift, simple autoregressive linear scale, first nonlinear block, progressive block addition, joint fine-tuning, followed by untouched validation. |
| Baseline | Independently tuned cold-start joint reverse-KL using the identical transport, batch, update budget, initialization family, and target. |
| Primary criterion | Separate untouched known-law proposal diagnostics for each model. Training loss selects checkpoints and LR candidates only; it cannot establish distributional validity. |
| Hard vetoes | Non-finite loss/gradient/state, empty or duplicate variable groups, optimizer update outside the declared group, failed XLA/batch-native path, broken exact target/sampler, or invalid artifact. |
| Continuation veto | Broken generic controller, target, sampler, or validation authority. A failed model candidate is a repair or scope result, not a reason to stop later planned models. |
| Explanatory diagnostics | Held-out reverse KL, gradient/clipping trace, group update norms, proposal importance ESS, log target/proposal-ratio variation, and model-specific moments. |
| Must not conclude | Passing these controls does not establish a universal NeuTra recipe, SSL-LSTM correctness, multimodal coverage, HMC readiness, or a repository default. |

## Generic API contract

The new repository function must not contain target names, funnel indices,
known coefficients, target formulas, or model-specific thresholds. It receives:

1. an invertible trainable transport and batched target log-density;
2. named, non-overlapping variable groups supplied by an adapter;
3. four training-stage specifications, where the progressive stage may contain
   multiple subphases;
4. a stateless batched latent supplier;
5. a disjoint held-out selection loss callback; and
6. an untouched final validation callback.

The returned result records stage/subphase checkpoints, selected states,
learning rates, trainable variable names, update counts, finite/clipping status,
validation output, and nonclaims. The controller uses TensorFlow only, keeps the
batch dimension native, and JIT-compiles each fixed variable-subset update.

## Default dense-IAF five-stage recipe

| Stage | Trainable structure | Purpose |
|---|---|---|
| 1. Affine | first-stage output shift bias only | establish location without permitting immediate global root-scale collapse |
| 2. Simple transport | affine group plus explicit strictly autoregressive linear log-scale path | learn first-order conditional scale before nonlinear co-adaptation |
| 3. Progressive | first full IAF block, then cumulative blocks one at a time | add nonlinear capacity through continuation |
| 4. Joint | every transport variable | remove freezing constraints after a viable basin is reached |
| 5. Validation | no optimizer update | untouched target-specific proposal-law audit |

This recipe is a hypothesis, not a universal default. In particular, shift-only
stage 1 is chosen because unrestricted affine reverse-KL training is known to
compress the funnel root. That choice may underfit nonzero scale geometry, which
the stage-2 and later diagnostics must expose.

## Model ladder and evidence contracts

### A. Paper reverse funnel, dimension 100

- Exact sampler and target already checked.
- Staged candidate uses three width-100 IAF blocks and an unbounded linear scale
  path in block 0. It may not use the known coefficient one.
- Baseline: tuned cold-start three-block joint route from the completed campaign.
- Gate: separate 99.9% intervals for root mean/second moment, standardized child
  residual mean/second moment, and both `|y|>2` tails.

### B. Correlated Gaussian, dimension 16

- Fixed deterministic mean and Cholesky factor generated in TensorFlow/Python.
- Same transport recipe; no exact affine factor is supplied to training.
- Gate: separate 99.9% intervals for whitened aggregate mean, second moment,
  maximum coordinate mean, and covariance error, plus log-ratio variation.

### C. Banana pushforward, dimension 16

- Exact law: `x0=z0`, `x1=z1+b*(z0^2-1)`, remaining coordinates unchanged.
- The target and exact sampler use the analytic inverse and unit Jacobian.
- Gate: transformed latent mean/second moment and selected marginal moments on
  untouched proposal draws, plus log-ratio variation.

### D. Symmetric three-component Gaussian mixture, dimension 2

- Boundary diagnostic, not an expected pass. Exact iid mixture sampling is
  available.
- Gate: component-region masses and basic moments on untouched proposal draws.
- A failure weakens generalization to separated multimodality; it does not
  invalidate success on unimodal controls or the generic controller.

## Tuning and budgets

Each phase and each cold baseline tune their peak LR independently over
`2e-4`, `5e-4`, and `1e-3`, using piecewise multipliers `1`, `0.1`, `0.01` at
60% and 85% of each phase. These values are bounded hypotheses from the funnel
campaign, not transferred defaults. The maximum selected training path is 1,000
updates for each low-dimensional control (`100 + 300 + 3*100 + 300`) and 5,000
for the funnel (`250 + 2000 + 3*500 + 1250`). The cold baseline receives the
same per-model update budget. Tuning work is larger because every phase evaluates
all three rates from the same incoming checkpoint; artifacts report both tuning
optimizer updates and selected-path updates. Batch size is 4,096. One matched
seed is diagnostic; a second seed is required only for a model where either
route passes the first untouched gate.

## Skeptical plan audit

| Risk | Audit disposition |
|---|---|
| Wrong baseline | Repaired: compare against cold joint training with the same transport and independently tuned LR. |
| Funnel knowledge hidden in generic code | Vetoed: target-specific indices and validation stay in the experiment adapter; the controller sees only variable groups and callbacks. |
| Proxy promoted to correctness | Vetoed: held-out reverse KL selects candidates; untouched known-law diagnostics decide viability. |
| Five stages only rename one optimizer loop | Vetoed: tests must show only declared variables change in each phase and optimizer state is phase-local. |
| Architecture change without retuning | Vetoed: staged and cold routes tune independently per model. |
| One success claimed as universal | Vetoed: conclusions are model-by-model; mixture is an explicit boundary test. |
| Run passes while selection data leak into audit | Vetoed: calibration, selection, and final audit use disjoint stateless seeds. |
| Failure caused by insufficient budget | Retain curves and terminal gradients; classify as undertrained if improvement is material at the cap. |
| Concurrent GPU work distorts runtime | Runtime remains descriptive; record device occupancy and memory growth. |

Audit verdict: execution is justified after focused mechanics tests pass. The
artifacts answer whether the controller is generic, whether staged training is
viable on each known-law model, and where it does not generalize. They do not
answer SSL-LSTM or HMC validity.

## Execution

1. Implement the generic controller and dense-IAF group adapter.
2. Test group validation, phase-local updates, checkpoint restoration,
   determinism, finite failure handling, and no-update validation.
3. Run a short GPU/XLA canary on Gaussian and funnel.
4. Tune and confirm funnel, Gaussian, banana, and mixture against matched cold
   baselines.
5. Write result and reset notes with decision and inference-status tables.

## Artifact root

`docs/plans/artifacts/neutra-generic-five-stage-training-2026-08-15/`
