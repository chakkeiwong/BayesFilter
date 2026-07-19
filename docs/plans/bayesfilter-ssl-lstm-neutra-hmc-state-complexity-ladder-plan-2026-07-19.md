# SSL-LSTM NeuTra-HMC State-Complexity Ladder Plan

Date: 2026-07-19  
Status: `PHASE_1_2_COMPLETE_RESOURCE_STOP_PENDING_GPU_AUTHORIZATION`  
Tier: Tier 3 bounded GPU/XLA research experiment

## Question And Estimand

Does the same four-coordinate Bayesian SSL-LSTM estimand remain numerically
valid and yield converged NeuTra-transformed HMC as filtering-state complexity
increases through `q in {1,2,5,10,20}`?

For rung `q`, use `latent_dim=hidden_dim=q`, scalar observation dimension,
augmented filtering-state dimension `3q`, and the general SSL-LSTM parameter
chart of size `9q^2+13q+2`. Estimate exactly these homologous coordinates:

1. `latent_mean_weight.0.0`;
2. `latent_mean_bias.0`;
3. `observation_weight.0.0`; and
4. `observation_bias.0`.

All other coordinates are fixed by a deterministic, hashed rung fixture. This
is a controlled **state/filter-complexity ladder**, not a full-parameter
dimension ladder. Estimating all 3,862 possible q=20 coefficients from 30
scalar observations would be underidentified and would not answer the stated
question. A full-parameter ladder requires a separate data-size and
identifiability design.

The q=1 rung must reproduce the existing locked four-coordinate target. The
q>1 rungs use deterministic synthetic observations generated prospectively
from their fixed full fixture at the same four-coordinate truth. Synthetic
data identity, truth, prior, horizon, and free-coordinate mask are frozen
before training or HMC.

## Research Intent Ledger

| Field | Frozen contract |
| --- | --- |
| Main question | Does NeuTra-HMC convergence survive increasing nonlinear filtering-state complexity for one controlled four-coordinate estimand? |
| Exact baseline | The locked q=1 four-parameter SVD-UKF target and its 32x32 three-stage dense-IAF NeuTra procedure. |
| Candidate mechanism | The same 32x32 three-stage dense-IAF family, Optuna-nominated training controls, plateau learning-rate repair, and separately tuned transformed HMC at every q. |
| Expected failure modes | Wrong generalized target, full-chart derivative materialization, nonfinite score, XLA/device failure, NeuTra saturation or seed instability, poor transformed geometry, HMC divergence, nonconvergence, or forecast-moment instability. |
| Primary promotion criterion | A rung passes target validity, two-seed NeuTra admissibility, transformed-target validity, and four-chain retained HMC diagnostics, then passes the declared synthetic recovery/predictive checks. |
| Promotion veto | Any required target/transport/sampler/predictive gate fails after its prospectively allowed repair. |
| Continuation veto | Invalid target math or fixture, corrupted artifact, unavailable trusted GPU/XLA path, host RSS above 64 GiB, GPU allocation failure, source drift during a rung, or exhausted declared wall/GPU budget. |
| Repair trigger | A candidate failure triggers the smallest declared repair: directional-score correction, one Optuna/plateau repair, HMC retuning, or additional retained segments. It does not reject the research direction. |
| Explanatory only | Training loss, runtime, RSS below cap, GPU allocator peak, acceptance within a viable band, jump size, and continuous cross-seed differences without uncertainty support. |
| Must not conclude | Posterior truth from another sampler, complete mode/tail coverage, full-parameter SSL-LSTM estimability, superiority of NeuTra, production readiness, or model adequacy outside the synthetic design. |

## Evidence Contract

| Role | Requirement |
| --- | --- |
| Scientific question | Whether one fixed four-coordinate Bayesian problem continues to admit valid NeuTra-HMC sampling as the latent/filter state grows. |
| Comparator | q=1 locked target; all rungs otherwise share horizon 30, scalar observations, prior SD 4, free-coordinate names/order, dtype, filter, NeuTra family, validation rules, and HMC diagnostics. |
| Primary pass | Every retained chain coordinate has rank-normalized split R-hat `<=1.01`, bulk ESS `>=400`, tail ESS `>=400`, and mean MCSE/SD `<=0.05`, with zero exposed native divergences and finite target values/scores. |
| Training admission | Both independent NeuTra seeds have finite value/score/round-trip probes, saturation `<=0.05`, and a one-sided paired 95% heldout reverse-KL improvement bound below zero. Training metrics nominate transports; they do not establish posterior correctness. |
| HMC tuning | Tune each frozen transport separately toward acceptance 0.70. Tuning samples are excluded from retained evidence. A fixed kernel is admitted only after a fresh confirmation. |
| Predictive/recovery check | On synthetic q>1 data, report truth coverage by marginal 95% intervals and standardized posterior-mean errors; compare independent retained-chain forecast laws using predeclared one-to-ten-step mean/log-variance influence statistics with MCSE-aware uncertainty. These are calibration and replication checks, not an oracle posterior. |
| Hard vetoes | Nonfinite values/scores/samples, wrong score, failed round trip, positive native divergence, any unmoved chain, invalid lineage, host RSS `>64 GiB`, GPU OOM, or artifact failure. |
| Explanatory only | Loss curves, runtime, memory below cap, acceptance, observed truth errors, and point differences without declared uncertainty. |
| Artifact | `docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/` plus a result note beside this plan. |

Passing a hard screen means the rung is viable under this design. It does not
statistically rank q values or establish that NeuTra is superior to plain HMC.

## Phase 1: Dimension-General Target And Directional Score

Implement a target that embeds four free coordinates into the rung's full
fixture and returns only the corresponding four analytic score directions.
The derivative engine must construct derivative tensors with leading dimension
four directly. It must not construct all `9q^2+13q+2` directions and gather
afterward.

Checks:

- exact q=1 value/score parity with the locked target at the prior center and
  fixed shell points;
- q=1,2,5,10,20 score finite differences on all four free coordinates;
- eager/XLA and scalar/batch parity;
- free-coordinate ordering, fixture embedding, prior, and target signature;
- q=20 isolated host/GPU memory measurement under the 64 GiB host ceiling;
- negative controls for wrong coordinate order and perturbed fixture identity.

Handoff: all target checks pass and q=20 does not allocate full-chart derivative
tensors. Otherwise write a blocker result and do not train.

## Phase 2: Timing Canary And Budget Freeze

Run one trusted GPU/XLA canary per new shape, in increasing q order. Each canary
compiles one target batch, executes 10 NeuTra training steps for each of two
independent seeds, executes a minimal transformed-HMC call, and records compile
and warm execution time plus host/GPU memory.

Project the full cost using the slower warm rate with a 50% margin and include
fresh-shape compilation. The current execution authority is bounded to Phase 1
and Phase 2 with at most **1.0 cumulative GPU-hour** and 64 GiB host RAM. A
material ladder whose projection exceeds the remaining one-hour envelope is a
resource stop requiring an explicit new GPU-hour authorization; it is not a
candidate failure.

Handoff: freeze per-rung and cumulative budgets, commands, seeds, output paths,
and sequential stop rules before material training.

## Phase 3: NeuTra Hyperparameter Nomination And Training

For each rung sequentially:

1. Use the existing 32x32, three-stage dense-IAF family with ELU activation,
   per-variable clipping, and `s_max=1`; do not substitute the historical 4x4
   capacity.
2. Run bounded Optuna nomination over learning rate `[1e-4,2e-3]` (log scale),
   initialization scale `{0.005,0.01,0.02}`, and clip norm `{5,10}` using two
   independent training/validation streams. Rungs `50,100,200,400` are
   nomination proxies only.
3. Train two fresh independent seeds under the nominated configuration to a
   maximum of 5,000 steps, batch size 480, with validation every 100 steps.
4. Use the existing statistically defined 500-step patience (five independent
   100-step validation checkpoints). If no paired heldout improvement occurs
   for one patience period, restore the best joint trainer/controller
   checkpoint and halve the learning rate. Stop only after a second 500-step
   period without improvement after that repair, at the 5,000-step maximum,
   or at a hard/resource veto. The timing canary may lengthen this to 1,000
   steps only when five validation checks cover less than ten measured warm
   target evaluations; it may never exceed 1,000 steps or vary by observed
   validation outcomes.
5. Freeze the best replayable transport from each seed. Training loss and an
   Optuna objective may nominate but cannot promote a transport.

If one seed fails, perform at most one bounded fresh-seed confirmation with the
same nominated hyperparameters. Do not search architectures within this plan.

## Phase 4: Transformed-Target Preflight And HMC Tuning

For every admitted transport:

- verify change-of-variables value and score identities, finite differences,
  forward/inverse round trip, serialization replay, and source/fixture binding;
- tune step size and leapfrog count separately with disjoint seeds;
- target acceptance is `0.70`; use a viable confirmation band `[0.60,0.80]`;
- freeze only a kernel that passes a fresh four-chain confirmation with finite
  telemetry, movement in every chain, and no exposed native divergences;
- tuning/confirmation samples are permanently excluded from retained evidence.

One adjacent trajectory-length repair is allowed for acceptance-only failure.
Geometry, target, or divergence failures return to the relevant earlier phase.

## Phase 5: Four-Chain Retained HMC With Sequential Stopping

For each of the two independent transports, acquire immutable four-chain
segments of 256 retained draws per chain. Burn in 256 transitions on the first
segment only. Evaluate cumulative checkpoints at `512,1024,2048,4096` retained
draws per chain. Continue after a diagnostic miss unless a hard/resource veto
fires.

A chart passes only at a checkpoint where all free coordinates satisfy:

- rank-normalized split R-hat `<=1.01`;
- rank-normalized bulk ESS `>=400`;
- rank-normalized tail ESS `>=400`;
- posterior-mean MCSE/SD `<=0.05`;
- finite mapped and transformed samples/value/score;
- zero exposed native divergences and movement in every chain.

Compare the two admitted charts in the common four-coordinate theta chart.
Require the absolute difference in each of four means and ten raw second
moments to be at most three combined MCSEs. This is a replication-stability
screen, not an oracle comparison or equivalence proof.

## Phase 6: Recovery And Predictive-Moment Validation

For q>1 synthetic rungs, report marginal 95% interval coverage of the frozen
truth and standardized mean error `abs(E[theta_j]-theta*_j)/posterior_sd_j`.
These diagnostics are descriptive for one synthetic dataset and cannot prove
frequentist calibration. A future multi-dataset SBC program is required for
that claim.

Using fresh stateless forecast innovations, compare the two independently
trained/sampled charts over horizons 1 through 10 using posterior-predictive
means and log variances. Use the documented influence-function/HAC procedure
and frozen calibration scales. Pass only when no predeclared MCSE-aware region
rejects replication stability. Preserve path plots from dispersed retained
draws as intuitive explanatory evidence.

## Sequential Program Stop And Handoff

Execute `q=1,2,5,10,20` in order. A target-invalidity or resource veto stops
the program. A NeuTra or HMC failure rejects that rung under the current
candidate and records the exact repair trigger; it does not claim the research
direction is invalid. Later rungs normally stop after a failed rung because
the ordered ladder no longer supports a clean complexity boundary, unless the
failure is repaired within the prospectively allowed single repair.

The 64 GiB host-RAM ceiling is `68,719,476,736` bytes. Record both isolated
process `ru_maxrss` and TensorFlow GPU allocator peak. The GPU allocator is
limited by the physical device; 64 GiB is not a GPU-memory claim. Run rungs in
fresh processes so host high-water marks and XLA caches are isolated.

## Skeptical Pre-Execution Audit

| Risk | Finding and repair |
| --- | --- |
| Wrong baseline | Repaired: q=1 is the locked four-coordinate target; q>1 uses the same coordinate semantics and prior. |
| Proxy promoted | Repaired: Optuna/loss/canaries only nominate or veto training mechanics; retained four-chain diagnostics and predictive replication are primary. |
| Hidden estimand drift | Repaired: q scales state/filter complexity while the estimated block remains four homologous coordinates. Full-parameter estimation is explicitly out of scope. |
| Underidentified comparison | Repaired: do not estimate 64--3,862 coefficients from 30 scalar observations. |
| Memory cap masks bad algorithm | Repaired: directional derivatives must have leading dimension four; full-chart score materialization is forbidden. |
| Missing stop | Repaired: 64 GiB host ceiling, physical GPU OOM, one-hour preflight authority, canary projection, per-rung sequential gates, and artifact/source failures are explicit. |
| Unfair comparison | Horizon, observation dimension, free coordinates, prior, filter, NeuTra family, seed count, HMC gates, and maximum retained opportunity are held fixed. Synthetic observations necessarily differ by q and are hashed before fitting. |
| No posterior oracle | Repaired: peer transport replication, truth recovery on synthetic data, forecast moments, and future SBC replace parameter agreement with an unavailable oracle. |
| Stale scalar code | Material: existing Phase 5--7 runners hard-code dimension four and the q=1 locked target. New dimension-general target/runner work is required before execution. |
| Misleading pass | Repaired: engineering, training, sampler, and scientific ledgers remain separate; no rung is called converged from short timings. |

Audit decision: `PASS_AFTER_TARGET_AND_DIRECTIONAL_SCORE_REPAIR`. Phase 1 and
the q=1/q=2 portion of Phase 2 completed. The Phase 2 resource gate fired
before material training: q=1 alone projects 30.9 GPU-hours for two 5,000-step
streams before Optuna, HMC tuning, retained chains, or predictive validation.
No q>1 material NeuTra/HMC run is authorized by this plan until a new GPU-hour
budget is explicitly approved and the per-rung ladder budget is refreshed.

## Required Result Record

The result note must include a run manifest, per-rung decision table, inference
status table, exact commands/environment/device/JIT/TF32/seeds/wall/RSS/output
paths, hard vetoes, viable candidates, whether any ranking is supported,
descriptive-only differences, the strongest alternative explanation, what
would overturn the conclusion, and the next evidence needed.

## Phase Close

Phase 1 target/score repair and q=1/q=2 mechanics canaries completed. The
proper end-to-end ladder did not execute: no 5,000-step NeuTra training,
Optuna study, transformed HMC tuning, retained four-chain acquisition, or
predictive validation was run. Continue only after recording a new GPU-hour
authorization and a revised resource ledger; the 64 GiB host-RAM ceiling is
not the active bottleneck.
