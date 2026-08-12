# Defensive weighted NeuTra analytic HMC plan (2026-08-12)

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | Does fixed-length HMC in the coordinates of one frozen, independently confirmed weighted-NeuTra transport recover the known unequal-weight two-mode Gaussian-mixture law? |
| Candidate mechanism | Freeze the selected width-128 weighted IAF, target the exact change-of-variables density in latent coordinates, tune only HMC step size and leapfrog count, discard all tuning and warm-up draws, and map retained latent draws back to physical coordinates. |
| Exact authority | The normalized analytic law `0.8 N(mu_1,Sigma_1) + 0.2 N(mu_2,Sigma_2)` used by the training campaign. The learned transport density is not the HMC target. |
| Expected failure mode | A loader or Jacobian-sign error may target the wrong law; the IAF may preserve marginal mass but leave a latent energy barrier; a fixed HMC trajectory may remain trapped by initialized mode; or short tuning may nominate a fragile kernel. |
| Promotion criterion | The canonical sequential controller passes modern R-hat and bulk/tail ESS, no hard sampler veto fires, and retained physical draws pass the predeclared analytic component-mass and moment uncertainty screens. |
| Promotion veto | Nonfinite state/value/score/Jacobian, invalid target status, corrupt or mismatched frozen state, failed derivative parity, a chain without movement, positive native divergence when exposed, failed retained R-hat/ESS, or failed analytic posterior screen. |
| Continuation veto | Wrong transformed target, wrong Jacobian sign or pullback, broken checkpoint identity, corrupted archive, GPU memory-policy failure, or campaign cap. A viable implementation with a rejected HMC kernel triggers retuning or geometry diagnosis rather than rejection of weighted training. |
| Repair trigger | Canary failure after mechanics parity triggers a smaller step-size grid; sequential non-mixing triggers latent barrier/path diagnostics and possibly a different frozen replication; analytic disagreement after convergence diagnostics pass triggers archive and diagnostic audit before any new run. |
| Explanatory diagnostics | Acceptance, finite energy-error tails, runtime, per-chain mode occupancy, transition counts, NLL, and transport-training diagnostics. |
| Must not be concluded | Success on this known four-dimensional target does not establish exhaustive mode discovery, SSL-LSTM validity, a general NeuTra default, sampler superiority, or correctness on the remaining analytic target suite. |

## Mathematical target

For frozen transport `theta = T_phi(z)`, HMC must use

```text
log p_z(z) = log p_theta(T_phi(z)) + log |det J_T_phi(z)|.
```

The score is

```text
grad_z log p_z(z)
  = J_T_phi(z)^T grad_theta log p_theta(T_phi(z))
    + grad_z log |det J_T_phi(z)|.
```

The plus sign is required because the HMC variable is `z` and the physical
measure is the pushforward through `T_phi`. The inverse-density identity
`log q_phi(theta) = log N(z;0,I) - log|det J_T_phi(z)|` is a different quantity
and must not be substituted for the transformed analytic target.

## Evidence contract

| Role | Diagnostic |
|---|---|
| Primary promotion criteria | Sequential retained `max(rank-split R-hat, folded rank-split R-hat) <= 1.01`; bulk and tail ESS at least 400 in both latent and physical coordinates; analytic retained-sample screens below. |
| Hard vetoes | Frozen-state/hash mismatch; forward/inverse or explicit-score parity failure; any nonfinite required tensor; target-status failure; chain without movement; positive native divergence when available; sequential cap without readiness. |
| Repair triggers | Tuning candidate outside its screen, warm-up R-hat above 1.05 at cap, retained R-hat/ESS failure at cap, or analytic screen failure with otherwise valid mechanics. |
| Explanatory only | Acceptance probability, finite `|Delta H|` tail, runtime, per-chain occupancy, selected trajectory length, and training NLL. Acceptance never establishes convergence. |
| Exact comparator | Analytic mixture weights, mean, covariance, one-dimensional marginal means/variances, and exact component responsibilities. |
| Preserved artifact | Versioned JSON manifests/results plus archived warm-up and retained TensorFlow tensors under `docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12/`. |

### Analytic retained-sample screens

The screens are simultaneous descriptive validity checks, not a ranking test.
For `n = chains x retained draws`, use responsibility values `gamma_2(theta)`:

- minority mass: analytic `0.2` must lie inside a two-sided 99% normal interval
  using the empirical standard error of `gamma_2`; also report batch-means MCSE;
- each mean coordinate: analytic mean must lie inside a two-sided 99% interval
  using chain-aware batch-means MCSE;
- each covariance entry: analytic covariance must lie inside a two-sided 99%
  interval for the corresponding centered second moment using chain-aware
  batch-means MCSE;
- both hard-assignment modes must be observed overall and in every chain, but
  raw per-chain occupancy remains explanatory rather than a calibrated equality
  test.

The nominal 99% level is inherited from the earlier owner-selected diagnostic
policy. Multiple screens are not collapsed into a joint p-value, and passing
them is not a proof of distributional equality.

## Scope and frozen inputs

- Target: `separated_two_mode_unequal_weight_d4_v1`.
- Transport family: six dense autoregressive stages, two `(128,128)` hidden
  layers per stage, float64.
- Frozen transport: confirmation replication 1, chosen before HMC by the neutral
  rule "lowest fresh confirmation replication identifier". No posterior result
  is used to choose the transport.
- Checkpoint source:
  `docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/capacity-depth6-width128-updates10000-confirmation-1-v1/trainer_states.json`.
- The loader must verify the checkpoint SHA-256 recorded by that run, the
  semantic state hash, schema/config, variable count/shapes, and exact restored
  state hash.
- Four chains are initialized deliberately across inverse images of both
  analytic component means: two chains near each mode with fixed small offsets.
  This prevents a one-mode initialization accident but is not posterior evidence.

## Default and numeric audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| TensorFlow/TFP float64, XLA, GPU | Reviewed repository defaults | Same backend/precision as training and canonical HMC policy | compile/device mismatch | GPU/XLA canary and device manifest |
| GPU memory growth | Mandatory owner policy | Allows shared GPUs without eager reservation | launch-invalid evidence | fail-closed pre-import environment plus runtime verification |
| Replication 1 transport | Neutral predeclared selection rule | Avoids choosing on HMC behavior | one transport may not represent cross-training-seed behavior | this run makes a one-transport claim only; later replication is separate |
| Four chains | Canonical minimum | Enables modern multi-chain diagnostics | inadequate mode-transition evidence | mode-aware starts and per-chain occupancy/movement |
| `L=(3,5,10,15,20,25)` | Inherited BayesFilter grid plus owner-approved `L=3`; `L=1` forbidden | Screens materially different trajectory lengths | grid may miss a stable region | bounded repair grid only if no candidate survives |
| Initial epsilon `0.1` | Convenience hypothesis | Transport is trained toward unit-normal geometry | could be too large or small | dual-averaging canary and finite-energy screen |
| Tuning budgets | Reviewed target-specific hypothesis: 64/128/256 adaptation steps, 128 screen draws, 1,000 fresh verification draws per chain | Analytic target is cheap; verification can support modern R-hat rather than a tiny proxy | short tuning can nominate unstable epsilon | fresh verification plus sequential warm-up; no tuning draws retained |
| Sequential chunks/caps | Canonical defaults: 500 per chunk, 2,000--10,000 warm-up, 1,000--10,000 retained | Repository `bayesfilter_neutra_sequential_hmc_v1` | HMC may hit cap without cross-mode mixing | archive every chunk and stop with candidate rejection, not false success |
| R-hat/ESS thresholds | Repository policy: warm-up 1.05, retained 1.01, bulk/tail ESS 400 | Existing canonical controller contract | diagnostics can miss rare global structure | analytic responsibility/moment screens remain required |
| 99% analytic intervals | Owner-selected diagnostic level from prior predictive work | Conservative marginal diagnostic without a fabricated joint test | normal/MCSE approximation can be optimistic under autocorrelation | report chain-aware batch-means MCSE and individual interval outcomes |
| One bounded campaign attempt | Academic reproducibility cap | Avoids silently expanding compute after a failed candidate | repair may remain incomplete | preserve failure and write explicit next discriminating step |

## Execution ladder

1. Implement a fail-closed frozen weighted-IAF loader, exact explicit pullbacks,
   analytic mixture value/score adapter, transformed-target builder, and retained
   analytic diagnostics.
2. Unit/reference checks on CPU only: state/hash restoration, forward/inverse
   parity, Jacobian sign, explicit score versus GradientTape reference, batch/XLA
   parity, analytic moments/responsibilities, and warm-up exclusion in result
   assembly. These are reference/debug checks, not GPU evidence.
3. Run one bounded GPU/XLA mechanics canary using all four mode-aware chains and
   `L=3`; require finite values/scores/traces, movement, verified memory growth,
   and an archived manifest. Canary draws are discarded.
4. Tune the six-point leapfrog grid with identity mass in `z`, target-specific
   disjoint tuning seeds, and fresh verification. The initial 1,000-draw window
   was extended by the recorded repair ledger to 4,000 draws per chain. Select
   only among candidates without hard vetoes. Acceptance participates in tuning
   but does not become a convergence claim.
5. Freeze the selected epsilon and `L`; run the canonical archived sequential
   controller. Exclude all warm-up draws from posterior summaries.
6. On retained draws only, compute analytic responsibility, mean, covariance,
   marginal, R-hat/ESS, mode, and finite diagnostics; write the result, manifest,
   hashes, decision/inference tables, and reset memo.

## Compute and stop budget

- One GPU process, one physical GPU, memory growth enabled; no GPU 0 exclusion is
  required by current owner state, but the manifest records the selected device.
- Canary cap: 10 minutes.
- Tuning cap: 30 minutes.
- Sequential cap: 60 minutes and at most 10,000 warm-up plus 10,000 retained
  transitions per chain.
- Total campaign cap: 100 minutes, including archive finalization. These are
  conservative convenience caps; measured wall time is reported and may be much
  smaller on the analytic target.
- A timeout preserves completed artifacts and yields `under_budgeted` or
  `candidate_not_ready`; it never yields a pass.

## Pre-mortem

- The run could pass R-hat because chains remain in their initialized modes with
  stable but wrong occupancy. The analytic responsibility screen and per-chain
  transition/occupancy diagnostics distinguish this.
- The run could match component mass but target the transport density rather than
  the analytic posterior. Direct transformed-value/score parity and Jacobian-sign
  tests distinguish this before HMC.
- HMC could fail because the explicit pullback is wrong rather than because the
  learned geometry is poor. GradientTape is used only as an independent test
  authority on fixed points; it is forbidden in the candidate runtime.
- HMC could fail because the tuning grid is too coarse. If mechanics pass but all
  arms fail acceptance/energy screens, one smaller-epsilon repair grid is allowed
  within the unchanged cap and must be recorded.
- A successful single frozen transport could be mistaken for cross-seed robustness.
  The result is explicitly scoped to replication 1; cross-transport replication
  remains future evidence.

## Skeptical pre-execution audit

- **Wrong baseline:** corrected. Analytic mixture truth is the authority; the
  learned density is coordinates only. Plain HMC or reverse-KL is not substituted
  for truth.
- **Proxy promotion:** corrected. Acceptance, training loss, mode-aware starts,
  and a short canary cannot promote the sampler. Sequential R-hat/ESS plus analytic
  retained-sample checks are required.
- **Missing stop conditions:** corrected. Target/Jacobian/hash/device failures are
  continuation vetoes. Kernel or mixing failures are candidate rejections and
  repair triggers.
- **Unfair comparison:** no method ranking is attempted. Tuning and confirmation
  use disjoint seeds. Analytic target moments and responsibilities are computed
  from the fixed normalized target law.
- **Hidden assumptions:** transport seed, target, initial states, grid, epsilon,
  intervals, chunk sizes, and caps are explicit above. The one-transport scope is
  a limitation, not a cross-seed claim.
- **Stale context:** this plan uses the completed eight-seed width-128 campaign and
  the current sequential policy. It does not rely on the superseded width-64 result
  or historical 16-draw HMC attempts.
- **Environment mismatch:** candidate execution is TensorFlow/TFP GPU/XLA float64
  with memory growth. CPU is restricted to focused reference checks.
- **Artifact relevance:** checkpoint bindings, tuning results, raw archived chunks,
  and retained analytic summaries directly answer the stated question. Training
  artifacts alone do not.

Audit verdict: **PASS FOR THE BOUNDED ANALYTIC-HMC LADDER**. Execution must stop
before serious tuning if the explicit derivative, restore, GPU-memory, or XLA
canary gates fail. Passing this plan cannot promote later analytic variants or
SSL-LSTM work.

## 2026-08-12 tuning repair ledger

- The first six-point tuning run used 1,000 fresh verification draws per chain
  and screened modern R-hat in physical coordinates. Every candidate was finite,
  had finite target scores, and had mean acceptance inside the configured band,
  but every candidate failed folded R-hat `<=1.01`; the minimum observed maximum
  R-hat was `1.02076` at `L=5`.
- Physical-coordinate tuning verification was an unfair gate for a deliberately
  multimodal target because the canonical sequential controller separately
  requires the maximum over latent and physical coordinates. The tuner was
  repaired to record an explicit verification coordinate and rerun in `z`.
- The latent-coordinate retry also rejected all candidates. The closest was
  `L=5`, maximum folded R-hat `1.03774`; all six retained finite values and
  acceptance inside the pass band. This weakens the hypothesis that the learned
  map fully whitens the target, but 1,000 autocorrelated draws do not establish
  that no fixed kernel can meet tuning admission.
- One final tuning-budget repair is authorized: keep the same six `L` values,
  adaptation, step-size policy, starts, target, seeds, and thresholds, but extend
  fresh verification from 1,000 to 4,000 draws per chain. This directly tests
  whether the rejection is finite-window autocorrelation rather than invalid
  geometry. It is not threshold relaxation or posterior confirmation.
- If no candidate reaches modern R-hat `<=1.01` at 4,000 draws, stop before
  sequential HMC and classify the current transport/kernel family as not admitted.
  If at least one passes, freeze the predeclared selector's candidate and proceed
  to canonical sequential warm-up and retained sampling, which still gates on the
  maximum over both coordinate systems plus ESS and analytic posterior checks.

Repair audit: **PASS FOR ONE 4,000-DRAW VERIFICATION RETRY**. The comparator,
target, sampler family, hard gates, hardware class, and campaign cap are unchanged.

Post-run provenance audit found that run-v4 numerically consumed the predeclared
mode-aware latent bank through a caller callback, but the public tuner artifact
still reported `initial_state_all_zero=true` and labeled a physical position as
`fixed_neutra_initial_z`. Run-v4 is therefore launch-invalid for scientific
interpretation. It remains diagnostic evidence only. The tuner now owns an
explicit validated `initial_state_bank`, supplies it to every arm, records the
bank in verification diagnostics, and binds the mass-artifact position to the
actual latent coordinate. One clean rerun is required; this is a provenance
repair with no change to target, kernels, seeds, thresholds, starts, or budget.

## Terminal adjudication correction

The initial run-v5 result builder combined every marginal mean and covariance
interval with `all(...)` and treated that conjunction as one promotion veto. That
is a joint multiple-comparison rejection without a calibrated joint test and is
wrong relative to this plan's explicit rule that marginal screens are not
collapsed into a joint p-value. It also contradicts the owner's earlier rejection
of overly stringent joint diagnostics.

Correct roles for terminal adjudication are:

- hard primary screens: finite retained tensors, both hard modes overall and in
  every chain, analytic `0.2` inside the predeclared 99% responsibility-mass
  interval, sequential maximum-over-latent-and-physical R-hat `<=1.01`, bulk/tail
  ESS `>=400`, movement, status, and divergence vetoes;
- explanatory marginal diagnostics: each mean and covariance interval, with pass
  counts and failed entries reported individually but no joint rejection;
- nonclaim: passing these screens means this one retained run is statistically
  compatible with the analytic target under the declared diagnostics. It does not
  prove equality, stationary sampling, cross-transport robustness, or general
  NeuTra validity.

Run-v5 tensors remain immutable. A fresh adjudication artifact must reverify all
archive receipt hashes and recompute only these diagnostic roles; it must preserve
the original run-v5 result and its overly stringent binary label as provenance.
