# Defensive weighted NeuTra validation plan (2026-08-11)

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | Can a defensive, target-weighted forward-KL training route learn a NeuTra map that preserves multimodal coverage and improves exact transformed-target geometry where ordinary reverse-KL training collapses to one mode? |
| Candidate mechanism | Generate globally covering weighted particles from a normalized defensive proposal, train an invertible TensorFlow transport by weighted maximum likelihood, retain defensive proposal support during later annealed-SMC refinement, and use exact Jacobian-corrected HMC only after transport gates pass. |
| Exact baseline | Plain BayesFilter reverse-KL NeuTra with the same trainable family, initialization seed, optimization budget, and target. Analytic target laws and component probabilities are the authority on synthetic rungs; plain NeuTra is a method comparator, not truth. |
| Expected failure mode | Importance weights may collapse; the inverse/log-density gradient may be wrong; a single continuous flow may cover modes but leave poor latent geometry; capacity selection may overfit one weighted cloud; later target integrations may fail despite analytic success. |
| Promotion criterion | Each rung has its own primary criterion below. No later rung starts unless all prior hard gates pass. Training or validation loss alone never promotes a transport. |
| Promotion veto | Wrong inverse/Jacobian gradient, invalid target value/score, nonfinite state, missing known component, truth outside the declared independent-run interval, failed exact corrected-HMC agreement, or a held-out weighted diagnostic that invalidates the selected map. |
| Continuation veto | Broken target/reference identity, corrupted artifacts, unavailable required GPU memory growth, attempt/campaign cap, or evidence that the implemented quantity differs from weighted forward KL. A failed candidate or capacity arm is a repair trigger, not automatically a direction veto. |
| Repair trigger | Weight degeneracy triggers proposal/annealing repair; coverage with poor geometry triggers capacity or componentwise-map repair; paper-target regression triggers objective-mixture review; surrogate failure after prior passes triggers target-specific tuning, not threshold relaxation. |
| Explanatory diagnostics | Loss curves, latent moments, curvature spectra, base-to-physical coverage, runtime, allocator peak, HMC acceptance, ESS, and tail summaries unless explicitly promoted by a rung contract. |
| Must not be concluded | Analytic-rung success does not prove SSL-LSTM validity; weighted particles are not an unweighted posterior archive; balanced proposal counts are not posterior weights; successful local maps do not prove exhaustive mode discovery. |

## Evidence contract by rung

| Rung | Targets | Primary evidence | Promotion veto | Artifact root |
|---:|---|---|---|---|
| 0 | Correlated and ill-conditioned Gaussian | Trainable inverse/log-density gradient agrees with analytic or finite-difference authority; held-out weighted mean/covariance and exact affine law agree with truth | Any inverse, Jacobian, normalization, XLA, finite, or exact-moment failure | `docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r0-gaussian/` |
| 1 | Equal-weight and unequal-weight two-component Gaussian mixtures with equal and unequal rotated covariances | Every known component is represented; exact component probabilities lie in predeclared independent-replication intervals; candidate held-out weighted NLL and corrected HMC pass | Missing component, collapsed weights, failed independent replication, or invalid transformed target | `.../r1-two-mode/` |
| 2 | Asymmetric four-component Gaussian mixture | All material components represented and exact four-way probabilities recovered with uncertainty | Any material component omitted or false binary-only partition logic | `.../r2-four-mode/` |
| 3 | NeuTra paper ill-conditioned Gaussian, funnel, and German-credit target in `~/python/dsge_hmc` | Enhanced route does not regress the repository's plain NeuTra posterior/reference and HMC gates at matched budgets | Earlier analytic gate open; target/reference mismatch; posterior/reference or sampler veto | `.../r3-paper-suite/` |
| 4 | Exchangeable Bayesian Gaussian mixture and an asymmetric-prior variant | Symmetric labeling probability and low-dimensional quadrature/reference summaries recovered | Label mode omitted, symmetry/weight reference rejected, or invariant posterior summaries fail | `.../r4-label-switching/` |
| 5 | BayesFilter LGSSM parameter posterior | Exact Kalman/dense-grid parameter posterior and transformed-HMC agreement | Kalman/reference disagreement or target-integration failure | `.../r5-lgssm/` |
| 6 | `nk_like_mild_smooth` and `nk_like_strong_smooth` from `~/python/dsge_hmc` | Exact transformed-HMC diagnostics and posterior/reference agreement under matched target-evaluation budgets | Earlier gate open or any sampler/reference veto | `.../r6-dsge-surrogates/` |
| 7 | Reduced SSL-LSTM preserving the q=20 parameter roles | Dense-grid, quadrature, or separately validated weighted reference agreement plus multimodal coverage | Reduced target does not preserve the mechanism or reference is unresolved | `.../r7-reduced-ssl-lstm/` |
| 8 | SSL-LSTM q=20 | Agreement between independent weighted path and exact retained sampling, followed by posterior-predictive diagnostics | Unresolved weights, warm-up-only samples, failed convergence, or predictive input ineligibility | `.../r8-q20/` |

## First executable campaign: rungs 0--2

### Comparators

1. Plain reverse-KL NeuTra.
2. The normalized defensive proposal without transport refinement.
3. Defensive weighted forward-KL NeuTra.
4. Optional forward-KL followed by reverse-KL refinement only after the plain weighted candidate passes coverage; refinement may not erase a known component.

### Exact synthetic suite

The first implementation supports normalized full-covariance Gaussian mixtures in
TensorFlow. The initial campaign uses dimension four so tensor shapes match the
SSL-LSTM parameter dimension. Frozen target families are:

- one correlated Gaussian with reviewed covariance;
- equal-weight two-mode, unequal covariance;
- target weights `(0.8,0.2)` with balanced defensive proposal;
- target weights `(0.95,0.05)` as a rare-material-mode stress test;
- four asymmetric components with unequal weights and rotated covariances.

These numbers are test hypotheses selected to span easy, material-minority, and
rare-minority cases. They are not universal defaults. Exact target probabilities,
moments, density, and score are analytic.

### Split and replication policy

- Training, checkpoint selection, audit, base-coverage, and exact-HMC random streams
  are disjoint stateless seed domains.
- At least four independent weighted-particle replications are required for a
  descriptive canary and eight for a claim-bearing exact-weight interval.
- Checkpoint selection uses held-out weighted negative log likelihood. The final
  audit is untouched by selection.
- Synthetic component labels are retained only as analytic diagnostics; training
  consumes physical rows and normalized importance weights, not labels.

### Numeric and budget provenance

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Dimension 4 | Target-matched convenience hypothesis | Matches SSL-LSTM parameter dimension while retaining analytic truth | Misses higher-dimensional weight collapse | Later moderate-dimensional analytic extension before paper suite if needed |
| Balanced defensive component weights | Coverage hypothesis | Ensures all known components are proposed; importance correction supplies target weights | High weight variance for extreme target imbalance | ESS fraction and maximum normalized weight per replication |
| TensorFlow `float64`, XLA | Repository reviewed default for this scientific lane | Matches target/NeuTra precision and differentiability | GPU kernels may compile slowly | Unit gradient test and one bounded GPU canary |
| GPU training | Repository NeuTra policy | Serious learned transport work belongs on GPU | Memory policy or device mismatch | Fail-closed memory-growth and placement receipt |
| CPU-only unit tests | Explicit reference/debug exception | Small analytic mechanics do not require GPU | Cannot support throughput claims | Label unit-only and run GPU canary before scientific interpretation |
| Training budgets | Unproven tuning hypotheses until r0 canary | No measured weighted-forward-KL curve exists yet | Under/overtraining | Small budget ladder with disjoint held-out curves; no fixed number promoted in advance |
| ESS/coverage thresholds | Must be calibrated on analytic truth | Avoid unsupported sacred constants | False pass or excessive rejection | Report continuous diagnostics first; bind thresholds only after r0/r1 canary review |

### Skeptical pre-execution audit

- **Wrong baseline:** corrected. Analytic laws are truth; plain reverse KL is the
  method baseline. The defensive proposal is retained separately so gains are not
  falsely attributed to training.
- **Proxy promotion:** corrected. Training loss and held-out weighted NLL may select
  checkpoints but cannot establish posterior correctness. Exact weights/moments and
  exact corrected-HMC behavior are required.
- **Missing stop conditions:** corrected. Nonfinite calculations, wrong gradients,
  missing components, weight collapse, device-policy failure, and caps stop the
  current rung. Candidate failure triggers repair unless the harness/reference is
  invalid.
- **Unfair comparison:** matched architecture, seeds, optimization updates, and
  target-evaluation accounting are required where plain reverse KL is compared.
- **Hidden assumptions:** known-mode defensive proposal, flow capacity, optimizer,
  weight clipping policy, resampling policy, thresholds, and budgets remain explicit
  hypotheses. Weight clipping is forbidden in the exact synthetic authority path.
- **Environment mismatch:** BayesFilter implementation is TensorFlow-only and writes
  only in this repository. `~/python/dsge_hmc` is inspected/read-only in this
  campaign; its benchmark execution begins only after the analytic gate and under
  its own repository instructions.
- **Artifact relevance:** r0/r1 artifacts contain exact target parameters, proposal
  parameters, independent seeds, raw weight diagnostics, trainer identity, device
  receipt, held-out results, and exact truth comparison. They directly answer the
  first-gate question.

Audit verdict: **PASS FOR IMPLEMENTATION AND UNIT/BOUNDED CANARY ONLY**. It does
not authorize jumping to paper, DSGE, or SSL-LSTM targets before terminal analytic
gate review.

## 2026-08-11 canary repair ledger

- The r0 Gaussian canary passed all exploratory gates at 500 updates per arm.
- The first r1 `(0.8, 0.2)` two-mode candidate represented both modes but assigned
  only `0.1298` base-pushforward mass to the minority component. The corrected
  importance cloud was healthy (ESS fraction `0.7357`, maximum normalized weight
  `2.44e-5`), so proposal-weight collapse is ruled out for this candidate.
- Target-weighted latent mean/covariance were close to standard normal, but those
  moment diagnostics did not detect the wrong generative mode mass. They remain
  explanatory only and cannot promote a multimodal map.
- The weighted heldout NLL was still falling at update 500. A single 2,000-update
  repair is therefore authorized as a budget-discrimination arm. The multiplier
  four is a convenience cap derived from the observed 500-update curve, not a
  promoted optimizer default.
- Repair decision: if the 2,000-update arm passes the same mode-mass screen, move
  to independent replications. If it still misses by more than `0.05` or the NLL
  has plateaued materially above analytic target entropy, stop increasing this
  architecture's budget and test capacity/componentwise structure. Do not start
  r2 or the paper suite while r1 remains open.
- Eight independent 2,000-update replications recovered minority mass with mean
  `0.1725` and 95% Student-t interval `[0.1628, 0.1822]`, excluding analytic truth
  `0.2`. All modes were observed and all runs were finite. This rejects the
  `(32,32)`/three-stage candidate for r1 promotion; it does not reject weighted
  forward KL.
- Capacity repair ladder: keep the measured 2,000-update budget and optimizer
  fixed, then test `(32,32)`/six stages (depth-only) and `(64,64)`/three stages
  (width-only). These doubled-capacity values are convenience hypotheses anchored
  to the failed baseline, not defaults. Held-out weighted NLL and held-out base
  mode mass nominate an arm; final audit remains explanatory until a fresh
  eight-replication interval passes. If neither arm repairs mass allocation, test
  the combined `(64,64)`/six-stage arm once before designing a componentwise or
  augmented-state transport.
- Capacity-arm nomination is lexicographic: minimize absolute minority-mass error
  first; use audit weighted NLL only if the mass errors differ by less than `0.005`.
  The `0.005` tie band is a convenience resolution threshold relative to the
  observed baseline error `0.0275`, not a promotion threshold. A nominated arm
  still requires fresh eight-run interval evidence.
- The six-stage width-32 replication interval was `[0.1860, 0.1939]`, still
  excluding truth `0.2`; depth alone repairs most but not all systematic bias.
  The predeclared combined six-stage width-64 canary recovered minority mass
  `0.19926`, audit NLL `3.97393`, and relative pushforward covariance error
  `0.0131`. This materially improves the depth-only canary and nominates the
  combined arm for eight-run confirmation. It remains a target-specific
  candidate, not a default.
- The combined six-stage width-64 eight-run confirmation completed after switching
  to one sequential GPU process and one attached terminal session at a time. The
  weighted minority-mass mean was `0.19478`, with 95% Student-t interval
  `[0.19045, 0.19910]`; analytic truth `0.2` is outside the interval. All runs were
  finite and represented both components. This rejects the combined generic IAF
  candidate under the predeclared r1 criterion, without rejecting weighted forward
  KL. The generic IAF capacity ladder is now exhausted; the next repair must use
  componentwise or augmented-state structure before any later rung starts.

## 2026-08-12 owner-requested width-128 budget canary

The owner requested one additional diagnostic at six stages, hidden layers
`(128,128)`, and 10,000 updates because the measured 2,000-update wall time was only
about 1.7 minutes and the held-out curves had not established optimization
convergence.

Evidence contract:

- Question: does substantially greater generic-IAF capacity and optimization budget
  remove the minority-mass and held-out-density mismatch seen at width 64?
- Exact baseline: the eight-run six-stage `(64,64)`, 2,000-update campaign; matched
  reverse KL remains in the harness at the same architecture and update budget.
- Primary canary diagnostic: analytic minority mass on the untouched base-pushforward
  audit; held-out weighted NLL selects a checkpoint but cannot promote the arm.
- Vetoes: nonfinite values, missing component, invalid XLA/GPU/memory-growth receipt,
  weight collapse, wrong target/capacity identity, or corrupt artifacts.
- Explanatory diagnostics: loss trajectory, selected update, clipping, latent
  moments, audit NLL, runtime, and allocator peak.
- Nonclaims: one seed cannot establish convergence, statistically supported
  superiority, r1 promotion, HMC validity, or transfer to unknown modes.

Default and numeric audit:

- Width 128 is an owner-requested capacity hypothesis, not a default. Six stages is
  inherited from the strongest tested depth arm.
- 10,000 updates is owner requested and equals 40.96 million fresh proposal rows per
  training arm. It is not an epoch count because batches are newly generated.
- Batch 4,096, Adam learning rate `1e-3`, tanh activation, gradient clip 10, float64,
  and fixed learning rate are inherited comparability hypotheses. Fixed learning
  rate may prevent terminal convergence; the checkpoint curve is the early
  diagnostic. A terminal best checkpoint does not prove convergence.
- Training, selection, audit, and base-pushforward streams remain disjoint. The
  output root is fresh and versioned.

Skeptical audit verdict: **PASS FOR ONE DIAGNOSTIC CANARY ONLY**. The command
directly tests the requested budget/capacity repair against analytic truth. A pass
nominates replication or learning-rate review; a failure with a still-improving
terminal curve does not prove generic-IAF inadequacy.

Canary outcome: the run completed in `652.72` seconds. Held-out NLL selected update
`8,500`, not the terminal update, and the post-4,000 checkpoint windows fluctuated
around a stable level rather than improving monotonically. The selected map's audit
NLL was `3.95192`, only `0.00271` nats above the independently estimated target
self-NLL; minority mass was `0.19719` (absolute error `0.00281`). This nominates the
width-128 arm for independent replication. It does not establish convergence across
seeds or pass r1.

### Width-128 confirmatory replication campaign

The owner requested four seeds on each of the two physical GPUs, eight seeds total.
The width-128 seed-0 canary selected this capacity and is therefore excluded from
the confirmatory interval. Confirmation uses fresh replication IDs 1--8: four
assigned to host GPU 0 and four to host GPU 1. This avoids capacity-selection
leakage while satisfying the requested allocation.

- Frozen protocol: six stages, hidden layers `(128,128)`, 10,000 updates, batch
  4,096, float64, XLA, fixed Adam `1e-3`, and the same analytic target/proposal.
- Primary criterion: analytic minority mass `0.2` lies inside the two-sided 95%
  Student-t interval across the eight fresh independently trained maps.
- Hard vetoes: nonfinite value, missing component, invalid hash, wrong seed/capacity
  identity, missing GPU/XLA/memory-growth receipt, or incomplete terminal artifact.
- Stop rule: one process per GPU; stop only the affected GPU lane on a veto and do
  not pool an incomplete or mismatched run.
- Allocation: replications 1--4 on GPU 0 and 5--8 on GPU 1, sequential within each
  GPU and concurrent across GPUs.
- Expected wall time: approximately 44 minutes, derived from the measured 652.72
  seconds per seed and four sequential seeds on the slower lane. This is an
  estimate, not a timeout or scientific threshold.
- Nonclaims: a passing interval applies only to this analytic target and frozen
  protocol; it does not establish HMC, posterior, paper-suite, or SSL-LSTM validity.

Skeptical audit verdict: **PASS**. The baseline is exact analytic truth; seed 0 is
not reused; the criterion and vetoes are frozen; artifacts directly answer the
question; and the two-GPU allocation changes throughput, not the statistical unit.

Confirmation outcome: all eight fresh seeds completed and passed the weighted-arm
finite/coverage/hash/device gates. Mean minority mass was `0.20000077`, with 95%
Student-t interval `[0.19782562, 0.20217592]`, containing analytic truth `0.2`.
This passes the predeclared r1 target screen. It is not proof of equality or
cross-target validity. The next justified action is the remaining r1 analytic
target variants, not the paper or SSL-LSTM rungs.

## Implementation phases

1. Add a TensorFlow trainable inverse/log-density route for the existing affine and
   dense autoregressive transports, without changing reverse-KL behavior.
2. Add weighted forward-KL loss, gradient, update, validation, and artifact schemas.
3. Add analytic Gaussian/mixture utilities and tests for exact normalization,
   gradients, inverse/logdet, weighted reduction, deterministic replay, and mode
   collapse detection.
4. Execute CPU-only mechanics tests.
5. Execute one GPU/XLA r0 canary with memory growth, then a bounded r1 unequal-weight
   canary only if r0 passes.
6. Record result, decision/inference tables, manifest, and reset memo. Review whether
   thresholds and budgets are supported before any larger suite.

## Run manifest requirements

Every serious artifact records git commit and dirty status, exact command, conda
environment, TensorFlow/TFP versions, device and GPU memory-growth receipt, allocator
peak, dtype/XLA, target/proposal parameters and hashes, architecture and optimizer,
seeds, wall time, output paths, this plan, and the terminal result file.
