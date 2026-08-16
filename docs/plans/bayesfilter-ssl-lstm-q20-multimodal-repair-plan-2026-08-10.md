# SSL-LSTM q=20 multimodal repair plan (2026-08-10)

Status: `AUDITED_STAGE_1_EXECUTION_AUTHORIZED`  
Execution date: 2026-08-10

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can an exact TensorFlow/TFP parallel-tempered fixed-HMC harness preserve and recover known multimodal distributions, and can it execute finite replica exchanges on the exact seed-B transformed target at bounded cost? |
| Candidate mechanism | TFP `ReplicaExchangeMC` with fixed HMC at each inverse temperature and exact Metropolis swap correction.  No NUTS. |
| Expected failure mode | Temperature ladder overlap may be too weak, hot replicas may retain basin identity, region-dependent curvature may invalidate a step at one temperature, or SSL target cost may make the design unaffordable. |
| Stage-1 promotion criterion | Synthetic equal- and unequal-weight fixtures recover both modes and their known cold-distribution weights within predeclared broad Monte Carlo bands, with finite HMC/swap traces and reconstructed replica travel. |
| Promotion veto | Non-finite states/target/log acceptance, invalid TFP swap permutations, a synthetic fixture outside its declared weight band, missing mode transitions, or eager/XLA mismatch beyond tolerance. |
| Continuation veto | Harness correctness failure; source/target identity mismatch; SSL compile/run wall time above its cap; any exact-target status failure; or an artifact that cannot reconstruct the executed settings. |
| Repair trigger | Synthetic under-mixing with otherwise valid mechanics triggers ladder/step diagnostics within the fixed attempt budget.  SSL zero transitions triggers ladder/timing design work, not rejection of tempering. |
| Explanatory diagnostics | Per-temperature HMC acceptance, adjacent swap acceptance, identity round trips, sign transitions, hot-replica basin forgetting, runtime, and raw occupancies. |
| Must not be concluded | Exact SSL mode weights, exhaustive mode discovery, posterior correctness, NeuTra repair, superiority over another sampler, default readiness, or posterior-predictive validity. |

## Evidence contract

The exact baseline comparator is ordinary fixed HMC started in one mode on the same
synthetic target.  The global candidate is TFP replica exchange using the same target
and fixed-HMC family.  Known analytic mixture weights are the primary synthetic
authority.  For SSL stage 1, success means mechanics and timing only: exact target
binding, finite traces, accepted swaps where proposed, identity reconstruction, and a
measured cost.  A mode crossing is desirable but is not required for the canary.

Hard vetoes are non-finite accepted states, target/status failure, malformed swap
permutations, failed analytic synthetic bounds, target/provenance drift, and budget
breach.  Swap acceptance, round trips, observed transitions, HMC acceptance, and
runtime are explanatory at SSL-canary scale.  Results are preserved under:

`docs/plans/artifacts/ssl-lstm-q20-multimodal-repair-2026-08-10/r1/`

No stage-1 result will establish posterior correctness or weights.  Replica cold-chain
occupancy will not be used as a mode-weight authority for SSL.  AIS or annealed SMC
with independent batches remains the planned weight lane.

## Baseline ladder

| Rung | Role | Target | Decision use |
|---|---|---|---|
| Plain fixed HMC | Naive known-failure comparator | Analytic mixtures | Must remain trapped under the deliberately separated fixture; explanatory contrast only. |
| TFP replica-exchange fixed HMC | Exact global-transition baseline | Analytic mixtures, then exact SSL pullback canary | May pass stage-1 mechanics/known-law validation. |
| TFP AIS fixed HMC | Independent weighted global-mass candidate | Analytic mixtures first, SSL only in a later plan | Required before SSL mode weights can be claimed. |
| Multimodally trained NeuTra plus exact HMC | Later learned candidate | SSL pullback | Not trained until a global weighted authority supplies coverage. |

Only the first two rungs execute in this stage.  Comparing only the failed NeuTra
chain with replica exchange would be unfair because no independent weight authority
would exist.

## Default and numeric audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
|---|---|---|---|
| TensorFlow 2.20 / TFP 0.25 | Installed environment; reviewed implementation baseline | `ReplicaExchangeMC` and AIS are present; TFP owns Metropolis/swap corrections | API shape or trace mismatch; bootstrap/one-step unit tests and XLA fixture expose it |
| XLA on | Repository default | Required for algorithmic TensorFlow paths | Unsupported trace operation or compile failure; focused compile test |
| CPU with GPU hidden | Explicit diagnostic exception | Current exact SSL value/score authority and prior timing evidence use multicore CPU/XLA; no training occurs | Accidental GPU initialization; environment/device manifest veto |
| Synthetic means `-4,+4`, scale `0.5` | Convenience fixture, not a scientific default | Creates visibly separated modes with exact weights | Too hard/easy to discriminate; plain-HMC trap and replica-transition counts diagnose it |
| Synthetic weights `0.5/0.5` and `0.8/0.2` | Reviewed test cases | Exercise symmetry and unequal mass; exact analytic authority | Pooled occupancy can look right from starts; require post-warm-up transitions and replica travel |
| Synthetic inverse temperatures | Geometric ladder hypothesis following installed TFP guidance | Small positive hot beta lowers barriers while retaining a proper power-tempered target | Poor adjacent overlap or hot forgetting; swap/round-trip diagnostics and at most three fixture attempts |
| Temperature-specific step `epsilon_1/sqrt(beta)` | Derived harmonic scaling hypothesis, also recommended by installed TFP docs | Curvature of a power-tempered smooth target scales with beta | Mixture geometry is not globally harmonic; per-temperature HMC acceptance/finite veto |
| SSL cold step `0.1`, `L=3` | Measured causal control, not a selected kernel | Passed local positive and negative stationary checks (`32/32`, `31/32`) | Too short for global travel; canary cannot promote it and reports transitions only descriptively |
| SSL betas `(1,0.5,0.25,0.125)` | Convenience geometric ladder hypothesis | Four replicas bound cost and reduce the sampled path barrier by up to eightfold | Ladder can miss overlap/forgetting; canary records adjacent swaps and explicitly makes no mixing claim |
| SSL four transitions | Convenience mechanics minimum | Exercises both even and odd adjacent swap parities twice | Insufficient for stationarity; declared mechanics/timing-only |
| Synthetic weight bands | Reviewed regression tolerances, not hypothesis-test alpha | Equal negative fraction `[0.40,0.60]`; unequal `[0.68,0.90]` are broad enough for correlated finite draws but reject collapse/wrong dominant mode | Single deterministic stream could pass accidentally; require mode transitions, multi-chain starts, and later statistical validation before promotion |
| Synthetic attempts `<=3` | Convenience compute cap | Allows repair of an obvious ladder/shape issue without open-ended tuning | Repeated tuning to fixtures can overfit; all attempts are preserved and no SSL claim follows |
| Stage-1 wall cap `3,600 s` | Derived from prior 748--773 s tiny SSL XLA runs plus bounded synthetic work | Covers one SSL canary and limited harness repair within user-granted headroom | Compile/runtime explosion; runner-enforced per-stage caps and no automatic material campaign |

The synthetic numeric settings are regression hypotheses, not repository defaults.
Changing them after observing a scientific SSL result is forbidden in this stage.

## Execution plan

1. Implement a standalone TensorFlow/TFP diagnostic helper around
   `ReplicaExchangeMC`.  Validate shapes, strictly positive decreasing inverse
   temperatures, positive per-temperature steps, explicit replica state, exact
   even/odd swaps, trace finiteness, and replica-identity reconstruction.
2. Add deterministic TensorFlow/XLA tests for swap identity, plain-HMC trapping,
   equal-weight recovery, unequal-weight recovery, and invalid configuration.
3. Run the synthetic runner and preserve settings, samples/sign traces, HMC
   acceptance, swap acceptance, identity positions, round trips, and analytic weight
   residuals.  Stop if the contract fails after at most three total attempts.
4. Reconstruct the exact archived seed-B transformed target through the already
   hash/parity-checked root-cause route.  Start the four temperatures across both
   known stationary regions, use the audited canary ladder, run four transitions,
   and preserve finite/status/swap/timing evidence.  This is not warm-up or retained
   posterior sampling.
5. Write the result and reset memo.  Decide whether a separately budgeted ladder
   tuning plus AIS campaign is justified.  Do not launch it in stage 1.

## Compute and attempt budget

- Synthetic tests and runner: at most three full fixture attempts and `900 s` total.
- SSL canary: one scientific attempt plus at most one localized harness retry,
  `2,400 s` total.
- Whole stage: `3,600 s` wall cap as recorded by artifacts; no detached service.
- CPU threads: 8 for this diagnostic lane; `CUDA_VISIBLE_DEVICES=-1` before import.
- Output root is versioned `r1`; every writer refuses overwrite.

The later campaign budget will be derived from the SSL canary.  No extrapolated
multi-hour campaign is authorized by this plan.

## Skeptical plan audit

| Audit risk | Finding and repair |
|---|---|
| Wrong baseline | Repaired by adding plain fixed HMC and exact analytic mixture laws; NeuTra is not the only comparator. |
| Proxy promoted to criterion | Swap acceptance, R-hat, transitions, and runtime are explicitly explanatory for SSL.  Known synthetic weights are the stage-1 primary authority. |
| Missing stop condition | Synthetic, identity, finite/status, provenance, attempt, and wall-time vetoes are explicit. |
| Unfair comparison | Plain HMC and replica exchange share targets and local kernel family.  No speed/superiority ranking is attempted. |
| Hidden assumption | The two known SSL sign regions may not exhaust modes; sign is not treated as a complete basin partition. |
| Stale context | Historical exact identity is unavailable and remains false; the canary reuses hash/parity-checked current reconstruction and records the dirty worktree. |
| Environment mismatch | Installed APIs and versions were inspected.  CPU is an explicit diagnostic exception, XLA remains on, and GPU is hidden before import. |
| Artifact would not answer question | Synthetic artifacts preserve analytic truth and travel; SSL artifact answers mechanics/timing only.  It cannot be cited for weights. |
| Beta zero/improper target | Repaired by requiring every power-tempering inverse temperature to be strictly positive. |
| Replica labels lost by state swaps | Repaired by tracing accepted swap matrices and reconstructing identity positions, not inferring round trips from state values. |
| Step valid only in one region | SSL cold step comes from explicit two-region control; every temperature reports its HMC finite/acceptance telemetry.  No candidate kernel is promoted. |
| Dirty shared worktree | Only new lane-specific files are touched; no existing inference/nonlinear file is edited or reverted. |

The audited stage is adequate because a failed synthetic test invalidates the harness,
while an SSL canary failure remains a discriminating ladder/cost result rather than a
premature rejection of the research direction.

## Pre-mortem

The run could misleadingly pass if pooled synthetic occupancy merely preserves its
initial balance.  Warm-up exclusion, deliberately nonstationary starts, within-cold
sign transitions, and replica identity travel reduce that risk.  It could fail for
step-size or ladder reasons rather than because tempering is ineffective; per-replica
HMC and adjacent-swap telemetry distinguish those.  SSL can pass mechanics while the
hot replica never forgets its basin; that is why canary success does not authorize a
posterior claim.

## Planned decision tables

The result will separately report engineering correctness, sampler validity, and
scientific interpretation.  It will include hard veto status, viable candidates,
whether any ranking is statistically supported, descriptive-only differences,
default readiness, next evidence, strongest alternative explanation, and what would
overturn the conclusion.

