# SSL-LSTM q=20 tempered reverse-KL transport ensemble implementation plan

Date: 2026-08-28  
Status: `PHASES_0_TO_7_COMPLETE_PHASE_8_C0_TO_C5_COMPLETE_PHASE9A_LOCALIZED_REPAIR_COMPLETE_FULL_REPLAY_PENDING_PHASE9B_BLOCKED`

Governing master program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`.
This file remains the detailed mathematical and implementation plan; future
phase authority and state transitions come from the master program and its
named active subplan.

Mathematical authority:
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.tex`.
The longer Markdown record at the same basename preserves the earlier replay
theorems and explains why replay is no longer the high-dimensional foundation.

Independent review records:

- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-math-review-reply-2026-08-28.md`
  (`AGREE`); and
- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-plan-review-reply-2026-08-28.md`
  (`REVISE`, adjudicated and repaired in this revision).

## Decision

Implement a new candidate with two separate uses of an ensemble of invertible
transports:

1. Train each transport with fresh IID standard-Gaussian draws and a reverse-KL
   objective along a proper reference-to-posterior temperature path.
2. After training, freeze every transport and use it as one exact coordinate
   chart in a fixed, state-independent mixture of Metropolis-corrected HMC
   kernels. Couple temperatures with exact adjacent replica exchange.

The ensemble is a categorical mixture of pushforward laws. It is not an
arithmetic average of maps. Adaptive particle replay is not an input to the
primary training or validation route. Joint mixture reverse-KL refinement is an
optional enhanced arm because its cross-density cost is quadratic in component
count and its fitted weights are not automatically posterior mode masses.

This plan originally authorized implementation, unit/reference tests, and short
mechanics smokes. On 2026-08-29 the user instructed the program to refresh the
missing Phase 8 details and continue execution. The active serious-campaign
authority is the bounded Phase 8 subplan at
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-calibration-subplan-2026-08-29.md`.
That subplan freezes the component-count candidate set and selection rule,
temperature-ladder candidates, training search, whitening diagnostics,
checkpoint semantics, compatibility gates, chain count, ESS/MCSE and
declared-region travel targets, attempt caps, and the total campaign wall cap.
The selected component count and ladder remain frozen before any untouched
confirmation stream is consumed.

## C3 closeout and C3B refresh, 2026-08-31

C3A completed all eight L3 lineage/overlap rows on the strict q=20 GPU route;
the artifact-only diversity repair then supplied the omitted covariance and
sign-occupancy summaries from fresh disjoint banks. All hard checks passed.
The result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-result-2026-08-31.md`.

Positive-temperature branching is not nominated: its cross-chart mean and
covariance distances are not consistently larger than pure continuation across
the two roots and two calibration architectures, and all sign occupancy stays
near one half. This is descriptive short-run evidence, not a rejection of
tempering or a ranking. C2's large pullback-score residuals remain unresolved;
no whitening, HMC, posterior, or default claim is open.

The missing-diagnostic repair is closed. The next bounded subplan is C3B:
repeat the pure-versus-single-restart comparison on an L5 ladder
`(0,0.25,0.5,0.75,1)` with the same two architectures, batch size, root count,
and positive-temperature restart at `beta=0.5`. Its purpose is to test whether
finer temperature spacing improves adjacent overlap or preserves diversity;
Phase 9 confirmation and all HMC work remain closed.

C3B completed its repaired attempt with all eight rows hard-valid. L5 adjacent
acceptance was descriptively higher than the L3 diagnostic, and the paired
branch-minus-pure mean-distance contrast was positive for both roots of both
architectures. Covariance and sign contrasts were mixed, however, and the
short finite banks are not chains. No branching or architecture arm is
promoted. The result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3b-l5-ladder-result-2026-08-31.md`.
The imported-helper provenance gap was closed by the metadata-only receipt at
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/provenance-repair-2026-08-31/attempt-01/provenance_manifest.json`.

The next bounded step is C4A, a q=20 GPU/XLA feasibility pilot for the optional
`K=4` joint mixture reverse-KL arm. It measures the quadratic cross-density
term before any candidate freeze; it cannot open whitening, posterior, HMC, or
mode-discovery claims.

C4A then passed its K=4 implementation and resource screen in 454.691 seconds:
the exact joint work count was 512 per `B=32` update, the 16-update forecast
was 117.723 seconds, and peak allocator use was 3.40 GiB. The joint held-out
objective was descriptively lower than the matched independent copies, but one
root and eight updates do not support a ranking. C4A is closed without joint-arm
promotion; the next bounded step is a fresh C4B root/architecture replication
before C5 freeze. Its result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4a-joint-feasibility-result-2026-08-31.md`.

C4B then replicated the K=4 implementation and resource result at fresh roots
for compact-high and compact-low. Both rows passed all exact-work, finite-state,
checkpoint, learned-map reliability, alpha, duplicate-state, GPU/XLA, allocator,
and forecast checks in 876.427 seconds. The held-out objective contrasts had
opposite signs and were evaluated on separate arm banks, while pullback score
residuals stayed large. Therefore no joint arm, architecture, whitening, HMC,
posterior, or default is promoted. The result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4b-joint-replication-result-2026-08-31.md`.
The active boundary is now a metadata-only C5 freeze refresh before any
separately reviewed Phase 9 proposal.

This C4B state supersedes earlier phase-table snapshots that described C1, C3,
or C4 continuation boundaries. The active execution boundary is the C5 freeze
refresh; Phase 9 and retained HMC remain closed.

C5 then passed as a metadata-only freeze. The terminal receipt is the fresh
`attempt-02` manifest under
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c5-freeze/`;
it hashes the finalized subplan, selects `phase8-k2-compact-high-l3-pure`, and
marks K=4 joint as `NOT_RETAINED_FOR_PHASE9`. The result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-result-2026-08-31.md`.
The next boundary is a separately audited Phase 9 tuning/validation subplan.

### User-authorized shared-GPU exception

On 2026-08-29 the user explicitly authorized the bounded Phase 7 mechanics
smoke to share GPU 0 despite the trusted probe's idle-compute veto. This is a
resource-scheduling exception only; it does not change the target, numerical
method, evidence contract, or promotion criteria. The launch must expose only
GPU 0, set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import, verify
memory growth before logical-device creation, record the preflight allocator and
compute-process snapshot, and use a new output directory. It must stop on an
allocation failure, visible contention, or any invalid numerical state. The
result remains a non-claim-bearing mechanics diagnostic and does not authorize
Phase 8 training, retained posterior sampling, or a capacity claim for larger
campaigns.

## Research intent ledger

| Field | Statement |
|---|---|
| Main question | Can blind, diversified reverse-KL lineages continued through a proper temperature path produce complementary frozen charts that improve declared multimodal exploration over single-map and physical-coordinate baselines at matched target-evaluation cost? |
| Candidate mechanism | Multiple independently initialized transports, fresh Gaussian reverse-KL training, positive-temperature branching/restarts, fixed multi-chart HMC, and replica exchange. |
| Expected failure mode | All lineages collapse to the same basin; the ladder has poor overlap; a chart is numerically unusable away from its training basin; per-chart tuning cost or `O(K^2 B)` mixture refinement becomes prohibitive; or exact kernels remain mode locked. |
| Primary promotion criterion | On untouched q=20 confirmation random streams, the candidate passes exactness and health vetoes, canonical retained-chain diagnostics, cold declared-sign-region transition and replica-travel requirements, and predeclared downstream/reference agreement, with uncertainty-aware comparison against the full baseline ladder under a common target-call budget. |
| Promotion veto | Wrong bridge endpoint, mismatched value/score, wrong inverse or Jacobian, state-dependent uncorrected chart selection, stale tuning identity, invalid Metropolis or swap ratio, invalid current/retained state, asymmetric or unrecorded invalid-proposal handling, unledgered NeuTra-HMC route, or violation of TensorFlow/GPU/batching/XLA policy. |
| Continuation veto | The q=20 target cannot supply a valid proof that every ladder law is proper on the same theta measure; a minimal analytic fixture contradicts the mathematical identities after focused repair; no transport architecture can provide a reliable inverse and log determinant after the bounded repair ladder; the numerical target is invalid on the predeclared blind initialization screen after bounded repair; or the authorized campaign budget is exhausted. |
| Repair trigger | Duplicate lineages, poor temperature overlap, candidate rejection, weak round trips, high cross-density cost, or a localized implementation/infrastructure failure. These reject or repair the current candidate; they do not refute reverse-KL transports or tempering in general. |
| What must not be concluded | No exhaustive mode discovery, IID Gaussian pullback, universal high-dimensional scaling, posterior mode masses from mixture weights, statistical superiority from descriptive metrics, underlying UKF model correctness, or default readiness. |

## Evidence contract

### Claimed target

The claim-bearing cold target is the existing batch-native q=20 SSL-LSTM
posterior in unconstrained `theta in R^4`. The 60-dimensional filtering state is
internal to target evaluation; it is not the sampled parameter dimension. The
temperature family must have the same theta measure and must satisfy
`pi_beta=1 = pi` exactly.

The inspected q=20 implementation computes

```text
log tilde_pi(theta) = log L(theta) - ||theta - PRIOR_CENTER||^2 / (2 * 16)
```

up to a constant. This supplies a candidate proper endpoint
`g0 = N(PRIOR_CENTER, 16 I)` and the likelihood-tempered path
`log tilde_pi_beta = log g0 + beta log L`. Phase 0 must expose and test this
decomposition rather than reconstruct it by subtracting two rounded outputs.

Endpoint parity alone is insufficient: every intermediate bridge law must be
proper. For this q=20 target, Phase 0 must turn the following source-derived
argument into a checked bridge-properness receipt. The finite-horizon
approximate likelihood is a product of one-dimensional Gaussian innovation
densities. Its unscented covariance weights are nonnegative at the declared
`alpha=1`, `beta=2`, `kappa=0` setting, and the fixed observation variance
`R` is strictly positive. Hence each innovation variance satisfies
`S_t(theta) >= R`, and

```text
0 < L(theta) <= M := (2 pi R)^(-HORIZON/2).
```

There are two notations here, because the runtime value program intentionally
omits theta-independent normalizing constants. Let `bar g0` be the
unnormalized Gaussian kernel returned by that program and let
`g0 = A_prior^(-1) bar g0` be the corresponding normalized Gaussian law, where
`A_prior = (2 pi prior_variance)^(d/2)`. For the normalized law, every
`beta in [0,1]` has

```text
0 < Z_beta = integral g0(theta) L(theta)^beta dtheta
           <= max(1, M) < infinity.
```

The runtime kernel has the matching bound

```text
0 < bar Z_beta = integral bar g0(theta) L(theta)^beta dtheta
               = A_prior Z_beta
               <= A_prior max(1, M) < infinity.
```

The positive constant `A_prior` cancels from Metropolis and replica-swap
ratios, but it is part of the runtime properness receipt and its source
identity is recorded. A finite sample of target evaluations is only a
numerical stress test and cannot replace this proof. If the covariance
weights, fixed positive observation variance, finite horizon, likelihood
factorization, or runtime normalizer convention changes, the receipt becomes
stale and a new sufficient argument is required before the bridge can be
admitted.

### Comparator ladder

All arms use the same cold target, data, theta coordinates, dtype, target-status
policy, and target-evaluation accounting.

| Arm | Purpose |
|---|---|
| Physical-coordinate HMC | Naive local baseline. |
| Single cold reverse-KL NeuTra | Original NeuTra baseline. |
| Physical-coordinate replica exchange | Classical tempering baseline under the identical proper bridge. |
| Single-chart-per-temperature replica exchange | Isolates temperature continuation and exchange from the ensemble effect. |
| Cold multi-start, multi-chart HMC | Isolates multiple charts from tempering. |
| Tempered independent reverse-KL ensemble | Plain proposed method. |
| Tempered ensemble plus joint mixture-RKL refinement | Enhanced optional method. |
| Mixture independence MH | Optional global-proposal diagnostic, never a substitute baseline. |
| Mode-informed transport initialization | Optional oracle discovery ceiling using the two known sign-separated MAP representatives; excluded from blind-candidate promotion, ranking, and default selection. |

Training target calls, HMC target/score calls, transport cross-density
evaluations, swap bridge recombinations or evaluations, compile time, peak
memory, and wall time are reported separately. The `O(K^2 B)` joint-loss term
counts transport inverse/density work, not `O(K^2 B)` SSL-LSTM likelihood
calls: the target term is evaluated on the `K B` outer samples. A
compute-matched comparison must cap expensive target evaluations and wall time,
not merely optimizer updates or retained draws.

Exact cache reuse is allowed only when target identity, physical state, bridge
identity, value, and status all match. For the q=20 decomposed bridge, cached
`log g0` and `log L` can be recombined at another beta without another filter
call. A generic bridge may require two cross-beta value calls per adjacent swap.
Manifests report cache hits and actual new calls rather than assigning every
swap a fictitious fixed call count; every comparison arm uses the same rule.

### Evidence roles

| Evidence | Role |
|---|---|
| Density, inverse, Jacobian, transformed-score, transition-replay, and swap-ratio fixtures | Hard implementation veto. |
| Proper bridge endpoints, score decomposition, and source-bound intermediate-law properness proof | Hard target veto. |
| Finite status, all-chain movement, energy-error policy, memory growth, device, batch-native target, and XLA receipts | Hard execution veto. |
| Training loss, latent moments, component separation, HMC acceptance, adjacent swap rate, and mixture weights | Explanatory or nomination only. |
| Warmup readiness and retained rank-normalized split/folded R-hat | Canonical sampler gates using repository policy. |
| Bulk/tail ESS and downstream MCSE | Promotion gates whose numeric targets must be derived and frozen before the serious run. |
| Replica round trips, hot-level declared-region forgetting, cold declared-region transitions, and initialization forgetting | Tempering-specific promotion gates under the explicit q=20 label protocol below; they are not formal basin-identification evidence. |
| Paired multi-seed uncertainty analysis | Required before ranking viable arms. |

## Current implementation findings

| Path | Reusable fact | Boundary that must not be crossed |
|---|---|---|
| `bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py` | Batch-native TensorFlow `float64` q=20 value/analytic score/status, XLA by default, explicit Gaussian prior term. | It exposes only the combined posterior externally; a claim-bearing temperature bridge is not yet an interface. |
| `bayesfilter/inference/neutra_weighted_training.py` | Invertible dense IAF, forward/inverse/log-density operations, matched single-map reverse-KL trainer, XLA default. | Existing numeric architecture and optimizer values are inherited hypotheses, not target-specific ensemble defaults. The weighted forward-KL trainer is not the new foundation. |
| `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py` | Active tuner binds one exact target and one frozen transport to one scope-specific artifact. | Separate component artifacts do not create a valid multi-chart controller. An artifact is valid only for its exact `(target, beta, chart, dtype, backend, XLA mode)` scope; a different scope needs a different artifact. Recovery of a preserved checkpoint for the identical scope is allowed. |
| `bayesfilter/inference/neutra_hmc.py` | Canonical sequential warmup/retained policy and modern diagnostics. | The present single-target controller cannot be bypassed by a new claim-bearing ensemble runner. It must be extended or supplied a tested exact transition abstraction. |
| `bayesfilter/inference/neutra_hmc_policy.py` | Discovery and route-ledger enforcement for NeuTra-HMC routes. | A new qualifying route is a test failure until classified exactly once. |
| `bayesfilter/testing/distributed_replica_exchange_tf.py` | Diagnostic adjacent-swap and identity-travel mechanics. | Its pure power target is not the proposed proper-reference bridge, and a testing module is not production inference authority. |
| `docs/reference/hmc-tuning-interface.md` | Active tuner selection and artifact-consumption rules. | A chain runner is not a tuner; acceptance alone is not handoff evidence. |

## Mathematical implementation contract

1. `T_i` always denotes a bijection from Gaussian latent `z` to physical
   `theta`. Every component must supply forward, inverse, forward log
   determinant, component log density, and exact transformed score.
2. `q_alpha(theta)` is computed by `reduce_logsumexp(log_alpha_i + log q_i)`.
   Sampling selects a categorical component. Maps are never averaged.
3. Independent component training minimizes `KL(q_i || pi_beta)` with fresh
   Gaussian batches. Optional joint training enumerates every outer component
   and every cross-component density in the mixture-RKL identity.
4. No invalid target row may be silently dropped, replaced, or resampled within
   an update. Before optimizer state exists, each proposed initialization is
   evaluated on a disjoint, fixed stateless standard-base-Gaussian preflight
   bank, folded by component identity and never reused for training. An invalid
   initialization may traverse only a predeclared finite reference-affine/scale
   repair ladder, using the same bank and applying no optimizer update. Exhaustion
   rejects the initialization or target. After admission, an invalid training
   row preserves and freezes the pre-update state, archives the attempt, and
   triggers an explicit rollback or numerical repair; drawing replacement rows
   until a batch happens to be valid is ineligible because it conditions the
   Gaussian objective on numerical admission.
5. `alpha` is a variational density parameter. `gamma` is a fixed
   state-independent chart-selection frequency. Neither is inferred posterior
   mass without an additional valid argument.
6. Each physical chart kernel is obtained by mapping the current physical state
   through that chart's inverse, applying a Metropolis-corrected kernel to the
   exact pullback target, and mapping back.
7. Each `(target, beta, chart, dtype, backend, XLA mode)` tuning scope has its
   own repository-issued artifact. Runtime retuning and cross-scope artifact
   reuse are forbidden. Checkpoint recovery is allowed only for an identical
   scope and must preserve the original lineage and tuning evidence.
8. Adjacent swaps use the two complete unnormalized bridge densities at the two
   exchanged physical states. The current pure-power shortcut is ineligible.
9. Only the `beta=1` retained state is posterior output. Training samples,
   warmup, hot replicas, and mixture samples are not posterior draws.
10. The q=20 declared-region label is the sign of physical coordinate 2,
    `observation_weight.0.0`: strict positive and strict negative half-spaces
    contain the two known sign-separated MAP representatives. A zero value is
    recorded as a boundary state. These labels establish region visits and
    crossings only; they do not establish formal basin membership, exhaustive
    mode count, or posterior regional mass. The representatives and label
    evidence must bind the exact current target signature before use.

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode and early diagnostic | Status |
|---|---|---|---|---|
| q=20 theta dimension `4` | Current target API | This is the active problem, despite a 60-dimensional internal filter state. | Shape/signature fixture catches measure drift. | Frozen target fact. |
| Gaussian endpoint with center `PRIOR_CENTER`, covariance `16 I` | Current target prior calculation | It is a proper law on the exact theta coordinates and gives prior-likelihood tempering. | Endpoint value/score parity plus the bounded-likelihood properness receipt; fail if either is unavailable. | Candidate reviewed default, pending Phase 0 receipt. |
| Number of components `K` | Not known | It controls discovery probability, quadratic joint-loss cost, and per-chart tuning cost. | Memory/runtime and duplicate-lineage pilot. | Unproven hypothesis; no numeric default. |
| Temperature count and spacing | Not known | Must provide overlap and hot-level forgetting. | Energy-overlap, adjacent exchange, round-trip, and basin-forgetting diagnostics. | Unproven hypothesis; no numeric default. |
| Pure continuation versus positive-temperature branching | Correction in the mathematical note | Full optimization at `beta=0` can erase distributional diversity. | Component-distance and basin-occupancy comparison on held-out Gaussian draws. | Required ablation. |
| Blind initial centers/scales drawn under `g0` | Avoids target-mode oracle leakage | Diversifies physical starts without posterior samples. | Fixed latent preflight bank and finite reference-affine/scale repair ladder; report every invalidity and component duplication without replacement-row conditioning. | Candidate hypothesis. Mode-informed seeds are oracle-only. |
| Existing IAF architecture and Adam settings | Legacy weighted campaign | Useful warm start for mechanics only. | Target-specific capacity/optimizer search and held-out Gaussian reverse-KL estimates. | Warm start, never a promoted default. |
| Joint mixture refinement | New derivation | Can coordinate component density coverage without target samples. | `O(K^2 B)` transport cross-density memory/time and numerical collapse. The unequal-error alpha bias is expected mathematics, not a failure. | Optional enhanced arm; admitted only if a measured budget gate passes. |
| Fixed uniform `gamma` | Simplest state-independent kernel mixture | Exact by the proved invariant-kernel mixture result. | Per-chart movement/cost imbalance. | Baseline kernel policy; other fixed values require held-out tuning. |
| Trainable `alpha` through logits | Standard simplex parameterization | Ensures positivity and normalization for joint mixture training. | Boundary collapse and misleading mass interpretation. | Optional; record entropy and never call posterior mass. |
| TensorFlow/TFP, static batch, XLA, GPU memory growth | Repository policy | Required implementation and execution backend. | Import/device/XLA/batch receipt; fail closed. | Required. |
| No pfor or row mapping | Repository policy | Target is already batch native. | Graph inspection and monkeypatch tests. | Required. |
| Canonical sequential thresholds | `bayesfilter_neutra_sequential_hmc_v1` | Current owner policy: warmup recent-window maximum R-hat `<=1.05`, retained tuning admission R-hat `<=1.01`. | Policy and route-ledger tests. | Inherited reviewed policy; ESS/downstream targets remain to be set. |
| Temperature checkpoint semantics | Phase 8 skeptical audit | Mutating one transport through the ladder would overwrite the chart required at earlier temperatures. | Exact snapshot/restore replay at every `(beta, chart)` checkpoint and immutable hashes before later training. | Required engineering invariant. |
| Optimizer state across temperatures | No established q=20 evidence | Resetting Adam at a changed objective avoids silently carrying stale moments, but may slow continuation. | Reset at each beta in the first candidate; record as a candidate hypothesis and preserve enough state for a future carry-state ablation. | Frozen Phase 8 candidate choice, not a general default. |
| Four chains | Shared sequential controller | The controller rejects fewer than four chains and modern split/folded R-hat needs multiple chains. | Route-policy and controller configuration checks. | Required minimum for Phase 9. |

## Phase program

### Phase 0: target bridge and interface proof

Implement a repository-owned batch-native tempered target interface that
returns bridge value, exact score, and target status for a static rank-2 theta
batch and scalar beta. Expose the likelihood and Gaussian-prior terms directly
from the existing computation. Do not compute the likelihood later by
subtracting separately rounded posterior and prior calls.

Required tests:

- `beta=0` equals the Gaussian prior value/score up to one theta-independent log
  normalizer;
- `beta=1` equals the current posterior value/score exactly within declared
  dtype tolerance and has the same status;
- arbitrary beta equals `prior + beta * likelihood` for value and score;
- a source-bound properness receipt checks finite horizon, the fixed strictly
  positive observation variance, the exact nonnegative covariance weights, and
  the Gaussian innovation factorization, then records the analytic `M` bound,
  the normalized-law `Z_beta` bound, and the runtime-kernel
  `bar Z_beta <= A_prior max(1,M)` bound;
- a disjoint blind `g0` stress bank and the two known MAP representatives have
  finite bridge values/scores/status at a predeclared beta grid; this is a
  numerical-suitability screen only, not an integrability proof;
- bridge and swap arithmetic remains in the log domain; observed dynamic ranges
  are recorded, and any exponentiation whose argument can exceed the dtype's
  derived safe log range is rejected rather than protected by a guessed
  constant;
- stable explicit `tf.function` signatures, XLA compatibility, batch size
  greater than one, and no sample-wise loop or pfor;
- target, theta-coordinate, data, q, dtype, backend, and bridge identities are
  present in the signature payload.

Exit: endpoint and score parity, the analytic properness receipt, and the
numerical-suitability screen pass. A finite grid alone cannot close the
properness gate. A failure is a hard design blocker because replica exchange
and reverse-KL training would otherwise target a wrong or undefined law.

### Phase 1: ensemble density primitives

Add an inference-owned transport-bank object over a statically sized tuple of
existing dense IAF transports. It must compute all component forward results,
all component log densities on a physical batch, stable mixture log density,
categorical sampling, and immutable component identities. Use TensorFlow
operations only.

Required fixtures:

- affine Gaussian components with analytic densities;
- permutation of component order leaves the mixture density unchanged;
- categorical empirical moments agree with analytic mixture moments as a
  diagnostic with uncertainty, not a proof;
- averaged-map counterexample differs from the mixture;
- extreme logits and tail points remain numerically stable;
- cross-density tensor shapes are fixed and `O(K^2 B)` transport work, peak
  memory, compile time, and steady-state update time are measured separately
  from the `O(K B)` target evaluations.

Exit: exact analytic density/inverse/Jacobian fixtures pass. A quadratic-cost
failure blocks only joint mixture refinement, not independent component
training or fixed chart kernels.

### Phase 2: reverse-KL trainers

Implement two explicit trainers:

- an independent-component trainer that applies the original reverse-KL
  Gaussian objective to every component; and
- an optional joint mixture trainer that enumerates the mixture expectation and
  cross-component log densities, with optional logits for `alpha`.

Every optimizer update uses a fresh stateless Gaussian batch with more than one
row. Target evaluation is batch native and compiled with XLA. The trainer must
record whether any replay, scalar fallback, row mapping, or invalid-row filtering
occurred; each must be false for an eligible run.

Tests cover analytic Gaussian optima, gradient parity on tiny diagnostic
fixtures, independent seed streams, alpha normalization, component
permutation, checkpoint round-trip, and whole-update failure on an invalid row.

Every completed `(beta, chart)` training state must also have an immutable,
TensorFlow-native snapshot containing the transport configuration, reference
affine wrapper, trainable tensors, component identity, beta, parent checkpoint,
and target/bridge identities. Restoring into a fresh object must reproduce the
state hash, forward value, inverse, and log determinant before the original
object may continue to the next beta. Earlier-temperature objects are never
mutated after their checkpoint is frozen.

Exit: independent training passes first. Joint training is a separate optional
exit and is rejected if it exceeds its predeclared memory/wall-time allocation,
is numerically invalid, or collapses components to identical parameters under
the declared collapse test. Fitted alpha following the separated-region
`p_i exp(-delta_i)` formula is expected behavior, not a rejection. Failure of
the optional arm does not stop the main route.

### Phase 3: tempered lineage controller

Build a deterministic controller that binds a bridge identity, a frozen beta
ladder, component identities, stateless seeds, and one of two discovery arms:

- pure continuation from the previous temperature; or
- continuation plus predeclared fresh restarts/branching at positive beta.

Initial physical locations and scales for the blind candidate come only from
the reference law and declared stateless rules. Known q=20 sign modes may be
used only in the optional oracle discovery-ceiling arm. The oracle arm is never
eligible to promote the blind candidate. Before the first optimizer update,
every blind initialization must pass the fixed preflight bank under the finite
identity/scale repair ladder from contract item 4. A fresh lucky batch is not a
repair. No lineage is pruned because of lower aggregate reverse-KL loss during
discovery. Selection, if needed for a later cap, uses a disjoint held-out
Gaussian audit and retains the loss/diversity tradeoff rather than treating
either as posterior evidence.

Exit: reproducible checkpoints at every beta, complete initialization-preflight
and repair receipts, all target-status checks, no seed collision, and a report
comparing pure continuation with positive-temperature branching. Duplicate
components are a repair trigger, not a correctness veto.

### Phase 4: exact fixed multi-chart HMC mechanics

For each frozen chart, reuse the existing fixed-transport adapter and active
public tuner. Query the capability registry at runtime. A tuning artifact must
bind the exact beta target and chart identity. Add a physical-state transition
that chooses a chart with fixed `gamma`, applies the artifact-bound transformed
HMC transition, and maps back.

The implementation must either extend the shared sequential controller through
a tested exact-transition abstraction or add a shared-core mode within it. A
standalone new chain loop cannot be promoted. Update the NeuTra route ledger and
its discovery test in the same phase.

Fixtures cover one-step replay, physical/latent round-trip, transformed
value/score parity, fixed-mixture invariance on analytic targets, rejection of
state-dependent gamma, stale/mismatched artifact rejection, and symmetric
handling of invalid leapfrog paths.

Implement a reusable learned-transport reliability screen here. It evaluates
forward/inverse round trips, forward/inverse log-determinant cancellation,
finite transformed scores, and Jacobian conditioning on self-component draws,
all cross-component draws, reference-law stress draws, and declared diagnostic
points. Tolerances are derived from dtype, coordinate scaling, and analytic-map
roundoff fixtures, then frozen before inspecting q=20 results. A self-only
round trip is insufficient because every selected chart must invert a current
physical state that may have arrived through another chart. Phase 8 applies
this screen to every learned q=20 chart before tuning. An invalid inverse at a
current or retained state is a hard veto, not a state-dependent chart-skipping
rule.

Exit: exact mechanics pass for at least two nonlinear frozen charts on analytic
targets. No retained posterior claim is made.

### Phase 5: proper-reference replica exchange

Implement an inference-owned replica-exchange transition over the complete
bridge target, using batch-native value/score calls and alternating adjacent
pair schedules. Preserve temperature slot and replica identity separately.
Do not promote the diagnostic pure-power testing route.

Required tests:

- generic cross-beta log-ratio equals the direct product-density ratio;
- likelihood-tempering simplification is tested but not hard-coded as the only
  bridge;
- forward/reverse swap detailed balance on finite analytic fixtures;
- odd/even pair schedules do not overlap;
- rejected swaps leave values, scores, status, and identities coherent;
- only the beta-one stream is exposed to posterior archiving;
- sequential chunk extension preserves exact continuation state and archives
  every discarded warmup chunk;
- the physical-coordinate replica-exchange baseline uses this same bridge
  object, endpoint identities, cache/status semantics, and swap ratio, with
  identity-coordinate within-temperature kernels; the diagnostic pure-power
  route is rejected as a comparator;
- the physical and charted routes share the same bridge/swap implementation,
  and every new qualifying physical route is classified in the route ledger.

Exit: product-target invariance fixtures, route policy, and artifact identity
tests pass.

### Phase 6: analytic end-to-end fixtures

Use tractable Gaussian and separated Gaussian-mixture targets to exercise the
full sequence: bridge, independent training, optional joint loss, chart tuning,
within-temperature transitions, swaps, identity travel, and cold marginal.
Include unequal regional mass and deliberately unequal component error to show
that fitted alpha follows the derived biased-weight formula rather than being
misreported as posterior mass.

Also exercise gamma independently of alpha. On an exact finite-state or
tractable Gaussian-mixture target, verify algebraically that fixed uniform and
fixed nonuniform gamma both satisfy `pi K = pi`. The controller API must make a
state-dependent gamma unrepresentable or reject it before execution. Preserve
the two-state counterexample as a negative fixture and verify exactly that its
state-dependent selector violates invariance; finite Monte Carlo moment drift
is explanatory only.

These fixtures test implementation and finite-sample behavior. Moment agreement
uses uncertainty intervals and cannot certify stationarity or general scaling.

Exit: all exact identities pass and finite stochastic differences are reported
without unsupported ranking.

### Phase 7: q=20 GPU/XLA mechanics smoke

Run a short, explicitly non-claim-bearing GPU smoke with memory growth enabled
and verified before device initialization. The normal route requires trusted
idle-GPU admission; the user-authorized shared-GPU exception above is an
allowed alternative for this bounded smoke when its preflight snapshot and
contention stop rule are recorded. Exercise bridge endpoints, more than one
transport, more than one positive temperature, one optimizer update, one
fixed-chart transition, and one swap. Record target status, device placement,
XLA, static batch, absence of row mapping/pfor, allocator policy, source hashes,
and output paths.

The permitted CPU-debug exception may hide CUDA and disable JIT solely to
localize implementation defects. A passing CPU-debug artifact is retained as
an implementation diagnostic, but it does not satisfy this GPU/XLA exit and
cannot support placement, performance, HMC, or posterior claims.

Exit: mechanics and environment pass. Loss, whitening, swap rate, and movement
from this smoke are not research evidence.

### Phase 8: q=20 calibration, selection, and candidate freeze

Before this phase, write one serious-campaign subplan with a fresh compute
budget, versioned output root, attempt cap, architecture/capacity search,
optimizer search, batch-size search, a bounded pilot set of `K` candidates,
beta-ladder candidates, branching rules, training seeds, and disjoint
calibration, validation, and untouched confirmation random streams. The
observed q=20 data and posterior target are fixed; "partition" refers to random
streams, initialization banks, and retained-chain evidence, not to changing or
splitting the likelihood data.

Before the first calibration run, the full
`tests/test_ssl_lstm_complexity_target_tf.py` and
`tests/test_ssl_lstm_predictive_tf.py` suites must be rerun under the subplan's
bounded compatibility budget. A test failure is a hard engineering veto until
repaired. A timeout without a failing assertion is an infrastructure/compile
repair trigger: preserve it, localize the stalled test, and retry once under the
remaining compatibility allocation. Focused Phase 0--7 passes do not silently
upgrade an incomplete full suite to a pass.

That subplan must restate the research question and evidence contract, audit
every numeric default, and include a pre-mortem distinguishing: a run that
passes local diagnostics while remaining globally locked; failure caused by an
invalid target, map, or controller; failure caused by insufficient capacity or
tuning; and evidence against the candidate mechanism. It must name the cheapest
diagnostic that separates those explanations before allocating the next budget
rung.

The subplan must freeze these definitions before any result used for selection
is inspected:

- **Cold declared-region transition:** use the strict sign of physical
  coordinate 2, the two half-spaces containing the known positive and negative
  MAP representatives, and the existing retained-label diagnostic. Every cold
  chain must visit both declared regions and make the predeclared minimum number
  of retained crossings; the binary region indicator must satisfy an
  uncertainty target derived from the desired regional-occupancy MCSE. This is
  region-crossing evidence, not formal mode or basin identification.
- **Initialization forgetting:** include cold chains initialized from both
  known sign regions for diagnosis after blind transport training. Predeclare
  posterior functionals and scientifically meaningful equivalence margins;
  compare start-stratified estimates with autocorrelation-aware MCSE intervals.
  Failure to reject equality is not evidence of forgetting.
- **Hot-level region forgetting:** for hot-slot states and tracked replica
  identities, require visits to both declared sign regions, an indicator-ESS or
  MCSE target, and absence of a start-region effect under a predeclared
  equivalence analysis. Raw acceptance or one sign change is insufficient.
- **Replica travel:** define complete cold-to-hot-to-cold and
  hot-to-cold-to-hot round trips by replica identity and freeze the required
  counts from the desired travel uncertainty and retained budget.

#### Phase 8A: calibration and search

Use only calibration and validation streams to compare `K`, ladder,
architecture, optimizer, branching, and batch hypotheses. Before admitting the
optional joint arm, use Phase 1 measurements to derive from its allocated
campaign wall-time and memory budget a maximum admissible `K`, `B`, and total
joint-update count. Record `K^2 B` transport cross-density work and `K B`
target work separately. Candidates outside that measured envelope skip the
joint arm rather than discovering infeasibility during confirmation.

Apply the learned-transport reliability screen from Phase 4 to self, cross,
reference, and declared diagnostic banks. A component may be removed only by a
predeclared validation rule before the final `K`, gamma, and tuning scopes are
frozen; the resulting candidate is then rebuilt and retuned. Preserve every
failure. If no chart survives the bounded architecture/regularization repair,
the continuation veto fires.

For a held-out base draw `z ~ N(0,I)`, define the unnormalized pullback log
density residual

```text
r_i,beta(z) = log tilde_pi_beta(T_i(z))
              + log |det J_T_i(z)| - log phi(z).
```

An exact Gaussianizing chart makes `r_i,beta` constant. Also define the
pullback-score residual

```text
u_i,beta(z) = J_T_i(z)^T score_beta(T_i(z))
              + grad_z log |det J_T_i(z)| + z,
```

which is zero for an exact Gaussian pullback. On disjoint held-out Gaussian and
fixed radial-stress banks, record centered residual RMS and quantiles,
score-residual RMS/max, reverse-KL loss, and declared-region occupancy by
component. These are explanatory and nomination diagnostics, except that a
nonfinite value or failed target status is a hard veto. The moments of the
forward input `z` are tautologically Gaussian and must not be reported as
whitening evidence. Pullback moments and tails computed from exact retained HMC
draws belong to Phase 9.

#### Phase 8B: candidate freeze

Select one frozen `K`, ladder, branching policy, transport architecture,
alpha/gamma policy, and per-scope identity for each comparator arm using only
the calibration/validation evidence. Preserve statistically indistinguishable
viable representatives without calling one best. The optional oracle
initialization arm, if run, is reported only as a discovery ceiling and cannot
select or promote the blind candidate. No confirmation stream is consumed in
Phase 8.

Exit: frozen candidates and protocols, a measured optional-arm feasibility
decision, complete learned-map reliability receipts, and a result note
separating hard vetoes, descriptive differences, and statistically supported
selection statements.

### Phase 9: q=20 tuning and sequential posterior validation

Tune every retained `(beta, chart)` scope through an active public tuner with
disjoint tuning draws. Then run the extended canonical sequential controller:
archive and discard warmup, enforce the current recent-window warmup policy,
grow retained beta-one sampling cumulatively, and apply finite/status/movement/
energy vetoes to every chunk.

Use at least four chains, as enforced by the shared controller. Apply the
subplan's frozen bulk/tail ESS, relative-MCSE, declared-region, start-forgetting,
and replica-round-trip definitions. The exact chain count and initialization
bank are part of the frozen confirmation identity and cannot be changed after
confirmation results are inspected.

Run every frozen comparator arm retained by Phase 8 under the common accounting
rules; do not validate only the proposed candidate. Comparator-specific tuning
must use its own eligible scope and disjoint tuning stream.

Consume only the untouched confirmation streams frozen by the Phase 8 subplan.
Bulk/tail ESS targets, declared-region transitions, hot-level forgetting,
initialization-forgetting equivalence margins, replica travel, and the
independent reference/downstream screen must already be frozen. Apply paired
multi-seed uncertainty analysis to comparisons. A passed hard screen means
viable; it does not support a ranking by itself.

Exit: terminal result with decision and inference-status tables. A valid but
declared-region-locked candidate rejects that arm and triggers the smallest discriminating
repair; it does not retroactively invalidate exact kernel mechanics.

### Phase 10: future-dimensional scaling ladder

Only after q=20 mechanics and posterior gates, test controlled targets with
increasing parameter dimension. The ladder must report component count,
temperature count, cross-density evaluations, target calls, compile time, peak
host/device memory, wall time, round trips, and uncertainty in posterior
functionals. It must include the single-map and physical replica-exchange
baselines.

Exit: a scaling result, including negative results. No q=20 success may be
extrapolated to high dimension without this phase.

## Between-phase repair and refresh

After each phase:

1. Classify failures as target/math, implementation, numerical, tuning,
   infrastructure, candidate-scientific, or evidence insufficiency.
2. Check whether the failure invalidates the harness or only rejects the current
   candidate. Continue to a planned repair unless a declared continuation veto
   fired.
3. Preserve the failed output; use a fresh versioned directory for the repair.
4. Rerun the smallest exact regression, then the phase suite.
5. Compare the implementation with the mathematical note and update the
   document-alignment audit if semantics changed.
6. Refresh the next phase's assumptions, commands, budget estimate, and
   stop/repair conditions from measured evidence.

No repair may change the cold target, data, theta measure, exact correction,
hardware class, or scientific comparison without a revised plan. Local
harness, serialization, compilation, or resource repairs may proceed within an
authorized campaign when its total budget and evidence contract are unchanged.

## Serious-run artifacts

The future campaign root is
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-<date>/`,
with a fresh attempt subdirectory for every launch. Each serious manifest must
record the Git commit and dirty state, exact command, conda environment,
TensorFlow/TFP versions, target/data/bridge/transport/tuning identities, CPU/GPU
and memory-growth status, dtype/TF32/XLA settings, static batch size, seeds,
wall time, target-call counts, output paths, plan, and result note.

## Skeptical pre-execution audit

| Risk | Finding and repair in this plan |
|---|---|
| Wrong baseline | Repaired with physical HMC, single NeuTra, matched physical replica exchange, single-chart tempering, and cold ensemble ablations. |
| Proxy promoted to success | Loss, whitening, component separation, acceptance, and swap rate are explanatory only. Cold posterior and travel gates are explicit. |
| Circular particle validation | Removed from the primary route; training stochasticity comes from IID Gaussian base draws and exact target calls. |
| `beta=0` destroys diversity | Repaired by requiring a pure-continuation versus positive-temperature restart/branching ablation. |
| Improper power-tempering endpoint | Repaired by Phase 0's exact proper Gaussian-prior bridge and endpoint parity. |
| Intermediate bridge law assumed proper | Repaired by the q=20 bounded Gaussian-innovation likelihood proof for the normalized law and the corresponding `A_prior`-scaled bound for the unnormalized runtime kernel; the receipt is stale on source or normalizer changes, and finite stress draws are numerical diagnostics only. |
| Blind initialization deadlock | Repaired by a fixed pre-optimizer bank and finite identity/scale ladder with no replacement-row conditioning; exhaustion is an explicit target/initialization veto. |
| Existing diagnostic route mistaken for implementation | Explicitly forbidden; Phase 5 is inference-owned and bridge-generic. |
| Single-map tuner mistaken for multi-chart authority | Explicit per-scope artifacts plus canonical controller integration and route-ledger update are required. |
| Unfair compute comparison | Target calls, cross-evaluations, compile time, and wall time are separately recorded and primary comparisons are target-call matched. |
| Joint-arm cost measured but ungated | Repaired by a Phase 8A budget-derived feasibility envelope over `K`, `B`, cross-density work, memory, and wall time. |
| Learned map trusted from analytic fixtures | Repaired by self- and cross-component inverse/log-determinant/score screens applied to every frozen q=20 map before tuning. |
| Undefined global-travel gate | Repaired with the existing two sign-region labels, MCSE-aware cold/hot/start-stratified protocols, and explicit nonclaims about formal basins. |
| Physical replica-exchange mismatch | Repaired by requiring the physical baseline and charted candidate to share the identical proper bridge and swap implementation. |
| Hidden numeric defaults | Repaired for the active q=20 calibration by the 2026-08-29 Phase 8 subplan. Values remain target-specific hypotheses and do not become cross-model defaults. |
| Temperature charts overwritten by continuation | Repaired by an immutable snapshot/restore round trip at every `(beta, chart)` state before training proceeds. |
| Tautological whitening diagnostic | Repaired by using pullback density/score residuals on held-out base and stress banks; forward-input Gaussian moments are explicitly ineligible evidence. |
| Incomplete broad compatibility suites | Repaired by a bounded pre-calibration full-suite gate with timeout localization and one retry. |
| Stale dimensional claim | The q=20 parameter target is recorded as four-dimensional; a separate scaling ladder is required for the future high-dimensional aim. |
| Environment mismatch | TensorFlow/TFP, batch-native static shapes, XLA default, trusted GPU execution, and memory growth are hard gates. |
| Candidate failure mistaken for direction failure | Between-phase classification distinguishes exactness, tuning, mixing, and evidence failures. |

Historical Phase 8 refresh verdict (superseded by the closeouts below):
`PHASE8_REFRESH_REQUIRES_SUBPLAN_SKEPTICAL_AUDIT_BEFORE_RESEARCH_LAUNCH`.
The historical trusted idle-GPU admission probe returned
`no_idle_policy_permitted_gpu`, but the user then authorized the bounded shared-
GPU exception. The GPU0/XLA smoke passed with memory growth and no allocation
failure; it remains non-claiming mechanics evidence. The refreshed Phase 8
subplan now freezes the component count, temperature ladder, training search
budget, optional-joint-arm envelope, ESS/MCSE and declared-region travel
targets, and attempt cap. The repository-default GPU route is operationally
defined by `scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh`; it
sets one GPU and memory growth before TensorFlow import and performs no
idle-GPU probe. Any outer Codex approval boundary is platform provenance, not a
scientific gate.
The repository/service distinction and one-time narrow-permission guidance are
recorded in `docs/plans/bayesfilter-gpu-default-execution-boundary-2026-08-29.md`.
The first Phase 8 cost-pilot launch (`attempt-02-default-gpu`) reached GPU 0 and
completed the beta-0 checkpoint but timed out with exit code `124` at the
1,800-second cap. This is a bounded graph/compile localization trigger, not a
candidate result. The next authorized repair is the launcher’s diagnostic-only
`target-localization` mode with a fresh output directory and a 900-second cap;
the preserved record is
`phase8-calibration/attempt-02-default-gpu/timeout.json`. A 900-second
localization then completed both B=8 target calls but timed out at B=256, and
the final 2,700-second B=8 validation-size cost retry completed both B=8 charts
and B=32 chart 0 before timing out during the remaining B=32 receipt. The
records are under `attempt-03-target-localization/` and
`attempt-04-cost-smallbank/timeout.json`. The Phase 8 C1 allocation is now
exhausted; C2--C5 do not launch without a new budget or a reviewed target-graph
optimization. This is a continuation blocker for the current execution, not a
rejection of the mathematical direction.

The user subsequently authorized a bounded graph-repair diagnostic after the
GPU service boundary was switched. The first repair ordering timed out while
compiling a q=20 direct `B=32` parity graph; the repaired `B=8`-chunk route
completed a finite 32-row prefix in `192.694` seconds but timed out before the
remaining 224 rows. The result is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-graph-repair-result-2026-08-30.md`.
This does not reopen C1 or authorize C2--C5: a new graph-level optimization or
execution design and a fresh reviewed budget are required.

The strict-backend localization and parity repair then passed on 2026-08-30.
The repaired parity receipt found value and score residuals below the frozen
backend tolerances and verified a GPU/XLA strict-backend trainer. The new
bounded cost-rescue subplan is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-strict-backend-cost-subplan-2026-08-30.md`.
Its full-256 K=2 cost pilot then passed in `261.52175762609113` seconds,
selected B=32, and stayed below the 4-GiB allocator cap. The old C1 allocation
remains closed. Before C2, refresh the calibration subplan to bind the strict
backend, require a larger-batch parity check, and set a new bounded training
allocation; C2--C5 and Phase 9 remain closed until that refresh is audited.

The refreshed C2 strict-backend subplan then passed its B=8 semantic parity
prerequisite and completed all eight architecture/root rows in
`1074.309582018992` seconds. Every row was finite and replayable, all four
architecture groups passed the learned-map reliability screen, and all
within-row paired held-out reverse-KL intervals were negative. The detailed
receipt and interpretation are in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-strict-calibration-result-2026-08-30.md`.
This closes C2 as calibration feasibility only: the held-out pullback-score
residuals remain large, so no whitening, architecture, HMC, or posterior claim
is supported. Compact-high and compact-low remain viable calibration
representatives without a statistically supported ranking. The next subplan
must test positive-temperature branching and temperature overlap under a fresh
budget; Phase 9 confirmation remains closed.

## Phase 9A fresh-map tuning preflight closeout, 2026-08-31

The separately audited Phase 9A subplan rebuilt two fresh K=2 compact-high
charts, used the strict q=20 bridge, and exercised the active fixed-transport
tuner.  The first harness attempt was repaired.  A localized cap repair then
passed chart-0/beta-0 with selected epsilon `0.810010` and acceptance
`0.859967`, one trace per reusable HMC graph, and `1402670592` bytes (about
1.31 GiB) peak allocation.  The first complete launch passed all three chart-0 scopes but
failed chart-1/beta-0 when the tuner requested epsilon `1.256879` above cap
`1.0`.  A final declared cap `2.0` repair repeated the scope failure: after
finite screens at epsilon `0.628978` (acceptance `0.998950`) and `1.205189`
(acceptance `0.939618`), the next requested epsilon was `2.410379`.

This is a reproducible chart-specific tuning-boundary failure, not a target or
bridge mathematical failure.  The six-scope handoff criterion failed, so the
shared replica-exchange controller was not run and Phase 9B confirmation,
whitening, HMC, posterior, and default-readiness claims remain blocked.  The
full decision and inference-status tables are in
`docs/plans/bayesfilter-ssl-lstm-q20-phase9a-fresh-tuning-preflight-result-2026-08-31.md`.
The next action, if authorized, is a new bounded chart-1/beta-0 repair plan
that chooses and justifies longer adaptation or a fixed candidate grid; no
additional cap widening is implicit.

## Phase 9A chart-1/beta-0 program-repair closeout, 2026-09-01

The localized repair subplan
`docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-subplan-2026-09-01.md`
was executed after a skeptical audit. The runner now binds source-owned
profiles and scope pins, fresh seeds, measured joint grids, durable start and
failure manifests, per-call progress records, and the GPU memory-growth
launcher. Attempt-04 completed the chart-1/beta-0 scope with all four
declared pairs measured, two selection replications, disjoint held-out
verification, and a finite/mobile provisional handoff `(epsilon=0.55,L=3)`,
but reused attempt-03 seeds for a provenance-only replay. The final fresh-seed
record is attempt-05, which repeats those mechanics with a distinct v4 seed
namespace.

This closes the localized mechanics repair only. The run status is
`PASS_PHASE9A_SCOPE_PREFLIGHT_PARTIAL`; five scope handoffs were intentionally
not attempted and the shared transition controller was not run. High
acceptance, very short chains, large descriptive R-hat values, and large
pullback-score residuals remain unresolved evidence limitations. Phase 9B,
whitening, posterior, convergence, and default-readiness work remain closed.
The detailed result and reset memo are
`docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-result-2026-09-01.md`
and
`docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-reset-memo-2026-09-01.md`.
Any full Phase 9A replay requires a new separately audited budget and plan.
