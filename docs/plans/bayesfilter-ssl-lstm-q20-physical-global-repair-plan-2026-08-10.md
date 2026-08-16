# SSL-LSTM q=20 physical-coordinate global repair plan (2026-08-10)

Status: `AUDITED_EXECUTION_AUTHORIZED`

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can independent weighted sampling and physical-coordinate parallel tempering represent both known seed-B SSL-LSTM posterior regions without inheriting the failed NeuTra geometry? |
| Candidate mechanisms | A normalized two-component local-Gaussian mixture proposal with exact target/proposal importance correction for mass; TFP replica-exchange fixed HMC in one fixed physical affine chart for transitions. |
| Expected failures | Proposal weights may collapse or miss another mode; the chart may retain anisotropy; the temperature ceiling may not cause basin forgetting; or exact target cost may be prohibitive. |
| Promotion criterion | Weight lane: all target rows valid, central proposal weight ESS at least 20% of draws, maximum normalized weight at most 0.05, independent-batch 95% interval half-width at most 0.10, and scale-sensitivity estimates within 0.10. Transition lane: finite/status-valid traces, every adjacent pair communicates, both starting signs reach the opposite cold sign, at least one complete round trip, and a hot replica changes physical sign through local HMC rather than only swapping initialized states. |
| Promotion veto | Non-finite or invalid target row; failed known-law importance test; weight-gate failure; target/source identity mismatch; invalid swap identity; or missing required trace. |
| Continuation veto | Harness invalidity, target identity/parity failure, wall cap, or both proposed physical charts failing local two-region integration.  Zero global transitions alone is a repair trigger, not a continuation veto. |
| Repair trigger | Low ESS/high max weight triggers AIS or annealed SMC.  Poor physical HMC acceptance triggers smaller step or revised chart.  Good swaps without hot forgetting triggers a hotter/denser ladder. |
| Explanatory diagnostics | Point estimates, proposal component counts, log-weight range, per-batch mode estimates, HMC/swap acceptance, sign paths, identity travel, and runtime. |
| Must not be concluded | Exhaustive mode discovery, default readiness, NeuTra repair, posterior predictive validity, or sampler superiority. |

## Evidence contract

The exact target is the original four-parameter batch-native SSL-LSTM q=20 target,
signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
The two checked physical stationary representatives and their finite-difference local
precisions come from the accepted root-cause geometry artifact.

The mass comparator is self-normalized importance sampling from a normalized
Gaussian mixture centered at those representatives.  The exact unnormalized target
value and normalized proposal density determine every weight.  This corrects the
proposal; it does not assume its equal component weights are posterior weights.  A
known unequal-weight and unequal-scale analytic mixture is the pre-run authority.

Replica exchange addresses movement, not weights.  Its comparator is the failed
NeuTra-coordinate canary and ordinary physical fixed HMC.  Raw cold occupancy and
initial chain balance cannot pass the mass criterion.  The primary artifacts are
written under:

`docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/r1/`

## Baseline ladder

| Rung | Role | Use |
|---|---|---|
| Equal local-Laplace plug-in weights | Naive mass baseline | Explanatory only; not an authority. |
| Direct Gaussian-mixture importance sampling | Exact corrected weighted baseline | Primary mode-mass candidate if weight diagnostics pass. |
| AIS / annealed SMC | Enhanced weighted repair | Execute only if direct importance weights fail. |
| Ordinary physical fixed HMC | Naive transition baseline | Expected to remain locally trapped. |
| Physical affine replica exchange | Exact transition candidate | Must demonstrate local hot sign forgetting and replica travel. |
| Multimodally trained NeuTra | Later learned component | Not trained until global weighted coverage is accepted. |

## Defaults and numeric audit

| Choice | Provenance/status | Justification | Failure mode / early diagnostic |
|---|---|---|---|
| Physical target coordinates | Derived repair | NeuTra separated source modes from `1.281` physical units to `23.707` latent units | Target signatures, representative value/score parity, and local HMC checks |
| Proposal centers and covariances | Measured local MAP/Hessian artifacts | Two checked stationary regions with stable SPD local precision | Local Gaussian tails may mismatch target or miss modes; ESS, max weight, invalid rows, scale sensitivity |
| Equal proposal component probability | Convenience sampling choice, not posterior hypothesis | Guarantees proposal support in both known regions; exact weights correct it | Proposal density must include the equal mixture and target weights must not be inferred from component counts |
| Proposal covariance scales `0.5,1,2` | Reviewed sensitivity ladder | Tests narrower and broader local coverage | Divergent estimates or weight collapse veto direct IS |
| Central batches `8 x 100` | Convenience uncertainty design | Reuses measured `25 workers x 4 rows` CPU/XLA lane and yields independent batch estimates | Eight batches remain modest; batch interval and ESS gates prevent overclaim |
| Sensitivity batches `2 x 100` per alternate scale | Convenience diagnostic | Enough to detect gross scale dependence without dominating budget | Descriptive only; failure triggers AIS/SMC |
| ESS fraction `>=0.20` | Reviewed diagnostic threshold | Requires at least 160 effective central draws, well above a few dominant weights | Still not proof of finite variance; maximum-weight and sensitivity gates also required |
| Maximum normalized weight `<=0.05` | Reviewed diagnostic threshold | Rules out any one row carrying more than 5% of central estimate | Heavy tails may remain unseen; independent batches and scale sensitivity required |
| Mode interval half-width `<=0.10` | Reviewed diagnostic precision goal | Enough to distinguish gross collapse from a near-balanced result | Not a scientific equivalence margin or posterior claim by itself |
| Sensitivity difference `<=0.10` | Reviewed stability diagnostic | Gross proposal-scale dependence indicates unreliable weights | Passing does not prove all proposal families agree |
| Pool `25 x 4`, XLA, CPU hidden-GPU | Measured inherited execution lane | About 7 s per warm 100-row evaluation in prior artifacts | Current load or code drift; timing canary and 1,800 s weight cap |
| Physical chart | Equal-weight law-of-total-covariance of two local Gaussians | Symmetric warm-start that includes within-mode and between-mode scales | Provisional equal weight can distort geometry; transformed local curvature and two-region canary expose it |
| Temperatures `(1,.5,.25,.125,.0625,.03125)` | Convenience geometric ladder hypothesis | Six replicas allow five adjacent exchanges and a 32x barrier flattening | Adjacent overlap or hot forgetting may fail; complete travel/sign traces required |
| Cold step `0.05`, scaled by `1/sqrt(beta)` | Derived from mapped maximum precision about 367, stability scale about 0.104 | Keeps cold step below half the local quadratic scale in both modes | Nonquadratic geometry; per-temperature acceptance/status veto |
| `L=8`, 12 transitions, two sign-separated chains | Convenience mechanics/travel canary | Hot trajectory length about 2.26 versus mapped mode separation 1.93; 12 steps allow a full six-temperature round trip in principle | Still short and not stationary; cannot produce posterior draws or weights |
| Transition cap `6,000 s` | Derived from stage-1 `503.9 s` for 4 transitions at `L=3`, adjusted for `L=8` and six replicas | Bounded within user-granted headroom | Runtime breach stops before a longer campaign |

All thresholds are stage-specific reviewed diagnostic gates, not repository defaults.

## Execution plan

1. Implement TensorFlow diagnostic helpers for Gaussian-mixture sampling/log density,
   self-normalized weights, ESS, maximum normalized weight, weighted mode fraction,
   and independent-batch interval.  Test equal/unequal-weight and unequal-scale known
   laws, proposal mismatch, replay, and weight degeneracy detection.
2. Run a 100-row SSL timing/parity canary through the existing batch-native
   multicore pool.  Verify target signature and both representative values against
   the geometry artifact before opening mass seeds.
3. Run eight central and four scale-sensitivity 100-row independent proposal batches.
   Archive rows, target values/status, proposal log density, log weights, component
   labels, seeds, pool topology, and uncertainty diagnostics.  Stop or trigger AIS/SMC
   if any primary weight gate fails.
4. Construct the fixed physical chart from both local covariances and between-mode
   displacement.  Record its exact factor, mapped mode distance, and each mapped
   local precision.  Run local two-region HMC mechanics before tempering.
5. If local mechanics pass, run the bounded six-temperature, two-chain, 12-transition
   replica-exchange canary.  Archive full accepted states, physical signs, local-HMC
   pre-swap signs, swap matrices, identity paths, and status.
6. Write result and reset memo.  Only if the weight lane and transition lane both pass
   may a later plan issue a posterior archive or begin multimodal NeuTra training.

## Compute and attempt budget

- Importance helper/tests and analytic validation: `300 s`.
- SSL weight lane including pool startup, canary, 12 batches, and one localized
  infrastructure retry: `1,800 s`, at most 1,200 target rows.
- Physical local/replica canary: one chart, one local step canary, one replica run,
  and at most one smaller-step retry if local acceptance fails; `6,000 s`.
- Whole stage cap: `8,100 s`.  No detached run and no posterior material campaign.

## Skeptical plan audit

| Risk | Resolution |
|---|---|
| Wrong baseline | Analytic known-law IS and ordinary physical HMC are explicit baselines. |
| Proxy promotion | Laplace weights, swap acceptance, raw occupancy, and runtime are explanatory only.  Corrected weights are primary for mass; travel/forgetting are primary for transitions. |
| Missing stop | Known-law, identity, target status, ESS/max-weight, uncertainty, sensitivity, local-HMC, travel, and wall-time vetoes are explicit. |
| Unfair comparison | Global authority uses original target coordinates; failed NeuTra coordinates are not forced on the repair. |
| Hidden equal-weight assumption | Equal mixture is proposal design only and appears in the proposal log density.  Posterior weights come from target/proposal correction. |
| Missed third mode | Two-mode proposal cannot certify exhaustive discovery.  This remains a nonclaim; later independent hot/SMC discovery is required for default readiness. |
| Historical drift | Mass uses current batch target only after exact target signature and representative value parity.  No archived trainer restoration is needed. |
| Weight heavy tails | ESS, maximum normalized weight, independent batches, and covariance-scale sensitivity are joint gates; failure triggers AIS/SMC. |
| Tempering self-validation | Initial balance and raw occupancy cannot validate weights.  Hot sign change before swaps distinguishes local forgetting from exchange of initialized states. |
| Artifact mismatch | Every scientific tensor is serialized with shape/dtype/SHA-256; JSON refuses overwrite. |

The audit passes for a bounded diagnostic campaign.  It does not authorize posterior
issuance, NeuTra training, or posterior-predictive claims.

## Pre-mortem

Direct importance sampling could look stable because all tested proposals share the
same two discovered modes; an undiscovered third mode would remain absent.  Tempering
could show cold sign changes by swapping the two initialized signs without a hot
replica ever crossing locally.  The plan therefore records pre-swap local-HMC signs
and labels the two-mode scope explicitly.  Conversely, a failed chart or ladder does
not invalidate the target or importance result; it rejects only that transition
candidate and triggers a revised physical kernel.

