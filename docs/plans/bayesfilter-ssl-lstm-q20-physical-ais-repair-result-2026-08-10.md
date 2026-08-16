# SSL-LSTM q=20 physical AIS repair result (2026-08-10)

## Verdict

The AIS campaign completed successfully as an experiment and rejected the
sparse-rejuvenation AIS candidate as a posterior-weight authority.

The exact 64-bridge central calculation materially repaired direct-importance
weight concentration: across 800 paths, ESS fraction was `0.4514`, maximum
normalized weight was `0.01659`, and the eight independent batch estimates gave
a 95% t interval `[0.4828,0.5681]` with half-width `0.04267`.  The pooled negative
region estimate was `0.5253`.

That estimate is not accepted posterior mass.  Two predeclared gates failed:

1. None of 800 central paths changed physical sign despite eight HMC moves per
   path.
2. The independent 32-bridge sensitivity estimate was `0.3598`, differing from
   the 64-bridge estimate by `0.1655`, above the `0.08` gate.  Its ESS fraction
   was only `0.1013` and one normalized weight was `0.2101`.

The correct decision is `STOP_AIS_WEIGHT_PROMOTION_AND_PLAN_ANNEALED_SMC`.
Resampling and ancestry diagnostics are the next repair.  No posterior archive or
predictive run is authorized.

## Claimed and computed quantities

| Item | Classification |
|---|---|
| Claimed target | Determine whether annealed importance sampling can produce reliable relative mass evidence for the two known SSL-LSTM physical sign regions. |
| Quantity computed | Exact TensorFlow/TFP linear AIS weights from a normalized equal-component two-local-Gaussian proposal, with bridge-correct freshly bootstrapped HMC in the fixed physical chart. |
| Relation to target | Correct finite AIS computation for the two-mode proposal; promotion failed because movement and schedule-stability gates failed. |
| Source anchor | Target `9a86e6...7278`, adapter `a8be6c...166f3`, geometry SHA-256 `dc3dd7...eeb`, direct-IS comparator SHA-256 `ed683d...075`, helper SHA-256 `091ec7...137`. |
| Not proved | Exhaustive mode discovery, posterior mass `0.5253`, HMC convergence, global stationarity, NeuTra repair, posterior correctness, or predictive validity. |

## Harness correction

Installed TFP 0.25's generic AIS driver passes prior kernel results between
changing bridge targets.  TFP HMC kernel results cache target values and gradients,
so using that driver directly with HMC would evaluate the first trajectory step
from stale bridge data.  The local TensorFlow helper re-bootstraps HMC under every
bridge where HMC is scheduled before calling `one_step`.

Known-law XLA tests passed identical normalized targets and unequal-weight,
unequal-scale Gaussian mixtures.  Bridges without rejuvenation use the identity
kernel, which is exactly invariant for each bridge.  Proposal and target values are
carried across identity steps; this changes mixing efficiency, not the AIS weight
identity.

## Timing repair

The full-rejuvenation `r1` canary was finite, status-valid, and accepted `98.44%`
of HMC proposals, but required `778.0 s` for 16 bridge moves.  The originally
planned material ladder extrapolated to about `28,008 s`, violating the frozen
`7,200 s` cap.  No material seeds were opened under that design.

The exact sparse repair kept 64/32 weight bridges and scheduled HMC every eighth
bridge.  Its `r2` canary passed in `107.9 s` with two HMC moves.  That admitted the
material `r3` campaign.  Twenty-five fresh spawned workers evaluated four paths
each on disjoint four-core CPU groups (`0--99`) for every independent batch.  The
ten material waves completed in `3391.40 s`.

## Material evidence

### Central 64-bridge lane

Batch estimates were:

`[0.4773, 0.4411, 0.5306, 0.5786, 0.5152, 0.5837, 0.5709, 0.5062]`

| Diagnostic | Result | Gate |
|---|---:|---|
| Valid and finite target paths | `800/800` | Pass |
| Pooled negative-region estimate | `0.525332` | Descriptive only |
| Pooled ESS | `361.09/800` (`0.45136`) | Pass, required `>=0.30` |
| Maximum normalized weight | `0.016594` | Pass, required `<=0.02` |
| Eight-batch mean | `0.525454` | Descriptive only |
| Eight-batch 95% t interval | `[0.482787,0.568122]` | Pass, half-width `0.042668 <=0.08` |
| Initial-to-terminal sign changes | `0/800` | **Fail**, required at least one |
| Log normalizer ratio estimate | `-34.3675` | Explanatory only |

High HMC acceptance (`0.971--0.984` by batch) does not rescue the zero-movement
failure.  It shows local proposals were numerically accepted, not that they crossed
the separated sign regions.

### 32-bridge sensitivity lane

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Valid and finite target paths | `200/200` | Hard validity pass |
| Negative-region estimate | `0.359783` | Descriptive sensitivity result |
| ESS | `20.27/200` (`0.10133`) | Severe concentration |
| Maximum normalized weight | `0.21005` | Severe concentration |
| Difference from 64 bridges | `0.165549` | **Fail**, required `<=0.08` |
| Initial-to-terminal sign changes | `0/200` | No global movement |

The 32-bridge result does not establish that negative mass is `0.360`.  It
establishes that the candidate's apparent mode-weight result is not robust to the
predeclared schedule comparison.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject sparse AIS as weight authority | Four reliability gates passed; movement and schedule gates failed | Target, finite, identity, XLA, receipt, and wall-time vetoes all passed | Whether resampling prevents weight collapse and enables ancestry to represent both regions | Implement annealed SMC with adaptive conditional-ESS placement, resampling, and ancestry diagnostics | That negative posterior mass is `0.5253` or `0.3598` |
| Retain 64-bridge design as an SMC warm start | ESS, maximum weight, and interval gates passed | No numerical veto | Lack of cross-sign transitions means local particles depend on proposal coverage | Use the same exact target/chart as a baseline schedule, not as accepted posterior evidence | Mode completeness or global transport |
| Keep posterior archive blocked | Required AIS gates failed | Upstream posterior-authority veto remains | Stable mass under resampling and exhaustive mode coverage | SMC, then independent travel/confirmation gates | HMC convergence, posterior correctness, or predictive validity |

## Inference-status table

| Inference item | Status |
|---|---|
| Hard veto screen | All 1,000 paths valid and finite; target identities, XLA execution, source hashes, and receipts passed |
| Viable candidates | Physical-chart annealed SMC remains viable; sparse AIS is rejected as a weight authority |
| Statistically supported ranking | None |
| Descriptive-only differences | Point estimates, acceptance, runtimes, log normalizers, and per-batch ESS/tails |
| Default readiness | Not ready |
| Next evidence needed | Resampling-based SMC with conditional-ESS temperature placement, ancestry diversity, stable region mass across independent batches/schedules, and explicit mode-coverage limits |

## Engineering, numerical, and scientific ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Five AIS known-law tests and three material-harness tests passed; CPU/XLA execution completed; every batch wrote target-status and tensor receipts. |
| Numerical/sampler validity | AIS weights are mathematically correct and 64-bridge weight concentration passed, but zero sign movement and schedule instability reject this candidate. |
| Scientific interpretation | Relative mode mass remains unresolved.  Results identify a specific need for resampling/global ancestry; they do not invalidate the target or physical multimodal direction. |

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`; shared worktree dirty and recorded |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`; Python 3.13.13; TensorFlow 2.20.0; TFP 0.25.0 |
| Device | CPU-only diagnostic exception; GPU hidden before import; XLA enabled |
| Topology | 25 fresh spawned workers per wave; four paths and four pinned cores per worker; CPUs `0--99` |
| Material design | Eight independent `100 x 64` central batches; two independent `100 x 32` sensitivity batches; HMC every eight bridges; step `0.03`, `L=4` |
| Seeds | 250 unique proposal and 250 disjoint unique AIS stateless seeds, bound by family/batch/worker |
| Wall time | `3391.3985 s`, below `7200 s` cap |
| Artifact root | `docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3/` |
| Terminal artifact | `docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3/material.json` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-physical-ais-repair-plan-2026-08-10.md` |
| Result | This file |

## Post-run red team

The strongest alternative explanation for zero sign changes is not an error in AIS
weights; the normalized two-mode proposal initializes particles in both regions, so
local movement is not required for unbiased weighting in the ideal infinite-sample
limit.  It is nevertheless a valid promotion veto here because finite reliability
would otherwise depend entirely on the proposal's assumed two-region coverage.  A
third missed mode or one badly scaled known region would not be repaired by these
local paths.

The strongest alternative explanation for the schedule failure is high Monte Carlo
variance in only 200 sensitivity paths.  That does not justify accepting the
central estimate: the observed sensitivity lane also has ESS `20.3` and maximum
weight `0.210`, exactly the instability the gate was designed to expose.

Evidence that would overturn this rejection is an independent resampling-based
annealed method with stable mass across schedules/batches, adequate ancestry from
both known regions, valid target receipts, and an explicit limit on undiscovered
modes.  The weakest evidence remains exhaustive mode discovery because every
proposal in this campaign was constructed from the two already known regions.
