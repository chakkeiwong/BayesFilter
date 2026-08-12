# SSL-LSTM q=20 physical annealed-SMC material result (2026-08-10)

## Verdict

Adaptive globally resampled SMC passed every predeclared finite-sample reliability
gate for relative mass over the two known proposal-supported sign regions.

Eight independent 100-particle cESS `0.80` runs gave negative-region estimates

`[0.5024,0.3623,0.4702,0.3361,0.5109,0.5271,0.5234,0.5345]`.

Their mean is `0.47087` with independent-batch 95% t interval
`[0.40573,0.53602]` and half-width `0.06514`.  Every terminal ESS fraction was
at least `0.8783`; every maximum normalized weight was at most `0.03854`; every
run retained 50--65 distinct initial roots with at least 23 roots from each known
sign region.

Two independent cESS `0.70` runs averaged `0.41895`.  The difference from cESS
`0.80` was `0.05192`, below the `0.08` sensitivity gate.

The direct classification is
`TWO_KNOWN_REGION_SMC_WEIGHT_AUTHORITY_VIABLE`.  This does not establish a full
posterior because every run began from the same two known-region proposal and no
HMC mutation changed sign.  Exhaustive mode discovery and global transition
mixing remain unresolved.  No posterior archive or predictive run is issued yet.

## Claimed and computed quantities

| Item | Classification |
|---|---|
| Claimed target | Stable finite-sample relative mass evidence for the two known SSL-LSTM physical sign regions. |
| Quantity computed | Ten independent TensorFlow/TFP annealed-SMC terminal beta-1 pre-resampling weighted sign measures: eight cESS `0.80`, two cESS `0.70`. |
| Relation | Equal to the declared two-known-region finite SMC target; different from exhaustive full-posterior mass. |
| Source anchor | Exact target `9a86e6...7278`, adapter `a8be6c...166f3`, physical chart/proposal from the bound AIS runner, verified canary, and post-run recovery verification of all 990 child tensors plus two aggregates. |
| Not proved | Absence of undiscovered modes, HMC stationarity, global transition mixing, NeuTra repair, full posterior correctness, predictive validity, or default readiness. |

## Material gate table

| Gate | Result | Status |
|---|---:|---|
| All ten child mechanics runs passed | `10/10` | Pass |
| Each central terminal ESS fraction `>=0.50` | minimum `0.87831` | Pass |
| Each central maximum weight `<=0.05` | maximum `0.03854` | Pass |
| Each central unique-root fraction `>=0.30` | minimum `0.50` | Pass |
| At least 10 positive and 10 negative roots per central run | minima `23` and `23` | Pass |
| Eight-batch interval half-width `<=0.08` | `0.06514` | Pass |
| cESS `0.70/0.80` mass difference `<=0.08` | `0.05192` | Pass |
| Wall time `<=4200 s` | `2487.58 s` | Pass |

## Central evidence

| Batch | Estimate | Terminal ESS fraction | Maximum weight | Unique roots | Positive roots | Negative roots | Stages |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.50243` | `0.93164` | `0.03352` | 65 | 36 | 29 | 5 |
| 1 | `0.36234` | `0.91722` | `0.02679` | 61 | 36 | 25 | 4 |
| 2 | `0.47021` | `0.99996` | `0.01026` | 50 | 27 | 23 | 6 |
| 3 | `0.33612` | `0.96577` | `0.02336` | 63 | 38 | 25 | 4 |
| 4 | `0.51092` | `0.93106` | `0.02996` | 57 | 23 | 34 | 4 |
| 5 | `0.52713` | `0.93767` | `0.02399` | 61 | 33 | 28 | 4 |
| 6 | `0.52335` | `0.97953` | `0.01893` | 60 | 28 | 32 | 5 |
| 7 | `0.53449` | `0.87831` | `0.03854` | 58 | 28 | 30 | 5 |

The between-run standard deviation is `0.07792`.  The interval is therefore the
primary mass statement; quoting `0.47087` without `[0.40573,0.53602]` would hide
the measured Monte Carlo uncertainty.

No HMC mutation changed physical sign in any central run.  This is explanatory,
not a weight-gate failure: SMC can correctly reweight and resample particles from
a normalized proposal that already covers both known regions.  It does limit the
claim to that proposal-supported scope.

## Sensitivity evidence

The two cESS `0.70` batch estimates averaged `0.41895`.  Their difference from the
cESS `0.80` central mean is `0.05192`, which passes the frozen `0.08` gate.  Passing
two cESS settings does not prove universal schedule invariance; it rejects the much
larger schedule instability observed in sparse AIS (`0.16555`) under this reviewed
control.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Admit SMC as a two-known-region weight authority | All terminal weight, uncertainty, ancestry, cESS sensitivity, validity, receipt, and budget gates passed | No declared material veto | Third/unseen modes and local mutation isolation | Freeze this mass evidence and return to physical replica-exchange travel validation | That `0.47087` is exhaustive full-posterior mass |
| Do not issue a posterior archive yet | Weight lane passed only within known-region scope | Global transition/round-trip and mode-coverage gates remain | Whether a frozen sampler traverses and represents the relevant target globally | Run longer physical replica exchange with repeated round-trip and cold convergence gates | HMC stationarity, full posterior correctness, or predictive validity |
| Do not retrain NeuTra yet | Independent global weight evidence improved, but no eligible posterior sample archive exists | Upstream transition gate remains | How to construct globally representative training samples | Wait for an eligible physical transition archive | That learned whitening is repaired |

## Inference-status table

| Inference item | Status |
|---|---|
| Hard veto screen | Passed all exact-target, finite, XLA, terminal-policy, receipt, ancestry, weight, sensitivity, and wall-time gates |
| Viable candidates | Adaptive globally resampled physical SMC is viable for two-known-region mass; physical replica exchange remains the transition candidate |
| Statistically supported ranking | None |
| Statistical evidence | Eight independent run estimates and their t interval support finite precision for the stated two-region mass target |
| Descriptive-only differences | Individual batch estimates, stages, acceptance, runtimes, beta paths, log normalizers, and zero HMC sign changes |
| Default readiness | Not ready |
| Next evidence needed | Repeated physical temperature round trips and cold convergence under frozen transition settings, plus an explicit mode-coverage limitation |

## Engineering, numerical, and scientific ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Seventeen focused SMC tests passed after the schema repair.  The original JSON manifests linked 780 child receipts; recovery verified those, the 210 immutable pre-resampling tensors hidden by key collisions, and two aggregates.  Every child used fresh pinned batch-4 CPU/XLA workers. |
| Numerical/sampler validity | Terminal weights, ESS, ancestry, uncertainty, and cESS sensitivity passed for ten independent runs.  Local HMC remained sign-trapped. |
| Scientific interpretation | Relative mass across the two known proposal-supported sign regions is now viable finite SMC evidence.  Full-posterior correctness remains unproved. |

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`; dirty concurrent worktree recorded |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`; Python 3.13.13; TensorFlow 2.20.0; TFP 0.25.0 |
| Device | CPU-only diagnostic exception; GPU hidden before import; XLA enabled |
| Topology | Fresh 25-worker waves; four particles/four pinned cores per worker; CPUs `0--99` |
| Central | 8 independent runs, 100 particles each, cESS `0.80` |
| Sensitivity | 2 independent runs, 100 particles each, cESS `0.70` |
| Mutation | One HMC move per nonterminal stage, step `0.03`, `L=4` |
| Resampling | Global systematic every nonterminal adaptive stage; none at beta 1 |
| Seeds | Ten disjoint child domains separated by at least 10,000; internal initialization/resampling/mutation domains disjoint |
| Wall time | `2487.5778 s` |
| Original manifest-linked receipts | `780` child plus `2` aggregate, zero mismatches |
| Recovered immutable receipts | `210` child pre-resampling tensors hidden by the v1 flat-map key collision, zero mismatches |
| Complete verified inventory | `990` child plus `2` aggregate tensor files, zero mismatches |
| Recovery artifact | `docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/receipt-recovery-v1.json`, SHA-256 `3aea988e7b27381a6b62e7a2d452db8251b9bd7d8b9f5e68ad08fcbe711b6d97` |
| Artifact root | `docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/` |
| Terminal artifact | `docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/material.json` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-physical-annealed-smc-material-plan-2026-08-10.md` |
| Result | This file |

## Post-run red team

The strongest alternative explanation for success is explicit proposal coverage:
every run began from local Gaussian approximations around the two already known
regions.  Global resampling can correct their relative finite weights but cannot
discover a region the proposal never samples.  Zero sign-changing HMC proposals
confirms that the mutation kernel did not independently repair that limitation.

The strongest concern inside the declared scope is the batch spread (`0.336--0.534`).
The independent interval honestly includes that variability and passed the frozen
precision gate, but more runs would narrow the interval.  A result that would
overturn admission is a fresh reviewed seed family or cESS control producing a
failed interval, weight, ancestry, or sensitivity gate.  Evidence needed to expand
the claim beyond two known regions is an independent mode-discovery/global-travel
argument, not more repetitions of the same local proposal.

## Post-run receipt recovery

The historical stage-v1 writer used one flat receipt map.  At each of 35
nonterminal stages, six post-resampling entries replaced same-named
pre-resampling entries in that map.  The tensor files themselves were not
overwritten, and terminal stages were unaffected.  Therefore the original
material artifact reported 780 linked child receipts although 990 child tensor
files existed.

The recovery diagnostic made no target calls and changed no historical source
artifact.  It hashed and parsed every tensor, checked dtype and shape, reproduced
normalized weights and ESS, checked resampling ancestry and stage-to-stage
continuity, and reproduced all ten terminal mass estimates and both aggregate
tensors.  All checks passed.  The estimator and scientific classification are
unchanged; the earlier receipt count was wrong.

Future stage-v2 artifacts use nested `receipts.pre` and `receipts.post` maps, so
same-named tensors cannot collide.  Historical v1 artifacts remain readable but
must be interpreted with the recovery inventory rather than silently rewritten.
