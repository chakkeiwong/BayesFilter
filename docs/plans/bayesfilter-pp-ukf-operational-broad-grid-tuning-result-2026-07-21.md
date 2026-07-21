# PP-UKF Operational Broad-Grid Tuning Result

Date: 2026-07-21

Decision: `WITHDRAWN_INVALID_ONE_PASS_TUNING_RESULT`

Correction: the original `viable_pair_set` interpretation is withdrawn. The
`L=5` point estimate was `0.75979`, outside the stated `[0.65,0.75]` practical
region. The historical classifier admitted it only because its working
interval overlapped the band and remained inside the wider repair region. That
admission rule was wrong relative to the stated target. This file remains
historical evidence of the one-pass run; it must not authorize validation or
sampling. The active repair plan is
`docs/plans/bayesfilter-pp-ukf-state-continuing-epsilon-repair-plan-2026-07-21.md`.

## Outcome

The historical broad-grid campaign completed all six independently tuned
primaries and both required same-epsilon neighbor guards. The terminal grid
status is `viable_pair_set`.

Under the historical, now-withdrawn heuristic, one primary was labeled viable:

`L=5`, epsilon `0.8426345584765329`.

Its `L=4` and `L=6` same-epsilon guards did not survive. Because the primary
mean itself was above the band, `L=5` is not a viable candidate and no fresh
fixed-kernel validation is authorized from this result. The next justified
action is state-continuing epsilon repair with fresh evidence.

## Complete Results

| Role | L | Epsilon | Mean acceptance probability | Working interval | Disposition | Hard veto |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Primary | 3 | 0.8724049589 | 0.82038 | [0.80822, 0.83254] | needs higher epsilon | none |
| Primary | 5 | 0.8426345585 | 0.75979 | [0.74416, 0.77541] | provisional viable | none |
| Primary | 9 | 0.7489709357 | 0.83207 | [0.82226, 0.84187] | needs higher epsilon | none |
| Primary | 13 | 0.6908655196 | 0.85382 | [0.84139, 0.86624] | needs higher epsilon | none |
| Primary | 18 | 0.6813265223 | 0.84624 | [0.83767, 0.85480] | needs higher epsilon | none |
| Primary | 25 | 0.6800917536 | 0.81148 | [0.79508, 0.82788] | needs higher epsilon | none |
| Same-epsilon guard of L=5 | 4 | 0.8426345585 | 0.83309 | [0.82071, 0.84547] | needs higher epsilon | none |
| Same-epsilon guard of L=5 | 6 | 0.8426345585 | 0.76570 | [0.75383, 0.77758] | needs higher epsilon | none |

Every pair used four chains and three fresh replications. The intervals are the
reviewed bounded tuning heuristic over twelve chain-run means, not confidence
intervals proving a stochastic ranking. All samples were discarded.

## Barrier And Resource Results

| Gate | Result |
| --- | --- |
| Fixed-identity handoff | Passed; identity metric content signature unchanged |
| Six-primary barrier | Passed, 6/6 completed |
| Viable-primary expansion | One primary generated exactly `L=4` and `L=6` guards |
| Same-epsilon guard barrier | Passed, 2/2 completed; epsilon retuning false |
| Target/value/score/status health | No hard rejection in any completed pair |
| Native divergence | Not exposed by the TFP HMC kernel; not claimed to be zero |
| GPU/XLA/memory policy | RTX 4080 SUPER, XLA on, float64, TF32 recorded, memory growth verified |
| Attempt 02 wall time | 3,063.630 s |
| Cumulative charged time | 6,994.005 s (1.943 h), below the 14,400 s campaign ceiling |
| Retained sampling | Not launched |

Attempt 01 first completed the real `L=3` canary and stopped after 267.798 s.
Its initial gate incorrectly charged the worst-case guards of every possible
viable primary before primary viability was known. This was a localized
resource-harness error, not a target or tuning failure. The plan amendment
corrected the dependency order, preserved Attempt 01, charged its wall time,
and used fresh complete evidence in Attempt 02.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Withdraw `(5, 0.8426345585)` | Point mean `0.75979` exceeded the practical band | No observed target/movement hard veto; native divergence unavailable | Historical interval-overlap rule was wrong relative to admission | Run state-continuing epsilon repair with fresh evidence | No viable-kernel or validation authorization |
| Reject other primaries for this tuning round | Their intervals were wholly above the practical acceptance region | No hard target veto | Longer or repaired per-L epsilon tuning might move them into band | Only revisit under a new tuning-repair plan if validation of L=5 fails or broader candidates are needed | No rejection of PP-UKF or broad-grid tuning |
| Do not promote L=4 or L=6 | Same-epsilon guard intervals were above the practical region | No hard target veto | Retuning them would answer a different question | Keep them as failed local-suitability guards | No claim that independently retuned L=4/L=6 cannot work |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No target, finiteness, status, movement, barrier, identity, GPU, or resource veto fired; native divergence was unavailable |
| Viable candidates | None under the corrected strict admission rule |
| Statistically supported ranking | None; the campaign preserves viability and does not rank stochastic candidates |
| Descriptive-only differences | Epsilon, acceptance means/intervals, and runtime across L values |
| Default readiness | Not established |
| Next evidence needed | Fresh state-continuing epsilon repair, then strict replicated tuning admission before any validation or sampling handoff |

## Evidence Ledgers

| Ledger | Verdict |
| --- | --- |
| Engineering correctness | Passed focused tests, complete barriers, exact target/transport/metric lineage, and terminal artifacts |
| Numerical/sampler validity | One bounded viable pair; no observed target-health veto; native divergence unavailable; convergence not tested |
| Scientific interpretation | The broad per-L procedure found a candidate missed by the old bootstrap/local-anchor attempt; no posterior or model claim follows |

## Run Manifest

- Git commit at execution: `0fff464ab456b72a010007e552c1e2d761624afe`
  with a preserved dirty worktree.
- Python: `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`, Python 3.11.14.
- TensorFlow: 2.19.1.
- Device: NVIDIA GeForce RTX 4080 SUPER, TensorFlow `/device:GPU:0`.
- XLA: enabled. Dtype: float64. TF32 execution setting: enabled and recorded.
- GPU memory mode: verified memory growth; full-device preallocation disabled.
- Target signature:
  `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Frozen transport SHA-256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Root seed: `(20260721, 9300)`; per-pair seeds are content-derived and preserved
  in the private result.
- Plan:
  `docs/plans/bayesfilter-pp-ukf-operational-broad-grid-tuning-plan-2026-07-21.md`.
- Terminal manifest:
  `docs/plans/artifacts/bayesfilter-pp-ukf-operational-broad-grid-20260721/attempt-02/run_manifest.json`.
- Private pair evidence:
  `docs/plans/artifacts/bayesfilter-pp-ukf-operational-broad-grid-20260721/attempt-02/private_result.json`.
- Public summary:
  `docs/plans/artifacts/bayesfilter-pp-ukf-operational-broad-grid-20260721/attempt-02/public_result.json`.

Command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONPYCACHEPREFIX=/tmp/bayesfilter-pp-ukf-grid-pycache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_pp_ukf_operational_broad_grid_20260721.py \
  --output-root \
  docs/plans/artifacts/bayesfilter-pp-ukf-operational-broad-grid-20260721/attempt-02 \
  --frozen-transport \
  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/PP-UKF/final/segments/steps-004001-005000/frozen_transport.json \
  --frozen-transport-sha256 \
  b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221
```

## Post-Run Red Team

The strongest alternative explanation is that the 64-step dual-averaging
budget remained too short for most L values, causing finite but overly small
epsilons and high acceptance. The five `needs_higher_epsilon` primaries support
that as a repair hypothesis but do not prove it. The weakest evidence is the
short, common-start tuning screen and unavailable native divergence field.

A fresh fixed-kernel validation that rejects `(5, 0.8426345585)`, exposes a
target-health failure, or shows unacceptable convergence/movement would
overturn the candidate handoff. Independently retuning `L=4` or `L=6` could
produce other viable candidates, but that would be a new refinement policy and
cannot retroactively make these same-epsilon guards pass.
