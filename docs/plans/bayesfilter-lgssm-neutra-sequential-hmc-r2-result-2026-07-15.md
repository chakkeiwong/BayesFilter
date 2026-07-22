# LGSSM NeuTra Sequential HMC Repair Phase R2 Result

Date: 2026-07-15  
Decision: `PASS_NEUTRA_ON_EXACT_FAVORABLE_LGSSM_FIXTURE`

## Outcome

The corrected campaign establishes that one specific frozen NeuTra candidate,
`dense_seed1201`, supports the recorded fixed HMC kernel on this exact favorable
18-dimensional LGSSM fixture. It passed fresh sequential warm-up, full
confirmatory convergence, tuned plain-HMC mean agreement, and truth recovery.

`dense_seed1202` did not pass confirmation. Its second fresh warm-up chunk had
one predeclared energy-error divergence (`log_accept_ratio < -1000`). All states,
target values, log acceptance values, and target-status telemetry were finite,
but the energy-error count of one is a genuine hard veto. No posterior draws
were collected for that candidate and it was not retried.

| Candidate | Warm-up | Retained | Max modern R-hat | Min bulk ESS | Min tail ESS | Max plain-HMC mean difference | Max truth-recovery distance | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dense_seed1201` | 2,000/chain | 4,000/chain | `1.002149` | `4571.61` | `3976.95` | `2.0802` combined MCSE | `1.6290` posterior SD | pass |
| `dense_seed1202` | 2,000/chain | 0 | N/A | N/A | N/A | N/A | N/A | hard veto: one warm-up energy-error divergence |

Seed1201 thresholds were modern R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS
`>=400`, plain-HMC mean disagreement `<=4` combined MCSE, and truth-recovery
distance `<=3` posterior SD. Every parameter passed. Its retained acceptance
rates were approximately `0.680` and `0.681` across the two 2,000-draw chunks;
acceptance is descriptive only and did not carry the decision.

## Artifacts And Manifest

Aggregate result:
`docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/sequential-repair-attempt-01/confirmation-attempt-01/result.json`,
artifact hash
`sha256:ced8f1f952b711bc2af932af771eff9ed68b0da0cf52d55f3ee2d55d66440858`.

Candidate results:

- seed1201 artifact hash
  `sha256:46bad6b8f920ce90946d177b13405298dab995f2c66b32cad421ef3c4ec15ec0`;
- seed1202 artifact hash
  `sha256:596f2b80f7443b984a20fe59944676b1d51b9c7a6a4e22042fb02519ccf0f61c`.

Both results bind Git commit `d269f5bbd8531b878d4f25897a357fbc8f172488`,
Python/TF/TFP environment, `CUDA_VISIBLE_DEVICES=-1`, XLA, float64, commands,
target/adapter/transport/kernel identities, seeds, comparator hash, artifact
paths, and wall times. The active R2 plan is
`docs/plans/bayesfilter-lgssm-neutra-sequential-hmc-r2-subplan-2026-07-15.md`;
the manifest's parent plan field points to the amended master campaign plan.

All warm-up chunks were retained in separate TensorFlow archives and excluded
from posterior summaries. Seed1201 has separate latent/raw archives for 2,000
warm-up and 4,000 retained draws per chain. Seed1202 has separate latent/raw
archives for its two warm-up chunks and no retained archive.

## Decision Table

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Positive exact-fixture result for seed1201 | all full convergence, comparator-agreement, and recovery gates pass | no seed1201 veto; seed1202 independently vetoed by one extreme energy error | performance and reliability across fixtures, training seeds, and models | use seed1201 as a validated exact-fixture case; test broader fixtures under new target-specific plans | no sampler superiority, calibration, robustness, generality, production readiness, or default claim |

## Inference Status

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | seed1201 passes; seed1202 fails one energy-error veto |
| Statistically supported ranking | none; the evidence does not rank candidates or methods |
| Descriptive-only differences | acceptance, runtime, R-hat/ESS magnitudes, and seed-specific outcomes |
| Default readiness | not established |
| Next evidence needed | multiple less favorable fixtures, more training seeds, and uncertainty-aware cross-method comparisons for any robustness or superiority claim |

## Claimed And Computed Quantities

Claimed target: confirmatory HMC in the frozen NeuTra latent coordinates,
transformed to the exact LGSSM raw parameter coordinates and compared with the
same tuned plain-HMC fixture summary.

Actually computed: four batched CPU/XLA HMC chains with the frozen candidate's
transport, step size `0.8`, and 10 leapfrog steps; modern raw-coordinate R-hat,
rank-based bulk/tail ESS, combined-MCSE mean agreement, and posterior-SD truth
recovery. These quantities match the stated exact-fixture gates. They do not
establish exact posterior equality or cross-problem validity.

## Negative Result Classification

Seed1202 is a numerical sampler/energy-error veto for this fresh confirmation
seed and fixed kernel. It is not an implementation, target, artifact, training,
or diagnostic failure, and it is not evidence that the NeuTra direction is
false. The run stopped before posterior sampling as required.

## Post-Run Red Team

The strongest alternative explanation for the positive result is that this
truth-centered synthetic fixture and one favorable training seed make the task
unusually easy. The seed1202 veto shows that reliability across learned
transports is not established. A less favorable fixture, multiple additional
training seeds, or posterior disagreement under a valid fresh run could
overturn any broader interpretation. The weakest evidence is generality, which
was not tested and is explicitly not claimed.

## Budget

R1 used about 18.8 minutes. R2 used about 13.7 minutes for seed1201 and 4.7
minutes for seed1202, plus short validation/finalization commands. The campaign
remained well inside the six-hour CPU ceiling and used no package mutation,
network fetch, paid compute, destructive action, or external publication.

## Final Checks And Review

| Check | Result |
| --- | --- |
| Focused campaign, training, protocol, and screen-finalizer suite | `51 passed` |
| Python compile | pass |
| TensorFlow-only active-route audit | pass |
| `git diff --check` | pass |
| Bounded one-path Claude terminal result review | `VERDICT: AGREE` |

An earlier final-suite command named a nonexistent strict-training test module
and therefore collected no tests. It was a command-list error, not a test
failure; the corrected command used the actual modules and produced the
`51 passed` result above.
