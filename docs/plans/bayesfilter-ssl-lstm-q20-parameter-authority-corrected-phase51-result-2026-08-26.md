# Corrected Parameter-Authority Phase 51 Result

Date: `2026-08-26`  
Version: `v3.3-mode-aware-proposal-geometry`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase51-subplan-2026-08-26.md`  
Status: `PASS_V3_3_MODE_AWARE_GEOMETRY_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 51 tested whether a two-mode, full-covariance independent proposal
component reduced the finite between-bank variability left by Phase 50's
isotropic support component. The target remained the batch-native q=20
SSL-LSTM target in `theta_R4`; the 60-dimensional UKF state remained internal
to target evaluation. The q-based annealing bridge, initial clouds, schedule,
particle count, seeds, and eight proposal steps were unchanged.

The candidate was

`s_geom(theta) = 0.5 N(m_minus, kappa^2 C_minus) +
                0.5 N(m_plus, kappa^2 C_plus)`

and `r_geom = (1-rho)q + rho*s_geom`, with hypotheses `rho=0.50` and
`kappa=2.0`. The independent-MH correction used `log r_geom` at both current
and candidate states. No density was assigned to an ETPF/GenUT transform and
no NeuTra/HMC/LEDH route was opened.

## Hard-gate evidence

| Gate | Result | Evidence |
|---|---|---|
| full-covariance algebra fixture | passed | `PASS_V3_3_MODE_AWARE_GEOMETRY_FIXTURE`; SPD matrices, normalized mixture, exact beta-zero/beta-one correction, finite XLA states, movement |
| target and measure | passed | target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d727`; `theta_R4`; retained rows `[256,4]` |
| q/r separation | passed | q used for tempering; `r_geom` used only in the independent-MH ratio |
| candidate validity | passed | no invalid candidate accepted in any active stage |
| paired replay | passed | all Phase 47 initial clouds and identity endpoint hashes reproduced |
| geometry provenance | passed | geometry SHA-256 `dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb` |
| GPU/XLA policy | passed | two RTX 4080 SUPER devices, memory growth before logical-device use, XLA and TF32 enabled |
| report/artifact integrity | passed | unique root, finite JSON/tensor artifacts, report status `PASS_V3_3_MODE_AWARE_GEOMETRY_REPORT` |

The fixture, GPU boundary, and report wall times were `1.3551 s`,
`5558.9085 s`, and `0.3726 s`, respectively.

## Descriptive result

The geometry arm sampled the intended geometry component at approximately
one-half of candidate draws. Mean active-stage acceptance was approximately
`0.311`, `0.311`, and `0.316` across the three banks; no invalid candidate was
accepted. These are implementation diagnostics, not convergence evidence.

| Spread across three banks | Geometry arm | Phase 50 support arm | Geometry <= support? |
|---|---:|---:|---|
| theta mean[0] | `0.7591391865` | `0.8011797849` | yes |
| maximum covariance off-diagonal | `1.5080273143` | `4.2534745351` | yes |
| negative-mode mass | `0.0225140891` | `0.0653321133` | yes |
| retained root count | `5.0` | `11.0` | yes |
| weighted ESS fraction | `0.2664594273` | `0.0441026833` | no |

The predeclared primary branch therefore is
`mode_aware_geometry_reduces_between_bank_variability_descriptive`. The
geometry arm is descriptively favorable on the three primary spread metrics,
but ESS variability moved in the opposite direction. Against Phase 49's
depth-eight arm the result is mixed, not uniformly favorable.

This is a nomination signal for the tested proposal configuration, not a
statistical ranking. Three banks, inherited mode representatives, and
untuned `rho/kappa` hypotheses are insufficient for a default or scientific
claim.

## Decision table

| Decision | Primary criterion | Status | Veto/limitation | Next action | Not concluded |
|---|---|---|---|---|---|
| retain theta target authority | target/status/measure/pairing/replay/provenance | pass | none | retain the R4 target boundary | posterior correctness |
| promote IID Gaussian whitening | finite mutation clouds | veto | finite clouds do not identify a Gaussian law | keep whitening closed | IID Gaussian law |
| promote geometry as default | paired spread screen | defer | three banks, no uncertainty model, ESS spread worsened | run fresh paired replication | superiority/default readiness |
| admit NeuTra HMC or canonical LEDH | downstream density/posterior gates | veto | whitening and posterior gates remain closed | keep those routes closed | HMC/LEDH readiness |

## Inference status

| Evidence class | Status |
|---|---|
| hard veto screen | passed |
| statistically supported ranking | none |
| descriptive-only differences | geometry versus Phase 50 support and Phase 49 depth-eight frozen summaries |
| default readiness | not ready |
| next evidence needed | fresh paired banks with the three arms on identical initial clouds and a predeclared uncertainty diagnostic |

## Classification and limitations

Engineering correctness passed. Numerical validity of the declared finite
kernel passed. Scientific interpretation is a descriptive nomination only.
The strongest alternative explanation is that local curvature improves finite
proposal overlap without estimating global mode mass or connecting geometry.

No new external MathDevMCP invocation was made for Phase 51. The fixture's
direct substituted-form residual check is an implementation check, not a
formal invariance proof. This limitation is preserved rather than upgraded.

## Red-team note

| Item | Statement |
|---|---|
| strongest alternative | local Hessian geometry changes proposal overlap but does not establish target mode mass or global support quality |
| overturning evidence | fresh paired replications fail to reproduce the primary spread pattern, or uncertainty-aware downstream target checks show no effect |
| weakest evidence | three paired banks, inherited representatives, and no uncertainty interval |

## Artifacts and provenance

- Fixture: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase51-mode-aware-proposal-geometry/fixture/`
- Boundary: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase51-mode-aware-proposal-geometry/q20-paired/result.json`
- Report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase51-mode-aware-proposal-geometry/report/result.json`
- Boundary SHA-256: `b6a5e89007cc34bf8d6a7daf9b3320045a488444bf6376bcee1a5ea9dd9bdc45`
- Fixture SHA-256: `0b4eb76e15cbcd45be8763497c70116947ffc7cf0a275f7ceb484a28f844b4f5`
- Report SHA-256: `942e396b1ea4c509c2547df64ed6c38ff7fc41aa48f91f2461de22cf206a2d25`

The machine-readable report is authoritative for exact hashes and metrics.
No mutation rows were used to train NeuTra, select an objective, admit HMC,
or promote a default.
