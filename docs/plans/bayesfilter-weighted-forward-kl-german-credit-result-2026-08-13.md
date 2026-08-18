# German-credit weighted NeuTra result (2026-08-13)

Plan: `docs/plans/bayesfilter-weighted-forward-kl-german-credit-plan-2026-08-13.md`

## Decision

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| German weighted arm stopped before training | The weighted proposal never reached the predeclared support screen; the reverse comparator also failed modern HMC at the historical capacity rung | Both finite proposal attempts failed ESS; valid reverse HMC failed modern R-hat for every fixed-`L` arm | A richer reverse transport or longer target-specific training might improve proposal support, but that is outside this bounded rung | Preserve this target-specific negative result and move to the next reviewed target/baseline | No weighted-vs-reverse ranking, posterior equality claim, default decision, or broad NeuTra conclusion |

## Evidence

- Immutable data/reference copies passed source binding:
  - data SHA-256 `2752b044394958ab6dd193a0b56ca0f0b3a2d8bc7cb8c008e35a5e84bbec02f8`;
  - reference SHA-256 `605fbca76b076bb23cf865f7210ef8e6da2b29c1c87964d13463126e71faeb09`.
- The TensorFlow target passed source preprocessing, closed-form value,
  autodiff score, batch shape, and GPU/XLA target-canary checks.
- Reverse training used dense IAF `(51,51)`, three stages, batch 1,024,
  float64 GPU/XLA. The 1,000-update rung selected update 1000 by disjoint
  heldout reverse-KL: selection/audit `553.10274 / 553.56417`; clipping was
  `898/1000`, so loss is descriptive only.

## Proposal support

| Proposal | Global ESS / 65,536 | Global ESS fraction | Median 4,096-row ESS fraction | Max global weight | Status |
|---|---:|---:|---:|---:|---|
| Reverse pushforward scale mixture | `1.17` | `1.79e-5` | `8.39e-4` | `0.9231` | Rejected |
| Reference-marginal + reverse repair | `7.10` | `1.08e-4` | `4.76e-4` | `0.2432` | Rejected |

The repair improved global ESS about `6.06x`, but remained far below the
predeclared global and median-batch ESS fraction floor `0.0625`. All target,
proposal, row, hash, and batch-native diagnostics were finite. This is a valid
proposal-support failure, not a target or artifact failure. Weighted replay and
training were therefore not launched.

## Reverse HMC comparator

The first `reverse-hmc-r1` artifact is launch-invalid: a shared weighted-IAF
ELU pullback mixed float32 Python literals with float64 state. The repair uses
dtype-preserving `tf.ones_like`/float64 constants, passes focused tests, and a
finite one-arm XLA check. The valid `reverse-hmc-r2` retry rejected every arm
under modern rank/folded R-hat. Representative maxima:

| `L` | Acceptance | Max rank-normalized R-hat | Max folded R-hat |
|---:|---:|---:|---:|
| 3 | `0.428` | `2.357` | `1.628` |
| 5 | `0.512` | `1.864` | `1.850` |
| 10 | `0.706` | `1.156` | `1.129` |
| 15 | `0.766` | `1.093` | `1.060` |
| 20 | `0.682` | `1.077` | `1.055` |
| 25 | `0.844` | `1.043` | `1.031` |
| 32 | `0.807` | `1.125` | `1.114` |

The current threshold is `1.01`; native divergence is unavailable and is not
claimed to be zero. This is valid reverse-comparator negative evidence after
the shared dtype defect was repaired.

## Failure classification

| Ledger | Verdict | Evidence |
|---|---|---|
| Engineering correctness | Target route passes; one shared HMC dtype defect was repaired | Target tests, GPU/XLA canary, focused weighted/reverse/HMC tests, finite one-arm retry |
| Numerical/sampler validity | Reverse transport fails current modern HMC at the declared rung | All valid `r2` arms finite but R-hat exceeds `1.01` |
| Weighted scientific mechanism | Not tested downstream because both full-support proposal screens failed | ESS failure stops this target; it does not reject weighted forward-KL generally |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Proposal support veto for both attempts; reverse HMC R-hat veto for all valid arms |
| Statistically supported ranking | None |
| Descriptive-only diagnostics | Reverse loss/clipping, proposal ESS, acceptance, R-hat, runtime, and repair behavior |
| Default readiness | Not assessed and not promoted |
| Next evidence needed | A separately reviewed richer German proposal/training plan, or the next master-plan target |

## Post-run red team

The strongest alternative explanation is undertrained reverse geometry: the
historical 1,000-update arm clipped heavily and failed the modern screen. A
richer transport or longer target-specific optimizer protocol could improve
both HMC and proposal support. That is a new experiment, not evidence from this
bounded campaign.

Artifacts:

- Target source/canary: `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/german-credit-source/`.
- Reverse training: `german-credit/reverse-canary-r1/`, `reverse-serious-r1/`.
- Proposals: `german-credit/proposal-r1/`, `proposal-r2-reference-augmented/`.
- Launch-invalid HMC: `german-credit/reverse-hmc-r1/`.
- Valid repaired HMC: `german-credit/reverse-hmc-r2/`.

The valid retry's `artifact_hashes.json` covers `result.json`,
`run_manifest.json`, and the nested `tuning/tuning_result.json`; the runner
now emits this manifest on both pass and tuning-rejection paths.
