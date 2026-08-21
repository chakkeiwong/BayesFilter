# NeuTra Full-Validation Result (2026-08-17)

## Provenance Correction

The three-mode lane below evaluated the obsolete `(64,64)`, three-stage,
1,000-update baseline with checkpoint SHA-256 `57b21cc99778b0e24e6c5809ebbb6137709edf8177e7faeeac9d259deb2e7b12`.
It did not evaluate the reviewed `(128,128)`, six-stage capacity repair selected
at update 8,750 with SHA-256 `b39c682030fb3ba8bafe863c747674db40b5d7c13e164c8445ddfab649ad93f6`.
That repaired candidate had already passed tuning, sequential HMC, and exact
three-mode component-law screens. Therefore `FAILED_TUNING_SCREEN` below is a
valid result for the small historical baseline, not the active three-mode
candidate. The active runner now fails closed on this identity distinction.

This result records execution of
`bayesfilter-neutra-full-validation-execution-plan-2026-08-17.md`.

## Decision Table

| Rung | Primary criterion | Veto status | Decision | Next action | Nonclaim |
|---|---|---|---|---|---|
| Harness/contracts | Target/score/transport/HMC contracts pass | None in focused suite | `PASS` | Continue to analytic controls | Does not establish scientific correctness |
| Two-mode Gaussian mixture | Sequential HMC plus retained analytic screens | No sequential hard veto; native divergence not exposed | `VIABLE_ANALYTIC_CONTROL` | Use as a positive-control baseline | One frozen transport/one analytic target only |
| Three-mode Gaussian mixture, obsolete small baseline | Admit a tuned `L>=2` kernel with modern verification | All `L={3,5,10,15,20,25}` candidates failed rank/folded R-hat verification | `FAILED_TUNING_SCREEN_SMALL_BASELINE` | Retain as capacity-ablation evidence; use the reviewed six-stage checkpoint for active validation | No posterior or method-wide conclusion |
| Three-mode Gaussian mixture, reviewed capacity repair | Original plus two fresh seeds passed support, tuned sequential HMC, and exact-law screens | No hard veto in the three component-aware seeds | `VIABLE_REPLICATED_COMPONENT_AWARE_CAPACITY_REPAIR` | Design a target-query-driven mode-discovery proposal; continue separate geometry/application controls | No unknown-mode discovery, universal validity, or cross-target transfer |

## Harness

Focused tests passed: `54 passed, 1 skipped`. The tested contracts cover
batch-native value/score paths, explicit transformed score parity, frozen
transport hash binding, sequential warmup/retained archive behavior, `L=1`
rejection, and Gaussian-mixture diagnostics.

## Two-Mode Control

Artifact root:
`docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/two-mode-full/`.

The run used the frozen checkpoint SHA-256
`af961871dcc3b626216d7500e695534f147ecfd9ba4fe0f9907f59018d40e8e5`, GPU/XLA,
float64, and verified TensorFlow memory growth. Target-specific tuning selected
`L=20` and step size `0.14091138276334744`. Sequential HMC passed with:

- warmup max R-hat `1.04299` under the `1.05` warmup threshold;
- retained max R-hat `1.00551` under the `1.01` threshold;
- retained minimum bulk ESS `6948.04`;
- retained minimum tail ESS `982.80`;
- all finite/status/movement hard vetoes clear;
- retained analytic primary screens passed, including both modes observed per
  chain and the 99% minority-mass screen.

The analytic diagnostics reported descriptive marginal moment misses: one mean
coordinate and one covariance entry did not contain the analytic truth in their
marginal intervals. The runner correctly treats these as explanatory and does
not form a joint rejection. This is a viable analytic control, not proof of
general posterior correctness.

## Three-Mode Control

Artifact root:
`docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/three-mode-full/`.

The run used the frozen checkpoint SHA-256
`57b21cc99778b0e24e6c5809ebbb6137709edf8177e7faeeac9d259deb2e7b12`, GPU/XLA,
float64, and verified memory growth. The canary passed finite target/score/status
checks. Full target-specific tuning then rejected every tested leapfrog value
because `verification_modern_rank_folded_rhat_failed`. No sequential HMC or
retained-sample claim was made.

This rejects the small frozen-transport/tuning hypothesis. It does not weaken
the already-passing six-stage capacity repair, establish a value/score bug, or
reject NeuTra generally. Two fresh component-aware replicas subsequently passed
the complete downstream path. The naive centered Student-t mode-blind proposal
failed support before training. See
`bayesfilter-neutra-three-mode-provenance-and-evidence-closure-result-2026-08-17.md`.

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Obsolete small three-mode baseline tuning veto; no veto for the previously completed six-stage three-mode result; no two-mode sequential hard veto |
| Statistically supported ranking | None; no candidate ranking was attempted |
| Descriptive-only differences | Acceptance, log-accept proxy, moment intervals, runtime |
| Default-readiness | Not established |
| Next evidence needed | Target-query-driven mode-discovery proposal, then separate geometry/application controls |

## Artifact Map

- Two-mode canary: `docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/two-mode-canary/`
- Two-mode full: `docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/two-mode-full/`
- Three-mode canary: `docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/three-mode-canary/`
- Three-mode full: `docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/three-mode-full/`
- Execution plan: `docs/plans/bayesfilter-neutra-full-validation-execution-plan-2026-08-17.md`
