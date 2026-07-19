# SSL-LSTM NeuTra Phase 6 Transformed-HMC Tuning Result

Date: 2026-07-16

Status: `PHASE6_COMPLETE_IDENTITY_MASS_KERNELS_FROZEN_AFTER_H_REPAIR`

## Decision

Phase 6 produced viable source-bound identity-mass tuning kernels for both
immutable trial-0 NeuTra charts:

| Chart | Mass | Step size | Leapfrog steps | Trajectory | Confirmation source |
| --- | --- | ---: | ---: | ---: | --- |
| Fresh G | Identity momentum covariance | `0.8` | `4` | `3.2` | `ladder-r2.json`, fresh seed `(20260716, 6701)` |
| Fresh H | Identity momentum covariance | `0.8` | `4` | `3.2` | `h-confirmation-repair.json`, fresh seed `(20260716, 6901)` |

Decision: `PHASE6_IDENTITY_MASS_KERNELS_FROZEN_AFTER_H_REPAIR`.

This is a sampler-mechanics/tuning handoff only. No tuning sample was retained
as posterior evidence. Phase 6 does not establish convergence, posterior
correctness, support or mode completeness, predictive validity, G/H ranking,
sampler superiority, or default readiness.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Freeze G/H identity-mass tuning kernels for Phase 7 | Both charts passed fresh 64-result, 32-burn-in confirmation with per-chain acceptance, movement, RMS jump, finiteness, and exposed-divergence gates | No final hard or acceptance veto; native divergence unavailable, not zero | Short tuning confirmations do not establish stationarity or global posterior exploration | Plan fresh retained-chain admission with independent seeds and chain-aware diagnostics | Convergence, posterior correctness, predictive validity, superiority, or defaults |

## Evidence Sequence

### Stage A Canary

The authoritative `r2` canary passed on trusted physical GPU 1 / logical GPU 0
in `328.3580 / 2400` seconds. Both G and H used all four reconstructed A0
starts, `epsilon=0.01`, `L=2`, four results, two burn-in steps, and independent
first/warm seeds. All chains moved and all samples, target/proposed-target,
log-acceptance, and correction telemetry were finite. Acceptance `1.0` was
correctly treated as a high-side scale-search trigger, not a pass.

Native TFP divergence booleans were not exposed. Every artifact records
`unavailable_not_zero`; no zero-divergence claim is made.

### Scale And Trajectory Ladder

Both charts had all-high acceptance on `epsilon=0.05,0.10,0.20,0.40` at `L=4`,
so the same prospective high expansion `0.8,1.6` ran for both. `epsilon=0.8`
was viable; `1.6` had low acceptance and insufficient movement. Scale
selection minimized worst per-chain deviation from target acceptance `0.70`.

At `epsilon=0.8`, G's viable trajectory rungs were `L=2,4`; H's were
`L=2,4,8`. Fixed priority selected G `L=4` and H `L=8`. G passed fresh
confirmation. H `L=8` passed all finite, movement, RMS-jump, and divergence
availability gates but had per-chain acceptance
`[0.546875, 0.609375, 0.671875, 0.53125]`; two chains missed the lower `0.55`
screen by one and two accepted decisions. This rejected that selected H kernel
without rejecting identity mass or the research direction.

### H One-Change Repair

The repair used the already prospectively tested viable adjacent H rung:
fixed `epsilon=0.8`, shortened `L=4`, 64 results, 32 burn-in steps, and fresh
seed `(20260716, 6901)`. It was not a new candidate search and changed no
threshold or mass convention.

| H repair check | Result | Gate |
| --- | --- | --- |
| Per-chain acceptance | `[0.609375, 0.625, 0.625, 0.6875]` | Each in `[0.55,0.85]` |
| Per-chain movement | `[0.609375, 0.625, 0.640625, 0.6875]` | Each `>=0.50` |
| Per-chain RMS jump | `[3.7737, 2.5783, 3.1236, 3.2188]` | Each `>=0.05` |
| Core finite telemetry | All true | Samples, accepted/proposed target, log-acceptance and correction |
| Native divergence | `unavailable_not_zero` | Positive exposed divergence would veto |
| Wall/cap | `168.9784 / 600` seconds | Passed resource stop |

## Artifact Failure Separation

The first Stage B process completed HMC calls but failed before writing a
receipt because shared explanatory HMC telemetry uses IEEE `NaN` for unavailable
summaries while the Phase 6 serializer required strict JSON. Its in-memory
outcomes were discarded and are not evidence. This was
`ARTIFACT_SERIALIZATION_FAILURE`, not a target, transport, HMC, geometry, or
candidate result.

The runner was repaired to encode only nonfinite explanatory scalars as explicit
strings (`"NaN"`, `"Infinity"`, `"-Infinity"`) before strict JSON encoding.
Core numerical gates remained numeric and unchanged. The source hash changed,
so a fresh `r2` canary and full fresh `r2` ladder were run; no failed-run
outcome or random stream was reused.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Final G and H confirmations passed all declared hard gates |
| Viable candidates | G identity `epsilon=0.8,L=4`; H identity `epsilon=0.8,L=4` |
| Statistically supported ranking | None; no G/H or candidate superiority ranking is supported |
| Descriptive-only differences | Acceptance, movement, RMS jump, runtime, and short-run G/H differences |
| Default readiness | Not assessed and not supported |
| Next evidence needed | Independent retained chains with R-hat, bulk/tail ESS, MCSE, mapped-coordinate diagnostics, and cross-replication stability |

## Evidence Ledgers

| Ledger | Status | Evidence |
| --- | --- | --- |
| Engineering correctness | `PASSED` | Strict receipt/source/payload binding; four distinct A0 starts; identity-mass fixture; focused and dynamic-runner tests |
| Numerical correctness | `PASSED_FOR_TUNING_SCOPE` | Finite core telemetry and explicit per-chain diagnostics on trusted GPU/XLA |
| Sampler mechanics | `PASSED_TUNING_HANDOFF` | Both frozen kernels passed fresh short confirmation |
| Sampler admission | `NOT_ASSESSED` | No retained chains, R-hat, ESS, or MCSE admission run |
| Posterior correctness | `NOT_ASSESSED` | No posterior oracle exists and no retained/predictive validation ran |
| Scientific interpretation | `TUNING_VIABILITY_ONLY` | Identity-mass NeuTra HMC remains viable for Phase 7 testing |

## Run Manifest

All authoritative runs used git commit
`ffaaaf903354e095da126dbfa47878c34717c5b8`, dirty worktree, conda `tfgpu`,
Python `3.13.13`, TensorFlow `2.20.0`, TensorFlow Probability `0.25.0`, physical
GPU 1 exposed as logical GPU 0, `float64`, TF32 enabled, whole-chain XLA, and
trust basis `owner_designated_managed_session_visible_gpu_trusted`.

| Run | Command role | Wall/cap seconds | Seeds | Output |
| --- | --- | ---: | --- | --- |
| Canary `r2` | Compile/mechanics timing | `328.3580 / 2400` | G `6101,6102`; H `6201,6202` with root `20260716` | `canary-r2.json` |
| Ladder `r2` | Scale, trajectory, G/H confirmation | `794.0802 / 1800` | Prospective `6300-6801` ledger | `ladder-r2.json` |
| H repair | One adjacent-trajectory confirmation | `168.9784 / 600` | `(20260716,6901)` | `h-confirmation-repair.json` |

The invalid first ladder attempt wrote no receipt and has no admissible result.
Its command was the same pre-repair Stage B command with output `ladder.json`;
it remained under the `1800`-second in-run and `2100`-second external caps.

## Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Authoritative canary `r2` | `09c656a304b9dced821b6b85abf586fcecf1495aa3c15e0720d58540af6c39bd` |
| Authoritative ladder `r2` | `6065d862f7dd6aeaea5db57a10f7d4a06be7292a93ffac4e4320e689f7533c51` |
| Final H repair / G-H kernel manifest | `dc340ab2570032a85062d0ec9cd8c9e020c41a133ec9d11b78982502ff08b9b2` |
| Final runner source | `4c9f16a78e25fc6866043207d29fc3da28bb53dcaaf745f369e6d5ecd1322ead` |
| Final focused tests | `636dc7a421da627d437a9db6417bb548db20584b899dcdc0dd0368c7fb520fc0` |

Final focused suite: `14 passed`; earlier serializer-repair suite including two
shared dynamic-runner tests: `15 passed`. `git diff --check` passed. A broader
shared-HMC subset had `24` passes and four unrelated stale exact-key assertions
that do not accept the current proposed-target/correction trace additions; this
lane changed neither shared HMC runtime nor those tests.

## Post-Run Red Team

The strongest alternative explanation is that 64-result confirmation is too
short to expose metastability, resonance, or support omission. Passing the
screen means the kernels are viable for retained admission, not that their
stationary samples are correct. Reverse-KL mode seeking may still make G and H
miss the same posterior region.

Evidence that would overturn this closeout includes source/payload drift,
failure to reproduce the exact transformed target, a retained-chain hard veto,
poor R-hat/ESS/MCSE, material G/H replication instability, or predictive-law
incompatibility. The weakest evidence is the short confirmation length and
unavailable native divergence boolean.
