# Phase A4 HMC Acquisition Repair-02 Result

Date: 2026-07-14 (Asia/Shanghai)

Status: `BLOCKED_PROMOTION_VETO_MORE_RETAINED_EVIDENCE_REQUIRED`

## Outcome

Fixed-mass dual averaging at target acceptance `0.70` repaired the immediate
tuning problem. The trusted GPU/XLA adaptation screen was `SELECTED`, with all
four chains moving, aggregate acceptance `0.6484375`, per-chain acceptance
`[0.84375,0.65625,0.65625,0.4375]`, finite telemetry, and a scalar step frozen
at `0.37613058552609946` across all 64 post-warmup draws.

The conditionally authorized fresh `250` burn-in plus `250` retained run used
that exact frozen step, four leapfrog steps, trajectory length
`1.5045223421043978`, and the exact adaptation-screen final state. It produced
four finite moving chains with acceptance
`[0.552,0.324,0.32,0.416]`. No hard veto fired.

The archive was not admitted because every inferential mixing family failed:

- maximum rank-normalized split R-hat was `1.5172837865185707` in latent
  coordinates and `1.513142434899695` in free coordinates, above `1.05`;
- minimum bulk ESS was `14.057948859944599` latent and
  `14.116288200251109` free, below `100`;
- minimum tail ESS was `14.172679649246772` in both systems, below `100`; and
- maximum mean MCSE/SD was `0.3776408311452806` latent and
  `0.37765712070103424` free, above `0.10`.

This is a promotion veto caused by insufficient retained mixing/length, not a
continuation veto caused by an invalid target, unmoved chain, nonfinite value,
or malformed artifact. The current authorization covered only the smallest
retained gate, so no extension and no forecast calibration were run.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve repair-02 as a viable frozen-kernel candidate but reject the 250-draw archive as A4 calibration input | `FAIL`: R-hat, ESS, and MCSE/SD promotion criteria fail in both coordinate systems | No hard veto; promotion veto only | Whether exact continuation with more retained draws is sufficient, or whether persistent chain separation indicates remaining mass/geometry limitations | Prospectively authorize an exact continuation from the repair-02 final state with the same frozen kernel and sequential diagnostics; do not rerun adaptation or select another seed | No posterior correctness, convergence proof, sampler superiority, predictive equivalence, NeuTra readiness, model adequacy, or default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed: all four chains moved; all retained samples and telemetry were finite; archive hashes and target/source lineage passed; native divergence was not exposed |
| Promotion screen | Failed for split R-hat, bulk/tail ESS, and MCSE/SD in latent and free coordinates |
| Statistically supported ranking | None; no uncertainty-supported comparison of kernels or samplers was run |
| Descriptive-only differences | Adaptation and retained acceptance rates, frozen step, runtime, R-hat/ESS/MCSE values, initialization-memory diagnostics, and target/log-accept extrema |
| Default-readiness | Not assessed and not supported |
| Next evidence needed | Sequential exact continuation under the same frozen kernel; if chain separation does not materially decline, a separately planned mass-geometry repair |

## Separate Evidence Ledgers

| Ledger | Status | Evidence |
| --- | --- | --- |
| Engineering correctness | `PASSED` | Repair-02 plus repository fixed-mass tests `21/21`; real TFP freeze semantics; compile and whitespace checks; exact state/config/hash replay |
| Numerical validity | `PASSED_FOR_EMITTED_ARTIFACTS` | Finite adaptation and retained tensors/telemetry; GPU/XLA placement; exact private hashes; no archive collision |
| Adaptation | `PASSED_DIAGNOSTIC_SCREEN` | Target `0.70`; aggregate `0.6484375`; every chain in `[0.20,0.95]`; frozen step `0.37613058552609946`; all chains moved |
| Sampler admission | `FAILED_PROMOTION_ONLY` | R-hat, ESS, and MCSE/SD fail at 250 retained draws; movement, acceptance, and finiteness pass |
| Posterior correctness | `NOT_ASSESSED` | No posterior-reference comparison; valid mechanics and mixing diagnostics are prerequisites only |
| Forecast calibration | `NOT_RUN` | Conditional authorization required an admitted retained archive |
| Scientific interpretation | `VIABLE_KERNEL_INSUFFICIENT_EVIDENCE` | Supports exact continuation as the next discriminator; does not establish convergence or reject HMC/moment validation |

## Evidence Summary

| Artifact | Status | Key evidence | SHA-256 |
| --- | --- | --- | --- |
| `repair-02/adaptation.json` | `SELECTED` | All chains moved; aggregate acceptance `0.6484375`; frozen scalar step; finite GPU/XLA telemetry | `97df6a564171deaeb101d20e5d81f93139d3294982519ce3781114bbbfbc2d7d` |
| `repair-02/segment-0.json` | `NOT_ADMITTED` | Movement and acceptance pass; R-hat/ESS/MCSE promotion vetoes | `58e3d9c19ae82450539ce4a16f98e63bb409a630beae0e8a4da5c16703d4c9e3` |

All paths above are relative to
`docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/`.

Private bindings:

| Private artifact | SHA-256 |
| --- | --- |
| Adaptation manifest | `bcfa9253be89f1551ea154477458d34d917f66cdc3719a9a67b60f8b7ca5bea3` |
| Adaptation screen samples | `f2166b1240528d08d61ef545338dd6ff3deb1d0dab1cdb042983052f7f3f1b95` |
| Adaptation final state | `3b2f7cbedf1dbe6508bc5f6b8e2fbb76ba55a6c3e31954d5899bdffd4583fa0c` |
| Adaptation step trace | `1c81eb8684aaf4839a95dd4e6ff370d7e604be30f6980bddbb2fe2c2e24713a2` |
| Retained manifest | `0a48a13852046ad5ae888ec369e0c107678a61f2ff600ba434a6af45a659d673` |
| Retained sample shard `[250,4,4]` | `14255bb3f15897eadccd84a1f295d69d6bbebee74689d0e46c0ecf499e76d43e` |
| Retained final state | `b0df9f30dee43e4b0fe7e545226e5c5a36a4c2a6e387b1d9b7cb01d45d7e38bf` |

The adaptation screen samples are diagnostic-only and must not be pooled into
retained or calibration evidence. The retained shard is a valid non-admitted
segment and may be combined only with an explicitly authorized exact
continuation under the same frozen kernel and diagnostic contract.

## Run Manifest

| Field | Adaptation | Retained segment |
| --- | --- | --- |
| Git commit | `3d353253dc93a102722e00cbca8803a1b3fce7fa` | Same |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TensorFlow Probability `0.25.0` | Same |
| Device/JIT | Two RTX 4080 SUPER devices visible; output on `GPU:0`; XLA JIT and TF32 enabled; `float64` target | Same |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` | Same |
| Seed | `[20260714,1620]` | `[20260714,1630]` |
| Wall time | `210.62788747507147s` | `273.4375842399895s` |
| HMC call | `207.7867655949667s` | `268.0461158310063s` |
| Input state | Original four dispersed starts | Exact adaptation-screen final state |
| Kernel | Fixed-mass dual averaging from step `0.3925`, `L=4`, target `0.70` | Frozen step `0.37613058552609946`, `L=4` |

Shared GPU accounting:

| Field | Value |
| --- | --- |
| Prior charged before repair-02 | `1556.734474526951s` |
| Repair-02 adaptation | `210.62788747507147s` |
| Repair-02 retained rung | `273.4375842399895s` |
| Total charged | `2040.799946242012s` = `0.5668888739561144h` |
| Shared cap | `28800s` = `8h` |
| Remaining | `26759.200053757988s` = `7.433111126043886h` |

Unspent budget is not authority to extend under this completed repair-02 plan.

## Failure Classification

| Question | Answer |
| --- | --- |
| Did adaptation fail? | No. It produced a finite frozen step and passed the prospective repair screen. |
| Did the harness or target fail? | No. Source, geometry, state handoff, hashes, finiteness, GPU/XLA placement, and archive semantics passed. |
| Did chains move and accept proposals? | Yes. Every chain moved; per-chain retained acceptance was `[0.552,0.324,0.32,0.416]`. |
| Did 250 retained draws establish sampler admission? | No. Cross-chain agreement and effective information were far below the frozen criteria. |
| Did HMC or predictive-moment validation fail? | No. This is evidence for a longer exact continuation, not against the research direction. |

The TensorFlow runtime emitted complex-to-`float64` cast warnings from the
existing A1 spectral path during post-run diagnostics. The run did not abort,
all checked samples, target values, and telemetry were finite, and target/source
identity was unchanged. The warnings are explanatory and should remain visible;
they are not silently promoted to a divergence or posterior-validity claim.

## Post-Run Red Team

The strongest alternative explanation is that the four chains still retain
substantial initialization memory after 250 retained draws. Maximum
standardized leave-one-chain-out differences were `4.4247` latent and `4.2874`
free. That can improve with exact continuation, but it may also reveal a mass
matrix or multimodality problem that more draws will not efficiently repair.

The next run should therefore be sequential and discriminating: append a
predeclared retained block from the exact final state, recompute cumulative
diagnostics, and stop if R-hat/ESS/MCSE pass or if a resource cap is reached.
The trend in chain means and initialization-memory diagnostics is explanatory;
it must not replace the frozen admission criteria. If meaningful improvement
does not occur after the first extension, a mass-geometry plan is more
justified than repeatedly extending or retuning seeds.

The weakest evidence is native divergence unavailability. Acceptance and
finite log-accept ratios are not native divergence substitutes. No posterior
reference comparison has occurred.

## Stop And Handoff

- Do not use the current retained shard for forecast calibration, A5,
  confirmation, NeuTra training, or NeuTra-HMC.
- Do not rerun adaptation, change seeds, or change the frozen kernel under this
  result.
- The next eligible experiment is an exact continuation from the repair-02
  retained final state with the same frozen step and leapfrog count, preserving
  the existing 250 draws and recomputing cumulative admission diagnostics.
- That continuation requires a concise prospective plan and explicit owner
  authorization. Forecast calibration remains blocked until cumulative draws
  pass every retained-admission gate.
