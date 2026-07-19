# Phase A4 HMC Repair-02 Continuation-01 Result

Date: 2026-07-14 (Asia/Shanghai)

Status: `BLOCKED_CUMULATIVE_MIXING_PROMOTION_VETO_MASS_GEOMETRY_REPAIR_INDICATED`

## Outcome

The authorized exact continuation ran once with no new burn-in, using the
repair-02 exact final state, frozen step `0.37613058552609946`, four leapfrog
steps, trajectory length `1.5045223421043978`, and fresh seed
`[20260714,1640]`. The new trusted GPU/XLA block contributed 250 retained draws
per chain. Cumulative diagnostics were recomputed over the immutable repair-02
segment 0 followed by the fresh block, giving shape `[500,4,4]`.

The cumulative archive remains `NOT_ADMITTED`. All four chains moved, every
sample and required telemetry value was finite, cumulative per-chain acceptance
`[0.48,0.448,0.47,0.39]` passed the `[0.20,0.95]` screen, and no hard veto
fired. However, R-hat, bulk/tail ESS, and MCSE/SD still fail in both coordinate
systems.

The continuation improved R-hat, bulk ESS, MCSE/SD, and initialization-memory
diagnostics, but tail ESS did not improve. At 500 draws per chain the archive
is still materially distant from every mixing threshold. Forecast calibration
was therefore not run.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the cumulative 500-draw archive as A4 calibration input and stop exact continuation | `FAIL`: R-hat, bulk/tail ESS, and MCSE/SD fail in latent and free coordinates | No hard veto; cumulative promotion veto only | Whether persistent separation and very low tail ESS arise from the fixed mass geometry, multimodality, or merely much longer autocorrelation than the current budget-efficient route can tolerate | Plan a fixed-target mass-geometry diagnostic/repair before more HMC draws; preserve all current artifacts and do not rerun seeds | No posterior incorrectness, convergence result, HMC-direction rejection, sampler ranking, predictive equivalence, NeuTra readiness, model adequacy, or default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed: all chains moved; samples, target values, and log-accept telemetry were finite; acceptance passed; hashes and source/kernel/state lineage passed; native divergence was not exposed |
| Promotion screen | Failed for split R-hat, bulk/tail ESS, and MCSE/SD in both coordinate systems |
| Statistically supported ranking | None; this was an exact continuation, not a sampler comparison |
| Descriptive-only differences | Diagnostic changes from 250 to 500 draws, acceptance, chain means, initialization memory, runtime, and telemetry extrema |
| Default-readiness | Not assessed and not supported |
| Next evidence needed | A prospective mass-geometry repair using the same target and original chain identities, followed by fresh retained admission; posterior-reference evidence remains separate |

## Diagnostic Change

| Diagnostic | 250 draws | 500 draws | Threshold | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Latent maximum split R-hat | `1.5172837865185707` | `1.2407799466181304` | `<=1.05` | Improved, still failed |
| Free maximum split R-hat | `1.513142434899695` | `1.2543474296182926` | `<=1.05` | Improved, still failed |
| Latent minimum bulk ESS | `14.057948859944599` | `21.97835059239677` | `>=100` | Improved, still failed |
| Free minimum bulk ESS | `14.116288200251109` | `22.191311358781963` | `>=100` | Improved, still failed |
| Latent minimum tail ESS | `14.172679649246772` | `13.33333333333334` | `>=100` | Did not improve; failed |
| Free minimum tail ESS | `14.172679649246772` | `13.33333333333334` | `>=100` | Did not improve; failed |
| Latent maximum MCSE/SD | `0.3776408311452806` | `0.24023983253081674` | `<=0.10` | Improved, still failed |
| Free maximum MCSE/SD | `0.37765712070103424` | `0.24016992367007667` | `<=0.10` | Improved, still failed |
| Latent maximum initialization memory | `4.424727304400373` | `3.0160667906062386` | Explanatory only | Lower, still substantial |
| Free maximum initialization memory | `4.287388726848259` | `2.9721813363124836` | Explanatory only | Lower, still substantial |

These changes are descriptive. They do not establish monotone convergence or
justify extrapolating the number of draws needed to pass.

## Separate Evidence Ledgers

| Ledger | Status | Evidence |
| --- | --- | --- |
| Engineering correctness | `PASSED` | Focused continuation tests `6/6`; compile and whitespace checks; exact public/private/source/kernel/state/budget replay; fresh no-overwrite namespace |
| Numerical validity | `PASSED_FOR_EMITTED_ARTIFACTS` | New and cumulative samples finite; trusted GPU/XLA placement; exact hashes; no hard veto |
| Sampler admission | `FAILED_PROMOTION_ONLY` | All movement/acceptance/finiteness gates pass; all R-hat/ESS/MCSE families fail |
| Posterior correctness | `NOT_ASSESSED` | No posterior-reference comparison; more finite draws cannot substitute for that separate evidence |
| Forecast calibration | `NOT_RUN` | Cumulative HMC input was not admitted |
| Scientific interpretation | `MASS_GEOMETRY_REPAIR_INDICATED` | One exact continuation improved central mixing metrics but left severe cross-chain and tail inefficiency; repeated unplanned extension is not justified |

## Artifact Evidence

| Artifact | Status | SHA-256 |
| --- | --- | --- |
| Public continuation receipt `repair-02-continuation-01/segment-1.json` | `NOT_ADMITTED` | `29ea0a7461f4c98043977dd02ed8afe8acff552860f40bc1723775ac638d5392` |
| Private continuation manifest | Hash verified | `39fbebba44cc180d7d266b2bafdab4336eceb6031540fdb934222ac59a427b95` |
| New retained shard `[250,4,4]` | Finite, diagnostic/non-admitted | `ded007411545841a172fb6f08ff04c725affb4fc04d026f24cd81ed80983a4fd` |
| New final state | Exact continuation handoff candidate only | `18a19caf2dbd42ccf9e34e68d4db777066b4149426488187a6c9aecf84ded47e` |
| Cumulative serialized sample identity | Recorded in public receipt | `8cd7b2cbd652a3592f9ca1a27a519d04adc33e09028051e5473a52d61be2a0e8` |

The cumulative identity binds the old and new shards in draw order. The new
shard and final state must not be used for calibration or another continuation
without a new prospective plan.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `3d353253dc93a102722e00cbca8803a1b3fce7fa` |
| Worktree | Dirty; unrelated Kalman/QR/Sylvester and runtime lane changes preserved and untouched |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_continuation_01_2026_07_14.py run` |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TensorFlow Probability `0.25.0` |
| Device/JIT | Two RTX 4080 SUPER devices visible; output on `GPU:0`; XLA JIT and TF32 enabled; `float64` target tensors |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Target | Locked A1 semantic SHA-256 `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e` |
| Seed | `[20260714,1640]` |
| Burn-in/new retained | `0/250` per chain |
| Wall time | `202.6476745409891s` |
| HMC call | `197.29824104101863s` |
| Prior charged GPU time | `2040.799946242012s` |
| Total charged GPU time | `2243.447620783001s` = `0.6231798946619448h` |
| Shared cap | `28800s` = `8h` |
| Remaining | `26556.552379217s` = `7.376820105338056h` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-repair-02-continuation-01-plan-2026-07-14.md` |
| Review | `docs/reviews/bayesfilter-ssl-lstm-a4-hmc-repair-02-continuation-01-native-review-2026-07-14.md` |

Unspent GPU budget is not authority for another extension or mass adaptation.

## Failure Classification

| Question | Answer |
| --- | --- |
| Did the continuation implementation fail? | No. Exact handoff, frozen kernel, archive hashes, source bindings, GPU/XLA placement, and cumulative concatenation passed. |
| Did the target or numerical path fail? | No. All required values and samples were finite and no hard veto fired. |
| Did additional draws help? | Descriptively, R-hat, bulk ESS, MCSE/SD, and initialization-memory diagnostics improved. Tail ESS did not. |
| Did cumulative 500 draws pass? | No. Every inferential mixing family remains outside its prospective threshold. |
| Does this reject HMC or predictive-moment validation? | No. It indicates the current fixed mass geometry is not an efficient route to admissible calibration draws at the tested length. |

The existing A1 spectral path again emitted complex-to-`float64` cast warnings
during diagnostics. They remain explanatory: the run completed, target/source
identity was unchanged, and all checked numerical values were finite. They are
not native divergence telemetry or posterior-validity evidence.

## Post-Run Red Team

It remains possible that substantially more exact draws would eventually pass.
The current result cannot disprove that. However, minimum tail ESS failed to
improve when draws doubled, maximum R-hat remains near `1.25`, and effective
information remains roughly one order of magnitude below the threshold. With
the scientific question being calibration-input acquisition rather than proof
of asymptotic convergence, a mass-geometry diagnostic is now more informative
than another unplanned block.

The strongest alternative explanation is multimodality or posterior structure
that no single global mass matrix can repair. A mass adaptation attempt must
therefore remain diagnostic and must preserve chain identities; failure to
reduce cross-chain separation would strengthen the multimodality/target-
geometry explanation rather than trigger seed deletion.

## Stop And Handoff

- Do not run forecast calibration, A5 confirmation, NeuTra training, or
  NeuTra-HMC from the cumulative archive.
- Do not add another continuation block under this plan.
- Do not delete chains, restart from favorable states, change seeds, or relax
  R-hat/ESS/MCSE thresholds.
- The next eligible action is a prospective fixed-target mass-geometry
  diagnostic/repair. It requires explicit authorization and must decide in
  advance whether it tests diagonal or dense/windowed mass adaptation.
