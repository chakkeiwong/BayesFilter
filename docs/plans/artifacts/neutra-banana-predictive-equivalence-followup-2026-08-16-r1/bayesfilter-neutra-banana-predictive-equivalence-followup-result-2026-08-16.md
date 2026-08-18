# NeuTra Banana Predictive-Equivalence Follow-up Result (2026-08-16)

## Outcome

The unchanged seed-15, 6,000-update learned transport with frozen identity z
mass, `L=10`, and step size `0.7709722545680272` completed the larger-window
predictive-law diagnostic in `178.01 s` on GPU 0. No retraining, retuning,
state regeneration, or kernel change occurred.

The larger diagnostic reduced the candidate point MMD by about a factor of four
relative to the 1,024-draw diagnostic, and the coordinate/banana moments are
now close to their exact values. However, the candidate's 99% moving-block
bootstrap upper interval remained above the exact-vs-exact q99 calibration
envelope at both tested archive windows and all three block lengths. The excess
is small: 4.7% to 13.8% relative to the empirical q99 envelope. The candidate
also exceeded the empirical q95 envelope in every cell.

This is persistent evidence of a small discrepancy under this finite screen,
not a formal equality-test rejection. The two candidate windows overlap and
are therefore sensitivity windows, not independent replications.

## Evidence contract

| Item | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-neutra-banana-predictive-equivalence-followup-plan-2026-08-16.md` |
| Artifact root | `docs/plans/artifacts/neutra-banana-predictive-equivalence-followup-2026-08-16-r1/` |
| Candidate | Existing confirmation archive; seed `15`, 6,000 updates, identity z mass, `L=10`, step `0.7709722545680272` |
| Draw window | 4,096 draws per chain from the 5,000-draw archive |
| Window offsets | `0` and `904`; overlapping, not independent |
| Statistic | Biased RBF MMD with bandwidths `(2,4,8)` |
| Dependence treatment | Four fixed chains; moving block lengths `32,64,128`; 512 bootstrap replicates |
| Calibration | 128 independent exact-vs-exact banks per offset, same geometry |
| Runtime | `178.01496 s` |
| Provenance | GPU 0, verified memory growth, float64, XLA, TF32 disabled |
| Integrity | All six listed artifact hashes passed |

## Primary results

The calibration values are empirical quantiles of the 128 exact-vs-exact upper
intervals. MMD values are squared discrepancies, not p-values.

| Offset | Block | Candidate point | Candidate upper 99% | Control q95 | Control q99 | Upper/q99 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 0.00010362 | 0.00026096 | 0.00022071 | 0.00024925 | 1.047 |
| 0 | 64 | 0.00010362 | 0.00026149 | 0.00022510 | 0.00023528 | 1.111 |
| 0 | 128 | 0.00010362 | 0.00026855 | 0.00022364 | 0.00023591 | 1.138 |
| 904 | 32 | 0.00009426 | 0.00025040 | 0.00021930 | 0.00023302 | 1.075 |
| 904 | 64 | 0.00009426 | 0.00025448 | 0.00021519 | 0.00023512 | 1.082 |
| 904 | 128 | 0.00009426 | 0.00025887 | 0.00021678 | 0.00022954 | 1.128 |

The prior 1,024-draw point values were approximately `0.0004006` and
`0.0004127`; the 4,096-draw values are `0.0001036` and `0.0000943`. This
decrease is descriptive and does not by itself prove a sampling-rate law.

## Explanatory results

For offset 0, the first four candidate coordinate means were
`(0.00164,-0.00093,-0.01799,-0.01412)` and variances
`(1.0084,1.2473,1.0282,1.0089)`. For offset 904 they were
`(0.00067,0.00657,-0.01270,-0.00693)` and
`(1.0140,1.2451,1.0326,1.0012)`. The analytic second-coordinate variance is
`1 + 2(0.35)^2 = 1.245`; the observed values are close. These summaries are
explanatory only and do not override the MMD screen.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Larger-window predictive screen | Candidate upper interval below control q99 at all windows/blocks | **Failed descriptively** in all 6 cells; no infrastructure veto | Candidate windows overlap; q99 is estimated from 128 controls and 512 bootstrap replicates | Use a feature/tail decomposition to locate the residual discrepancy before any training change | Formal equality rejection or universal HMC failure |
| Sampling-size sensitivity | Point discrepancy decreases strongly from 1,024 to 4,096 draws | **Passed as a descriptive trend** | Same candidate archive and overlapping windows; no independent long chain | Preserve as evidence that the prior excess was partly finite-sample | A guaranteed `1/N` scaling law |
| Harness validity | Finite exact controls, source/candidate hashes, GPU/XLA provenance | **Passed** | No formal calibration theorem for the custom envelope | Keep the harness and inspect the residual feature | Scientific correctness in all metrics |
| HMC candidate status | Prior HMC convergence and exact-law moment gates remain valid | **Unchanged** | Limited exact-law summaries and unavailable native divergence telemetry | Do not retune or retrain yet; diagnose where the MMD excess comes from | Default/production readiness |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | No numerical, artifact, or provenance veto; predictive screen remains above the empirical q99 envelope |
| Statistically supported ranking | None; no competing candidate was evaluated |
| Descriptive-only differences | MMD point/interval, q95/q99 envelope ratios, moments, block sensitivity, and runtime |
| Default-readiness | Not supported |
| Next evidence needed | Decompose the MMD by latent/model coordinates and nonlinear tail features using the same frozen archive and exact controls |

## Interpretation and red-team note

The strongest alternative explanation is residual finite-sample calibration
error: the candidate point fell sharply with four times as many draws, and the
two large windows overlap. A second explanation is block-length misspecification
for the HMC dependence. A third is a genuine small learned-transport law error
concentrated in a nonlinear or tail feature that the coordinate screens do not
resolve. The current result cannot choose among these explanations.

The correct next action is a diagnostic decomposition, not a change to the
training objective, architecture, learning rate, or HMC kernel. If the residual
comes from one identifiable tail/nonlinear feature, repair can be targeted. If
it disappears under a correctly calibrated feature-wise control, classify the
MMD excess as underpowered finite-sample evidence.

## Nonclaims

- No formal equality p-value was computed.
- The two candidate windows are overlapping sensitivity windows, not independent replications.
- No posterior correctness proof, universal kernel claim, superiority claim, SSL-LSTM transfer, or production/default readiness is supported.
