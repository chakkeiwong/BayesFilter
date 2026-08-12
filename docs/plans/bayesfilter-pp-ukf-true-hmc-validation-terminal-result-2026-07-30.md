# PP-UKF true-HMC validation terminal result

Date: 2026-07-30

Status: `COMPLETED_UNRANKED_VIABLE_SET`

Plan: `docs/plans/bayesfilter-pp-ukf-true-hmc-continuation-repair-plan-2026-07-30.md`

Terminal artifact:
`docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11/public_result.json`

Run manifest:
`docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11/run_manifest.json`

## Result

Attempt 11 completed the three continuations selected after attempt 09:
`L=(9,12,17)`. All three passed the frozen sequential HMC screen. The merged
artifact preserves the seven candidates admitted in attempt 09, so all ten
planned candidates are viable under this screen. No hard veto fired and no
ranking was performed.

The campaign used `51,376.631626442 s` of the authorized `86,400 s`: the
attempt-11 wall time was `8,629.763232442 s`, added to the conservative
`42,746.868394 s` carry-in. Warmup draws remain excluded from posterior
summaries.

## Candidate decision table

| L | Epsilon | Retained draws per chain | Max R-hat | Min bulk ESS | Min tail ESS | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 0.845201109 | 1,000 | 1.005913 | 3,306.2 | 1,812.8 | Passed screen; viable |
| 9 | 0.983032751 | 10,000 | 1.006980 | 18,487.0 | 2,881.5 | Passed screen after continuation; viable |
| 12 | 0.790480953 | 6,500 | 1.007653 | 25,073.4 | 5,477.1 | Passed screen after continuation; viable |
| 13 | 0.790480953 | 1,000 | 1.005027 | 1,922.7 | 1,304.5 | Passed screen; viable |
| 14 | 0.790480953 | 2,000 | 1.005784 | 1,358.7 | 2,433.6 | Passed screen; viable |
| 17 | 0.887851484 | 4,500 | 1.007966 | 7,894.7 | 2,794.1 | Passed screen after continuation; viable |
| 18 | 0.887851484 | 1,000 | 1.008701 | 1,708.1 | 915.8 | Passed screen; viable |
| 19 | 0.887851484 | 2,000 | 1.004593 | 1,242.5 | 1,935.3 | Passed screen; viable |
| 24 | 0.836099880 | 1,500 | 1.006858 | 1,462.8 | 1,107.6 | Passed screen; viable |
| 25 | 0.836099880 | 2,000 | 1.007256 | 2,664.5 | 1,389.6 | Passed screen; viable |

The frozen thresholds were maximum all-parameter rank-normalized/folded split
R-hat `<=1.01`, minimum bulk ESS `>=1000`, and minimum tail ESS `>=400`.
Each row also passed finite-state, finite-target, finite-log-acceptance,
all-chain-movement, and target-status checks.

## Decision

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain all ten as an unranked viable HMC-kernel set | All ten passed the frozen R-hat, bulk-ESS, tail-ESS, and health screen | No declared hard veto fired | Candidate draw counts differ; no uncertainty-aware comparison or posterior-reference check was performed; native divergence is unavailable | Resolve the complex-cast warning, then predeclare and run a downstream posterior-correctness or uncertainty-aware comparison if a single candidate must be selected | No best candidate, superiority, posterior correctness, default-readiness, production-readiness, or broad PP-UKF validity claim |

## Inference status

| Evidence class | Status | Interpretation |
| --- | --- | --- |
| Hard veto screen | Passed for all ten candidates | No nonfinite state/target/log acceptance, invalid telemetry, no-movement condition, or convergence-diagnostic veto was observed |
| Statistically supported ranking | None | The campaign intentionally did not rank candidates and has no predeclared paired uncertainty analysis |
| Descriptive-only differences | R-hat, ESS, acceptance, runtime, draw count, and extreme-log-acceptance counts | These explain behavior but do not establish that one viable candidate is better |
| Default readiness | Not established | Passing the HMC-kernel screen does not establish posterior correctness or a default kernel |
| Next evidence needed | Complex-cast localization plus downstream posterior/reference validation or a predeclared uncertainty-aware selection experiment | A new decision experiment is required before selecting one candidate as best or default |

## Artifact verification

The terminal inspection established:

- exactly ten rows, ten unique candidate identifiers, and ten unique values of
  `L`;
- exact candidate-row equality between `public_result.json` and the terminal
  `progress.json`;
- all 332 unique latent/raw archive files referenced by the ten rows exist and
  match their recorded SHA-256 values; and
- no attempt-11 PP-UKF process remains active.

Compact artifact hashes are preserved in
`docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11/artifact_hashes.json`.
Raw `.tftensor` files, mutable progress, and the launch log remain ignored by
the repository hygiene policy. The public result, run manifest, hash receipt,
and this terminal note are claim-supporting evidence and remain trackable.

## Residual numerical risk

TFP `HamiltonianMonteCarlo` did not expose a native divergence indicator. The
recorded value `not_exposed_by_tfp_hamiltonian_monte_carlo` is not evidence of
zero divergences. Finite log-acceptance values below `-1000` remain explanatory
extreme proposals, not an energy veto.

The original launch log emitted 72 warnings that a `complex128` value was cast
to `float64`, discarding its imaginary component. Source tracing showed that
all 72 came from TFP's FFT-backed ESS diagnostic: 24 retained diagnostic
evaluations times three ESS calls each. The PP-UKF target and HMC transition
are real-valued.

This warning was repaired after the run in
`docs/plans/bayesfilter-real-fft-hmc-ess-cast-warning-repair-plan-2026-07-30.md`.
The replacement uses TensorFlow `rfft`/`irfft` while matching TFP 0.25's ESS
formula. On the final `L=9` attempt-11 model-coordinate tensor, the maximum
direct ESS difference from TFP was `3.64e-11`; the diagnostic reproduced the
same pass metrics and emitted zero complex-cast warnings. The historical
attempt-11 artifact is unchanged, and no rerun is required for this
post-sampling diagnostic repair.

## Post-run red team

The strongest alternative explanation is that the screen admits chains that
mix internally but target a numerically altered density because the complex
cast discards a non-negligible imaginary component. A second alternative is
that unequal stopping times make the descriptive ESS and runtime values
unsuitable for candidate ranking. A source-localized derivation showing that
the imaginary component is analytically zero up to floating-point error, or a
corrected real-valued implementation with posterior/reference agreement,
would address the first concern. A predeclared equal-budget or cost-normalized
comparison with uncertainty would be required to support a ranking.
