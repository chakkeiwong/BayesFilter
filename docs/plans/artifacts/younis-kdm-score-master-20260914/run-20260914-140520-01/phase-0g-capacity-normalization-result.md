# Normalization, consistency and capacity results

All 56 numerical rows in this slice completed: eight GPU capacity rows and 48 CPU normalization rows. This closes the bounded mechanics allocation; it does not complete the master scientific comparison.

The same finite-program score and its constructed unnormalized derivative remain different from an established unbiased model-score estimator. Reports preserve this distinction explicitly.

## Normalization and consistency

| N | LEDH mean score MSE | Prior SIS mean score MSE | LEDH mean denominator CV | Prior SIS mean denominator CV |
|---:|---:|---:|---:|---:|
| 8 | 0.8356 | 7.4933 | 0.3052 | 0.4717 |
| 16 | 0.5256 | 2.5621 | 0.1688 | 0.3903 |
| 32 | 0.1311 | 1.9822 | 0.0805 | 0.2904 |

The two datasets and four streams per method/count are descriptive only. Both methods had smaller observed score error at higher N in this fixture, but no monotonicity, asymptotic consistency, ranking or unbiasedness is established. The normalizer is scaled by the Kalman likelihood. Reports include its mean and MCSE, the constructed derivative mean, individual-score mean, ratio of means, reciprocal tails and empirical covariance identities. The latter passed exact-array tests, including the Zhat=theta+X ratio-bias counterexample.

For the N-to-2N difference norm versus N-score error norm, pooled descriptive correlation was 0.0578 for LEDH and 0.5521 for prior SIS. These eight rows pool only two datasets and cannot support an inferential correlation or independent calibration. In particular, this fixture does not nominate the LEDH consistency measure as a useful bias predictor. Per-dataset associations are preserved in normalization-consistency-01/consistency.json. A controlled oracle-bearing calibration with fresh evaluation data remains Phase 4B work.

Numerical source is frozen at d5654905 in .localresources/worktrees/younis-score-ratio-20260915. The three normalization-n{8,16,32}-cpu-01 runs each contain 16 complete rows, taking 14.619, 14.368 and 14.452 seconds. CPU FP64/XLA is an explicit reference exception with GPU intentionally hidden. Normalization/report tests: 14 passed; coordinator tests: 13 passed; commit oracle hooks: three passed. The exact commands, seeds, settings, source hashes, environment and per-row data digests are preserved in state.json, initial-manifest.json and row results. Source code is TensorFlow/TFP; aggregation is TensorFlow on CPU.

## Capacity

| State dimension | Total process seconds | Process allocator peak MiB | Rows |
|---:|---:|---:|---:|
| 4 | 53.922 | 160.02 | 4 complete |
| 12 | 54.952 | 160.03 | 4 complete |

Each run used Kalman, shared LEDH, SGQF covariance and persistent-mixture covariance, N=64, T=3, observation dimension two, RTX 4080 SUPER, FP32/TF32/XLA and verified GPU memory growth. Source a99a1c55 is preserved in the nonlinear checkout. Peak allocation is cumulative process high-water memory, not an isolated per-method peak, and the first numerical call includes compilation. Neither throughput nor a ranking follows. One trace per method kernel was observed. Twelve dimensions is a bounded capacity check, not broad high-dimensional readiness.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue implementation integration | All requested consumers and diagnostic reports complete | No failed numerical rows | New main loop still needs integration checks | Compose callbacks with current loop; test main call chains | Whole master completion |
| Keep consistency diagnostic unpromoted | No calibrated association criterion evaluated | Tiny sample and no independent calibration | Bias correlation may change across N/model/regime | Oracle-bearing calibration after scope tuning | Useful bias prediction or valid score control variate |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical rows passed; scientific promotion remains unsupported |
| Statistically supported ranking | None |
| Descriptive-only differences | MSE, CV, correlations and timing |
| Default-readiness | Not established |
| Next evidence needed | Scoped calibration and heldout comparisons; cost and uncertainty protocol |

Post-run red team: decreasing errors may reflect particular data/streams, and common seed labels do not prove nested coupling across different tensor shapes. A larger independent calibration could reverse the correlation conclusion. The weakest evidence is four particle replicates per conditional distribution, which is especially inadequate for reciprocal tails and a Taylor bias approximation. No reported ratio correction is applied to a runtime estimator.

Refresh: phase-0g-integration-plan.md starts the next bounded implementation slice inside the unchanged parent envelope. Eight of twelve GPU launches have been used. Historical run roots are unchanged. Scope-specific tuning, publication-faithful fitted iAPF, consistency calibration, remaining model coverage and final replication remain explicit master tasks.
