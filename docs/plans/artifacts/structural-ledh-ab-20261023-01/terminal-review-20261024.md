# Structural Model A/B terminal review — 24 October 2026

The executable master finished at 10:14 UTC (18:14 Hong Kong time). Both four-chain sampling runs met the declared R-hat threshold. This completes the scheduled execution, while Model B long-horizon posterior approximation accuracy remains unresolved.

| Model | Warmup / retained draws per chain | Maximum R-hat across transformed and physical parameters | Pooled acceptance | Chain acceptance range | Truths in marginal 95% intervals |
|---|---:|---:|---:|---:|---|
| A | 2,000 / 3,000 | 1.003236 | 71.56% | 70.40%–72.47% | True |
| B | 2,000 / 1,000 | 1.003858 | 73.08% | 70.50%–76.80% | True |

The legacy storage key `verification` contains all retained inference draws; there is no additional validation sampling phase. ESS and finite energy errors remain diagnostics, not stopping criteria. Model B chain four accepted 76.8%, above the preferred 65–75% range; pooled acceptance was 73.075%. This is a tuning observation, not evidence against the posterior target.

Model A used exact Kalman endpoint values with LEDH proposal forces; its posterior means agreed with the refined exact posterior reference under the declared tolerance. Model B used a frozen finite-particle LEDH likelihood and its trained NeuTra map. Mixing and truth coverage do not establish agreement with the exact nonlinear posterior.

The repaired Model B map completed 256 accepted updates in 256 attempts. The production precision check passed before retraining: conditional FP64 reset arithmetic is an explicit opt-in, with FP32 particle/filter calculations retained. Both HMC manifests report finite values, valid endpoints, readiness, movement and health checks passing, with zero force fallbacks. These force-fallback counts are distinct from conditional reset-precision fallbacks. The 36 saved warmup/inference tensor artifacts referenced by both manifests exist and match their recorded SHA-256 hashes.

Budget ledger: 11.5215 GPU worker-hours charged of 48 authorized; 36.4785 remain. No campaign worker remains running. This is not a measured GPU-kernel utilization total.

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Scheduled execution complete; retain both results | Every parameter R-hat below 1.01 | Saved manifest health checks pass | Model B finite-design posterior error | Report completed inference and keep reference accuracy as an open question | Exact nonlinear posterior accuracy or a new global default |

## Inference status

| Category | Finding |
|---|---|
| Hard veto screen | No terminal HMC veto in either saved manifest; previous failed attempts remain preserved |
| Statistically supported ranking | None; no method superiority comparison is claimed |
| Descriptive-only differences | Acceptance, ESS, runtime, training loss and point-score comparisons |
| Default readiness | No global numerical-policy or algorithm promotion |
| Next evidence needed | Posterior/reference comparison with reference refinement and Monte Carlo uncertainty for Model B |

Post-run review: the strongest alternative explanation for apparently successful Model B inference is good mixing within a biased finite-LEDH posterior. Independent long-horizon posterior comparison could overturn an accuracy claim, which is therefore not made. This is the weakest remaining evidence, rather than a reason to discard valid draws from the explicitly defined approximate target.

Evidence: `state.json`, `analysis.json`, `attempts/model_a-hmc-01/result.json`, `attempts/model_b-reset-hmc-01/result.json`, `attempts/B-reset-train-01/result.json`, and `attempts/reset-posterior-assessment-01/result.json`. The base experiment and recovery plans remain `docs/plans/structural-ledh-model-ab-master-2026-10-23.md` and `docs/plans/structural-ledh-ab-endpoint-recovery-2026-10-24.md`.
