# Compact covariance implementation comparison, 2026-10-13

All arms use the same finite filter, observations, fixed design, controls, N4032 and K2016. The change computes each distinct proposal covariance before broadcasting. This comparison measures implementation parity and cost, not model-oracle accuracy or posterior convergence.

| Model | Status | Scaled value gap | Scaled score gap | Baseline B2 seconds | Candidate B2 seconds | Candidate/baseline |
|---|---|---:|---:|---:|---:|---:|
| lgssm | COVARIANCE_IMPLEMENTATION_PARITY_PASS_COST_MEASURED | 0 | 0 | 64.1318 | 63.7385 | 0.993867 |
| ksc | COVARIANCE_IMPLEMENTATION_PARITY_PASS_COST_MEASURED | 0 | 0 | 18.3842 | 18.3441 | 0.997819 |
| predator_prey | COVARIANCE_IMPLEMENTATION_PARITY_PASS_COST_MEASURED | 0 | 0 | 37.4079 | 38.5419 | 1.03031 |
| sir_d18 | COVARIANCE_IMPLEMENTATION_PARITY_PASS_COST_MEASURED | 0 | 0 | 93.5581 | 77.1453 | 0.824571 |
| range_bearing | COVARIANCE_IMPLEMENTATION_PARITY_PASS_COST_MEASURED | 0 | 0 | 16.7062 | 16.6952 | 0.999343 |

Each completed timing uses three warmed calls per implementation with alternating order. Observed timing differences are descriptive only. SIR and range use the explicitly selected mixed-arithmetic diagnostic scope; FP32 remains the campaign default. Controls are comparison inputs; old source-bound tuning artifacts cannot authorize HMC on the changed source.

| Decision | Primary criterion | Vetoes | Uncertainty | Next action | What is not concluded |
|---|---|---|---|---|---|
| Retain compact implementation | All five scopes pass with exact value/score/particle/tangent parity | No newly invalid rows observed | Three timing pairs only; other workload on GPU1; untested posterior region | Localize exact range force failure, then fresh-scope tuning and mechanics | Full repair, statistically supported speed superiority or posterior correctness |

| Inference status | Finding |
|---|---|
| Hard veto screen | All five paired full-horizon comparisons pass |
| Statistically supported ranking | None |
| Descriptive differences | Paired warmed timings above |
| Default readiness | No default-policy change or HMC admission from parity |
| Next evidence needed | Resolve start-region health, refresh own-scope tuning, and then corrected-HMC diagnostics |

Budget consumed or currently reserved: 29985.649/36000 seconds; 56/60 attempts.

Sources:
- [lgssm](docs/plans/artifacts/surrogate-hmc-compact-lgssm-20261013-01/lgssm/covariance-comparison.json)
- [ksc](docs/plans/artifacts/surrogate-hmc-compact-ksc-20261013-01/ksc/covariance-comparison.json)
- [predator_prey](docs/plans/artifacts/surrogate-hmc-compact-predator-20261013-01/predator_prey/covariance-comparison.json)
- [sir_d18](docs/plans/artifacts/surrogate-hmc-compact-sir-20261013-01/sir_d18/covariance-comparison.json)
- [range_bearing](docs/plans/artifacts/surrogate-hmc-compact-range-20261013-01/range_bearing/covariance-comparison.json)

SIR mean warm time decreased from93.56 to77.15 seconds (17.5%) in these three repetitions; this is descriptive. The other four scopes show no clear measured gain. First-call timings start after explicit TensorFlow tracing and include XLA compilation plus execution; total worker wall time includes tracing. Range nearby-start comparisons used FP32-rounded offsets/beta, so they do not replay the failed FP64 pilot exactly. The separate force-health plan addresses that distinction.

Post-run red-team: the compiler may already eliminate duplicated work, so source simplification is not evidence of speed. Identical outputs at nominal points do not establish healthy forces over the HMC initialization region. Mixed FP64 recursion is not an exact model-score oracle.
