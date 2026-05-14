# BayesFilter V1 Model B/C HMC Ladder

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-model-bc-hmc-ladder-2026-05-15.json`.

## Claim Scope

cpu_hmc_readiness_candidate_not_convergence

## Rows

| Target | Classification | Chains | Draws | Acceptance | Max R-hat | Nonfinite |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| model_b_nonlinear_accumulation_tf_svd_cubature | `candidate` | 3 | 16 | 1.000, 1.000, 1.000 | 1.995 | 0 |
| model_b_nonlinear_accumulation_tf_svd_ukf | `candidate` | 3 | 16 | 1.000, 1.000, 1.000 | 1.995 | 0 |
| model_b_nonlinear_accumulation_tf_svd_cut4 | `candidate` | 3 | 16 | 1.000, 1.000, 1.000 | 1.995 | 0 |
| model_c_autonomous_nonlinear_growth_tf_svd_cubature | `candidate` | 3 | 16 | 1.000, 1.000, 1.000 | 2.024 | 0 |
| model_c_autonomous_nonlinear_growth_tf_svd_ukf | `candidate` | 3 | 16 | 1.000, 1.000, 1.000 | 2.024 | 0 |
| model_c_autonomous_nonlinear_growth_tf_svd_cut4 | `candidate` | 3 | 16 | 1.000, 1.000, 1.000 | 2.024 | 0 |

## Interpretation

Candidate means finite branch-gated CPU HMC diagnostics suitable for a future convergence ladder.  It is not a convergence or posterior-recovery claim.
