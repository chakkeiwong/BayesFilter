# GenUT Three-Model Simple Feasibility Result

Status: `diagnostic_feasibility_pass_all_three`.

All differences are descriptive one-seed diagnostics. They do not rank methods.

| Model | Route | Value | Score | Role/status |
|---|---|---:|---|---|
| reduced_continuous_preclip_sir_j1_v1 | GenUT | -11.7710438 | `[-0.005946041084825993, 0.08329159021377563, 5.175644874572754]` | diagnostic_feasibility_pass |
| reduced_continuous_preclip_sir_j1_v1 | dense_manual_score | -10.8591241 | `[-0.004605806472455861, 0.064172346167723, 2.2132342579662017]` | same_target_accuracy_anchor |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | GenUT | -15.9853563 | `[-0.09018072485923767, -0.1266472339630127, 0.01989034377038479]` | diagnostic_feasibility_pass |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | zhao_cui_fixed_branch | -16.019873 | `[-0.12547016510243214, -0.15484276223278626, 0.02226093458781045]` | same_target_diagnostic_not_oracle |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | GenUT | -19.9852524 | `[-0.6998224258422852, 0.8116505146026611]` | diagnostic_feasibility_pass |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | fixed_sgqf | -19.9509416 | `[-0.692474876700316, 0.6095781629434605]` | same_KSC_surrogate_diagnostic_not_exact_SV_oracle |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | principal_sqrt_ukf | -19.9509416 | `[-0.6924748767003159, 0.6095781629434573]` | same_KSC_surrogate_diagnostic_not_exact_SV_oracle |

## Inference status

| Question | Verdict |
|---|---|
| Hard veto screen | `passed for all three short-prefix phases` |
| Statistically supported ranking | None; one seed and one short prefix per model |
| Descriptive-only differences | All GenUT-minus-comparator value and score differences |
| Default readiness | Not evaluated |
| Next evidence needed | Target-specific tuning, particle/seed ladders, uncertainty intervals, then untouched full-horizon runs |

JSON: `/home/chakwong/BayesFilter/docs/benchmarks/artifacts/genut_three_model_simple_feasibility_20260722/attempt01/result.json`
