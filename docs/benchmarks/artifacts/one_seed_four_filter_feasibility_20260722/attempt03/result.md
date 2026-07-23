# One-Seed Four-Filter Feasibility

All values are descriptive one-seed diagnostics; no method ranking is supported.

| Model | Method | Status | Value | Score | Dtype | Reason |
|---|---|---|---:|---|---|---|
| ksc_sv | ukf | executed | -19.9509417 | `[-0.6924748646961102, 0.6095782715450404]` | float64 |  |
| ksc_sv | sgqf | executed | -19.9509417 | `[-0.6924748646961102, 0.6095782715450418]` | float64 |  |
| ksc_sv | zhao_cui | executed | -19.9562888 | `[-0.7056718029203304, 0.6354886692801854]` | float64 |  |
| ksc_sv | genut | executed | -19.9733963 | `[-0.6754307746887207, 0.5055976510047913]` | float32 |  |
| exact_transformed_sv | ukf | not_comparable | n/a | `n/a` | n/a | available UKF is an augmented-noise raw-observation Gaussian closure, not this exact transformed target |
| exact_transformed_sv | sgqf | executed | -19.7376715 | `[-0.5324662803728515, 0.7453561843984522]` | float64 |  |
| exact_transformed_sv | zhao_cui | executed | -19.995663 | `[-0.7072002788826178, 0.5905715364278991]` | float64 |  |
| exact_transformed_sv | genut | executed | -20.0184021 | `[-0.6781377792358398, 0.48294341564178467]` | float32 |  |
| generalized_sv | sgqf | executed | -16.0194552 | `[-0.12200644916137315, -0.1539074236623366, 0.0222873809519662]` | float64 |  |
| generalized_sv | zhao_cui | executed | -16.0198728 | `[-0.12547017508737268, -0.1548427704221113, 0.022260932828017707]` | float64 |  |
| generalized_sv | ukf | not_comparable | n/a | `n/a` | n/a | no reviewed same-target generalized-SV UKF route is implemented |
| generalized_sv | genut | executed | -16.0158463 | `[-0.10781475156545639, -0.15403185784816742, 0.021872445940971375]` | float32 |  |
| predator_prey | sgqf | executed | -102.622704 | `[-27.641142846745378, 0.0841067805671752, -0.08414331945605757, 0.8556990579673663, 17.525597768735736, -22.634978370035732]` | float64 |  |
| predator_prey | ukf | not_comparable | n/a | `n/a` | n/a | available UKF uses initial-observation-first timing, while this canonical source row is transition-then-observe |
| predator_prey | zhao_cui | not_implemented_or_not_comparable | n/a | `n/a` | n/a | fixed-variant Zhao-Cui predator-prey source-route evaluator is not implemented; historical retained-grid route is demoted |
| predator_prey | genut | executed | -102.581879 | `[-26.750307083129883, 0.1759563535451889, -0.09248145669698715, 0.5654255747795105, 19.851070404052734, -25.527313232421875]` | float32 |  |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | finite executed cells and explicit unavailable reasons recorded |
| Statistically supported ranking | none; one seed |
| Descriptive differences | all value/score differences only |
| Default/leaderboard readiness | not evaluated |
| Next evidence | target-specific tuning and multi-seed uncertainty on rows with complete coverage |

JSON: `docs/benchmarks/artifacts/one_seed_four_filter_feasibility_20260722/attempt03/result.json`
