# Moment-Retuned GenUT Whole Leaderboard

All cross-method differences are descriptive unless the row records a paired interval.

| Model | Method | Status | Value | Score |
|---|---|---|---:|---|
| lgssm_T50 | genut | executed_value_score | -136.333493 | `[5.795313209295273, -4.049524396657944, 0.23992155399173498, -1.983576636761427, 5.537582151591778]` |
| lgssm_T50 | sgqf | executed_value_score | -136.075975 | `[5.65544620078834, -3.835056867532365, 0.3023618942730066, -1.9171764199511037, 4.35427564127945]` |
| lgssm_T50 | zhao_cui | executed_value_score | -136.075975 | `[5.65544620078834, -3.835056867532365, 0.3023618942730066, -1.9171764199511037, 4.35427564127945]` |
| ksc_sv_T10 | genut | executed_value_score | -19.9539498 | `[-0.6944250017404556, 0.6076753847301006]` |
| ksc_sv_T10 | sgqf | executed_value_score | -19.9509417 | `[-0.6924748646961102, 0.6095782715450418]` |
| ksc_sv_T10 | zhao_cui | executed_value_score | -19.9562888 | `[-0.7056718029203304, 0.6354886692801854]` |
| exact_sv_T10 | genut | executed_value_score | -19.9944105 | `[-0.6982585825026035, 0.5679645072668791]` |
| exact_sv_T10 | sgqf | executed_value_score | -19.7376715 | `[-0.5324662803728515, 0.7453561843984522]` |
| exact_sv_T10 | zhao_cui | executed_value_score | -19.995663 | `[-0.7072002788826178, 0.5905715364278991]` |
| generalized_sv_T10 | genut | executed_value_score | -16.017544 | `[-0.12287971889600158, -0.15284854173660278, 0.022309970343485475]` |
| generalized_sv_T10 | sgqf | executed_value_score | -16.0194552 | `[-0.12200644916137315, -0.1539074236623366, 0.0222873809519662]` |
| generalized_sv_T10 | zhao_cui | executed_value_score | -16.0198728 | `[-0.12547017508737268, -0.1548427704221113, 0.022260932828017707]` |
| predator_prey_T20 | genut | executed_value_score | -102.739536 | `[-27.7752343416214, 0.07764652464538813, -0.0874873218126595, 1.0422719791531563, 18.367236852645874, -23.650980949401855]` |
| predator_prey_T20 | sgqf | executed_value_score | -102.622704 | `[-27.641142846745264, 0.08410678056716742, -0.0841433194560576, 0.8556990579673749, 17.52559776873574, -22.634978370035743]` |
| predator_prey_T20 | zhao_cui | executed_value_score | -102.419676 | `[-22.67643303781743, 0.1382796511647915, -0.08341679699102532, 0.24588743845934002, 17.605348980194737, -22.815861809468686]` |
| austria_sir_T20 | genut | executed_value_score | -683.363808 | `[-865.9230951070786, 170.8852949142456, 114.98120737075806]` |
| austria_sir_T20 | sgqf | executed_value_score | -682.348006 | `[28.739453057371584, -106.65885657030441, 9.43117639262833]` |
| austria_sir_T20 | zhao_cui | blocked | n/a | `n/a` |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | See `hard_valid` and per-cell status in JSON |
| Statistically supported ranking | None across methods |
| Descriptive-only differences | All cross-method value and score gaps |
| Default readiness | Not established |
| Next evidence | Repair blocked observed-data Zhao-Cui SIR score and run target-specific replication |

JSON: `docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/attempt05_final/result.json`
