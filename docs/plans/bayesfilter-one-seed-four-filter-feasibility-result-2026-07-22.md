# One-Seed Four-Filter Feasibility Result

Date: 2026-07-23  
Plan: `docs/plans/bayesfilter-one-seed-four-filter-feasibility-plan-2026-07-22.md`  
Artifact: `docs/benchmarks/artifacts/one_seed_four_filter_feasibility_20260722/attempt03/`

## Decision

The one-seed feasibility campaign completed on the RTX 4080 SUPER in FP32 GenUT/TF32/XLA with `N=1002`. All 12 executed cells were finite. All four GenUT cells passed the program-valid and residual checks; the largest reported GenUT residual was `1.5259e-5`, below the `5e-4` diagnostic gate. TensorFlow allocator peak was `134,453,504` bytes (about 128.2 MiB) for the process.

This is a route-feasibility result, not a ranking or leaderboard result. The only complete four-way row is the explicitly amended initial-observation-first KSC-SV prefix. The source-order predator-prey row has no admissible fixed-variant Zhao-Cui evaluator and its UKF uses a different time order, so those cells are correctly unavailable.

## Results

Values and scores are at the declared truth parameter point. Scores are in each row's listed parameter coordinates. Differences are descriptive one-seed quantities.

| Model / target | Method | Status | Value | Score |
|---|---|---|---:|---|
| KSC-SV amended initial-observation-first prefix `T=10` | UKF | executed | `-19.9509417466` | `[-0.6924748647, 0.6095782715]` |
| KSC-SV amended initial-observation-first prefix `T=10` | SGQF | executed | `-19.9509417466` | `[-0.6924748647, 0.6095782715]` |
| KSC-SV amended initial-observation-first prefix `T=10` | Zhao-Cui fixed variant | executed | `-19.9562888343` | `[-0.7056718029, 0.6354886693]` |
| KSC-SV amended initial-observation-first prefix `T=10` | GenUT `N=1002` | executed | `-19.9733963013` | `[-0.6754307747, 0.5055976510]` |
| Exact transformed SV amended prefix `T=10` | UKF | not comparable | n/a | n/a |
| Exact transformed SV amended prefix `T=10` | SGQF | executed | `-19.7376714545` | `[-0.5324662804, 0.7453561844]` |
| Exact transformed SV amended prefix `T=10` | Zhao-Cui fixed variant | executed | `-19.9956629857` | `[-0.7072002789, 0.5905715364]` |
| Exact transformed SV amended prefix `T=10` | GenUT `N=1002` | executed | `-20.0184020996` | `[-0.6781377792, 0.4829434156]` |
| Generalized SV source-row prefix `T=10` | UKF | not comparable | n/a | n/a |
| Generalized SV source-row prefix `T=10` | SGQF | executed | `-16.01945524697` | `[-0.1220064492, -0.1539074237, 0.0222873810]` |
| Generalized SV source-row prefix `T=10` | Zhao-Cui fixed variant | executed | `-16.01987281813` | `[-0.1254701751, -0.1548427704, 0.0222609328]` |
| Generalized SV source-row prefix `T=10` | GenUT `N=1002` | executed | `-16.0158462524` | `[-0.1078147516, -0.1540318578, 0.0218724459]` |
| Predator-prey source row `T=20` | UKF | not comparable | n/a | n/a |
| Predator-prey source row `T=20` | SGQF | executed | `-102.6227035213` | `[-27.64114285, 0.08410678, -0.08414332, 0.85569906, 17.52559777, -22.63497837]` |
| Predator-prey source row `T=20` | Zhao-Cui fixed variant | not implemented / not comparable | n/a | n/a |
| Predator-prey source row `T=20` | GenUT `N=1002` | executed | `-102.5818786621` | `[-26.75030708, 0.17595635, -0.09248146, 0.56542557, 19.85107040, -25.52731323]` |

Parameter order is `(z_gamma, log_beta)` for both SV rows, `(z_gamma, log_tau, mu_over_tau)` for generalized SV, and physical `(r,K,a,s,u,v)` for predator-prey.

## Pairwise Differences

For each row, `GenUT - comparator` is purely descriptive.

| Row | Comparator | Value difference | Score difference |
|---|---|---:|---|
| KSC amended prefix | UKF | `-0.0224545547` | `[0.01704409, -0.10398062]` |
| KSC amended prefix | SGQF | `-0.0224545547` | `[0.01704409, -0.10398062]` |
| KSC amended prefix | Zhao-Cui | `-0.0171074669` | `[0.03024103, -0.12989102]` |
| Exact transformed SV amended prefix | SGQF | `-0.2807306452` | `[-0.14567150, -0.26241277]` |
| Exact transformed SV amended prefix | Zhao-Cui | `-0.0227391139` | `[0.02906250, -0.10762812]` |
| Generalized SV source prefix | SGQF | `0.0036089945` | `[0.01419170, -0.00012443, -0.00041494]` |
| Generalized SV source prefix | Zhao-Cui | `0.0040265657` | `[0.01765542, 0.00081091, -0.00038849]` |
| Predator-prey source row | SGQF | `0.0408248592` | `[0.89083576, 0.09184957, -0.00833814, -0.29027348, 2.32547264, -2.89233486]` |

## Route And Target Qualifications

- KSC and exact transformed SV use the repository's explicitly amended initial-observation-first fixture (`y0` is assimilated before the first transition). They are not source-order `x0 -> transition -> y1` leaderboard rows.
- Generalized SV uses the source-row transition-before-every-observation convention and its raw Gaussian SV observation law.
- Predator-prey uses the sealed source-order `x0 -> 20 transitions -> y1:y20` fixture and the fixed SGQF physical-score route.
- The historical predator-prey Zhao-Cui multistate retained-grid route was not run. It is demoted and cannot fill the fixed-variant cell.
- The exact-SV UKF cell is unavailable because the existing UKF is an augmented-noise raw-observation Gaussian closure, not the exact transformed log-chi-square target.
- The generalized-SV UKF cell is unavailable because no reviewed same-target UKF route is implemented.
- All executed comparator scores are manual analytical/fixed-branch recursive routes. GenUT's runtime score is recursive forward sensitivity of the same finite value program. Finite differences are not used at runtime.

## Engineering And Inference Status

| Item | Status |
|---|---|
| GPU/XLA/TF32 execution | pass; logical `/GPU:0`, XLA compilation observed, TF32 enabled |
| TensorFlow memory policy | pass; memory growth configured before logical-device initialization |
| GenUT finite/program-valid gate | pass for all four rows |
| GenUT residual gate | pass; maximum `1.5259e-5` |
| Finite values and scores | pass for all executed cells |
| Statistically supported ranking | none; one seed and mixed approximation targets |
| Default or leaderboard readiness | not evaluated |
| Additional evidence needed | source-order UKF/Zhao-Cui wiring, target-specific tuning, and multi-seed uncertainty on any row intended for ranking |

## Attempt History

- Attempt 1 stopped before the first GenUT call because the harness passed `tf.float32` positionally to `tf.random.stateless_normal`. No numerical cell was produced.
- Attempt 2 passed GPU/XLA compilation and ran the KSC and exact-SV GenUT cells, then stopped because `generalized_sv_sgqf_value_score_status` is module-local rather than re-exported from `bayesfilter.highdim`.
- Attempt 3 fixed both localized harness imports/calls and completed the full matrix. It is the terminal artifact.

## Post-Run Red Team

The strongest alternative explanation is approximation error or precision/target-convention effects, not implementation failure. The residual and finite checks verify the implemented GenUT scalar, not the exact nonlinear likelihood. The KSC UKF/SGQF equality is expected for this scalar component-collapse route and does not establish superiority. No one-seed difference should guide a default change.

## Terminal Nonclaims

- no method ranking or statistically supported superiority;
- no exact nonlinear likelihood or score certification;
- no source-order SV four-way leaderboard row;
- no admissible Zhao-Cui predator-prey comparison;
- no default, HMC, posterior-correctness, or production-readiness claim; and
- no inference from unavailable cells to numerical failure.
