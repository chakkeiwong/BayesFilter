# Q3 Canonical LEDH Leaderboard (2026-08-25)

Program: Q3 of the completion program; plan: `bayesfilter-q3-leaderboard-execution-plan-2026-08-24.md`. All rows G-5 stamped; seeding uses the fidelity-#7-fixed independent replication streams. N=1008 particles per arm; 16 value seeds / 8 score seeds unless a row's manifest says otherwise.

## Configuration status (READ BEFORE ANY NUMBER)

A cell whose program is not `production` or whose tuning is UNTUNED must not be debugged or interpreted as the production algorithm's performance. Per the per-scope tuning rule, UNTUNED cells carry no per-model claims.

| row | cell | program | tuning |
|---|---|---|---|
| linear2d | canonical_ledh | production v1 (registry: contract_e reset + dual-cap trust region ON) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| linear2d | bootstrap_pf | comparator (bootstrap PF, systematic resampling) | N/A (no tunables beyond N) |
| dlgssm | canonical_ledh | production v1 (registry: contract_e reset + dual-cap trust region ON) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| dlgssm | bootstrap_pf | comparator (bootstrap PF, systematic resampling) | N/A (no tunables beyond N) |
| dlgssm | score_dir0 | production v1 (registry: S6 contract_e reset + S7 dual-cap trust region) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| predator_prey | canonical_ledh | production v1 (registry: contract_e reset + dual-cap trust region ON) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| predator_prey | bootstrap_pf | comparator (bootstrap PF, systematic resampling) | N/A (no tunables beyond N) |
| predator_prey | score_dir0 | production v1 (registry: S6 contract_e reset + S7 dual-cap trust region) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| ksc_sv | canonical_ledh | production v1 (registry: contract_e reset + dual-cap trust region ON) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| ksc_sv | bootstrap_pf | comparator (bootstrap PF, systematic resampling) | N/A (no tunables beyond N) |
| ksc_sv | score_dir0 | production v1 (registry: S6 contract_e reset + S7 dual-cap trust region) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| generalized_sv | canonical_ledh | production v1 (registry: contract_e reset + dual-cap trust region ON) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| generalized_sv | bootstrap_pf | comparator (bootstrap PF, systematic resampling) | N/A (no tunables beyond N) |
| generalized_sv | score_dir4 | production v1 (registry: S6 contract_e reset + S7 dual-cap trust region) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| austria_sir | canonical_ledh | production v1 (registry: contract_e reset + dual-cap trust region ON) | UNTUNED (no per-scope tuning artifact; registry warm-start values: trust radius 0.5 warm start per Q2 Curve-2, epsilon 2.0 warm start per gap A2) |
| austria_sir | bootstrap_pf | comparator (bootstrap PF, systematic resampling) | N/A (no tunables beyond N) |
| austria_sir | score_dir0 | production v1 (registry + annealed k=4 per Q2 calibration) | Q2 Curve-1 artifact (k=4/c=8) + Curve-3 damping; epsilon/substeps/reset controls untuned |

## Hard-veto screen

- linear2d: clean
- dlgssm: clean
- predator_prey: clean
- ksc_sv: clean
- generalized_sv: clean
- austria_sir: clean

## Value cells

| row | data | exact ref | canonical mean (spread) | bootstrap mean (spread) | UKF-GF |
|---|---|---|---|---|---|
| linear2d | test-fixture linear 2d, T=5 | -14.390 | -14.428 (0.192), |err| 0.038 | -14.407 (0.101), |err| 0.017 | -14.390, |err| 0.0000 |
| dlgssm | frozen benchmark_lgssm_m3_T50_seed81100 | -173.123 | -173.993 (0.127), |err| 0.870 | -173.125 (0.193), |err| 0.002 | -173.123, |err| 0.0000 |
| predator_prey | frozen predator_prey_T20 | — | -656.066 (0.577) | -886.583 (39.454) | -645.158 |
| ksc_sv | frozen zhao_cui_sv_ksc_T1000 (full mixture horizon) | — | -2439.335 (1.224) | -2439.069 (0.881) | -2491.603 |
| generalized_sv | simulated gen-SV T=20 (seed 501; heteroskedastic law) | — | -34.927 (0.112) | -35.063 (0.126) | -34.605 |
| austria_sir | frozen austria_sir_y1_y20; Q2-calibrated annealed k=4/c=8 canonical arm (per-scope calibration applies to this row only) | — | -681.756 (0.480) | -682.428 (0.612) | -681.686 |

## Score cells (canonical analytical)

| row | cell | dir | mode | mean (seed spread) | self-consistency rel err (seed 0, kind) | exact ref |
|---|---|---|---|---|---|---|
| dlgssm | score_dir0 | 0 | plain | -18.045 (0.283) | 2.99e-11 (central_fd_seed0) | -15.6731 (|err of mean| 2.3722) |
| predator_prey | score_dir0 | 0 | plain | -99.915 (9.151) | 2.55e-06 (central_fd_seed0) | — |
| ksc_sv | score_dir0 | 0 | plain | 237.790 (0.964) | 6.56e-10 (central_fd_seed0) | — |
| generalized_sv | score_dir4 | 4 | plain | 2.209 (0.227) | 1.13e-10 (central_fd_seed0) | — |
| austria_sir | score_dir0 | 0 | annealed k=4 | -411.128 (201.198) | 1.26e-09 (oracle_seed0) | — |

Score-cell reading guide: self-consistency columns compare the analytical score against a reference derivative OF THE SAME estimator (central FD for plain cells; the autodiff oracle for annealed cells, since FD is invalid across resampling-boundary crossings) — machine-precision values prove the derivation, not unbiasedness. The exact-reference column shows that the particle score MEAN deviates from the exact score at claim scale: on dlgssm T=50 both modes deviate by ~4 (~30x seed-SE) with OPPOSITE signs (plain -11.5, exact -15.7, annealed -20.1). Finite-N score bias is estimator-variant-dependent; the Austria Fisher result (bias shrinking under annealing) does NOT transfer as a general rule, and score cells must not be read as unbiased score estimates. This is the board's standing caution for Q5.

## Inference-status table

| question | status |
|---|---|
| hard-veto screen | see screen above; a vetoed row's cells are not interpreted |
| statistically supported ranking | NONE claimed — no predeclared uncertainty analysis ranks arms; exact-reference absolute errors are the only absolute claims |
| descriptive-only differences | all cross-arm value/score differences without an exact reference; seed spreads are descriptive |
| default-readiness | not established by this board (Q5 scope); per-scope tuning rule stands |
| next evidence needed | SGQF/zhao-cui comparator hookup (slice B); paired-seed uncertainty analysis before any ranking language |

## Notes

- UKF-GF is a Gaussian-approximation comparator; on KSC (mixture) and gen-SV (heteroskedastic) rows it is density-misspecified (recorded in the row artifacts).
- The Austria row runs the Q2-calibrated annealed lane (k=4, c=8); per-scope calibration applies to that row only.
- Absent rows are reported ABSENT, not silently dropped.

## Stamps

- linear2d: `ledh-canonical-conformance-v1-2026-08-24@7944c4cbcff9`
- dlgssm: `ledh-canonical-conformance-v1-2026-08-24@b1db29624efe`
- predator_prey: `ledh-canonical-conformance-v1-2026-08-24@b1db29624efe`
- ksc_sv: `ledh-canonical-conformance-v1-2026-08-24@b1db29624efe`
- generalized_sv: `ledh-canonical-conformance-v1-2026-08-24@b1db29624efe`
- austria_sir: `ledh-canonical-conformance-v1-2026-08-24@6bc7a37558e1`
