# ledh-phase5-lgssm-score-memory-n10000-2026-07-06

Historical note: this artifact used `dtype=float64`. Under the active LEDH
execution directive in
`docs/plans/bayesfilter-ledh-tf32-only-execution-policy-reset-2026-07-10.md`,
it is historical/reference evidence only and must not be cited as current TF32
LEDH production, admission, or leaderboard evidence.

- Row: `benchmark_lgssm_exact_oracle_m3_T50`
- Score route: `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`
- Particles: `10000`
- Score directional: `0.3782809953387445`
- FD directional: `0.378280994883795`
- Absolute error: `4.5494946698809713e-10`
- Relative error: `1.202675980538481e-09`
- GPU memory peak MiB: `197.2001953125`
- Memory budget MiB: `14000.0`
- Primary pass: `true`

## Nonclaims

- Phase 5 score-memory evidence only
- not HMC/NUTS readiness
- not posterior correctness
- not scientific superiority
