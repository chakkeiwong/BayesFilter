# KSC-SV Same-Target LEDH Score

- JSON artifact: `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-tiny-compact-score-2026-07-08.json`
- Row: `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`
- Score admission status: `tiny_score_diagnostic_not_admitted`
- Correctness: `{'kind': 'same_scalar_finite_difference', 'status': 'pass', 'max_abs_error': 1.688629603341374e-05, 'max_rel_error': 6.364022943512981e-05, 'fd_step': 1e-05, 'atol': 0.001, 'rtol': 0.002}`
- Memory diagnostics: `{'n10000_memory_pass': False, 'peak_mib': None, 'budget_mib': 14000.0}`

## Nonclaims

- not full KSC-SV score admission
- not exact native actual-SV likelihood evidence
- not actual-SV admission
- not generalized-SV admission
- not raw Gaussian observation likelihood evidence
- not augmented-noise Gaussian-closure evidence
- not HMC readiness evidence
- not posterior correctness evidence
- not scientific superiority evidence
- not runtime ranking evidence
