# Phase 7 KSC-SV Forward Scalar Artifact

- JSON artifact: `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`
- Row id: `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`
- Admission status: `n10000_same_target_value_admitted`
- Target scalar: `observed_data_log_likelihood_estimator`
- Target observation policy: `ksc_log_chi_square_gaussian_mixture_surrogate`
- Flow observation policy: `gaussianized_ksc_log_square_surrogate_flow_observation`
- Target density used for correction: `True`
- Batch seeds: `[81120, 81121, 81122, 81123, 81124]`
- Num particles: `10000`
- Time steps: `1000`
- Log likelihood by seed: `[-2288.165771484375, -2287.877685546875, -2287.852294921875, -2287.529296875, -2288.34375]`
- Output devices: `['/job:localhost/replica:0/task:0/device:GPU:0']`
- Finite output: `True`

## Nonclaims

- not score admission
- not score correctness
- not exact native actual-SV likelihood evidence
- not actual-SV admission
- not raw Gaussian observation likelihood evidence
- not augmented-noise Gaussian-closure evidence
- not generalized-SV admission
- not HMC readiness evidence
- not posterior correctness evidence
- not scientific superiority evidence
- not runtime ranking evidence
