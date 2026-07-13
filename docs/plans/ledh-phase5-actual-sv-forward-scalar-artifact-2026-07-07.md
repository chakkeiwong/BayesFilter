# Phase 5 Actual-SV Full-Row Forward Scalar Artifact

- JSON artifact: `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`
- Row id: `zhao_cui_sv_actual_nongaussian_T1000`
- Admission status: `n10000_same_target_value_admitted`
- Target scalar: `observed_data_log_likelihood_estimator`
- Target observation policy: `transformed_actual_sv_log_y_square`
- Flow observation policy: `gaussianized_exact_log_square_actual_sv_flow_observation`
- Target density used for correction: `True`
- Batch seeds: `[81120, 81121, 81122, 81123, 81124]`
- Num particles: `10000`
- Time steps: `1000`
- Log likelihood by seed: `[-2290.10205078125, -2289.888916015625, -2289.83154296875, -2289.517333984375, -2290.427490234375]`
- Output devices: `['/job:localhost/replica:0/task:0/device:GPU:0']`
- Finite output: `True`

## Nonclaims

- not score admission
- not score correctness
- not KSC surrogate likelihood evidence
- not raw Gaussian observation likelihood evidence
- not augmented-noise Gaussian-closure evidence
- not generalized-SV admission
- not HMC readiness evidence
- not posterior correctness evidence
- not scientific superiority evidence
- not runtime ranking evidence
