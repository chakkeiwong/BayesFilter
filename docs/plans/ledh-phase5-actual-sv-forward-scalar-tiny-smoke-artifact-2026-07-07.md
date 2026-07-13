# Phase 5 Actual-SV Tiny Adapter Smoke Artifact

- JSON artifact: `docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`
- Row id: `zhao_cui_sv_actual_nongaussian_T1000`
- Admission status: `tiny_executed_not_full_row`
- Target scalar: `observed_data_log_likelihood_estimator`
- Target observation policy: `transformed_actual_sv_log_y_square`
- Flow observation policy: `gaussianized_exact_log_square_actual_sv_flow_observation`
- Target density used for correction: `True`
- Batch seeds: `[81120]`
- Num particles: `128`
- Time steps: `4`
- Log likelihood by seed: `[-7.566841125488281]`
- Output devices: `['/job:localhost/replica:0/task:0/device:GPU:0']`
- Finite output: `True`

## Nonclaims

- not full actual-SV row admission
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
