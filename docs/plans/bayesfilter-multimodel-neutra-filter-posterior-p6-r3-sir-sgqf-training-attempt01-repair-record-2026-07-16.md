# P6 R3 SIR-SGQF Training Attempt 01 Repair Record

Date: 2026-07-16

Attempt root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/training/screen/candidates/dim3_lr1e3/attempt-01`

Classification: `TRAINER_TELEMETRY_INTERFACE_DEFECT`.

The trusted RTX 4080 SUPER GPU initialized with memory growth, but tracing the
compiled training graph stopped before step 1 because the generic NeuTra
training status normalizer required `innovation_condition_estimate`. The
SIR-SGQF target supplies every validity-bearing field used by training:
`status_code`, `valid_pre_regularized_score`, `floor_count_value`, and
`min_innovation_eigenvalue`; condition number is explanatory telemetry only.

No optimizer update, loss observation, trained weight, heldout result, frozen
transport, or scientific result was produced. The only file is the immutable
training configuration written before graph tracing. Attempt 1 must not be
resumed or used as a recipe outcome.

The smallest identity-preserving repair makes condition-estimate telemetry
optional in the generic batch-status normalizer. Absence is represented by an
explicit `innovation_condition_estimate_available=false` flag and `null` in
training records. A neutral internal tensor exists only to preserve the static
compiled output structure and is never reported as measured telemetry. Target
value, reviewed score, validity, optimizer, target source, typed identity,
recipe, seed, GPU/XLA policy, and budget remain unchanged.

Editing the SIR target function was rejected because its source closure is
bound into the admitted typed target identity and would invalidate the
same-target comparator boundary.
