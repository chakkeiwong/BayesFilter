# Student DPF future-work usability gates result

## Status

Final label: `future_work_usability_gates_complete`.

This report is student-lane, comparison-only evidence.  It does not
promote student code, the clean-room prototype, or any neural/artifact
surface into production.

## Counts

- planned execution probes: `15`
- observed execution records: `15`
- total records including contract audit: `30`
- missing planned probes: `[]`
- artifact size bytes: `89721`

## Status Counts

- `blocked`: `6`
- `ok`: `24`

## Classification Counts

- `api_smoke_only`: `13`
- `blocked_environment_drift`: `2`
- `blocked_missing_assumption`: `4`
- `usable_component_only`: `5`
- `usable_for_clean_room_spec`: `6`

## Family Decisions

- `differentiable_resampling`: `component_spec_next`
- `dpf`: `clean_room_spec_next`
- `dpfpf`: `debug_gate_next`
- `neural_ot`: `component_spec_next`
- `neural_resampling`: `debug_gate_next`
- `stochastic_flow`: `clean_room_spec_next`

## Execution Records

| Probe | Family | Implementation | Status | Classification | Next decision |
| --- | --- | --- | --- | --- | --- |
| `advanced_soft_resampler` | `differentiable_resampling` | `advanced_particle_filter` | `ok` | `usable_component_only` | `component_spec_next` |
| `advanced_sinkhorn_resampler` | `differentiable_resampling` | `advanced_particle_filter` | `ok` | `usable_component_only` | `component_spec_next` |
| `mlcoe_soft_resampler` | `differentiable_resampling` | `2026MLCOE` | `ok` | `usable_component_only` | `component_spec_next` |
| `mlcoe_sinkhorn_resampler` | `differentiable_resampling` | `2026MLCOE` | `ok` | `usable_component_only` | `component_spec_next` |
| `advanced_amortized_ot_resampler` | `neural_ot` | `advanced_particle_filter` | `ok` | `usable_component_only` | `component_spec_next` |
| `mlcoe_transformer_resampler` | `neural_resampling` | `2026MLCOE` | `blocked` | `blocked_environment_drift` | `debug_gate_next` |
| `advanced_stochastic_pff_flow` | `stochastic_flow` | `advanced_particle_filter` | `ok` | `usable_for_clean_room_spec` | `clean_room_spec_next` |
| `advanced_stochastic_pfpf` | `stochastic_flow` | `advanced_particle_filter` | `ok` | `usable_for_clean_room_spec` | `clean_room_spec_next` |
| `mlcoe_stochastic_flow` | `stochastic_flow` | `2026MLCOE` | `blocked` | `blocked_missing_assumption` | `defer_until_artifacts_or_assumptions` |
| `advanced_tf_dpf_soft` | `dpf` | `advanced_particle_filter` | `blocked` | `blocked_environment_drift` | `debug_gate_next` |
| `advanced_tf_dpf_sinkhorn` | `dpf` | `advanced_particle_filter` | `ok` | `usable_for_clean_room_spec` | `clean_room_spec_next` |
| `advanced_tf_dpf_amortized` | `dpf` | `advanced_particle_filter` | `ok` | `usable_for_clean_room_spec` | `clean_room_spec_next` |
| `mlcoe_dpf_soft` | `dpf` | `2026MLCOE` | `ok` | `usable_for_clean_room_spec` | `clean_room_spec_next` |
| `mlcoe_dpf_sinkhorn` | `dpf` | `2026MLCOE` | `ok` | `usable_for_clean_room_spec` | `clean_room_spec_next` |
| `mlcoe_dpfpf` | `dpfpf` | `2026MLCOE` | `blocked` | `blocked_missing_assumption` | `defer_until_artifacts_or_assumptions` |

## Blockers

| Probe | Classification | Execution attempted | Blocker |
| --- | --- | ---: | --- |
| `mlcoe_transformer_resampler` | `blocked_environment_drift` | `True` | InvalidArgumentError: Exception encountered when calling WeightedMultiHeadAttention.call().  [1m{{function_node __wrapped__Mul_device_/job:localhost/replica:0/task:0/device:CPU... |
| `mlcoe_stochastic_flow` | `blocked_missing_assumption` | `False` | complete model object with exact flow/dPF field semantics; validated covariance and observation contract |
| `advanced_tf_dpf_soft` | `blocked_environment_drift` | `True` | ValueError: in user code:      File "/home/ubuntu/python/BayesFilter/experiments/student_dpf_baselines/vendor/advanced_particle_filter/tf_filters/differentiable_particle.py", li... |
| `mlcoe_dpfpf` | `blocked_missing_assumption` | `False` | complete model object with exact flow/dPF field semantics; validated covariance and observation contract |

## Successful Probe Highlights

| Probe | Runtime seconds | Key finite/gradient evidence |
| --- | ---: | --- |
| `advanced_soft_resampler` | 0.364283 | gradient_finite=True; finite_outputs=True |
| `advanced_sinkhorn_resampler` | 0.743984 | gradient_finite=True; finite_outputs=True |
| `mlcoe_soft_resampler` | 0.0182423 | gradient_finite=True; finite_outputs=True |
| `mlcoe_sinkhorn_resampler` | 0.0371381 | gradient_finite=True; finite_outputs=True |
| `advanced_amortized_ot_resampler` | 1.47081 | gradient_finite=True; finite_outputs=True |
| `advanced_stochastic_pff_flow` | 0.0149568 | finite_means=True; position_rmse=0.179145; average_ess=16 |
| `advanced_stochastic_pfpf` | 0.0150201 | finite_means=True; position_rmse=0.0954115; average_ess=3.2956 |
| `advanced_tf_dpf_sinkhorn` | 1.57272 | gradient_finite=True; finite_particles=True; finite_log_evidence=True |
| `advanced_tf_dpf_amortized` | 4.29489 | gradient_finite=True; finite_particles=True; finite_log_evidence=True |
| `mlcoe_dpf_soft` | 0.0373226 | gradient_finite=True; finite_particles=True |
| `mlcoe_dpf_sinkhorn` | 0.0739161 | gradient_finite=True; finite_particles=True |

## Interpretation

Non-neural differentiable resampling and any successful filter-level
smokes can inform later clean-room specifications.  Neural paths remain
artifact- or training-semantics dependent unless classified otherwise in
the machine-readable records.  Blocked paths are evidence of current
readiness limits, not failures of the future research direction.
