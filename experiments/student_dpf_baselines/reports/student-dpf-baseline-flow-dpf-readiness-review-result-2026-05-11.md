# Student DPF baseline flow and DPF readiness review result

## Date

2026-05-11

## Scope

This report covers MP4 of the quarantined student DPF experimental-baseline
stream.  It uses static inventory plus import/signature probes only.
No student filter was instantiated, no filter/update/step method was
called, no notebook or experiment script was executed, and no vendored
student code was modified.

## Command

`python -m experiments.student_dpf_baselines.runners.run_flow_dpf_readiness_review`

Working directory: `/home/ubuntu/python/BayesFilter`

## Provenance

- `advanced_particle_filter`: `d2a797c330e11befacbb736b5c86b8d03eb4a389`
- `2026MLCOE`: `020cfd7f2f848afa68432e95e6c6e747d3d2402d`

## Overall

- records: 28
- probe status counts: `{'importable': 28}`
- readiness counts: `{'candidate_for_bounded_comparison': 9, 'reproduction_gate_required': 14, 'excluded_pending_debug': 3, 'adapter_internal_only': 2}`
- candidate decision: `selected_bounded_candidate`

## Inventory

| Candidate | Implementation | Category | Probe | Readiness | Signature |
| --- | --- | --- | --- | --- | --- |
| `advanced_edh_flow` | advanced_particle_filter | `edh_ledh_flow` | `importable` | `candidate_for_bounded_comparison` | `(self, n_particles: int = 500, n_flow_steps: int = 29, flow_step_ratio: float = 1.2, re...` |
| `advanced_edh_particle_filter` | advanced_particle_filter | `edh_ledh_pfpf` | `importable` | `candidate_for_bounded_comparison` | `(self, n_particles: int = 500, n_flow_steps: int = 29, flow_step_ratio: float = 1.2, re...` |
| `advanced_ledh_flow` | advanced_particle_filter | `edh_ledh_flow` | `importable` | `candidate_for_bounded_comparison` | `(self, n_particles: int = 500, n_flow_steps: int = 29, flow_step_ratio: float = 1.2, re...` |
| `advanced_ledh_particle_filter` | advanced_particle_filter | `edh_ledh_pfpf` | `importable` | `candidate_for_bounded_comparison` | `(self, n_particles: int = 500, n_flow_steps: int = 29, flow_step_ratio: float = 1.2, re...` |
| `advanced_stochastic_pff_flow` | advanced_particle_filter | `stochastic_flow` | `importable` | `reproduction_gate_required` | `(self, n_particles: int = 100, n_flow_steps: int = 29, step_schedule: Literal['fine', '...` |
| `advanced_stochastic_pfpf` | advanced_particle_filter | `stochastic_flow` | `importable` | `reproduction_gate_required` | `(self, n_particles: int = 100, n_flow_steps: int = 29, flow_step_ratio: float = 1.2, Q_...` |
| `advanced_scalar_kernel_pff` | advanced_particle_filter | `kernel_pff` | `importable` | `excluded_pending_debug` | `(self, **kwargs)` |
| `advanced_matrix_kernel_pff` | advanced_particle_filter | `kernel_pff` | `importable` | `excluded_pending_debug` | `(self, **kwargs)` |
| `advanced_tf_dpf` | advanced_particle_filter | `dpf` | `importable` | `reproduction_gate_required` | `(self, n_particles: int = 100, resampler: str = 'soft', alpha: float = 0.5, epsilon: fl...` |
| `advanced_soft_resample` | advanced_particle_filter | `differentiable_resampling` | `importable` | `reproduction_gate_required` | `(particles: tensorflow.python.framework.tensor.Tensor, log_w_norm: tensorflow.python.fr...` |
| `advanced_sinkhorn_resample` | advanced_particle_filter | `differentiable_resampling` | `importable` | `reproduction_gate_required` | `(particles: tensorflow.python.framework.tensor.Tensor, log_w: tensorflow.python.framewo...` |
| `advanced_amortized_ot_resampler` | advanced_particle_filter | `neural_ot` | `importable` | `reproduction_gate_required` | `(self, ckpt_dir: Optional[str] = None, d: int = 2, N: int = 1000, eps: float = 0.5, dty...` |
| `advanced_hmc_corenflos_lg_dpf` | advanced_particle_filter | `hmc_parameter_inference` | `importable` | `reproduction_gate_required` | `(y_obs: tensorflow.python.framework.tensor.Tensor, truth: dict, *, resampler: str, n_pa...` |
| `mlcoe_particle_flow_filter` | 2026MLCOE | `edh_ledh_flow` | `importable` | `candidate_for_bounded_comparison` | `(self, model, N=100, mode='ledh', is_pfpf=True, mu=0.2)` |
| `mlcoe_edh` | 2026MLCOE | `edh_ledh_flow` | `importable` | `candidate_for_bounded_comparison` | `(self, model, N=100, steps=30)` |
| `mlcoe_ledh` | 2026MLCOE | `edh_ledh_flow` | `importable` | `candidate_for_bounded_comparison` | `(self, model, N=100, steps=30)` |
| `mlcoe_pfpf_edh` | 2026MLCOE | `edh_ledh_pfpf` | `importable` | `candidate_for_bounded_comparison` | `(self, model, N=100, steps=30)` |
| `mlcoe_pfpf_ledh` | 2026MLCOE | `edh_ledh_pfpf` | `importable` | `candidate_for_bounded_comparison` | `(self, model, N=100, steps=30)` |
| `mlcoe_kpff` | 2026MLCOE | `kernel_pff` | `importable` | `excluded_pending_debug` | `(self, ensemble, y_obs, obs_idx, R_var, kernel_type='matrix')` |
| `mlcoe_edh_solver` | 2026MLCOE | `flow_solver` | `importable` | `adapter_internal_only` | `(self, name=None)` |
| `mlcoe_ledh_solver` | 2026MLCOE | `flow_solver` | `importable` | `adapter_internal_only` | `(self, name=None)` |
| `mlcoe_ledh_optimized_flow` | 2026MLCOE | `stochastic_flow` | `importable` | `reproduction_gate_required` | `(self, model, n_steps=100, mu=0.2)` |
| `mlcoe_dpf` | 2026MLCOE | `dpf` | `importable` | `reproduction_gate_required` | `(self, transition_fn, observation_fn, num_particles)` |
| `mlcoe_dpfpf` | 2026MLCOE | `dpfpf` | `importable` | `reproduction_gate_required` | `(self, model, solver=None, resampler=None, N=30, n_steps=20)` |
| `mlcoe_soft_resampler` | 2026MLCOE | `differentiable_resampling` | `importable` | `reproduction_gate_required` | `(self, num_particles, alpha=0.5)` |
| `mlcoe_sinkhorn_resampler` | 2026MLCOE | `differentiable_resampling` | `importable` | `reproduction_gate_required` | `(self, epsilon=0.05, n_iter=10, thresh=0.001)` |
| `mlcoe_transformer_resampler` | 2026MLCOE | `neural_resampling` | `importable` | `reproduction_gate_required` | `(self, num_particles, latent_dim=128, num_heads=4)` |
| `mlcoe_phmc_pfpf` | 2026MLCOE | `hmc_parameter_inference` | `importable` | `reproduction_gate_required` | `(model, observations, solver=None, resampler=None, num_samples=50, beta=0.1, stepsize=0...` |

## Candidate Decision

- family: `importance_corrected_edh_pfpf`
- advanced path: `advanced_particle_filter.filters.edh.EDHParticleFilter`
- MLCOE path: `src.filters.flow_filters.PFPF_EDH`
- fixture: existing nonlinear Gaussian range-bearing fixture
- runtime cap: short horizon, <=64 particles, <=10 flow steps for first adapter spike
- adapter scope: adapter-owned bridges only; no vendored-code edits and no production bayesfilter imports
- blocker plan: If either API requires unrecorded model assumptions, stop and record blocked_missing_assumption rather than widening scope.
- reason: Both paths represent EDH particle-flow particle filters with importance correction; this is more semantically aligned than pure flow, kernel PFF, stochastic flow, neural DPF, or HMC.

Metrics for the later bounded adapter spike:
- `latent_state_rmse`
- `final_position_rmse`
- `average_ess_if_exposed`
- `resampling_count_if_exposed`
- `runtime_seconds`
- `finite_output_checks`
- `EKF_UKF_proxy_comparison`

## Hypothesis Results

- `F1_edh_ledh_first_candidate`: `supported_with_adapter_caveat`
- `F2_kernel_pff_excluded`: `supported`
- `F3_dpf_neural_ot_hmc_reproduction_gates`: `supported`
- `F4_import_signature_probe_sufficient`: `supported`
- `F5_reuse_existing_fixtures`: `supported`

## Interpretation

The EDH/PFPF-EDH pair is the only immediate comparison candidate.
It is still not ready for performance claims; it is ready only for a
small adapter-owned spike that reuses the existing nonlinear Gaussian
range-bearing fixture and records proxy metrics.

Kernel PFF remains excluded from routine comparison because MP3 found
that reduced runs completed but consistently hit the iteration cap.

DPF, differentiable resampling, neural OT, transformer resampling,
stochastic flow, and HMC paths are importable surfaces, not comparison
evidence.  Each needs its own reproduction gate before any result claim.

## Next Phase Recommendation

Create a scoped MP4 follow-up adapter spike for advanced EDHParticleFilter versus MLCOE PFPF_EDH on the existing nonlinear Gaussian range-bearing fixture.
