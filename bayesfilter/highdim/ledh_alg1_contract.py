"""Machine-readable Li(2017) Algorithm 1 contract for the canonical LEDH lane.

Transcribed from ``docs/chapters/ch19c_dpf_implementation_literature.tex``
(lines 230-340), which is the repository's implementation contract for
LEDH-PF-PF OT with dual-cap trust-region GenUT and the UKF per-particle
covariance lifecycle. This module is the source of truth the conformance
suite imports; prose documents are advisory.

Owner rulings encoded (2026-08-21):
- Only the canonical algorithm may hold claim-bearing status.
- The claim-bearing score path is the ANALYTICAL recursive gradient;
  autodiff is permitted only under ``*_oracle_*`` namespaces.
- Identity/constant placeholder covariances are forbidden for claim-bearing
  model callbacks (conformance C-10).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContractStep:
    step_id: str
    title: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    forbidden_shortcuts: tuple[str, ...]
    source_anchor: str
    conformance_tests: tuple[str, ...]


ALGORITHM_1_STEPS: tuple[ContractStep, ...] = (
    ContractStep(
        "ukf_predict",
        "Per-particle unscented covariance prediction",
        requires=("ancestor_states", "ancestor_covariances", "transition_mean_fn", "process_noise_covariance"),
        produces=("predicted_means", "predicted_covariances"),
        forbidden_shortcuts=(
            "identity_covariance",
            "shared_covariance_across_particles",
            "constant_covariance_across_time",
        ),
        source_anchor="ch19c eq. bf-pfpf-alg1-covariance-lifecycle (prediction arrow)",
        conformance_tests=("C-1",),
    ),
    ContractStep(
        "zero_noise_anchor",
        "Dual-state objects: zero-noise anchor and actual pre-flow sample",
        requires=("ancestor_states", "transition_mean_fn", "process_noise_sample"),
        produces=("anchor_states", "pre_flow_states"),
        forbidden_shortcuts=("anchor_equals_actual", "linearize_at_actual_sample"),
        source_anchor="ch19c eqs. bf-pfpf-alg1-zero-noise-anchor / preflo-sample",
        conformance_tests=("C-3",),
    ),
    ContractStep(
        "ledh_flow",
        "Per-particle LEDH affine flow with dual-state integration",
        requires=("anchor_states", "pre_flow_states", "predicted_covariances", "observation", "observation_model"),
        produces=("post_flow_states", "forward_log_det"),
        forbidden_shortcuts=(
            "shared_prior_precision",
            "flow_ignores_predicted_covariance",
            "single_state_integration",
        ),
        source_anchor="ch19c eqs. bf-pfpf-alg1-auxiliary-step / actual-step",
        conformance_tests=("C-2", "C-4"),
    ),
    ContractStep(
        "theta_product",
        "Forward determinant accumulated per pseudo-time step",
        requires=("flow_coefficients_per_substep",),
        produces=("forward_log_det",),
        forbidden_shortcuts=("determinant_of_different_map", "dropped_theta"),
        source_anchor="ch19c eqs. bf-pfpf-alg1-theta-product / det-product",
        conformance_tests=("C-4",),
    ),
    ContractStep(
        "pfpf_weight",
        "PF-PF importance weight with proposal-density denominator",
        requires=("post_flow_states", "pre_flow_states", "ancestor_states", "forward_log_det", "transition_density", "observation_density", "proposal_density"),
        produces=("weights",),
        forbidden_shortcuts=(
            "same_density_numerator_denominator",
            "transition_density_at_pre_flow",
            "dropped_log_det",
        ),
        source_anchor="ch19c eq. bf-pfpf-alg1-weight",
        conformance_tests=("C-5",),
    ),
    ContractStep(
        "ukf_update",
        "Per-particle unscented covariance update; covariance recursion",
        requires=("predicted_means", "predicted_covariances", "observation", "observation_model"),
        produces=("posterior_covariances",),
        forbidden_shortcuts=("skip_update", "shared_posterior_covariance"),
        source_anchor="ch19c eq. bf-pfpf-alg1-covariance-lifecycle (update arrow)",
        conformance_tests=("C-6",),
    ),
    ContractStep(
        "triple_resampling",
        "Ancestry moves the triple {state, covariance, weight} together",
        requires=("post_flow_states", "posterior_covariances", "weights", "ancestry_indices"),
        produces=("resampled_states", "resampled_covariances", "reset_weights"),
        forbidden_shortcuts=("states_only_resampling", "covariances_left_in_old_order"),
        source_anchor="ch19c eq. bf-pfpf-alg1-resampling-triple",
        conformance_tests=("C-7",),
    ),
    ContractStep(
        "ot_reset",
        "Sinkhorn OT + Contract E-Chol reset (BayesFilter reviewed extension)",
        requires=("post_flow_states", "weights", "reset_design"),
        produces=("reset_states", "reset_diagnostics"),
        forbidden_shortcuts=("non_canonical_reset_semantics",),
        source_anchor="AGENTS.md Contract E Canonical LEDH Reset Policy",
        conformance_tests=("C-8",),
    ),
    ContractStep(
        "dual_cap_trust_correction",
        "Dual-cap trust-region GenUT higher-moment correction (full surface)",
        requires=("reset_states", "weighted_moment_targets"),
        produces=("corrected_states", "correction_diagnostics"),
        forbidden_shortcuts=(
            "diagonal_only_reduction",
            "uncapped_undamped_steps",
            "value_score_surface_mismatch",
        ),
        source_anchor="general surface: higher_moment_contract_e.higher_moment_shape_jvp",
        conformance_tests=("C-8",),
    ),
    ContractStep(
        "analytical_score",
        "Analytical recursive parameter gradient; autodiff forbidden",
        requires=("all_stage_analytical_derivatives",),
        produces=("score",),
        forbidden_shortcuts=(
            "forward_accumulator",
            "gradient_tape",
            "finite_difference_shipping",
        ),
        source_anchor="owner directive 2026-08-21; ch19c score identity sections",
        conformance_tests=("C-9",),
    ),
)


@dataclass(frozen=True)
class EntryPoint:
    lane: str
    role: str  # "canonical" | "oracle" | "scaffold"
    module: str
    callable_name: str
    expiry_phase: str | None = None
    notes: str = ""


ENTRY_POINTS: tuple[EntryPoint, ...] = (
    EntryPoint(
        lane="single_cloud",
        role="canonical",
        module="bayesfilter.highdim.ledh_canonical_filter_tf",
        callable_name="canonical_value_and_diagnostics",
    ),
    EntryPoint(
        lane="single_cloud",
        role="canonical",
        module="bayesfilter.highdim.ledh_canonical_score_tf",
        callable_name="canonical_value_and_analytical_score",
        notes="analytical recursive score; P4",
    ),
    EntryPoint(
        lane="batch",
        role="canonical",
        module="bayesfilter.highdim.ledh_canonical_batch_tf",
        callable_name="canonical_batch_value_score",
        notes="P5",
    ),
    EntryPoint(
        lane="single_cloud",
        role="oracle",
        module="bayesfilter.highdim.ledh_canonical_autodiff_oracle_tf",
        callable_name="oracle_forward_autodiff_score",
        notes="parity judge only; never claim-bearing",
    ),
    EntryPoint(
        lane="batch_fused",
        role="canonical",
        module="bayesfilter.highdim.ledh_canonical_batch_fused_tf",
        callable_name="canonical_batch_fused_value_score",
        notes="NeuTra-eligible fused lane; parity-gated vs single-cloud authority",
    ),
    EntryPoint(
        lane="neutra_target",
        role="canonical",
        module="bayesfilter.highdim.ledh_canonical_neutra_targets_tf",
        callable_name="make_canonical_neutra_target",
        notes="P7 rebind: NeuTra targets on the canonical stack; fresh signatures",
    ),
    EntryPoint(
        lane="models",
        role="canonical",
        module="bayesfilter.highdim.ledh_canonical_models_tf",
        callable_name="austria_sir_canonical_model",
        notes="model adapters with C-10 covariance provenance",
    ),
)


# G-5 conformance stamp (conformance test plan, implemented 2026-08-24
# for the Q3 leaderboard): benchmark artifacts carry this version string
# plus the run commit in `alg1_conformance`, stamped only by runners
# that verified the canonical gate battery green at launch. Aggregators
# must refuse unstamped artifacts (historical-result quarantine).
ALG1_CONFORMANCE_SUITE_VERSION = "ledh-canonical-conformance-v1-2026-08-24"

# THE production program, defined in one place so that "production" is
# machine-checkable and the omission of a required mechanism is a loud,
# labeled deviation — never a silent default. Owner directive 2026-08-26:
# the dual-cap TRUST REGION is REQUIRED — it is the designed control for
# the covariance-explosion problem (reset-less probe: flow collapse at
# chaos-takeoff steps from unbounded cloud spread; Q2 Curve-2 seed-1:
# uncapped correction steps measurably harm on takeoff clouds).
# Parameter provenance: lm_damping 1e-2 justified (Q2 Curve 3);
# trust_radius 0.5 WARM START (Curve 2 criterion failed honestly; the
# multi-iteration calibration contract is the R2 gating item); dual-cap
# family constants = owner family decision (2026-08-07 spec, Curve 5);
# sinkhorn epsilon 2.0 WARM START (measured epsilon-sensitive, Curve-4/
# gap A2). Warm-start fields are production-REQUIRED mechanisms with
# not-yet-calibrated values — cells must say so in `tuning`, and must
# not omit the mechanism.
LEDH_PRODUCTION_PROGRAM_V1 = {
    # value filter kwargs (canonical_value_and_diagnostics)
    "filter": {
        "dual_cap_enabled": True,
        "trust_region_enabled": True,
        "trust_region_lm_damping": 1.0e-2,
        "trust_region_lm_scale_floor": 1.0e-4,
        "trust_region_radius": 0.5,
    },
    # dual-cap family constants enforced via _restore_cloud_primal
    # defaults (wiring-tested against this registry)
    "dual_cap_family": {
        "dual_cap_diagonal_steps": 4,
        "dual_cap_diagonal_strength": 0.2,
        "dual_cap_pairwise_steps": 4,
        "dual_cap_pairwise_strength": 0.02,
        "dual_cap_pairwise_particle_rms_cap": 2.0,
        "dual_cap_coordinate_cap": 0.98,
        "dual_cap_coordinate_cap_power": 8,
    },
    # score-lane kwargs (canonical_value_and_analytical_score); the
    # reset design is scope-shaped and supplied by the runner
    "score": {
        "reset_policy": "contract_e",
        "reset_sinkhorn_steps": 8,
        "reset_balance_steps": 8,
        "correction_steps": 4,
        "correction_strength": 0.2,
        "correction_lm_damping": 1.0e-2,
        "correction_trust_radius": 0.5,
        "pairwise_steps": 4,
        "pairwise_strength": 0.02,
        "pairwise_rms_cap": 2.0,
        "coordinate_cap": 0.98,
    },
}

FORBIDDEN_CALLBACK_PLACEHOLDERS = (
    "identity_covariance_without_reviewed_exception",
    "constant_covariance_without_provenance",
    "identity_transition_matrix_for_nonlinear_dynamics",
)

COVARIANCE_PROVENANCE_KINDS = (
    "sigma_point_predicted",
    "analytical_jacobian_propagated",
    "model_exact",
    "reviewed_exception",
)


def step(step_id: str) -> ContractStep:
    for entry in ALGORITHM_1_STEPS:
        if entry.step_id == step_id:
            return entry
    raise KeyError(step_id)


__all__ = [
    "ALGORITHM_1_STEPS",
    "ENTRY_POINTS",
    "ContractStep",
    "EntryPoint",
    "FORBIDDEN_CALLBACK_PLACEHOLDERS",
    "COVARIANCE_PROVENANCE_KINDS",
    "step",
]
