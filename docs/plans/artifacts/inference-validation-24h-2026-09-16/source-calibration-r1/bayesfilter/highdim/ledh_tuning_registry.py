"""Explicit per-route LEDH tuning families.

This registry is deliberately descriptive. Route-specific tuner programs bind
these control names to their own finite value-and-score callables; no generic
OT tuner is applied to Contract E--TP routes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LEDHTuningRoute:
    model_id: str
    target_id: str
    route_id: str
    reset_contract_id: str
    control_family_id: str
    tunable_controls: tuple[str, ...]
    fixed_controls: tuple[str, ...]
    tuner_program: str
    tuner_status: str
    status: str = "ACTIVE_REQUIRES_SCOPE_TUNING"


ROUTES: tuple[LEDHTuningRoute, ...] = (
    LEDHTuningRoute(
        model_id="canonical_lgssm_m3",
        target_id="canonical_lgssm",
        route_id="ledh_contract_e_canonical_lgssm_tf",
        reset_contract_id="contract_e_chol_v1",
        control_family_id="streaming_ot_sinkhorn_balance_v1",
        tunable_controls=("sinkhorn_steps", "balance_steps"),
        fixed_controls=("epsilon", "scaling", "prepared_ridge", "particle_count"),
        tuner_program="docs/benchmarks/run_ledh_offline_ot_tuning_campaign.py",
        tuner_status="IMPLEMENTED",
    ),
    LEDHTuningRoute(
        model_id="latent_preclip_sir",
        target_id="latent_preclip_sir",
        route_id="ledh_contract_e_latent_sir_tf",
        reset_contract_id="contract_e_chol_v1",
        control_family_id="streaming_ot_sinkhorn_balance_v1",
        tunable_controls=(
            "sinkhorn_steps",
            "balance_steps",
            "epsilon",
            "scaling",
            "prepared_ridge",
        ),
        fixed_controls=("particle_count", "preclip_semantics", "teacher_route"),
        tuner_program="docs/benchmarks/run_ledh_latent_sir_scope_tuning.py",
        tuner_status="REQUIRED_NOT_IMPLEMENTED",
    ),
    LEDHTuningRoute(
        model_id="actual_sv",
        target_id="actual_sv",
        route_id="ledh_contract_e_tp_scalar_sv_tf",
        reset_contract_id="contract_e_tp_v1",
        control_family_id="tp_feature_chart_v1",
        tunable_controls=(
            "teacher_quadrature_order",
            "continuation_quadrature_order",
            "lookahead_steps",
            "active_feature_basis",
            "row_scale_policy",
            "ridge_policy",
        ),
        fixed_controls=("time_order", "parameterization", "dtype", "jit_compile"),
        tuner_program="docs/benchmarks/run_ledh_tp_scope_tuning.py",
        tuner_status="REQUIRED_NOT_IMPLEMENTED",
    ),
    LEDHTuningRoute(
        model_id="generalized_sv",
        target_id="generalized_sv",
        route_id="ledh_contract_e_tp_structural_tf",
        reset_contract_id="contract_e_tp_v1",
        control_family_id="tp_structural_feature_chart_v1",
        tunable_controls=(
            "teacher_quadrature_order",
            "continuation_quadrature_order",
            "lookahead_steps",
            "active_feature_basis",
            "row_scale_policy",
            "ridge_policy",
        ),
        fixed_controls=("time_order", "parameterization", "dtype", "jit_compile"),
        tuner_program="docs/benchmarks/run_ledh_tp_scope_tuning.py",
        tuner_status="REQUIRED_NOT_IMPLEMENTED",
    ),
    LEDHTuningRoute(
        model_id="ksc_sv",
        target_id="ksc_sv",
        route_id="ledh_contract_e_tp_structural_tf",
        reset_contract_id="contract_e_tp_v1",
        control_family_id="tp_structural_feature_chart_v1",
        tunable_controls=(
            "teacher_quadrature_order",
            "continuation_quadrature_order",
            "lookahead_steps",
            "active_feature_basis",
            "row_scale_policy",
            "ridge_policy",
        ),
        fixed_controls=("time_order", "parameterization", "dtype", "jit_compile"),
        tuner_program="docs/benchmarks/run_ledh_tp_scope_tuning.py",
        tuner_status="REQUIRED_NOT_IMPLEMENTED",
    ),
    LEDHTuningRoute(
        model_id="predator_prey",
        target_id="predator_prey",
        route_id="ledh_contract_e_tp_predator_prey_tf",
        reset_contract_id="contract_e_tp_v1",
        control_family_id="tp_predator_prey_feature_chart_v1",
        tunable_controls=(
            "teacher_quadrature_order",
            "continuation_quadrature_order",
            "lookahead_steps",
            "active_feature_basis",
            "row_scale_policy",
            "ridge_policy",
        ),
        fixed_controls=("time_order", "parameterization", "dtype", "jit_compile"),
        tuner_program="docs/benchmarks/run_ledh_tp_scope_tuning.py",
        tuner_status="REQUIRED_NOT_IMPLEMENTED",
    ),
)


def route_for_model(model_id: str) -> LEDHTuningRoute:
    matches = [route for route in ROUTES if route.model_id == model_id]
    if len(matches) != 1:
        raise KeyError(f"expected one LEDH tuning route for {model_id!r}")
    return matches[0]


def require_active_route_tuning(model_id: str, *, selected_scope_sha256: str | None) -> LEDHTuningRoute:
    route = route_for_model(model_id)
    if route.status != "ACTIVE_REQUIRES_SCOPE_TUNING":
        raise ValueError(f"LEDH route {model_id!r} is not active for admission")
    if route.tuner_status != "IMPLEMENTED":
        raise ValueError(f"LEDH route {model_id!r} has no implemented scope tuner")
    if not selected_scope_sha256:
        raise ValueError(f"LEDH route {model_id!r} has no selected scope tuning artifact")
    return route


__all__ = ["LEDHTuningRoute", "ROUTES", "require_active_route_tuning", "route_for_model"]
