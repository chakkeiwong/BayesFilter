"""Target-specific model boundaries for experimental Contract E--TP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    FIXED_SIR_AUSTRIA_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
    PREDATOR_PREY_ROW_ID,
)
from bayesfilter.highdim.models import (
    GeneralizedSVPriorMeanSSM,
    ParameterizedZhaoCuiSIRSSM,
    PredatorPreySSM,
    parameterized_zhao_cui_sir_austria_model,
    p30_predator_prey_fixture_model,
)
from bayesfilter.highdim.sv_mixture_cut4 import (
    ExactTransformedSVSSM,
    KSCMixtureTransformedSVSSM,
)


TransitionPush = Callable[[tf.Tensor, tf.Tensor, tf.Tensor, int], tf.Tensor]
SupportPredicate = Callable[[tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class ContractETPModelAdapter:
    row_id: str
    parameter_order: tuple[str, ...]
    theta_coordinate_system: str
    target_observation_policy: str
    innovation_dimension: int
    support_policy: str
    model: object
    transition_push: TransitionPush
    support_valid: SupportPredicate
    proposal_flow_status: str
    preparation_region_status: str

    @property
    def parameter_dimension(self) -> int:
        return len(self.parameter_order)


def _all_finite(points: tf.Tensor) -> tf.Tensor:
    return tf.reduce_all(tf.math.is_finite(points))


def _strictly_positive_and_finite(points: tf.Tensor) -> tf.Tensor:
    return _all_finite(points) & tf.reduce_all(points > 0.0)


def _actual_sv_push(
    model: ExactTransformedSVSSM,
) -> TransitionPush:
    def push(
        theta: tf.Tensor, parents: tf.Tensor, standard_noise: tf.Tensor, time_index: int
    ) -> tf.Tensor:
        del time_index
        parameters = model.physical_parameters(theta)
        parents = tf.convert_to_tensor(parents, tf.float64)
        noise = tf.convert_to_tensor(standard_noise, tf.float64)
        return parameters["gamma"] * parents + model.sigma * noise

    return push


def _generalized_sv_push(
    model: GeneralizedSVPriorMeanSSM,
) -> TransitionPush:
    def push(
        theta: tf.Tensor, parents: tf.Tensor, standard_noise: tf.Tensor, time_index: int
    ) -> tf.Tensor:
        del time_index
        parameters = model.physical_parameters(theta)
        parents = tf.convert_to_tensor(parents, tf.float64)
        noise = tf.convert_to_tensor(standard_noise, tf.float64)
        return (
            parameters["mu"]
            + parameters["gamma"] * (parents - parameters["mu"])
            + model.process_scale * noise
        )

    return push


def _predator_prey_push(model: PredatorPreySSM) -> TransitionPush:
    chol = tf.linalg.cholesky(model.process_covariance)

    def push(
        theta: tf.Tensor, parents: tf.Tensor, standard_noise: tf.Tensor, time_index: int
    ) -> tf.Tensor:
        del time_index
        mean = model.transition_mean(theta, parents)
        noise = tf.convert_to_tensor(standard_noise, tf.float64)
        return mean + tf.linalg.matmul(noise, chol, transpose_b=True)

    return push


def _sir_push(model: ParameterizedZhaoCuiSIRSSM) -> TransitionPush:
    def push(
        theta: tf.Tensor, parents: tf.Tensor, standard_noise: tf.Tensor, time_index: int
    ) -> tf.Tensor:
        return model.transition_push_from_standard_normal(
            theta, parents, standard_noise, time_index
        )

    return push


def make_actual_sv_contract_e_tp_adapter() -> ContractETPModelAdapter:
    model = ExactTransformedSVSSM(sigma=1.0)
    return ContractETPModelAdapter(
        row_id=ACTUAL_SV_ROW_ID,
        parameter_order=("gamma_unconstrained", "log_beta"),
        theta_coordinate_system="synthetic_unconstrained",
        target_observation_policy="transformed_actual_sv_log_y_square",
        innovation_dimension=1,
        support_policy="real_line_finite",
        model=model,
        transition_push=_actual_sv_push(model),
        support_valid=_all_finite,
        proposal_flow_status="gaussianized_ledh_surface_exists_requires_tp_binding",
        preparation_region_status="region_design_required_before_chart_preparation",
    )


def make_ksc_sv_contract_e_tp_adapter() -> ContractETPModelAdapter:
    model = KSCMixtureTransformedSVSSM(sigma=1.0, transform_offset=1.0e-8)
    return ContractETPModelAdapter(
        row_id=KSC_SV_ROW_ID,
        parameter_order=("gamma_unconstrained", "log_beta"),
        theta_coordinate_system="synthetic_unconstrained",
        target_observation_policy="ksc_log_chi_square_gaussian_mixture_surrogate",
        innovation_dimension=1,
        support_policy="real_line_finite",
        model=model,
        transition_push=_actual_sv_push(model),
        support_valid=_all_finite,
        proposal_flow_status="mixture_conditioned_gaussian_surfaces_requires_tp_binding",
        preparation_region_status="region_design_required_before_chart_preparation",
    )


def make_generalized_sv_contract_e_tp_adapter() -> ContractETPModelAdapter:
    model = GeneralizedSVPriorMeanSSM(process_scale=1.0)
    return ContractETPModelAdapter(
        row_id=GENERALIZED_SV_ROW_ID,
        parameter_order=("gamma_unconstrained", "log_tau", "mu"),
        theta_coordinate_system="source_route_active_transformed_prior_mean",
        target_observation_policy="source_route_prior_mean_generalized_sv_raw_observation",
        innovation_dimension=1,
        support_policy="real_line_finite",
        model=model,
        transition_push=_generalized_sv_push(model),
        support_valid=_all_finite,
        proposal_flow_status="log_square_gaussianized_surface_separate_from_raw_target_requires_tp_binding",
        preparation_region_status="region_design_required_before_chart_preparation",
    )


def make_predator_prey_contract_e_tp_adapter() -> ContractETPModelAdapter:
    model = p30_predator_prey_fixture_model()
    return ContractETPModelAdapter(
        row_id=PREDATOR_PREY_ROW_ID,
        parameter_order=("r", "K", "a", "s", "u", "v"),
        theta_coordinate_system="physical",
        target_observation_policy="additive_gaussian_predator_prey",
        innovation_dimension=2,
        support_policy="real_plane_finite_additive_gaussian_target",
        model=model,
        transition_push=_predator_prey_push(model),
        support_valid=_all_finite,
        proposal_flow_status="nonlinear_ledh_surface_exists_requires_tp_binding",
        preparation_region_status="model_parameter_box_exists_tp_region_not_reviewed",
    )


def make_sir_contract_e_tp_adapter() -> ContractETPModelAdapter:
    model = parameterized_zhao_cui_sir_austria_model()
    return ContractETPModelAdapter(
        row_id=FIXED_SIR_AUSTRIA_ROW_ID,
        parameter_order=(
            "log_kappa_scale",
            "log_nu_scale",
            "log_obs_noise_scale",
        ),
        theta_coordinate_system="sir_log_scale_theta",
        target_observation_policy="additive_gaussian_spatial_sir",
        innovation_dimension=18,
        support_policy="source_clip_susceptible_after_noise_and_finite",
        model=model,
        transition_push=_sir_push(model),
        support_valid=_all_finite,
        proposal_flow_status="fixed_sir_ledh_surface_exists_requires_tp_structural_binding",
        preparation_region_status="reviewed_log_scale_box_minus0p5_plus0p5",
    )


def contract_e_tp_model_adapters() -> tuple[ContractETPModelAdapter, ...]:
    return (
        make_actual_sv_contract_e_tp_adapter(),
        make_ksc_sv_contract_e_tp_adapter(),
        make_generalized_sv_contract_e_tp_adapter(),
        make_predator_prey_contract_e_tp_adapter(),
        make_sir_contract_e_tp_adapter(),
    )


__all__ = [
    "ContractETPModelAdapter",
    "contract_e_tp_model_adapters",
    "make_actual_sv_contract_e_tp_adapter",
    "make_generalized_sv_contract_e_tp_adapter",
    "make_ksc_sv_contract_e_tp_adapter",
    "make_predator_prey_contract_e_tp_adapter",
    "make_sir_contract_e_tp_adapter",
]
