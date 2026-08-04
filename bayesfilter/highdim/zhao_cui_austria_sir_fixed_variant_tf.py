"""Fixed source-order Austria SIR observed-data value and score mechanics.

Squared-TT and conditional KR operations used by later proposal builders are
source-grounded Zhao-Cui operations.  The latent pre-clipping state, frozen APF
assembly, and recursive score are BayesFilter extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping

import tensorflow as tf

from bayesfilter.highdim.models import zhao_cui_sir_austria_model
from bayesfilter.highdim.zhao_cui_predator_prey_fixed_variant_tf import (
    PreparedSourceOrderFrozenBranch,
    SourceOrderFrozenAPFProgram,
    prepare_source_order_frozen_apf_program,
    prepare_source_order_frozen_branch,
)
from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
    SIR_DATASET_SEED,
    SIR_HORIZON,
    SIR_OBSERVATION_DIM,
    SIR_OBSERVATION_SHA256,
    SIR_PARAMETER_DIM,
    SIR_STATE_DIM,
    SIR_STATE_SHA256,
    generate_frozen_sir_dataset_tf,
)


ROUTE_ID = "zhao_cui_austria_sir_source_order_fixed_branch_extension_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
TARGET_ID = "zhao_cui_austria_sir_seed81120_latent_preclip_y1_y20_v1"
ROW_ID = "austria_sir_T20"
EVENT_ORDER = "x0_then_20_transition_then_observe_steps_y1_y20"
MEASURE_ID = "full_state_lebesgue_v1"
SCORE_BACKEND_ID = "analytical_parameter_score_no_autodiff_v1"
RUNTIME_FP32_OBSERVATION_SHA256 = (
    "40c793fb374e84fcd347c66b189352b5997740cc753ea0be03441ecf32828009"
)
CLAIM_PARTICLE_COUNT = 1008
THETA_REFERENCE = (0.0, 0.0, 0.0)
TARGET_SCHEMA = "bayesfilter.zhao_cui_austria_sir_observed_data_target.v1"

_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), tf.float32)
_BASE_KAPPA = tf.constant(0.1, tf.float32)
_BASE_NU = tf.constant(18.0, tf.float32)
_BASE_OBSERVATION_VARIANCE = tf.constant(100.0, tf.float32)
_RK4_STEP = tf.constant(0.005, tf.float32)
_RK4_SUBSTEPS = 4
_PARAMETER_EYE = tf.eye(SIR_PARAMETER_DIM, dtype=tf.float32)


def _tensor_hash(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AustriaSIRObservedDataTarget:
    """Repository-issued exact Austria target in source and runtime dtypes."""

    source_states: tf.Tensor
    source_observations: tf.Tensor
    observations: tf.Tensor
    manifest: Mapping[str, object]
    target_identity: str

    def __post_init__(self) -> None:
        source_states = tf.convert_to_tensor(self.source_states, tf.float64)
        source_observations = tf.convert_to_tensor(self.source_observations, tf.float64)
        observations = tf.convert_to_tensor(self.observations, tf.float32)
        if source_states.shape != (SIR_HORIZON + 1, SIR_STATE_DIM):
            raise ValueError("Austria source states must have shape [21,18]")
        if source_observations.shape != (SIR_HORIZON, SIR_OBSERVATION_DIM):
            raise ValueError("Austria source observations must have shape [20,9]")
        if observations.shape != (SIR_HORIZON, SIR_OBSERVATION_DIM):
            raise ValueError("Austria runtime observations must have shape [20,9]")
        if _tensor_hash(source_states) != SIR_STATE_SHA256:
            raise ValueError("Austria source-state identity rejected")
        if _tensor_hash(source_observations) != SIR_OBSERVATION_SHA256:
            raise ValueError("Austria source-observation identity rejected")
        if _tensor_hash(observations) != RUNTIME_FP32_OBSERVATION_SHA256:
            raise ValueError("Austria runtime-observation identity rejected")
        manifest = dict(self.manifest)
        if self.target_identity != _semantic_hash(manifest):
            raise ValueError("Austria target manifest identity rejected")
        object.__setattr__(self, "source_states", source_states)
        object.__setattr__(self, "source_observations", source_observations)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))


def make_austria_sir_observed_data_target() -> AustriaSIRObservedDataTarget:
    """Load and seal the seed-81120 `y1:y20` comparison target."""

    source_states, source_observations, _all_observations = (
        generate_frozen_sir_dataset_tf()
    )
    observations = tf.cast(source_observations, tf.float32)
    manifest: dict[str, object] = {
        "schema": TARGET_SCHEMA,
        "row_id": ROW_ID,
        "target_id": TARGET_ID,
        "dataset_seed": SIR_DATASET_SEED,
        "event_order": EVENT_ORDER,
        "horizon": SIR_HORIZON,
        "state_dimension": SIR_STATE_DIM,
        "observation_dimension": SIR_OBSERVATION_DIM,
        "parameter_dimension": SIR_PARAMETER_DIM,
        "parameter_order": (
            "log_kappa_scale",
            "log_nu_scale",
            "log_observation_noise_scale",
        ),
        "theta_reference": THETA_REFERENCE,
        "source_state_sha256": SIR_STATE_SHA256,
        "source_observation_sha256": SIR_OBSERVATION_SHA256,
        "runtime_fp32_observation_sha256": RUNTIME_FP32_OBSERVATION_SHA256,
        "filtering_state": "latent_pre_clipping_gaussian_z_t",
        "physical_state": "x0=z0; susceptible coordinates clipped for t>=1",
        "measure_id": MEASURE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
    }
    return AustriaSIRObservedDataTarget(
        source_states=source_states,
        source_observations=source_observations,
        observations=observations,
        manifest=manifest,
        target_identity=_semantic_hash(manifest),
    )


def _rows(value: tf.Tensor, width: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float32)
    if tensor.shape.rank != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape [batch,{width}]")
    return tensor


class AustriaSIRLatentPreclipFP32Model:
    """Graph-native FP32 density and manual-score model for a frozen branch."""

    def __init__(self) -> None:
        source = zhao_cui_sir_austria_model()
        self._initial_mean = tf.cast(source.initial_mean, tf.float32)
        self._adjacency = tf.cast(source._adjacency_matrix, tf.float32)  # noqa: SLF001
        self._degree = tf.reduce_sum(self._adjacency, axis=1)

    def parameter_dim(self) -> int:
        return SIR_PARAMETER_DIM

    def state_dim(self) -> int:
        return SIR_STATE_DIM

    def observation_dim(self) -> int:
        return SIR_OBSERVATION_DIM

    def frozen_apf_measure_id(self) -> str:
        return MEASURE_ID

    def frozen_apf_score_backend_id(self) -> str:
        return SCORE_BACKEND_ID

    def initial_log_density(self, theta: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        del theta
        values = _rows(state, SIR_STATE_DIM, "state")
        residual = values - self._initial_mean[tf.newaxis, :]
        return -0.5 * (
            tf.constant(float(SIR_STATE_DIM), tf.float32) * _LOG_TWO_PI
            + tf.reduce_sum(tf.square(residual), axis=1)
        )

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, state: tf.Tensor
    ) -> tf.Tensor:
        del theta
        values = _rows(state, SIR_STATE_DIM, "state")
        return tf.zeros([tf.shape(values)[0], SIR_PARAMETER_DIM], tf.float32)

    def transition_mean(
        self, theta: tf.Tensor, previous: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        mean, _jacobian = self._transition_mean_and_parameter_jacobian(
            theta, previous, time_index
        )
        return mean

    def transition_log_density(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        next_values = _rows(current, SIR_STATE_DIM, "current")
        mean = self.transition_mean(theta, previous, time_index)
        residual = next_values - mean
        return -0.5 * (
            tf.constant(float(SIR_STATE_DIM), tf.float32) * _LOG_TWO_PI
            + tf.reduce_sum(tf.square(residual), axis=1)
        )

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        next_values = _rows(current, SIR_STATE_DIM, "current")
        mean, jacobian = self._transition_mean_and_parameter_jacobian(
            theta, previous, time_index
        )
        residual = next_values - mean
        return tf.reduce_sum(jacobian * residual[:, :, tf.newaxis], axis=1)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del time_index
        values = _rows(state, SIR_STATE_DIM, "state")
        y = tf.reshape(tf.convert_to_tensor(observation, tf.float32), [SIR_OBSERVATION_DIM])
        variance = _BASE_OBSERVATION_VARIANCE * tf.exp(2.0 * theta[2])
        residual = y[tf.newaxis, :] - values[:, 1::2]
        return -0.5 * tf.reduce_sum(
            _LOG_TWO_PI + tf.math.log(variance) + tf.square(residual) / variance,
            axis=1,
        )

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del time_index
        values = _rows(state, SIR_STATE_DIM, "state")
        y = tf.reshape(tf.convert_to_tensor(observation, tf.float32), [SIR_OBSERVATION_DIM])
        variance = _BASE_OBSERVATION_VARIANCE * tf.exp(2.0 * theta[2])
        residual = y[tf.newaxis, :] - values[:, 1::2]
        direct = tf.reduce_sum(tf.square(residual) / variance - 1.0, axis=1)
        return direct[:, tf.newaxis] * _PARAMETER_EYE[2][tf.newaxis, :]

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "AustriaSIRLatentPreclipFP32Model",
            "target_id": TARGET_ID,
            "state_dimension": SIR_STATE_DIM,
            "observation_dimension": SIR_OBSERVATION_DIM,
            "parameter_dimension": SIR_PARAMETER_DIM,
            "dtype": "float32",
            "rk4_variant": "zhao_cui_half_step_fourth_stage",
            "rk4_substeps": _RK4_SUBSTEPS,
            "rk4_step": 0.005,
            "process_covariance": "I18",
            "observation_covariance": "100*exp(2*theta[2])*I9",
            "measure_id": MEASURE_ID,
            "score_backend_id": SCORE_BACKEND_ID,
            "runtime_autodiff": False,
            "runtime_finite_difference": False,
            "classification": ROUTE_CLASSIFICATION,
        }

    def _transition_mean_and_parameter_jacobian(
        self, theta: tf.Tensor, previous: tf.Tensor, time_index: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        parameters = tf.reshape(tf.convert_to_tensor(theta, tf.float32), [SIR_PARAMETER_DIM])
        latent_previous = _rows(previous, SIR_STATE_DIM, "previous")
        clipped = tf.reshape(
            tf.stack(
                [
                    tf.maximum(latent_previous[:, 0::2], 0.0),
                    latent_previous[:, 1::2],
                ],
                axis=2,
            ),
            [tf.shape(latent_previous)[0], SIR_STATE_DIM],
        )
        current = tf.where(tf.equal(time_index, 1), latent_previous, clipped)
        tangent = tf.zeros(
            [tf.shape(current)[0], SIR_STATE_DIM, SIR_PARAMETER_DIM], tf.float32
        )

        def body(
            index: tf.Tensor, state: tf.Tensor, state_tangent: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            k1, d1 = self._rhs_with_parameter_jacobian(parameters, state, state_tangent)
            k2, d2 = self._rhs_with_parameter_jacobian(
                parameters,
                state + 0.5 * _RK4_STEP * k1,
                state_tangent + 0.5 * _RK4_STEP * d1,
            )
            k3, d3 = self._rhs_with_parameter_jacobian(
                parameters,
                state + 0.5 * _RK4_STEP * k2,
                state_tangent + 0.5 * _RK4_STEP * d2,
            )
            k4, d4 = self._rhs_with_parameter_jacobian(
                parameters,
                state + 0.5 * _RK4_STEP * k3,
                state_tangent + 0.5 * _RK4_STEP * d3,
            )
            return (
                index + 1,
                state + _RK4_STEP / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
                state_tangent
                + _RK4_STEP / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4),
            )

        result = tf.while_loop(
            lambda index, *_unused: index < _RK4_SUBSTEPS,
            body,
            (tf.zeros([], tf.int32), current, tangent),
            maximum_iterations=_RK4_SUBSTEPS,
            parallel_iterations=1,
        )
        return result[1], result[2]

    def _rhs_with_parameter_jacobian(
        self, theta: tf.Tensor, state: tf.Tensor, tangent: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        kappa = _BASE_KAPPA * tf.exp(theta[0])
        nu = _BASE_NU * tf.exp(theta[1])
        susceptible = state[:, 0::2]
        infectious = state[:, 1::2]
        d_susceptible = tangent[:, 0::2, :]
        d_infectious = tangent[:, 1::2, :]
        susceptible_neighbor = (
            tf.linalg.matmul(susceptible, self._adjacency, transpose_b=True)
            - susceptible * self._degree[tf.newaxis, :]
        )
        infectious_neighbor = (
            tf.linalg.matmul(infectious, self._adjacency, transpose_b=True)
            - infectious * self._degree[tf.newaxis, :]
        )
        d_susceptible_neighbor = (
            tf.einsum("njp,kj->nkp", d_susceptible, self._adjacency)
            - d_susceptible * self._degree[tf.newaxis, :, tf.newaxis]
        )
        d_infectious_neighbor = (
            tf.einsum("njp,kj->nkp", d_infectious, self._adjacency)
            - d_infectious * self._degree[tf.newaxis, :, tf.newaxis]
        )
        infection = kappa * susceptible * infectious
        d_infection = kappa * (
            infectious[:, :, tf.newaxis] * d_susceptible
            + susceptible[:, :, tf.newaxis] * d_infectious
        ) + infection[:, :, tf.newaxis] * _PARAMETER_EYE[0][tf.newaxis, tf.newaxis, :]
        rhs_susceptible = -infection + 0.5 * susceptible_neighbor
        rhs_infectious = infection - nu * infectious + 0.5 * infectious_neighbor
        d_rhs_susceptible = -d_infection + 0.5 * d_susceptible_neighbor
        d_rhs_infectious = (
            d_infection
            - nu * d_infectious
            - (nu * infectious)[:, :, tf.newaxis]
            * _PARAMETER_EYE[1][tf.newaxis, tf.newaxis, :]
            + 0.5 * d_infectious_neighbor
        )
        return (
            tf.reshape(
                tf.stack([rhs_susceptible, rhs_infectious], axis=2),
                [tf.shape(state)[0], SIR_STATE_DIM],
            ),
            tf.reshape(
                tf.stack([d_rhs_susceptible, d_rhs_infectious], axis=2),
                [tf.shape(state)[0], SIR_STATE_DIM, SIR_PARAMETER_DIM],
            ),
        )


@dataclass(frozen=True)
class AustriaSIRSourceOrderProgram:
    """Austria target wrapper around the shared source-order APF recursion."""

    target: AustriaSIRObservedDataTarget
    branch: PreparedSourceOrderFrozenBranch
    require_claim_scope: bool = False
    model: AustriaSIRLatentPreclipFP32Model = field(
        default_factory=AustriaSIRLatentPreclipFP32Model
    )
    _delegate: SourceOrderFrozenAPFProgram = field(init=False, repr=False)
    program_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_austria_branch(self.target, self.branch, self.require_claim_scope)
        delegate = prepare_source_order_frozen_apf_program(self.model, self.branch)
        payload = {
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "target_identity": self.target.target_identity,
            "delegate_program_id": delegate.program_id,
            "require_claim_scope": bool(self.require_claim_scope),
        }
        object.__setattr__(self, "_delegate", delegate)
        object.__setattr__(self, "program_id", _semantic_hash(payload))

    def evaluate(self, theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
        return self._delegate.evaluate(theta)

    def compiled(
        self, *, jit_compile: bool = True
    ) -> Callable[[tf.Tensor], Mapping[str, tf.Tensor]]:
        return self._delegate.compiled(jit_compile=jit_compile)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            **self._delegate.manifest_payload(),
            "austria_route_id": ROUTE_ID,
            "austria_route_classification": ROUTE_CLASSIFICATION,
            "austria_program_id": self.program_id,
            "target_identity": self.target.target_identity,
            "source_observation_sha256": SIR_OBSERVATION_SHA256,
            "runtime_fp32_observation_sha256": RUNTIME_FP32_OBSERVATION_SHA256,
            "require_claim_scope": bool(self.require_claim_scope),
            "production_kr_closure": False,
        }


def prepare_austria_sir_source_order_branch(
    *,
    target: AustriaSIRObservedDataTarget,
    observations: tf.Tensor,
    states: tf.Tensor,
    initial_log_proposal_density: tf.Tensor,
    ancestors: tf.Tensor,
    auxiliary_log_probabilities: tf.Tensor,
    transition_log_proposal_density: tf.Tensor,
    proposal_compiler_id: str,
    require_claim_scope: bool = False,
) -> PreparedSourceOrderFrozenBranch:
    """Issue an Austria branch identity from actual tensors and target data."""

    values = tf.convert_to_tensor(observations, tf.float32)
    if values.shape.rank != 2 or values.shape[1] != SIR_OBSERVATION_DIM:
        raise ValueError("Austria observations must have shape [T,9]")
    horizon = int(values.shape[0])
    if horizon < 1 or horizon > SIR_HORIZON:
        raise ValueError("Austria branch horizon must be in [1,20]")
    tf.debugging.assert_equal(values, target.observations[:horizon])
    source_state_hash = _tensor_hash(target.source_states[: horizon + 1])
    observation_hash = _tensor_hash(values)
    branch = prepare_source_order_frozen_branch(
        observations=values,
        states=tf.convert_to_tensor(states, tf.float32),
        initial_log_proposal_density=initial_log_proposal_density,
        ancestors=ancestors,
        auxiliary_log_probabilities=auxiliary_log_probabilities,
        transition_log_proposal_density=transition_log_proposal_density,
        target_id=(TARGET_ID if horizon == SIR_HORIZON else f"{TARGET_ID}_prefix_t{horizon}"),
        event_order=EVENT_ORDER,
        target_seed=SIR_DATASET_SEED,
        target_state_sha256=source_state_hash,
        target_observation_sha256=observation_hash,
        proposal_compiler_id=proposal_compiler_id,
    )
    _require_austria_branch(target, branch, require_claim_scope)
    return branch


def prepare_austria_sir_source_order_program(
    branch: PreparedSourceOrderFrozenBranch,
    *,
    target: AustriaSIRObservedDataTarget | None = None,
    require_claim_scope: bool = False,
) -> AustriaSIRSourceOrderProgram:
    return AustriaSIRSourceOrderProgram(
        target=target or make_austria_sir_observed_data_target(),
        branch=branch,
        require_claim_scope=require_claim_scope,
    )


def make_bootstrap_mechanics_branch(
    *,
    particle_count: int,
    horizon: int,
    proposal_seed: int,
    target: AustriaSIRObservedDataTarget | None = None,
) -> PreparedSourceOrderFrozenBranch:
    """Build a fixed bootstrap branch for parity and memory-safe mechanics only."""

    if int(particle_count) < 2:
        raise ValueError("particle_count must be at least two")
    if int(horizon) < 1 or int(horizon) > SIR_HORIZON:
        raise ValueError("horizon must be in [1,20]")
    target = target or make_austria_sir_observed_data_target()
    model = AustriaSIRLatentPreclipFP32Model()
    theta_reference = tf.zeros([SIR_PARAMETER_DIM], tf.float32)
    initial_noise = tf.random.stateless_normal(
        [int(particle_count), SIR_STATE_DIM],
        seed=tf.constant([int(proposal_seed), 0], tf.int32),
        dtype=tf.float32,
    )
    initial_state = model._initial_mean[tf.newaxis, :] + initial_noise  # noqa: SLF001
    states = [initial_state]
    initial_log_q = model.initial_log_density(theta_reference, initial_state)
    ancestors = tf.tile(
        tf.range(int(particle_count), dtype=tf.int32)[tf.newaxis, :],
        [int(horizon), 1],
    )
    auxiliary_log = tf.fill(
        [int(horizon), int(particle_count)],
        -tf.math.log(tf.cast(int(particle_count), tf.float32)),
    )
    transition_log_q = []
    previous = initial_state
    for time_index in range(1, int(horizon) + 1):
        mean = model.transition_mean(
            theta_reference, previous, tf.constant(time_index, tf.int32)
        )
        noise = tf.random.stateless_normal(
            [int(particle_count), SIR_STATE_DIM],
            seed=tf.constant([int(proposal_seed), time_index], tf.int32),
            dtype=tf.float32,
        )
        current = mean + noise
        transition_log_q.append(
            model.transition_log_density(
                theta_reference,
                previous,
                current,
                tf.constant(time_index, tf.int32),
            )
        )
        states.append(current)
        previous = current
    return prepare_austria_sir_source_order_branch(
        target=target,
        observations=target.observations[: int(horizon)],
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=ancestors,
        auxiliary_log_probabilities=auxiliary_log,
        transition_log_proposal_density=tf.stack(transition_log_q),
        proposal_compiler_id=(
            "austria_bootstrap_uniform_identity_mechanics_only_"
            f"seed{int(proposal_seed)}"
        ),
        require_claim_scope=False,
    )


def _require_austria_branch(
    target: AustriaSIRObservedDataTarget,
    branch: PreparedSourceOrderFrozenBranch,
    require_claim_scope: bool,
) -> None:
    if branch.event_order != EVENT_ORDER:
        raise ValueError("Austria event order rejected")
    if branch.target_seed != SIR_DATASET_SEED:
        raise ValueError("Austria target seed rejected")
    if branch.state_dimension != SIR_STATE_DIM:
        raise ValueError("Austria state dimension rejected")
    if branch.observation_dimension != SIR_OBSERVATION_DIM:
        raise ValueError("Austria observation dimension rejected")
    horizon = branch.transition_count
    if horizon < 1 or horizon > SIR_HORIZON:
        raise ValueError("Austria branch horizon rejected")
    if branch.target_id != (
        TARGET_ID if horizon == SIR_HORIZON else f"{TARGET_ID}_prefix_t{horizon}"
    ):
        raise ValueError("Austria target id rejected")
    expected_observations = target.observations[:horizon]
    if branch.observations.shape != expected_observations.shape or not bool(
        tf.reduce_all(tf.equal(branch.observations, expected_observations)).numpy()
    ):
        raise ValueError("Austria observation tensor rejected")
    if branch.target_observation_sha256 != _tensor_hash(expected_observations):
        raise ValueError("Austria observation hash rejected")
    if branch.target_state_sha256 != _tensor_hash(target.source_states[: horizon + 1]):
        raise ValueError("Austria source-state hash rejected")
    if require_claim_scope:
        if horizon != SIR_HORIZON or branch.particle_count != CLAIM_PARTICLE_COUNT:
            raise ValueError("Austria claim scope requires T=20 and N=1008")
        if branch.target_observation_sha256 != RUNTIME_FP32_OBSERVATION_SHA256:
            raise ValueError("Austria claim runtime observation hash rejected")
        if "bootstrap" in branch.proposal_compiler_id:
            raise ValueError("bootstrap mechanics branch is not claim eligible")


__all__ = [
    "AustriaSIRLatentPreclipFP32Model",
    "AustriaSIRObservedDataTarget",
    "AustriaSIRSourceOrderProgram",
    "CLAIM_PARTICLE_COUNT",
    "EVENT_ORDER",
    "MEASURE_ID",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "ROW_ID",
    "RUNTIME_FP32_OBSERVATION_SHA256",
    "SCORE_BACKEND_ID",
    "TARGET_ID",
    "make_austria_sir_observed_data_target",
    "make_bootstrap_mechanics_branch",
    "prepare_austria_sir_source_order_branch",
    "prepare_austria_sir_source_order_program",
]
