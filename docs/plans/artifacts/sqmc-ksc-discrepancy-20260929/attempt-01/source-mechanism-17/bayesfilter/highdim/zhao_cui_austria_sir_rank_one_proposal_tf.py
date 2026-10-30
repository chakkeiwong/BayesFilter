"""XLA-native rank-one squared-TT/KR proposal for Austria SIR.

At the frozen reference parameter, the SIR transition covariance is identity
and the infectious-only observation covariance is ``100 I``. Conditional on a
fixed parent, the locally optimal proposal is therefore diagonal Gaussian. In
its Gaussian-quantile coordinates the density is constant, so it has an exact
rank-one squared-TT representation and an analytic triangular KR map.

The assembled auxiliary particle branch is a BayesFilter extension. Only the
squared-TT/KR proposal and importance-correction operations are Zhao-Cui
source-grounded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_tf import (
    AustriaSIRLatentPreclipFP64Model,
    AustriaSIRObservedDataTarget,
    EVENT_ORDER,
    TARGET_ID,
    make_austria_sir_observed_data_target,
    prepare_austria_sir_source_order_branch,
)
from bayesfilter.highdim.zhao_cui_predator_prey_fixed_variant_tf import (
    PreparedSourceOrderFrozenBranch,
)
from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
    SIR_DATASET_SEED,
    SIR_OBSERVATION_DIM,
    SIR_PARAMETER_DIM,
    SIR_STATE_DIM,
)


DTYPE = tf.float64
COMPILER_ROUTE_ID = "zhao_cui_austria_rank_one_gaussian_ttsirt_kr_compiler_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
PROPOSAL_OPERATION_CLASSIFICATION = "source_faithful_operation_only"
REFERENCE_THETA = (0.0, 0.0, 0.0)
_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), DTYPE)
_PREDICTIVE_VARIANCE = tf.constant(101.0, DTYPE)
_INFECTIOUS_POSTERIOR_VARIANCE = tf.constant(100.0 / 101.0, DTYPE)
_INFECTIOUS_POSTERIOR_SCALE = tf.sqrt(_INFECTIOUS_POSTERIOR_VARIANCE)
_UNIFORM_EPSILON = tf.constant(1e-12, DTYPE)


def _tensor_hash(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    return hashlib.sha256(bytes(serialized.numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def _standard_normal_from_uniform(uniforms: tf.Tensor) -> tf.Tensor:
    clipped = tf.clip_by_value(
        tf.convert_to_tensor(uniforms, DTYPE),
        _UNIFORM_EPSILON,
        1.0 - _UNIFORM_EPSILON,
    )
    return tf.sqrt(tf.constant(2.0, DTYPE)) * tf.math.erfinv(2.0 * clipped - 1.0)


def _standard_normal_log_density(standardized: tf.Tensor) -> tf.Tensor:
    return -0.5 * tf.reduce_sum(
        _LOG_TWO_PI + tf.square(standardized), axis=1
    )


@dataclass(frozen=True)
class AustriaSIRRankOneProposalCompilation:
    """Frozen branch plus the literal reference variables that generated it."""

    branch: PreparedSourceOrderFrozenBranch
    initial_reference_uniforms: tf.Tensor
    ancestor_uniforms: tf.Tensor
    transition_reference_uniforms: tf.Tensor
    compiler_id: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        initial = tf.convert_to_tensor(self.initial_reference_uniforms, DTYPE)
        ancestor = tf.convert_to_tensor(self.ancestor_uniforms, DTYPE)
        transition = tf.convert_to_tensor(self.transition_reference_uniforms, DTYPE)
        if initial.shape != (self.branch.particle_count, SIR_STATE_DIM):
            raise ValueError("initial reference uniforms have the wrong shape")
        if ancestor.shape != (
            self.branch.transition_count,
            self.branch.particle_count,
        ):
            raise ValueError("ancestor uniforms have the wrong shape")
        if transition.shape != (
            self.branch.transition_count,
            self.branch.particle_count,
            SIR_STATE_DIM,
        ):
            raise ValueError("transition reference uniforms have the wrong shape")
        for name, value in (
            ("initial_reference_uniforms", initial),
            ("ancestor_uniforms", ancestor),
            ("transition_reference_uniforms", transition),
        ):
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
            tf.debugging.assert_greater_equal(value, tf.zeros([], DTYPE))
            tf.debugging.assert_less(value, tf.ones([], DTYPE))
        if len(str(self.compiler_id)) != 64:
            raise ValueError("compiler_id must be a SHA-256 digest")
        object.__setattr__(self, "initial_reference_uniforms", initial)
        object.__setattr__(self, "ancestor_uniforms", ancestor)
        object.__setattr__(self, "transition_reference_uniforms", transition)
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class AustriaSIRRankOneMixtureProposalCompilation:
    """Frozen exactly scored mixture branch and all literal reference inputs."""

    branch: PreparedSourceOrderFrozenBranch
    guide_thetas: tf.Tensor
    initial_reference_uniforms: tf.Tensor
    ancestor_uniforms: tf.Tensor
    component_uniforms: tf.Tensor
    transition_reference_uniforms: tf.Tensor
    compiler_id: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        guides = tf.convert_to_tensor(self.guide_thetas, DTYPE)
        initial = tf.convert_to_tensor(self.initial_reference_uniforms, DTYPE)
        ancestor = tf.convert_to_tensor(self.ancestor_uniforms, DTYPE)
        component = tf.convert_to_tensor(self.component_uniforms, DTYPE)
        transition = tf.convert_to_tensor(self.transition_reference_uniforms, DTYPE)
        if guides.shape.rank != 2 or guides.shape[1] != SIR_PARAMETER_DIM:
            raise ValueError("guide_thetas must have shape [component,3]")
        if initial.shape != (self.branch.particle_count, SIR_STATE_DIM):
            raise ValueError("initial reference uniforms have the wrong shape")
        expected_selection_shape = (
            self.branch.transition_count,
            self.branch.particle_count,
        )
        if ancestor.shape != expected_selection_shape:
            raise ValueError("ancestor uniforms have the wrong shape")
        if component.shape != expected_selection_shape:
            raise ValueError("component uniforms have the wrong shape")
        if transition.shape != expected_selection_shape + (SIR_STATE_DIM,):
            raise ValueError("transition reference uniforms have the wrong shape")
        for value in (initial, ancestor, component, transition):
            tf.debugging.assert_all_finite(value, "reference inputs must be finite")
            tf.debugging.assert_greater_equal(value, tf.zeros([], DTYPE))
            tf.debugging.assert_less(value, tf.ones([], DTYPE))
        if len(str(self.compiler_id)) != 64:
            raise ValueError("compiler_id must be a SHA-256 digest")
        object.__setattr__(self, "guide_thetas", guides)
        object.__setattr__(self, "initial_reference_uniforms", initial)
        object.__setattr__(self, "ancestor_uniforms", ancestor)
        object.__setattr__(self, "component_uniforms", component)
        object.__setattr__(self, "transition_reference_uniforms", transition)
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class AustriaSIRPersistentGuideProgram:
    """Batched persistent-guide branches and their exact combined score."""

    observations: tf.Tensor
    guide_thetas: tf.Tensor
    states: tf.Tensor
    initial_log_proposal_density: tf.Tensor
    ancestors: tf.Tensor
    auxiliary_log_probabilities: tf.Tensor
    transition_log_proposal_density: tf.Tensor
    program_id: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        observations = tf.convert_to_tensor(self.observations, DTYPE)
        guides = tf.convert_to_tensor(self.guide_thetas, DTYPE)
        states = tf.convert_to_tensor(self.states, DTYPE)
        initial_log_q = tf.convert_to_tensor(
            self.initial_log_proposal_density, DTYPE
        )
        ancestors = tf.convert_to_tensor(self.ancestors, tf.int32)
        auxiliary = tf.convert_to_tensor(self.auxiliary_log_probabilities, DTYPE)
        transition_log_q = tf.convert_to_tensor(
            self.transition_log_proposal_density, DTYPE
        )
        if guides.shape.rank != 2 or guides.shape[1] != SIR_PARAMETER_DIM:
            raise ValueError("guide_thetas must have shape [guide,3]")
        guide_count = int(guides.shape[0])
        if states.shape.rank != 4 or states.shape[0] != guide_count:
            raise ValueError("states must have shape [guide,T+1,particle,state]")
        horizon = int(states.shape[1]) - 1
        particle_count = int(states.shape[2])
        if states.shape[3] != SIR_STATE_DIM:
            raise ValueError("persistent-guide state dimension rejected")
        if observations.shape != (horizon, SIR_OBSERVATION_DIM):
            raise ValueError("persistent-guide observations have the wrong shape")
        if initial_log_q.shape != (guide_count, particle_count):
            raise ValueError("initial proposal density has the wrong shape")
        expected = (guide_count, horizon, particle_count)
        if ancestors.shape != expected:
            raise ValueError("persistent-guide ancestors have the wrong shape")
        if auxiliary.shape != expected or transition_log_q.shape != expected:
            raise ValueError("persistent-guide proposal arrays have the wrong shape")
        for value in (observations, guides, states, initial_log_q, auxiliary, transition_log_q):
            tf.debugging.assert_all_finite(value, "persistent-guide tensors must be finite")
        if len(str(self.program_id)) != 64:
            raise ValueError("program_id must be a SHA-256 digest")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "guide_thetas", guides)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "initial_log_proposal_density", initial_log_q)
        object.__setattr__(self, "ancestors", ancestors)
        object.__setattr__(self, "auxiliary_log_probabilities", auxiliary)
        object.__setattr__(self, "transition_log_proposal_density", transition_log_q)
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def guide_count(self) -> int:
        return int(self.guide_thetas.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])

    @property
    def particle_count(self) -> int:
        return int(self.states.shape[2])

    def prefix(self, horizon: int) -> "AustriaSIRPersistentGuideProgram":
        """Issue a repository-owned prefix view of one longer frozen program."""

        steps = int(horizon)
        if steps < 1 or steps > self.horizon:
            raise ValueError("prefix horizon must be within the frozen program")
        if steps == self.horizon:
            return self
        observations = self.observations[:steps]
        states = self.states[:, : steps + 1]
        ancestors = self.ancestors[:, :steps]
        auxiliary = self.auxiliary_log_probabilities[:, :steps]
        transition_log_q = self.transition_log_proposal_density[:, :steps]
        identity_payload = {
            "prefix_route_id": (
                "zhao_cui_austria_persistent_guide_program_prefix_v1"
            ),
            "parent_program_id": self.program_id,
            "parent_horizon": self.horizon,
            "horizon": steps,
            "observations_sha256": _tensor_hash(observations),
            "guide_thetas_sha256": _tensor_hash(self.guide_thetas),
            "states_sha256": _tensor_hash(states),
            "initial_log_proposal_density_sha256": _tensor_hash(
                self.initial_log_proposal_density
            ),
            "ancestors_sha256": _tensor_hash(ancestors),
            "auxiliary_log_probabilities_sha256": _tensor_hash(auxiliary),
            "transition_log_proposal_density_sha256": _tensor_hash(
                transition_log_q
            ),
        }
        program_id = _semantic_hash(identity_payload)
        manifest = {
            **dict(self.manifest),
            **identity_payload,
            "program_id": program_id,
            "prefix_identity": "literal_tensor_prefix_of_parent_program",
        }
        return AustriaSIRPersistentGuideProgram(
            observations=observations,
            guide_thetas=self.guide_thetas,
            states=states,
            initial_log_proposal_density=self.initial_log_proposal_density,
            ancestors=ancestors,
            auxiliary_log_probabilities=auxiliary,
            transition_log_proposal_density=transition_log_q,
            program_id=program_id,
            manifest=manifest,
        )

    def compiled(
        self, *, jit_compile: bool = True
    ) -> Callable[[tf.Tensor], Mapping[str, tf.Tensor]]:
        model = AustriaSIRLatentPreclipFP64Model()
        observations = self.observations
        states = self.states
        initial_log_q = self.initial_log_proposal_density
        ancestors = self.ancestors
        auxiliary = self.auxiliary_log_probabilities
        transition_log_q = self.transition_log_proposal_density
        guide_count = self.guide_count
        horizon = self.horizon
        particle_count = self.particle_count
        log_particle_count = tf.math.log(tf.cast(particle_count, DTYPE))
        log_guide_count = tf.math.log(tf.cast(guide_count, DTYPE))

        @tf.function(
            input_signature=(tf.TensorSpec([SIR_PARAMETER_DIM], DTYPE),),
            jit_compile=bool(jit_compile),
            autograph=False,
        )
        def evaluate(theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
            flat_initial = tf.reshape(states[:, 0], [-1, SIR_STATE_DIM])
            initial_log_density = tf.reshape(
                model.initial_log_density(theta, flat_initial),
                [guide_count, particle_count],
            )
            initial_score = tf.reshape(
                model.initial_log_density_parameter_score(theta, flat_initial),
                [guide_count, particle_count, SIR_PARAMETER_DIM],
            )
            log_unnormalized = initial_log_density - initial_log_q
            log_sum = tf.reduce_logsumexp(log_unnormalized, axis=1)
            log_weights = log_unnormalized - log_sum[:, tf.newaxis]
            weights = tf.exp(log_weights)
            increment = log_sum - log_particle_count
            increment_score = tf.reduce_sum(
                weights[:, :, tf.newaxis] * initial_score, axis=1
            )
            derivative_log_weights = initial_score - increment_score[:, tf.newaxis, :]
            ess = tf.math.reciprocal(tf.reduce_sum(tf.square(weights), axis=1))
            maximum_weight = tf.reduce_max(weights, axis=1)
            finite = tf.reduce_all(
                tf.math.is_finite(log_unnormalized), axis=1
            ) & tf.reduce_all(tf.math.is_finite(initial_score), axis=(1, 2))
            branch_values = increment
            branch_scores = increment_score
            ess_values = tf.TensorArray(
                DTYPE,
                size=horizon + 1,
                clear_after_read=False,
                element_shape=tf.TensorShape([guide_count]),
            ).write(0, ess)
            maximum_weight_values = tf.TensorArray(
                DTYPE,
                size=horizon + 1,
                clear_after_read=False,
                element_shape=tf.TensorShape([guide_count]),
            ).write(0, maximum_weight)
            increment_values = tf.TensorArray(
                DTYPE,
                size=horizon + 1,
                clear_after_read=False,
                element_shape=tf.TensorShape([guide_count]),
            ).write(0, increment)
            increment_score_values = tf.TensorArray(
                DTYPE,
                size=horizon + 1,
                clear_after_read=False,
                element_shape=tf.TensorShape([guide_count, SIR_PARAMETER_DIM]),
            ).write(0, increment_score)

            def body(
                row: tf.Tensor,
                previous_log_weights: tf.Tensor,
                previous_marks: tf.Tensor,
                values: tf.Tensor,
                scores: tf.Tensor,
                all_finite: tf.Tensor,
                ess_array: tf.TensorArray,
                maximum_weight_array: tf.TensorArray,
                increment_array: tf.TensorArray,
                increment_score_array: tf.TensorArray,
            ) -> tuple[object, ...]:
                time_index = row + 1
                current_ancestors = tf.gather(ancestors, row, axis=1)
                previous_state = tf.gather(
                    tf.gather(states, row, axis=1),
                    current_ancestors,
                    axis=1,
                    batch_dims=1,
                )
                current_state = tf.gather(states, time_index, axis=1)
                selected_log_weights = tf.gather(
                    previous_log_weights,
                    current_ancestors,
                    axis=1,
                    batch_dims=1,
                )
                selected_marks = tf.gather(
                    previous_marks,
                    current_ancestors,
                    axis=1,
                    batch_dims=1,
                )
                selected_auxiliary = tf.gather(
                    tf.gather(auxiliary, row, axis=1),
                    current_ancestors,
                    axis=1,
                    batch_dims=1,
                )
                flat_previous = tf.reshape(previous_state, [-1, SIR_STATE_DIM])
                flat_current = tf.reshape(current_state, [-1, SIR_STATE_DIM])
                observation = tf.gather(observations, row)
                transition_density = tf.reshape(
                    model.transition_log_density(
                        theta, flat_previous, flat_current, time_index
                    ),
                    [guide_count, particle_count],
                )
                observation_density = tf.reshape(
                    model.observation_log_density(
                        theta, flat_current, observation, time_index
                    ),
                    [guide_count, particle_count],
                )
                transition_score = tf.reshape(
                    model.transition_log_density_parameter_score(
                        theta, flat_previous, flat_current, time_index
                    ),
                    [guide_count, particle_count, SIR_PARAMETER_DIM],
                )
                observation_score = tf.reshape(
                    model.observation_log_density_parameter_score(
                        theta, flat_current, observation, time_index
                    ),
                    [guide_count, particle_count, SIR_PARAMETER_DIM],
                )
                current_log_unnormalized = (
                    selected_log_weights
                    + transition_density
                    + observation_density
                    - selected_auxiliary
                    - tf.gather(transition_log_q, row, axis=1)
                )
                local_marks = selected_marks + transition_score + observation_score
                current_log_sum = tf.reduce_logsumexp(
                    current_log_unnormalized, axis=1
                )
                current_log_weights = (
                    current_log_unnormalized - current_log_sum[:, tf.newaxis]
                )
                current_weights = tf.exp(current_log_weights)
                current_increment = current_log_sum - log_particle_count
                current_increment_score = tf.reduce_sum(
                    current_weights[:, :, tf.newaxis] * local_marks, axis=1
                )
                current_marks = (
                    local_marks - current_increment_score[:, tf.newaxis, :]
                )
                current_ess = tf.math.reciprocal(
                    tf.reduce_sum(tf.square(current_weights), axis=1)
                )
                current_maximum_weight = tf.reduce_max(current_weights, axis=1)
                current_finite = tf.reduce_all(
                    tf.math.is_finite(current_log_unnormalized), axis=1
                ) & tf.reduce_all(tf.math.is_finite(local_marks), axis=(1, 2))
                return (
                    row + 1,
                    current_log_weights,
                    current_marks,
                    values + current_increment,
                    scores + current_increment_score,
                    all_finite & current_finite,
                    ess_array.write(time_index, current_ess),
                    maximum_weight_array.write(
                        time_index, current_maximum_weight
                    ),
                    increment_array.write(time_index, current_increment),
                    increment_score_array.write(
                        time_index, current_increment_score
                    ),
                )

            (
                _,
                _,
                _,
                branch_values,
                branch_scores,
                finite,
                ess_values,
                maximum_weight_values,
                increment_values,
                increment_score_values,
            ) = tf.while_loop(
                lambda row, *_unused: row < horizon,
                body,
                (
                    tf.zeros([], tf.int32),
                    log_weights,
                    derivative_log_weights,
                    branch_values,
                    branch_scores,
                    finite,
                    ess_values,
                    maximum_weight_values,
                    increment_values,
                    increment_score_values,
                ),
                maximum_iterations=horizon,
                parallel_iterations=1,
            )
            branch_combination_weights = tf.nn.softmax(branch_values)
            combined_value = tf.reduce_logsumexp(branch_values) - log_guide_count
            combined_score = tf.reduce_sum(
                branch_combination_weights[:, tf.newaxis] * branch_scores,
                axis=0,
            )
            branch_effective_count = tf.math.reciprocal(
                tf.reduce_sum(tf.square(branch_combination_weights))
            )
            return {
                "log_likelihood": combined_value,
                "score": combined_score,
                "branch_values": branch_values,
                "branch_scores": branch_scores,
                "branch_combination_weights": branch_combination_weights,
                "branch_effective_count": branch_effective_count,
                "ess_by_time_and_guide": ess_values.stack(),
                "maximum_weight_by_time_and_guide": maximum_weight_values.stack(),
                "branch_log_increments": increment_values.stack(),
                "branch_increment_scores": increment_score_values.stack(),
                "finite_by_guide": finite,
                "finite": tf.reduce_all(finite)
                & tf.math.is_finite(combined_value)
                & tf.reduce_all(tf.math.is_finite(combined_score)),
            }

        return evaluate


def symmetric_guide_grid(half_width: float = 0.03) -> tf.Tensor:
    """Return the fixed 27-point Cartesian guide grid without host loops."""

    radius = tf.convert_to_tensor(half_width, DTYPE)
    axis = tf.stack([-radius, tf.zeros([], DTYPE), radius])
    coordinates = tf.meshgrid(axis, axis, axis, indexing="ij")
    return tf.reshape(tf.stack(coordinates, axis=-1), [-1, SIR_PARAMETER_DIM])


def kappa_nu_guide_grid(half_width: float = 0.03) -> tf.Tensor:
    """Return the nine-mode guide grid with observation-noise guide fixed at zero."""

    radius = tf.convert_to_tensor(half_width, DTYPE)
    axis = tf.stack([-radius, tf.zeros([], DTYPE), radius])
    kappa, nu = tf.meshgrid(axis, axis, indexing="ij")
    return tf.reshape(
        tf.stack([kappa, nu, tf.zeros_like(kappa)], axis=-1),
        [-1, SIR_PARAMETER_DIM],
    )


def _transition_mean_grid(
    model: AustriaSIRLatentPreclipFP64Model,
    guide_thetas: tf.Tensor,
    previous_state: tf.Tensor,
    time_index: tf.Tensor,
) -> tf.Tensor:
    """Vectorized RK4 means for every guide and parent particle."""

    guides = tf.convert_to_tensor(guide_thetas, DTYPE)
    previous = tf.convert_to_tensor(previous_state, DTYPE)
    clipped = tf.reshape(
        tf.stack(
            [tf.maximum(previous[:, 0::2], 0.0), previous[:, 1::2]], axis=2
        ),
        [tf.shape(previous)[0], SIR_STATE_DIM],
    )
    initial = tf.where(tf.equal(time_index, 1), previous, clipped)
    state = tf.broadcast_to(
        initial[tf.newaxis, :, :],
        [tf.shape(guides)[0], tf.shape(previous)[0], SIR_STATE_DIM],
    )
    kappa = tf.constant(0.1, DTYPE) * tf.exp(guides[:, 0])
    nu = tf.constant(18.0, DTYPE) * tf.exp(guides[:, 1])
    adjacency = model._adjacency  # noqa: SLF001
    degree = model._degree  # noqa: SLF001
    step = tf.constant(0.005, DTYPE)

    def rhs(values: tf.Tensor) -> tf.Tensor:
        susceptible = values[:, :, 0::2]
        infectious = values[:, :, 1::2]
        susceptible_neighbor = (
            tf.einsum("cnj,kj->cnk", susceptible, adjacency)
            - susceptible * degree[tf.newaxis, tf.newaxis, :]
        )
        infectious_neighbor = (
            tf.einsum("cnj,kj->cnk", infectious, adjacency)
            - infectious * degree[tf.newaxis, tf.newaxis, :]
        )
        infection = (
            kappa[:, tf.newaxis, tf.newaxis] * susceptible * infectious
        )
        return tf.reshape(
            tf.stack(
                [
                    -infection + 0.5 * susceptible_neighbor,
                    infection
                    - nu[:, tf.newaxis, tf.newaxis] * infectious
                    + 0.5 * infectious_neighbor,
                ],
                axis=3,
            ),
            [tf.shape(guides)[0], tf.shape(previous)[0], SIR_STATE_DIM],
        )

    def body(index: tf.Tensor, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        k1 = rhs(values)
        k2 = rhs(values + 0.5 * step * k1)
        k3 = rhs(values + 0.5 * step * k2)
        k4 = rhs(values + 0.5 * step * k3)
        return index + 1, values + step / 6.0 * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    _, result = tf.while_loop(
        lambda index, *_unused: index < 4,
        body,
        (tf.zeros([], tf.int32), state),
        maximum_iterations=4,
        parallel_iterations=1,
    )
    return result


def _transition_mean_persistent_guides(
    model: AustriaSIRLatentPreclipFP64Model,
    guide_thetas: tf.Tensor,
    previous_state: tf.Tensor,
    time_index: tf.Tensor,
) -> tf.Tensor:
    """Paired RK4 means with shapes `[guide,particle,state]` and `[guide,3]`."""

    guides = tf.convert_to_tensor(guide_thetas, DTYPE)
    previous = tf.convert_to_tensor(previous_state, DTYPE)
    clipped = tf.reshape(
        tf.stack(
            [
                tf.maximum(previous[:, :, 0::2], 0.0),
                previous[:, :, 1::2],
            ],
            axis=3,
        ),
        [tf.shape(guides)[0], tf.shape(previous)[1], SIR_STATE_DIM],
    )
    state = tf.where(tf.equal(time_index, 1), previous, clipped)
    kappa = tf.constant(0.1, DTYPE) * tf.exp(guides[:, 0])
    nu = tf.constant(18.0, DTYPE) * tf.exp(guides[:, 1])
    adjacency = model._adjacency  # noqa: SLF001
    degree = model._degree  # noqa: SLF001
    step = tf.constant(0.005, DTYPE)

    def rhs(values: tf.Tensor) -> tf.Tensor:
        susceptible = values[:, :, 0::2]
        infectious = values[:, :, 1::2]
        susceptible_neighbor = (
            tf.einsum("cnj,kj->cnk", susceptible, adjacency)
            - susceptible * degree[tf.newaxis, tf.newaxis, :]
        )
        infectious_neighbor = (
            tf.einsum("cnj,kj->cnk", infectious, adjacency)
            - infectious * degree[tf.newaxis, tf.newaxis, :]
        )
        infection = (
            kappa[:, tf.newaxis, tf.newaxis] * susceptible * infectious
        )
        return tf.reshape(
            tf.stack(
                [
                    -infection + 0.5 * susceptible_neighbor,
                    infection
                    - nu[:, tf.newaxis, tf.newaxis] * infectious
                    + 0.5 * infectious_neighbor,
                ],
                axis=3,
            ),
            [tf.shape(guides)[0], tf.shape(previous)[1], SIR_STATE_DIM],
        )

    def body(index: tf.Tensor, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        k1 = rhs(values)
        k2 = rhs(values + 0.5 * step * k1)
        k3 = rhs(values + 0.5 * step * k2)
        k4 = rhs(values + 0.5 * step * k3)
        return index + 1, values + step / 6.0 * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    _, result = tf.while_loop(
        lambda index, *_unused: index < 4,
        body,
        (tf.zeros([], tf.int32), state),
        maximum_iterations=4,
        parallel_iterations=1,
    )
    return result


def make_rank_one_mixture_branch_tensor_compiler(
    *,
    particle_count: int,
    horizon: int,
    guide_thetas: tf.Tensor,
    proposal_standard_deviation_scale: float = 1.0,
) -> Callable[[tf.Tensor, tf.Tensor], Mapping[str, tf.Tensor]]:
    """Create an XLA compiler for an exactly scored analytic guide mixture."""

    particles = int(particle_count)
    steps = int(horizon)
    guides = tf.convert_to_tensor(guide_thetas, DTYPE)
    if guides.shape.rank != 2 or guides.shape[1] != SIR_PARAMETER_DIM:
        raise ValueError("guide_thetas must have static shape [component,3]")
    component_count = int(guides.shape[0])
    proposal_scale = tf.convert_to_tensor(
        proposal_standard_deviation_scale, DTYPE
    )
    if proposal_scale.shape.rank != 0 or float(proposal_standard_deviation_scale) < 1.0:
        raise ValueError("proposal_standard_deviation_scale must be at least one")
    proposal_variance_scale = tf.square(proposal_scale)
    if particles < 2 or component_count < 1:
        raise ValueError("particles and guide components must be positive")
    if steps < 1 or steps > 20:
        raise ValueError("horizon must be in [1,20]")
    model = AustriaSIRLatentPreclipFP64Model()
    log_component_prior = -tf.math.log(tf.cast(component_count, DTYPE))
    log_particle_count = tf.math.log(tf.cast(particles, DTYPE))

    @tf.function(
        input_signature=(
            tf.TensorSpec([steps, SIR_OBSERVATION_DIM], DTYPE),
            tf.TensorSpec([2], tf.int32),
        ),
        jit_compile=True,
        autograph=False,
    )
    def compile_tensors(
        observations: tf.Tensor, seed: tf.Tensor
    ) -> Mapping[str, tf.Tensor]:
        roots = tf.random.experimental.stateless_split(seed, 4)
        initial_uniforms = tf.random.stateless_uniform(
            [particles, SIR_STATE_DIM],
            roots[0],
            minval=_UNIFORM_EPSILON,
            maxval=1.0 - _UNIFORM_EPSILON,
            dtype=DTYPE,
        )
        ancestor_uniforms = tf.random.stateless_uniform(
            [steps, particles], roots[1], dtype=DTYPE
        )
        component_uniforms = tf.random.stateless_uniform(
            [steps, particles], roots[2], dtype=DTYPE
        )
        transition_uniforms = tf.random.stateless_uniform(
            [steps, particles, SIR_STATE_DIM],
            roots[3],
            minval=_UNIFORM_EPSILON,
            maxval=1.0 - _UNIFORM_EPSILON,
            dtype=DTYPE,
        )
        initial_standard = _standard_normal_from_uniform(initial_uniforms)
        initial_state = model._initial_mean[tf.newaxis, :] + initial_standard  # noqa: SLF001
        initial_log_q = _standard_normal_log_density(initial_standard)
        previous_guide_log_weights = tf.fill(
            [component_count, particles], -log_particle_count
        )

        states = tf.TensorArray(
            DTYPE,
            size=steps + 1,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles, SIR_STATE_DIM]),
        ).write(0, initial_state)
        ancestors = tf.TensorArray(
            tf.int32,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles]),
        )
        selected_components = tf.TensorArray(
            tf.int32,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles]),
        )
        auxiliary_log = tf.TensorArray(
            DTYPE,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles]),
        )
        transition_log_q = tf.TensorArray(
            DTYPE,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles]),
        )

        def body(
            row: tf.Tensor,
            previous_state: tf.Tensor,
            guide_log_weights: tf.Tensor,
            state_values: tf.TensorArray,
            ancestor_values: tf.TensorArray,
            component_values: tf.TensorArray,
            auxiliary_values: tf.TensorArray,
            log_q_values: tf.TensorArray,
        ) -> tuple[object, ...]:
            time_index = row + 1
            means = _transition_mean_grid(
                model, guides, previous_state, time_index
            )
            observation = tf.gather(observations, row)
            observation_variance = 100.0 * tf.exp(2.0 * guides[:, 2])
            predictive_variance = observation_variance + 1.0
            predictive_residual = (
                observation[tf.newaxis, tf.newaxis, :] - means[:, :, 1::2]
            )
            predictive_log_density = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI
                + tf.math.log(predictive_variance)[:, tf.newaxis, tf.newaxis]
                + tf.square(predictive_residual)
                / predictive_variance[:, tf.newaxis, tf.newaxis],
                axis=2,
            )
            joint_logits = (
                log_component_prior + guide_log_weights + predictive_log_density
            )
            normalization = tf.reduce_logsumexp(joint_logits)
            current_auxiliary_log = (
                tf.reduce_logsumexp(joint_logits, axis=0) - normalization
            )
            ancestor_cdf = tf.cumsum(tf.exp(current_auxiliary_log))
            current_ancestors = tf.reduce_sum(
                tf.cast(
                    tf.gather(ancestor_uniforms, row)[:, tf.newaxis]
                    > ancestor_cdf[tf.newaxis, :],
                    tf.int32,
                ),
                axis=1,
            )
            current_ancestors = tf.minimum(current_ancestors, particles - 1)
            selected_joint_logits = tf.gather(
                joint_logits, current_ancestors, axis=1
            )
            conditional_component_log = selected_joint_logits - tf.reduce_logsumexp(
                selected_joint_logits, axis=0, keepdims=True
            )
            component_cdf = tf.cumsum(tf.exp(conditional_component_log), axis=0)
            current_components = tf.reduce_sum(
                tf.cast(
                    tf.gather(component_uniforms, row)[tf.newaxis, :]
                    > component_cdf,
                    tf.int32,
                ),
                axis=0,
            )
            current_components = tf.minimum(
                current_components, component_count - 1
            )
            parent_means = tf.gather(means, current_ancestors, axis=1)
            sample_indices = tf.range(particles, dtype=tf.int32)
            selected_mean = tf.gather_nd(
                parent_means, tf.stack([current_components, sample_indices], axis=1)
            )
            selected_observation_variance = tf.gather(
                observation_variance, current_components
            )
            selected_posterior_variance = (
                selected_observation_variance
                / (selected_observation_variance + 1.0)
            )
            standard = _standard_normal_from_uniform(
                tf.gather(transition_uniforms, row)
            )
            susceptible = (
                selected_mean[:, 0::2] + proposal_scale * standard[:, 0::2]
            )
            infectious_location = (
                selected_observation_variance[:, tf.newaxis]
                * selected_mean[:, 1::2]
                + observation[tf.newaxis, :]
            ) / (selected_observation_variance[:, tf.newaxis] + 1.0)
            infectious = infectious_location + tf.sqrt(
                selected_posterior_variance * proposal_variance_scale
            )[:, tf.newaxis] * standard[:, 1::2]
            current_state = tf.reshape(
                tf.stack([susceptible, infectious], axis=2),
                [particles, SIR_STATE_DIM],
            )

            all_observation_variance = observation_variance[:, tf.newaxis]
            all_posterior_variance = (
                all_observation_variance / (all_observation_variance + 1.0)
            )
            all_infectious_location = (
                all_observation_variance[:, :, tf.newaxis]
                * parent_means[:, :, 1::2]
                + observation[tf.newaxis, tf.newaxis, :]
            ) / (all_observation_variance[:, :, tf.newaxis] + 1.0)
            susceptible_standard = (
                current_state[tf.newaxis, :, 0::2] - parent_means[:, :, 0::2]
            ) / proposal_scale
            infectious_standard = (
                current_state[tf.newaxis, :, 1::2] - all_infectious_location
            ) / tf.sqrt(
                all_posterior_variance * proposal_variance_scale
            )[:, :, tf.newaxis]
            component_log_q = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI
                + tf.math.log(proposal_variance_scale)
                + tf.square(susceptible_standard),
                axis=2,
            ) - 0.5 * tf.reduce_sum(
                _LOG_TWO_PI
                + tf.math.log(
                    all_posterior_variance * proposal_variance_scale
                )[:, :, tf.newaxis]
                + tf.square(infectious_standard),
                axis=2,
            )
            current_log_q = tf.reduce_logsumexp(
                conditional_component_log + component_log_q, axis=0
            )

            selected_previous_weights = tf.gather(
                guide_log_weights,
                tf.broadcast_to(
                    current_ancestors[tf.newaxis, :],
                    [component_count, particles],
                ),
                axis=1,
                batch_dims=1,
            )
            selected_auxiliary = tf.gather(
                current_auxiliary_log, current_ancestors
            )
            transition_residual = current_state[tf.newaxis, :, :] - parent_means
            transition_log_density = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI + tf.square(transition_residual), axis=2
            )
            observation_residual = (
                observation[tf.newaxis, tf.newaxis, :]
                - current_state[tf.newaxis, :, 1::2]
            )
            observation_log_density = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI
                + tf.math.log(observation_variance)[:, tf.newaxis, tf.newaxis]
                + tf.square(observation_residual)
                / observation_variance[:, tf.newaxis, tf.newaxis],
                axis=2,
            )
            guide_log_unnormalized = (
                selected_previous_weights
                + transition_log_density
                + observation_log_density
                - selected_auxiliary[tf.newaxis, :]
                - current_log_q[tf.newaxis, :]
            )
            next_guide_log_weights = guide_log_unnormalized - tf.reduce_logsumexp(
                guide_log_unnormalized, axis=1, keepdims=True
            )
            return (
                row + 1,
                current_state,
                next_guide_log_weights,
                state_values.write(time_index, current_state),
                ancestor_values.write(row, current_ancestors),
                component_values.write(row, current_components),
                auxiliary_values.write(row, current_auxiliary_log),
                log_q_values.write(row, current_log_q),
            )

        (
            _,
            _,
            final_guide_log_weights,
            states,
            ancestors,
            selected_components,
            auxiliary_log,
            transition_log_q,
        ) = tf.while_loop(
            lambda row, *_unused: row < steps,
            body,
            (
                tf.zeros([], tf.int32),
                initial_state,
                previous_guide_log_weights,
                states,
                ancestors,
                selected_components,
                auxiliary_log,
                transition_log_q,
            ),
            maximum_iterations=steps,
            parallel_iterations=1,
        )
        return {
            "states": states.stack(),
            "initial_log_proposal_density": initial_log_q,
            "ancestors": ancestors.stack(),
            "selected_components": selected_components.stack(),
            "auxiliary_log_probabilities": auxiliary_log.stack(),
            "transition_log_proposal_density": transition_log_q.stack(),
            "final_guide_log_weights": final_guide_log_weights,
            "initial_reference_uniforms": initial_uniforms,
            "ancestor_uniforms": ancestor_uniforms,
            "component_uniforms": component_uniforms,
            "transition_reference_uniforms": transition_uniforms,
        }

    return compile_tensors


def make_persistent_guide_tensor_compiler(
    *, particle_count: int, horizon: int, guide_thetas: tf.Tensor
) -> Callable[[tf.Tensor, tf.Tensor], Mapping[str, tf.Tensor]]:
    """Create one XLA compiler for independent persistent guide branches."""

    particles = int(particle_count)
    steps = int(horizon)
    guides = tf.convert_to_tensor(guide_thetas, DTYPE)
    if guides.shape.rank != 2 or guides.shape[1] != SIR_PARAMETER_DIM:
        raise ValueError("guide_thetas must have shape [guide,3]")
    guide_count = int(guides.shape[0])
    if particles < 2 or guide_count < 1 or steps < 1 or steps > 20:
        raise ValueError("invalid persistent-guide scope")
    model = AustriaSIRLatentPreclipFP64Model()
    log_particle_count = tf.math.log(tf.cast(particles, DTYPE))

    @tf.function(
        input_signature=(
            tf.TensorSpec([steps, SIR_OBSERVATION_DIM], DTYPE),
            tf.TensorSpec([2], tf.int32),
        ),
        jit_compile=True,
        autograph=False,
    )
    def compile_tensors(
        observations: tf.Tensor, seed: tf.Tensor
    ) -> Mapping[str, tf.Tensor]:
        roots = tf.random.experimental.stateless_split(seed, 3)
        initial_uniforms = tf.random.stateless_uniform(
            [guide_count, particles, SIR_STATE_DIM],
            roots[0],
            minval=_UNIFORM_EPSILON,
            maxval=1.0 - _UNIFORM_EPSILON,
            dtype=DTYPE,
        )
        # Time-major generation makes every shorter horizon a literal prefix of
        # the same stateless stream before guide-major runtime indexing.
        ancestor_uniforms = tf.transpose(
            tf.random.stateless_uniform(
                [steps, guide_count, particles], roots[1], dtype=DTYPE
            ),
            [1, 0, 2],
        )
        transition_uniforms = tf.transpose(
            tf.random.stateless_uniform(
                [steps, guide_count, particles, SIR_STATE_DIM],
                roots[2],
                minval=_UNIFORM_EPSILON,
                maxval=1.0 - _UNIFORM_EPSILON,
                dtype=DTYPE,
            ),
            [1, 0, 2, 3],
        )
        initial_standard = _standard_normal_from_uniform(initial_uniforms)
        initial_state = (
            model._initial_mean[tf.newaxis, tf.newaxis, :]  # noqa: SLF001
            + initial_standard
        )
        initial_log_q = -0.5 * tf.reduce_sum(
            _LOG_TWO_PI + tf.square(initial_standard), axis=2
        )
        previous_guide_log_weights = tf.fill(
            [guide_count, particles], -log_particle_count
        )
        states = tf.TensorArray(
            DTYPE,
            size=steps + 1,
            clear_after_read=False,
            element_shape=tf.TensorShape(
                [guide_count, particles, SIR_STATE_DIM]
            ),
        ).write(0, initial_state)
        ancestors = tf.TensorArray(
            tf.int32,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([guide_count, particles]),
        )
        auxiliary_log = tf.TensorArray(
            DTYPE,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([guide_count, particles]),
        )
        transition_log_q = tf.TensorArray(
            DTYPE,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([guide_count, particles]),
        )

        def body(
            row: tf.Tensor,
            previous_state: tf.Tensor,
            guide_log_weights: tf.Tensor,
            state_values: tf.TensorArray,
            ancestor_values: tf.TensorArray,
            auxiliary_values: tf.TensorArray,
            log_q_values: tf.TensorArray,
        ) -> tuple[object, ...]:
            time_index = row + 1
            means = _transition_mean_persistent_guides(
                model, guides, previous_state, time_index
            )
            observation = tf.gather(observations, row)
            observation_variance = 100.0 * tf.exp(2.0 * guides[:, 2])
            predictive_variance = observation_variance + 1.0
            predictive_residual = (
                observation[tf.newaxis, tf.newaxis, :] - means[:, :, 1::2]
            )
            predictive_log_density = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI
                + tf.math.log(predictive_variance)[:, tf.newaxis, tf.newaxis]
                + tf.square(predictive_residual)
                / predictive_variance[:, tf.newaxis, tf.newaxis],
                axis=2,
            )
            current_auxiliary_log = tf.nn.log_softmax(
                guide_log_weights + predictive_log_density, axis=1
            )
            ancestor_cdf = tf.cumsum(
                tf.exp(current_auxiliary_log), axis=1
            )
            uniforms = tf.gather(ancestor_uniforms, row, axis=1)
            current_ancestors = tf.reduce_sum(
                tf.cast(
                    uniforms[:, :, tf.newaxis]
                    > ancestor_cdf[:, tf.newaxis, :],
                    tf.int32,
                ),
                axis=2,
            )
            current_ancestors = tf.minimum(current_ancestors, particles - 1)
            parent_mean = tf.gather(
                means, current_ancestors, axis=1, batch_dims=1
            )
            standard = _standard_normal_from_uniform(
                tf.gather(transition_uniforms, row, axis=1)
            )
            posterior_variance = (
                observation_variance / (observation_variance + 1.0)
            )
            susceptible = parent_mean[:, :, 0::2] + standard[:, :, 0::2]
            infectious_location = (
                observation_variance[:, tf.newaxis, tf.newaxis]
                * parent_mean[:, :, 1::2]
                + observation[tf.newaxis, tf.newaxis, :]
            ) / (observation_variance[:, tf.newaxis, tf.newaxis] + 1.0)
            infectious = infectious_location + tf.sqrt(
                posterior_variance
            )[:, tf.newaxis, tf.newaxis] * standard[:, :, 1::2]
            current_state = tf.reshape(
                tf.stack([susceptible, infectious], axis=3),
                [guide_count, particles, SIR_STATE_DIM],
            )
            current_log_q = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI + tf.square(standard[:, :, 0::2]), axis=2
            ) - 0.5 * tf.reduce_sum(
                _LOG_TWO_PI
                + tf.math.log(posterior_variance)[:, tf.newaxis, tf.newaxis]
                + tf.square(standard[:, :, 1::2]),
                axis=2,
            )
            selected_previous_weights = tf.gather(
                guide_log_weights,
                current_ancestors,
                axis=1,
                batch_dims=1,
            )
            selected_auxiliary = tf.gather(
                current_auxiliary_log,
                current_ancestors,
                axis=1,
                batch_dims=1,
            )
            transition_residual = current_state - parent_mean
            transition_log_density = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI + tf.square(transition_residual), axis=2
            )
            observation_residual = (
                observation[tf.newaxis, tf.newaxis, :]
                - current_state[:, :, 1::2]
            )
            observation_log_density = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI
                + tf.math.log(observation_variance)[:, tf.newaxis, tf.newaxis]
                + tf.square(observation_residual)
                / observation_variance[:, tf.newaxis, tf.newaxis],
                axis=2,
            )
            guide_log_unnormalized = (
                selected_previous_weights
                + transition_log_density
                + observation_log_density
                - selected_auxiliary
                - current_log_q
            )
            next_guide_log_weights = guide_log_unnormalized - tf.reduce_logsumexp(
                guide_log_unnormalized, axis=1, keepdims=True
            )
            return (
                row + 1,
                current_state,
                next_guide_log_weights,
                state_values.write(time_index, current_state),
                ancestor_values.write(row, current_ancestors),
                auxiliary_values.write(row, current_auxiliary_log),
                log_q_values.write(row, current_log_q),
            )

        (
            _,
            _,
            final_guide_log_weights,
            states,
            ancestors,
            auxiliary_log,
            transition_log_q,
        ) = tf.while_loop(
            lambda row, *_unused: row < steps,
            body,
            (
                tf.zeros([], tf.int32),
                initial_state,
                previous_guide_log_weights,
                states,
                ancestors,
                auxiliary_log,
                transition_log_q,
            ),
            maximum_iterations=steps,
            parallel_iterations=1,
        )
        return {
            "states": tf.transpose(states.stack(), [1, 0, 2, 3]),
            "initial_log_proposal_density": initial_log_q,
            "ancestors": tf.transpose(ancestors.stack(), [1, 0, 2]),
            "auxiliary_log_probabilities": tf.transpose(
                auxiliary_log.stack(), [1, 0, 2]
            ),
            "transition_log_proposal_density": tf.transpose(
                transition_log_q.stack(), [1, 0, 2]
            ),
            "final_guide_log_weights": final_guide_log_weights,
            "initial_reference_uniforms": initial_uniforms,
            "ancestor_uniforms": ancestor_uniforms,
            "transition_reference_uniforms": transition_uniforms,
        }

    return compile_tensors


def make_rank_one_branch_tensor_compiler(
    *, particle_count: int, horizon: int
) -> Callable[[tf.Tensor, tf.Tensor], Mapping[str, tf.Tensor]]:
    """Create the static-shape XLA branch compiler for one proposal scope."""

    particles = int(particle_count)
    steps = int(horizon)
    if particles < 2:
        raise ValueError("particle_count must be at least two")
    if steps < 1 or steps > 20:
        raise ValueError("horizon must be in [1,20]")
    model = AustriaSIRLatentPreclipFP64Model()
    theta = tf.zeros([SIR_PARAMETER_DIM], DTYPE)
    uniform_log_weight = -tf.math.log(tf.cast(particles, DTYPE))

    @tf.function(
        input_signature=(
            tf.TensorSpec([steps, SIR_OBSERVATION_DIM], DTYPE),
            tf.TensorSpec([2], tf.int32),
        ),
        jit_compile=True,
        autograph=False,
    )
    def compile_tensors(
        observations: tf.Tensor, seed: tf.Tensor
    ) -> Mapping[str, tf.Tensor]:
        roots = tf.random.experimental.stateless_split(seed, 3)
        initial_uniforms = tf.random.stateless_uniform(
            [particles, SIR_STATE_DIM],
            roots[0],
            minval=_UNIFORM_EPSILON,
            maxval=1.0 - _UNIFORM_EPSILON,
            dtype=DTYPE,
        )
        ancestor_uniforms = tf.random.stateless_uniform(
            [steps, particles],
            roots[1],
            minval=tf.zeros([], DTYPE),
            maxval=tf.ones([], DTYPE),
            dtype=DTYPE,
        )
        transition_uniforms = tf.random.stateless_uniform(
            [steps, particles, SIR_STATE_DIM],
            roots[2],
            minval=_UNIFORM_EPSILON,
            maxval=1.0 - _UNIFORM_EPSILON,
            dtype=DTYPE,
        )

        initial_standard = _standard_normal_from_uniform(initial_uniforms)
        initial_state = model._initial_mean[tf.newaxis, :] + initial_standard  # noqa: SLF001
        initial_log_q = _standard_normal_log_density(initial_standard)
        states = tf.TensorArray(
            DTYPE,
            size=steps + 1,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles, SIR_STATE_DIM]),
        ).write(0, initial_state)
        ancestors = tf.TensorArray(
            tf.int32,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles]),
        )
        auxiliary_log = tf.TensorArray(
            DTYPE,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles]),
        )
        transition_log_q = tf.TensorArray(
            DTYPE,
            size=steps,
            clear_after_read=False,
            element_shape=tf.TensorShape([particles]),
        )

        def body(
            row: tf.Tensor,
            previous_state: tf.Tensor,
            state_values: tf.TensorArray,
            ancestor_values: tf.TensorArray,
            auxiliary_values: tf.TensorArray,
            log_q_values: tf.TensorArray,
        ) -> tuple[object, ...]:
            time_index = row + 1
            transition_mean = model.transition_mean(
                theta, previous_state, time_index
            )
            observation = tf.gather(observations, row)
            predictive_residual = (
                observation[tf.newaxis, :] - transition_mean[:, 1::2]
            )
            predictive_log_density = -0.5 * tf.reduce_sum(
                _LOG_TWO_PI
                + tf.math.log(_PREDICTIVE_VARIANCE)
                + tf.square(predictive_residual) / _PREDICTIVE_VARIANCE,
                axis=1,
            )
            current_auxiliary_log = tf.nn.log_softmax(
                predictive_log_density + uniform_log_weight
            )
            cdf = tf.cumsum(tf.exp(current_auxiliary_log))
            uniforms = tf.gather(ancestor_uniforms, row)
            current_ancestors = tf.reduce_sum(
                tf.cast(uniforms[:, tf.newaxis] > cdf[tf.newaxis, :], tf.int32),
                axis=1,
            )
            current_ancestors = tf.minimum(current_ancestors, particles - 1)
            parent_mean = tf.gather(transition_mean, current_ancestors)
            standard = _standard_normal_from_uniform(
                tf.gather(transition_uniforms, row)
            )
            susceptible = parent_mean[:, 0::2] + standard[:, 0::2]
            infectious_location = (
                100.0 * parent_mean[:, 1::2] + observation[tf.newaxis, :]
            ) / 101.0
            infectious = (
                infectious_location
                + _INFECTIOUS_POSTERIOR_SCALE * standard[:, 1::2]
            )
            current_state = tf.reshape(
                tf.stack([susceptible, infectious], axis=2),
                [particles, SIR_STATE_DIM],
            )
            current_log_q = _standard_normal_log_density(standard) - (
                tf.cast(SIR_OBSERVATION_DIM, DTYPE)
                * tf.math.log(_INFECTIOUS_POSTERIOR_SCALE)
            )
            return (
                row + 1,
                current_state,
                state_values.write(time_index, current_state),
                ancestor_values.write(row, current_ancestors),
                auxiliary_values.write(row, current_auxiliary_log),
                log_q_values.write(row, current_log_q),
            )

        (
            _,
            _,
            states,
            ancestors,
            auxiliary_log,
            transition_log_q,
        ) = tf.while_loop(
            lambda row, *_unused: row < steps,
            body,
            (
                tf.zeros([], tf.int32),
                initial_state,
                states,
                ancestors,
                auxiliary_log,
                transition_log_q,
            ),
            maximum_iterations=steps,
            parallel_iterations=1,
        )
        return {
            "states": states.stack(),
            "initial_log_proposal_density": initial_log_q,
            "ancestors": ancestors.stack(),
            "auxiliary_log_probabilities": auxiliary_log.stack(),
            "transition_log_proposal_density": transition_log_q.stack(),
            "initial_reference_uniforms": initial_uniforms,
            "ancestor_uniforms": ancestor_uniforms,
            "transition_reference_uniforms": transition_uniforms,
        }

    return compile_tensors


def compile_austria_sir_rank_one_proposal_branch(
    *,
    particle_count: int,
    horizon: int,
    seed: int,
    target: AustriaSIRObservedDataTarget | None = None,
    require_claim_scope: bool = False,
) -> AustriaSIRRankOneProposalCompilation:
    """Compile one literal theta-independent proposal branch on GPU/XLA."""

    target = target or make_austria_sir_observed_data_target()
    observations = target.source_observations[: int(horizon)]
    compiler = make_rank_one_branch_tensor_compiler(
        particle_count=int(particle_count), horizon=int(horizon)
    )
    tensors = compiler(observations, tf.constant([int(seed), 901], tf.int32))
    identity_payload = {
        "compiler_route_id": COMPILER_ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "proposal_operation_classification": PROPOSAL_OPERATION_CLASSIFICATION,
        "target_id": TARGET_ID,
        "event_order": EVENT_ORDER,
        "target_seed": SIR_DATASET_SEED,
        "horizon": int(horizon),
        "particle_count": int(particle_count),
        "seed": int(seed),
        "dtype": "float64",
        "reference_theta": REFERENCE_THETA,
        "rank": 1,
        "kr_map": "analytic_diagonal_gaussian_quantile",
        "initial_reference_uniforms_sha256": _tensor_hash(
            tensors["initial_reference_uniforms"]
        ),
        "ancestor_uniforms_sha256": _tensor_hash(tensors["ancestor_uniforms"]),
        "transition_reference_uniforms_sha256": _tensor_hash(
            tensors["transition_reference_uniforms"]
        ),
    }
    compiler_id = _semantic_hash(identity_payload)
    branch = prepare_austria_sir_source_order_branch(
        target=target,
        observations=observations,
        states=tensors["states"],
        initial_log_proposal_density=tensors["initial_log_proposal_density"],
        ancestors=tensors["ancestors"],
        auxiliary_log_probabilities=tensors["auxiliary_log_probabilities"],
        transition_log_proposal_density=tensors[
            "transition_log_proposal_density"
        ],
        proposal_compiler_id=compiler_id,
        require_claim_scope=require_claim_scope,
    )
    manifest = {
        **identity_payload,
        "compiler_id": compiler_id,
        "branch_id": branch.branch_id,
        "initial_proposal": "exact_p_x0",
        "transition_proposal": "exact_q_ref_x_t_given_parent_y_t",
        "auxiliary_law": "softmax(log_W_ref_plus_log_predictive_y_given_parent)",
        "susceptible_conditional_variance": 1.0,
        "infectious_conditional_variance": 100.0 / 101.0,
        "predictive_observation_variance": 101.0,
        "xla_jit_compile": True,
        "python_numerical_loop": False,
        "numpy_numerical_path": False,
        "paper_anchor": (
            ".localresources/papers/"
            "zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:807-924"
        ),
        "author_source_anchor": (
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/"
            "models/full_sol.m:21-43"
        ),
        "nonclaims": (
            "assembled proposal is not source-faithful Austria parameter inference",
            "rank-one origin optimality does not prove off-origin or T20 viability",
            "no exact physical-likelihood, HMC, posterior, default, production, or superiority claim",
        ),
    }
    return AustriaSIRRankOneProposalCompilation(
        branch=branch,
        initial_reference_uniforms=tensors["initial_reference_uniforms"],
        ancestor_uniforms=tensors["ancestor_uniforms"],
        transition_reference_uniforms=tensors["transition_reference_uniforms"],
        compiler_id=compiler_id,
        manifest=manifest,
    )


def compile_austria_sir_rank_one_mixture_proposal_branch(
    *,
    particle_count: int,
    horizon: int,
    seed: int,
    guide_half_width: float = 0.03,
    guide_family: str = "full_cartesian_27",
    proposal_standard_deviation_scale: float = 1.0,
    target: AustriaSIRObservedDataTarget | None = None,
    require_claim_scope: bool = False,
) -> AustriaSIRRankOneMixtureProposalCompilation:
    """Compile the fixed 27-component analytic mixture proposal on GPU/XLA."""

    target = target or make_austria_sir_observed_data_target()
    if guide_family == "full_cartesian_27":
        guides = symmetric_guide_grid(guide_half_width)
    elif guide_family == "kappa_nu_cartesian_9":
        guides = kappa_nu_guide_grid(guide_half_width)
    else:
        raise ValueError("unsupported guide_family")
    observations = target.source_observations[: int(horizon)]
    compiler = make_rank_one_mixture_branch_tensor_compiler(
        particle_count=int(particle_count),
        horizon=int(horizon),
        guide_thetas=guides,
        proposal_standard_deviation_scale=proposal_standard_deviation_scale,
    )
    tensors = compiler(observations, tf.constant([int(seed), 1901], tf.int32))
    identity_payload = {
        "compiler_route_id": (
            "zhao_cui_austria_rank_one_gaussian_ttsirt_kr_mixture_compiler_v1"
        ),
        "route_classification": ROUTE_CLASSIFICATION,
        "proposal_operation_classification": PROPOSAL_OPERATION_CLASSIFICATION,
        "target_id": TARGET_ID,
        "event_order": EVENT_ORDER,
        "target_seed": SIR_DATASET_SEED,
        "horizon": int(horizon),
        "particle_count": int(particle_count),
        "seed": int(seed),
        "dtype": "float64",
        "guide_half_width": float(guide_half_width),
        "guide_family": str(guide_family),
        "guide_count": int(guides.shape[0]),
        "proposal_standard_deviation_scale": float(
            proposal_standard_deviation_scale
        ),
        "guide_thetas_sha256": _tensor_hash(guides),
        "rank_per_component": 1,
        "kr_map": "analytic_diagonal_gaussian_quantile",
        "mixture_density": "exact_logsumexp_of_all_component_conditionals",
        "initial_reference_uniforms_sha256": _tensor_hash(
            tensors["initial_reference_uniforms"]
        ),
        "ancestor_uniforms_sha256": _tensor_hash(tensors["ancestor_uniforms"]),
        "component_uniforms_sha256": _tensor_hash(
            tensors["component_uniforms"]
        ),
        "transition_reference_uniforms_sha256": _tensor_hash(
            tensors["transition_reference_uniforms"]
        ),
    }
    compiler_id = _semantic_hash(identity_payload)
    branch = prepare_austria_sir_source_order_branch(
        target=target,
        observations=observations,
        states=tensors["states"],
        initial_log_proposal_density=tensors["initial_log_proposal_density"],
        ancestors=tensors["ancestors"],
        auxiliary_log_probabilities=tensors["auxiliary_log_probabilities"],
        transition_log_proposal_density=tensors[
            "transition_log_proposal_density"
        ],
        proposal_compiler_id=compiler_id,
        require_claim_scope=require_claim_scope,
    )
    manifest = {
        **identity_payload,
        "compiler_id": compiler_id,
        "branch_id": branch.branch_id,
        "selected_components_sha256": _tensor_hash(
            tensors["selected_components"]
        ),
        "final_guide_log_weights_sha256": _tensor_hash(
            tensors["final_guide_log_weights"]
        ),
        "xla_jit_compile": True,
        "python_numerical_loop": False,
        "numpy_numerical_path": False,
        "paper_anchor": (
            ".localresources/papers/"
            "zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:807-924"
        ),
        "author_source_anchor": (
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/"
            "models/full_sol.m:21-43"
        ),
        "nonclaims": (
            "assembled guide mixture is not in the Zhao-Cui Austria example",
            "fixed domain-spanning mixture is an extension_or_invention",
            "no exact physical-likelihood, HMC, posterior, default, production, or superiority claim",
        ),
    }
    return AustriaSIRRankOneMixtureProposalCompilation(
        branch=branch,
        guide_thetas=guides,
        initial_reference_uniforms=tensors["initial_reference_uniforms"],
        ancestor_uniforms=tensors["ancestor_uniforms"],
        component_uniforms=tensors["component_uniforms"],
        transition_reference_uniforms=tensors["transition_reference_uniforms"],
        compiler_id=compiler_id,
        manifest=manifest,
    )


def compile_austria_sir_persistent_guide_program(
    *,
    particle_count: int,
    horizon: int,
    seed: int,
    guide_half_width: float = 0.03,
    target: AustriaSIRObservedDataTarget | None = None,
) -> AustriaSIRPersistentGuideProgram:
    """Compile the fixed nine-branch kappa/nu guide family on GPU/XLA."""

    target = target or make_austria_sir_observed_data_target()
    guides = kappa_nu_guide_grid(guide_half_width)
    observations = target.source_observations[: int(horizon)]
    compiler = make_persistent_guide_tensor_compiler(
        particle_count=int(particle_count),
        horizon=int(horizon),
        guide_thetas=guides,
    )
    tensors = compiler(observations, tf.constant([int(seed), 2901], tf.int32))
    identity_payload = {
        "compiler_route_id": (
            "zhao_cui_austria_persistent_rank_one_guide_family_compiler_v1"
        ),
        "route_classification": ROUTE_CLASSIFICATION,
        "proposal_operation_classification": PROPOSAL_OPERATION_CLASSIFICATION,
        "target_id": TARGET_ID,
        "event_order": EVENT_ORDER,
        "target_seed": SIR_DATASET_SEED,
        "horizon": int(horizon),
        "particle_count_per_guide": int(particle_count),
        "seed": int(seed),
        "dtype": "float64",
        "guide_family": "persistent_kappa_nu_cartesian_9",
        "guide_half_width": float(guide_half_width),
        "guide_count": int(guides.shape[0]),
        "guide_thetas_sha256": _tensor_hash(guides),
        "states_sha256": _tensor_hash(tensors["states"]),
        "initial_log_proposal_density_sha256": _tensor_hash(
            tensors["initial_log_proposal_density"]
        ),
        "ancestors_sha256": _tensor_hash(tensors["ancestors"]),
        "auxiliary_log_probabilities_sha256": _tensor_hash(
            tensors["auxiliary_log_probabilities"]
        ),
        "transition_log_proposal_density_sha256": _tensor_hash(
            tensors["transition_log_proposal_density"]
        ),
        "initial_reference_uniforms_sha256": _tensor_hash(
            tensors["initial_reference_uniforms"]
        ),
        "ancestor_uniforms_sha256": _tensor_hash(tensors["ancestor_uniforms"]),
        "transition_reference_uniforms_sha256": _tensor_hash(
            tensors["transition_reference_uniforms"]
        ),
    }
    program_id = _semantic_hash(identity_payload)
    manifest = {
        **identity_payload,
        "program_id": program_id,
        "rank_per_guide": 1,
        "kr_map": "analytic_diagonal_gaussian_quantile",
        "guide_persistence": "one_guide_per_full_genealogy",
        "branch_combination_scalar": "logmeanexp_of_guide_branch_likelihoods",
        "branch_combination_score": (
            "likelihood_weighted_guide_branch_score_average"
        ),
        "xla_jit_compile": True,
        "python_numerical_loop": False,
        "numpy_numerical_path": False,
        "paper_anchor": (
            ".localresources/papers/"
            "zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:807-924"
        ),
        "author_source_anchor": (
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/"
            "models/full_sol.m:21-43"
        ),
        "nonclaims": (
            "persistent guide family is not in the Zhao-Cui Austria example",
            "guide-domain construction is extension_or_invention",
            "no exact physical-likelihood, HMC, posterior, default, production, or superiority claim",
        ),
    }
    return AustriaSIRPersistentGuideProgram(
        observations=observations,
        guide_thetas=guides,
        states=tensors["states"],
        initial_log_proposal_density=tensors[
            "initial_log_proposal_density"
        ],
        ancestors=tensors["ancestors"],
        auxiliary_log_probabilities=tensors["auxiliary_log_probabilities"],
        transition_log_proposal_density=tensors[
            "transition_log_proposal_density"
        ],
        program_id=program_id,
        manifest=manifest,
    )


__all__ = [
    "AustriaSIRRankOneProposalCompilation",
    "AustriaSIRRankOneMixtureProposalCompilation",
    "AustriaSIRPersistentGuideProgram",
    "COMPILER_ROUTE_ID",
    "PROPOSAL_OPERATION_CLASSIFICATION",
    "ROUTE_CLASSIFICATION",
    "compile_austria_sir_rank_one_proposal_branch",
    "compile_austria_sir_rank_one_mixture_proposal_branch",
    "compile_austria_sir_persistent_guide_program",
    "make_rank_one_branch_tensor_compiler",
    "make_rank_one_mixture_branch_tensor_compiler",
    "make_persistent_guide_tensor_compiler",
    "kappa_nu_guide_grid",
    "symmetric_guide_grid",
]
