from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import pytest
import tensorflow as tf

from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
)
from bayesfilter.highdim.filtering import IdentityCoordinateMap
from bayesfilter.highdim.models import PredatorPreySSM
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.zhao_cui_predator_prey_fixed_variant_tf import (
    COMPILER_CLASSIFICATION,
    EVENT_ORDER,
    ROUTE_CLASSIFICATION,
    TARGET_HORIZON,
    TARGET_ID,
    compile_source_order_ttsirt_proposal_branch,
    prepare_predator_prey_fixed_variant_program,
    prepare_predator_prey_source_order_branch,
    prepare_source_order_frozen_apf_program,
    prepare_source_order_frozen_branch,
)
from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
    generate_source_order_predator_prey_dataset_tf,
)


DTYPE = tf.float64
LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), DTYPE)


def _constant_transport(dimension: int) -> FixedTTSIRTTransport:
    convention = MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    product = ProductBasis(
        [LegendreBasis1D(BoundedInterval(-1.0, 1.0), 0) for _ in range(dimension)],
        convention,
    )
    functional_tt = FunctionalTT(
        [TTCore(tf.ones([1, 1, 1], DTYPE)) for _ in range(dimension)],
        product,
        convention,
    )
    defensive = TensorProductReferenceDensity(product, convention)
    tau = tf.constant(0.25, DTYPE)
    floor = tf.constant(1e-12, DTYPE)
    density = SquaredTTDensity(
        sqrt_tt=functional_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=floor,
        denominator_floor=floor,
        measure_convention=convention,
        branch_identity=SquaredTTDensity.expected_branch_identity(
            sqrt_tt=functional_tt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=floor,
            denominator_floor=floor,
            measure_convention=convention,
        ),
    )
    return FixedTTSIRTTransport(
        density=density,
        cdf_config=KRCDFConfig(
            grid_size=33,
            bisection_steps=16,
            monotonicity_tolerance=1e-12,
            bracket_tolerance=1e-12,
            denominator_floor=1e-12,
            max_floor_count=0,
        ),
    )


def _isotropic_log_density(
    value: tf.Tensor, mean: tf.Tensor, variance: float
) -> tf.Tensor:
    residual = tf.convert_to_tensor(value, DTYPE) - tf.convert_to_tensor(mean, DTYPE)
    dimension = tf.cast(tf.shape(residual)[-1], DTYPE)
    variance_tensor = tf.constant(variance, DTYPE)
    return -0.5 * (
        dimension * (LOG_TWO_PI + tf.math.log(variance_tensor))
        + tf.reduce_sum(tf.square(residual), axis=-1) / variance_tensor
    )


@dataclass(frozen=True)
class _SourceOrderLocationModel:
    dimension: int

    def parameter_dim(self) -> int:
        return 2

    def state_dim(self) -> int:
        return self.dimension

    def observation_dim(self) -> int:
        return self.dimension

    def frozen_apf_measure_id(self) -> str:
        return "full_state_lebesgue_v1"

    def frozen_apf_score_backend_id(self) -> str:
        return "analytical_parameter_score_no_autodiff_v1"

    def initial_log_density(self, theta: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        return _isotropic_log_density(
            state, tf.fill([self.dimension], theta[0]), 1.3
        )

    def transition_log_density(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del time_index
        return _isotropic_log_density(current, 0.7 * previous + theta[0], 0.8)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del time_index
        return _isotropic_log_density(
            observation[tf.newaxis, :], state + theta[1], 1.1
        )

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, state: tf.Tensor
    ) -> tf.Tensor:
        residual = state - theta[0]
        first = tf.reduce_sum(residual, axis=1) / 1.3
        return tf.stack([first, tf.zeros_like(first)], axis=1)

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del time_index
        residual = current - (0.7 * previous + theta[0])
        first = tf.reduce_sum(residual, axis=1) / 0.8
        return tf.stack([first, tf.zeros_like(first)], axis=1)

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del time_index
        residual = observation[tf.newaxis, :] - (state + theta[1])
        second = tf.reduce_sum(residual, axis=1) / 1.1
        return tf.stack([tf.zeros_like(second), second], axis=1)

    def manifest_payload(self) -> Mapping[str, object]:
        return {"family": "source_order_location_fixture", "dimension": self.dimension}


def _fixture_branch():
    states = tf.constant(
        [
            [[-0.7, 0.1], [0.0, -0.3], [0.5, 0.8], [1.1, -0.4]],
            [[-0.2, 0.5], [0.4, -0.1], [0.8, 0.3], [-0.5, 0.9]],
            [[0.1, 0.7], [0.9, 0.0], [-0.3, 0.4], [0.6, -0.6]],
        ],
        DTYPE,
    )
    observations = tf.constant([[0.3, -0.2], [0.8, 0.1]], DTYPE)
    ancestors = tf.constant([[3, 0, 2, 1], [1, 3, 0, 2]], tf.int32)
    auxiliary = tf.constant(
        [[0.1, 0.2, 0.3, 0.4], [0.35, 0.15, 0.25, 0.25]], DTYPE
    )
    return prepare_source_order_frozen_branch(
        observations=observations,
        states=states,
        initial_log_proposal_density=_isotropic_log_density(
            states[0], tf.zeros([2], DTYPE), 1.7
        ),
        ancestors=ancestors,
        auxiliary_log_probabilities=tf.math.log(auxiliary),
        transition_log_proposal_density=tf.stack(
            [
                _isotropic_log_density(states[1], tf.zeros([2], DTYPE), 1.5),
                _isotropic_log_density(states[2], tf.zeros([2], DTYPE), 1.9),
            ]
        ),
        target_id="source_order_location_fixture_v1",
        event_order="x0_then_transition_then_y1",
        target_seed=17,
        target_state_sha256="0" * 64,
        target_observation_sha256="1" * 64,
    )


def _direct_source_order_scalar(
    model: _SourceOrderLocationModel,
    theta: tf.Tensor,
    branch,
) -> tf.Tensor:
    """Independent scalar statement; deliberately has no observation at x0."""

    log_n = tf.math.log(tf.cast(branch.particle_count, DTYPE))
    current = (
        model.initial_log_density(theta, branch.states[0])
        - branch.initial_log_proposal_density
    )
    log_sum = tf.reduce_logsumexp(current)
    value = log_sum - log_n
    previous_log_weights = current - log_sum
    for time_index in range(1, branch.transition_count + 1):
        row = time_index - 1
        ancestor = branch.ancestors[row]
        parent = tf.gather(branch.states[time_index - 1], ancestor)
        current = (
            tf.gather(previous_log_weights, ancestor)
            + model.transition_log_density(
                theta, parent, branch.states[time_index], time_index
            )
            + model.observation_log_density(
                theta,
                branch.states[time_index],
                branch.observations[row],
                time_index,
            )
            - tf.gather(branch.auxiliary_log_probabilities[row], ancestor)
            - branch.transition_log_proposal_density[row]
        )
        log_sum = tf.reduce_logsumexp(current)
        value = value + log_sum - log_n
        previous_log_weights = current - log_sum
    return value


def test_source_order_program_matches_independent_scalar_and_not_y0_program() -> None:
    model = _SourceOrderLocationModel(2)
    theta = tf.constant([0.17, -0.09], DTYPE)
    branch = _fixture_branch()
    program = prepare_source_order_frozen_apf_program(model, branch)

    result = program.evaluate(theta)
    expected = _direct_source_order_scalar(model, theta, branch)
    y0_first_initial = (
        model.initial_log_density(theta, branch.states[0])
        + model.observation_log_density(
            theta, branch.states[0], branch.observations[0], 0
        )
        - branch.initial_log_proposal_density
    )
    y0_first_initial_increment = tf.reduce_logsumexp(y0_first_initial) - tf.math.log(
        tf.cast(branch.particle_count, DTYPE)
    )

    tf.debugging.assert_near(result["log_likelihood"], expected, atol=2e-12)
    tf.debugging.assert_near(
        result["log_likelihood"], tf.reduce_sum(result["log_increments"]), atol=2e-12
    )
    tf.debugging.assert_near(
        result["score"], tf.reduce_sum(result["increment_scores"], axis=0), atol=2e-12
    )
    assert abs(
        float((result["log_increments"][0] - y0_first_initial_increment).numpy())
    ) > 1e-3
    assert result["log_increments"].shape == (3,)
    assert result["increment_scores"].shape == (3, 2)
    assert bool(result["finite"].numpy())


def test_source_order_recursive_score_matches_same_scalar_fd_and_graph() -> None:
    model = _SourceOrderLocationModel(2)
    theta = tf.constant([0.17, -0.09], DTYPE)
    branch = _fixture_branch()
    program = prepare_source_order_frozen_apf_program(model, branch)
    result = program.evaluate(theta)

    step = tf.constant(1e-5, DTYPE)
    finite_difference = []
    for parameter_index in range(model.parameter_dim()):
        direction = tf.one_hot(parameter_index, model.parameter_dim(), dtype=DTYPE)
        finite_difference.append(
            (
                _direct_source_order_scalar(model, theta + step * direction, branch)
                - _direct_source_order_scalar(model, theta - step * direction, branch)
            )
            / (2.0 * step)
        )

    tf.debugging.assert_near(
        result["score"], tf.stack(finite_difference), atol=3e-8, rtol=3e-8
    )
    graph_result = program.compiled(jit_compile=False)(theta)
    tf.debugging.assert_near(
        graph_result["log_likelihood"], result["log_likelihood"], atol=2e-12
    )
    tf.debugging.assert_near(graph_result["score"], result["score"], atol=2e-12)


def test_source_order_ttsirt_compiler_emits_t_plus_one_branch() -> None:
    particle_count = 4
    auxiliary = tf.math.log(
        tf.constant(
            [[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]], DTYPE
        )
    )
    keyword_arguments = {
        "observations": tf.constant([[0.2, -0.1], [0.4, 0.3]], DTYPE),
        "initial_transport": _constant_transport(2),
        "transition_transports": (_constant_transport(4), _constant_transport(4)),
        "previous_coordinate_maps": (IdentityCoordinateMap(2), IdentityCoordinateMap(2)),
        "current_coordinate_maps": (IdentityCoordinateMap(2), IdentityCoordinateMap(2)),
        "initial_reference_points": tf.constant(
            [[0.1, 0.3, 0.6, 0.9], [0.2, 0.4, 0.7, 0.8]], DTYPE
        ),
        "ancestor_uniforms": tf.constant(
            [[0.05, 0.15, 0.45, 0.95], [0.1, 0.35, 0.65, 0.9]], DTYPE
        ),
        "auxiliary_log_probabilities": auxiliary,
        "transition_reference_points": tf.constant(
            [
                [[0.2, 0.4, 0.7, 0.8], [0.1, 0.3, 0.6, 0.9]],
                [[0.15, 0.45, 0.65, 0.85], [0.25, 0.35, 0.55, 0.75]],
            ],
            DTYPE,
        ),
        "target_id": "source_order_compiler_fixture_v1",
        "event_order": "x0_then_transition_then_y1",
        "target_seed": 29,
        "target_state_sha256": "4" * 64,
        "target_observation_sha256": "5" * 64,
        "online_dtype": tf.float32,
    }
    compilation = compile_source_order_ttsirt_proposal_branch(**keyword_arguments)
    branch = compilation.branch

    assert branch.states.shape == (3, particle_count, 2)
    assert branch.observations.shape == (2, 2)
    assert branch.ancestors.shape == (2, particle_count)
    assert branch.transition_log_proposal_density.shape == (2, particle_count)
    assert branch.dtype == tf.float32
    assert compilation.manifest["classification"] == COMPILER_CLASSIFICATION
    assert compilation.manifest["production_kr_closure"] is False
    operations = compilation.manifest["operation_classifications"]
    assert operations["squared_tt_defensive_density"]["classification"] == "source_faithful"
    assert operations["paired_core_prefix_conditional"]["classification"] == "source_faithful"
    assert operations["frozen_randomness_and_settings"]["classification"] == "fixed_hmc_adaptation"
    assert operations["source_order_fixed_branch_value_and_score"]["classification"] == "extension_or_invention"

    changed = dict(keyword_arguments)
    changed["initial_reference_points"] = tf.constant(
        [[0.11, 0.3, 0.6, 0.9], [0.2, 0.4, 0.7, 0.8]], DTYPE
    )
    second = compile_source_order_ttsirt_proposal_branch(**changed)
    assert second.branch.branch_id != branch.branch_id
    assert second.compiler_id != compilation.compiler_id


def _predator_prey_tiny_branch(model: PredatorPreySSM):
    theta = model.true_parameters()
    initial = tf.constant(
        [[49.4, 4.7], [50.2, 5.1], [50.8, 5.5], [49.9, 4.4]], DTYPE
    )
    ancestor_rows = tf.constant([[0, 2, 3, 1], [1, 3, 0, 2]], tf.int32)
    states = [initial]
    observations = []
    transition_log_q = []
    for time_index in range(2):
        parents = tf.gather(states[-1], ancestor_rows[time_index])
        mean = model.transition_mean(theta, parents)
        residual = tf.constant(
            [[0.2, -0.3], [-0.4, 0.1], [0.3, 0.25], [-0.15, -0.2]], DTYPE
        ) * tf.cast(time_index + 1, DTYPE)
        current = mean + residual
        states.append(current)
        observations.append(
            tf.reduce_mean(current, axis=0)
            + tf.constant([0.35, -0.25], DTYPE)
        )
        transition_log_q.append(
            _isotropic_log_density(current, mean, 3.5 + 0.25 * time_index)
        )
    state_tensor = tf.stack(states)
    auxiliary = tf.constant(
        [[0.15, 0.25, 0.35, 0.25], [0.3, 0.2, 0.1, 0.4]], DTYPE
    )
    return prepare_source_order_frozen_branch(
        observations=tf.stack(observations),
        states=state_tensor,
        initial_log_proposal_density=_isotropic_log_density(
            initial, model.initial_mean, 1.4
        ),
        ancestors=ancestor_rows,
        auxiliary_log_probabilities=tf.math.log(auxiliary),
        transition_log_proposal_density=tf.stack(transition_log_q),
        target_id="predator_prey_tiny_mechanics_fixture_v1",
        event_order="x0_then_transition_then_y1",
        target_seed=23,
        target_state_sha256="2" * 64,
        target_observation_sha256="3" * 64,
    )


def test_predator_prey_all_six_manual_score_coordinates_match_fd() -> None:
    model = PredatorPreySSM()
    theta = model.true_parameters()
    branch = _predator_prey_tiny_branch(model)
    program = prepare_source_order_frozen_apf_program(model, branch)
    result = program.evaluate(theta)
    steps = tf.constant([2e-6, 2e-4, 2e-5, 2e-6, 2e-6, 2e-6], DTYPE)
    finite_difference = []
    for parameter_index in range(model.parameter_dim()):
        direction = tf.one_hot(parameter_index, model.parameter_dim(), dtype=DTYPE)
        step = steps[parameter_index]
        plus = program.evaluate(theta + step * direction)["log_likelihood"]
        minus = program.evaluate(theta - step * direction)["log_likelihood"]
        finite_difference.append((plus - minus) / (2.0 * step))

    tf.debugging.assert_near(
        result["score"], tf.stack(finite_difference), atol=3e-5, rtol=2e-5
    )
    graph_result = program.compiled(jit_compile=False)(theta)
    tf.debugging.assert_near(graph_result["score"], result["score"], atol=2e-9)
    assert bool(result["finite"].numpy())


def test_predator_prey_float32_online_model_matches_float64_reference() -> None:
    reference = PredatorPreySSM(dtype=tf.float64)
    online = PredatorPreySSM(dtype=tf.float32)
    previous64 = tf.constant(
        [[49.7, 4.9], [75.0, 7.2], [109.0, 4.1], [113.0, -0.2]],
        tf.float64,
    )
    mean64 = reference.transition_mean(reference.true_parameters(), previous64)
    current64 = mean64 + tf.constant(
        [[0.2, -0.1], [-0.3, 0.15], [0.1, 0.25], [-0.2, -0.3]],
        tf.float64,
    )
    transition64 = reference.transition_log_density(
        reference.true_parameters(), previous64, current64, 1
    )
    score64 = reference.transition_log_density_parameter_score(
        reference.true_parameters(), previous64, current64, 1
    )

    previous32 = tf.cast(previous64, tf.float32)
    current32 = tf.cast(current64, tf.float32)
    mean32 = online.transition_mean(online.true_parameters(), previous32)
    transition32 = online.transition_log_density(
        online.true_parameters(), previous32, current32, 1
    )
    score32 = online.transition_log_density_parameter_score(
        online.true_parameters(), previous32, current32, 1
    )

    assert mean32.dtype == tf.float32
    assert transition32.dtype == tf.float32
    assert score32.dtype == tf.float32
    tf.debugging.assert_near(tf.cast(mean32, tf.float64), mean64, atol=2e-4, rtol=2e-5)
    tf.debugging.assert_near(
        tf.cast(transition32, tf.float64), transition64, atol=3e-5, rtol=2e-5
    )
    tf.debugging.assert_near(
        tf.cast(score32, tf.float64), score64, atol=4e-4, rtol=4e-4
    )
    assert online.manifest_payload()["dtype"] == "float32"


def test_sealed_factory_rejects_non_t20_branch_and_manifest_is_nonadmitted() -> None:
    model = PredatorPreySSM()
    tiny = _predator_prey_tiny_branch(model)
    with pytest.raises(ValueError, match="sealed predator-prey target mismatch"):
        prepare_predator_prey_fixed_variant_program(model, tiny)

    particle_count = 2
    _truth_states, sealed_observations = (
        generate_source_order_predator_prey_dataset_tf()
    )
    sealed = prepare_predator_prey_source_order_branch(
        observations=sealed_observations,
        states=tf.zeros([TARGET_HORIZON + 1, particle_count, 2], DTYPE),
        initial_log_proposal_density=tf.zeros([particle_count], DTYPE),
        ancestors=tf.zeros([TARGET_HORIZON, particle_count], tf.int32),
        auxiliary_log_probabilities=tf.fill(
            [TARGET_HORIZON, particle_count], -tf.math.log(tf.constant(2.0, DTYPE))
        ),
        transition_log_proposal_density=tf.zeros(
            [TARGET_HORIZON, particle_count], DTYPE
        ),
    )
    program = prepare_predator_prey_fixed_variant_program(model, sealed)
    manifest = program.manifest_payload()
    assert manifest["target_id"] == TARGET_ID
    assert manifest["event_order"] == EVENT_ORDER
    assert manifest["route_classification"] == ROUTE_CLASSIFICATION
    assert manifest["identity_role"] == "reproducibility_fingerprint_not_admission"
    assert manifest["retained_grid_route"] is False
    assert manifest["pseudo_marginal_exact_target_claimed"] is False
    assert manifest["runtime_autodiff"] is False
    assert manifest["runtime_finite_difference"] is False

    with pytest.raises(ValueError, match="observations"):
        prepare_predator_prey_source_order_branch(
            observations=tf.zeros([TARGET_HORIZON, 2], DTYPE),
            states=tf.zeros([TARGET_HORIZON + 1, particle_count, 2], DTYPE),
            initial_log_proposal_density=tf.zeros([particle_count], DTYPE),
            ancestors=tf.zeros([TARGET_HORIZON, particle_count], tf.int32),
            auxiliary_log_probabilities=tf.fill(
                [TARGET_HORIZON, particle_count],
                -tf.math.log(tf.constant(2.0, DTYPE)),
            ),
            transition_log_proposal_density=tf.zeros(
                [TARGET_HORIZON, particle_count], DTYPE
            ),
        )
