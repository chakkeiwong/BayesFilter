from __future__ import annotations

import math

import tensorflow as tf

from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
)
from bayesfilter.highdim.filtering import IdentityCoordinateMap
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    AlgebraicCoordinateMap,
    TTSIRT_COMPILER_CLASSIFICATION,
    combine_fixed_ttsirt_block_compilations,
    compile_fixed_ttsirt_proposal_branch,
)


DTYPE = tf.float64


def _convention() -> MeasureConvention:
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _constant_transport(dimension: int) -> FixedTTSIRTTransport:
    convention = _convention()
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
    normalizer_floor = tf.constant(1e-12, DTYPE)
    denominator_floor = tf.constant(1e-12, DTYPE)
    density = SquaredTTDensity(
        sqrt_tt=functional_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=normalizer_floor,
        denominator_floor=denominator_floor,
        measure_convention=convention,
        branch_identity=SquaredTTDensity.expected_branch_identity(
            sqrt_tt=functional_tt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=normalizer_floor,
            denominator_floor=denominator_floor,
            measure_convention=convention,
        ),
    )
    return FixedTTSIRTTransport(
        density=density,
        cdf_config=KRCDFConfig(
            grid_size=65,
            bisection_steps=24,
            monotonicity_tolerance=1e-12,
            bracket_tolerance=1e-12,
            denominator_floor=1e-12,
            max_floor_count=0,
        ),
    )


def test_algebraic_coordinate_map_roundtrip_and_jacobian_directions() -> None:
    coordinate_map = AlgebraicCoordinateMap(tf.constant([0.75, 2.0], DTYPE))
    reference = tf.constant(
        [[-0.8, 0.4], [-0.2, 0.0], [0.6, -0.7]], DTYPE
    )

    physical, log_dxdz = coordinate_map.forward(reference)
    reconstructed, log_dzdx = coordinate_map.inverse(physical)

    tf.debugging.assert_near(reconstructed, reference, atol=2e-12)
    tf.debugging.assert_near(log_dxdz + log_dzdx, tf.zeros([3], DTYPE), atol=2e-12)
    assert coordinate_map.manifest_payload()["family"] == "AlgebraicCoordinateMap"


def test_conditional_proposal_density_uses_paired_prefix_marginal() -> None:
    transport = _constant_transport(4)
    conditioning = tf.constant(
        [[-0.8, 0.1, 0.6], [0.4, -0.2, 0.9]], DTYPE
    )
    generated = tf.constant(
        [[-0.3, 0.7, 0.2], [0.8, -0.5, 0.0]], DTYPE
    )

    log_density = transport.conditional_proposal_log_density(
        conditioning_points=conditioning,
        generated_points=generated,
    )

    expected = tf.fill([3], tf.constant(-2.0 * math.log(2.0), DTYPE))
    tf.debugging.assert_near(log_density, expected, atol=2e-12)


def test_fixed_ttsirt_compiler_emits_correct_uniform_branch_and_ancestors() -> None:
    particle_count = 4
    initial_transport = _constant_transport(1)
    transition_transport = _constant_transport(2)
    log_auxiliary = tf.math.log(
        tf.constant([[0.1, 0.2, 0.3, 0.4]], DTYPE)
    )
    compilation = compile_fixed_ttsirt_proposal_branch(
        observations=tf.constant([[0.0], [0.2]], DTYPE),
        initial_transport=initial_transport,
        transition_transports=(transition_transport,),
        coordinate_map=IdentityCoordinateMap(1),
        initial_reference_points=tf.constant([[0.1, 0.3, 0.6, 0.9]], DTYPE),
        ancestor_uniforms=tf.constant([[0.05, 0.15, 0.45, 0.95]], DTYPE),
        auxiliary_log_probabilities=log_auxiliary,
        transition_reference_points=tf.constant(
            [[[0.2, 0.4, 0.7, 0.8]]], DTYPE
        ),
    )
    branch = compilation.branch

    tf.debugging.assert_equal(branch.ancestors, tf.constant([[0, 1, 2, 3]], tf.int32))
    tf.debugging.assert_near(
        branch.states,
        tf.constant(
            [[[-0.8], [-0.4], [0.2], [0.8]], [[-0.6], [-0.2], [0.4], [0.6]]],
            DTYPE,
        ),
        atol=3e-4,
    )
    expected_log_q = tf.fill([particle_count], tf.constant(-math.log(2.0), DTYPE))
    tf.debugging.assert_near(branch.initial_log_proposal_density, expected_log_q, atol=2e-12)
    tf.debugging.assert_near(
        branch.transition_log_proposal_density[0], expected_log_q, atol=2e-12
    )
    assert len(compilation.compiler_id) == 64
    assert compilation.manifest["classification"] == TTSIRT_COMPILER_CLASSIFICATION
    assert compilation.manifest["axis_order"] == ("x_previous", "x_current")
    assert compilation.manifest["production_kr_closure"] is False


def test_block_compiler_concatenates_states_and_sums_log_proposals() -> None:
    keyword_arguments = {
        "observations": tf.constant([[0.0], [0.2]], DTYPE),
        "initial_transport": _constant_transport(1),
        "transition_transports": (_constant_transport(2),),
        "coordinate_map": IdentityCoordinateMap(1),
        "initial_reference_points": tf.constant([[0.1, 0.3, 0.6, 0.9]], DTYPE),
        "ancestor_uniforms": tf.constant([[0.05, 0.15, 0.45, 0.95]], DTYPE),
        "auxiliary_log_probabilities": tf.math.log(
            tf.constant([[0.1, 0.2, 0.3, 0.4]], DTYPE)
        ),
        "transition_reference_points": tf.constant(
            [[[0.2, 0.4, 0.7, 0.8]]], DTYPE
        ),
    }
    first = compile_fixed_ttsirt_proposal_branch(**keyword_arguments)
    second = compile_fixed_ttsirt_proposal_branch(**keyword_arguments)

    combined = combine_fixed_ttsirt_block_compilations((first, second))

    tf.debugging.assert_equal(combined.branch.ancestors, first.branch.ancestors)
    tf.debugging.assert_near(
        combined.branch.states,
        tf.concat([first.branch.states, second.branch.states], axis=2),
        atol=0.0,
    )
    tf.debugging.assert_near(
        combined.branch.initial_log_proposal_density,
        2.0 * first.branch.initial_log_proposal_density,
        atol=0.0,
    )
    tf.debugging.assert_near(
        combined.branch.transition_log_proposal_density,
        2.0 * first.branch.transition_log_proposal_density,
        atol=0.0,
    )
    assert combined.manifest["block_count"] == 2
    assert combined.manifest["block_state_dimensions"] == (1, 1)
    assert combined.manifest["observation_mode"] == "shared"
    assert combined.manifest["shared_ancestor_genealogy"] is True


def test_block_compiler_can_concatenate_block_observations() -> None:
    common = {
        "initial_transport": _constant_transport(1),
        "transition_transports": (_constant_transport(2),),
        "coordinate_map": IdentityCoordinateMap(1),
        "initial_reference_points": tf.constant([[0.1, 0.3, 0.6, 0.9]], DTYPE),
        "ancestor_uniforms": tf.constant([[0.05, 0.15, 0.45, 0.95]], DTYPE),
        "auxiliary_log_probabilities": tf.math.log(
            tf.constant([[0.1, 0.2, 0.3, 0.4]], DTYPE)
        ),
        "transition_reference_points": tf.constant(
            [[[0.2, 0.4, 0.7, 0.8]]], DTYPE
        ),
    }
    first = compile_fixed_ttsirt_proposal_branch(
        observations=tf.constant([[0.0], [0.2]], DTYPE),
        **common,
    )
    second = compile_fixed_ttsirt_proposal_branch(
        observations=tf.constant([[1.0], [1.2]], DTYPE),
        **common,
    )

    combined = combine_fixed_ttsirt_block_compilations(
        (first, second), observation_mode="concatenate"
    )

    tf.debugging.assert_near(
        combined.branch.observations,
        tf.constant([[0.0, 1.0], [0.2, 1.2]], DTYPE),
        atol=0.0,
    )
    assert combined.manifest["observation_mode"] == "concatenate"
    assert combined.manifest["block_observation_dimensions"] == (1, 1)
