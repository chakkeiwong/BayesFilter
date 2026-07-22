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


def _correlated_transport() -> FixedTTSIRTTransport:
    convention = _convention()
    product = ProductBasis(
        [LegendreBasis1D(BoundedInterval(-1.0, 1.0), 1) for _ in range(2)],
        convention,
    )
    functional_tt = FunctionalTT(
        [
            TTCore(
                tf.constant(
                    [[[1.0, 0.0], [0.0, 1.0]]],
                    DTYPE,
                )
            ),
            TTCore(
                tf.constant(
                    [[[1.0], [0.0]], [[0.0], [0.1]]],
                    DTYPE,
                )
            ),
        ],
        product,
        convention,
    )
    defensive = TensorProductReferenceDensity(product, convention)
    tau = tf.constant(0.05, DTYPE)
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


def test_batched_inverse_matches_original_scalar_grid_bisection() -> None:
    transport = _correlated_transport()
    reference = tf.constant(
        [[0.1, 0.35, 0.8], [0.2, 0.55, 0.9]], DTYPE
    )

    batched = transport.inverse_transport(reference)
    scalar_columns = []
    for sample_index in range(int(reference.shape[1])):
        first, _ = transport._inverse_axis(
            0,
            tf.zeros([1, 0], DTYPE),
            reference[0, sample_index],
        )
        second, _ = transport._inverse_axis(
            1,
            tf.reshape(first.z_value, [1, 1]),
            reference[1, sample_index],
        )
        scalar_columns.append(tf.stack([first.z_value, second.z_value]))

    tf.debugging.assert_near(
        batched,
        tf.stack(scalar_columns, axis=1),
        atol=2e-12,
        rtol=2e-12,
    )


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
    assert compilation.manifest["classification"] == "extension_or_invention"
    assert compilation.manifest["axis_order"] == ("x_previous", "x_current")
    assert compilation.manifest["production_kr_closure"] is False
    operations = compilation.manifest["operation_classifications"]
    assert operations["squared_tt_defensive_density"]["classification"] == "source_faithful"
    assert operations["squared_tt_defensive_density"]["paper_anchor"]
    assert operations["squared_tt_defensive_density"]["author_source_anchor"]
    assert operations["paired_core_prefix_conditional"]["classification"] == "source_faithful"
    assert operations["paired_core_prefix_conditional"]["paper_anchor"]
    assert operations["paired_core_prefix_conditional"]["author_source_anchor"]
    assert operations["frozen_randomness_and_settings"]["classification"] == "fixed_hmc_adaptation"
    assert operations["frozen_randomness_and_settings"]["paper_anchor"]
    assert operations["frozen_randomness_and_settings"]["author_source_anchor"]
    assert operations["previous_current_prefix_axis_order"]["classification"] == "extension_or_invention"
    assert operations["finite_grid_trapezoid_bisection_inverse"]["classification"] == "extension_or_invention"
    initial_manifest = compilation.manifest["initial_transport"]
    assert initial_manifest["route_classification"] == "extension_or_invention"
    assert initial_manifest["proposition2_marginal_classification"] == "source_faithful"
    assert initial_manifest["conditional_cdf_route_class"].startswith(
        "extension_or_invention"
    )
    assert initial_manifest["paper_anchors"]
    assert initial_manifest["author_source_anchors"]


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

    tf.debugging.assert_equal(
        combined.branch.observations,
        tf.constant([[0.0, 1.0], [0.2, 1.2]], DTYPE),
    )
    assert combined.manifest["observation_mode"] == "concatenate"
    assert combined.manifest["block_observation_dimensions"] == (1, 1)
