from __future__ import annotations

import tensorflow as tf

import bayesfilter.highdim as highdim


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _product_basis(dimension: int) -> highdim.ProductBasis:
    return highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 0)
            for _ in range(dimension)
        ],
        _convention(),
    )


def _density(dimension: int) -> highdim.SquaredTTDensity:
    product = _product_basis(dimension)
    ftt = highdim.FunctionalTT(
        [highdim.TTCore(tf.constant([[[1.0]]], dtype=tf.float64)) for _ in range(dimension)],
        product,
        _convention(),
    )
    defensive = highdim.TensorProductReferenceDensity(product, _convention())
    branch_identity = highdim.SquaredTTDensity.expected_branch_identity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tf.constant(0.0, dtype=tf.float64),
        normalizer_floor=tf.constant(1e-12, dtype=tf.float64),
        denominator_floor=tf.constant(1e-12, dtype=tf.float64),
        measure_convention=_convention(),
    )
    return highdim.SquaredTTDensity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tf.constant(0.0, dtype=tf.float64),
        normalizer_floor=tf.constant(1e-12, dtype=tf.float64),
        denominator_floor=tf.constant(1e-12, dtype=tf.float64),
        measure_convention=_convention(),
        branch_identity=branch_identity,
    )


def _transport(dimension: int) -> highdim.FixedTTSIRTTransport:
    return highdim.FixedTTSIRTTransport(
        density=_density(dimension),
        cdf_config=highdim.KRCDFConfig(
            grid_size=129,
            bisection_steps=28,
            monotonicity_tolerance=1e-12,
            bracket_tolerance=1e-12,
            denominator_floor=1e-12,
            max_floor_count=0,
        ),
    )


def test_p57_m4_fixed_ttsirt_transport_declares_source_contract() -> None:
    transport = _transport(2)
    protocol = highdim.SourceRouteTransportProtocol(transport)
    payload = protocol.manifest_payload()

    assert payload["source_contract_level"] == "fixed_ttsirt"
    assert payload["transport_object"]["source_map_semantics"] == "ttsirt_eval_irt_rt_cirt_reference_style"
    assert payload["transport_object"]["tt_cores_declared"] is True
    assert payload["transport_object"]["defensive_density_declared"] is True


def test_p57_m4_forward_inverse_roundtrip_uniform_source_map() -> None:
    transport = _transport(2)
    local = tf.constant([[-0.8, 0.0, 0.5], [0.25, -0.5, 0.75]], dtype=tf.float64)
    reference = transport.forward_transport(local)
    roundtrip = transport.inverse_transport(reference)

    expected_reference = 0.5 * (local + 1.0)
    tf.debugging.assert_near(reference, expected_reference, atol=2e-3)
    tf.debugging.assert_near(roundtrip, local, atol=2e-4)


def test_p57_m4_conditional_inverse_matches_joint_inverse_suffix() -> None:
    transport = _transport(2)
    conditioning = tf.constant([[-0.25]], dtype=tf.float64)
    suffix_reference = tf.constant([[0.75, 0.2]], dtype=tf.float64)
    suffix = transport.conditional_inverse_transport(conditioning, suffix_reference)

    tf.debugging.assert_near(suffix, 2.0 * suffix_reference - 1.0, atol=2e-4)


def test_p57_m4_upper_suffix_conditional_roundtrip() -> None:
    transport = _transport(2)
    conditioning = tf.constant([[0.25, -0.5]], dtype=tf.float64)
    generated = tf.constant([[-0.75, 0.6]], dtype=tf.float64)
    uniforms = transport.conditional_forward_transport_suffix(
        conditioning, generated
    )
    reconstructed = transport.conditional_inverse_transport_suffix(
        conditioning, uniforms
    )
    tf.debugging.assert_near(reconstructed, generated, atol=2e-3)


def test_p57_m4_density_jacobian_tieout_for_source_map() -> None:
    transport = _transport(2)
    local = tf.constant([[-0.5, 0.5], [0.25, -0.25]], dtype=tf.float64)
    reference = transport.forward_transport(local)
    reconstructed = transport.inverse_transport(reference)
    log_density = tf.math.log(transport.eval_pdf(local))
    log_jacobian = transport.forward_log_jacobian(local)

    tf.debugging.assert_near(reconstructed, local, atol=2e-4)
    tf.debugging.assert_near(log_jacobian, log_density, atol=1e-12)
    tf.debugging.assert_near(
        log_density,
        tf.fill([2], tf.math.log(tf.constant(0.25, dtype=tf.float64))),
        atol=1e-12,
    )


def test_algebraic_source_map_uses_finite_reference_cdf_and_physical_density() -> None:
    product = highdim.ProductBasis(
        [highdim.LagrangePiecewiseBasis1D(highdim.AlgebraicMap(1.0), 1, 1)],
        _convention(),
    )
    ftt = highdim.FunctionalTT(
        [highdim.TTCore(tf.ones([1, 2, 1], dtype=tf.float64))],
        product,
        _convention(),
    )
    defensive = highdim.TensorProductReferenceDensity(product, _convention())
    identity = highdim.SquaredTTDensity.expected_branch_identity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tf.constant(0.0, dtype=tf.float64),
        normalizer_floor=tf.constant(1e-12, dtype=tf.float64),
        denominator_floor=tf.constant(1e-12, dtype=tf.float64),
        measure_convention=_convention(),
    )
    density = highdim.SquaredTTDensity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tf.constant(0.0, dtype=tf.float64),
        normalizer_floor=tf.constant(1e-12, dtype=tf.float64),
        denominator_floor=tf.constant(1e-12, dtype=tf.float64),
        measure_convention=_convention(),
        branch_identity=identity,
    )
    transport = highdim.FixedTTSIRTTransport(
        density=density,
        cdf_config=highdim.KRCDFConfig(
            grid_size=257,
            bisection_steps=32,
            monotonicity_tolerance=1e-12,
            bracket_tolerance=1e-12,
            denominator_floor=1e-12,
            max_floor_count=0,
        ),
    )
    local = tf.constant([[-2.0, 0.0, 2.0]], dtype=tf.float64)
    reference_coordinate = local / tf.sqrt(1.0 + tf.square(local))
    expected_uniform = 0.5 * (reference_coordinate + 1.0)
    expected_density = 0.5 * tf.pow(1.0 + tf.square(local[0]), -1.5)

    uniform = transport.forward_transport(local)
    reconstructed = transport.inverse_transport(uniform)

    tf.debugging.assert_near(uniform, expected_uniform, atol=2e-4)
    tf.debugging.assert_near(reconstructed, local, atol=2e-3)
    tf.debugging.assert_near(transport.eval_pdf(local), expected_density, atol=1e-12)
    tf.debugging.assert_near(
        transport.forward_log_jacobian(local),
        tf.math.log(expected_density),
        atol=2e-4,
    )
