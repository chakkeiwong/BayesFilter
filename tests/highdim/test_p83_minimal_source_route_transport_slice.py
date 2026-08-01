from __future__ import annotations

import pytest
import tensorflow as tf

import bayesfilter.highdim as highdim


DTYPE = tf.float64


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _product_basis(degrees: tuple[int, ...]) -> highdim.ProductBasis:
    return highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), degree)
            for degree in degrees
        ],
        _convention(),
    )


def _identity(
    ftt: highdim.FunctionalTT,
    defensive: highdim.TensorProductReferenceDensity,
    tau: float,
) -> highdim.BranchIdentity:
    return highdim.SquaredTTDensity.expected_branch_identity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tf.constant(tau, dtype=DTYPE),
        normalizer_floor=tf.constant(1e-12, dtype=DTYPE),
        denominator_floor=tf.constant(1e-12, dtype=DTYPE),
        measure_convention=_convention(),
    )


def _nonuniform_density(tau: float = 0.25) -> highdim.SquaredTTDensity:
    product = _product_basis((1, 1))
    core0 = highdim.TTCore(
        tf.constant([[[1.0, 0.35], [0.2, -0.15]]], dtype=DTYPE)
    )
    core1 = highdim.TTCore(
        tf.constant([[[1.1], [0.25]], [[-0.4], [0.8]]], dtype=DTYPE)
    )
    ftt = highdim.FunctionalTT([core0, core1], product, _convention())
    defensive = highdim.TensorProductReferenceDensity(product, _convention())
    return highdim.SquaredTTDensity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tf.constant(tau, dtype=DTYPE),
        normalizer_floor=tf.constant(1e-12, dtype=DTYPE),
        denominator_floor=tf.constant(1e-12, dtype=DTYPE),
        measure_convention=_convention(),
        branch_identity=_identity(ftt, defensive, tau),
    )


def _constant_density(dimension: int, tau: float = 0.25) -> highdim.SquaredTTDensity:
    product = _product_basis(tuple(0 for _ in range(dimension)))
    ftt = highdim.FunctionalTT(
        [highdim.TTCore(tf.constant([[[1.0]]], dtype=DTYPE)) for _ in range(dimension)],
        product,
        _convention(),
    )
    defensive = highdim.TensorProductReferenceDensity(product, _convention())
    return highdim.SquaredTTDensity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tf.constant(tau, dtype=DTYPE),
        normalizer_floor=tf.constant(1e-12, dtype=DTYPE),
        denominator_floor=tf.constant(1e-12, dtype=DTYPE),
        measure_convention=_convention(),
        branch_identity=_identity(ftt, defensive, tau),
    )


def _cdf_config() -> highdim.KRCDFConfig:
    return highdim.KRCDFConfig(
        grid_size=65,
        bisection_steps=20,
        monotonicity_tolerance=1e-12,
        bracket_tolerance=1e-12,
        denominator_floor=1e-12,
        max_floor_count=0,
    )


def _fixed_transport(
    density: highdim.SquaredTTDensity,
) -> highdim.FixedTTSIRTTransport:
    return highdim.FixedTTSIRTTransport(density=density, cdf_config=_cdf_config())


def _frame() -> highdim.SourceRouteCoordinateFrame:
    return highdim.SourceRouteCoordinateFrame(
        mu=tf.constant([0.5, -0.25, 0.75], dtype=DTYPE),
        matrix=tf.constant(
            [
                [1.5, 0.0, 0.0],
                [0.1, 1.25, 0.0],
                [0.2, -0.1, 1.1],
            ],
            dtype=DTYPE,
        ),
        expansion_factor=1.0,
    )


def _transition_log_density(points: tf.Tensor, time_index: int) -> tf.Tensor:
    del time_index
    return -0.5 * tf.square(points[1, :] - points[2, :])


def _likelihood_log_density(points: tf.Tensor, time_index: int) -> tf.Tensor:
    del time_index
    return -0.25 * tf.square(points[1, :] - 1.0)


def _prior_log_density(points: tf.Tensor) -> tf.Tensor:
    center = tf.constant([[0.25], [-0.1]], dtype=DTYPE)
    return -0.5 * tf.reduce_sum(tf.square(points - center), axis=0)


def _components(
    *,
    include_prior: bool,
) -> highdim.SourceRouteSequentialDensityComponents:
    return highdim.SourceRouteSequentialDensityComponents(
        parameter_dim=1,
        state_dim=1,
        transition_log_density_fn=_transition_log_density,
        likelihood_log_density_fn=_likelihood_log_density,
        prior_log_density_fn=_prior_log_density if include_prior else None,
    )


def _target(
    *,
    time_index: int,
    shift_constant: float,
    components: highdim.SourceRouteSequentialDensityComponents | None,
) -> highdim.SourceRouteTarget:
    def negative_log_physical(points: tf.Tensor) -> tf.Tensor:
        if components is None:
            return tf.zeros([int(points.shape[1])], dtype=DTYPE)
        return components.negative_log_physical_density(
            physical_points=points,
            time_index=time_index,
            previous_retained_object=None,
        )

    return highdim.build_source_route_target(
        negative_log_physical_density_fn=negative_log_physical,
        coordinate_frame=_frame(),
        shift_constant=tf.constant(shift_constant, dtype=DTYPE),
        time_index=time_index,
    )


def _protocol(
    dimension: int = 3,
    tau: float = 0.25,
) -> highdim.SourceRouteTransportProtocol:
    return highdim.SourceRouteTransportProtocol(
        _fixed_transport(_constant_density(dimension, tau=tau))
    )


def test_p83_manifest_honestly_marks_grid_cdf_as_nonproduction() -> None:
    fixed = _fixed_transport(_nonuniform_density(tau=0.25))
    payload = fixed.manifest_payload()

    assert payload["source_contract_level"] == "fixed_ttsirt"
    assert payload["defensive_mass_positive"] is True
    assert (
        payload["proposition2_marginal_backend"]
        == "paired_core_mass_contraction_prefix_suffix"
    )
    assert payload["conditional_cdf_backend"] == "numerical_grid_trapezoid_bisection"
    assert (
        payload["conditional_cdf_route_class"]
        == "extension_or_invention_diagnostic_approximation"
    )
    assert payload["production_kr_closure"] is False
    assert payload["proposal_density_backend"] == "eval_pdf_on_local_samples"
    assert "no production KR closure" in payload["p83_nonclaims"]

    direct = highdim.p83_minimal_transport_slice_readiness(payload)
    protocol = highdim.SourceRouteTransportProtocol(fixed)
    wrapped = highdim.p83_minimal_transport_slice_readiness(
        protocol.manifest_payload()
    )

    assert direct.status == highdim.P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS
    assert direct.ready_for_p83_minimal_slice is True
    assert wrapped.status == highdim.P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS


def test_p83_readiness_blocks_zero_defensive_mass_and_production_kr_claim() -> None:
    zero_payload = _fixed_transport(_nonuniform_density(tau=0.0)).manifest_payload()
    zero_result = highdim.p83_minimal_transport_slice_readiness(zero_payload)

    production_payload = dict(_fixed_transport(_nonuniform_density(tau=0.25)).manifest_payload())
    production_payload["production_kr_closure"] = True
    production_result = highdim.p83_minimal_transport_slice_readiness(
        production_payload
    )

    assert zero_result.status == highdim.P83_MINIMAL_TRANSPORT_SLICE_BLOCK_STATUS
    assert "missing_positive_defensive_mass" in zero_result.blockers
    assert production_result.status == highdim.P83_MINIMAL_TRANSPORT_SLICE_BLOCK_STATUS
    assert "production_kr_closure_not_false" in production_result.blockers

    with pytest.raises(ValueError, match="ready status"):
        highdim.P83MinimalTransportSliceReadiness(
            status=highdim.P83_MINIMAL_TRANSPORT_SLICE_READY_STATUS,
            blockers=("hidden_blocker",),
            manifest=zero_payload,
        )


def test_p83_proposal_correction_uses_eval_pdf_not_reference_density() -> None:
    fixed = _fixed_transport(_nonuniform_density(tau=0.25))
    transport = highdim.SourceRouteTransportProtocol(fixed)
    reference = tf.constant([[0.2, 0.8], [0.3, 0.7]], dtype=DTYPE)
    local = transport.inverse_transport(reference)

    proposal = transport.proposal_log_density(
        local_points=local,
        reference_points=reference,
    )
    expected = tf.math.log(transport.eval_pdf(local))
    base_uniform_log_density = tf.zeros([2], dtype=DTYPE)

    tf.debugging.assert_near(proposal, expected)
    assert bool(tf.reduce_any(tf.abs(proposal - base_uniform_log_density) > 1e-6).numpy())


def test_p83_previous_marginal_uses_paired_core_marginal_evaluator() -> None:
    fixed = _fixed_transport(_constant_density(3, tau=0.25))
    marginal = fixed.marginalize((0, 1))
    local = tf.constant([[-0.5, 0.25], [0.0, 0.5]], dtype=DTYPE)

    values = marginal.normalized_retained_density_values(tf.transpose(local))

    assert marginal.diagnostics["semantics"] == "source_style_squared_tt_marginal"
    assert "mass-matrix recursion" in marginal.diagnostics["source_anchor"]
    tf.debugging.assert_near(values, tf.ones([2], dtype=DTYPE))


def test_p83_two_step_fixed_ttsirt_retained_object_mechanics_nonclaim() -> None:
    components1 = _components(include_prior=True)
    components2 = _components(include_prior=False)
    spec1 = highdim.SourceRouteSequentialStepSpec(
        target=_target(time_index=1, shift_constant=0.2, components=components1),
        transport=_protocol(),
        reference_samples=tf.constant(
            [
                [0.2, 0.6],
                [0.3, 0.7],
                [0.4, 0.8],
            ],
            dtype=DTYPE,
        ),
        measure_convention=_convention(),
        density_components=components1,
    )
    spec2 = highdim.SourceRouteSequentialStepSpec(
        target=_target(time_index=2, shift_constant=-0.1, components=None),
        transport=_protocol(),
        reference_samples=tf.constant(
            [
                [0.25, 0.75],
                [0.35, 0.65],
                [0.45, 0.85],
            ],
            dtype=DTYPE,
        ),
        measure_convention=_convention(),
        density_components=components2,
        previous_marginal_keep_axes=(0, 1),
        previous_marginal_input_axes=(0, 2),
    )

    result = highdim.source_route_run_sequential_fixed_hmc(
        step_specs=(spec1, spec2)
    )
    transport_payload = result.steps[0].retained_object.transport_object.manifest_payload()

    assert result.steps[1].previous_retained_object is result.steps[0].retained_object
    assert result.steps[1].previous_marginal_density is not None
    assert result.steps[1].previous_marginal_density.keep_axes == (0, 1)
    assert transport_payload["production_kr_closure"] is False
    assert "no d18 correctness" in transport_payload["p83_nonclaims"]
