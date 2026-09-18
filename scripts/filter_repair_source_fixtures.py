"""Frozen source-route weight and coordinate calculations for comparison."""

FIXTURES = ("source_route_weights", "source_route_retained", "source_route_previous")


def _transport(tf):
    from bayesfilter import highdim

    convention = highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE, reference_weight_name="omega")
    basis = highdim.ProductBasis(tuple(highdim.LegendreBasis1D(
        highdim.BoundedInterval(-1., 1.), 1) for _ in range(2)), convention)
    cores = (highdim.TTCore(tf.constant([[[1., 0.], [0., 1.]]], tf.float64)),
             highdim.TTCore(tf.constant([[[1.], [0.]], [[0.], [.1]]], tf.float64)))
    ftt = highdim.FunctionalTT(cores, basis, convention)
    arguments = dict(sqrt_tt=ftt,
        defensive_density=highdim.TensorProductReferenceDensity(basis, convention),
        tau=tf.constant(.05, tf.float64), normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64), measure_convention=convention)
    density = highdim.SquaredTTDensity(**arguments,
        branch_identity=highdim.SquaredTTDensity.expected_branch_identity(**arguments))
    return highdim.FixedTTSIRTTransport(density, highdim.KRCDFConfig(
        grid_size=9, bisection_steps=8, monotonicity_tolerance=1e-12,
        bracket_tolerance=1e-12, denominator_floor=1e-12, max_floor_count=0)), convention


def fixture(tf, name, size, jit):
    del jit
    if name not in FIXTURES:
        raise ValueError(name)
    from bayesfilter.highdim import source_route as source

    rows = 4 * size
    matrix = tf.constant([[1.3, .2], [-.1, .7]], tf.float64)
    mu = tf.constant([.1, -.2], tf.float64)
    frame = source.SourceRouteCoordinateFrame(mu=mu, matrix=matrix, expansion_factor=1.)

    def density(points):
        return .5 * tf.reduce_sum(tf.square(points), axis=0)

    target = source.build_source_route_target(negative_log_physical_density_fn=density,
        coordinate_frame=frame, shift_constant=tf.constant(.35, tf.float64), time_index=1)

    if name != "source_route_weights":
        transport, convention = _transport(tf)
        protocol = source.SourceRouteTransportProtocol(transport)
        points = tf.reshape(tf.linspace(tf.constant(.15, tf.float64), .85, 2 * rows), [2, rows])
        if name == "source_route_retained":
            def evaluate(query):
                value = source.source_route_generate_retained_samples(target=target,
                    transport=protocol, reference_samples=query, time_index=1)
                return (value.retained_batch.samples, value.proposal_log_density,
                    value.target_log_density, value.correction_log_weights, value.retained_batch.log_weights,
                    value.diagnostics.effective_sample_size, value.normalizer.log_increment())
        else:
            components = source.SourceRouteSequentialDensityComponents(parameter_dim=0, state_dim=1,
                transition_log_density_fn=lambda x, t: -.1 * x[0]**2,
                likelihood_log_density_fn=lambda x, t: -.2 * x[0]**2,
                prior_log_density_fn=lambda x: -.1 * x[0]**2)
            spec = source.SourceRouteSequentialStepSpec(target=target, transport=protocol,
                reference_samples=points, measure_convention=convention, density_components=components)
            previous = source._p59_retained_object_from_spec(spec)
            points = points[:1]

            def evaluate(query):
                with tf.GradientTape() as tape:
                    tape.watch(query)
                    value = source.source_route_previous_marginal_log_density(
                        previous_retained_object=previous, physical_points=query, keep_axes=(0,))
                    loss = tf.reduce_sum(value.log_density)
                return value.local_points, value.log_density, tape.gradient(loss, query)

        return evaluate, (points,), {"rows": rows, "dimension": 2,
            "boundary": name + "_complete_values_and_declared_pullback",
            "classification": "fixed_hmc_adaptation_execution_only", "canonical_admitted": False}

    def evaluate(points, proposal):
        physical = target.physical_points_from_reference(points)
        log_target = source.source_route_reference_log_density_from_physical(
            log_physical_density=-density(physical), coordinate_frame=frame)
        shifted = source.source_route_shifted_negative_log_target(
            negative_log_target=-log_target, shift_constant=target.shift_constant)
        correction = source.source_route_proposal_log_weights(log_target_density=-shifted,
            log_proposal_density=proposal)
        return (physical, shifted, correction, source.normalize_log_weights(correction),
                source.effective_sample_size_from_log_weights(correction),
                source.source_route_equal_weight_log_normalizer_estimate(correction),
                source.source_route_log_normalizer_update(log_transport_normalizer=tf.constant(.8, tf.float64),
                    shift_constant=target.shift_constant))

    points = tf.reshape(tf.linspace(tf.constant(-.5, tf.float64), .4, 2 * rows), [2, rows])
    proposal = tf.linspace(tf.constant(-.2, tf.float64), .3, rows)
    return evaluate, (points, proposal), {"rows": rows, "dimension": 2,
        "boundary": "source_affine_density_shift_proposal_weights_ess_normalizer",
        "classification": "fixed_hmc_adaptation_execution_only", "canonical_admitted": False}
