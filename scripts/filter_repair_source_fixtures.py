"""Frozen source-route weight and coordinate calculations for comparison."""

FIXTURES = ("source_route_weights", "source_route_retained", "source_route_previous", "source_route_sequence",
            "source_guard_gates")


def _gate_fixture(tf, size, jit, public_boundary):
    """The same public gate statistics, with an explicit compiled-kernel arm."""
    from importlib.util import find_spec

    from bayesfilter.highdim import source_route as source

    fitted = _transport(tf)[0].density.sqrt_tt
    rows = 16 * size
    fit = tf.reshape(tf.linspace(tf.constant(-.8, tf.float64), .7, 2*rows), [2, rows])
    points = tf.reshape(tf.linspace(tf.constant(-.6, tf.float64), .9, 2*rows), [2, rows])
    targets = tf.linspace(tf.constant(.8, tf.float64), 1.2, rows)
    starts = tf.constant([.9, 1.1], tf.float64)
    indices = tf.math.floormod(tf.range(rows), 2)
    spectra = tuple(tf.constant([2., 1., 1e-14], tf.float64) for _ in range(4*size))
    support_fields = ("nearest_fit_distance_min", "nearest_fit_distance_median",
                      "nearest_fit_distance_max", "fit_leave_one_out_distance_max",
                      "point_any_saturated_fraction")
    line_fields = ("line_prediction_max_abs", "line_residual_max_abs", "line_residual_rms",
                   "endpoint_growth_ratio_max")
    has_native = find_spec("bayesfilter.highdim.source_route_gate_runtime_tf") is not None
    if has_native and not public_boundary:
        from bayesfilter.highdim import source_route_gate_runtime_tf as native

        line = native.line_probe_program(fitted, (2, rows), (2,), (rows,), "indices", jit_compile=jit)
        rank = native.spectrum_rank_program(tuple(tuple(value.shape) for value in spectra), jit_compile=jit)

        def evaluate(fit_cloud, cloud, target_values, start_values, start_indices, *singular_values):
            support = native.support_statistics.python_function(cloud, fit_cloud)
            line_values = line.python_function(cloud, target_values, start_values, start_indices,
                                                tf.constant(1., tf.float64))
            ranks, _ = rank.python_function(singular_values, tf.constant(source.P72_EFFECTIVE_RANK_TOL, tf.float64))
            return tf.stack((*support[3:], *line_values[1:5], tf.constant(100., tf.float64),
                             tf.reduce_min(ranks)))
        evaluate.timing_scope = "complete_numerical_source_gate_statistics"
        evaluate.numerical_execution = "xla" if jit else "graph_reference"
    else:
        def evaluate(fit_cloud, cloud, target_values, start_values, start_indices, *singular_values):
            support = source.p72_support_clipping_coverage(role="guard", points=cloud, fit_points=fit_cloud)
            line = source.p72_line_probe_diagnostics(fitted_tt=fitted, line_points=cloud,
                line_target_values=target_values, start_prediction_values=start_values,
                line_start_indices=start_indices, target_scale=1.)
            records = tuple({"condition_number": 100., "scaled_augmented_singular_values": value}
                            for value in singular_values)
            condition = source.p72_condition_effective_rank_gate(records)
            return tf.stack((*tuple(tf.constant(support[key], tf.float64) for key in support_fields),
                *tuple(tf.constant(line[key], tf.float64) for key in line_fields),
                tf.constant(condition["condition_number_max"], tf.float64),
                tf.constant(condition["effective_rank_min"], tf.float64)))
        evaluate.timing_scope = "complete_public_source_gate_statistics_and_provenance"
        evaluate.numerical_execution = "default_xla_gate_programs" if has_native else "legacy_host_gates"
    return evaluate, (fit, points, targets, starts, indices, *spectra), {
        "rows": rows, "dimension": 2, "spectra": len(spectra),
        "fields": (*support_fields, *line_fields, "condition_number_max", "effective_rank_min"),
        "boundary": "public_source_gate_numeric_fields; decisions_and_hashes_checked_in_correctness_suite",
        "classification": "extension_or_invention_execution_only", "canonical_admitted": False}


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
    arguments = {"sqrt_tt": ftt,
        "defensive_density": highdim.TensorProductReferenceDensity(basis, convention),
        "tau": tf.constant(.05, tf.float64), "normalizer_floor": tf.constant(1e-12, tf.float64),
        "denominator_floor": tf.constant(1e-12, tf.float64), "measure_convention": convention}
    density = highdim.SquaredTTDensity(**arguments,
        branch_identity=highdim.SquaredTTDensity.expected_branch_identity(**arguments))
    return highdim.FixedTTSIRTTransport(density, highdim.KRCDFConfig(
        grid_size=9, bisection_steps=8, monotonicity_tolerance=1e-12,
        bracket_tolerance=1e-12, denominator_floor=1e-12, max_floor_count=0)), convention


def fixture(tf, name, size, jit, *, public_boundary=False):
    if name not in FIXTURES:
        raise ValueError(name)
    if name == "source_guard_gates":
        return _gate_fixture(tf, size, jit, public_boundary)
    from bayesfilter.highdim import source_route as source

    rows = 4 * size
    matrix = tf.constant([[1.3, .2], [-.1, .7]], tf.float64)
    mu = tf.constant([.1, -.2], tf.float64)
    frame = source.SourceRouteCoordinateFrame(mu=mu, matrix=matrix, expansion_factor=1.)

    def density(points):
        return .5 * tf.reduce_sum(tf.square(points), axis=0)

    target = source.build_source_route_target(negative_log_physical_density_fn=density,
        coordinate_frame=frame, shift_constant=tf.constant(.35, tf.float64), time_index=1)

    if name == "source_route_sequence":
        from dataclasses import replace
        from importlib.util import find_spec

        transport, convention = _transport(tf)
        protocol = source.SourceRouteTransportProtocol(transport)
        points = tf.reshape(tf.linspace(tf.constant(.15, tf.float64), .85, 8), [2, 4])
        components = source.SourceRouteSequentialDensityComponents(parameter_dim=0, state_dim=1,
            transition_log_density_fn=lambda x, t: -.1 * (x[0] - x[1])**2,
            likelihood_log_density_fn=lambda x, t: -.2 * x[0]**2,
            prior_log_density_fn=lambda x: -.1 * x[0]**2)
        first = source.SourceRouteSequentialStepSpec(target=target, transport=protocol,
            reference_samples=points, measure_convention=convention, density_components=components)
        count = 2 * size
        specs = (first, *(replace(first, target=replace(target, time_index=index+1),
            density_components=replace(components, prior_log_density_fn=None),
            previous_marginal_keep_axes=(0,), previous_marginal_input_axes=(1,))
            for index in range(1, count)))
        has_native_dates = find_spec("bayesfilter.highdim.source_route_sequential_tf") is not None
        if not public_boundary and has_native_dates:
            from bayesfilter.highdim.source_route_sequential_tf import (
                sequential_program,
                unpack_step,
            )

            program, prepared, lengths = sequential_program(specs, jit_compile=jit)

            def evaluate(*queries):
                packed = program.python_function(*queries) if tf.inside_function() else program(*queries)
                # Fixed output schema only; every date calculation is in program.
                fields = tuple(unpack_step(packed[index], row[2], lengths[index])
                               for index, row in enumerate(prepared))
                return tuple(row[:7] if index == 0 else row for index, row in enumerate(fields))
            evaluate.timing_scope = "complete_numerical_date_kernel"
            evaluate.numerical_execution = "xla" if jit else "graph_reference"
        else:
            def evaluate(*queries):
                # The same public endpoint in each source arm, including its
                # result assembly. Candidate numerical execution defaults XLA.
                current = tuple(replace(spec, reference_samples=query)
                                for spec, query in zip(specs, queries, strict=True))
                result = source.source_route_run_sequential_fixed_hmc(step_specs=current)

                def fields(step):
                    value = step.retained_samples
                    previous = step.previous_marginal_density
                    return (value.retained_batch.samples, value.proposal_log_density,
                        value.target_log_density, value.correction_log_weights, value.retained_batch.log_weights,
                        value.diagnostics.effective_sample_size, value.normalizer.log_transport_normalizer,
                        *((previous.physical_points, previous.local_points, previous.log_density)
                          if previous is not None else ()))
                return tuple(fields(step) for step in result.steps)
            evaluate.timing_scope = "complete_public_date_endpoint"
            evaluate.numerical_execution = "default_xla_date_program" if has_native_dates else "legacy_host_date_loop"

        return evaluate, tuple(spec.reference_samples for spec in specs), {
            "rows": 4, "dimension": 2, "dates": count,
            "boundary": "complete_frozen_source_date_values_with_previous_marginals",
            "classification": "fixed_hmc_adaptation_execution_only", "canonical_admitted": False}

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
