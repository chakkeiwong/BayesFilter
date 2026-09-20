"""Matched diagnostic scopes for batched localization and optional progress."""

FIXTURES = ("sequential_batched_locator", "sequential_batched_locator_progress")


def fixture(tf, name, size, jit, *, public_boundary=False):
    from bayesfilter.inference import sequential_map_covariance as geometry

    if name not in FIXTURES:
        raise ValueError(name)
    progress = name.endswith("_progress")
    dtype = tf.float64
    count, dimension = 2 * size, 2
    config = geometry.SequentialMapCovarianceConfig(locator_max_iterations=4,
        locator_max_line_search_iterations=7, max_exact_evaluations=1)
    starts = tf.reshape(tf.linspace(tf.constant(-.4, dtype), tf.constant(.3, dtype),
        count * dimension), [count, dimension])
    scale = tf.constant([.7, 1.3], dtype)
    precision = tf.constant([[2., .3], [.3, 4.]], dtype)
    mode = tf.constant([.15, -.2], dtype)

    def scalar(point):
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    def batched(points):
        delta = points - mode
        scores = -delta @ precision
        return .5 * tf.reduce_sum(delta * scores, axis=1), scores

    native = hasattr(geometry, "batched_locator_program")
    if native and not public_boundary:
        arguments = (scalar, batched, count, dimension,
            config.locator_standardized_box_radius, config.locator_gradient_tolerance,
            config.locator_max_iterations, config.locator_max_line_search_iterations,
            config.locator_stopping_condition)
        if progress:
            owner = geometry.buffered_batched_locator_program(*arguments, device=starts.device, jit_compile=jit)
            program = owner.compiled.python_function
        else:
            program = geometry.batched_locator_program(*arguments, jit_compile=jit).python_function

        def evaluate(starts, scale):
            result = program(starts, scale)
            diagnostics = tf.stack([tf.cast(result["converged"], tf.int32),
                tf.cast(result["failed"], tf.int32), tf.fill([count], result["iterations"]),
                tf.fill([count], result["objective_calls"])], axis=1)
            return (result["selected"]["position"],
                result["exact_evaluations"] + result["objective_evaluations"], diagnostics,
                result["endpoint_finite"], result["endpoint_standardized_norm"],
                result["trace_count"] if progress else tf.constant(0, tf.int64),
                result["trace_overflow"] if progress else tf.constant(False))
    else:
        def evaluate(starts, scale):
            observations = []
            result = geometry.estimate_sequential_map_covariance(scalar, starts,
                batched_locator_value_and_score_fn=batched, scale=scale, config=config,
                progress_callback=observations.append if progress else None)
            if result.status != "maximum_exact_evaluations_after_bounded_locator":
                raise RuntimeError("The matched diagnostic must stop after locator accounting")
            reports = result.diagnostics["locator"]
            return (tf.convert_to_tensor(result.map_candidate, dtype),
                tf.constant(result.diagnostics["exact_evaluations"]),
                tf.constant([[int(row["converged"]), int(row["failed"]), row["iterations"],
                    row["objective_calls"]] for row in reports]),
                tf.constant([row["finite"] for row in reports]),
                tf.constant([row["endpoint_standardized_norm"] for row in reports], dtype),
                tf.constant(sum(event["stage"] == "locator_objective_completed" for event in observations), tf.int64),
                tf.constant(False))

    evaluate.timing_scope = ("complete_tensor_batched_locator_with_trace_writes" if progress else
        "complete_tensor_batched_locator") if native and not public_boundary else (
        "complete_public_batched_locator_with_progress_delivery" if progress else
        "complete_public_batched_locator_budget_rejection_records")
    evaluate.execution_backend = "tensorflow" if native else "legacy_python_batched_locator_diagnostic_only"
    return evaluate, (starts, scale), {"dimension": dimension, "starts": count,
        "max_iterations": config.locator_max_iterations,
        "max_line_search_iterations": config.locator_max_line_search_iterations,
        "gradient_tolerance": config.locator_gradient_tolerance,
        "box_radius": config.locator_standardized_box_radius,
        "progress_requested": progress, "trace_values_checked_by_focused_original_event_parity": True,
        "boundary": "complete_locator_budget_rejection_before_refinement", "random_inputs": False}
