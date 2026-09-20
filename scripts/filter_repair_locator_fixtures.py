"""Matched diagnostic measurements of original and compiled scalar localization."""

FIXTURES = ("sequential_scalar_locator",)


def fixture(tf, name, size, jit, *, public_boundary=False):
    from bayesfilter.inference import sequential_map_covariance as geometry

    if name not in FIXTURES:
        raise ValueError(name)
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

    native = hasattr(geometry, "scalar_locator_program")
    if native and not public_boundary:
        program = geometry.scalar_locator_program(scalar, count, dimension,
            config.locator_standardized_box_radius, config.locator_gradient_tolerance,
            config.locator_max_iterations, config.locator_max_line_search_iterations,
            config.locator_stopping_condition, jit_compile=jit).python_function

        def evaluate(starts, scale):
            result = program(starts, scale)
            return (result["selected"]["position"],
                result["exact_evaluations"] + result["objective_evaluations"],
                result["optimizer_diagnostics"], result["endpoint_finite"],
                result["endpoint_standardized_norm"])
    else:
        def evaluate(starts, scale):
            result = geometry.estimate_sequential_map_covariance(scalar, starts,
                scale=scale, config=config)
            if result.status != "maximum_exact_evaluations_after_bounded_locator":
                raise RuntimeError("The matched diagnostic must stop after locator accounting")
            reports = result.diagnostics["locator"]
            return (tf.convert_to_tensor(result.map_candidate, dtype),
                tf.constant(result.diagnostics["exact_evaluations"]),
                tf.constant([[int(row["converged"]), int(row["failed"]), row["iterations"],
                    row["objective_evaluations"]] for row in reports]),
                tf.constant([row["finite"] for row in reports]),
                tf.constant([row["endpoint_standardized_norm"] for row in reports], dtype))

    evaluate.timing_scope = ("complete_tensor_scalar_locator" if native and not public_boundary
        else "complete_public_scalar_locator_budget_rejection_records")
    evaluate.execution_backend = "tensorflow" if native else "legacy_python_scalar_locator_diagnostic_only"
    return evaluate, (starts, scale), {"dimension": dimension, "starts": count,
        "max_iterations": config.locator_max_iterations,
        "max_line_search_iterations": config.locator_max_line_search_iterations,
        "gradient_tolerance": config.locator_gradient_tolerance,
        "box_radius": config.locator_standardized_box_radius,
        "boundary": "complete_locator_budget_rejection_before_refinement", "random_inputs": False}
