"""Native execution of the existing centered-density quadratic solve.

This preserves the finite conjugate-gradient algorithm, convergence threshold,
curvature veto and trace schedule. It makes no Zhao-Cui source-faithfulness
claim. The callback is the same fixed affine quadratic gradient as before.
"""

from functools import lru_cache

import tensorflow as tf

D = tf.float64


@lru_cache(maxsize=16)
def quadratic_cg_program(value_and_gradient, specification, max_iterations, trace_interval):
    """One bounded specialization of the complete callback/solve recurrence."""
    @tf.function(input_signature=[specification, tf.TensorSpec([], D)],
                 jit_compile=True, autograph=False)
    def solve(initial_position, tolerance):
        _, affine = value_and_gradient(tf.zeros_like(initial_position))

        def action(vector):
            _, gradient = value_and_gradient(vector)
            result = gradient - affine
            return result, tf.reduce_all(tf.math.is_finite(result))

        rhs = -affine
        initial_action, finite = action(initial_position)
        residual = rhs - initial_action
        squared = tf.reduce_sum(tf.square(residual))
        initial_norm = tf.sqrt(squared)
        scale = tf.maximum(tf.linalg.norm(rhs), tf.constant(1., D))
        threshold = tolerance * scale
        converged = initial_norm <= threshold
        norms = tf.TensorArray(D, max_iterations + 1, element_shape=[]).unstack(
            tf.zeros([max_iterations + 1], D)).write(0, initial_norm)
        keep = tf.TensorArray(tf.bool, max_iterations + 1, element_shape=[]).unstack(
            tf.zeros([max_iterations + 1], tf.bool)).write(0, True)

        def step(iteration, position, residual, direction, squared, converged,
                 failed, completed, minimum, finite, norms, keep):
            product, valid = action(direction)
            finite = finite & valid
            curvature = tf.tensordot(direction, product, 1)
            minimum = tf.minimum(minimum, curvature)
            failed = ~tf.math.is_finite(curvature) | (curvature <= 0.)

            def update():
                step_size = squared / curvature
                next_position = position + step_size * direction
                next_residual = residual - step_size * product
                next_squared = tf.reduce_sum(tf.square(next_residual))
                norm = tf.sqrt(next_squared)
                done = norm <= threshold
                # The baseline leaves direction unchanged on convergence.
                next_direction = tf.cond(done, lambda: direction,
                    lambda: next_residual + (next_squared / squared) * direction)
                record = done | (iteration % trace_interval == 0) | (iteration == max_iterations)
                return (next_position, next_residual, next_direction, next_squared,
                        done, iteration, norms.write(iteration, norm), keep.write(iteration, record))

            position, residual, direction, squared, converged, completed, norms, keep = tf.cond(
                failed, lambda: (position, residual, direction, squared, converged, completed, norms, keep),
                update)
            return (iteration + 1, position, residual, direction, squared, converged,
                    failed, completed, minimum, finite, norms, keep)

        state = tf.while_loop(
            lambda iteration, position, residual, direction, squared, converged,
                   failed, *_: (iteration <= max_iterations) & ~converged & ~failed,
            step, (tf.constant(1), initial_position, residual, residual, squared,
                   converged, tf.constant(False), tf.constant(0), tf.constant(float("inf"), D),
                   finite, norms, keep), maximum_iterations=max_iterations, parallel_iterations=1)
        _, position, _, _, squared, converged, failed, completed, minimum, finite, norms, keep = state
        final_norm = tf.sqrt(squared)
        return {"position": position, "converged": converged, "failed": failed,
                "num_iterations": completed, "initial_residual_norm": initial_norm,
                "residual_norm": final_norm, "relative_residual_norm": final_norm / scale,
                "minimum_curvature": minimum, "finite_actions": finite,
                "trace_norms": norms.stack(), "trace_relative": norms.stack() / scale,
                "trace_keep": keep.stack()}

    return solve
