"""Native execution of existing centered-density initialization and training.

This preserves the finite conjugate-gradient algorithm, convergence threshold,
curvature veto and trace schedule. It makes no Zhao-Cui source-faithfulness
claim. The callback is the same fixed affine quadratic gradient as before.
"""

from functools import lru_cache, wraps
from inspect import signature

import tensorflow as tf

from bayesfilter.ops.fixed_signature_tf import fixed_signature_function
from bayesfilter.ops.stateless_random_tf import philox_normal_float64

D = tf.float64


@tf.custom_gradient
def metric_sqrt(value):
    """Keep unused zero-valued metrics from contributing a 0/0 pullback.

    A compiled multi-output VJP receives zero cotangents for unused metrics.
    An active cotangent at zero retains the ordinary singular sqrt derivative.
    """
    root = tf.sqrt(value)
    return root, lambda cotangent: tf.math.xdivy(cotangent, 2. * root)


def _method_validity(trainer, operation, arguments, outputs):
    from bayesfilter.highdim.stochastic_training_native_tf import flat_cores

    valid = tf.reduce_all(tf.math.is_finite(flat_cores(tf.nest.flatten(outputs))))
    if operation == "absolute_density_loss":
        valid &= tf.reduce_all(arguments["batch"].theta[0] == 0.)
    if operation == "absolute_density_loss_arrays" and arguments["derivative_points"] is None:
        valid &= arguments["derivative_weight"] == 0.
    if operation in ("origin_global_score_metrics_arrays", "origin_prefix_score_metrics_arrays"):
        valid &= tf.reduce_all(arguments["score_standard_error"] >= 0.)
    return valid


def compiled_trainer_method(*, result_type=None, state_attribute="residual_variables"):
    """Use explicit mutable-core inputs and the existing compiled full pullback."""
    from bayesfilter.highdim.stochastic_training_native_tf import call_method

    def decorate(method):
        api = signature(method)

        @wraps(method)
        def evaluate(trainer, *args, **kwargs):
            if tf.inside_function():
                return method(trainer, *args, **kwargs)
            bound = api.bind(trainer, *args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.pop(next(iter(api.parameters)))
            # These options were cast by the original loss. In particular,
            # tf.cast(Python float, float64) preserves its float32 conversion.
            arguments = {name: tf.cast(value, D) if name in (
                "l1_weight", "l2_weight", "derivative_weight") else value
                for name, value in arguments.items()}
            return call_method(trainer, method.__name__, result_type=result_type,
                state_attribute=state_attribute, validity_fn=_method_validity, **arguments)

        return evaluate

    return decorate


@lru_cache(maxsize=16)
def default_residual_program(shape, component_count):
    """Scale only the first parent core for every existing residual feature."""
    @tf.function(input_signature=[tf.TensorSpec(shape, D)], jit_compile=True, autograph=False)
    def evaluate(parent):
        scale = tf.constant(1e-3, D) * tf.cast(tf.range(component_count) + 1, D)
        factors = tf.where(tf.range(shape[0])[None, :] == 0, scale[:, None], tf.constant(1., D))
        return parent[None] * factors[:, :, None, None, None]

    return evaluate


@lru_cache(maxsize=16)
def balanced_core_program(ranks, widths):
    dimension, rank, width = len(widths), max(ranks), max(widths)
    if widths[0] < 2 and ranks[1] > 1:
        raise ValueError("seeded channels require at least two first-axis basis functions")

    @tf.function(input_signature=[tf.TensorSpec([], D)], jit_compile=True, autograph=False)
    def evaluate(seeded_scale):
        axis = tf.range(dimension)[:, None, None, None]
        left = tf.range(rank)[None, :, None, None]
        basis = tf.range(width)[None, None, :, None]
        right = tf.range(rank)[None, None, None, :]
        left_rank = tf.constant(ranks[:-1])[:, None, None, None]
        right_rank = tf.constant(ranks[1:])[:, None, None, None]
        basis_width = tf.constant(widths)[:, None, None, None]
        shape_mask = (left < left_rank) & (right < right_rank) & (basis < basis_width)
        constant = (left == 0) & (right == 0)
        middle = (left > 0) & (left == right)
        last = (axis == dimension - 1) & (left > 0) & (right == 0)
        index = 1 + (axis + right - 1) % tf.maximum(basis_width - 1, 1)
        first = (axis == 0) & (left == 0) & (right > 0) & (basis == index)
        return tf.where(shape_mask, tf.cast(constant | middle | last, D) + tf.cast(first, D) * seeded_scale, 0.)

    return evaluate


@lru_cache(maxsize=16)
def residual_noise_program(shapes, component_count):
    dimension = len(shapes)
    rank = max(max(shape[0], shape[2]) for shape in shapes)
    width = max(shape[1] for shape in shapes)
    distinct_shapes = tuple(dict.fromkeys(shapes))
    shape_indices = tuple(distinct_shapes.index(shape) for shape in shapes)

    @tf.function(input_signature=[tf.TensorSpec([dimension, rank, width, rank], D),
        tf.TensorSpec([], tf.int32), tf.TensorSpec([], D), tf.TensorSpec([], D)],
        jit_compile=True, autograph=False)
    def evaluate(balanced, seed, amplitude, perturbation):
        def component(index):
            def axis_value(axis):
                def branch(shape):
                    def draw():
                        noise = philox_normal_float64(shape, tf.stack([seed + 104729 * index, axis + 1]))
                        return tf.pad(noise, [[0, rank-shape[0]], [0, width-shape[1]], [0, rank-shape[2]]])
                    return draw

                branches = tuple(branch(shape) for shape in distinct_shapes)
                noise = tf.switch_case(tf.gather(shape_indices, axis), branches)
                return balanced[axis] + perturbation * noise

            values = tf.map_fn(axis_value, tf.range(dimension),
                fn_output_signature=tf.TensorSpec([rank, width, rank], D), parallel_iterations=1)
            scale = tf.where(tf.range(dimension) == 0, amplitude, tf.constant(1., D))
            return values * scale[:, None, None, None]

        return tf.map_fn(component, tf.range(component_count),
            fn_output_signature=tf.TensorSpec([dimension, rank, width, rank], D), parallel_iterations=1)

    return evaluate


@lru_cache(maxsize=16)
def connected_channels_program(widths, target_rank, old_rank):
    dimension, width = len(widths), max(widths)
    if dimension < 2:
        raise ValueError("connected channels require at least two TT cores")

    @tf.function(input_signature=[tf.TensorSpec([dimension, target_rank, width, target_rank], D),
        tf.TensorSpec([], tf.int32), tf.TensorSpec([], D)], jit_compile=True, autograph=False)
    def evaluate(packed, seed, epsilon):
        def noises(channel):
            first = philox_normal_float64([widths[0]], tf.stack([seed + channel, 1]))
            last = philox_normal_float64([widths[-1]], tf.stack([seed + channel, 2]))
            return tf.stack([tf.pad(first, [[0, width - widths[0]]]),
                             tf.pad(last, [[0, width - widths[-1]]])])

        noise = tf.map_fn(noises, tf.range(old_rank, target_rank),
            fn_output_signature=tf.TensorSpec([2, width], D), parallel_iterations=1)
        channels = tf.one_hot(tf.range(old_rank, target_rank), target_rank, dtype=D)
        zero = tf.one_hot(0, target_rank, dtype=D)
        first = tf.einsum("a,kc,kb->abc", zero, channels, noise[:, 0]) * epsilon
        last = tf.einsum("ka,kb,c->abc", channels, noise[:, 1], zero) * epsilon
        middle = tf.einsum("ka,kc,b->abc", channels, channels, tf.ones([width], D))
        axis = tf.range(dimension)[:, None, None, None]
        addition = tf.where(axis == 0, first[None], tf.where(axis == dimension - 1, last[None], middle[None]))
        width_mask = tf.range(width)[None, None, :, None] < tf.constant(widths)[:, None, None, None]
        return packed + tf.where(width_mask, addition, tf.constant(0., D))

    return evaluate


@fixed_signature_function(floating_dtype=D)
def additive_core_banks(coefficients):
    """Pack every additive feature TT as [component, axis, left, basis, right]."""
    one, zero = tf.ones_like(coefficients), tf.zeros_like(coefficients)
    upper = tf.stack([one, coefficients], axis=-1)
    lower = tf.stack([zero, one], axis=-1)
    return tf.stack([upper, lower], axis=2)


@fixed_signature_function(floating_dtype=D)
def additive_pair_core_banks(additive, pair):
    """Encode the existing disjoint adjacent-pair automaton in one tensor."""
    components, dimension, width = additive.shape
    one, zero = tf.ones_like(additive), tf.zeros_like(additive)
    even = tf.range(dimension) % 2 == 0
    opening = tf.where(even[None, :, None, None],
        tf.broadcast_to(tf.eye(width, dtype=D), [components, dimension, width, width]), 0.)
    upper = tf.concat([one[..., None], additive[..., None], opening], axis=-1)
    lower = tf.concat([zero[..., None], one[..., None], tf.zeros_like(opening)], axis=-1)
    closing = tf.where(even[None, :, None, None], tf.constant(0., D),
                       tf.gather(pair, tf.range(dimension) // 2, axis=1))
    channels = tf.concat([tf.zeros_like(closing)[..., None], closing[..., None],
        tf.zeros([components, dimension, width, width, width], D)], axis=-1)
    return tf.concat([upper[:, :, None], lower[:, :, None], channels], axis=2)


def unpack_additive_bank(packed):
    """Restore the fixed boundary-rank schema after the batched encoding."""
    if packed.shape[0] == 1:
        return (packed[0, :1],)
    middle = () if packed.shape[0] == 2 else tuple(tf.unstack(packed[1:-1], axis=0))
    return (packed[0, :1], *middle, packed[-1, :, :, 1:2])


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
