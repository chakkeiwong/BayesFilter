"""Diagnostic-only binary64 Jacobi refinement; not an admitted eigensolver."""

import tensorflow as tf


def refine_eigh(matrix, *, sweeps=3):
    """Refine a static even-size batch inside the caller's signed XLA graph."""
    size = int(matrix.shape[-1])
    if matrix.dtype != tf.float64 or matrix.shape.rank != 3 or size % 2:
        raise ValueError("This diagnostic requires even-size batched float64 matrices")
    half = size // 2
    circle = list(range(size))
    permutations = []
    for _ in range(size - 1):
        permutations.append(circle[:half] + list(reversed(circle[half:])))
        circle = [circle[0], circle[-1], *circle[1:-1]]
    pairs = {tuple(sorted((permutation[index], permutation[index + half])))
             for permutation in permutations for index in range(half)}
    if len(pairs) != size * (size - 1) // 2:
        raise RuntimeError("Round-robin schedule does not cover all index pairs")
    schedule = tf.constant(permutations, tf.int32)
    inverse_schedule = tf.argsort(schedule, axis=-1)
    _, vectors = tf.linalg.eigh(matrix)
    working = tf.linalg.matrix_transpose(vectors) @ matrix @ vectors
    working = 0.5 * (working + tf.linalg.matrix_transpose(working))

    def rotate(index, working, vectors):
        round_index = tf.math.floormod(index, size - 1)
        permutation = schedule[round_index]
        inverse = inverse_schedule[round_index]
        block = tf.gather(tf.gather(working, permutation, axis=1), permutation, axis=2)
        reordered_vectors = tf.gather(vectors, permutation, axis=2)
        first_diagonal = tf.linalg.diag_part(block[:, :half, :half])
        second_diagonal = tf.linalg.diag_part(block[:, half:, half:])
        off_diagonal = tf.linalg.diag_part(block[:, :half, half:])
        delta = second_diagonal - first_diagonal
        twice_off_diagonal = 2.0 * off_diagonal
        scale = tf.maximum(tf.abs(delta), tf.abs(twice_off_diagonal))
        safe_scale = tf.where(scale > 0.0, scale, tf.ones_like(scale))
        scaled_delta = delta / safe_scale
        scaled_off_diagonal = twice_off_diagonal / safe_scale
        radius = tf.sqrt(tf.square(scaled_delta) + tf.square(scaled_off_diagonal))
        denominator = scaled_delta + tf.where(scaled_delta >= 0.0, radius, -radius)
        tangent = scaled_off_diagonal / tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
        cosine = tf.math.rsqrt(1.0 + tf.square(tangent))
        sine = tangent * cosine
        rotated_columns = tf.concat((
            block[:, :, :half] * cosine[:, None, :] - block[:, :, half:] * sine[:, None, :],
            block[:, :, :half] * sine[:, None, :] + block[:, :, half:] * cosine[:, None, :],
        ), axis=2)
        rotated = tf.concat((
            rotated_columns[:, :half, :] * cosine[:, :, None] - rotated_columns[:, half:, :] * sine[:, :, None],
            rotated_columns[:, :half, :] * sine[:, :, None] + rotated_columns[:, half:, :] * cosine[:, :, None],
        ), axis=1)
        first_block = tf.linalg.set_diag(rotated[:, :half, :half], first_diagonal - tangent * off_diagonal)
        second_block = tf.linalg.set_diag(rotated[:, half:, half:], second_diagonal + tangent * off_diagonal)
        upper_block = tf.linalg.set_diag(rotated[:, :half, half:], tf.zeros_like(off_diagonal))
        lower_block = tf.linalg.set_diag(rotated[:, half:, :half], tf.zeros_like(off_diagonal))
        rotated = tf.concat((tf.concat((first_block, upper_block), axis=2), tf.concat((lower_block, second_block), axis=2)), axis=1)
        rotated_vectors = tf.concat((
            reordered_vectors[:, :, :half] * cosine[:, None, :] - reordered_vectors[:, :, half:] * sine[:, None, :],
            reordered_vectors[:, :, :half] * sine[:, None, :] + reordered_vectors[:, :, half:] * cosine[:, None, :],
        ), axis=2)
        working = tf.gather(tf.gather(rotated, inverse, axis=1), inverse, axis=2)
        vectors = tf.gather(rotated_vectors, inverse, axis=2)
        return index + 1, working, vectors

    _, working, vectors = tf.while_loop(
        lambda index, working, vectors: index < sweeps * (size - 1),
        rotate, (tf.constant(0), working, vectors), parallel_iterations=1,
    )
    values = tf.linalg.diag_part(working)
    order = tf.argsort(values, axis=-1)
    return tf.gather(values, order, batch_dims=1), tf.gather(vectors, order, axis=2, batch_dims=1)
