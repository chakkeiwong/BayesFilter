def _refined_symmetric_eigh(matrix: tf.Tensor, *, sweeps: int=4) -> tuple[tf.Tensor, tf.Tensor]:
    """Refine XLA's float64 eigensystem; return NaN on failed residual checks.

    Four sweeps and the residual bound are the tested Phase 9B repair, not a
    universal eigensolver guarantee. See the 2026-09-11 runtime-health plan.
    Callers own the stable tf.function signature and XLA compilation.
    """
    matrix = _symmetrize(tf.convert_to_tensor(matrix, dtype=tf.float64))
    if matrix.shape.rank != 3 or matrix.shape[-1] is None:
        raise ValueError('refined eigensolver requires a rank-3 static matrix')
    dimension = int(matrix.shape[-1])
    if dimension < 1 or matrix.shape[-2] != dimension:
        raise ValueError('refined eigensolver requires square matrices')
    if type(sweeps) is not int or sweeps <= 0:
        raise ValueError('refined eigensolver sweeps must be a positive integer')
    if dimension == 1:
        return (tf.linalg.diag_part(matrix), tf.ones_like(matrix))
    finite_rows = tf.reduce_all(tf.math.is_finite(matrix), axis=[-2, -1])
    safe_matrix = tf.where(finite_rows[:, tf.newaxis, tf.newaxis], matrix, tf.eye(dimension, dtype=tf.float64)[tf.newaxis])
    tournament_size = dimension + dimension % 2
    half = dimension // 2
    paired = 2 * half
    circle = list(range(tournament_size))
    permutations: list[list[int]] = []
    for _ in range(tournament_size - 1):
        pairs = [(first, second) for first, second in zip(circle[:tournament_size // 2], reversed(circle[tournament_size // 2:])) if first < dimension and second < dimension]
        active_indices = [first for first, _ in pairs] + [second for _, second in pairs]
        permutations.append(active_indices + [index for index in range(dimension) if index not in active_indices])
        circle = [circle[0], circle[-1], *circle[1:-1]]
    schedule = tf.constant(permutations, tf.int32)
    inverse_schedule = tf.argsort(schedule, axis=-1)
    _, vectors = tf.linalg.eigh(safe_matrix)
    working = tf.linalg.matrix_transpose(vectors) @ safe_matrix @ vectors
    working = _symmetrize(working)

    def rotate(index: tf.Tensor, current: tf.Tensor, current_vectors: tf.Tensor):
        round_index = tf.math.floormod(index, tournament_size - 1)
        permutation = schedule[round_index]
        inverse = inverse_schedule[round_index]
        block = tf.gather(tf.gather(current, permutation, axis=1), permutation, axis=2)
        reordered_vectors = tf.gather(current_vectors, permutation, axis=2)
        first_diagonal = tf.linalg.diag_part(block[:, :half, :half])
        second_diagonal = tf.linalg.diag_part(block[:, half:paired, half:paired])
        off_diagonal = tf.linalg.diag_part(block[:, :half, half:paired])
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
        rotated_columns = tf.concat((block[:, :, :half] * cosine[:, tf.newaxis, :] - block[:, :, half:paired] * sine[:, tf.newaxis, :], block[:, :, :half] * sine[:, tf.newaxis, :] + block[:, :, half:paired] * cosine[:, tf.newaxis, :], block[:, :, paired:]), axis=2)
        rotated = tf.concat((rotated_columns[:, :half, :] * cosine[:, :, tf.newaxis] - rotated_columns[:, half:paired, :] * sine[:, :, tf.newaxis], rotated_columns[:, :half, :] * sine[:, :, tf.newaxis] + rotated_columns[:, half:paired, :] * cosine[:, :, tf.newaxis], rotated_columns[:, paired:, :]), axis=1)
        first_block = tf.linalg.set_diag(rotated[:, :half, :half], first_diagonal - tangent * off_diagonal)
        second_block = tf.linalg.set_diag(rotated[:, half:paired, half:paired], second_diagonal + tangent * off_diagonal)
        upper_block = tf.linalg.set_diag(rotated[:, :half, half:paired], tf.zeros_like(off_diagonal))
        lower_block = tf.linalg.set_diag(rotated[:, half:paired, :half], tf.zeros_like(off_diagonal))
        rotated = tf.concat((tf.concat((first_block, upper_block, rotated[:, :half, paired:]), axis=2), tf.concat((lower_block, second_block, rotated[:, half:paired, paired:]), axis=2), rotated[:, paired:, :]), axis=1)
        rotated_vectors = tf.concat((reordered_vectors[:, :, :half] * cosine[:, tf.newaxis, :] - reordered_vectors[:, :, half:paired] * sine[:, tf.newaxis, :], reordered_vectors[:, :, :half] * sine[:, tf.newaxis, :] + reordered_vectors[:, :, half:paired] * cosine[:, tf.newaxis, :], reordered_vectors[:, :, paired:]), axis=2)
        next_current = tf.gather(tf.gather(rotated, inverse, axis=1), inverse, axis=2)
        next_vectors = tf.gather(rotated_vectors, inverse, axis=2)
        return (index + 1, next_current, next_vectors)
    _, working, vectors = tf.while_loop(lambda index, _current, _vectors: index < sweeps * (tournament_size - 1), rotate, (tf.constant(0, tf.int32), working, vectors), parallel_iterations=1)
    values = tf.linalg.diag_part(working)
    order = tf.argsort(values, axis=-1)
    values = tf.gather(values, order, batch_dims=1)
    vectors = tf.gather(vectors, order, axis=2, batch_dims=1)
    residual = safe_matrix @ vectors - vectors * values[:, tf.newaxis, :]
    orthogonality = tf.linalg.matrix_transpose(vectors) @ vectors - tf.eye(dimension, dtype=tf.float64)[tf.newaxis]
    roundoff_bound = tf.constant(64.0 * dimension * math.ulp(1.0), tf.float64)
    residual_norm = tf.linalg.norm(residual, axis=[-2, -1])
    matrix_norm = tf.linalg.norm(safe_matrix, axis=[-2, -1])
    valid = finite_rows & tf.math.is_finite(residual_norm) & tf.math.is_finite(matrix_norm) & (residual_norm <= roundoff_bound * matrix_norm) & (tf.linalg.norm(orthogonality, axis=[-2, -1]) <= roundoff_bound)
    return (values, vectors, valid, residual_norm, matrix_norm, tf.linalg.norm(orthogonality, axis=[-2, -1]), roundoff_bound, working)
