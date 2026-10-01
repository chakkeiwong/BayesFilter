"""Historical diagnostic snapshot of the unqualified 01554 active-row solver.

Independent test reference only; never import this module in a runtime route.
The isolated sliced-dot branches below fail complete fitter parity on CPU.

Rank-aware least squares without forming normal equations.

The primal uses pivoted Householder QR and complete orthogonal decomposition.
Its full-rank pullback
is the same equation used by the former CompleteOrthogonalDecomposition
wrapper. Condition checks use singular values of the triangular factor, not
eigenvalues of a Gram matrix (which would square the condition number).
"""

import sys
from functools import lru_cache

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_svd

BACKEND = "tensorflow_native_complete_orthogonal_decomposition"


@lru_cache(maxsize=16)
def make_complete_orthogonal_lstsq(rows, columns):
    # The enclosing endpoint owns JIT. This stable local boundary keeps the
    # custom pullback's captures in one FuncGraph inside nested ALS branches.
    return tf.function(
        complete_orthogonal_lstsq,
        autograph=False,
        input_signature=[
            tf.TensorSpec([rows, columns], tf.float64),
            tf.TensorSpec([rows, 1], tf.float64),
        ],
    )


def condition_number(matrix, *, jit_compile=True):
    matrix = tf.stop_gradient(matrix)
    # Reduce tall systems without sacrificing singular-value conditioning.
    if matrix.shape[0] > matrix.shape[1]:
        matrix = tf.linalg.qr(matrix, full_matrices=False)[1]
    singular = (
        xla_svd(
            matrix, max_iter=100, epsilon=sys.float_info.epsilon, precision_config=""
        ).s
        if jit_compile
        else tf.linalg.svd(matrix, compute_uv=False)
    )
    smallest, largest = tf.reduce_min(singular), tf.reduce_max(singular)
    return tf.stop_gradient(
        tf.where(
            (smallest > 0) & tf.math.is_finite(smallest) & tf.math.is_finite(largest),
            largest / smallest,
            tf.constant(float("inf"), matrix.dtype),
        )
    )


@tf.custom_gradient
def full_rank_lstsq(matrix, rhs):
    q, upper = tf.linalg.qr(matrix, full_matrices=False)
    solution = tf.linalg.triangular_solve(
        upper, tf.matmul(q, rhs, transpose_a=True), lower=False
    )

    def grad(upstream):
        _, upper = tf.linalg.qr(matrix, full_matrices=False)
        intermediate = tf.linalg.triangular_solve(
            upper, upstream, lower=False, adjoint=True
        )
        z = tf.linalg.triangular_solve(upper, intermediate, lower=False)
        xzt = tf.matmul(solution, z, transpose_b=True)
        grad_matrix = -tf.matmul(
            matrix, xzt + tf.linalg.matrix_transpose(xzt)
        ) + tf.matmul(rhs, z, transpose_b=True)
        return grad_matrix, tf.matmul(matrix, z)

    return solution, grad


def _active_row_operation(operation, active_rows, minimum_rows, capacity):
    """Bind static row shapes; execute one branch inside TensorFlow/XLA."""
    def shape_case(rows):
        def execute():
            return operation(rows)
        return execute

    branches = tuple(shape_case(rows) for rows in range(minimum_rows, capacity + 1))
    index = tf.clip_by_value(active_rows, minimum_rows, capacity) - minimum_rows
    return tf.switch_case(index, branches)


def _active_sum(values, active_rows, minimum_rows):
    return _active_row_operation(lambda rows: tf.reduce_sum(values[:rows], axis=0),
        active_rows, minimum_rows, int(values.shape[0]))


def _active_matvec(matrix, vector, active_rows, minimum_rows):
    return _active_row_operation(lambda rows: tf.linalg.matvec(matrix[:rows], vector[:rows], transpose_a=True),
        active_rows, minimum_rows, int(matrix.shape[0]))


def _reflector(vector, pivot, active, active_rows=None, minimum_rows=None):
    # StridedSlice requires data-dependent bounds as compile-time constants
    # on TF/XLA. Gather keeps the pivot and its upstream matrix at runtime.
    first = tf.gather(vector, pivot)
    length = vector.shape[0] if vector.shape[0] is not None else tf.shape(vector)[0]
    tail = tf.where(active & (tf.range(length) != pivot), vector, tf.zeros_like(vector))
    square = (tf.reduce_sum(tail * tail) if active_rows is None
              else _active_sum(tail * tail, active_rows, minimum_rows))
    beta = tf.where(
        first >= 0, -tf.sqrt(first * first + square), tf.sqrt(first * first + square)
    )
    nonzero = square > tf.constant(sys.float_info.min, vector.dtype)
    essential = tail / tf.where(nonzero, first - beta, tf.ones_like(beta))
    tau = tf.where(
        nonzero,
        (beta - first) / tf.where(nonzero, beta, tf.ones_like(beta)),
        tf.zeros_like(beta),
    )
    return tf.one_hot(pivot, length, dtype=vector.dtype) + essential, tau


@tf.custom_gradient
def complete_orthogonal_lstsq(matrix, rhs):
    """Native CPQR/COD with the original two-tensor tracing signature."""
    return _complete_orthogonal_lstsq(matrix, rhs)


@tf.custom_gradient
def complete_orthogonal_lstsq_active_rows(matrix, rhs, active_rows):
    """Shared COD for padded tall systems, with a runtime active-row count."""
    return _complete_orthogonal_lstsq(matrix, rhs, active_rows)


def _complete_orthogonal_lstsq(matrix, rhs, active_rows=None):
    """Native CPQR/COD with Eigen's epsilon*min(rows,columns) rank threshold.

    Anchors: Eigen ColPivHouseholderQR.h::rank/threshold and
    CompleteOrthogonalDecomposition.h::computeInPlace/_solve_impl, bundled
    with TensorFlow. Rank truncation remains observable in the primal; the
    original wrapper's full-rank gradient formula is intentionally preserved.

    Optional active rows support fixed-capacity overdetermined systems with
    at least twice as many active rows as columns. Row reductions use the
    compact operand shape, with shared factorization control. Inactive rows
    and their pullbacks are zero; invalid active counts produce nonfinite
    results rather than a silently accepted clipped system.
    """
    rows, cols = matrix.shape
    if cols is None:
        raise ValueError(
            "complete orthogonal decomposition requires a fixed column count"
        )
    dynamic_rows = rows is None
    if active_rows is not None:
        if dynamic_rows or rows < 2 * cols:
            raise ValueError("active-row COD requires fixed capacity of at least twice the column count")
        active_rows = tf.ensure_shape(active_rows, [])
        if active_rows.dtype != tf.int32:
            raise TypeError("active-row count must have dtype int32")
        active_mask = tf.range(rows) < active_rows
        matrix = tf.where(active_mask[:, None], matrix, tf.zeros_like(matrix))
        rhs = tf.where(active_mask[:, None], rhs, tf.zeros_like(rhs))
    rows = tf.shape(matrix)[0] if dynamic_rows else rows
    diagonal_size = tf.minimum(rows, cols) if dynamic_rows else min(rows, cols)
    row_indices, col_indices = tf.range(rows), tf.range(cols)

    def factor(k, a, b, permutation):
        squared = tf.where(row_indices[:, None] >= k, a * a, tf.zeros_like(a))
        norms = (tf.reduce_sum(squared, axis=0) if active_rows is None
                 else _active_sum(squared, active_rows, 2 * cols))
        pivot = tf.argmax(
            tf.where(
                col_indices >= k, norms, tf.fill([cols], tf.constant(-1.0, a.dtype))
            ),
            output_type=tf.int32,
        )
        swap = tf.tensor_scatter_nd_update(
            col_indices, tf.stack((k, pivot))[:, None], tf.stack((pivot, k))
        )
        a, permutation = tf.gather(a, swap, axis=1), tf.gather(permutation, swap)
        v, tau = _reflector(tf.gather(a, k, axis=1), k, row_indices >= k, active_rows, 2 * cols)
        product_a = (tf.linalg.matvec(a, v, transpose_a=True) if active_rows is None
                     else _active_matvec(a, v, active_rows, 2 * cols))
        a = a - tau * v[:, None] * product_a[None, :]
        # Exact structural zeros avoid rounded residuals entering pivot norms.
        a = tf.where(
            (row_indices[:, None] > k) & (col_indices[None, :] == k),
            tf.zeros_like(a),
            a,
        )
        product_b = (tf.linalg.matvec(b, v, transpose_a=True) if active_rows is None
                     else _active_matvec(b, v, active_rows, 2 * cols))
        b = b - tau * v[:, None] * product_b[None, :]
        return k + 1, a, b, permutation

    _, upper, transformed, permutation = tf.while_loop(
        lambda k, *_: k < diagonal_size,
        factor,
        (tf.constant(0), matrix, rhs, col_indices),
        maximum_iterations=diagonal_size,
        parallel_iterations=1,
    )
    upper = tf.pad(upper[:diagonal_size], [[0, cols - diagonal_size], [0, 0]])
    transformed = tf.pad(
        transformed[:diagonal_size], [[0, cols - diagonal_size], [0, 0]]
    )
    upper = tf.ensure_shape(upper, [cols, cols])
    transformed = tf.ensure_shape(transformed, [cols, rhs.shape[1]])
    pivots = tf.abs(tf.linalg.diag_part(upper))
    threshold = tf.reduce_max(pivots) * (
        tf.constant(sys.float_info.epsilon, matrix.dtype)
        * tf.cast(diagonal_size, matrix.dtype)
    )
    rank = tf.reduce_sum(tf.cast(pivots > threshold, tf.int32))

    # Right Householder transformations zero the trailing columns without
    # dynamic-size slicing. Padded identity rows preserve a static solve shape.
    def complete(i, upper, right):
        k = rank - 1 - i
        v, tau = _reflector(tf.gather(upper, k), k, (col_indices == k) | (col_indices >= rank))
        upper = upper - tau * tf.linalg.matvec(upper, v)[:, None] * v[None, :]
        right = right - tau * tf.linalg.matvec(right, v)[:, None] * v[None, :]
        return i + 1, upper, right

    _, upper, right = tf.while_loop(
        lambda i, *_: i < rank,
        complete,
        (tf.constant(0), upper, tf.eye(cols, dtype=matrix.dtype)),
        maximum_iterations=cols,
        parallel_iterations=1,
    )
    live = col_indices < rank
    safe_upper = tf.where(
        live[:, None] & live[None, :], upper, tf.zeros_like(upper)
    ) + tf.linalg.diag(tf.cast(~live, matrix.dtype))
    padded_rhs = tf.where(live[:, None], transformed, tf.zeros_like(transformed))
    permuted = tf.matmul(
        right, tf.linalg.triangular_solve(safe_upper, padded_rhs, lower=False)
    )
    solution = tf.scatter_nd(permutation[:, None], permuted, [cols, rhs.shape[1]])
    if active_rows is not None:
        count_valid = (active_rows >= 2 * cols) & (active_rows <= rows)
        solution = tf.where(count_valid, solution, tf.constant(float("nan"), solution.dtype))

    def grad(upstream):
        def underdetermined():
            # Full-row-rank minimum-norm derivative. Use QR of A^T to solve
            # AA^T systems without forming a Gram matrix.
            _, upper = tf.linalg.qr(tf.transpose(matrix), full_matrices=False)
            if dynamic_rows:
                upper = tf.pad(upper, [[0, rows - tf.shape(upper)[0]], [0, 0]])

            def solve_row_system(value):
                intermediate = tf.linalg.triangular_solve(
                    upper, value, lower=False, adjoint=True
                )
                return tf.linalg.triangular_solve(upper, intermediate, lower=False)

            y = solve_row_system(rhs)
            z = solve_row_system(tf.matmul(matrix, upstream))
            zy = tf.matmul(z, y, transpose_b=True)
            return (
                tf.matmul(y, upstream, transpose_b=True)
                - tf.matmul(zy + tf.transpose(zy), matrix)
            ), z

        def overdetermined():
            upper = (tf.linalg.qr(matrix, full_matrices=False)[1] if active_rows is None
                else _active_row_operation(lambda count: tf.linalg.qr(matrix[:count], full_matrices=False)[1],
                    active_rows, 2 * cols, rows))
            if dynamic_rows:
                upper = tf.pad(upper, [[0, cols - tf.shape(upper)[0]], [0, 0]])
            intermediate = tf.linalg.triangular_solve(
                upper, upstream, lower=False, adjoint=True
            )
            z = tf.linalg.triangular_solve(upper, intermediate, lower=False)
            xzt = tf.matmul(solution, z, transpose_b=True)
            return -tf.matmul(
                matrix, xzt + tf.linalg.matrix_transpose(xzt)
            ) + tf.matmul(rhs, z, transpose_b=True), tf.matmul(matrix, z)

        if active_rows is not None:
            grad_matrix, grad_rhs = overdetermined()
            return (tf.where(active_mask[:, None], grad_matrix, tf.zeros_like(grad_matrix)),
                tf.where(active_mask[:, None], grad_rhs, tf.zeros_like(grad_rhs)), None)
        if dynamic_rows:
            return tf.cond(rows < cols, underdetermined, overdetermined)
        return underdetermined() if rows < cols else overdetermined()

    return solution, grad
