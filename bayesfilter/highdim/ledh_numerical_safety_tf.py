"""Numerical safety utilities for LEDH computations.

This module provides graceful failure handling for numerically unstable
operations that can occur during HMC parameter space exploration.

The key insight: instead of crashing on ill-conditioned matrices, we detect
failures and return sentinel values (-inf log probability, zero gradient) that
allow Metropolis-Hastings rejection to work as designed.
"""

import tensorflow as tf


def safe_cholesky(
    matrix: tf.Tensor,
    name: str = "cholesky"
) -> tuple[tf.Tensor, tf.Tensor]:
    """Cholesky decomposition with per-matrix nonfinite-factor rejection.

    TensorFlow's tf.linalg.cholesky returns NaN when the input matrix is not
    positive definite (e.g., due to numerical errors, ill-conditioning, or
    actual indefiniteness). This wrapper detects the NaN result and returns
    a validity flag instead of propagating NaN through the computation.

    This enables graceful failure handling: when LEDH encounters a pathological
    parameter configuration during HMC exploration, we can return -inf log
    probability and let Metropolis-Hastings reject the proposal naturally,
    rather than crashing the entire chain.

    Parameters
    ----------
    matrix : tf.Tensor
        Symmetric positive-definite matrix to factorize, shape [..., n, n].
    name : str, optional
        Name for the operation (used in error messages and profiling).

    Returns
    -------
    valid : tf.Tensor
        Boolean batch mask with all singleton axes squeezed, preserving the
        existing API. Unbatched and singleton-batch inputs return a scalar.
        True means the computed factor is finite, not that it is well conditioned.
    chol : tf.Tensor
        Cholesky factor L such that matrix = L @ L^T when valid=True.
        Returns zeros when valid=False for clean propagation through graph.

    Notes
    -----
    **Why factor inspection instead of an eigenvalue pre-check:**

    - **Performance**: O(n²) finite scan vs O(n³) eigenvalue decomposition
    - **XLA-compatible**: Pure TensorFlow ops, no Python control flow
    - **Empirically validated**: Phase 4a logs show Cholesky returning NaN
      with warning messages, not crashing

    **Failure modes detected:**

    - Negative eigenvalues (indefinite matrices)
    - Zero eigenvalues (singular matrices)
    - Numerical errors accumulating to non-positive-definite result
    - Nonfinite factors, including infinity

    A finite factor can still be ill-conditioned. This helper does not compute
    a condition number, check symmetry or establish an error bound; callers
    requiring those properties must check them separately. Numerical operations
    are TensorFlow-native and inherit compilation from their enclosing owner.

    **Example usage:**

    ```python
    # In Contract E reset policy:
    gap = target_cov - plus_cov + ridge * I
    valid_gap, gap_chol = safe_cholesky(gap, "gap")

    if not valid_gap:
        # Return sentinel values for -inf log probability
        return zeros, zeros, ..., False
    ```

    References
    ----------
    Phase 4a diagnostic failure (2026-09-16): Cholesky decomposition failed
    during HMC exploration with message "Cholesky decomposition was not
    successful for batch 0. The input might not be valid. Filling lower-
    triangular output with NaNs."
    """
    with tf.name_scope(name):
        # Attempt Cholesky decomposition
        # On failure, TensorFlow returns NaN in the output tensor
        chol_attempt = tf.linalg.cholesky(matrix)

        # Check each batch element; finite factors need no numerical alteration.
        # For shape [..., n, n], reduce over last two dimensions only
        # This gives per-batch validity flags for batched inputs
        is_valid = tf.reduce_all(
            tf.math.is_finite(chol_attempt),
            axis=[-2, -1]
        )

        # Expand is_valid for broadcasting if needed
        # For batched inputs, we need [..., 1, 1] shape for tf.where
        broadcast_valid = is_valid[..., tf.newaxis, tf.newaxis]

        # Return zeros when invalid (for clean propagation)
        # This prevents NaN from contaminating downstream computations
        chol = tf.where(
            broadcast_valid,
            chol_attempt,
            tf.zeros_like(chol_attempt)
        )

        # Squeeze back to scalar for non-batched case
        is_valid = tf.squeeze(is_valid)

        return is_valid, chol
