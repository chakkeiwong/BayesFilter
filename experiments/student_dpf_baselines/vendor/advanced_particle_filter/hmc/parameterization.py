"""
HMC parameterization for the SVSSM PoC.

Maps HMC's unconstrained parameter vector z to constrained model
parameters (mu, Phi, Sigma_eta_chol). Batched over HMC chains.

Layout (2D state, d = 2):
    z[:, 0:2]  -> mu               [B, 2]
    z[:, 2:6]  -> Phi (row-major)  [B, 2, 2]     free, stability via barrier
    z[:, 6:9]  -> Sigma_eta chol   Cholesky: log(L_11), log(L_22), L_21

Total: 9 unconstrained dims per chain.

All operations are fully tensor-only and JIT-safe under tf.function.
"""

import tensorflow as tf
from ..tf_models.svssm import SVSSMParams


STATE_DIM = 2

# Layout constants
MU_START, MU_END = 0, 2          # [0:2]  mu
PHI_START, PHI_END = 2, 6        # [2:6]  Phi flat
SIGMA_START, SIGMA_END = 6, 9    # [6:9]  Sigma_eta Cholesky

TOTAL_DIM = 9


@tf.function(reduce_retracing=True)
def unpack_batched(z: tf.Tensor) -> SVSSMParams:
    """
    Unpack a batched unconstrained parameter vector.

    Identification constraints applied:
      - mu ordering: mu_1 = z[0], mu_2 = z[0] + exp(z[1])
        ensures mu_1 < mu_2, breaking the permutation symmetry.
      - Sigma_eta Cholesky: L_11 = exp(z[6]), L_22 = exp(z[7]) (positive),
        L_21 = z[8] (free).

    Args:
        z: [B, 9] unconstrained parameters

    Returns:
        SVSSMParams with leading batch dim B.
    """
    dtype = z.dtype
    B = tf.shape(z)[0]

    # Mu with ordering constraint: mu_1 < mu_2
    # z[0] -> mu_1 directly
    # z[1] -> mu_2 = mu_1 + exp(z[1])  (always > mu_1)
    mu_1 = z[:, 0]                                         # [B]
    mu_2 = z[:, 0] + tf.exp(z[:, 1])                      # [B]
    mu = tf.stack([mu_1, mu_2], axis=-1)                   # [B, 2]

    Phi = tf.reshape(
        z[:, PHI_START:PHI_END], [B, STATE_DIM, STATE_DIM]
    )                                                       # [B, 2, 2]

    # Sigma_eta Cholesky:
    #   L_11 = exp(z[6]),   L_22 = exp(z[7]),   L_21 = z[8]
    #   L = [[L_11,  0 ],
    #        [L_21, L_22]]
    z_sigma = z[:, SIGMA_START:SIGMA_END]                  # [B, 3]
    L_11 = tf.exp(z_sigma[:, 0])
    L_22 = tf.exp(z_sigma[:, 1])
    L_21 = z_sigma[:, 2]
    zero = tf.zeros_like(L_11)

    row0 = tf.stack([L_11, zero], axis=-1)                 # [B, 2]
    row1 = tf.stack([L_21, L_22], axis=-1)                 # [B, 2]
    Sigma_eta_chol = tf.stack([row0, row1], axis=1)        # [B, 2, 2]

    return SVSSMParams(mu=mu, Phi=Phi, Sigma_eta_chol=Sigma_eta_chol)


@tf.function(reduce_retracing=True)
def _spectral_radius_2x2(Phi: tf.Tensor) -> tf.Tensor:
    """
    Closed-form spectral radius of a batch of 2x2 matrices.

    Eigenvalues are (tr +/- sqrt(tr^2 - 4 det)) / 2.
      discriminant >= 0 (real eigs):     rho = (|tr| + sqrt(discrim)) / 2
      discriminant <  0 (complex eigs):  rho = sqrt(det)

    Args:
        Phi: [B, 2, 2]
    Returns:
        rho: [B]
    """
    dtype = Phi.dtype
    tr = Phi[:, 0, 0] + Phi[:, 1, 1]
    det = (Phi[:, 0, 0] * Phi[:, 1, 1]
           - Phi[:, 0, 1] * Phi[:, 1, 0])
    discrim = tr * tr - 4.0 * det

    # Safe branches for differentiability at the boundary
    safe_real = tf.maximum(discrim, tf.cast(0.0, dtype))
    rho_real = (tf.abs(tr) + tf.sqrt(safe_real + tf.cast(1e-12, dtype))) / 2.0
    safe_det = tf.maximum(det, tf.cast(1e-12, dtype))
    rho_complex = tf.sqrt(safe_det)

    return tf.where(discrim >= 0.0, rho_real, rho_complex)


@tf.function(reduce_retracing=True)
def log_prior_batched(
    z: tf.Tensor,
    mu_scale: float = 2.0,
    phi_diag_center: float = 0.85,
    phi_diag_scale: float = 0.3,
    phi_offdiag_scale: float = 0.3,
    sigma_log_diag_scale: float = 1.5,
    sigma_offdiag_scale: float = 0.1,
    barrier_weight: float = 100.0,
    barrier_threshold: float = 0.98,
) -> tf.Tensor:
    """
    Batched log-prior on z with identification-aware structure:

      - mu: N(0, mu_scale^2) on z[0] and z[1] (unconstrained space;
        mu_1 = z[0], mu_2 = z[0] + exp(z[1]) via ordering constraint)
      - Phi diagonal: N(phi_diag_center, phi_diag_scale^2) — centered at
        high persistence, reflecting prior belief that assets are persistent
      - Phi off-diagonal: N(0, phi_offdiag_scale^2) — tight, encouraging
        sparsity in cross-asset effects (breaks Phi-Sigma ridge)
      - Sigma_eta log-diag: N(0, sigma_log_diag_scale^2) — weakly
        informative on innovation scale
      - Sigma_eta L_21: N(0, sigma_offdiag_scale^2) — TIGHT (default 0.1),
        encouraging uncorrelated innovations unless data demands otherwise
      - Spectral-radius barrier on Phi
      - Jacobian of bijection (exp on Cholesky diag + mu ordering exp)

    Args:
        z: [B, 9]

    Returns:
        log_prior: [B]
    """
    dtype = z.dtype
    mu_s = tf.cast(mu_scale, dtype)
    phi_d_c = tf.cast(phi_diag_center, dtype)
    phi_d_s = tf.cast(phi_diag_scale, dtype)
    phi_o_s = tf.cast(phi_offdiag_scale, dtype)
    sig_d = tf.cast(sigma_log_diag_scale, dtype)
    sig_o = tf.cast(sigma_offdiag_scale, dtype)
    b_w = tf.cast(barrier_weight, dtype)
    b_t = tf.cast(barrier_threshold, dtype)

    z_mu    = z[:, MU_START:MU_END]        # [B, 2]
    z_phi   = z[:, PHI_START:PHI_END]      # [B, 4]  row-major: [00, 01, 10, 11]
    z_sigma = z[:, SIGMA_START:SIGMA_END]  # [B, 3]

    # --- Mu prior (on unconstrained z[0], z[1]) ---
    # z[0] is mu_1 directly; z[1] maps to log(mu_2 - mu_1)
    lp_mu = -0.5 * tf.reduce_sum(tf.square(z_mu / mu_s), axis=-1)     # [B]

    # --- Phi prior: separate diagonal and off-diagonal ---
    # z_phi layout: [Phi_00, Phi_01, Phi_10, Phi_11]
    phi_diag = tf.stack([z_phi[:, 0], z_phi[:, 3]], axis=-1)           # [B, 2]
    phi_offdiag = tf.stack([z_phi[:, 1], z_phi[:, 2]], axis=-1)        # [B, 2]
    lp_phi_diag = -0.5 * tf.reduce_sum(
        tf.square((phi_diag - phi_d_c) / phi_d_s), axis=-1
    )                                                                   # [B]
    lp_phi_offdiag = -0.5 * tf.reduce_sum(
        tf.square(phi_offdiag / phi_o_s), axis=-1
    )                                                                   # [B]

    # --- Sigma_eta prior ---
    lp_sigma_diag = -0.5 * tf.reduce_sum(
        tf.square(z_sigma[:, :2] / sig_d), axis=-1
    )                                                                   # [B]
    lp_sigma_offdiag = -0.5 * tf.square(z_sigma[:, 2] / sig_o)        # [B]

    # --- Stability barrier ---
    params = unpack_batched(z)
    rho = _spectral_radius_2x2(params.Phi)                             # [B]
    excess = tf.maximum(rho - b_t, tf.cast(0.0, dtype))
    lp_barrier = -b_w * tf.square(excess)                              # [B]

    # --- Jacobian of bijection ---
    # exp() on z[6], z[7] (Cholesky diag): contributes z[6] + z[7]
    # exp() on z[1] (mu ordering): contributes z[1]
    log_det_J = z_sigma[:, 0] + z_sigma[:, 1] + z[:, 1]               # [B]

    return (lp_mu + lp_phi_diag + lp_phi_offdiag
            + lp_sigma_diag + lp_sigma_offdiag
            + lp_barrier + log_det_J)
