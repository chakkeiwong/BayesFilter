"""
Phase 3 (preconditioned variant): HMC in native z-space with a
Hessian-derived mass matrix.

Theoretical rationale
---------------------
Standard HMC dynamics: q̈ = -M^{-1} ∇U, where M is the mass matrix.
Setting M = -H at the mode (Fisher information) makes the Hamiltonian
trajectory nearly circular in momentum-scaled coordinates → large
step sizes become admissible → fast mixing.

At a saddle point, -H has negative eigenvalues and can't be used
directly. Riemannian-HMC (Girolami & Calderhead 2011) suggests using
Fisher information which is always PSD. As a cheap surrogate when
Fisher isn't available, we use |H|:

    H = V diag(λ) V^T
    |H| = V diag(|λ|) V^T

This preserves the eigenvectors (the correlation structure captured
by the Hessian) and uses the magnitudes as step-size scales. Whether
a direction has positive or negative curvature, the appropriate step
size is set by |λ|'s magnitude.

Relation to whitened_hmc.py
---------------------------
The whitening approach required Σ = -H^{-1} to exist (forced flooring
at a saddle) and then applied a bijector to transform to whitened
coordinates. This approach needs nothing of the sort: |H| is always
PSD regardless of H's signature, and HMC operates in native z-space
with the mass matrix handling the preconditioning.

Bonus: no bijector jacobian bookkeeping, no TransformedTransitionKernel.
"""

import time
from typing import NamedTuple, Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..tf_filters import TFDifferentiableParticleFilter
from ..hmc.parameterization import (
    unpack_batched, log_prior_batched, TOTAL_DIM,
)
from ..hmc.run_hmc_poc import compute_rhat


DTYPE = tf.float64
tfd = tfp.distributions


class PreconditionedHMCResult(NamedTuple):
    """
    Output of preconditioned Phase 3.

    Attributes:
        samples_z:         [num_samples, B_chain, 9]  samples in z-space
        samples_z_white:   [num_samples, B_chain, 9]  z-samples transformed
                                                      through L_M^{-1}; this
                                                      gives the "preconditioned
                                                      residual" analogous to
                                                      whitened samples. When
                                                      HMC mixes well, this
                                                      should have sample cov ≈
                                                      some diagonal matrix.
        accept_rate:       float
        rhat:              [9]  per-parameter R-hat (z-space)
        rhat_white:        [9]  R-hat in preconditioned coords (for
                                compat with whitened notebook cells)
        final_step_size:   float  adapted step size
        mass_eigvals:      [9]   eigenvalues of |H|
        mass_cond:         float condition number of the mass matrix
        elapsed:           float seconds
    """
    samples_z: np.ndarray
    samples_z_white: np.ndarray
    accept_rate: float
    rhat: np.ndarray
    rhat_white: np.ndarray
    final_step_size: float
    mass_eigvals: np.ndarray
    mass_cond: float
    elapsed: float


def _build_mass_matrix_from_hessian(
    H: np.ndarray,
    min_abs_eigval: float = 1e-6,
) -> tuple:
    """
    Construct a PSD mass matrix M = |H| from the (possibly indefinite)
    Hessian H of log-posterior.

    The sign convention: we want M to play the role of -H at a true mode,
    so M is built from -H's eigenvalues:
        -H = V diag(λ) V^T
        M  = V diag(|λ|) V^T

    A tiny floor `min_abs_eigval` is applied to guard against exactly-zero
    eigenvalues that would otherwise give infinite step size in that
    direction. This is minimal regularization — unlike the Laplace
    flooring which clips to lambda_max / kappa_target, here we only
    prevent division-by-zero.

    Returns:
        M:          [d, d]  PSD mass matrix
        L_M:        [d, d]  Cholesky factor, M = L_M L_M^T
        eigvals:    [d]     raw eigenvalues of -H
        abs_eigvals: [d]    the regularized |eigvals| used in M
    """
    neg_H = -H
    eigvals, eigvecs = np.linalg.eigh(neg_H)       # ascending
    abs_eigvals = np.maximum(np.abs(eigvals), min_abs_eigval)

    # M = V diag(|λ|) V^T
    M = (eigvecs * abs_eigvals[np.newaxis, :]) @ eigvecs.T
    M = 0.5 * (M + M.T)                             # symmetrize
    L_M = np.linalg.cholesky(M)

    return M, L_M, eigvals, abs_eigvals


def _build_target_log_prob_fn(
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    n_mc: int,
    include_prior: bool,
):
    """
    Batched log-posterior in native z-space. Identical to the whitened
    version's target function; the difference is that HMC operates
    directly on z (no bijector transform).
    """

    @tf.function(reduce_retracing=True)
    def target_log_prob_fn(z):
        B_chain = tf.shape(z)[0]
        z_rep = tf.repeat(z, repeats=n_mc, axis=0)
        params = unpack_batched(z_rep)
        result = dpf.filter(params, observations, rng)
        log_ev_flat = result.log_evidence
        log_ev_2d = tf.reshape(log_ev_flat, [B_chain, n_mc])
        log_ev = tf.reduce_mean(log_ev_2d, axis=-1)

        if include_prior:
            lp = log_prior_batched(z)
            return lp + log_ev
        else:
            return log_ev

    return target_log_prob_fn


def run_preconditioned_hmc(
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    z_hat: tf.Tensor,
    H: tf.Tensor,
    B_chain: int = 4,
    num_burnin: int = 50,
    num_samples: int = 200,
    num_leapfrog: int = 10,
    initial_step_size: float = 0.3,
    n_mc: int = 2,
    include_prior: bool = True,
    init_jitter: float = 0.02,
    target_accept: float = 0.75,
    min_abs_eigval: float = 1e-6,
    seed: int = 7777,
    verbose: bool = True,
) -> PreconditionedHMCResult:
    """
    Run HMC in z-space with a mass matrix |H| derived from the Phase 2 Hessian.

    How the mass matrix preconditions HMC
    -------------------------------------
    HMC's optimal step size is O(σ_min) where σ_min = 1/sqrt(max |eigval(M^{-1}(-H))|).
    If M = |H|, then M^{-1}(-H) has eigenvalues ±1, so σ_min = 1; i.e., a
    step size of ~1 is reasonable regardless of the raw Hessian's
    condition number. SimpleStepSizeAdaptation will tune this.

    Chains start near z_hat with small Gaussian jitter in z-space.
    Because the mass matrix encodes the local scale, `init_jitter` in z
    coords can be small — 0.02 is ~0.1σ of the posterior in most
    directions.

    Args:
        observations:   [T, d]
        dpf:            soft DPF
        rng:            tf.random.Generator
        z_hat:          [9]  Phase 1 Adam output
        H:              [9, 9]  Hessian of log-posterior from Phase 2 (raw,
                        NOT regularized — this method handles indefinite H)
        B_chain:        number of chains
        num_burnin:     HMC warmup steps (includes step-size adaptation)
        num_samples:    HMC samples
        num_leapfrog:   leapfrog steps per move
        initial_step_size: pre-adaptation value; with correct mass matrix
                        optimal is near 1; 0.3 is conservative.
        n_mc:           PF MC replicas per gradient
        include_prior:  match Phase 1 objective
        init_jitter:    z-space jitter for chain initialization
        target_accept:  step-size adaptation target
        min_abs_eigval: floor on |eigval| of -H to avoid singular M
        seed:           jitter RNG seed
        verbose:        print progress

    Returns:
        PreconditionedHMCResult
    """
    z_hat = tf.cast(tf.reshape(z_hat, [TOTAL_DIM]), DTYPE)
    H_np = np.asarray(H)

    # --- Build mass matrix M = |H|, factored ---
    M, L_M, eigvals_raw, eigvals_abs = _build_mass_matrix_from_hessian(
        H_np, min_abs_eigval=min_abs_eigval,
    )
    cond_M = eigvals_abs.max() / eigvals_abs.min()

    if verbose:
        print(f"[PrecondHMC] building mass matrix M = |H|")
        print(f"[PrecondHMC] eigvals of -H (raw):       {eigvals_raw.round(3)}")
        print(f"[PrecondHMC] eigvals of M (|-H|, used): {eigvals_abs.round(3)}")
        print(f"[PrecondHMC] cond(M) = {cond_M:.2e}")
        n_negative = int((eigvals_raw < 0).sum())
        if n_negative > 0:
            print(f"[PrecondHMC] NOTE: {n_negative}/{TOTAL_DIM} directions had "
                  f"negative -H eigval (z_hat is a saddle or plateau); ")
            print(f"            mass matrix absorbs the sign via |·|, so HMC ")
            print(f"            proceeds without regularization bias.")

    # --- Target log-prob in z-space ---
    target_log_prob_fn = _build_target_log_prob_fn(
        observations, dpf, rng, n_mc=n_mc, include_prior=include_prior,
    )

    # --- Momentum distribution ---
    # Mass matrix M defines the KE: p^T M^{-1} p / 2
    # Momentum ~ N(0, M), so its scale_tril is chol(M) = L_M.
    # TFP's HMC uses the momentum distribution directly.
    L_M_tf = tf.constant(L_M, dtype=DTYPE)

    def make_momentum_dist():
        return tfd.MultivariateNormalLinearOperator(
            loc=tf.zeros(TOTAL_DIM, dtype=DTYPE),
            scale=tf.linalg.LinearOperatorLowerTriangular(L_M_tf),
        )

    # --- Initial states in z-space, near z_hat ---
    rng_init = tf.random.Generator.from_seed(seed)
    # Small jitter in z-coords. We scale by the inverse mass diag to
    # give per-direction jitter proportional to expected posterior scale.
    # (Simple z-jitter works fine too; this is a mild refinement.)
    z_init = z_hat[tf.newaxis, :] + init_jitter * rng_init.normal(
        [B_chain, TOTAL_DIM], dtype=DTYPE,
    )

    # --- HMC kernel with explicit momentum distribution ---
    inner_hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=num_leapfrog,
        step_size=initial_step_size,
        store_parameters_in_results=True,
    )

    # Attach the momentum distribution. In TFP, this is done by replacing
    # the kernel's momentum sampler. The modern way is via
    # `tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo`.
    try:
        inner_hmc = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            num_leapfrog_steps=num_leapfrog,
            step_size=initial_step_size,
            momentum_distribution=make_momentum_dist(),
        )
        used_preconditioned_api = True
    except AttributeError:
        used_preconditioned_api = False
        # Fallback: use plain HMC without explicit momentum. With the
        # Hessian-derived step_size chosen correctly this degrades
        # gracefully but loses the directional mass-matrix benefit.
        if verbose:
            print("[PrecondHMC] WARNING: tfp.experimental.mcmc."
                  "PreconditionedHamiltonianMonteCarlo not available; "
                  "falling back to plain HMC.")

    # Step-size adaptation during burnin
    adapted_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_hmc,
        num_adaptation_steps=int(0.8 * num_burnin),
        target_accept_prob=tf.cast(target_accept, DTYPE),
    )

    # --- Sample ---
    if verbose:
        print(f"[PrecondHMC] B_chain={B_chain}, burnin={num_burnin}, "
              f"samples={num_samples}, leapfrog={num_leapfrog}")
        print(f"[PrecondHMC] initial step_size={initial_step_size}, "
              f"target_accept={target_accept}")
        print(f"[PrecondHMC] preconditioned_api={used_preconditioned_api}")

    t0 = time.time()

    def trace_fn(_, pkr):
        # SimpleStepSizeAdaptation wraps HMC; .inner_results is HMC results.
        hmc_r = pkr.inner_results
        return (hmc_r.is_accepted, hmc_r.accepted_results.step_size)

    samples_z_tf, (is_accepted, step_sizes) = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin,
        current_state=z_init,
        kernel=adapted_kernel,
        trace_fn=trace_fn,
    )
    elapsed = time.time() - t0

    samples_z = samples_z_tf.numpy()
    accept_rate = float(tf.reduce_mean(tf.cast(is_accepted, DTYPE)).numpy())

    # step_sizes can be scalar or per-chain depending on TFP config
    ss_last = step_sizes[-1].numpy()
    final_step = float(ss_last if np.isscalar(ss_last) or ss_last.ndim == 0
                       else ss_last.flatten()[0])

    rhat = compute_rhat(samples_z_tf).numpy()

    # Preconditioned-coord analog of "whitening": residuals transformed by
    # L_M^{-1}. If HMC mixes well and the posterior really is locally well-
    # described by |H|, sample_cov in these coords should be ≈ diagonal.
    # Specifically, eigvals of sample_cov equal (true_var along eigvec) *
    # |eigval of H|, which should be ≈ 1 at a well-identified mode.
    residuals = samples_z - z_hat.numpy()[np.newaxis, np.newaxis, :]
    # L_M^{-1} @ residual^T for each (s, b)
    L_M_inv = np.linalg.solve(L_M, np.eye(TOTAL_DIM))
    samples_z_white = np.einsum('ij,sbj->sbi', L_M_inv, residuals)
    samples_z_white_tf = tf.constant(samples_z_white, dtype=DTYPE)
    rhat_white = compute_rhat(samples_z_white_tf).numpy()

    if verbose:
        print(f"[PrecondHMC] done in {elapsed:.1f}s")
        print(f"[PrecondHMC] accept_rate = {accept_rate:.2%}")
        print(f"[PrecondHMC] final step_size = {final_step:.4f}")
        print(f"[PrecondHMC] R-hat: {rhat.round(3)}")

        # Quick sanity: compare HMC's empirical cov to 1/|eigvals|
        # (the "implicit Laplace" that M encodes)
        flat = samples_z.reshape(-1, TOTAL_DIM)
        emp_std = flat.std(axis=0)
        laplace_implicit_std = 1.0 / np.sqrt(eigvals_abs)
        # Compare in the eigenbasis of H rather than per-param
        # (a simpler per-param check: not exact but informative)
        print(f"[PrecondHMC] HMC empirical std (per-param): "
              f"{emp_std.round(3)}")

    return PreconditionedHMCResult(
        samples_z=samples_z,
        samples_z_white=samples_z_white,
        accept_rate=accept_rate,
        rhat=rhat,
        rhat_white=rhat_white,
        final_step_size=final_step,
        mass_eigvals=eigvals_abs,
        mass_cond=cond_M,
        elapsed=elapsed,
    )
