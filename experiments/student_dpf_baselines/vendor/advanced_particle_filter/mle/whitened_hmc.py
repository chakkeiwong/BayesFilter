"""
Phase 3 (optional): Local HMC in Laplace-whitened coordinates.

Theoretical rationale
---------------------
The Laplace approximation gives posterior ≈ N(z_hat, Σ_Laplace). Define
the whitening transform

    z = z_hat + L @ z_white    where   Σ_Laplace = L L^T

Then under the Laplace approximation, z_white ~ N(0, I). If the true
posterior is also approximately Gaussian near z_hat, the posterior of
z_white is approximately isotropic unit-variance → a scalar HMC step
size works well across all directions, no mass-matrix tuning needed.

Why this helps with DPF gradient bias
-------------------------------------
The soft resampler's ~25-30% gradient attenuation accumulates over long
HMC trajectories far from the mode. Here we:
  (a) start all chains at z_hat (the mode) — no burn-in distance to
      accumulate bias over
  (b) stay in a small ball around the mode — leapfrog trajectories of a
      few dozen steps in whitened coordinates correspond to steps of
      size O(sigma) in original coordinates
  (c) use short chains (50 burnin + 200 samples) — total distance
      traveled is limited

Together these keep DPF attenuation below the level that biased the
original raw HMC's Phi to 0.55 vs truth 0.85.

What Phase 3 gives you that Phase 2 didn't
------------------------------------------
Non-Gaussian posterior features: skew, heavy tails, multi-modality
within the local basin. If samples in z_white space look ~ N(0, I)
(sample covariance ≈ I, marginals look Gaussian), then Laplace was
sufficient and Phase 3 adds little. If they don't, Phase 3 corrects
the Gaussian approximation.
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


class WhitenedHMCResult(NamedTuple):
    """
    Output of Phase 3.

    Attributes:
        samples_z:         [num_samples, B_chain, 9]  samples in original coords
        samples_z_white:   [num_samples, B_chain, 9]  samples in whitened coords
        accept_rate:       float
        rhat:              [9]  potential scale reduction per parameter
                                (in original coords)
        rhat_white:        [9]  same in whitened coords (often more uniform)
        final_step_size:   float  adapted step size
        elapsed:           float  seconds
    """
    samples_z: np.ndarray
    samples_z_white: np.ndarray
    accept_rate: float
    rhat: np.ndarray
    rhat_white: np.ndarray
    final_step_size: float
    elapsed: float


def _build_target_in_original_coords(
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    n_mc: int,
    include_prior: bool,
):
    """
    Target log-posterior in the original z parameterization. The
    TransformedTransitionKernel wraps this with the L-whitening bijector
    to obtain the target in whitened coords (handling the log-abs-det
    Jacobian automatically).
    """

    @tf.function(reduce_retracing=True)
    def target_log_prob_fn(z):
        # z: [B_chain, 9]
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


def run_whitened_hmc(
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    z_hat: tf.Tensor,
    L_chol: tf.Tensor,
    B_chain: int = 4,
    num_burnin: int = 50,
    num_samples: int = 200,
    num_leapfrog: int = 10,
    initial_step_size: float = 0.5,
    n_mc: int = 2,
    include_prior: bool = True,
    white_jitter: float = 0.1,
    target_accept: float = 0.75,
    seed: int = 7777,
    verbose: bool = True,
) -> WhitenedHMCResult:
    """
    Run HMC in Laplace-whitened coordinates, starting all chains at the mode.

    Implementation note: we use TFP's TransformedTransitionKernel to let
    HMC operate in whitened space transparently. The bijector is
        z = z_hat + L @ z_white           (forward)
        z_white = L^{-1} @ (z - z_hat)    (inverse)
    which is an affine (Shift ∘ ScaleMatvecTriL) bijector. TFP handles
    the constant log-abs-det(L) Jacobian correctly.

    Args:
        observations:     [T, d]
        dpf:              soft DPF, N=1000 recommended
        rng:              tf.random.Generator for PF sampling
        z_hat:            [9]   mode from Phase 1
        L_chol:           [9, 9] Cholesky from Phase 2 (Σ = L L^T)
        B_chain:          number of chains
        num_burnin:       HMC warmup steps (step-size adaptation)
        num_samples:      HMC sampling steps
        num_leapfrog:     leapfrog steps per HMC move
        initial_step_size: in whitened space. 0.5 is a sensible default
                          since whitened posterior ≈ N(0, I) and
                          optimal HMC step is O(d^{-1/4}) ≈ 0.5 for d=9.
        n_mc:             PF MC replicas per gradient
        include_prior:    match Phase 1 objective
        white_jitter:     initial chain scatter in whitened coords.
                          0.1 = chains start at 0.1σ around the mode.
        target_accept:    step-size adaptation target
        seed:             jitter RNG seed
        verbose:          print progress

    Returns:
        WhitenedHMCResult
    """
    z_hat = tf.cast(tf.reshape(z_hat, [TOTAL_DIM]), DTYPE)
    L_chol = tf.cast(L_chol, DTYPE)

    # --- Build bijector: z_white -> z = z_hat + L @ z_white ---
    # TFP composition order: Shift(z_hat)(ScaleMatvecTriL(L)(z_white))
    scale = tfp.bijectors.ScaleMatvecTriL(
        scale_tril=L_chol, validate_args=False,
    )
    shift = tfp.bijectors.Shift(z_hat)
    bijector = tfp.bijectors.Chain([shift, scale])   # applied right-to-left

    # --- Target in original coords ---
    target_log_prob_fn = _build_target_in_original_coords(
        observations, dpf, rng, n_mc=n_mc, include_prior=include_prior,
    )

    # --- Initial states in WHITENED coords, at 0 + jitter ---
    rng_init = tf.random.Generator.from_seed(seed)
    z_white_init = white_jitter * rng_init.normal(
        [B_chain, TOTAL_DIM], dtype=DTYPE,
    )

    # --- Build HMC kernel wrapped in TransformedTransitionKernel ---
    inner_hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=num_leapfrog,
        step_size=initial_step_size,
    )

    # Adapt step size during burnin; adaptation target ~0.75 for HMC
    adapted_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_hmc,
        num_adaptation_steps=int(0.8 * num_burnin),
        target_accept_prob=tf.cast(target_accept, DTYPE),
    )

    # Wrap with bijector: HMC walks in z_white space
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=adapted_kernel,
        bijector=bijector,
    )

    # --- Sample ---
    if verbose:
        print(f"[WhitenedHMC] B_chain={B_chain}, burnin={num_burnin}, "
              f"samples={num_samples}, leapfrog={num_leapfrog}")
        print(f"[WhitenedHMC] initial step_size={initial_step_size}, "
              f"target_accept={target_accept}")

    t0 = time.time()

    # trace_fn reports accept + inner_kernel's step size (adapted in burnin).
    # With TransformedTransitionKernel wrapping SimpleStepSizeAdaptation
    # wrapping HMC, the nested pkr structure is:
    #   pkr.inner_results = SimpleStepSizeAdaptationResults
    #     .inner_results = HamiltonianMonteCarloResults
    #       .is_accepted, .accepted_results.step_size
    def trace_fn(_, pkr):
        ssa = pkr.inner_results
        hmc_r = ssa.inner_results
        return (
            hmc_r.is_accepted,
            hmc_r.accepted_results.step_size,
        )

    samples_white, (is_accepted, step_sizes) = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin,
        current_state=z_white_init,
        kernel=transformed_kernel,
        trace_fn=trace_fn,
    )
    elapsed = time.time() - t0

    # samples_white is [num_samples, B_chain, 9] in WHITENED space
    # because TransformedTransitionKernel exposes the inner (whitened) state.
    samples_z_white = samples_white.numpy()

    # --- Bring samples back to original coords ---
    # z = z_hat + L @ z_white, broadcast over (num_samples, B_chain)
    samples_z_tf = bijector.forward(samples_white)   # [S, B, 9]
    samples_z_np = samples_z_tf.numpy()

    accept_rate = float(tf.reduce_mean(tf.cast(is_accepted, DTYPE)).numpy())
    final_step = float(step_sizes[-1].numpy() if step_sizes.shape.rank == 1
                       else step_sizes[-1][0].numpy())

    # R-hat in both coord systems
    rhat_orig = compute_rhat(samples_z_tf).numpy()   # [9]
    rhat_white = compute_rhat(samples_white).numpy() # [9]

    if verbose:
        print(f"[WhitenedHMC] done in {elapsed:.1f}s")
        print(f"[WhitenedHMC] accept_rate = {accept_rate:.2%}")
        print(f"[WhitenedHMC] final step_size = {final_step:.4f}")
        print(f"[WhitenedHMC] R-hat (orig coords):  {rhat_orig.round(3)}")
        print(f"[WhitenedHMC] R-hat (white coords): {rhat_white.round(3)}")

        # Quick sample-cov sanity check in whitened coords
        flat_white = samples_z_white.reshape(-1, TOTAL_DIM)
        sample_cov_white = np.cov(flat_white, rowvar=False)
        diag_dev = np.abs(np.diag(sample_cov_white) - 1.0).max()
        offdiag_max = np.abs(
            sample_cov_white - np.diag(np.diag(sample_cov_white))
        ).max()
        print(f"[WhitenedHMC] sample cov diag deviation from 1: "
              f"{diag_dev:.3f}")
        print(f"[WhitenedHMC] sample cov max off-diag: {offdiag_max:.3f}")
        if diag_dev < 0.3 and offdiag_max < 0.3:
            print("[WhitenedHMC] -> whitened posterior ≈ N(0,I); "
                  "Laplace (Phase 2) was a good approximation.")
        else:
            print("[WhitenedHMC] -> whitened posterior deviates from N(0,I); "
                  "Phase 3 samples capture non-Gaussian features "
                  "that Laplace missed.")

    return WhitenedHMCResult(
        samples_z=samples_z_np,
        samples_z_white=samples_z_white,
        accept_rate=accept_rate,
        rhat=rhat_orig,
        rhat_white=rhat_white,
        final_step_size=final_step,
        elapsed=elapsed,
    )
