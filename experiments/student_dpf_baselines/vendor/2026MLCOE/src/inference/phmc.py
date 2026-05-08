"""
Particle Hamiltonian Monte Carlo (PHMC) Inference Module
========================================================
Implements the PHMC engine for gradient-based parameter estimation in
non-linear state-space models. This module utilizes the modular
DifferentiablePFPF (defined in a separate dpfpf.py) as its likelihood engine.

Mathematical Context:
---------------------
PHMC bridges the gap between Sequential Monte Carlo (SMC) and MCMC.
By using a differentiable filter (dPFPF) to estimate the marginal likelihood,
we can use automatic differentiation to obtain the gradient of the likelihood
with respect to the model parameters (theta). This allows the use of HMC,
which is significantly more efficient than random-walk Metropolis-Hastings
in high-dimensional or complex parameter spaces.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

# Modular math and filter imports
from src.filters.classical import DTYPE, safe_inv
from src.filters.dpfpf import DifferentiablePFPF
from src.filters.flows.EDH import EDHSolver
from src.filters.resampling.optimal_transport import SinkhornResampler

tfd = tfp.distributions

# =============================================================================
# PHMC INFERENCE ENGINE
# =============================================================================


def hmc_pfpf(model, observations, solver=None, resampler=None, num_samples=50,
             beta=0.1, stepsize=0.1, num_leapfrog_steps=2, callback=None):
    """
    Main PHMC sampling function.

    Args:
        model: The state-space model instance.
        observations: The sequence of data used for likelihood estimation.
        solver: The flow solver for the internal dPFPF.
        resampler: The differentiable resampler for the internal dPFPF.
        num_samples (int): Number of MCMC iterations to run.
        beta (float): Inverse temperature for tempering the posterior.
        stepsize (float): Initial HMC step size for the leapfrog integrator.
        num_leapfrog_steps (int): Number of steps in each HMC trajectory.
        callback (callable): Function called after each iteration for logging/monitoring.

    Returns:
        samples (np.array): The trace of accepted parameter values.
        acc_rate (float): The final acceptance rate of the chain.
    """
    N_particles = 100

    # Initialize the filter engine which acts as the likelihood evaluator.
    # Note that this filter is differentiable, allowing HMC to compute gradients.
    filter_engine = DifferentiablePFPF(model, solver=solver, resampler=resampler, N=N_particles)

    @tf.function
    def target_log_prob(theta):
        """
        Computes the log-target probability: log p(theta) + log p(y_{1:T} | theta).
        """
        theta = tf.reshape(theta, [1])

        # 1. Prior Evaluation: log p(theta)
        # Using a normal prior centered at 10.0 (based on the sigma_v_sq interest).
        prior = tf.reduce_sum(tfd.Normal(tf.cast(10.0, DTYPE), tf.cast(5.0, DTYPE)).log_prob(theta))

        # 2. Model Update: Push the proposed theta into the SSM parameters.
        model.update_parameters(theta)

        # 3. Filter Initialization
        X = tfd.Normal(tf.cast(0.0, DTYPE), tf.sqrt(tf.cast(5.0, DTYPE))).sample((N_particles, 1))
        W = tf.ones(N_particles, dtype=DTYPE) / tf.cast(N_particles, DTYPE)
        P = tf.eye(model.state_dim, dtype=DTYPE) * 5.0
        total_log_l = tf.cast(0.0, DTYPE)

        # 4. Sequential Likelihood Estimation via dPFPF
        for n in range(len(observations)):
            model.current_n.assign(tf.cast(n + 1, DTYPE))

            # The filter 'step' must be differentiable for the GradientTape below.
            X, W, P = filter_engine.step(observations[n], X, W, P)

            # Accumulate incremental log-likelihood contributions.
            innov = tf.cast(observations[n], DTYPE) - model.h_func(X)
            log_obs_weights = -0.5 * (tf.square(innov) / (model.sigma_w_sq + 1e-6))
            log_obs_weights = tf.clip_by_value(log_obs_weights, -15.0, 0.0)
            total_log_l += tf.reduce_logsumexp(log_obs_weights) - tf.math.log(tf.cast(N_particles, DTYPE))

        # Apply the tempering factor 'beta' to the total unnormalized log-posterior.
        return beta * tf.reshape(total_log_l + prior, [])

    @tf.function
    def clipped_value_and_grad(theta):
        """
        Wrapper to compute gradients and perform gradient clipping for stability.
        """
        with tf.GradientTape() as tape:
            tape.watch(theta)
            val = target_log_prob(theta)

        grad = tape.gradient(val, theta)
        # Norm clipping prevents "exploding gradients" during the initial
        # phase of the MCMC chain when theta might be in high-energy regions.
        grad = tf.clip_by_norm(grad, 5.0)
        return val, grad

    # Initial parameter guess
    current_state = tf.constant([8.0], dtype=DTYPE)

    # Configure the Hamiltonian Monte Carlo Kernel
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=stepsize,
        num_leapfrog_steps=num_leapfrog_steps
    )

    # Overwrite the standard gradient call with our clipped version.
    hmc_kernel._target_log_prob_fn = clipped_value_and_grad

    # Add adaptive step-size logic to maintain a target acceptance rate (0.65).
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_kernel,
        num_adaptation_steps=int(num_samples * 0.8),
        target_accept_prob=0.65
    )

    # Prepare for sampling loop
    pkr = adaptive_kernel.bootstrap_results(current_state)
    results, acceptance_history = [], []

    print(f"\nStarting Tempered PHMC (Beta={beta}, N={N_particles}, Samples={num_samples})...")

    # Primary MCMC Loop
    for i in tqdm(range(num_samples), desc="HMC Chain"):
        # Take a leapfrog step and determine acceptance via Metropolis-Hastings.
        current_state, pkr = adaptive_kernel.one_step(current_state, pkr)

        # Execute external callback for logging (e.g., printing current sigma^2).
        if callback:
            callback(i, current_state, pkr)

        results.append(current_state.numpy())
        acceptance_history.append(pkr.inner_results.is_accepted.numpy())

    return np.array(results), np.mean(acceptance_history)
