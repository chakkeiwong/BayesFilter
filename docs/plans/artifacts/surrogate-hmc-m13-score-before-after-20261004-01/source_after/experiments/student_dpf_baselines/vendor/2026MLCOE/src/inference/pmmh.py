"""
Particle Marginal Metropolis-Hastings (PMMH) Inference Module
=============================================================
Implements the PMMH engine for parameter estimation in non-linear state-space
models. This module uses a standard Bootstrap Particle Filter (BPF) as a
likelihood engine.

Mathematical Context:
---------------------
PMMH is a specific type of Particle MCMC (PMCMC) algorithm that allows for
Bayesian parameter inference when the marginal likelihood p(y_{1:T} | theta)
is intractable. It replaces the exact likelihood in the Metropolis-Hastings
ratio with an unbiased estimate obtained from a Particle Filter. Despite
the noise in the likelihood estimate, the algorithm targets the exact
posterior distribution of the parameters.

Author: Joowon Lee
Date: 2026-02-27
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

# Modular math and utility imports
from src.filters.classical import DTYPE, safe_inv

tfd = tfp.distributions

class PMMH_Engine:
    """
    Standard PMMH Likelihood Evaluator.
    ----------------------------------
    Encapsulates the logic for computing the log-marginal likelihood estimate
    using a Bootstrap Particle Filter (BPF).
    """
    def __init__(self, model, N=100):
        """
        Initializes the likelihood engine.

        Args:
            model: The state-space model instance.
            N (int): Number of particles used in the internal BPF.
        """
        self.m = model
        self.N = N

    @tf.function
    def log_likelihood(self, theta, obs):
        """
        Estimates the log-marginal likelihood p(y_{1:T} | theta) using a BPF.

        Args:
            theta: Proposed parameter value (e.g., sigma_v_sq).
            obs: The sequence of observations.

        Returns:
            log_L: Scalar tensor representing the estimated log-marginal likelihood.
        """
        # Update the model with the current parameter proposal.
        self.m.update_parameters([theta])

        # Initialize the particle ensemble from the state prior.
        X = tfd.Normal(tf.cast(0, DTYPE), tf.sqrt(tf.cast(5.0, DTYPE))).sample((self.N, 1))
        log_L = tf.cast(0.0, DTYPE)

        # Sequential Importance Resampling (SIR) loop over observations.
        for n in range(len(obs)):
            self.m.current_n.assign(tf.cast(n + 1, DTYPE))

            # 1. Prediction Step: Move particles through the transition density.
            X = self.m.propagate(X) + tfd.Normal(
                tf.cast(0, DTYPE),
                tf.sqrt(self.m.sigma_v_sq)
            ).sample((self.N, 1))

            # 2. Weighting Step: Compute incremental weights based on current observation.
            log_w = -0.5 * (tf.square(tf.cast(obs[n], DTYPE) - self.m.h_func(X)) / self.m.sigma_w_sq)

            # 3. Likelihood Accumulation: Update the total marginal likelihood estimate.
            # Using Log-Sum-Exp trick for numerical stability to avoid underflow.
            max_log_w = tf.reduce_max(log_w)
            log_L += max_log_w + tf.math.log(tf.reduce_mean(tf.exp(log_w - max_log_w)))

            # 4. Resampling Step: Multinomial resampling to mitigate particle degeneracy.
            # This step is non-differentiable (stochastic categorical sampling).
            resample_indices = tf.random.categorical(
                tf.math.log(tf.nn.softmax(tf.reshape(log_w, [-1]))[None, :]),
                self.N
            )[0]
            X = tf.gather(X, resample_indices)

        return log_L

def pmmh_bpf(model, observations, num_iter=50):
    """
    Executes the Particle Marginal Metropolis-Hastings (PMMH) sampling loop.

    Args:
        model: The state-space model instance.
        observations: The sequence of data.
        num_iter (int): Total number of MCMC iterations.

    Returns:
        samples (np.array): Trace of accepted parameter values.
        acc_rate (float): Overall acceptance rate.
    """
    # Instantiate the likelihood engine.
    engine = PMMH_Engine(model, N=100)

    # Initialization: Starting value for theta and its initial log-likelihood.
    curr_theta = tf.constant(8.0, dtype=DTYPE)
    curr_log_l = engine.log_likelihood(curr_theta, observations)

    samples, accepted = [], 0
    # Standard deviation for the Gaussian Random Walk proposal.
    proposal_std = 0.8

    print("Starting PMMH-BPF sampling...")

    # Metropolis-Hastings Chain
    for _ in tqdm(range(num_iter), desc="PMMH Chain"):
        # 1. Propose: Generate a new candidate parameter via Random Walk.
        prop_theta = curr_theta + tf.random.normal([], stddev=proposal_std, dtype=DTYPE)

        # 2. Evaluate & Decide: Standard MH acceptance step.
        if prop_theta > 0:  # Positivity constraint for variance parameters.
            prop_log_l = engine.log_likelihood(prop_theta, observations)

            # Acceptance Ratio (Log-domain): log(alpha) = log_p(prop) - log_p(curr).
            # Note: We assume a flat (uniform) prior for simplicity here.
            if tf.math.log(tf.random.uniform([], dtype=DTYPE)) < (prop_log_l - curr_log_l):
                curr_theta, curr_log_l = prop_theta, prop_log_l
                accepted += 1

        # Record the current state of the chain (whether it moved or stayed).
        samples.append(curr_theta.numpy())

    return np.array(samples), accepted / num_iter
