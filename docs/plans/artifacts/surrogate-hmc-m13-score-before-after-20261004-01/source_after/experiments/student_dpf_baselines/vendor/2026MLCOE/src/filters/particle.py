"""
Particle Filters Module
=======================
This module implements Sequential Monte Carlo (SMC) methods for non-linear
and non-Gaussian state estimation. It includes the Bootstrap Particle Filter
(BPF), Gaussian Sum Monte Carlo (GSMC), and the Unscented Particle Filter (UPF).

These methods represent the posterior distribution using a weighted ensemble
of particles, allowing them to capture multi-modality and complex non-linear
dependencies that analytical filters cannot.

Author: Joowon Lee
Date: 2026-02-26
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Import classical components from the sibling module
from .classical import UKF, ESRF

# Preserve global precision and jitter settings for numerical consistency
DTYPE = tf.float64
JITTER = 1e-6
tfd = tfp.distributions


class BPF:
    """
    Bootstrap Particle Filter (BPF).
    --------------------------------
    The standard Sequential Importance Resampling (SIR) filter.
    It uses the transition prior p(x_t | x_{t-1}) as the proposal distribution.
    While simple and highly parallelizable, it requires a large number of particles
    (N) to avoid degeneracy in high-dimensional or low-noise systems.

    Mathematical Context:
    ---------------------
    The BPF is the foundational approach to particle filtering, defined by three steps:
    1. Proposal: Samples x_t^(i) ~ p(x_t | x_{t-1}^(i)) directly from the transition model.
    2. Weighting: Updates weights using the likelihood w_t^(i) proportional to p(y_t | x_t^(i)).
    3. Resampling: Stochastically eliminates particles with low weights and multiplies
       particles with high weights to refocus computational effort on likely state regions.
    """
    def __init__(self, model, N=1000):
        self.m = model; self.N = N
        self.dim = model.state_dim
        self.X = tf.Variable(tf.zeros((N, self.dim), dtype=DTYPE))
        self.W = tf.Variable(tf.ones(N, dtype=DTYPE)/N); self.ess = tf.Variable(float(N), dtype=DTYPE)

    def init(self, x, P):
        """
        Initializes the particle ensemble from the prior distribution.
        """
        dist = tfd.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(P))
        self.X.assign(dist.sample(self.N))
        self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))

    @tf.function
    def step(self, z):
        """
        Executes a standard SIR filtering step.
        """
        # 1. Prediction (Proposal): Sample from the transition dynamics prior
        # Draw process noise samples based on the filter's Q matrix.
        noise = tfd.MultivariateNormalTriL(loc=tf.zeros(self.dim, dtype=DTYPE), scale_tril=tf.linalg.cholesky(self.m.Q_filter)).sample(self.N)

        # Propagate particles through the linear transition matrix F and add noise
        # Note: If F were non-linear, this would be f(X) + noise.
        self.X.assign(tf.transpose(self.m.F @ tf.transpose(self.X)) + noise)

        # 2. Measurement Update (Weighting)
        # Evaluate the distance between the proposed states and the actual observation 'z'
        preds = self.m.h_func(self.X); res = z - preds
        quad = tf.reduce_sum(tf.matmul(res, self.m.R_inv_filter) * res, axis=1)
        log_lik = -0.5 * quad

        # Sequential weight update: W_t = W_{t-1} * p(y_t | x_t)
        # Safe log transformation adding a small epsilon (1e-30) to prevent log(0)
        log_w_prev = tf.math.log(self.W + 1e-30); log_w_unnorm = log_lik + log_w_prev

        # Normalize weights safely
        w = tf.exp(log_w_unnorm - tf.reduce_max(log_w_unnorm)); w = w / tf.reduce_sum(w)
        self.ess.assign(1.0 / tf.reduce_sum(tf.square(w)))

        # 3. Conditional Resampling
        # BPF generally uses a harsher resampling threshold (e.g., 10% or 0.1 * N)
        # compared to UPF/GSMC, because prior-based proposals naturally decay faster.
        if self.ess < self.N * 0.1:
            idx = tf.random.categorical(tf.math.log(w[None, :]), self.N)[0]
            self.X.assign(tf.gather(self.X, idx))
            self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
        else: self.W.assign(w)

        # Return the weighted posterior mean
        return tf.reduce_sum(self.X * self.W[:, None], axis=0)


class UPF:
    """
    Unscented Particle Filter (UPF).
    --------------------------------
    Improves upon the standard BPF by using an Unscented Kalman Filter (UKF)
    to generate the proposal distribution for *each* particle. This incorporates
    the current observation 'z' into the proposal, moving particles to regions
    of high likelihood before the weighting step, severely reducing degeneracy.

    Mathematical Context:
    ---------------------
    In standard Sequential Importance Resampling, the proposal density is typically
    the prior: q(x_t | x_{t-1}, y_t) = p(x_t | x_{t-1}). This ignores the current
    observation y_t, meaning many particles are proposed in low-likelihood regions,
    wasting computational resources.

    The UPF uses a localized UKF for each particle to construct a Gaussian
    approximation of the optimal proposal: N(x_t | mu_UKF, P_UKF). Because the
    UKF conditions on y_t, particles are drawn directly in the high-probability
    regions of the true posterior, allowing the filter to survive complex non-linear
    dynamics with a drastically smaller ensemble size (N).
    """
    def __init__(self, model, N=50):
        # The UPF requires a much smaller N than BPF due to highly accurate UKF proposals.
        self.m = model; self.N = N; self.ukf = UKF(model)
        self.dim = model.state_dim
        # Maintain individual states (X) and covariances (Ps) for each particle's UKF tracking
        self.X = tf.Variable(tf.zeros((N, self.dim), dtype=DTYPE))
        self.Ps = tf.Variable(tf.zeros((N, self.dim, self.dim), dtype=DTYPE))
        self.W = tf.Variable(tf.ones(N, dtype=DTYPE)/N); self.ess = tf.Variable(float(N), dtype=DTYPE)

    def init(self, x, P):
        """
        Samples the initial ensemble from the prior Multivariate Normal distribution.
        """
        dist = tfd.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(P))
        self.X.assign(dist.sample(self.N))
        # Initialize the same prior covariance P for all individual UKFs
        self.Ps.assign(tf.tile(P[None, :, :], [self.N, 1, 1]))
        self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))

    @tf.function
    def step(self, z):
        """
        Executes a single filtering step using UKF-generated optimal proposals.
        """
        def single_ukf_step(inp):
            """
            Internal function mapped across all particles.
            Runs a full UKF predict/update cycle for a single particle using observation z.
            """
            x_i, P_i = inp; self.ukf.init(x_i, P_i); self.ukf.step(z)

            # Robust Cholesky decomposition: Fallback to adding jitter to the diagonal
            # if the UKF covariance matrix loses positive-definiteness due to numerical limits.
            try: L = tf.linalg.cholesky(self.ukf.P)
            except: L = tf.linalg.cholesky(self.ukf.P + tf.eye(self.dim, dtype=DTYPE)*1e-3)

            # Sample the new particle state from the UKF-updated posterior (the proposal)
            # x_new = mu_UKF + L * standard_normal
            samp = self.ukf.x + tf.linalg.matvec(L, tf.random.normal((self.dim,), dtype=DTYPE))
            return samp, self.ukf.P

        # 1. Proposal Generation: Map the UKF step across the entire ensemble in parallel
        X_new, P_new = tf.map_fn(single_ukf_step, (self.X, self.Ps), dtype=(DTYPE, DTYPE))
        self.X.assign(X_new); self.Ps.assign(P_new)

        # 2. Importance Weighting
        preds = self.m.h_func(self.X); res = z - preds
        quad = tf.reduce_sum(tf.matmul(res, self.m.R_inv_filter) * res, axis=1)
        log_lik = -0.5 * quad

        # Normalize weights safely in log-space using the max-shift trick
        w = tf.exp(log_lik - tf.reduce_max(log_lik)); w = w / tf.reduce_sum(w)

        # 3. Effective Sample Size & Resampling
        self.ess.assign(1.0 / tf.reduce_sum(tf.square(w)))
        if self.ess < self.N/2:
            # Resample both the particle states (X) and their associated UKF covariances (Ps)
            # This ensures that the cloned particles carry their local covariance structure forward.
            idx = tf.random.categorical(tf.math.log(w[None, :]), self.N)[0]
            self.X.assign(tf.gather(self.X, idx))
            self.Ps.assign(tf.gather(self.Ps, idx))
            self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
        else: self.W.assign(w)

        # Return the weighted mean estimate
        return tf.reduce_sum(self.X * self.W[:, None], axis=0)


class GSMC:
    """
    Gaussian Sum Monte Carlo (GSMC).
    --------------------------------
    Approximates the posterior distribution as a Gaussian Mixture Model (GMM).
    It uses an Ensemble Square Root Filter (ESRF) to propagate a set of
    particles, effectively maintaining a deterministic ensemble update coupled
    with a particle-based likelihood weighting and resampling mechanism.

    Mathematical Context:
    ---------------------
    Standard particle filters suffer from sample impoverishment because they rely
    purely on stochastic propagation. GSMC bridges the gap between deterministic
    Kalman filtering and stochastic SMC. By using the ESRF to propose the new
    particle locations, GSMC ensures that the ensemble is deterministically moved
    toward the observation before the likelihood weighting is applied. The
    posterior is thus represented as a weighted sum of Gaussian components,
    drastically reducing the variance of the importance weights.
    """
    def __init__(self, model, N=100):
        # Initialize the underlying state-space model, ensemble size (N),
        # and the underlying ESRF which handles the deterministic transitions.
        self.m = model; self.N = N; self.esrf = ESRF(model, N)
        # Initialize uniform importance weights (1/N) and Effective Sample Size (ESS)
        self.W = tf.Variable(tf.ones(N, dtype=DTYPE)/N); self.ess = tf.Variable(float(N), dtype=DTYPE)

    def init(self, x, P):
        """
        Initializes the GSMC filter.
        Delegates state and covariance initialization to the underlying ESRF.
        """
        self.esrf.init(x, P); self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))

    @tf.function
    def step(self, z):
        """
        Executes a single filtering step: Propagate -> Weight -> Resample.
        """
        # 1. Prediction Step: Propagate the ensemble deterministically via ESRF
        # The ESRF handles the process noise and state transition inherently.
        self.esrf.step(z)
        X_prop = self.esrf.X

        # 2. Observation Mapping: Map propagated particles to measurement space
        preds = self.m.h_func(X_prop)
        res = z - preds

        # 3. Likelihood Evaluation
        # Compute the Mahalanobis distance (quadratic form) for the Gaussian likelihood:
        # (z - h(x))^T * R^-1 * (z - h(x))
        quad = tf.reduce_sum(tf.matmul(res, self.m.R_inv_filter) * res, axis=1)
        log_lik = -0.5 * quad

        # Log-Sum-Exp Trick: Shift log-likelihoods by the maximum value to prevent
        # numerical underflow/overflow when taking the exponential for weights.
        w = tf.exp(log_lik - tf.reduce_max(log_lik)); w = w / tf.reduce_sum(w)

        # 4. Effective Sample Size (ESS) calculation
        # ESS = 1 / sum(w^2). A low ESS indicates weight degeneracy (few particles carry all the weight).
        self.ess.assign(1.0 / tf.reduce_sum(tf.square(w)))

        # 5. Conditional Resampling
        # Trigger multinomial resampling if ESS drops below 50% of the ensemble size.
        if self.ess < self.N/2:
            # Categorical sampling using the log-weights to select indices
            idx = tf.random.categorical(tf.math.log(w[None, :]), self.N)[0]
            # Hard-resample the underlying ESRF states based on drawn indices
            self.esrf.X.assign(tf.gather(self.esrf.X, idx))
            # Reset weights to uniform after resampling to maintain unbiasedness
            self.W.assign(tf.ones(self.N, dtype=DTYPE)/float(self.N))
        else: self.W.assign(w)

        # Return the weighted mean of the ensemble as the final state estimate
        return tf.reduce_sum(self.esrf.X * self.W[:, None], axis=0)
