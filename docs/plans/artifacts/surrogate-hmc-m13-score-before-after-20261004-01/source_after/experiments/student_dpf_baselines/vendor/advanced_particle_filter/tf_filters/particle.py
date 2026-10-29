"""
TensorFlow Particle Filter implementation.

Mirrors: filters/particle.py (NumPy version)

Bootstrap/SIR particle filter. Uses tf.while_loop for JIT compilation.
Resampling uses tf.random.Generator for reproducibility.
"""

import tensorflow as tf
from typing import Optional

from .base import TFFilterResult
from ..tf_models.base import TFStateSpaceModel
from ..tf_utils.resampling import (
    systematic_resample,
    stratified_resample,
    multinomial_resample,
    effective_sample_size,
    normalize_log_weights,
)


class TFBootstrapParticleFilter:
    """
    Bootstrap Particle Filter (Sequential Importance Resampling) — TF version.

    Uses the transition prior as proposal distribution.
    """

    def __init__(
        self,
        n_particles: int = 1000,
        resample_method: str = "systematic",
        resample_criterion: str = "ess",
        ess_threshold: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.n_particles = n_particles
        self.resample_method = resample_method
        self.resample_criterion = resample_criterion
        self.ess_threshold = ess_threshold
        self.seed = seed

    def _get_resampler(self):
        """Get resampling function based on method."""
        resamplers = {
            "systematic": systematic_resample,
            "stratified": stratified_resample,
            "multinomial": multinomial_resample,
        }
        if self.resample_method not in resamplers:
            raise ValueError(f"Unknown resample method: {self.resample_method}")
        return resamplers[self.resample_method]

    def filter(
        self,
        model: TFStateSpaceModel,
        observations: tf.Tensor,
        rng: Optional[tf.random.Generator] = None,
    ) -> TFFilterResult:
        """
        Run bootstrap particle filter.

        Args:
            model: TFStateSpaceModel
            observations: [T, ny] Observations
            rng: Optional tf.random.Generator (uses self.seed if None)

        Returns:
            TFFilterResult
        """
        if rng is None:
            rng = tf.random.Generator.from_seed(self.seed if self.seed is not None else 0)

        resample_fn = self._get_resampler()

        # Determine resampling mode as int for tf.cond compatibility
        # 0 = always, 1 = ess, 2 = never
        if self.resample_criterion == "always":
            resample_mode = 0
        elif self.resample_criterion == "ess":
            resample_mode = 1
        elif self.resample_criterion == "never":
            resample_mode = 2
        else:
            raise ValueError(f"Unknown resample criterion: {self.resample_criterion}")

        means, covs, ess_arr, loglik_arr, resampled_arr = self._filter_impl(
            model, observations, rng, resample_fn,
            resample_mode, self.ess_threshold,
        )
        return TFFilterResult(
            means=means,
            covariances=covs,
            ess=ess_arr,
            log_likelihood=tf.reduce_sum(loglik_arr),
            log_likelihood_increments=loglik_arr,
            resampled=resampled_arr,
        )

    @tf.function
    def _filter_impl(self, model, observations, rng, resample_fn,
                     resample_mode, ess_threshold):
        dtype = model.dtype
        T = tf.shape(observations)[0]
        N = self.n_particles
        nx = model.state_dim

        N_f = tf.cast(N, dtype)
        ess_thresh_val = tf.constant(ess_threshold, dtype=dtype) * N_f

        # Initialize particles from prior
        particles = model.sample_initial(N, rng)  # [N, nx]
        log_weights = tf.fill([N], -tf.math.log(N_f))

        # TensorArrays
        means_ta = tf.TensorArray(dtype=dtype, size=T + 1, dynamic_size=False)
        covs_ta = tf.TensorArray(dtype=dtype, size=T + 1, dynamic_size=False)
        ess_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)
        loglik_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)
        resampled_ta = tf.TensorArray(dtype=tf.bool, size=T, dynamic_size=False)

        # Initial estimate
        weights = tf.exp(log_weights)
        m0 = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
        diff0 = particles - m0
        P0 = tf.einsum('n,ni,nj->ij', weights, diff0, diff0)
        P0 = 0.5 * (P0 + tf.transpose(P0))
        means_ta = means_ta.write(0, m0)
        covs_ta = covs_ta.write(0, P0)

        def body(t, particles, log_weights, means_ta, covs_ta, ess_ta, loglik_ta, resampled_ta):
            y = observations[t]

            # Propagate particles through dynamics
            particles = model.sample_dynamics(particles, rng)

            # Compute log likelihoods
            log_lik = model.observation_log_prob(particles, y)  # [N]

            # Update weights
            log_weights = log_weights + log_lik

            # Log marginal likelihood increment
            max_lw = tf.reduce_max(log_weights)
            log_Z_t = max_lw + tf.math.log(tf.reduce_sum(tf.exp(log_weights - max_lw)))
            loglik_ta = loglik_ta.write(t, log_Z_t)

            # Normalize
            weights, _ = normalize_log_weights(log_weights)
            log_weights = tf.math.log(weights + 1e-300)

            # ESS
            ess = effective_sample_size(weights)
            ess_ta = ess_ta.write(t, ess)

            # Filtered estimate
            m = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
            diff = particles - m
            P = tf.einsum('n,ni,nj->ij', weights, diff, diff)
            P = 0.5 * (P + tf.transpose(P))
            means_ta = means_ta.write(t + 1, m)
            covs_ta = covs_ta.write(t + 1, P)

            # Resample decision
            do_resample = tf.cond(
                tf.equal(resample_mode, 0),
                lambda: True,
                lambda: tf.cond(
                    tf.equal(resample_mode, 2),
                    lambda: False,
                    lambda: ess < ess_thresh_val,
                ),
            )
            resampled_ta = resampled_ta.write(t, do_resample)

            # Conditional resampling
            def do_resample_fn():
                indices = resample_fn(weights, rng)
                new_particles = tf.gather(particles, indices)
                new_log_w = tf.fill([N], -tf.math.log(N_f))
                return new_particles, new_log_w

            def no_resample_fn():
                return particles, log_weights

            particles, log_weights = tf.cond(
                do_resample,
                do_resample_fn,
                no_resample_fn,
            )

            return (t + 1, particles, log_weights,
                    means_ta, covs_ta, ess_ta, loglik_ta, resampled_ta)

        _, _, _, means_ta, covs_ta, ess_ta, loglik_ta, resampled_ta = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=body,
            loop_vars=(0, particles, log_weights,
                       means_ta, covs_ta, ess_ta, loglik_ta, resampled_ta),
        )

        return (means_ta.stack(), covs_ta.stack(), ess_ta.stack(),
                loglik_ta.stack(), resampled_ta.stack())
