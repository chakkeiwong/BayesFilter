"""Small TensorFlow/TFP HMC-readiness fixtures for BayesFilter tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_linear_gaussian_score_hessian,
)
from bayesfilter.linear.kalman_qr_tf import tf_qr_linear_gaussian_log_likelihood
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)


tfm = tfp.mcmc


@dataclass(frozen=True)
class QRStaticLGSSMTarget:
    """Deterministic static LGSSM target used for first v1 HMC smoke tests."""

    observations: tf.Tensor
    initial_parameters: tf.Tensor
    prior_scale: tf.Tensor
    jitter: tf.Tensor

    @staticmethod
    def default() -> "QRStaticLGSSMTarget":
        return QRStaticLGSSMTarget(
            observations=tf.constant([[0.18], [0.05], [0.16], [0.11]], dtype=tf.float64),
            initial_parameters=tf.constant([0.20, -1.05], dtype=tf.float64),
            prior_scale=tf.constant([1.0, 1.0], dtype=tf.float64),
            jitter=tf.constant(1e-9, dtype=tf.float64),
        )

    def model_and_derivatives(
        self,
        parameters: tf.Tensor,
    ) -> tuple[TFLinearGaussianStateSpace, TFLinearGaussianStateSpaceDerivatives]:
        rho_param, log_measurement_noise = tf.unstack(
            tf.convert_to_tensor(parameters, dtype=tf.float64)
        )
        tanh_rho = tf.math.tanh(rho_param)
        rho = 0.75 * tanh_rho
        drho = 0.75 * (1.0 - tanh_rho**2)
        d2rho = -1.5 * tanh_rho * (1.0 - tanh_rho**2)
        measurement_variance = tf.exp(2.0 * log_measurement_noise)
        d_measurement_variance = 2.0 * measurement_variance
        d2_measurement_variance = 4.0 * measurement_variance

        model = TFLinearGaussianStateSpace(
            initial_mean=tf.constant([0.1], dtype=tf.float64),
            initial_covariance=tf.constant([[0.35]], dtype=tf.float64),
            transition_offset=tf.constant([0.02], dtype=tf.float64),
            transition_matrix=tf.reshape(rho, [1, 1]),
            transition_covariance=tf.constant([[0.07]], dtype=tf.float64),
            observation_offset=tf.constant([0.01], dtype=tf.float64),
            observation_matrix=tf.constant([[1.2]], dtype=tf.float64),
            observation_covariance=tf.reshape(measurement_variance, [1, 1]),
        )
        derivatives = TFLinearGaussianStateSpaceDerivatives(
            d_initial_mean=tf.zeros([2, 1], dtype=tf.float64),
            d_initial_covariance=tf.zeros([2, 1, 1], dtype=tf.float64),
            d_transition_offset=tf.zeros([2, 1], dtype=tf.float64),
            d_transition_matrix=tf.reshape(tf.stack([drho, 0.0]), [2, 1, 1]),
            d_transition_covariance=tf.zeros([2, 1, 1], dtype=tf.float64),
            d_observation_offset=tf.zeros([2, 1], dtype=tf.float64),
            d_observation_matrix=tf.zeros([2, 1, 1], dtype=tf.float64),
            d_observation_covariance=tf.reshape(
                tf.stack([0.0, d_measurement_variance]),
                [2, 1, 1],
            ),
            d2_initial_mean=tf.zeros([2, 2, 1], dtype=tf.float64),
            d2_initial_covariance=tf.zeros([2, 2, 1, 1], dtype=tf.float64),
            d2_transition_offset=tf.zeros([2, 2, 1], dtype=tf.float64),
            d2_transition_matrix=tf.reshape(
                tf.stack([d2rho, 0.0, 0.0, 0.0]),
                [2, 2, 1, 1],
            ),
            d2_transition_covariance=tf.zeros([2, 2, 1, 1], dtype=tf.float64),
            d2_observation_offset=tf.zeros([2, 2, 1], dtype=tf.float64),
            d2_observation_matrix=tf.zeros([2, 2, 1, 1], dtype=tf.float64),
            d2_observation_covariance=tf.reshape(
                tf.stack([0.0, 0.0, 0.0, d2_measurement_variance]),
                [2, 2, 1, 1],
            ),
        )
        return model, derivatives

    def log_likelihood(self, parameters: tf.Tensor) -> tf.Tensor:
        model, _ = self.model_and_derivatives(parameters)
        return tf_qr_linear_gaussian_log_likelihood(
            self.observations,
            model,
            backend="tf_qr",
            jitter=self.jitter,
        ).log_likelihood

    def log_likelihood_and_autodiff_score(
        self,
        parameters: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(params)
            value = self.log_likelihood(params)
        score = tape.gradient(value, params)
        return value, score

    def log_likelihood_autodiff_score_hessian(
        self,
        parameters: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        with tf.GradientTape() as outer:
            outer.watch(params)
            with tf.GradientTape() as inner:
                inner.watch(params)
                value = self.log_likelihood(params)
            score = inner.gradient(value, params)
        hessian = outer.jacobian(score, params)
        return value, score, hessian

    def analytic_score_hessian(self, parameters: tf.Tensor):
        model, derivatives = self.model_and_derivatives(parameters)
        return tf_qr_linear_gaussian_score_hessian(
            self.observations,
            model,
            derivatives,
            jitter=self.jitter,
        )

    def target_log_prob(self, parameters: tf.Tensor) -> tf.Tensor:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        value = self.log_likelihood(params)
        prior_quadratic = tf.reduce_sum(tf.square(params / self.prior_scale))
        return value - 0.5 * prior_quadratic

    def target_log_prob_and_grad(self, parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        value, score = self.log_likelihood_and_autodiff_score(params)
        prior_score = -(params / tf.square(self.prior_scale))
        prior_quadratic = tf.reduce_sum(tf.square(params / self.prior_scale))
        return value - 0.5 * prior_quadratic, score + prior_score

    def curvature_diagnostics(self, parameters: tf.Tensor) -> Mapping[str, tf.Tensor]:
        result = self.analytic_score_hessian(parameters)
        autodiff_value, autodiff_score, autodiff_hessian = (
            self.log_likelihood_autodiff_score_hessian(parameters)
        )
        prior_precision = tf.linalg.diag(1.0 / tf.square(self.prior_scale))
        target_hessian = result.hessian - prior_precision
        return {
            "log_likelihood": result.log_likelihood,
            "score": result.score,
            "hessian": result.hessian,
            "autodiff_log_likelihood": autodiff_value,
            "autodiff_score": autodiff_score,
            "autodiff_hessian": autodiff_hessian,
            "value_residual": tf.abs(result.log_likelihood - autodiff_value),
            "score_residual": tf.reduce_max(tf.abs(result.score - autodiff_score)),
            "hessian_residual": tf.reduce_max(tf.abs(result.hessian - autodiff_hessian)),
            "target_hessian": target_hessian,
            "hessian_symmetry_residual": tf.reduce_max(
                tf.abs(target_hessian - tf.transpose(target_hessian))
            ),
            "negative_hessian_eigenvalues": tf.linalg.eigvalsh(-target_hessian),
        }


def run_qr_static_lgssm_hmc_smoke(
    *,
    num_results: int = 16,
    num_burnin_steps: int = 8,
    step_size: float = 0.05,
    num_leapfrog_steps: int = 3,
    seed: tuple[int, int] = (20260511, 17),
) -> Mapping[str, tf.Tensor]:
    """Run a tiny CPU-oriented HMC smoke for the first QR v1 target."""

    target = QRStaticLGSSMTarget.default()
    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target.target_log_prob,
        step_size=tf.constant(step_size, dtype=tf.float64),
        num_leapfrog_steps=num_leapfrog_steps,
    )
    initial_state = target.initial_parameters
    samples, trace = tfm.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=kernel,
        trace_fn=lambda _state, kernel_results: {
            "is_accepted": kernel_results.is_accepted,
            "log_accept_ratio": kernel_results.log_accept_ratio,
            "target_log_prob": kernel_results.accepted_results.target_log_prob,
        },
        seed=tf.constant(seed, dtype=tf.int32),
    )
    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_stddev = tf.math.reduce_std(samples, axis=0)
    value, gradient = target.target_log_prob_and_grad(initial_state)
    curvature = target.curvature_diagnostics(initial_state)
    return {
        "samples": samples,
        "sample_mean": sample_mean,
        "sample_stddev": sample_stddev,
        "acceptance_rate": tf.reduce_mean(tf.cast(trace["is_accepted"], tf.float64)),
        "finite_sample_count": tf.reduce_sum(
            tf.cast(tf.reduce_all(tf.math.is_finite(samples), axis=-1), tf.int32)
        ),
        "nonfinite_sample_count": tf.reduce_sum(
            tf.cast(tf.logical_not(tf.reduce_all(tf.math.is_finite(samples), axis=-1)), tf.int32)
        ),
        "initial_target_log_prob": value,
        "initial_gradient": gradient,
        "initial_gradient_finite": tf.reduce_all(tf.math.is_finite(gradient)),
        "initial_hessian_symmetry_residual": curvature["hessian_symmetry_residual"],
        "initial_negative_hessian_eigenvalues": curvature["negative_hessian_eigenvalues"],
        "min_target_log_prob": tf.reduce_min(trace["target_log_prob"]),
        "max_target_log_prob": tf.reduce_max(trace["target_log_prob"]),
        "max_abs_log_accept_ratio": tf.reduce_max(tf.abs(trace["log_accept_ratio"])),
    }
