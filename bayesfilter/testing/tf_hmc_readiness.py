"""Small TensorFlow/TFP HMC-readiness fixtures for BayesFilter tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_linear_gaussian_score,
    tf_qr_linear_gaussian_score_hessian,
)
from bayesfilter.linear.kalman_qr_tf import tf_qr_linear_gaussian_log_likelihood
from bayesfilter.linear.kalman_tf import tf_kalman_filter
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import tf_svd_cut4_score
from bayesfilter.testing.nonlinear_diagnostics_tf import (
    nonlinear_sigma_point_score_branch_summary,
)
from bayesfilter.testing.nonlinear_models_tf import (
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    model_b_observations_tf,
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
        # Independent diagnostic: keep the entire reverse pass inside XLA so
        # TensorList intermediates do not cross an eager/compiled boundary.
        @tf.function(input_signature=[tf.TensorSpec([2], tf.float64)], jit_compile=True)
        def reference(position):
            with tf.GradientTape() as tape:
                tape.watch(position)
                value = self.log_likelihood(position)
            return value, tape.gradient(value, position)

        return reference(params)

    def log_likelihood_autodiff_score_hessian(
        self,
        parameters: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        # Eager covariance/Joseph reference is independent of the compiled QR
        # derivative recurrence, including its second-order factor rules.
        with tf.GradientTape(persistent=True) as outer:
            outer.watch(params)
            with tf.GradientTape() as inner:
                inner.watch(params)
                model, _ = self.model_and_derivatives(params)
                value, _, _ = tf_kalman_filter.python_function(
                    observations=self.observations,
                    transition_offset=model.transition_offset,
                    transition_matrix=model.transition_matrix,
                    transition_covariance=model.transition_covariance,
                    observation_offset=model.observation_offset,
                    observation_matrix=model.observation_matrix,
                    observation_covariance=model.observation_covariance,
                    initial_state_mean=model.initial_mean,
                    initial_state_covariance=model.initial_covariance,
                    jitter=self.jitter,
                    return_filtered=False,
                )
            score = inner.gradient(value, params)
        hessian = outer.jacobian(score, params, experimental_use_pfor=False)
        del outer
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

        @tf.custom_gradient
        def analytical_target(position):
            value, score = self.target_log_prob_and_grad(position)
            return value, lambda cotangent: cotangent * score

        return analytical_target(params)

    def target_log_prob_and_grad(self, parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        model, derivatives = self.model_and_derivatives(params)
        result = tf_qr_linear_gaussian_score(self.observations, model, derivatives, jitter=self.jitter)
        value, score = result.log_likelihood, result.score
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

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float64)], jit_compile=True)
    def sample_reference(position):
        return tfm.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=position,
            kernel=kernel,
            trace_fn=lambda _state, kernel_results: {
                "is_accepted": kernel_results.is_accepted,
                "log_accept_ratio": kernel_results.log_accept_ratio,
                "target_log_prob": kernel_results.accepted_results.target_log_prob,
            },
            seed=tf.constant(seed, dtype=tf.int32),
        )

    samples, trace = sample_reference(initial_state)
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


@dataclass(frozen=True)
class ModelBNonlinearSVDTarget:
    """Smooth Model B target for the first nonlinear V1 HMC smoke."""

    observations: tf.Tensor
    initial_parameters: tf.Tensor
    prior_mean: tf.Tensor
    prior_scale: tf.Tensor
    backend: str

    @staticmethod
    def default() -> "ModelBNonlinearSVDTarget":
        return ModelBNonlinearSVDTarget(
            observations=model_b_observations_tf(),
            initial_parameters=tf.constant([0.70, 0.25, 0.80], dtype=tf.float64),
            prior_mean=tf.constant([0.70, 0.25, 0.80], dtype=tf.float64),
            prior_scale=tf.constant([0.25, 0.15, 0.25], dtype=tf.float64),
            backend="tf_svd_cut4",
        )

    def model_and_derivatives(self, parameters: tf.Tensor):
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        return (
            make_nonlinear_accumulation_model_tf(
                rho=params[0],
                sigma=params[1],
                beta=params[2],
            ),
            make_nonlinear_accumulation_first_derivatives_tf(
                rho=params[0],
                sigma=params[1],
                beta=params[2],
            ),
        )

    def log_likelihood_and_score(self, parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        model, derivatives = self.model_and_derivatives(parameters)
        result = tf_svd_cut4_score(
            self.observations,
            model,
            derivatives,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        )
        return result.log_likelihood, result.score

    def target_log_prob(self, parameters: tf.Tensor) -> tf.Tensor:
        value, _score = self.log_likelihood_and_score(parameters)
        centered = (tf.convert_to_tensor(parameters, dtype=tf.float64) - self.prior_mean)
        prior_quadratic = tf.reduce_sum(tf.square(centered / self.prior_scale))
        return value - 0.5 * prior_quadratic

    def target_log_prob_and_grad(self, parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        params = tf.convert_to_tensor(parameters, dtype=tf.float64)
        value, score = self.log_likelihood_and_score(params)
        centered = params - self.prior_mean
        prior_score = -(centered / tf.square(self.prior_scale))
        prior_quadratic = tf.reduce_sum(tf.square(centered / self.prior_scale))
        return value - 0.5 * prior_quadratic, score + prior_score

    def branch_summary(self) -> Mapping[str, tf.Tensor | float | int | tuple[str, ...]]:
        grid = tf.constant(
            [
                [0.62, 0.20, 0.70],
                [0.66, 0.23, 0.75],
                [0.70, 0.25, 0.80],
                [0.74, 0.27, 0.85],
                [0.78, 0.30, 0.90],
            ],
            dtype=tf.float64,
        )
        summary = nonlinear_sigma_point_score_branch_summary(
            self.observations,
            grid,
            lambda params: make_nonlinear_accumulation_model_tf(
                rho=params[0],
                sigma=params[1],
                beta=params[2],
            ),
            lambda params: make_nonlinear_accumulation_first_derivatives_tf(
                rho=params[0],
                sigma=params[1],
                beta=params[2],
            ),
            backend=self.backend,
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        )
        return {
            "ok_count": summary.ok_count,
            "total_count": summary.total_count,
            "ok_fraction": summary.ok_fraction,
            "active_floor_count": summary.active_floor_count,
            "weak_spectral_gap_count": summary.weak_spectral_gap_count,
            "nonfinite_count": summary.nonfinite_count,
            "failure_labels": summary.failure_labels,
            "max_deterministic_residual": summary.max_deterministic_residual,
            "max_support_residual": summary.max_support_residual,
        }


def run_model_b_nonlinear_svd_cut4_hmc_smoke(
    *,
    num_results: int = 12,
    num_burnin_steps: int = 6,
    step_size: float = 0.01,
    num_leapfrog_steps: int = 2,
    seed: tuple[int, int] = (20260514, 23),
) -> Mapping[str, tf.Tensor | tuple[str, ...]]:
    """Run a tiny CPU-oriented HMC smoke for nonlinear Model B with SVD-CUT4."""

    target = ModelBNonlinearSVDTarget.default()
    branch = target.branch_summary()
    if branch["ok_count"] != branch["total_count"]:
        raise ValueError(f"Model B branch gate failed before HMC smoke: {branch}")
    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target.target_log_prob,
        step_size=tf.constant(step_size, dtype=tf.float64),
        num_leapfrog_steps=num_leapfrog_steps,
    )
    samples, trace = tfm.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=target.initial_parameters,
        kernel=kernel,
        trace_fn=lambda _state, kernel_results: {
            "is_accepted": kernel_results.is_accepted,
            "log_accept_ratio": kernel_results.log_accept_ratio,
            "target_log_prob": kernel_results.accepted_results.target_log_prob,
        },
        seed=tf.constant(seed, dtype=tf.int32),
    )
    value, gradient = target.target_log_prob_and_grad(target.initial_parameters)
    return {
        "samples": samples,
        "sample_mean": tf.reduce_mean(samples, axis=0),
        "sample_stddev": tf.math.reduce_std(samples, axis=0),
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
        "branch_ok_count": tf.constant(branch["ok_count"], dtype=tf.int32),
        "branch_total_count": tf.constant(branch["total_count"], dtype=tf.int32),
        "branch_active_floor_count": tf.constant(branch["active_floor_count"], dtype=tf.int32),
        "branch_weak_spectral_gap_count": tf.constant(
            branch["weak_spectral_gap_count"],
            dtype=tf.int32,
        ),
        "branch_nonfinite_count": tf.constant(branch["nonfinite_count"], dtype=tf.int32),
        "branch_failure_labels": branch["failure_labels"],
        "min_target_log_prob": tf.reduce_min(trace["target_log_prob"]),
        "max_target_log_prob": tf.reduce_max(trace["target_log_prob"]),
        "max_abs_log_accept_ratio": tf.reduce_max(tf.abs(trace["log_accept_ratio"])),
    }
