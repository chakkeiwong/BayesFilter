"""Actual Kalman and sigma-point filter targets for campaign K0--K7.

No independent-reference computation is imported here. Graphs have a stable
signature for each supported batch size and are reused across evaluations.
"""
from __future__ import annotations

import math
from pathlib import Path
import tensorflow as tf

from .designs import digest
from .ssm_campaign_profiles import validate_parameters


class CampaignSSMTarget:
    batch_rank_policy = "rank2_required"

    def __init__(self, target, parameters=None, data=None, *, control="baseline", jit_compile=True):
        self.profile = validate_parameters(target, parameters)
        self.target_id, self.parameters, self.control = target, {}, control
        if control not in {"baseline", "noop", "wrong_score", "ignore_data"}:
            raise ValueError("unsupported campaign target control")
        self.parameter_dim = len(self.profile.raw_names)
        self.jit_compile = bool(jit_compile)
        self._graphs = {}
        self.bundle = None
        self.prior_mean = self.profile.prior_mean
        self.prior_scale = self.profile.prior_scale
        if self.profile.family == "nonlinear":
            # This fixture module owns TensorFlow constants. Import outside any
            # traced graph so a later adapter never captures another graph's
            # tensors. The numerical call remains inside our stable graph.
            from bayesfilter.testing.simple_nonlinear_generic_target_adapter_tf import make_batched_model_b_svd_ukf_components
            from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import tf_batched_svd_sigma_point_value_and_score
            self._make_nonlinear_components = make_batched_model_b_svd_ukf_components
            self._nonlinear_filter = tf_batched_svd_sigma_point_value_and_score
        if self.profile.family == "multivariate":
            from bayesfilter.testing.deterministic_lgssm_exact_target_tf import load_deterministic_lgssm_exact_target
            from bayesfilter.testing.multidim_triangular_lgssm_tf import raw_truth_from_contract
            self.bundle = load_deterministic_lgssm_exact_target()
            actual = self.bundle.fixture["observations"]
            if data is not None and digest(data) != digest(actual):
                raise ValueError("K6 data must match the checked 120x4 fixture")
            data = actual
            # This is the existing model's prior center, which happens to be
            # the synthetic truth template. No fitted posterior supplies starts.
            self.prior_mean = tuple(raw_truth_from_contract(self.bundle.contract).numpy().tolist())
            self.prior_scale = (0.5,)*4 + (0.6,)*6 + (0.35,)*8
        if data is None:
            raise ValueError("campaign targets require explicitly frozen observations")
        self.data = data
        tensor = tf.convert_to_tensor(data, tf.float64)
        expected = (120, 4) if self.bundle else (self.profile.horizon,)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{target} observations require shape {expected}")
        tf.debugging.assert_all_finite(tensor, "nonfinite campaign observations")
        self.observations = tensor if self.bundle else tensor[:, None]
        self.spec = None

    def adapter_signature(self):
        return digest({"schema": "validation.ssm_campaign.v1", "profile": self.profile.payload(),
            "data": self.data, "control": self.control,
            "existing_target": self.bundle.target_signature if self.bundle else None})

    def parameter_names(self):
        return self.profile.raw_names

    def value_score_capability(self):
        from bayesfilter.inference.posterior_adapter import ValueScoreCapability
        # All eight profiles pass full-chain compilation and deterministic
        # graph/XLA leapfrog parity in test_ssm_xla_full_chain.py. This permits
        # the campaign preflight; actual GPU qualification still happens there.
        return ValueScoreCapability(value_score_authority="graph_native", xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=self.jit_compile, target_scope="inference_validation",
            runtime_backend="tensorflow_actual_ssm_filter",
            evidence_path="docs/plans/bayesfilter-hmc-ssm-xla-preflight-repair-2026-09-29.md",
            nonclaims=("synthetic validation target; GPU/XLA execution requires campaign preflight",
                       "K7 samples a sigma-point approximation; K6 has no posterior reference",
                       "no posterior correctness, convergence or default-readiness claim"))

    def source_paths(self):
        from bayesfilter.linear import kalman_qr_derivatives_tf, compiled_recurrence_tf
        from . import ssm_campaign_profiles
        paths = [__file__, ssm_campaign_profiles.__file__, kalman_qr_derivatives_tf.__file__,
                 compiled_recurrence_tf.__file__]
        if self.profile.family == "nonlinear":
            from bayesfilter.testing import simple_nonlinear_generic_target_adapter_tf, nonlinear_models_tf
            from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf, sigma_points_tf, svd_sigma_point_derivatives_tf
            paths += [m.__file__ for m in (simple_nonlinear_generic_target_adapter_tf, nonlinear_models_tf,
                experimental_batched_svd_sigma_point_tf, sigma_points_tf, svd_sigma_point_derivatives_tf)]
        if self.bundle:
            from bayesfilter.testing import deterministic_lgssm_exact_target_tf, multidim_triangular_lgssm_tf, multidim_triangular_lgssm_batched_tf
            from bayesfilter.linear import batched_kalman_svd_derivatives_tf, kalman_svd_derivatives_tf
            paths += [m.__file__ for m in (deterministic_lgssm_exact_target_tf, multidim_triangular_lgssm_tf,
                multidim_triangular_lgssm_batched_tf, batched_kalman_svd_derivatives_tf, kalman_svd_derivatives_tf)]
            paths += [str(self.bundle.config_path), str(self.bundle.fixture_path), str(self.bundle.contract_path)]
        return tuple(str(Path(p).resolve()) for p in paths)

    def initial_starts(self):
        # Four modest prior-scale displacements, declared before observing data.
        shifts = tf.constant([-1., -.3, .4, 1.], tf.float64)[:, None]
        return tf.constant(self.prior_mean, tf.float64)[None, :] + shifts*tf.constant(self.prior_scale, tf.float64)[None, :]

    def to_model(self, q):
        q = tf.convert_to_tensor(q, tf.float64)
        if self.profile.family == "persistence_noise":
            return tf.stack((self.profile.persistence_cap*tf.tanh(q[..., 0]), tf.exp(q[..., 1])), -1)
        if self.profile.family == "two_noises":
            return tf.exp(q)
        return q

    def _evaluate(self, q):
        q = tf.convert_to_tensor(q, tf.float64)
        scalar = q.shape.rank == 1
        batch = q[None, :] if scalar else q
        if batch.shape.rank != 2 or batch.shape[-1] != self.parameter_dim:
            raise ValueError("campaign target expects [parameters] or [batch, parameters]")
        size = batch.shape[0]
        limit = 64 if self.profile.family in {"nonlinear", "multivariate"} else 4096
        if size is None or not 1 <= size <= limit:
            raise ValueError(f"campaign graph requires a static batch size in [1,{limit}]")
        if size not in self._graphs:
            self._graphs[size] = tf.function(self._score_status,
                input_signature=[tf.TensorSpec([size, self.parameter_dim], tf.float64)],
                autograph=False, jit_compile=self.jit_compile)
        result = self._graphs[size](batch)
        # Shared filter helpers may generalize their traced shape after a
        # different model dimension. Restore this adapter's declared static
        # contract at the boundary; HMC must still know its event dimension.
        value, score, status = result
        result = (tf.ensure_shape(value, [size]),
                  tf.ensure_shape(score, [size, self.parameter_dim]),
                  {key:tf.ensure_shape(tensor, [size]) for key,tensor in status.items()})
        return tf.nest.map_structure(lambda x: x[0], result) if scalar else result

    def log_prob_and_grad(self, q):
        value, score, _ = self._evaluate(q)
        return value, score

    def log_density(self, q):
        return self.log_prob_and_grad(q)[0]

    def _batch_score(self, q):
        return self.log_prob_and_grad(q)

    def target_status_telemetry(self, q):
        return self._evaluate(q)[2]

    def _score_status(self, q):
        if self.bundle:
            value, score, status = self.bundle.adapter.neutra_batch_log_prob_and_grad_status(q)
            if self.control == "ignore_data":
                mean = tf.constant(self.prior_mean,tf.float64)
                scale = tf.constant(self.prior_scale,tf.float64)
                value = -.5*tf.reduce_sum(((q-mean)/scale)**2,-1)
                score = -(q-mean)/scale**2
        else:
            if self.profile.family == "nonlinear":
                value, score, status = self._nonlinear(q)
            else:
                value, score = self._kalman(q)
                finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score), -1)
                # The scalar QR path has strictly positive covariances and
                # jitter=0; it uses no spectral floors. Check its actual outputs.
                status = {"status_code": tf.where(finite, 0, 1),
                          "valid_pre_regularized_score": finite,
                          "floor_count_value": tf.zeros([q.shape[0]], tf.int32)}
            mean, scale = tf.constant(self.prior_mean, tf.float64), tf.constant(self.prior_scale, tf.float64)
            prior = -0.5*tf.reduce_sum(((q-mean)/scale)**2, -1)
            prior_score = -(q-mean)/scale**2
            if self.control == "ignore_data":
                value, score = prior, prior_score
            else:
                value, score = value+prior, score+prior_score
        finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score), -1)
        status = {**status, "valid_pre_regularized_score": status["valid_pre_regularized_score"] & finite,
                  "status_code": tf.where(finite, status["status_code"], 1)}
        if self.control == "wrong_score":
            score += tf.constant(.25, tf.float64)
        return value, score, status

    def _kalman(self, q):
        from bayesfilter.linear.kalman_qr_derivatives_tf import tf_qr_sqrt_kalman_score_batched_static
        p, b, d = self.profile, q.shape[0], self.parameter_dim
        zeros = tf.zeros([b], tf.float64)
        dz = tf.zeros([b, d], tf.float64)
        rho = tf.fill([b], tf.constant(p.rho, tf.float64))
        process = tf.fill([b], tf.constant(p.process_sd**2, tf.float64))
        obs = tf.fill([b], tf.constant(p.observation_sd**2, tf.float64))
        offset, d_offset, d_rho, d_process, d_obs = zeros, dz, dz, dz, dz
        if p.family == "location":
            offset = q[:, 0]
            d_offset = tf.ones([b, 1], tf.float64)
        elif p.family == "persistence_noise":
            rho = p.persistence_cap*tf.tanh(q[:, 0])
            d_rho = tf.stack((p.persistence_cap*(1-tf.tanh(q[:, 0])**2), zeros), -1)
            obs = tf.exp(2*q[:, 1])
            d_obs = tf.stack((zeros, 2*obs), -1)
        else:
            process, obs = tf.exp(2*q[:, 0]), tf.exp(2*q[:, 1])
            d_process = tf.stack((2*process, zeros), -1)
            d_obs = tf.stack((zeros, 2*obs), -1)
        initial_cov = process/(1-rho**2)
        d_initial_cov = d_process/(1-rho[:, None]**2) + (
            2*process*rho/(1-rho**2)**2)[:, None]*d_rho
        return tf_qr_sqrt_kalman_score_batched_static(
            observations=self.observations,
            transition_offset=zeros[:, None], transition_matrix=rho[:, None, None],
            transition_covariance=process[:, None, None],
            observation_offset=offset[:, None], observation_matrix=tf.ones([b, 1, 1], tf.float64),
            observation_covariance=obs[:, None, None], initial_state_mean=zeros[:, None],
            initial_state_covariance=initial_cov[:, None, None],
            d_initial_state_mean=dz[:, :, None], d_initial_state_covariance=d_initial_cov[:, :, None, None],
            d_transition_offset=dz[:, :, None], d_transition_matrix=d_rho[:, :, None, None],
            d_transition_covariance=d_process[:, :, None, None], d_observation_offset=d_offset[:, :, None],
            d_observation_matrix=dz[:, :, None, None], d_observation_covariance=d_obs[:, :, None, None],
            jitter=tf.constant(0., tf.float64))

    def _nonlinear(self, q):
        full = tf.stack((q[:, 0], tf.fill([q.shape[0]], tf.constant(.25, tf.float64)), q[:, 1]), -1)
        model, derivatives = self._make_nonlinear_components(full)
        # The enclosing stable parameter-to-result graph owns XLA compilation.
        # An uncompiled inner while_loop avoids a new nested XLA function for
        # every target evaluation; it is still inside that enclosing graph.
        value, score, diagnostics = self._nonlinear_filter(
            self.observations, model, derivatives, backend="tf_svd_ukf", jit_compile=False)
        floors = diagnostics["placement_floor_count"] + diagnostics["innovation_floor_count"]
        valid = (diagnostics["compiled_branch_valid"] & (floors == 0)
                 & (diagnostics["principal_sqrt_target_classified_invalid_count"] == 0))
        status = {"status_code": tf.where(valid, 0, 1), "valid_pre_regularized_score": valid,
                  "floor_count_value": floors,
                  "min_innovation_eigenvalue": diagnostics["min_innovation_eigenvalue"],
                  # Model B has a scalar observation: every positive scalar
                  # innovation covariance has spectral condition number one.
                  "innovation_condition_estimate": tf.where(
                      diagnostics["min_innovation_eigenvalue"] > 0,
                      tf.ones([q.shape[0]], tf.float64),
                      tf.fill([q.shape[0]], tf.constant(float("inf"), tf.float64)))}
        return value, tf.gather(score, [0, 2], axis=1), status
