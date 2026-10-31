"""Offline SIR message diagnostic; not canonical admission or original KDM.

The replicated multivariate cSMC fitter supplies a next-policy integral.
Quadratic and positive RBF approximations are frozen before score evaluation.
"""

import math

import tensorflow as tf

from bayesfilter.highdim.ledh_backward_guide_tf import (
    FrozenMixtureBackwardGuide,
    FrozenQuadraticBackwardGuide,
)
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.replication.multivariate_controlled_smc_tf import (
    MultivariateGaussianControlledSMC,
    covariance_from_precision,
    multivariate_twist,
    quadratic,
)


def positive_information_projection(precision, linear, center):
    """Project to positive information, preserving the score at the center.

    A dimension-scaled roundoff floor protects reconstructed zero eigenvalues.
    Well-conditioned positive inputs are returned exactly, without projection.
    """
    symmetric = .5 * (precision + tf.linalg.matrix_transpose(precision))
    eigenvalues, eigenvectors = tf.linalg.eigh(symmetric)
    scale = tf.reduce_max(tf.abs(eigenvalues), axis=-1)
    floor = 64 * 2.220446049250313e-16 * precision.shape[-1] * scale
    changed = tf.reduce_min(eigenvalues, axis=-1) < floor
    clipped = tf.maximum(eigenvalues, floor[..., None])
    projected = tf.linalg.matmul(eigenvectors * clipped[..., None, :],
                                 eigenvectors, transpose_b=True)
    projected = .5 * (projected + tf.linalg.matrix_transpose(projected))
    projected = tf.where(changed[..., None, None], projected, precision)
    vector = linear + tf.linalg.matvec(projected - precision, center)
    return projected, vector, changed, eigenvalues[..., 0]


def next_policy_log_integral(transition_mean, points, policy, precision, logdet):
    """States are x_1,...,x_(T-1); policies include x_0,...,x_T."""
    return multivariate_twist(
        transition_mean(points), precision, logdet,
        policy["A"][2:, 0], policy["b"][2:, 0], policy["c"][2:, 0],
    )


class SIRGuideFitter:
    """One cSMC fit, independent controlled pilot, disjoint message validation."""

    def __init__(self, horizon, particles=512, validation_count=256,
                 components=32, jit_compile=True):
        if horizon < 2 or particles < 190 or components > particles:
            raise ValueError("SIR requires T>=2, full quadratic rank and enough centers")
        self.horizon = horizon
        self.particles = particles
        self.validation_count = validation_count
        self.components = components
        self.dimension = 18
        self.dtype = tf.float64
        self.spec = NonlinearSQMCSpec("sir_d18")
        self.theta = self.spec.default_theta(self.dtype)
        self.model, _ = self.spec.model(self.theta, tf.zeros_like(self.theta))
        identity = tf.eye(self.dimension, dtype=self.dtype)
        settings = dict(
            particles=particles, dimension=self.dimension, observation_dimension=10,
            initial_mean=self.spec.initial_mean(self.dtype), initial_covariance=identity,
            transition_covariance=identity, transition_mean=self.transition,
            observation_log_density=self.observation, solver="centered_qr",
            jit_compile=jit_compile, precision_policy="none",
        )
        self.engine = MultivariateGaussianControlledSMC(
            observations=horizon + 1, replicas=1, **settings)
        self.regression = MultivariateGaussianControlledSMC(
            observations=1, replicas=horizon - 1, **settings)
        self.export = tf.function(
            self._export, autograph=False, jit_compile=jit_compile,
            input_signature=[
                tf.TensorSpec([horizon + 1, 1, particles, self.dimension], self.dtype),
                tf.TensorSpec([horizon + 1, 1, particles], self.dtype),
                tf.TensorSpec([horizon + 1, 1, self.dimension, self.dimension], self.dtype),
                tf.TensorSpec([horizon + 1, 1, self.dimension], self.dtype),
                tf.TensorSpec([horizon + 1, 1], self.dtype),
                tf.TensorSpec([horizon - 1, particles + validation_count, self.dimension], self.dtype),
            ])

    def transition(self, states):
        flat = tf.reshape(states, [-1, self.dimension])
        return tf.reshape(self.model.transition_mean_fn(self.theta, flat), tf.shape(states))

    def observation(self, observed, states):
        residual = observed[..., :9] - states[..., 1::2]
        density = -.5 * tf.reduce_sum(residual ** 2 / 100, axis=-1)
        density -= tf.constant(4.5 * math.log(200 * math.pi), self.dtype)
        return observed[..., 9] * density

    def _export(self, clouds, logweights, policy_quadratic, policy_linear,
                policy_constant, design_noise):
        points = clouds[1:-1, 0]
        weights = tf.nn.softmax(logweights[1:-1, 0], axis=-1)
        center = tf.reduce_sum(weights[..., None] * points, axis=1)
        residual = points - center[:, None]
        covariance = tf.einsum("tn,tni,tnj->tij", weights, residual, residual)
        _, _, design_margin, design_valid = covariance_from_precision(covariance)
        identity = tf.eye(self.dimension, batch_shape=[self.horizon - 1], dtype=self.dtype)
        factor = tf.linalg.cholesky(tf.where(design_valid[:, None, None], covariance, identity))
        design = center[:, None] + tf.linalg.matmul(design_noise, factor, transpose_b=True)
        policy = {"A": policy_quadratic, "b": policy_linear, "c": policy_constant}
        integrated = next_policy_log_integral(
            self.transition, design, policy, self.engine.p, self.engine.ldq)
        log_message = integrated[2]
        training = design[:, :self.particles]
        validation = design[:, self.particles:]
        target = -log_message[:, :self.particles]
        fit = self.regression.regress(training, target)
        precision, linear, projected, raw_minimum = positive_information_projection(
            2 * fit["A"], -fit["b"], center)
        raw_prediction = -quadratic(validation, fit["A"], fit["b"], fit["c"])
        projected_prediction = -.5 * tf.einsum("tni,tij,tnj->tn", validation, precision, validation)
        projected_prediction += tf.einsum("tni,ti->tn", validation, linear)
        raw_at_center = -quadratic(center[:, None], fit["A"], fit["b"], fit["c"])[:, 0]
        projected_at_center = -.5 * tf.einsum("ti,tij,tj->t", center, precision, center)
        projected_at_center += tf.reduce_sum(center * linear, axis=-1)
        projected_prediction += (raw_at_center - projected_at_center)[:, None]
        heldout = log_message[:, self.particles:]
        quadratic_error = tf.reduce_mean((raw_prediction - heldout) ** 2, axis=-1)
        projected_error = tf.reduce_mean((projected_prediction - heldout) ** 2, axis=-1)
        offset = tf.reduce_max(log_message[:, :self.particles], axis=-1, keepdims=True)
        positive_target = tf.exp(log_message - offset)
        training_target = positive_target[:, :self.particles]
        validation_target = positive_target[:, self.particles:]
        constant_error = tf.reduce_mean(
            (validation_target - tf.reduce_mean(training_target, axis=-1, keepdims=True)) ** 2,
            axis=-1)
        centers = training[:, :self.components]
        normalized_centers = design_noise[:, :self.components]
        distances = tf.reduce_sum(
            (design_noise[:, :, None] - normalized_centers[:, None]) ** 2, axis=-1)
        bandwidths = tf.constant([.5, 1., 2.], self.dtype)
        bases = tf.exp(-.5 * distances[None] / bandwidths[:, None, None, None] ** 2)
        training_basis = bases[:, :, :self.particles]
        validation_basis = bases[:, :, self.particles:]
        gram = tf.linalg.matmul(training_basis, training_basis, transpose_a=True)
        response = tf.einsum("atnk,tn->atk", training_basis, training_target)
        coefficients = tf.ones([3, self.horizon - 1, self.components], self.dtype)
        def update(iteration, coefficients):
            denominator = tf.linalg.matvec(gram, coefficients)
            safe = tf.where(denominator > 0, denominator, tf.ones_like(denominator))
            return iteration + 1, coefficients * response / safe
        _, coefficients = tf.while_loop(
            lambda iteration, _: iteration < 200, update,
            (tf.constant(0), coefficients), parallel_iterations=1)
        prediction = tf.einsum("atnk,atk->atn", validation_basis, coefficients)
        kernel_error = tf.reduce_mean((prediction - validation_target[None]) ** 2, axis=-1)
        selection = tf.argmin(kernel_error, axis=0, output_type=tf.int32)
        selected_coefficients = tf.gather(tf.transpose(coefficients, [1, 0, 2]), selection, batch_dims=1)
        selected_error = tf.gather(tf.transpose(kernel_error), selection, batch_dims=1)
        selected_bandwidth = tf.gather(bandwidths, selection)
        selected_covariance = covariance * selected_bandwidth[:, None, None] ** 2
        active = selected_error < constant_error
        positive = tf.reduce_all(selected_coefficients > 0, axis=-1)
        selected_logits = tf.math.log(tf.where(selected_coefficients > 0, selected_coefficients,
                                               tf.ones_like(selected_coefficients)))
        selected_logits -= tf.reduce_logsumexp(selected_logits, axis=-1, keepdims=True)
        valid = tf.reduce_all(design_valid) & tf.reduce_all(integrated[4]) & fit["valid"]
        valid &= tf.reduce_all(positive) & tf.reduce_all(tf.math.is_finite(kernel_error))
        valid &= tf.reduce_all(tf.math.is_finite(projected_error))
        return dict(
            raw_precision=tf.concat([2 * fit["A"], tf.zeros([1, 18, 18], self.dtype)], axis=0),
            raw_linear=tf.concat([-fit["b"], tf.zeros([1, 18], self.dtype)], axis=0),
            precision=tf.concat([precision, tf.zeros([1, 18, 18], self.dtype)], axis=0),
            linear=tf.concat([linear, tf.zeros([1, 18], self.dtype)], axis=0),
            logits=tf.concat([selected_logits, tf.zeros([1, self.components], self.dtype)], axis=0),
            centers=tf.concat([centers, tf.zeros([1, self.components, 18], self.dtype)], axis=0),
            covariances=tf.concat([
                tf.repeat(selected_covariance[:, None], self.components, axis=1),
                tf.eye(18, batch_shape=[1, self.components], dtype=self.dtype)], axis=0),
            active=tf.concat([active, [False]], axis=0), valid=valid,
            design_margin=design_margin, integrated_margin=integrated[3],
            projected=projected, raw_minimum=raw_minimum, quadratic_error=quadratic_error,
            projected_error=projected_error, kernel_error=kernel_error,
            constant_error=constant_error, selected_bandwidth=selected_bandwidth,
            selected_error=selected_error, regression_valid=fit["valid"],
            qr_condition=fit["qr_reciprocal_condition_lower_bound"],
        )

    def fit(self, observations, seed, record=None):
        observed = tf.concat([
            tf.zeros([1, 10], self.dtype),
            tf.concat([observations, tf.ones([self.horizon, 1], self.dtype)], axis=-1),
        ], axis=0)[:, None]
        zeros = (tf.zeros([self.horizon + 1, 1, 18, 18], self.dtype),
                 tf.zeros([self.horizon + 1, 1, 18], self.dtype),
                 tf.zeros([self.horizon + 1, 1], self.dtype))
        shape = [self.horizon + 1, 1, self.particles, 18]
        initial = self.engine.forward(
            *zeros, observed, tf.random.stateless_normal(shape, [seed, 1], dtype=self.dtype),
            tf.random.stateless_uniform([self.horizon + 1, 1], [seed, 2], dtype=self.dtype))
        if record is not None:
            record("initial", {key: initial[key] for key in ("valid", "ess", "logZ", "precision_margin")})
        if not bool(initial["valid"]):
            raise ValueError("Invalid initial SIR cSMC pilot")
        policy = self.engine.fit(*zeros, initial["x"], initial["log_incremental_weight"])
        if record is not None:
            record("policy", policy)
        if not bool(policy["valid"]):
            raise ValueError("Invalid SIR cSMC fit: precision or regression failed")
        pilot = self.engine.forward(
            *(policy[key] for key in ("A", "b", "c")), observed,
            tf.random.stateless_normal(shape, [seed, 3], dtype=self.dtype),
            tf.random.stateless_uniform([self.horizon + 1, 1], [seed, 4], dtype=self.dtype))
        if record is not None:
            record("pilot", {key: pilot[key] for key in ("valid", "ess", "logZ", "precision_margin")})
        if not bool(pilot["valid"]):
            raise ValueError("Invalid independent controlled SIR pilot")
        exported = self.export(
            pilot["x"], pilot["log_incremental_weight"],
            *(policy[key] for key in ("A", "b", "c")),
            tf.random.stateless_normal(
                [self.horizon - 1, self.particles + self.validation_count, 18],
                [seed, 5], dtype=self.dtype))
        if record is not None:
            record("export", exported)
        if not bool(exported["valid"]):
            raise ValueError("Invalid SIR message export")
        quadratic_guide = FrozenQuadraticBackwardGuide(exported["precision"], exported["linear"])
        kernel_guide = FrozenMixtureBackwardGuide(
            exported["logits"], exported["centers"], exported["covariances"], exported["active"])
        diagnostics = dict(exported, policy=policy, initial_ess=initial["ess"],
                           controlled_ess=pilot["ess"], initial_logZ=initial["logZ"],
                           controlled_logZ=pilot["logZ"], fit_seed=seed)
        return quadratic_guide, kernel_guide, diagnostics
