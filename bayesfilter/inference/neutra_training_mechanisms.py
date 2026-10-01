"""Historical q20 training mechanisms; no HMC admission or production defaults.

The affine/scalar correction maps are HISTORICAL — UNFAITHFUL TO THE AUTHOR'S
CODE as full NeuTra architectures (owner directive 2026-09-25). They are
diagnostic mechanisms, not the canonical IAF or full conditional NAF. See
docs/reference/neutra-implementation.md and the canonical-policy notice.

The plan is docs/plans/bayesfilter-q20-three-mechanisms-plan-2026-09-23.md.
Maps operate on independent leading-batch rows. The compiled trainer owns the
stable shape/XLA boundary. Direct map calls are also useful for reference tests.
The generic proposal-score engine uses dimension-wise VJPs, never pfor or a
sample-wise target loop. It is a correctness implementation for small dimension,
not the optimized coupling-flow algorithm in Vaitl et al. (2024).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import tensorflow as tf

from bayesfilter.inference import neutra_transport_core as _transport_core


DTYPE = tf.float64


def _vector(values: Sequence[float], name: str) -> tf.Tensor:
    values = tuple(float(v) for v in values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return tf.constant(values, DTYPE)


def _valid_log_scale(values):
    scale = tf.exp(values)
    reciprocal = tf.exp(-values)
    return tf.reduce_all(tf.math.is_finite(scale) & (scale > 0)
                         & tf.math.is_finite(reciprocal) & (reciprocal > 0))


def _valid_weight_logits(values):
    weights = tf.nn.softmax(values)
    return tf.reduce_all(tf.math.is_finite(weights) & (weights > 0))


class FreeDiagonalAffine:
    """Free global scale, separate from any conditional scale cap in a base map."""

    def __init__(self, shift: Sequence[float], log_scale: Sequence[float]):
        shift_tensor = _vector(shift, "shift")
        scale_tensor = _vector(log_scale, "log_scale")
        if shift_tensor.shape != scale_tensor.shape:
            raise ValueError("shift and log_scale must have equal shape")
        self.parameter_dim = int(shift_tensor.shape[0])
        self.shift = tf.Variable(shift_tensor, name="mechanism_affine_shift")
        self.log_scale = tf.Variable(scale_tensor, name="mechanism_affine_log_scale")
        self.trainable_variables = (self.shift, self.log_scale)
        self.parameter_constraints = {id(self.log_scale): _valid_log_scale}

    def forward_and_logdet(self, latent):
        values = tf.ensure_shape(tf.convert_to_tensor(latent, DTYPE), [None, self.parameter_dim])
        scale = tf.where(_valid_log_scale(self.log_scale), tf.exp(self.log_scale), tf.constant(float("nan"), DTYPE))
        return _transport_core.affine_forward(values, self.shift, scale, logdet=tf.reduce_sum(self.log_scale))

    def inverse_and_forward_logdet(self, output):
        values = tf.ensure_shape(tf.convert_to_tensor(output, DTYPE), [None, self.parameter_dim])
        scale = tf.where(_valid_log_scale(self.log_scale), tf.exp(self.log_scale), tf.constant(float("nan"), DTYPE))
        return _transport_core.affine_inverse(values, self.shift, scale, logdet=tf.reduce_sum(self.log_scale))

    def parameter_state(self):
        return {"kind": "free_diagonal_affine", "hmc_admitted": False,
                "shift": self.shift.numpy().tolist(),
                "log_scale": self.log_scale.numpy().tolist()}


class ScalarSigmoidMixture:
    """A full-range DSF in one zero-based coordinate; other coordinates pass through.

    All initialization and inverse controls are explicit. Inverse failure emits
    nonfinite rows as well as a validity flag; it never returns an apparently
    valid approximation at the iteration cap. Inverse gradients are deliberately
    not provided: this optional component currently supports forward RKL only.
    """

    def __init__(self, *, dimension: int, coordinate: int,
                 log_slopes: Sequence[float], offsets: Sequence[float],
                 weight_logits: Sequence[float], inverse_atol: float,
                 inverse_rtol: float, inverse_max_iterations: int):
        if (isinstance(dimension, bool) or not isinstance(dimension, int)
                or dimension < 1 or isinstance(coordinate, bool)
                or not isinstance(coordinate, int) or not 0 <= coordinate < dimension):
            raise ValueError("dimension/coordinate must be valid integers")
        if (not math.isfinite(inverse_atol) or inverse_atol <= 0
                or not math.isfinite(inverse_rtol) or inverse_rtol < 0
                or isinstance(inverse_max_iterations, bool)
                or not isinstance(inverse_max_iterations, int)
                or inverse_max_iterations < 1):
            raise ValueError("invalid explicit inverse controls")
        slopes = _vector(log_slopes, "log_slopes")
        biases = _vector(offsets, "offsets")
        logits = _vector(weight_logits, "weight_logits")
        if not slopes.shape == biases.shape == logits.shape:
            raise ValueError("mixture parameter vectors must have equal shape")
        self.parameter_dim = dimension
        self.coordinate = coordinate
        self.inverse_atol = float(inverse_atol)
        self.inverse_rtol = float(inverse_rtol)
        self.inverse_max_iterations = inverse_max_iterations
        self.log_slopes = tf.Variable(slopes, name="mechanism_dsf_log_slopes")
        self.offsets = tf.Variable(biases, name="mechanism_dsf_offsets")
        self.weight_logits = tf.Variable(logits, name="mechanism_dsf_weight_logits")
        self.trainable_variables = (self.log_slopes, self.offsets, self.weight_logits)
        self.parameter_constraints = {id(self.log_slopes): _valid_log_scale,
                                      id(self.weight_logits): _valid_weight_logits}

    def scalar_value_logdet_score(self, x):
        return _transport_core.sigmoid_mixture(tf.convert_to_tensor(x, DTYPE),
            self.log_slopes, self.offsets, self.weight_logits)

    def _replace(self, values, scalar):
        mask = tf.one_hot(self.coordinate, self.parameter_dim, on_value=True,
                          off_value=False, dtype=tf.bool)
        return tf.where(mask, scalar[..., tf.newaxis], values)

    def forward_and_logdet(self, latent):
        values = tf.ensure_shape(tf.convert_to_tensor(latent, DTYPE), [None, self.parameter_dim])
        scalar, logdet, _ = self.scalar_value_logdet_score(values[:, self.coordinate])
        return self._replace(values, scalar), logdet

    def log_abs_det_jacobian_score_batch(self, latent):
        values = tf.ensure_shape(tf.convert_to_tensor(latent, DTYPE), [None, self.parameter_dim])
        _, _, score = self.scalar_value_logdet_score(values[:, self.coordinate])
        return self._replace(tf.zeros_like(values), score)

    def inverse_with_status(self, output):
        values = tf.ensure_shape(tf.convert_to_tensor(output, DTYPE), [None, self.parameter_dim])
        solution, logdet, valid, iterations = _transport_core.sigmoid_inverse(
            values[:, self.coordinate], self.log_slopes, self.offsets, self.weight_logits,
            atol=self.inverse_atol, rtol=self.inverse_rtol, max_iterations=self.inverse_max_iterations)
        valid &= tf.reduce_all(tf.math.is_finite(values), axis=-1)
        nan = tf.constant(float("nan"), DTYPE)
        recovered = self._replace(values, solution)
        return (tf.stop_gradient(tf.where(valid[:, None], recovered, nan)),
                tf.stop_gradient(tf.where(valid, logdet, nan)), valid, iterations)

    def inverse_and_forward_logdet(self, output):
        latent, logdet, _, _ = self.inverse_with_status(output)
        return latent, logdet

    def parameter_state(self):
        return {"kind": "scalar_sigmoid_mixture", "hmc_admitted": False,
                "dimension": self.parameter_dim, "coordinate": self.coordinate,
                "log_slopes": self.log_slopes.numpy().tolist(),
                "offsets": self.offsets.numpy().tolist(),
                "weight_logits": self.weight_logits.numpy().tolist(),
                "inverse_atol": self.inverse_atol, "inverse_rtol": self.inverse_rtol,
                "inverse_max_iterations": self.inverse_max_iterations}


def mechanism_from_state(state):
    """Reconstruct optional parameter state, never issue a frozen HMC payload."""
    if state.get("hmc_admitted") is not False:
        raise ValueError("mechanism state must be explicitly non-admitted")
    if state.get("kind") == "free_diagonal_affine":
        return FreeDiagonalAffine(state["shift"], state["log_scale"])
    if state.get("kind") == "scalar_sigmoid_mixture":
        return ScalarSigmoidMixture(**{key: state[key] for key in (
            "dimension", "coordinate", "log_slopes", "offsets", "weight_logits",
            "inverse_atol", "inverse_rtol", "inverse_max_iterations")})
    raise ValueError("unknown mechanism state")


class ComposedMechanism:
    """second(first(z)); freezing controls optimizer variables, not input derivatives."""

    def __init__(self, first, second, *, train_first: bool, train_second: bool):
        if first.parameter_dim != second.parameter_dim:
            raise ValueError("composition dimensions differ")
        self.first, self.second = first, second
        self.parameter_dim = first.parameter_dim
        self.trainable_variables = tuple(
            (first.trainable_variables if train_first else ())
            + (second.trainable_variables if train_second else ()))
        if len({id(v) for v in self.trainable_variables}) != len(self.trainable_variables):
            raise ValueError("composition repeats a trainable variable")
        self.parameter_constraints = {
            **getattr(first, "parameter_constraints", {}),
            **getattr(second, "parameter_constraints", {}),
        }

    def forward_and_logdet(self, latent):
        return _transport_core.compose_forward((self.first, self.second), latent)

    def inverse_and_forward_logdet(self, output):
        middle, ld2 = self.second.inverse_and_forward_logdet(output)
        latent, ld1 = self.first.inverse_and_forward_logdet(middle)
        return latent, ld1 + ld2


def _forward_proposal_score(transport, latent):
    return _transport_core.forward_proposal_score(transport, latent)


class MechanismReverseKLTrainer:
    """Batched standard/path RKL with supplied first target score and fail-closed updates.

    target_value_score returns (values[B], score[B,D], valid[B]). Its value is
    treated as opaque; no target Hessian or autodiff through target code is used.
    The actual loss is always returned, separately from the gradient carrier.
    """

    def __init__(self, transport, target_value_score: Callable, *, batch_size: int,
                 estimator: str, learning_rate: float, jit_compile: bool = True):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("training requires an explicit batch size greater than one")
        if estimator not in ("standard", "path"):
            raise ValueError("estimator must be standard or path")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.transport = transport
        self.target_value_score = target_value_score
        self.estimator = estimator
        self.variables = tuple(transport.trainable_variables)
        if not self.variables:
            raise ValueError("transport has no trainable variables")
        # SGD isolates the derivative mechanics; q20 Adam continuation is a separate integration.
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.optimizer.build(self.variables)
        signature = (tf.TensorSpec([batch_size, transport.parameter_dim], DTYPE),)
        self.evaluate = tf.function(self._evaluate, input_signature=signature,
                                    jit_compile=jit_compile, autograph=False)
        self.train_step = tf.function(self._train_step, input_signature=signature,
                                      jit_compile=jit_compile, autograph=False)

    def _evaluate(self, latent):
        return _transport_core.reverse_kl_evaluate(self.transport, self.target_value_score, latent,
            estimator=self.estimator, variables=self.variables)

    def _train_step(self, latent):
        result = self._evaluate(latent)
        # Check the explicit SGD proposal before mutating parameters, including overflow.
        rate = tf.cast(self.optimizer.learning_rate, DTYPE)
        constraints = getattr(self.transport, "parameter_constraints", {})
        checks = []
        for v, g in zip(self.variables, result["gradients"]):
            proposed = v - rate * g
            checks.append(tf.reduce_all(tf.math.is_finite(proposed)))
            if id(v) in constraints:
                checks.append(constraints[id(v)](proposed))
        valid = result["valid"] & tf.reduce_all(tf.stack(checks))

        def update():
            self.optimizer.apply_gradients(zip(result["gradients"], self.variables))
            return tf.identity(self.optimizer.iterations)

        iteration = tf.cond(valid, update, lambda: tf.identity(self.optimizer.iterations))
        return {**result, "valid": valid, "iteration": iteration}


class AdamMechanismCanary(MechanismReverseKLTrainer):
    """Fresh-slot Adam for bounded correction fits, not a production checkpoint codec.

    Invalid numerical proposals restore parameters AND all optimizer state.
    All optimizer controls are explicit hypotheses of the caller's canary plan.
    """

    def __init__(self, transport, target_value_score, *, batch_size, estimator,
                 learning_rate, beta1, beta2, epsilon, gradient_clip_norm,
                 jit_compile=True):
        if (not 0 <= beta1 < 1 or not 0 <= beta2 < 1
                or not math.isfinite(epsilon) or epsilon <= 0
                or not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0):
            raise ValueError("invalid explicit Adam canary controls")
        super().__init__(transport, target_value_score, batch_size=batch_size,
                         estimator=estimator, learning_rate=learning_rate,
                         jit_compile=jit_compile)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
        self.optimizer.build(self.variables)
        self.gradient_clip_norm = tf.Variable(gradient_clip_norm, dtype=DTYPE,
                                             trainable=False)

    def _train_step(self, latent):
        result = self._evaluate(latent)
        norm = tf.linalg.global_norm(result["gradients"])
        gradients, _ = tf.clip_by_global_norm(result["gradients"],
                                             self.gradient_clip_norm, use_norm=norm)
        state_variables = self.variables + tuple(self.optimizer.variables)
        old = tuple(tf.identity(v) for v in state_variables)
        constraints = getattr(self.transport, "parameter_constraints", {})

        def update():
            with tf.control_dependencies(old):
                self.optimizer.apply_gradients(zip(gradients, self.variables))
            checks = [tf.reduce_all(tf.math.is_finite(tf.cast(v, DTYPE)))
                      for v in state_variables]
            checks.extend(constraints[id(v)](v) for v in self.variables if id(v) in constraints)
            accepted = tf.reduce_all(tf.stack(checks))
            movement = tf.linalg.global_norm(tuple(
                v - previous for v, previous in zip(self.variables, old)))

            def rollback():
                for variable, previous in zip(state_variables, old):
                    variable.assign(previous)
                return tf.constant(False), tf.constant(0., DTYPE)

            return tf.cond(accepted, lambda: (accepted, movement), rollback)

        valid, movement = tf.cond(
            result["valid"] & tf.math.is_finite(norm), update,
            lambda: (tf.constant(False), tf.constant(0., DTYPE)))
        return {**result, "valid": valid, "iteration": tf.identity(self.optimizer.iterations),
                "gradient_norm": norm, "clipped": norm > self.gradient_clip_norm,
                "parameter_update_norm": movement}
