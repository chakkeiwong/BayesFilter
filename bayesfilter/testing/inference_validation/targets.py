"""TensorFlow target laws for validation, separate from reference computation.

Internal coordinates are unconstrained. Model quantities use explicit exp,
sigmoid or centered-softmax maps with the corresponding density Jacobian.
The batch value/score graph has one stable input signature per target instance.
"""
from __future__ import annotations

import math
from typing import Any
import tensorflow as tf

from .catalog import get_target
from .designs import digest


class ValidationTarget:
    def __init__(self, target_id: str, parameters: dict[str, Any] | None = None,
                 data: list | None = None, control: str = "baseline", *, jit_compile=True):
        self.spec = get_target(target_id)
        if not self.spec.available:
            raise ValueError(self.spec.unavailable_reason)
        self.target_id = target_id
        self.parameters = dict(parameters or {})
        self.data = list(data) if data is not None else None
        self.control = control
        self.parameter_dim = self.spec.dimension
        if control == "ignore_data" and not self.spec.generative:
            raise ValueError("ignored-data mutation requires a generative target")
        if control == "omit_jacobian" and self.spec.support not in {"positive", "unit_interval", "simplex3"}:
            raise ValueError("Jacobian mutation must change a nonconstant Jacobian")
        for key in ("scale", "condition", "tau", "sigma", "alpha", "beta", "df", "rate", "state_variance"):
            if key in self.parameters and (not math.isfinite(self.parameters[key]) or self.parameters[key] <= 0):
                raise ValueError(f"positive finite {key} required")
        if "rho" in self.parameters and not abs(self.parameters["rho"]) < 1:
            raise ValueError("stationary LGSSM requires |rho| < 1")
        if "n" in self.parameters and (type(self.parameters["n"]) is not int or self.parameters["n"] < 1):
            raise ValueError("n must be a positive integer")
        if "concentration" in self.parameters:
            alpha = self.parameters["concentration"]
            if len(alpha) != 3 or any(not math.isfinite(v) or v <= 0 for v in alpha):
                raise ValueError("three positive finite Dirichlet concentrations required")
        if self.data is not None:
            if not self.spec.generative or not self.data or any(not math.isfinite(v) for v in self.data):
                raise ValueError("finite nonempty data required for a generative target")
            if target_id == "beta_binomial" and (len(self.data) != 2 or
                    any(type(v) is not int for v in self.data) or not 0 <= self.data[0] <= self.data[1]):
                raise ValueError("binomial data must contain integer successes and total")
        if target_id == "student_t" and self.parameters.get("df", 5.) <= 2:
            raise ValueError("student_t fixture declares finite variance; use cauchy for infinite moments")
        if target_id == "mixture" and not 0 < self.parameters.get("weight", .3) < 1:
            raise ValueError("mixture weight must be strictly inside (0,1)")
        self._batch = tf.function(self._batch_score,
            input_signature=[tf.TensorSpec([None, self.parameter_dim], tf.float64)],
            autograph=False, jit_compile=jit_compile)

    def adapter_signature(self):
        return digest({"law": "inference_validation_target.v1", "target": self.target_id,
                       "parameters": self.parameters, "data": self.data, "control": self.control})

    def value_score_capability(self):
        from bayesfilter.inference.posterior_adapter import ValueScoreCapability
        return ValueScoreCapability(value_score_authority="graph_native", xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True, target_scope="inference_validation",
            runtime_backend="tensorflow", evidence_path=__file__,
            nonclaims=("validation fixture only; exact law checked by independent reference",))

    def parameter_names(self):
        return self.spec.parameters if len(self.spec.parameters) == self.parameter_dim else tuple(
            f"z{i}" for i in range(self.parameter_dim))

    def to_model(self, q):
        if self.target_id == "funnel_noncentered":
            return tf.concat([q[..., :1], tf.exp(q[..., :1] / 2) * q[..., 1:]], -1)
        if self.spec.support == "positive":
            return tf.exp(q)
        if self.spec.support == "unit_interval":
            return tf.math.sigmoid(q)
        if self.spec.support == "simplex3":
            return tf.nn.softmax(tf.concat([q, tf.zeros_like(q[..., :1])], -1), -1)
        return q

    def jacobian(self, q):
        if self.spec.support == "positive":
            return tf.reduce_sum(q, -1)
        if self.spec.support == "unit_interval":
            return tf.reduce_sum(-tf.nn.softplus(-q) - tf.nn.softplus(q), -1)
        if self.spec.support == "simplex3":
            return tf.reduce_sum(tf.nn.log_softmax(tf.concat([q, tf.zeros_like(q[..., :1])], -1), -1), -1)
        return tf.zeros(tf.shape(q)[:-1], tf.float64)

    def _number(self, name, default):
        return tf.constant(self.parameters.get(name, default), tf.float64)

    @staticmethod
    def _normal(x, scale):
        scale = tf.cast(scale, tf.float64)
        return -.5 * tf.square(x / scale) - tf.math.log(scale) - .5 * math.log(2 * math.pi)

    def log_density(self, q):
        kind = self.target_id
        if kind in {"gaussian", "rotated_gaussian"}:
            if kind == "rotated_gaussian":
                angle = self._number("angle", .6)
                x = tf.cos(angle) * q[..., 0] + tf.sin(angle) * q[..., 1]
                y = -tf.sin(angle) * q[..., 0] + tf.cos(angle) * q[..., 1]
                return self._normal(x, 1.) + self._normal(y, tf.sqrt(self._number("condition", 9.)))
            return tf.reduce_sum(self._normal(q, self._number("scale", 1.)), -1)
        if kind == "banana":
            x = q[..., 0]
            y = q[..., 1] - self._number("bend", .5) * (x*x - 1.)
            return self._normal(x, 1.) + self._normal(y, 1.)
        if kind == "funnel":
            return self._normal(q[..., 0], self._number("scale", 3.)) + tf.reduce_sum(
                self._normal(q[..., 1:], tf.exp(q[..., :1] / 2)), -1)
        if kind == "funnel_noncentered":
            # x_i = exp(v/2) z_i for two children: log |J| = v cancels
            # the two centered conditional scale terms exactly.
            return self._normal(q[..., 0], self._number("scale", 3.)) + tf.reduce_sum(
                self._normal(q[..., 1:], 1.), -1)
        if kind in {"student_t", "cauchy"}:
            df = self._number("df", 5.) if kind == "student_t" else tf.constant(1., tf.float64)
            const = tf.math.lgamma((df+1)/2) - tf.math.lgamma(df/2) - .5*tf.math.log(df*math.pi)
            return tf.reduce_sum(const - (df+1)/2 * tf.math.log1p(q*q/df), -1)
        if kind == "mixture":
            a, w = self._number("separation", 5.), self._number("weight", .3)
            left = self._normal(q[..., 0]+a, 1.) + tf.math.log(w)
            right = self._normal(q[..., 0]-a, 1.) + tf.math.log1p(-w)
            return tf.reduce_logsumexp(tf.stack([left, right], -1), -1) + self._normal(q[..., 1], 1.)
        if kind in {"gamma", "beta", "beta_binomial", "dirichlet"}:
            if kind == "gamma":
                a, rate = self._number("alpha", 2.), self._number("rate", 1.)
                value = tf.reduce_sum(a*tf.math.log(rate)-tf.math.lgamma(a)+(a-1)*q-rate*tf.exp(q), -1)
            elif kind in {"beta", "beta_binomial"}:
                a, b = self._number("alpha", 2.), self._number("beta", 3.)
                if kind == "beta_binomial" and self.control != "ignore_data" and self.data is not None:
                    a += self.data[0]
                    b += self.data[1]-self.data[0]
                value = tf.reduce_sum((a-1)*(-tf.nn.softplus(-q))+(b-1)*(-tf.nn.softplus(q)), -1)
                value -= tf.math.lgamma(a)+tf.math.lgamma(b)-tf.math.lgamma(a+b)
            else:
                alpha = tf.constant(self.parameters.get("concentration", [2.,3.,4.]), tf.float64)
                logp = tf.nn.log_softmax(tf.concat([q, tf.zeros_like(q[..., :1])], -1), -1)
                value = tf.reduce_sum((alpha-1)*logp, -1) + tf.math.lgamma(tf.reduce_sum(alpha))-tf.reduce_sum(tf.math.lgamma(alpha))
            return value if self.control == "omit_jacobian" else value + self.jacobian(q)
        if kind in {"normal_conjugate", "lgssm_location"}:
            value = self._normal(q[..., 0], self._number("tau", 2.))
            if self.control == "ignore_data" or self.data is None:
                return value
            residual = tf.constant(self.data, tf.float64) - q[..., :1]
            if kind == "normal_conjugate":
                return value + tf.reduce_sum(self._normal(residual, self._number("sigma", 1.)), -1)
            # Marginal covariance of a stationary scalar AR(1) latent state plus
            # independent measurement noise; its dense Gaussian law is the target.
            n = len(self.data)
            rho, state_var, noise = self._number("rho", .6), self._number("state_variance", 1.), self._number("sigma", .5)
            lag = tf.abs(tf.range(n)[:,None]-tf.range(n)[None,:])
            cov = state_var*tf.pow(rho, tf.cast(lag, tf.float64)) + tf.eye(n, dtype=tf.float64)*noise**2
            chol = tf.linalg.cholesky(cov)
            solved = tf.linalg.triangular_solve(chol, tf.transpose(residual))
            return value - .5*tf.reduce_sum(solved**2, 0) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol))) - n*.5*math.log(2*math.pi)
        raise ValueError(kind)

    def _batch_score(self, q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            value = self.log_density(q)
        score = tape.gradient(value, q)
        if self.control == "wrong_score":
            score = score + tf.constant(.25, tf.float64)
        return value, score

    def log_prob_and_grad(self, position):
        q = tf.convert_to_tensor(position, tf.float64)
        vector = q.shape.rank == 1
        value, score = self._batch(q[None] if vector else q)
        return (value[0], score[0]) if vector else (value, score)

    def target_status_telemetry(self, position):
        q = tf.convert_to_tensor(position, tf.float64)
        finite = tf.reduce_all(tf.math.is_finite(q), -1)
        return {"status_code": tf.where(finite, 0, 1), "valid_pre_regularized_score": finite,
                "floor_count_value": tf.zeros(tf.shape(q)[:-1], tf.int32)}
