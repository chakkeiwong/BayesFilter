"""Dimension-general four-coordinate SSL-LSTM posterior targets.

The state-complexity experiment estimates a fixed four-coordinate block while
holding the remaining chart coordinates at a hashed synthetic fixture.  The
target deliberately requests selected analytic derivative directions so the
filter score does not materialize the full chart derivative tensor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    ssl_lstm_parameter_slices,
    ssl_lstm_transition,
    tf_ssl_lstm_svd_ukf_score,
    unpack_ssl_lstm_parameters,
)


FREE_NAMES = (
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
)
PRIOR_CENTER = tf.constant((0.35, -0.08, 0.65, 0.05), tf.float64)
PRIOR_STANDARD_DEVIATION = 4.0
HORIZON = 30
Q1_FULL_FIXTURE = tf.constant(
    (
        0.09, -0.07, 0.05, 0.04, 0.03, -0.02, 0.06, -0.05,
        0.01, 0.04, -0.03, 0.02, 0.35, -0.08, 0.65, 0.05,
        0.15, -0.10, 0.20, -0.35, 0.15, 0.55, 0.35, -0.15,
    ), tf.float64
)
Q1_OBSERVATIONS = tf.reshape(tf.constant(
    (
        0.348783333509205, -0.06319427221788393, 0.938603323083808,
        1.4622688902045144, -0.44815739683239364, 0.22003438506565143,
        0.12802423285807635, 0.09088861589914976, -0.30892992513107187,
        1.2888202099980806, -0.5062346637379318, 0.23141030375951993,
        -0.7398852277577778, -1.637711895122823, 0.7463924366306034,
        -0.015159995809434501, 0.821621152232911, 0.943395801287454,
        0.7983413928708691, -0.44994456871443267, 0.5986856559902419,
        0.9655453011734912, 0.27912167846629843, -1.0212217577883904,
        -0.7212056110030904, 1.675807439349596, -1.0454402378094254,
        -0.5329910449431029, -1.6360645459528094, -0.6635502479829377,
    ), tf.float64), [HORIZON, 1])


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _sha(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _free_indices(config: SSLLSTMStaticConfig) -> tuple[int, ...]:
    slices = ssl_lstm_parameter_slices(config)
    return (
        slices.latent_weight_start,
        slices.latent_bias_start,
        slices.observation_weight_start,
        slices.observation_bias_start,
    )


def make_complexity_config(q: int) -> SSLLSTMStaticConfig:
    return SSLLSTMStaticConfig(
        horizon=HORIZON,
        latent_dim=int(q),
        hidden_dim=int(q),
        observation_dim=1,
    )


def make_full_fixture(config: SSLLSTMStaticConfig) -> tf.Tensor:
    """Create a deterministic finite fixture with the same free center."""

    if config.latent_dim == 1 and config.hidden_dim == 1:
        return tf.identity(Q1_FULL_FIXTURE)
    p = int(config.parameter_dim)
    theta = 0.025 * tf.sin(tf.cast(tf.range(p), tf.float64) * tf.constant(0.371, tf.float64))
    slices = ssl_lstm_parameter_slices(config)
    n = int(config.augmented_state_dim)
    k = int(config.latent_dim)
    updates: list[tuple[int, float]] = []
    for index in range(n):
        updates.append((slices.initial_std_start + index, -0.85 + 0.011 * index))
    for index in range(k):
        updates.append((slices.process_std_start + index, 0.55 + 0.017 * index))
    updates.append((slices.observation_std_start, -0.2))
    # The four estimated coordinates have the same prior-center truth at every q.
    updates.extend((index, float(value)) for index, value in zip(_free_indices(config), PRIOR_CENTER))
    indices = tf.constant([[index] for index, _ in updates], tf.int32)
    values = tf.constant([value for _, value in updates], tf.float64)
    return tf.tensor_scatter_nd_update(theta, indices, values)


def make_synthetic_observations(config: SSLLSTMStaticConfig, fixture: tf.Tensor) -> tf.Tensor:
    """Generate one fixed synthetic observation path for q>1."""

    if config.latent_dim == 1 and config.hidden_dim == 1:
        return tf.identity(Q1_OBSERVATIONS)
    params = unpack_ssl_lstm_parameters(fixture, config)
    state = tf.identity(params.initial_mean)
    rows: list[tf.Tensor] = []
    process_seed = tf.constant((20260719, 1000 + config.latent_dim), tf.int32)
    obs_seed = tf.constant((20260719, 2000 + config.latent_dim), tf.int32)
    for step in range(HORIZON):
        noise = tf.random.stateless_normal(
            [config.latent_dim],
            tf.random.experimental.stateless_fold_in(process_seed, step),
            dtype=tf.float64,
        ) * params.process_std
        obs_noise = tf.random.stateless_normal([1], tf.random.experimental.stateless_fold_in(obs_seed, step), dtype=tf.float64)
        deterministic = ssl_lstm_transition(params, state[tf.newaxis, :])[0]
        state = tf.concat(
            (
                deterministic[: config.latent_dim] + noise,
                deterministic[config.latent_dim :],
            ),
            axis=0,
        )
        observation = tf.reshape(
            tf.linalg.matvec(params.observation_weight, state[: config.latent_dim])
            + params.observation_bias,
            [1],
        ) + params.observation_std * obs_noise
        rows.append(observation)
    return tf.stack(rows, axis=0)


@dataclass(frozen=True)
class ComplexityTargetConfig:
    static_config: SSLLSTMStaticConfig
    fixture: tf.Tensor
    observations: tf.Tensor
    free_indices: tuple[int, ...]
    prior_center: tf.Tensor = PRIOR_CENTER
    prior_standard_deviation: float = PRIOR_STANDARD_DEVIATION

    def signature_payload(self) -> dict[str, Any]:
        return {
            "schema": "bayesfilter.ssl_lstm.complexity_target.v1",
            "horizon": self.static_config.horizon,
            "latent_dim": self.static_config.latent_dim,
            "hidden_dim": self.static_config.hidden_dim,
            "observation_dim": self.static_config.observation_dim,
            "parameter_dim": self.static_config.parameter_dim,
            "free_indices": list(self.free_indices),
            "free_names": list(FREE_NAMES),
            "fixture": [float(v) for v in tf.reshape(self.fixture, [-1]).numpy()],
            "observations": [float(v) for v in tf.reshape(self.observations, [-1]).numpy()],
            "prior_center": [float(v) for v in self.prior_center.numpy()],
            "prior_standard_deviation": float(self.prior_standard_deviation),
            "parameter_transform": {
                "orientation": "identity",
                "inverse_orientation": "identity",
            },
        }


class SSLLSTMComplexityPosteriorTarget:
    """Graph-native four-coordinate value/score target for one q rung."""

    def __init__(self, q: int, *, jit_compile: bool = True) -> None:
        config = make_complexity_config(q)
        fixture = make_full_fixture(config)
        observations = make_synthetic_observations(config, fixture)
        free_indices = _free_indices(config)
        self.config = ComplexityTargetConfig(config, fixture, observations, free_indices)
        self.q = int(q)
        self._jit_compile = bool(jit_compile)
        self._signature = _sha(self.config.signature_payload())
        self._adapter_signature = _sha({"target_signature": self._signature, "free_indices": list(free_indices), "directional_score": True})
        self._compiled = tf.function(
            self._value_score_impl,
            input_signature=[tf.TensorSpec([4], tf.float64)],
            jit_compile=self._jit_compile,
            reduce_retracing=True,
        )
        self._compiled_batches: dict[int, Any] = {}

    @property
    def parameter_dim(self) -> int:
        return 4

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return FREE_NAMES

    @property
    def target_scope(self) -> str:
        return f"ssl_lstm_neutra_state_complexity:q{self.q}"

    def target_signature(self) -> str:
        return self._signature

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=bool(self._jit_compile),
            runtime_backend="bayesfilter.nonlinear.ssl_lstm_complexity_target_tf",
            target_scope=self.target_scope,
            evidence_path="docs/plans/bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md",
            nonclaims=("controlled synthetic target only", "no posterior oracle", "no model adequacy claim"),
        )

    def full_theta(self, free: tf.Tensor) -> tf.Tensor:
        return tf.tensor_scatter_nd_update(self.config.fixture, tf.constant([[i] for i in self.config.free_indices], tf.int32), free)

    def _value_score_impl(self, free: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        full = self.full_theta(free)
        score_result, _components = tf_ssl_lstm_svd_ukf_score(
            self.config.observations,
            full,
            self.config.static_config,
            evidence_path="docs/plans/bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md",
            derivative_parameter_indices=self.config.free_indices,
            spectral_gap_tolerance=tf.constant(1.0e-10, tf.float64),
        )
        delta = free - self.config.prior_center
        variance = tf.constant(self.config.prior_standard_deviation ** 2, tf.float64)
        value = score_result.log_likelihood - 0.5 * tf.reduce_sum(tf.square(delta) / variance)
        score = score_result.score - delta / variance
        return tf.ensure_shape(value, []), tf.ensure_shape(score, [4])

    def _batch_value_score_impl(self, free: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values, scores = tf.map_fn(self._value_score_impl, free, fn_output_signature=(tf.float64, tf.TensorSpec([4], tf.float64)))
        return values, scores

    def value_and_score(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self._compiled(tf.convert_to_tensor(free, tf.float64))

    def eager_value_and_score(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        """Evaluate one target point without XLA or a target batch loop.

        This entry point is for the persistent CPU worker route.  It keeps the
        scientific target equations identical to ``value_and_score`` while
        making the worker boundary explicit: one process evaluates one scalar
        point at a time and returns its value and analytic score to the parent.
        """

        values = tf.convert_to_tensor(free, tf.float64)
        if values.shape.rank != 1 or values.shape[-1] != 4:
            raise ValueError("eager target point must have shape [4]")
        return self._value_score_impl(values)

    def log_prob_and_grad(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.value_and_score(free)

    def batch_value_and_score(self, free: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(free, tf.float64)
        if values.shape.rank != 2 or values.shape[-1] != 4:
            raise ValueError("batch_value_and_score requires static shape [batch,4]")
        batch_size = values.shape[0]
        if batch_size is None:
            raise ValueError("batch_value_and_score requires a static batch size")
        size = int(batch_size)
        compiled = self._compiled_batches.get(size)
        if compiled is None:
            compiled = tf.function(
                self._batch_value_score_impl,
                input_signature=[tf.TensorSpec([size, 4], tf.float64)],
                jit_compile=self._jit_compile,
                reduce_retracing=True,
            )
            self._compiled_batches[size] = compiled
        return compiled(values)


def complexity_posterior_target(q: int, *, jit_compile: bool = True) -> SSLLSTMComplexityPosteriorTarget:
    return SSLLSTMComplexityPosteriorTarget(q, jit_compile=jit_compile)


def complexity_target_worker_factory(config: dict[str, Any]) -> SSLLSTMComplexityPosteriorTarget:
    """Build the scalar, non-XLA target used by a spawned CPU worker."""

    if not isinstance(config, dict) or "q" not in config:
        raise ValueError("complexity worker config requires q")
    return complexity_posterior_target(int(config["q"]), jit_compile=False)


__all__ = [
    "FREE_NAMES",
    "SSLLSTMComplexityPosteriorTarget",
    "complexity_posterior_target",
    "complexity_target_worker_factory",
    "make_complexity_config",
    "make_full_fixture",
    "make_synthetic_observations",
]
