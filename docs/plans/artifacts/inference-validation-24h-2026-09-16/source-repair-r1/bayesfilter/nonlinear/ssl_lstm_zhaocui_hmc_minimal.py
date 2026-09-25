"""Internal minimal scalar SSL-LSTM Zhao-Cui HMC target helpers.

This module contains the frozen one-dimensional ``zhaocui_fixed`` fixture and
the Gaussian-prior HMC target adapter used by the minimal debug/reference HMC
ladder. It is an internal test/benchmark support surface, not a public API and
not source-faithful Zhao-Cui parity evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from bayesfilter.inference import ValueScoreCapability
from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig
from bayesfilter.nonlinear.ssl_lstm_zhaocui_fixed_adapter import (
    SSLLSTMZhaoCuiFixedManifest,
    make_ssl_lstm_zhaocui_fixed_components,
    tf_ssl_lstm_zhaocui_fixed_score,
)


MINIMAL_SSL_LSTM_ZHAOCUI_HMC_NONCLAIMS = (
    "Phase 1 target-adapter admission only",
    "CPU-hidden debug/reference exception only",
    "not an HMC sample or canary result",
    "not HMC convergence evidence",
    "not posterior correctness evidence",
    "not a method ranking or superiority claim",
    "not source-faithful SSL-LSTM Zhao-Cui parity evidence",
    "not GPU/XLA production-readiness evidence",
    "not default-readiness evidence",
    "not LEDH evidence",
)

MINIMAL_SSL_LSTM_ZHAOCUI_HMC_TARGET_SCOPE = (
    "minimal_ssl_lstm_zhaocui_hmc_ladder:zhaocui_fixed:phase1"
)
DEFAULT_MINIMAL_SSL_LSTM_ZHAOCUI_HMC_EVIDENCE_PATH = (
    "docs/plans/"
    "bayesfilter-minimal-ssl-lstm-zhaocui-hmc-ladder-"
    "phase1-target-adapter-subplan-2026-07-06.md"
)


class MinimalZhaoCuiHMCTargetAdapter:
    """Gaussian-prior HMC target wrapper for the frozen scalar fixture."""

    def __init__(
        self,
        *,
        prior_center: tf.Tensor | None = None,
        prior_scale: float = 5.0,
        evidence_path: str = DEFAULT_MINIMAL_SSL_LSTM_ZHAOCUI_HMC_EVIDENCE_PATH,
        manifest: SSLLSTMZhaoCuiFixedManifest | None = None,
    ) -> None:
        self.config = minimal_ssl_lstm_config()
        self.observations = minimal_ssl_lstm_observations()
        self.prior_center = tf.reshape(
            tf.convert_to_tensor(
                minimal_ssl_lstm_theta() if prior_center is None else prior_center,
                dtype=tf.float64,
            ),
            [self.config.parameter_dim],
        )
        self.prior_scale = float(prior_scale)
        if not math.isfinite(self.prior_scale) or self.prior_scale <= 0.0:
            raise ValueError("prior_scale must be positive and finite")
        self.evidence_path = str(evidence_path)
        self.manifest = manifest or minimal_ssl_lstm_zhaocui_manifest()
        self.parameter_dim = int(self.config.parameter_dim)
        self.target_scope = MINIMAL_SSL_LSTM_ZHAOCUI_HMC_TARGET_SCOPE
        self.filter_name = "zhaocui_fixed"
        self._components = make_ssl_lstm_zhaocui_fixed_components(
            self.prior_center,
            self.config,
            evidence_path=self.evidence_path,
            manifest=self.manifest,
        )

    def adapter_signature(self) -> str:
        payload = {
            "filter_name": self.filter_name,
            "horizon": int(self.config.horizon),
            "latent_dim": int(self.config.latent_dim),
            "hidden_dim": int(self.config.hidden_dim),
            "observation_dim": int(self.config.observation_dim),
            "parameter_dim": self.parameter_dim,
            "prior_scale": self.prior_scale,
            "evidence_path": self.evidence_path,
            "manifest": self.manifest.as_dict(),
            "target_scope": self.target_scope,
        }
        blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def value_score_capability(self) -> ValueScoreCapability:
        base = self._components.protocol.contract.value_score
        return ValueScoreCapability(
            value_score_authority=base.value_score_authority,
            xla_hmc_ready=base.xla_hmc_ready,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend=base.runtime_backend,
            evidence_path=self.evidence_path,
            target_scope=self.target_scope,
            nonclaims=tuple(base.nonclaims) + MINIMAL_SSL_LSTM_ZHAOCUI_HMC_NONCLAIMS,
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
        if theta_tensor.shape.rank == 2:
            values = []
            scores = []
            for index in range(int(theta_tensor.shape[0])):
                value, score = self._scalar_log_prob_and_grad(theta_tensor[index])
                values.append(value)
                scores.append(score)
            return tf.stack(values), tf.stack(scores)
        return self._scalar_log_prob_and_grad(theta_tensor)

    def _scalar_log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta_vector = tf.reshape(
            tf.convert_to_tensor(theta, dtype=tf.float64),
            [self.parameter_dim],
        )
        result, _components = tf_ssl_lstm_zhaocui_fixed_score(
            self.observations,
            theta_vector,
            self.config,
            evidence_path=self.evidence_path,
            manifest=self.manifest,
        )
        delta = theta_vector - self.prior_center
        variance = tf.constant(self.prior_scale * self.prior_scale, dtype=tf.float64)
        prior_value = -0.5 * tf.reduce_sum(tf.square(delta) / variance)
        prior_score = -delta / variance
        score = tf.reshape(
            tf.convert_to_tensor(result.score, dtype=tf.float64),
            [self.parameter_dim],
        )
        return (
            tf.convert_to_tensor(result.log_likelihood, dtype=tf.float64)
            + prior_value,
            score + prior_score,
        )


def minimal_ssl_lstm_config() -> SSLLSTMStaticConfig:
    """Return the frozen scalar SSL-LSTM fixture dimensions."""

    return SSLLSTMStaticConfig(
        horizon=2,
        latent_dim=1,
        hidden_dim=1,
        observation_dim=1,
    )


def minimal_ssl_lstm_theta() -> tf.Tensor:
    """Return the frozen scalar fixture parameter vector."""

    values = [
        0.09,
        -0.07,
        0.05,
        0.04,
        0.03,
        -0.02,
        0.06,
        -0.05,
        0.01,
        0.04,
        -0.03,
        0.02,
        0.35,
        -0.08,
        0.65,
        0.05,
        0.15,
        -0.10,
        0.20,
        -0.35,
        0.15,
        0.55,
        0.35,
        -0.15,
    ]
    return tf.constant(values, dtype=tf.float64)


def minimal_ssl_lstm_observations() -> tf.Tensor:
    """Return the frozen two-step scalar observation fixture."""

    return tf.constant([[0.12], [-0.03]], dtype=tf.float64)


def minimal_ssl_lstm_zhaocui_manifest() -> SSLLSTMZhaoCuiFixedManifest:
    """Return the fixed replay manifest used by the scalar test fixture."""

    return SSLLSTMZhaoCuiFixedManifest(
        reference_sample_count=9,
        initial_seed=(20260705, 41),
        process_seed=(20260705, 43),
    )


def initial_minimal_ssl_lstm_hmc_state(offset_scale: float = 1.0e-3) -> tf.Tensor:
    """Return a deterministic near-center initial state for later HMC phases."""

    scale = tf.constant(float(offset_scale), dtype=tf.float64)
    return minimal_ssl_lstm_theta() + scale * tf.linspace(
        tf.constant(-1.0, dtype=tf.float64),
        tf.constant(1.0, dtype=tf.float64),
        minimal_ssl_lstm_config().parameter_dim,
    )


def minimal_ssl_lstm_fixture_payload() -> dict[str, Any]:
    """Return a JSON-ready fixture payload for benchmark artifacts."""

    config = minimal_ssl_lstm_config()
    observations = minimal_ssl_lstm_observations()
    theta = minimal_ssl_lstm_theta()
    return {
        "horizon": int(config.horizon),
        "latent_dim": int(config.latent_dim),
        "hidden_dim": int(config.hidden_dim),
        "observation_dim": int(config.observation_dim),
        "augmented_state_dim": int(config.augmented_state_dim),
        "parameter_dim": int(config.parameter_dim),
        "observations": [
            [float(value) for value in row]
            for row in tf.convert_to_tensor(observations, dtype=tf.float64).numpy()
        ],
        "theta": [
            float(value)
            for value in tf.reshape(
                tf.convert_to_tensor(theta, dtype=tf.float64),
                [-1],
            ).numpy()
        ],
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value
