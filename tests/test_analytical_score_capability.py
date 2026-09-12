"""Contract tests for explicit analytical/manual score provenance."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    FixedTransportValueScoreAdapter,
    LatentAffineBatchValueScoreAdapter,
    LatentAffineHMCTransform,
    ValueScoreCapability,
    bind_batch_native_neutra_target,
)
from bayesfilter.inference.native_tfp_hmc import _capability_payload


class AnalyticalFixture:
    parameter_dim = 2
    parameter_names = ("alpha", "beta")
    config = object()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="analytical_manual",
            xla_hmc_ready=True,
            evidence_path="tests/test_analytical_score_capability.py",
            target_scope="analytical_fixture",
        )

    def adapter_signature(self) -> str:
        return "a" * 64

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        theta = tf.convert_to_tensor(theta, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(theta), axis=-1), -theta

    def neutra_batch_log_prob_and_grad_status(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
        theta = tf.convert_to_tensor(theta, tf.float64)
        value = -0.5 * tf.reduce_sum(tf.square(theta), axis=-1)
        score = -theta
        leading = tf.shape(value)
        return value, score, {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
        }


class AffineTransport:
    parameter_dim = 2

    def __init__(self) -> None:
        self.factor = tf.constant([[2.0, 0.0], [0.25, 1.5]], tf.float64)
        self.shift = tf.constant([0.1, -0.2], tf.float64)

    def manifest_payload(self) -> dict[str, object]:
        return {"schema": "analytical_affine_transport.v1", "parameter_dim": 2}

    def forward(self, z: tf.Tensor) -> tf.Tensor:
        return self.shift + tf.linalg.matvec(self.factor, z)

    def forward_batch(self, z: tf.Tensor) -> tf.Tensor:
        return self.shift + tf.linalg.matmul(z, self.factor, transpose_b=True)

    def log_abs_det_jacobian(self, z: tf.Tensor) -> tf.Tensor:
        del z
        return tf.linalg.slogdet(self.factor)[1]

    def log_abs_det_jacobian_batch(self, z: tf.Tensor) -> tf.Tensor:
        return tf.fill(tf.shape(z)[:1], self.log_abs_det_jacobian(z[0]))

    def pullback_score(self, z: tf.Tensor, score: tf.Tensor) -> tf.Tensor:
        del z
        return tf.linalg.matvec(self.factor, score, transpose_a=True)

    def pullback_score_batch(self, z: tf.Tensor, score: tf.Tensor) -> tf.Tensor:
        del z
        return tf.linalg.matmul(score, self.factor)

    def log_abs_det_jacobian_score(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(z)

    def log_abs_det_jacobian_score_batch(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(z)


def test_analytical_manual_authority_is_explicit_and_fail_closed() -> None:
    capability = ValueScoreCapability(
        value_score_authority="analytical_manual",
        score_provenance="analytical_manual",
        xla_hmc_ready=True,
        evidence_path="tests/test_analytical_score_capability.py",
        target_scope="analytical_fixture",
    )
    assert capability.is_analytical_manual_score is True
    assert capability.is_accepted_xla_hmc_authority is True
    assert capability.is_accepted_analytical_xla_hmc_authority is True
    assert _capability_payload(capability)["score_provenance"] == "analytical_manual"

    with pytest.raises(ValueError, match="requires analytical_manual"):
        ValueScoreCapability(
            value_score_authority="analytical_manual",
            score_provenance="unspecified",
            xla_hmc_ready=False,
        )
    with pytest.raises(ValueError, match="requires analytical_manual authority"):
        ValueScoreCapability(
            value_score_authority="graph_native",
            score_provenance="analytical_manual",
            xla_hmc_ready=False,
        )


def test_latent_and_fixed_transport_wrappers_preserve_provenance() -> None:
    transform = LatentAffineHMCTransform(
        center=np.zeros(2),
        factor=np.eye(2),
        covariance_provenance="analytical_fixture_covariance",
    )
    latent = LatentAffineBatchValueScoreAdapter(
        base_adapter=AnalyticalFixture(),
        transform=transform,
        target_scope="analytical_fixture_latent",
        xla_hmc_ready=True,
        evidence_path="tests/test_analytical_score_capability.py",
    )
    assert latent.value_score_capability().score_provenance == "analytical_manual"
    assert latent.value_score_capability().is_accepted_xla_hmc_authority is True

    fixed = FixedTransportValueScoreAdapter(
        base_adapter=AnalyticalFixture(),
        transport=AffineTransport(),
        target_scope="analytical_fixture_fixed",
        xla_hmc_ready=True,
        evidence_path="tests/test_analytical_score_capability.py",
    )
    capability = fixed.value_score_capability()
    assert capability.value_score_authority == "analytical_manual"
    assert capability.score_provenance == "analytical_manual"
    assert fixed.adapter_signature_payload()["score_provenance"] == "analytical_manual"
    assert fixed.manifest_payload()["score_provenance"] == "analytical_manual"


def test_neutra_binding_accepts_analytical_authority_and_records_it() -> None:
    adapter = AnalyticalFixture()
    binding = bind_batch_native_neutra_target(
        adapter,
        target_signature="b" * 64,
    )
    payload = binding.payload()
    assert payload["value_score_authority"] == "analytical_manual"
    assert payload["score_provenance"] == "analytical_manual"


@pytest.mark.parametrize(
    "authority",
    (
        "gradient_tape_fallback",
        "debug_only",
        "unavailable",
    ),
)
def test_neutra_binding_rejects_non_analytical_or_unaccepted_authority(
    authority: str,
) -> None:
    class NonAnalyticalFixture(AnalyticalFixture):
        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority=authority,  # type: ignore[arg-type]
                xla_hmc_ready=authority == "reviewed_gradient_tape_xla_exception",
                evidence_path="tests/test_analytical_score_capability.py",
                target_scope="non_analytical_fixture",
            )

    with pytest.raises(ValueError, match="accepted XLA|requires an accepted"):
        bind_batch_native_neutra_target(
            NonAnalyticalFixture(),
            target_signature="c" * 64,
        )


def test_reviewed_tape_authority_is_not_analytical_only() -> None:
    capability = ValueScoreCapability(
        value_score_authority="reviewed_gradient_tape_xla_exception",
        xla_hmc_ready=True,
        evidence_path="tests/test_analytical_score_capability.py",
        target_scope="taped_fixture",
    )
    assert capability.is_accepted_xla_hmc_authority is True
    assert capability.is_accepted_analytical_xla_hmc_authority is False
