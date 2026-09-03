"""Small frozen-transport call to BayesFilter's public HMC tuner.

The identity map keeps the example short while still satisfying the complete
frozen-transport protocol. The tiny budget is interface evidence only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tensorflow as tf

from bayesfilter.inference import (
    FIXED_TRANSPORT_HMC_MEASURED_POLICY,
    FixedTransportHMCKernelTuningConfig,
    ValueScoreCapability,
    tune_fixed_transport_hmc_kernel,
)


class AnalyticGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "docs-fixed-transport-analytic-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            target_scope="docs_fixed_transport_analytic_gaussian",
            nonclaims=("tiny documentation fixture only",),
        )

    def log_prob_and_grad(self, position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        position = tf.convert_to_tensor(position, dtype=tf.float64)
        log_prob = -0.5 * tf.reduce_sum(tf.square(position), axis=-1)
        return log_prob, -position


class FrozenIdentityTransport:
    parameter_dim = 2

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema": "docs.frozen_identity_transport.v1",
            "parameter_dim": self.parameter_dim,
            "kind": "identity",
        }

    def forward(self, latent: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(latent, dtype=tf.float64)

    def forward_batch(self, latent: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(latent, dtype=tf.float64)

    def log_abs_det_jacobian(self, latent: tf.Tensor) -> tf.Tensor:
        del latent
        return tf.constant(0.0, dtype=tf.float64)

    def log_abs_det_jacobian_batch(self, latent: tf.Tensor) -> tf.Tensor:
        latent = tf.convert_to_tensor(latent, dtype=tf.float64)
        return tf.zeros(tf.shape(latent)[:1], dtype=tf.float64)

    def pullback_score(
        self, latent: tf.Tensor, position_score: tf.Tensor
    ) -> tf.Tensor:
        del latent
        return tf.convert_to_tensor(position_score, dtype=tf.float64)

    def pullback_score_batch(
        self, latent: tf.Tensor, position_score: tf.Tensor
    ) -> tf.Tensor:
        del latent
        return tf.convert_to_tensor(position_score, dtype=tf.float64)

    def log_abs_det_jacobian_score(self, latent: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(tf.convert_to_tensor(latent, dtype=tf.float64))

    def log_abs_det_jacobian_score_batch(self, latent: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(tf.convert_to_tensor(latent, dtype=tf.float64))


result = tune_fixed_transport_hmc_kernel(
    base_adapter=AnalyticGaussianAdapter(),
    fixed_transport=FrozenIdentityTransport(),
    initial_position=[0.25, -0.25],
    config=FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1,
        # Claim-bearing tuning measures every declared (epsilon, L) pair.
        step_size_candidates=(0.05, 0.1, 0.2),
        leapfrog_grid=(2, 4),
        chain_count=4,
        selection_replications=2,
        selection_num_results=16,
        selection_num_burnin_steps=4,
        verification_num_results=8,
        verification_num_burnin_steps=2,
        chain_execution_mode="eager",
        use_xla=False,
        target_scope="docs_fixed_transport_analytic_gaussian",
        tuning_policy=FIXED_TRANSPORT_HMC_MEASURED_POLICY,
    ),
)
payload = result.payload()

assert payload["schema"] == "bayesfilter.fixed_transport_hmc_kernel_tuning_result.v5"
assert payload["reports_posterior_convergence"] is False
print(payload["final_status"])
