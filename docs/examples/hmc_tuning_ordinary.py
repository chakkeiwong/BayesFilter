"""Small ordinary-coordinate call to BayesFilter's public HMC tuner.

The smoke preset exercises the interface with a deliberately tiny budget. Its
status is contract evidence only; this example does not require or claim tuning
admission, convergence, or sampler quality.
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
    HMCKernelTuningConfig,
    ValueScoreCapability,
    tune_hmc_kernel,
)


class AnalyticGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "docs-ordinary-analytic-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            target_scope="docs_ordinary_analytic_gaussian",
            nonclaims=("tiny documentation fixture only",),
        )

    def log_prob_and_grad(self, position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        position = tf.convert_to_tensor(position, dtype=tf.float64)
        log_prob = -0.5 * tf.reduce_sum(tf.square(position), axis=-1)
        return log_prob, -position


result = tune_hmc_kernel(
    adapter=AnalyticGaussianAdapter(),
    initial_position=[0.25, -0.25],
    config=HMCKernelTuningConfig.smoke(
        target_scope="docs_ordinary_analytic_gaussian",
    ),
)
payload = result.payload()

assert payload["schema"] == "bayesfilter.hmc_kernel_tuning_result.v1"
assert payload["smoke_result_is_contract_only"] is True
assert payload["reports_posterior_convergence"] is False
print(payload["final_status"])
