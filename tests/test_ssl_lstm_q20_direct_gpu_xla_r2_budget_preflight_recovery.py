from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import tensorflow as tf


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_direct_gpu_xla_r2_budget_preflight_recovery_2026_07_30.py"
)
PLAN_PATH = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-recovery-plan-2026-07-30.md"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("q20_r2_recovery", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_surface_is_bounded_and_preflight_only() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    plan = PLAN_PATH.read_text(encoding="utf-8")

    assert "numpy" not in imports
    assert "sample_chain(" not in source
    assert "HamiltonianMonteCarlo(" not in source
    assert "MATERIAL_CAP_SECONDS = 6992.0" in source
    assert "WARM_UPDATE_COUNT = 3" in source
    assert "TensorSpec([int(batch_size), 4]" in source
    assert "SOURCE_SHA256 != live_source_sha256()" in source
    assert "No tuning, final training, or HMC" in plan


def test_exact_shape_validation_preserves_static_batch_dimension() -> None:
    runner = load_runner()

    class Target:
        def batch_value_score_status(self, theta):
            rows = tf.shape(theta)[0]
            return (
                tf.zeros([rows], tf.float64),
                tf.zeros_like(theta),
                {
                    "hard_valid_for_training": tf.ones([rows], tf.bool),
                    "floor_count_value": tf.zeros([rows], tf.int32),
                    "min_innovation_eigenvalue": tf.ones([rows], tf.float64),
                },
            )

    class Trainer:
        def _validation_impl(self, z):
            assert z.shape == (256, 4)
            zeros = tf.zeros([256], tf.float64)
            return zeros, zeros, z, zeros, z, z, z

    result = runner.exact_shape_validation_result(
        Trainer(), Target(), tf.zeros([256, 4], tf.float64), batch_size=256
    )

    assert result["static_input_shape"] == [256, 4]
    assert result["row_count"] == 256
    assert result["all_target_status_valid"] is True


def test_recovery_projection_prices_prior_protocol_and_substitutions() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "updates=100" in source
    assert "updates=1000" in source
    assert "validation_calls=11" in source
    assert "support_calls=12" in source
    assert "audit_calls=2" in source
    assert "direct compile-inclusive cost used for first and warm calls" in source
    assert "conservative_requested_campaign_days" in source
