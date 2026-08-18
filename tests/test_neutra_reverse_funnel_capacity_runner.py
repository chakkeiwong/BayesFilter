"""Static and analytic checks for the reverse-funnel capacity runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_neutra_reverse_funnel_capacity_2026_08_14.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("reverse_funnel_capacity_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_is_reverse_only_gpu_xla_and_proposal_gated() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"objective": "reverse_kl"' in source
    assert "MatchedReverseKLNeuTraTrainer" in source
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert "enable_tensor_float_32_execution(False)" in source
    assert "jit_compile=True" in source
    assert "proposal_gate_passed" in source
    assert "stage_s_max=caps" in source
    assert "stage_scale_linear_skip=" in source
    assert "first_stage_scale_linear_skip" in source
    assert "stage_unbounded_scale_linear=" in source
    assert "first_stage_unbounded_scale_linear" in source
    assert "permutation_policy=str(args.permutation_policy)" in source
    assert "learning_rate_schedule" in source
    assert 'args.run_mode == "confirmation"' in source
    assert "import numpy" not in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source


def test_stage_cap_parser_rejects_wrong_width_and_nonpositive_values() -> None:
    runner = _load_runner()
    assert runner._parse_stage_caps("3,0.5,0.5") == (3.0, 0.5, 0.5)
    assert runner._parse_stage_caps("4", 1) == (4.0,)
    for value in ("3,0.5", "3,0,0.5", "3,nan,0.5"):
        try:
            runner._parse_stage_caps(value, 3)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid caps were accepted: {value}")


def test_learning_rate_schedule_has_predeclared_fractional_drops() -> None:
    runner = _load_runner()
    assert runner._scheduled_learning_rate(1.0e-3, "constant", 900, 1000) == 1.0e-3
    assert runner._scheduled_learning_rate(1.0e-3, "piecewise_60_85", 599, 1000) == 1.0e-3
    assert runner._scheduled_learning_rate(1.0e-3, "piecewise_60_85", 600, 1000) == 1.0e-4
    assert runner._scheduled_learning_rate(1.0e-3, "piecewise_60_85", 850, 1000) == 1.0e-5


def test_iid_intervals_accept_exact_like_values_and_reject_shift() -> None:
    runner = _load_runner()
    exact_like = tf.random.stateless_normal((131072,), seed=(20260814, 72001), dtype=tf.float64)
    mean = runner._mean_interval(tf, exact_like, 0.0)
    second = runner._mean_interval(tf, tf.square(exact_like), 1.0)
    shifted = runner._mean_interval(tf, exact_like + tf.constant(0.1, tf.float64), 0.0)
    tail = runner._wilson_interval(
        tf,
        tf.reduce_sum(tf.cast(exact_like < -2.0, tf.int64)),
        131072,
        runner.EXACT_TAIL_PROBABILITY,
    )
    assert bool(mean["passed"])
    assert bool(second["passed"])
    assert bool(tail["passed"])
    assert not bool(shifted["passed"])
