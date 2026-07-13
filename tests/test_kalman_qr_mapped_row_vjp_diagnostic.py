from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = ROOT / (
    "docs/benchmarks/diagnose_kalman_qr_mapped_row_vjp_counterfactual_2026_07_13.py"
)
BENCHMARK_PATH = ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"


def _load(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_mapped_row_diagnostic_is_tensorflow_native_and_benchmark_stays_map_free() -> None:
    diagnostic = _load(DIAGNOSTIC_PATH, "kalman_qr_mapped_row_diagnostic_source")
    source = inspect.getsource(diagnostic.build_mapped_row_autodiff_fn)
    tree = ast.parse(source)

    assert "tf.map_fn" in source
    assert "parallel_iterations=1" in source
    assert "_batched_model_tensors" in source
    assert not any(isinstance(node, (ast.For, ast.ListComp)) for node in ast.walk(tree))
    assert "tf.map_fn" not in BENCHMARK_PATH.read_text(encoding="utf-8")


def test_mapped_row_diagnostic_matches_true_batched_on_tiny_fixture() -> None:
    diagnostic = _load(DIAGNOSTIC_PATH, "kalman_qr_mapped_row_diagnostic_numeric")
    benchmark = _load(BENCHMARK_PATH, "kalman_qr_mapped_row_benchmark")
    fixture = benchmark.make_fixture(2, 3, 4, dtype=tf.float64)
    params = benchmark._make_parameter_batch(fixture, 4)
    baseline = benchmark.build_batch_native_autodiff_fn(
        fixture, batch_size=4, jit_compile=False
    )
    mapped = diagnostic.build_mapped_row_autodiff_fn(
        benchmark, fixture, batch_size=4, jit_compile=False
    )

    baseline_value, baseline_score = baseline(params)
    mapped_value, mapped_score = mapped(params)

    np.testing.assert_allclose(mapped_value.numpy(), baseline_value.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(mapped_score.numpy(), baseline_score.numpy(), rtol=1e-8, atol=1e-9)
