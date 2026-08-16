from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_direct_gpu_xla_r2_budget_preflight_2026_07_30.py"
)
PLAN_PATH = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-plan-2026-07-30.md"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("q20_r2_budget_preflight", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_result(*, warm_count: int = 5):
    operations = [
        ("target_and_binding_construction", 1.0),
        ("trainer_construction", 2.0),
        ("validation_64_first", 3.0),
        ("optimizer_update_1", 4.0),
    ]
    operations.extend(
        (f"optimizer_update_{index}", 10.0 + index)
        for index in range(2, 2 + warm_count)
    )
    operations.extend(
        (
            ("validation_64_warm", 5.0),
            ("status_probe_2", 6.0),
            ("support_export_first", 7.0),
            ("support_export_warm", 8.0),
            ("audit_shape_256_first", 9.0),
            ("audit_shape_256_warm", 10.0),
            ("hlo_extraction", 11.0),
        )
    )
    return {
        "operations": [
            {"name": name, "duration_seconds": duration, "result": {}}
            for name, duration in operations
        ],
        "process_elapsed_seconds": 123.0,
    }


def test_runner_has_bounded_preflight_only_surface() -> None:
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
    assert "tf.map_fn(" not in source
    assert "tf.vectorized_map(" not in source
    assert "sample_chain(" not in source
    assert "HamiltonianMonteCarlo(" not in source
    assert 'GPU_MODES = {"gpu-identity", "timing"}' in source
    assert "MATERIAL_CAP_SECONDS = 12000.0" in source
    assert "WARM_UPDATE_COUNT = 5" in source
    assert "active_operation" in source
    assert "12,000 s" in plan
    assert "No tuning, final training, HMC" in plan


def test_timing_summary_separates_first_and_warm_costs() -> None:
    runner = load_runner()

    summary = runner.timing_summary(synthetic_result())

    assert summary["construction_seconds"] == 3.0
    assert summary["optimizer_update_first_seconds"] == 4.0
    assert summary["optimizer_update_warm_seconds"] == [12.0, 13.0, 14.0, 15.0, 16.0]
    assert summary["optimizer_update_warm_median_seconds"] == 14.0
    assert summary["optimizer_update_warm_max_seconds"] == 16.0
    assert summary["validation_64_first_seconds"] == 3.0
    assert summary["validation_64_warm_seconds"] == 5.0
    assert summary["audit_shape_256_warm_seconds"] == 10.0


def test_timing_summary_rejects_too_few_warm_receipts() -> None:
    runner = load_runner()

    with pytest.raises(runner.PreflightError, match="insufficient warm"):
        runner.timing_summary(synthetic_result(warm_count=2))


def test_projected_process_cost_prices_every_declared_call() -> None:
    runner = load_runner()
    timing = runner.timing_summary(synthetic_result())

    actual = runner.projected_process_cost(
        timing,
        updates=4,
        validation_calls=3,
        support_calls=2,
        audit_calls=2,
        use_max=False,
    )

    expected = (
        3.0  # construction
        + 3.0  # first validation
        + 4.0  # first update
        + 3 * 14.0  # warm updates
        + 2 * 5.0  # warm validations
        + 7.0
        + 8.0  # first/warm support
        + 9.0
        + 10.0  # first/warm audit
        + 11.0  # HLO extraction
    )
    assert actual == expected


def test_projection_uses_architecture_specific_timing_and_25_percent_margin() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'for label in ("32x32", "64x64")' in source
    assert "* 2.0" in source
    assert "CONTINGENCY_FACTOR = 1.25" in source
    assert "buffered_warm_max_seconds" in source
    assert "conservative_requested_campaign_seconds" in source


def test_timing_requires_explicit_architecture() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'if args.mode == "timing" and args.architecture is None:' in source
    assert "timing mode requires --architecture" in source
