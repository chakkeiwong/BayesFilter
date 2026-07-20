from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bayesfilter.inference.hmc_fixed_metric_grid_search import DEFAULT_L_GRID
from bayesfilter.testing.ssl_lstm_q20_fixed_metric_worker import (
    SCREEN_BURNIN,
    TARGET_SIGNATURE,
    TUNE_BURNIN,
    TUNE_RESULTS,
    expected_lineage_payload,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_process_grid_hmc_tuning_2026_07_20.py"
)
WORKER_PATH = ROOT / "bayesfilter/testing/ssl_lstm_q20_fixed_metric_worker.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("ssl_lstm_q20_process_grid", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_parent_and_worker_module_scope_do_not_import_tensorflow() -> None:
    for path in (RUNNER_PATH, WORKER_PATH):
        imports = imported_modules(path)
        assert "tensorflow" not in imports
        assert not any(name.startswith("tensorflow.") for name in imports)
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert source.index("_require_parent_launch_environment()") < source.index(
        "from bayesfilter.inference"
    )


def test_contract_binds_plain_q20_target_and_reviewed_policy() -> None:
    runner = load_runner()
    payload = runner.contract_payload()

    assert payload["status"] == "PASSED"
    assert payload["target_type"] == "plain_q20_posterior_no_transport"
    assert payload["target_signature"] == TARGET_SIGNATURE
    assert payload["lineage"] == expected_lineage_payload()
    assert payload["config"]["l_grid"] == DEFAULT_L_GRID
    assert payload["config"]["replication_count"] == 3
    assert payload["acceptance_policy"]["target"] == pytest.approx(0.70)
    assert payload["tune"]["num_results"] == TUNE_RESULTS == 1
    assert payload["tune"]["num_burnin_steps"] == TUNE_BURNIN == 64
    assert payload["screen_burnin_steps"] == SCREEN_BURNIN == 1


def test_projection_counts_complete_candidate_work_and_margin() -> None:
    runner = load_runner()
    leapfrog = 3
    without_extensions = leapfrog * (
        TUNE_BURNIN + TUNE_RESULTS + 3 * (SCREEN_BURNIN + 64)
    )
    with_extensions = without_extensions + leapfrog * 3 * (SCREEN_BURNIN + 128)

    assert runner.candidate_transition_leapfrogs(
        leapfrog, include_all_extensions=False
    ) == without_extensions
    assert runner.candidate_transition_leapfrogs(
        leapfrog, include_all_extensions=True
    ) == with_extensions
    projection = runner.projected_seconds(
        (3,),
        seconds_per_transition_leapfrog=2.0,
        effective_workers=1.0,
        include_all_extensions=False,
        cold_seconds_per_worker=10.0,
        worker_process_count=1,
    )
    assert projection == pytest.approx(1.5 * (without_extensions * 2.0 + 10.0))


def test_resource_launch_fails_closed_above_remaining_cap() -> None:
    runner = load_runner()
    runner.validate_resource_launch(
        cap_seconds=100.0, prior_seconds=10.0, projection_seconds=90.0
    )
    with pytest.raises(runner.Q20TuningError, match="remaining cap"):
        runner.validate_resource_launch(
            cap_seconds=100.0, prior_seconds=10.0, projection_seconds=90.01
        )
    with pytest.raises(runner.Q20TuningError, match="eight-GPU-hour"):
        runner.validate_resource_launch(
            cap_seconds=8.0 * 3600.0 + 1.0,
            prior_seconds=0.0,
            projection_seconds=1.0,
        )


def test_hmc_test_gate_requires_complete_successful_grid() -> None:
    runner = load_runner()
    base = {
        "schema": runner.SCHEMA,
        "mode": "grid",
        "status": "NO_SURVIVOR",
        "round0_complete": True,
        "grid_private": {"survivor_pairs": []},
    }
    with pytest.raises(runner.Q20TuningError, match="successful tuning"):
        runner.representative_from_grid(base)

    incomplete = {
        **base,
        "status": "TUNING_SUCCEEDED",
        "round0_complete": False,
        "grid_private": {
            "survivor_pairs": [
                {"num_leapfrog_steps": 3, "tuned_step_size": 0.1}
            ]
        },
    }
    with pytest.raises(runner.Q20TuningError, match="complete broad"):
        runner.representative_from_grid(incomplete)


def test_representative_is_deterministic_and_not_ranked_by_diagnostics() -> None:
    runner = load_runner()
    payload = {
        "schema": runner.SCHEMA,
        "mode": "grid",
        "status": "TUNING_SUCCEEDED",
        "round0_complete": True,
        "grid_private": {
            "lineage": expected_lineage_payload(),
            "survivor_pairs": [
                {"num_leapfrog_steps": 9, "tuned_step_size": 0.2},
                {"num_leapfrog_steps": 3, "tuned_step_size": 0.4},
                {"num_leapfrog_steps": 3, "tuned_step_size": 0.3},
            ],
        },
        "source_bindings": {"execution_source_signature": "source"},
    }

    representative = runner.representative_from_grid(
        json.loads(json.dumps(payload))
    )

    assert representative["num_leapfrog_steps"] == 3
    assert representative["step_size"] == pytest.approx(0.3)
    assert "no_stochastic_ranking" in representative["selection_rule"]
