"""Focused tests for the Phase 4A campaign runner contract."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "benchmarks"
    / "run_ledh_younis_kdm_phase4a_campaign.py"
)
sys.path.insert(0, str(RUNNER_PATH.parent))

from run_ledh_younis_kdm_phase4a_campaign import (  # noqa: E402
    _bootstrap_ci,
    _parse_float_list,
    _parse_int_list,
    _paired_error_summary,
    _path_seed,
    _sample_variance,
    _validate_seed_schedule,
)


def test_declared_campaign_seed_schedule_is_unique_and_disjoint() -> None:
    cells = [(32, 5), (32, 20), (128, 5), (128, 20)]
    _validate_seed_schedule(cells, calibration_reps=20, validation_reps=100)

    calibration = {
        _path_seed("calibration", n, horizon, replicate)
        for n, horizon in cells
        for replicate in range(20)
    }
    validation = {
        _path_seed("validation", n, horizon, replicate)
        for n, horizon in cells
        for replicate in range(100)
    }
    assert calibration.isdisjoint(validation)


def test_seed_schedule_rejects_collisions() -> None:
    # These coordinates have the same N*100 + T*10 offset.
    with pytest.raises(ValueError, match="calibration seed schedule"):
        _validate_seed_schedule([(32, 20), (33, 10)], 2, 2)


def test_seed_coordinates_and_split_are_checked() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        _path_seed("audit", 32, 5, 0)
    with pytest.raises(ValueError, match="outside"):
        _path_seed("calibration", 32, 5, -1)


def test_bootstrap_interval_is_deterministic() -> None:
    values = [-2.0, -1.0, 1.0, 2.0]
    first = _bootstrap_ci(values, seed=1234, replicates=2000)
    second = _bootstrap_ci(values, seed=1234, replicates=2000)
    assert first == second
    assert first[0] <= 0.0 <= first[1]
    assert math.isclose(_sample_variance(values), 10.0 / 3.0)


def test_paired_error_summary_uses_candidate_minus_atom() -> None:
    summary = _paired_error_summary(
        [1.0, 4.0, 9.0],
        [2.0, 3.0, 7.0],
        bootstrap_seed=7,
        bootstrap_reps=2000,
    )
    assert summary["mse"] == pytest.approx(14.0 / 3.0)
    assert summary["relative_gain"] == pytest.approx(-1.0 / 6.0)
    assert summary["paired_difference_mean"] == pytest.approx(2.0 / 3.0)
    assert summary["paired_difference_mcse"] == pytest.approx(
        math.sqrt((7.0 / 3.0) / 3.0)
    )


def test_list_parsers_reject_invalid_scopes() -> None:
    assert _parse_int_list("32, 128") == (32, 128)
    assert _parse_float_list("0,.4,.8") == (0.0, 0.4, 0.8)
    with pytest.raises(ValueError, match="positive"):
        _parse_int_list("32,0")
    with pytest.raises(ValueError, match="nonnegative"):
        _parse_float_list("0,-.1")


def test_runner_has_no_numpy_runtime_import() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "numpy" not in imported_roots


def test_all_rho_diagnostic_is_explicitly_opt_in() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"--retain-all-validation-rhos"' in source
    assert '"retained_all_validation_rhos"' in source
