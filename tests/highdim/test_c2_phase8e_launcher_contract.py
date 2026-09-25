"""Contract tests for the bounded Phase 8E campaign launcher."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "docs/benchmarks/run_c2_phase8e_statistical_replication_20260907.py"
SPEC = importlib.util.spec_from_file_location("c2_phase8e_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "output_root": str(tmp_path / "campaign"),
        "cuda_visible_devices": "0",
        "particle_count": 4096,
        "branch_count": 2,
        "state_seed": 20260912,
        "analysis_seed": 20260907,
        "observation_seeds": list(range(424300, 424312)),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_primary_launcher_validation_is_frozen_and_clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(MODULE, "_git", lambda *args: "")
    output_root, seeds = MODULE._validate_inputs(_args(tmp_path))
    assert output_root == (tmp_path / "campaign").resolve()
    assert seeds == tuple(range(424300, 424312))


@pytest.mark.parametrize(
    "overrides",
    [
        {"observation_seeds": [424300]},
        {"analysis_seed": 11},
        {"particle_count": 8192},
        {"branch_count": 1},
        {"state_seed": 52},
    ],
)
def test_primary_launcher_rejects_design_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: dict[str, object]
) -> None:
    monkeypatch.setattr(MODULE, "_git", lambda *args: "")
    with pytest.raises(ValueError):
        MODULE._validate_inputs(_args(tmp_path, **overrides))


def test_launcher_keeps_fixture_generation_cpu_only() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'generation_environment["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "campaign_manifest.json" in source
