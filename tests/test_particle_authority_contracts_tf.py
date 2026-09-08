"""Focused tests for the Phase 1 particle-authority fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "bayesfilter/testing/particle_authority_contracts_tf.py"


@pytest.fixture(scope="module")
def contracts():
    spec = importlib.util.spec_from_file_location("particle_authority_contracts", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_affine_density_identity_passes(contracts) -> None:
    result = contracts.affine_density_identity()
    assert result["status"] == "PASS"
    assert result["max_abs_log_density_residual"] <= 1.0e-12


def test_protocol_hash_detects_material_change(contracts) -> None:
    result = contracts.frozen_protocol_hash_check()
    assert result["status"] == "PASS"
    assert result["exact_replay"]
    assert result["material_change_detected"]


def test_known_density_mass_and_mode_missing_fixture(contracts) -> None:
    mass = contracts.known_density_mass_fixture(sample_count=4096)
    mode = contracts.mode_missing_transform_fixture()
    assert mass["status"] == "PASS"
    assert mode["status"] == "PASS"
    assert mode["input_negative_fraction"] == 0.0
    assert mode["bridge_fraction"] > 0.0


def test_mutation_tail_and_metadata_contracts(contracts) -> None:
    assert contracts.mutation_invariance_fixture(sample_count=4096)["status"] == "PASS"
    assert contracts.defensive_tail_fixture(grid_count=4001)["status"] == "PASS"
    assert contracts.replay_metadata_parity_fixture()["status"] == "PASS"


def test_runtime_lane_does_not_import_numpy() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "from numpy" not in source


def test_benchmark_wrapper_resolves_repository_root() -> None:
    runner = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_contracts_2026_08_25.py"
    source = runner.read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parents[2]" in source
    assert "sys.path.insert(0, str(REPOSITORY_ROOT))" in source
