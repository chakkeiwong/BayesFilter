"""Tests for the reduced innovation-coordinate density fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_particle_authority_ledh_reduced_fixture_2026_08_25.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("q20_ledh_reduced_fixture", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load reduced fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reduced_fixture_passes_but_keeps_target_binding_veto(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    module = _module()
    result = module.build_fixture()
    assert result["status"] == "REDUCED_COORDINATE_DENSITY_FIXTURE_PASS_TARGET_UNBOUND"
    assert result["hard_checks"]["inverse_roundtrip_residual"] <= 1.0e-10
    assert result["hard_checks"]["density_identity_residual"] <= 1.0e-10
    assert result["target"]["binding_to_parameter_target"] is False
    assert result["decision"]["direct_q20_ledh"] == (
        "closed_wrong_relative_to_declared_four_parameter_target"
    )
