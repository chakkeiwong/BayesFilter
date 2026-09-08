"""Focused tests for the q20 LEDH adapter audit contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_particle_authority_ledh_adapter_audit_2026_08_25.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("q20_ledh_adapter_audit", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load audit runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_requires_cpu_memory_policy_environment(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    module = _module()
    assert module.TARGET.exists()
    assert module.STRUCTURAL.exists()


def test_audit_records_missing_density_lifecycle(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    module = _module()
    result = module.build_audit()
    assert result["status"] == "ADAPTER_NOT_READY_REPAIRABLE"
    assert result["hard_checks"]["model_shape_contract"] is True
    assert result["hard_checks"]["transition_finite"] is True
    assert result["hard_checks"]["derivatives_finite"] is True
    assert "transition_log_density" in result["missing_required_terms"]
    assert "pre_flow_proposal_density" in result["missing_required_terms"]
    assert "per_step_covariance_lifecycle" in result["missing_required_terms"]
    assert result["callback_checks"]["transition"]["rank_matches"] is True
    assert result["callback_checks"]["observation"]["rank_matches"] is True
