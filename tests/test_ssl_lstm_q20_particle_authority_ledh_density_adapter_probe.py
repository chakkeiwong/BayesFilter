"""Tests for the q20 LEDH target-measure probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_particle_authority_ledh_density_adapter_probe_2026_08_25.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("q20_ledh_density_probe", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load density probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_probe_separates_state_and_innovation_measures(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    module = _module()
    result = module.build_probe()
    assert result["status"] == (
        "DIRECT_FULL_STATE_LEDH_BLOCKED_SINGULAR_MEASURE_REDUCED_REPAIR"
    )
    assert result["dimensions"]["state_dim"] == 60
    assert result["dimensions"]["innovation_dim"] == 20
    assert result["dimensions"]["parameter_dim"] == 4
    assert result["hard_checks"]["innovation_density_finite"] is True
    assert result["measure"]["common_full_state_lebesgue_density"] is False
    ranks = result["rank_receipt"]["rank_counts"]["1e-10"]
    assert all(int(rank) == 20 for row in ranks for rank in row)
