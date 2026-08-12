from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "docs/benchmarks"
LOADER = BENCHMARKS / "ssl_lstm_q20_neutra_seed_b_terminal.py"
TUNER = BENCHMARKS / (
    "run_ssl_lstm_q20_seed_b_terminal_six_l_tuning_2026_08_07.py"
)
SEQUENTIAL = BENCHMARKS / (
    "run_ssl_lstm_q20_seed_b_terminal_sequential_hmc_2026_08_07.py"
)
SUPERVISOR = BENCHMARKS / (
    "run_ssl_lstm_q20_seed_b_terminal_neutra_validation_supervisor_2026_08_07.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_b_terminal_binding_is_clean_and_selected() -> None:
    module = _load(LOADER, "seed_b_terminal_loader_test")
    binding = module.binding_payload()
    assert binding["optimizer_step"] == 6250
    assert binding["continuation_update"] == 4000
    assert binding["training_result_status"] == (
        "GPU_CONTINUATION_COMPLETED_CANDIDATE_NOMINATED"
    )
    assert binding["vetoes"] == []
    assert binding["target_validity_failure_count"] is None
    assert "not_explicitly_recorded" in binding["target_validity_event_telemetry"]
    assert binding["support_all_finite"] is True
    assert binding["roundtrip_max_abs"] < 1.0e-9


def test_seed_b_tuner_uses_fresh_grid_and_forbids_l1() -> None:
    module = _load(TUNER, "seed_b_terminal_tuner_test")
    assert module.base.CANONICAL_GRID == (5, 10, 15, 20, 25, 3)
    assert 1 not in module.base.CANONICAL_GRID
    assert module.TUNE_SEED_BASE[0] == 20260807
    assert module.SCREEN_SEED_BASE[0] == 20260807
    assert module.VERIFICATION_SEED_BASE[0] == 20260807
    assert "seed-b-terminal-step-6250" in module.TARGET_SCOPE
    with pytest.raises(ValueError, match="L=1"):
        module._config_for_worker(leapfrog=1, candidate_index=0)


def test_seed_b_sequential_kernel_loader_is_fail_closed(tmp_path: Path) -> None:
    module = _load(SEQUENTIAL, "seed_b_terminal_sequential_test")
    tuner = _load(TUNER, "seed_b_terminal_tuner_scope_test")
    assert module.TARGET_SCOPE == tuner.TARGET_SCOPE
    with pytest.raises(module.base.CampaignError, match="does not exist"):
        module.load_frozen_kernel(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "passed": True,
                "final_status": "passed",
                "final_kernel_payload": {
                    "num_leapfrog_steps": 1,
                    "step_size": 0.1,
                    "mass_policy": "fixed_identity_z",
                    "use_xla": True,
                },
            }
        ),
        encoding="ascii",
    )
    with pytest.raises(module.base.CampaignError, match="L=1"):
        module.load_frozen_kernel(invalid)


def test_seed_b_sequential_source_keeps_full_policy_minima() -> None:
    source = SEQUENTIAL.read_text(encoding="utf-8")
    base_source = (
        BENCHMARKS / "run_ssl_lstm_q20_chart_a_l10_sequential_hmc_2026_08_04.py"
    ).read_text(encoding="utf-8")
    assert 'keywords["acceptance_min"] = 0.0' not in source
    assert "warmup_min_results=2000" in base_source
    assert "retained_min_results=1000" in base_source
    assert "bulk_ess_min=400.0" in base_source
    assert "tail_ess_min=400.0" in base_source
    assert 'keywords["archive_label"] = "seed-b-terminal"' in source


def test_seed_b_supervisor_chains_tuning_before_sequential() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert 'merged.get("passed") is not True' in source
    assert source.index('label="seed_b_fixed_hmc_tuning"') < source.index(
        'label="seed_b_sequential_preflight"'
    )
    assert source.index('label="seed_b_sequential_preflight"') < source.index(
        'label="seed_b_sequential_hmc"'
    )
    assert "TUNING_CAP_SECONDS = 43_200.0" in source
    assert "SEQUENTIAL_CAP_SECONDS = 86_400.0" in source
    assert '"CUDA_VISIBLE_DEVICES": "-1"' in source
