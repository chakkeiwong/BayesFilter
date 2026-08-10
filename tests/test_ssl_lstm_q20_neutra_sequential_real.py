from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_q20_neutra_sequential_real_2026_07_22.py"
)


def load_runner():
    name = "ssl_lstm_q20_neutra_sequential_real_runner"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_contract_uses_repository_sequential_policy_and_two_coordinates() -> None:
    payload = runner.contract_payload()
    assert payload["policy_id"] == "bayesfilter_neutra_sequential_hmc_v1"
    assert payload["charts"] == ["chart-a", "chart-b"]
    assert payload["sequential_policy"]["warmup_min_results"] == 2000
    assert payload["sequential_policy"]["retained_min_results"] == 1000
    assert payload["diagnostic_coordinate_systems"] == [
        "hmc_coordinates",
        "model_parameters",
    ]
    assert payload["energy_error_identity"] == (
        "delta_h_equals_negative_log_accept_ratio"
    )
    assert payload["material_execution_authorized"] is False


def test_launcher_has_no_numpy_and_configures_growth_before_project_imports() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    growth = source.index("configure_tensorflow_gpu_memory_growth(")
    controller = source.index("from bayesfilter.inference.neutra_hmc import")
    assert growth < controller
    assert "run_sequential_neutra_hmc(" in source
    assert "np." not in source


def test_active_launcher_is_the_only_new_claim_bearing_sequential_route() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "NEUTRA_SEQUENTIAL_HMC_POLICY_ID" in source
    assert "run_sequential_neutra_hmc(" in source
    plan = (
        ROOT
        / "docs/plans/bayesfilter-ssl-lstm-q20-neutra-sequential-real-run-plan-2026-07-22.md"
    ).read_text(encoding="utf-8")
    assert "Active NeuTra-HMC Route Ledger" in plan
    assert "historical fixed-burn-in/checkpoint route" in plan


def test_every_fixed_transport_neutra_launcher_is_classified() -> None:
    plan = (
        ROOT
        / "docs/plans/bayesfilter-ssl-lstm-q20-neutra-sequential-real-run-plan-2026-07-22.md"
    ).read_text(encoding="utf-8")
    candidates = []
    for path in sorted((ROOT / "docs/benchmarks").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "FixedTransportValueScoreAdapter" in source and (
            "HamiltonianMonteCarlo" in source
            or "sample_chain" in source
            or "build_retained_sample_hmc_archive_runner" in source
        ):
            candidates.append(path.relative_to(ROOT).as_posix())
    missing = [path for path in candidates if f"`{path}`" not in plan]
    assert not missing, missing


def test_material_mode_requires_explicit_cli_gate() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--mode", "acquire"])
    args = runner.parse_args(
        [
            "--mode",
            "acquire",
            "--authorize-material-run",
            "--cap-seconds",
            "77700",
        ]
    )
    assert args.cap_seconds == pytest.approx(77700.0)


def test_acquisition_rejects_cap_above_reviewed_bound() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--mode",
                "acquire",
                "--authorize-material-run",
                "--cap-seconds",
                "77701",
            ]
        )


def test_budget_uses_measured_rate_and_margin(monkeypatch) -> None:
    budget = runner.CampaignBudget(10000.0)
    monkeypatch.setattr(budget, "started", budget.started)
    assert budget.allow(2000) is True
    expected = 2000 * runner.MEASURED_SECONDS_PER_TRANSITION_LEAPFROG * 1.5
    assert expected < 10000.0
