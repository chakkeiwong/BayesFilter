from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_hmc_pipeline_2026_07_20.py"
SPEC = importlib.util.spec_from_file_location("ssl_lstm_unified_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_contract_smoke_is_nonmaterial() -> None:
    args = MODULE.parse_args(
        ["--mode", "contract-smoke", "--q", "20", "--batch-size", "100"]
    )
    payload = MODULE.contract_payload(args)
    assert payload["status"] == "PASSED"
    assert payload["batch_size"] == 100
    assert payload["material_execution_authorized"] is False


def test_material_args_require_caps_params_and_output() -> None:
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--q", "20"])
    with pytest.raises(SystemExit):
        MODULE.parse_args(
            ["--q", "20", "--authorize-material-run", "--batch-size", "100"]
        )


def test_require_status_rejects_vetoed_handoff(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"status": "VETOED"}), encoding="utf-8")
    with pytest.raises(MODULE.PipelineError, match="not one of"):
        MODULE.require_status(path, label="training summary", statuses={"COMPLETED"})


def test_repo_path_rejects_escape() -> None:
    with pytest.raises(MODULE.PipelineError, match="inside"):
        MODULE.repo_path(Path("../outside"), label="output root")
