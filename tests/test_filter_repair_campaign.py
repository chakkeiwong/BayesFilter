"""The campaign must fail closed on incomplete or numerically unequal evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unfinished_attempt_consumes_reserved_budget():
    driver = load("run_filter_repair_campaign")
    rows = [{"device": "GPU", "timeout_seconds": 300}, {"device": "GPU", "timeout_seconds": 300, "elapsed_seconds": 17.5}, {"device": "CPU", "timeout_seconds": 900}]
    assert driver.charged_seconds(rows, "GPU") == 317.5


def test_gate_rejects_empty_or_unsupported_closure(tmp_path, monkeypatch, capsys):
    driver = load("run_filter_repair_campaign")
    ledger = tmp_path / "docs/plans/filter_gradient_repair_ledger_20260917.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"findings": [{"id": f"F{i:02d}", "status": "closed"} for i in range(1, 21)]}))
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "records", list)
    monkeypatch.setattr(driver, "source_hashes", dict)
    assert driver.gate() == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["merge_allowed"] is False
    assert len(verdict["open_findings"]) == 20


def test_parity_rejects_nonfinite_and_discrete_mismatch():
    comparison = load("compare_filter_repair_campaign")
    for before, after in (([float("nan")], [0.]), ([1], [2]), ([0.5], [0.6]), ([True], [False])):
        with pytest.raises(ValueError):
            comparison.compare_values(before, after)
    assert comparison.compare_values([[1.0, 2.0]], [[1.0, 2.0 + 1e-12]]) < 2e-12
