"""Recovery and resource guards for the KSC campaign (CPU diagnostics only)."""
import json
import sys

import pytest

from scripts import prepare_surrogate_hmc_ksc as runner
from bayesfilter.inference.surrogate_hmc_dctr_selection import MULTIPLIERS, issue_selection


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "bayesfilter/engine.py"
    source.parent.mkdir()
    source.write_text("# fixed numerical engine\n")
    hashes = {"bayesfilter/engine.py": runner.sha(source)}
    scope = dict(model="M03-KSC-T2", attempt=1)
    curves, data = {}, {}
    for partition in ("calibration", "validation"):
        directory = tmp_path / partition
        directory.mkdir()
        data[partition] = directory / "data.json"
        runner.save(data[partition], dict(model_id=scope["model"], attempt=1, partition=partition))
        arms = []
        for i, multiplier in enumerate(MULTIPLIERS):
            path = directory / f"arm-{i}.json"
            runner.save(path, dict(multiplier=multiplier, situations=dict(nominal={}, stress={})))
            arms.append(dict(path=str(path), sha256=runner.sha(path)))
        pairs = [dict(non_harm_passed=True, normalized_maximum_change=1e-5, tolerance=1e-3,
                      healthy_nominal=True, stress_safe=True, no_new_stress_failure=True)] * 3
        curves[partition] = directory / "curve.json"
        runner.save(curves[partition], dict(scope=scope, source_hashes=hashes, data_sha256=runner.sha(data[partition]),
                                           pairs=pairs, arms=arms))
    selected = issue_selection(output=tmp_path / "selection.json", scope=scope, evidence_paths=curves,
                               partition_data_paths=data, source_hashes=hashes)
    runner.save(tmp_path / "run-manifest.json", dict(source_hashes=hashes))
    runner.save(tmp_path / "report.json", dict(status="INFRASTRUCTURE_OR_RESOURCE_FAILURE", exit_code=-24,
                                              selected=selected, scope=scope))
    runner.save(tmp_path / "accounting.json", dict(exit_code=-24))
    return selected


def test_completed_selection_survives_later_cpu_failure(tmp_path, monkeypatch):
    selected = fixture(tmp_path, monkeypatch)
    assert runner.check_selection(tmp_path) == selected
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "INFRASTRUCTURE_OR_RESOURCE_FAILURE"


@pytest.mark.parametrize("damage", ["arm", "source", "incomplete", "unclassified"])
def test_recovery_rejects_damaged_or_unclassified_evidence(tmp_path, monkeypatch, damage):
    fixture(tmp_path, monkeypatch)
    if damage == "arm":
        (tmp_path / "calibration/arm-0.json").write_text("{}")
    elif damage == "source":
        (tmp_path / "bayesfilter/engine.py").write_text("# drift\n")
    else:
        report = json.loads((tmp_path / "report.json").read_text())
        if damage == "incomplete":
            report["selected"] = None
        else:
            report["exit_code"] = 1
        runner.save(tmp_path / "report.json", report)
    with pytest.raises(ValueError):
        runner.check_selection(tmp_path)


def test_wall_watchdog_stops_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    code, record = runner.run_bounded_worker([sys.executable, "-c", "import time; time.sleep(30)"],
        log_path=tmp_path / "log", cpu_budget=10, wall_budget=.1)
    assert code != 0 and record["stop_reason"] == "WALL_WATCHDOG"
    assert 1 <= len(record["cpu_affinity"]) <= 4


def test_cpu_watchdog_stops_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    code, record = runner.run_bounded_worker([sys.executable, "-c", "while True: pass"],
        log_path=tmp_path / "log", cpu_budget=4.2, wall_budget=5)
    assert code != 0 and record["stop_reason"] == "CPU_WATCHDOG"
