"""CPU-only process-accounting regression for interrupted SIR supervision."""

import json
import os

import pytest

from scripts.run_sir_neural_kdm_study import recover_completed_study, recover_screen


def saved_screen(path):
    names = ["evaluation-probe"] + [
        f"screen-k{components}-w{width}-beta{beta}"
        for components, width in ((2, 16), (4, 32)) for beta in (.5, .9)]
    records = []
    for name in names:
        target = path / name
        target.mkdir()
        (target / "result.json").write_text(json.dumps(dict(
            status="probe_complete" if name == "evaluation-probe" else "family_complete",
            source_check=dict(unchanged=True))))
        records.append(dict(name=name, status="complete", returncode=0,
                            charged_seconds=10, wall_seconds=9.5, pid=999999999,
                            start_utc="1970-01-01T00:01:40Z"))
    records[-1].update(status="running", returncode=None, charged_seconds=0)
    ledger = dict(allowance_seconds=1000., charged_seconds=40., runs=records)
    (path / "ledger.json").write_text(json.dumps(ledger))
    return ledger


def test_recovery_preserves_evidence_and_conservatively_charges_orphan(tmp_path):
    saved_screen(tmp_path)
    before = (tmp_path / "ledger.json").read_bytes()
    records, charged = recover_screen(tmp_path, 1000., now=125.1)
    assert charged == 66
    assert records[-1]["charged_seconds"] == 26
    assert records[-1]["returncode"] is None
    assert records[-1]["status"] == "complete"
    assert (tmp_path / "ledger.json").read_bytes() == before


@pytest.mark.parametrize("defect", ["alive", "missing", "source", "incomplete", "failed", "allowance"])
def test_recovery_rejects_duplicate_or_invalid_evidence(tmp_path, defect):
    ledger = saved_screen(tmp_path)
    result = tmp_path / ledger["runs"][-1]["name"] / "result.json"
    allowance = 1000.
    if defect == "alive":
        ledger["runs"][-1]["pid"] = os.getpid()
    elif defect == "missing":
        result.unlink()
    elif defect == "source":
        result.write_text(json.dumps(dict(status="family_complete", source_check=dict(unchanged=False))))
    elif defect == "incomplete":
        result.write_text(json.dumps(dict(status="invalid_training_candidate", source_check=dict(unchanged=True))))
    elif defect == "failed":
        ledger["runs"][0].update(status="failed", returncode=1)
    elif defect == "allowance":
        allowance = 1200.
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        recover_screen(tmp_path, allowance, now=125.1)


def saved_completed_study(path):
    ledger = saved_screen(path)
    ledger["runs"][-1].update(status="complete", returncode=0, charged_seconds=10)
    ledger["recovered_runs"] = [dict(record, evidence_root=str(path)) for record in ledger["runs"]]
    ledger["runs"] = []
    for seed in (263044001, 263044002):
        name = f"rung128-seed{seed}"
        target = path / name
        target.mkdir()
        (target / "result.json").write_text(json.dumps(dict(status="fit_complete",
            source_check=dict(unchanged=True))))
        ledger["runs"].append(dict(name=name, status="complete", charged_seconds=20))
    name = "rung512-seed263044001"
    target = path / name
    target.mkdir()
    (target / "failure.json").write_text(json.dumps(dict(
        error="TimeoutError('Neural KDM campaign process budget exhausted')",
        source_check=dict(unchanged=True))))
    ledger["runs"].append(dict(name=name, status="failed", charged_seconds=30))
    ledger["charged_seconds"] = 120
    (path / "ledger.json").write_text(json.dumps(ledger))
    return ledger


def test_finalization_preserves_failed_charge_and_excludes_failed_checkpoints(tmp_path):
    saved_completed_study(tmp_path)
    before = (tmp_path / "ledger.json").read_bytes()
    records, charged, completed = recover_completed_study(tmp_path, 1000.)
    assert charged == 120
    assert len(records) == 8
    assert len(completed) == 7
    assert all(not name.startswith("rung512") for name in completed)
    assert (tmp_path / "ledger.json").read_bytes() == before


@pytest.mark.parametrize("defect", ["source", "incomplete", "duplicate", "accounting", "failure", "allowance"])
def test_finalization_rejects_invalid_evidence(tmp_path, defect):
    ledger = saved_completed_study(tmp_path)
    allowance = 1000.
    if defect in ("source", "incomplete"):
        path = tmp_path / "rung128-seed263044001" / "result.json"
        result = json.loads(path.read_text())
        if defect == "source":
            result["source_check"]["unchanged"] = False
        else:
            result["status"] = "invalid_training_candidate"
        path.write_text(json.dumps(result))
    elif defect == "duplicate":
        ledger["runs"].append(dict(ledger["runs"][0]))
    elif defect == "accounting":
        ledger["charged_seconds"] -= 1
    elif defect == "failure":
        path = tmp_path / "rung512-seed263044001" / "failure.json"
        result = json.loads(path.read_text())
        result["error"] = "ValueError('Invalid gradient')"
        path.write_text(json.dumps(result))
    else:
        allowance = 1200.
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        recover_completed_study(tmp_path, allowance)
