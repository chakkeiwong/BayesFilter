"""CPU-only rejection checks for the stricter runtime-repair evidence contract."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("pairing,request_count", [("block", 1), ("call", 24)])
def test_timing_pairing_preserves_complete_experiment(monkeypatch, pairing, request_count):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    specification = importlib.util.spec_from_file_location(
        "runtime_pairing", scripts / "diagnose_sir_postmerge_runtime.py")
    runtime = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runtime)
    datasets = [263043001, 263043004, 263043021, 263043052]
    arms = ["plain", "conditional", "physical"]
    requests = runtime.paired_requests(datasets, arms, pairing)
    assert len(requests) == request_count
    realized = [(dataset, arm, direction) for request in requests
                for dataset in request["datasets"] for arm in request["arms"]
                for direction in request["directions"]]
    assert realized == [(dataset, arm, direction) for dataset in datasets
                        for arm in arms for direction in (-1, 0)]


@pytest.mark.parametrize("failure", [None, "raw_moment", "flag_change"])
def test_repair_checks_cannot_inherit_looser_replay_acceptance(monkeypatch, failure):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    specification = importlib.util.spec_from_file_location(
        "runtime_repair_reporting", scripts / "analyze_sir_runtime_repair.py")
    reporting = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reporting)
    baseline, candidate = Path("baseline"), Path("candidate")

    def original_comparison(*args, **kwargs):
        return dict(equivalence_passed=True, holdout=dict(changed_checks=[]),
            directional=dict(changed_checks=["force_valid"] if failure == "flag_change" else []))

    def rows(path, directional):
        value = 3e-8 if failure == "raw_moment" and path.parent == candidate else 0.
        return {(123, "plain", 0): dict(raw_blocks=[value], raw_estimate=[value])}

    monkeypatch.setattr(reporting, "verify_pair", original_comparison)
    monkeypatch.setattr(reporting, "read_rows", rows)
    report = reporting.verify_repaired_replay(baseline, candidate)
    assert report["equivalence_passed"]
    assert report["repair_acceptance"] == (failure is None)
    if failure == "raw_moment":
        assert not report["repair_raw_moment_checks"]["holdout"]["passed"]
    if failure == "flag_change":
        assert not report["repair_raw_moment_checks"]["directional"]["exact_flags"]
