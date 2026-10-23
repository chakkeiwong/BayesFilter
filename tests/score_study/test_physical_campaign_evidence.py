"""CPU evidence-integrity regressions at the executed campaign endpoint."""

import copy
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.score_study import physical_proposal_diagnostic_campaign as campaign_module
from bayesfilter.score_study.physical_campaign_evidence import (
    HEURISTIC_ARMS, PARTICLE_ARMS, TEST_PARTICLE_SEED_BASE,
)
from bayesfilter.score_study.run_source_snapshot import capture_source_snapshot, verify_source_snapshot


def _records():
    campaign = object.__new__(campaign_module.PhysicalCampaign)
    campaign.partitions = dict(test=[101, 102])
    rows = []
    for index, dataset in enumerate(campaign.partitions["test"]):
        common = dict(dataset=dataset, regime="low" if index == 0 else "high", loss=1., valid=True)
        rows.extend(dict(common, arm=arm, replicate=None) for arm in HEURISTIC_ARMS)
        rows.extend(dict(common, arm=arm, replicate=replicate,
                         particle_seed=TEST_PARTICLE_SEED_BASE + 2 * index + replicate)
                    for arm in PARTICLE_ARMS for replicate in range(2))
    return campaign, rows


def test_complete_paired_grid_is_accepted():
    campaign, rows = _records()
    summary = campaign.summarize(rows)
    assert summary["integrity"] == dict(complete=True, paired=True, rows=34,
                                        datasets=2, particle_replications=2)
    assert len(summary["comparisons"]) == 42
    assert all(row["valid"] and row["mean_difference"] == 0. for row in summary["comparisons"])


@pytest.mark.parametrize("defect", ["missing", "duplicate", "seed", "extra", "field",
                                    "nan", "infinite", "invalid", "regime", "replicate"])
def test_malformed_grid_rejected_before_aggregation(defect):
    campaign, rows = _records()
    target = next(row for row in rows if row["arm"] == "csmc_trained")
    if defect == "missing":
        rows.remove(target)
    elif defect == "duplicate":
        rows.append(copy.copy(target))
    elif defect == "seed":
        target["particle_seed"] += 1
    elif defect == "extra":
        target["arm"] = "undeclared"
    elif defect == "field":
        del target["valid"]
    elif defect in ("nan", "infinite"):
        target["loss"] = float("nan" if defect == "nan" else "inf")
    elif defect == "invalid":
        target["valid"] = False
    elif defect == "regime":
        target["regime"] = "high"
    elif defect == "replicate":
        target["replicate"] = True
    with pytest.raises(ValueError):
        campaign.summarize(rows)


def test_source_snapshot_captures_omitted_and_untracked_dependencies(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    package = repository / "bayesfilter" / "highdim"
    package.mkdir(parents=True)
    dependency = package / "higher_moment_contract_e.py"
    dependency.write_text("value = 1\n")
    untracked = package / "new_proposal.py"
    untracked.write_text("value = 2\n")
    commands = []

    def git_output(command, **options):
        commands.append(command)
        return "test-commit\n" if options.get("text") else b"dirty-source-record\n"

    monkeypatch.setattr("bayesfilter.score_study.run_source_snapshot.subprocess.check_output", git_output)
    snapshot = capture_source_snapshot(repository, tmp_path / "snapshot")
    assert len(snapshot["source_hashes"]) == 2
    assert (tmp_path / "snapshot" / dependency.relative_to(repository)).read_bytes() == dependency.read_bytes()
    assert verify_source_snapshot(repository, snapshot)["unchanged"]
    dependency.write_text("value = 3\n")
    assert verify_source_snapshot(repository, snapshot)["changed"] == [str(dependency.relative_to(repository))]
    assert not verify_source_snapshot(repository, snapshot)["unchanged"]
    assert any("diff" in command for command in commands)
    with pytest.raises(FileExistsError):
        capture_source_snapshot(repository, tmp_path / "snapshot")


def test_campaign_captures_sources_before_device_initialization(tmp_path, monkeypatch):
    calls = []

    def capture(repository, destination):
        calls.append((repository, destination))
        return {}

    def stop_before_device(*arguments, **options):
        raise RuntimeError("intentional CPU wiring stop")

    monkeypatch.setattr(campaign_module, "capture_source_snapshot", capture)
    monkeypatch.setattr(campaign_module, "configure_tensorflow_gpu_memory_growth", stop_before_device)
    with pytest.raises(RuntimeError, match="intentional CPU wiring stop"):
        campaign_module.PhysicalCampaign(SimpleNamespace(output=tmp_path / "run"))
    assert len(calls) == 1
    assert calls[0][1] == tmp_path / "run" / "source"
