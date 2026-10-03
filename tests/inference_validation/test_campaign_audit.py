"""Synthetic accounting/availability regressions; no sampler is launched."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_inference_validation_campaign.py"
SPEC = importlib.util.spec_from_file_location("campaign_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    design = {"design_id": "sbc", "device": "gpu", "engine": "sbc",
              "scenario": {"route": "ordinary"}, "replications": 2, "rank_draws": 2}
    result = write(root / "profile/sbc/result.json", {
        "design_identity": "design-1", "execution_status": "complete",
        "assessment": {"finding": "calibration_incomplete"}})
    index = {"source": {"identity": audit.canonical_hash({})},
             "plan": {"jobs": [{"design": design, "identity": "design-1"}]},
             "jobs": {"sbc": {"status": "complete", "result": str(result),
                              "result_sha256": audit.file_hash(result),
                              "attempts": [{"elapsed_seconds": 3., "status": "failed"},
                                           {"elapsed_seconds": 7., "status": "complete"}]}}}
    write(root / "profile/run_index.json", index)
    write(root / "terminal_audit.json", {
        "campaign_allowance_seconds": {"gpu": 30., "cpu_reference": 10.},
        "indexed_worker_seconds": {"gpu": 10., "cpu_reference": 0.},
        "profiles": [{"profile": "profile"}]})
    write(root / "profile/sbc/dataset-0000.json", {
        "dataset_id": 0, "status": "missing_fit", "ranks": {}, "fit_outputs_active": [[1.]],
        "fits": [{"member_id": "candidate"}, {"reason": "no_member_in_declared_L_group"}]})
    write(root / "profile/sbc/dataset-0001/fit-0000/pipeline.json", {"members": []})
    return root, index


def test_retries_are_charged_and_partial_datasets_are_not_completed(campaign, tmp_path):
    root, _ = campaign
    result = audit.audit_campaign(root, repo=tmp_path)
    assert result["invalid_artifacts"] == []
    assert result["indexed_worker_seconds"] == {"gpu": 10., "cpu_reference": 0.}
    assert result["remaining_seconds"] == {"gpu": 20., "cpu_reference": 10.}
    sbc = result["full_procedure_sbc"][0]
    assert sbc["dataset_counts"] == {"complete": 0, "missing_fit": 1,
                                     "started_without_dataset_record": 1, "unstarted": 0}
    assert sbc["recorded_missing_fit_reasons"] == {"no_member_in_declared_L_group": 1}
    assert not sbc["calibration_complete"]
    assert not sbc["pooled_with_other_attempts"]


def test_running_reservation_is_charged_conservatively(campaign, tmp_path):
    root, index = campaign
    index["jobs"]["sbc"].update(status="running", reserved_seconds=8.)
    write(root / "profile/run_index.json", index)
    result = audit.audit_campaign(root, repo=tmp_path)
    assert result["indexed_worker_seconds"]["gpu"] == 18.
    assert any("running" in row["reason"] for row in result["invalid_artifacts"])


def test_live_ledger_charges_cancelled_attempt_and_releases_unstarted(campaign):
    root,index=campaign
    index["jobs"]["sbc"]["status"]="cancelled"
    index["cancellation"]={"reason":"baseline mismatch"}
    index["plan"]["jobs"].append({"design":{"design_id":"not-run","device":"gpu","budget_seconds":999}})
    write(root/"profile/run_index.json",index)
    result=audit.campaign_status(root,{"gpu":30.,"cpu_reference":10.},verify_artifacts=True)
    assert result["indexed_worker_seconds"]["gpu"]==10.
    assert result["reserved_seconds"]["gpu"]==0.
    assert result["uncommitted_seconds"]["gpu"]==20.
    assert not result["invalid_artifacts"]


def test_live_ledger_reserves_queued_jobs_while_coordinator_is_active(campaign):
    root,index=campaign
    index["jobs"]["sbc"].update(status="running",reserved_seconds=8.)
    index["plan"]["jobs"].append({"design":{"design_id":"not-run","device":"gpu","budget_seconds":5.}})
    write(root/"profile/run_index.json",index)
    result=audit.campaign_status(root,{"gpu":30.,"cpu_reference":10.})
    assert result["indexed_worker_seconds"]["gpu"]==10.
    assert result["reserved_seconds"]["gpu"]==13.
    assert result["uncommitted_seconds"]["gpu"]==7.


def test_cancelled_suite_keeps_unstarted_sbc_denominator(campaign, tmp_path):
    root, index = campaign
    index["cancellation"] = {"reason": "wrong baseline"}
    design = {**index["plan"]["jobs"][0]["design"], "design_id": "unstarted"}
    index["plan"]["jobs"].append({"design": design, "identity": "design-2"})
    write(root / "profile/run_index.json", index)
    result = audit.audit_campaign(root, repo=tmp_path)
    assert result["invalid_artifacts"] == []
    assert result["indexed_worker_seconds"]["gpu"] == 10.
    assert result["profiles"][0]["statuses"] == {"complete": 1, "not_run": 1}
    missing = result["full_procedure_sbc"][1]
    assert missing["process_status"] == "not_run"
    assert missing["planned_datasets"] == 2
    assert missing["dataset_counts"]["unstarted"] == 2
    assert not missing["calibration_complete"]


def test_overhead_counts_toward_budget_without_becoming_worker_time(campaign, tmp_path):
    root, _ = campaign
    overhead = {"gpu": 0., "cpu_reference": 11.}
    live = audit.campaign_status(root, {"gpu": 30., "cpu_reference": 10.},
                                 overhead_seconds=overhead)
    assert live["budget_exceeded"]
    assert live["indexed_worker_seconds"]["cpu_reference"] == 0.
    assert live["total_charged_seconds"]["cpu_reference"] == 11.
    terminal = audit.read_json(root / "terminal_audit.json")
    terminal["overhead_charged_seconds"] = overhead
    write(root / "terminal_audit.json", terminal)
    result = audit.audit_campaign(root, repo=tmp_path)
    assert result["invalid_artifacts"] == []
    assert result["budget_exceeded"]
    assert result["remaining_seconds"]["cpu_reference"] == -1.


def test_corrupt_result_tensor_and_missing_index_fail_audit(campaign, tmp_path):
    root, index = campaign
    write(Path(index["jobs"]["sbc"]["result"]), {"design_identity": "wrong"})
    path = root / "profile/x.tensor"
    path.write_bytes(b"damaged")
    write(path.with_suffix(".tensor.json"), {"sha256": "wrong"})
    terminal = audit.read_json(root / "terminal_audit.json")
    terminal["profiles"].append({"profile": "missing-profile"})
    write(root / "terminal_audit.json", terminal)
    result = audit.audit_campaign(root, repo=tmp_path)
    assert len(result["invalid_artifacts"]) == 3
    assert result["tensor_checksums_checked"] == 1


def test_complete_dataset_requires_every_fit(campaign, tmp_path):
    root, _ = campaign
    path = root / "profile/sbc/dataset-0000.json"
    record = audit.read_json(path)
    record.update(status="complete", ranks={"parameter_0": 1})
    write(path, record)
    assert audit.audit_campaign(root, repo=tmp_path)["invalid_artifacts"]
    record.update(fits=[{"member_id": "a"}, {"member_id": "b"}], fit_outputs_active=[[1.], [2.]])
    write(path, record)
    result = audit.audit_campaign(root, repo=tmp_path)
    assert not result["invalid_artifacts"]
    assert result["full_procedure_sbc"][0]["dataset_counts"]["complete"] == 1
    assert not result["full_procedure_sbc"][0]["calibration_complete"]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "2"])
def test_invalid_worker_seconds_rejected(value):
    with pytest.raises(ValueError):
        audit.checked_seconds(value)


def test_cli_does_not_import_frameworks_or_overwrite_evidence(campaign, tmp_path):
    root, _ = campaign
    output = tmp_path / "new/preflight.json"
    program = ("import runpy,sys; "
               f"m=runpy.run_path({str(SCRIPT)!r}); "
               f"assert m['main']([{str(root)!r},'--output',{str(output)!r}])==0; "
               "assert not {'tensorflow','tensorflow_probability','numpy'} & sys.modules.keys()")
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.main([str(root), "--output", str(output)])
    assert output.read_bytes() == saved
