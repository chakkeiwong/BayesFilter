"""Terminal CPU audit of the corrected GPU deployment's bytes, calls and exports."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
directory = Path(__file__).resolve().parent
run = directory / "gpu-deployment-02/run"
read = lambda p: json.loads(p.read_text())
manifest = read(run / "manifest.json")
source = Path(manifest["command"][1]).resolve().parents[1]
sys.path.insert(0, str(source))
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCTuningCandidateSetController, _sha256, _json_native_sha256)
from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig
from bayesfilter.inference.hmc_acceptance_trials import work_seed
from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence

started = time.monotonic()
supervisor = read(run / "supervisor.json")
summary = read(run / "result.json")
assert manifest["status"] == supervisor["status"] == summary["status"] == "completed"
assert supervisor["exit_code"] == 0
assert manifest["memory_policy"]["all_physical_devices_memory_growth"] is True
assert manifest["use_xla"] is True
for name, expected in manifest["source_and_input_sha256"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
tuning = run / "search/tuning"
checkpoint = read(tuning / "tuning_checkpoint.json")
assert checkpoint.pop("content_hash") == _sha256(checkpoint)
spec = read(tuning / "execution_spec.json")
assert _sha256(spec["execution"]) == spec["binding_hash"] == checkpoint["binding_hash"]
payload = checkpoint["result"]
assert payload["result_hash"] == _sha256({k: v for k, v in payload.items() if k != "result_hash"})
controller = HMCTuningCandidateSetController.from_result_payload(payload)
result = controller.result()
execution = spec["execution"]
evidence = {}
for digest in checkpoint["numerical_evidence_hashes"]:
    row = read(tuning / "numerical_evidence" / (digest + ".json"))
    assert _json_native_sha256(row) == digest
    assert row["binding_hash"] == spec["binding_hash"]
    evidence[digest] = row
assert not checkpoint["partial_chunks"]
view = SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(execution["config"]),
    scope=controller.scope, _runtime=execution["runtime_policy"], _spec=execution,
    _evidence=evidence, _partial={})
view.work_seed = lambda work: work_seed(view, work)
audit = audit_native_evidence(result, view)
assert audit["attempted_chunk_count"] == audit["returned_chunk_count"] == 32
assert audit["charged_gradient_work"] == audit["expected_completed_gradient_work"] == 32768
assert len(audit["verified_candidate_ids"]) == 1
assert audit["verified_candidate_ids"] == summary["verified_candidate_ids"]
stage_statistics = {}
for row in evidence.values():
    acceptance = row["analysis"]["acceptance_evidence"]
    assert acceptance["invalid_trials"] == 0
    assert acceptance["completed_trials"] == 16
    assert not row["analysis"]["hard_vetoes"]
    assert acceptance["statistics"]["acceptance_decision"] == "passed"
    stage_statistics[row["work"]["stage"]] = acceptance["statistics"]
assert set(stage_statistics) == {"measurement", "verification"}
members = []
for path in (run / "search/members").glob("*-member.json"):
    member = read(path)
    assert member.pop("content_hash") == _json_native_sha256(member)
    bundle = read(path.parent / member["evidence_bundle"]["path"])
    content = bundle.pop("content_hash")
    assert content == _json_native_sha256(bundle) == member["evidence_bundle"]["content_hash"]
    assert bundle["binding_hash"] == spec["binding_hash"]
    assert bundle["candidate_set_result"]["verified_candidate_ids"] == audit["verified_candidate_ids"]
    members.append(member["candidate_id"])
assert sorted(members) == sorted(audit["verified_candidate_ids"])
record = dict(status="passed", source_commit=manifest["git_commit"],
    source_and_input_hashes_checked=len(manifest["source_and_input_sha256"]),
    native_work=audit, stage_statistics=stage_statistics, checked_export_candidate_ids=members,
    gpu_supervisor_seconds=supervisor["elapsed_seconds"],
    cpu_diagnostic_gpu_hidden=True, runtime_replay_performed_by_audit=False,
    posterior_assessed=False, adaptation_assessed=False, method_ranking=None,
    wall_seconds=time.monotonic()-started, command=[sys.executable, str(Path(__file__).resolve())],
    environment=sys.executable, audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    plan="docs/plans/bayesfilter-modern-hmc-adaptive-consumer-repair-2026-10-12.md")
with (directory / "deployment-terminal-audit-02.json").open("x") as out:
    out.write(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: record[k] for k in ("status", "source_and_input_hashes_checked", "gpu_supervisor_seconds", "wall_seconds")}))
