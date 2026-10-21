"""Read-only CPU audit of archived GPU bytes/work; no runtime replay authority."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(ROOT))
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCTuningCandidateSetController, _sha256, _json_native_sha256)
from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig
from bayesfilter.inference.hmc_acceptance_trials import work_seed
from bayesfilter.inference.hmc_acceptance_statistics import replicated_qualification_attainability
from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence

started = time.monotonic()
directory = Path(__file__).parent
run = directory / "gpu-deployment-01/run"
tuning = run / "search/tuning"
read = lambda p: json.loads(p.read_text())
manifest = read(run / "manifest.json")
for name, expected in manifest["source_and_input_sha256"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
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
assert not checkpoint["partial_chunks"]  # This terminal attempt has complete trials only.
view = SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(execution["config"]),
    scope=controller.scope, _runtime=execution["runtime_policy"], _spec=execution,
    _evidence=evidence, _partial={})
view.work_seed = lambda work: work_seed(view, work)
audit = audit_native_evidence(result, view)
assert audit["attempted_chunk_count"] == audit["returned_chunk_count"] == 8
assert audit["charged_gradient_work"] == audit["expected_completed_gradient_work"] == 8192
assert audit["verified_candidate_ids"] == []
p = view.config.acceptance_policy
old = replicated_qualification_attainability(policy=p, evidence_rungs=controller.config.evidence_rungs,
                                             jit_compile=False)
corrected = replicated_qualification_attainability(policy=replace(p, base_repetitions=16, max_repetitions=16),
    evidence_rungs=controller.config.evidence_rungs, jit_compile=False)
assert old["proven_impossible"] and not corrected["proven_impossible"]
record = dict(status="passed", audited_source_commit=manifest["git_commit"],
    source_and_input_hashes_checked=len(manifest["source_and_input_sha256"]),
    native_work=audit, failed_design_preflight=old, proposed_design_preflight=corrected,
    cpu_diagnostic_gpu_hidden=True, runtime_replay_performed=False, method_ranking=None,
    wall_seconds=time.monotonic()-started, command=[sys.executable, str(Path(__file__).resolve())],
    environment=sys.executable, audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    plan="docs/plans/bayesfilter-modern-hmc-adaptive-consumer-repair-2026-10-12.md")
with (directory / "deployment-terminal-audit-01.json").open("x") as stream:
    stream.write(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: record[k] for k in ("status", "source_and_input_hashes_checked", "wall_seconds")}))
