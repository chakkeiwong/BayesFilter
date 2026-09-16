"""Activate the tested eight-sweep repair with the original campaign budget."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as element_tree


ROOT = Path(__file__).resolve().parents[6]
DIAGNOSTIC = Path(__file__).resolve().parent
CAMPAIGN = DIAGNOSTIC.parent
sys.path.insert(0, str(ROOT))
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Migration activation requires intentionally hidden GPUs")
spec = importlib.util.spec_from_file_location("eight_sweep_activation", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
gpu_root = CAMPAIGN / "launches/eight-sweep-gpu-validation-5c70410169"
manifest_path = gpu_root / "worker/run_manifest.json"
manifest = json.loads(manifest_path.read_bytes())
gpu_summary = json.loads((gpu_root / "summary.json").read_bytes())
current = runner.source_hashes()
if (manifest["status"] != "completed" or manifest.get("passed") is not True
        or manifest["sources"] != current
        or not manifest["memory_policy"]["all_physical_devices_memory_growth"]
        or gpu_summary["result"]["status"] != "completed"
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != gpu_summary["result"]["required_artifact_sha256"]):
    raise RuntimeError("GPU fixed-bank evidence is incomplete or changed")
regression = DIAGNOSTIC / "eight-sweep-regression-r1.xml"
validation_path = DIAGNOSTIC / "eight-sweep-validation-a8df05dcf2/validation.json"
validation = json.loads(validation_path.read_bytes())
suites = list(element_tree.parse(regression).getroot().iter("testsuite"))
if (current != validation["current_sources"] or not suites
        or any(int(suite.attrib[field]) for suite in suites for field in ("failures", "errors", "skipped"))
        or hashlib.sha256(regression.read_bytes()).hexdigest() != validation["artifact_sha256"][regression.name]):
    raise RuntimeError("CPU repair validation is incomplete or stale")
namespace = "numerical-repairs/eigh-refinement-r2"
with (CAMPAIGN / ".coordinator.lock").open("a+b") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    start_path = CAMPAIGN / "campaign-start.json"
    ledger_path = CAMPAIGN / "campaign_budget_ledger.json"
    start_bytes, ledger_bytes = start_path.read_bytes(), ledger_path.read_bytes()
    start, ledger = json.loads(start_bytes), json.loads(ledger_bytes)
    previous_path = CAMPAIGN / runner.SOURCE_MIGRATION_FILENAME
    previous = json.loads(previous_path.read_bytes())
    runner._validate_source_migration(start, CAMPAIGN, previous["to_sources"])
    changed = sorted(path for path in set(previous["to_sources"]) | set(current)
                     if previous["to_sources"].get(path) != current.get(path))
    if (changed != [runner.NUMERICAL_CORE_PATH]
            or previous.get("execution_namespace") != "numerical-repairs/eigh-refinement-r1"
            or previous.get("serializer_only") is not False):
        raise RuntimeError("Expected the preserved four-sweep numerical predecessor")
    if ledger["reserved_seconds"] != 0 or any(row["status"] == "running" for row in ledger["attempts"]):
        raise RuntimeError("Live reservations must settle before migration")
    if hashlib.sha256(runner.PLAN.read_bytes()).hexdigest() != start["plan_hash"]:
        raise RuntimeError("Governing plan changed")
    if (CAMPAIGN / namespace).exists():
        raise RuntimeError("Eight-sweep namespace must be fresh")
    output = DIAGNOSTIC / ("eight-sweep-migration-" + uuid.uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    shutil.copy2(previous_path, output / "previous-source-migration.json")
    shutil.copy2(__file__, output / Path(__file__).name)
    for relative in (str(runner.SCRIPT.relative_to(ROOT)), runner.NUMERICAL_CORE_PATH,
                     "tests/test_principal_sqrt_eigen_refinement_tf.py", "tests/test_ssl_lstm_q20_phase9b_recovery_runtime.py"):
        destination = output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    migration = {
        "schema": runner.SOURCE_MIGRATION_SCHEMA, "migration_id": runner.NUMERICAL_MIGRATION_ID,
        "from_sources": start["sources"], "to_sources": current,
        "changed_paths": sorted(path for path in set(start["sources"]) | set(current)
                                if start["sources"].get(path) != current.get(path)),
        "plan_hash": start["plan_hash"], "claim_boundary": runner.CLAIM_BOUNDARY,
        "scientific_contract_unchanged": True, "serializer_only": False,
        "execution_namespace": namespace, "created_at_unix": time.time(),
        "previous_migration_archive": str(output / "previous-source-migration.json"),
        "repair_plan": str(ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md"),
        "cpu_regression": str(regression), "cpu_regression_sha256": hashlib.sha256(regression.read_bytes()).hexdigest(),
        "candidate_gpu_manifest": str(manifest_path), "candidate_gpu_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "numerical_repair": "binary64_batched_jacobi_eight_sweeps_unchanged_residual_SPD_derivative_and_health_checks",
        "fresh_scopes": list(runner.ARMS), "old_tuning_and_streams_reused": False,
    }
    runner.durable_json(output / "source-migration.json", migration)
    runner.durable_json(previous_path, migration)
    restored, same_ledger = runner.initialize_campaign(CAMPAIGN, resume=True)
    execution, effective = runner.prepare_execution_namespace(CAMPAIGN, restored, current, migration)
    runner.verify_binding(effective, execution)
    verification = runner.verify_campaign_bundles(CAMPAIGN)
    if start_bytes != start_path.read_bytes() or ledger_bytes != ledger_path.read_bytes():
        raise RuntimeError("Migration changed original start or budget")
    receipt = {
        "status": "EIGHT_SWEEP_NUMERICAL_MIGRATION_READY", "created_at_unix": time.time(),
        "command": [sys.executable, *sys.argv], "gpu_intentionally_hidden": True,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "execution_root": str(execution), "shared_ledger": str(same_ledger.path),
        "consumed_gpu_seconds": ledger["consumed_seconds"], "remaining_gpu_seconds": same_ledger.remaining_seconds(),
        "original_start_sha256": hashlib.sha256(start_bytes).hexdigest(),
        "unchanged_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "bundle_verification": verification,
        "cpu_tests_passed": sum(int(suite.attrib["tests"]) for suite in suites),
        "source_snapshot": str(output / "sources"), "migration": str(previous_path),
    }
    runner.durable_json(output / "activation-result.json", receipt)
    print(json.dumps({"output": str(output), **receipt}, indent=2))
