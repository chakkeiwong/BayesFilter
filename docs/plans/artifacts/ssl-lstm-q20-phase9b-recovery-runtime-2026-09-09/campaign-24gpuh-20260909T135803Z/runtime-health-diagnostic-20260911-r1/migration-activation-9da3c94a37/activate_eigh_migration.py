"""Record the tested numerical repair without changing the campaign allocation."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as element_tree
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
DIAGNOSTIC = Path(__file__).resolve().parent
CAMPAIGN = DIAGNOSTIC.parent
WRAPPER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py"


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("migration preparation is intentionally CPU-only")
    sys.path.insert(0, str(ROOT))
    specification = importlib.util.spec_from_file_location("phase9b_numerical_migration", WRAPPER)
    runner = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = runner
    specification.loader.exec_module(runner)
    regression = DIAGNOSTIC / "numerical-migration-regression-r2.xml"
    suites = list(element_tree.parse(regression).getroot().iter("testsuite"))
    if not suites or any(int(suite.attrib[field]) for suite in suites for field in ("failures", "errors", "skipped")):
        raise RuntimeError("complete passing CPU regression evidence required")
    candidate_root = CAMPAIGN / "launches/candidate-diagnostic-a7e029690b/strict"
    candidate = json.loads((candidate_root / "run_manifest.json").read_bytes())
    if candidate["status"] != "completed" or not candidate["memory_policy"]["all_physical_devices_memory_growth"]:
        raise RuntimeError("passing integrated GPU validation required")
    current = runner.source_hashes()
    changed_since_gpu = sorted(path for path in set(current) | set(candidate["sources"])
                               if current.get(path) != candidate["sources"].get(path))
    if changed_since_gpu != [str(WRAPPER.relative_to(ROOT))]:
        raise RuntimeError("numerical source changed after integrated GPU validation")
    with (CAMPAIGN / ".coordinator.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        start_path = CAMPAIGN / "campaign-start.json"
        ledger_path = CAMPAIGN / "campaign_budget_ledger.json"
        start_bytes, ledger_bytes = start_path.read_bytes(), ledger_path.read_bytes()
        start, ledger = json.loads(start_bytes), json.loads(ledger_bytes)
        if ledger["reserved_seconds"] != 0 or any(row["status"] == "running" for row in ledger["attempts"]):
            raise RuntimeError("live campaign reservations must settle before numerical migration")
        previous_path = CAMPAIGN / runner.SOURCE_MIGRATION_FILENAME
        previous = json.loads(previous_path.read_bytes())
        if previous["migration_id"] != runner.SOURCE_MIGRATION_ID or previous["serializer_only"] is not True:
            raise RuntimeError("expected the preserved serializer-only predecessor")
        if previous["from_sources"] != start["sources"]:
            raise RuntimeError("original source binding changed")
        namespace = "numerical-repairs/eigh-refinement-r1"
        if (CAMPAIGN / namespace).exists():
            raise RuntimeError("numerical namespace must be fresh for first activation")
        output = DIAGNOSTIC / f"migration-activation-{uuid.uuid4().hex[:10]}"
        output.mkdir()
        shutil.copy2(previous_path, output / "previous-source-migration.json")
        shutil.copy2(Path(__file__), output / Path(__file__).name)
        for path in (WRAPPER, ROOT / runner.NUMERICAL_CORE_PATH,
                     ROOT / "tests/test_ssl_lstm_q20_phase9b_recovery_runtime.py",
                     ROOT / "tests/test_principal_sqrt_eigen_refinement_tf.py"):
            destination = output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        migration = {
            "schema": runner.SOURCE_MIGRATION_SCHEMA, "migration_id": runner.NUMERICAL_MIGRATION_ID,
            "from_sources": start["sources"], "to_sources": current,
            "changed_paths": sorted(path for path in set(current) | set(start["sources"])
                                    if current.get(path) != start["sources"].get(path)),
            "plan_hash": start["plan_hash"], "claim_boundary": runner.CLAIM_BOUNDARY,
            "scientific_contract_unchanged": True, "serializer_only": False,
            "execution_namespace": namespace, "created_at_unix": time.time(),
            "repair_plan": str(ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md"),
            "previous_migration_archive": str(output / "previous-source-migration.json"),
            "cpu_regression": str(regression),
            "cpu_regression_sha256": hashlib.sha256(regression.read_bytes()).hexdigest(),
            "candidate_gpu_manifest": str(candidate_root / "run_manifest.json"),
            "candidate_gpu_manifest_sha256": hashlib.sha256((candidate_root / "run_manifest.json").read_bytes()).hexdigest(),
            "numerical_repair": "binary64_batched_jacobi_refinement_no_clipping_or_changed_thresholds",
            "fresh_scopes": list(runner.ARMS), "old_tuning_and_streams_reused": False,
        }
        runner.durable_json(output / "source-migration.json", migration)
        runner.durable_json(previous_path, migration)
        restored, same_ledger = runner.initialize_campaign(CAMPAIGN, resume=True)
        execution, effective = runner.prepare_execution_namespace(CAMPAIGN, restored, current, migration)
        runner.verify_binding(effective, execution)
        verification = runner.verify_campaign_bundles(CAMPAIGN)
        if start_path.read_bytes() != start_bytes or ledger_path.read_bytes() != ledger_bytes:
            raise RuntimeError("numerical migration modified original start or campaign accounting")
        receipt = {
            "status": "NUMERICAL_MIGRATION_READY_NO_GPU_WORK_LAUNCHED", "at_unix": time.time(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "command": [sys.executable, *sys.argv], "gpu_intentionally_hidden": True,
            "execution_root": str(execution), "shared_ledger": str(same_ledger.path),
            "consumed_gpu_seconds": ledger["consumed_seconds"],
            "remaining_gpu_seconds": same_ledger.remaining_seconds(),
            "original_start_sha256": hashlib.sha256(start_bytes).hexdigest(),
            "unchanged_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "bundle_verification": verification,
            "cpu_tests_passed": sum(int(suite.attrib["tests"]) for suite in suites),
            "source_snapshot": str(output / "sources"), "migration": str(previous_path),
        }
        runner.durable_json(output / "activation-result.json", receipt)
        print(json.dumps({"output": str(output), **receipt}, indent=2))


if __name__ == "__main__":
    main()
