#!/usr/bin/env python3
"""Close the imported-helper provenance gap for the completed C3B run.

This is an artifact-only repair.  It never imports TensorFlow, evaluates the
target, restores a map, or changes the original immutable C3B manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/attempt-02/run_manifest.json"
HELPER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase8_c3_lineage_overlap_2026_08_30.py"
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase8_c3b_l5_ladder_2026_08_31.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3b-l5-ladder-subplan-2026-08-31.md"
OUTPUT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/provenance-repair-2026-08-31/attempt-01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    required = (ORIGINAL, HELPER, RUNNER, PLAN)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing provenance inputs: {missing}")
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    if original.get("status") != "PASS_PHASE8_C3B_L5_LADDER":
        raise RuntimeError("original C3B manifest is not the completed pass")
    recorded_hash = str(original.get("manifest_hash", ""))
    if not recorded_hash:
        raise RuntimeError("original C3B manifest has no manifest_hash")
    without_hash = dict(original)
    without_hash.pop("manifest_hash", None)
    if stable_hash(without_hash) != recorded_hash:
        raise RuntimeError("original C3B manifest hash does not round-trip")
    helper_source = HELPER.read_text(encoding="utf-8")
    import_line = 'importlib.import_module("run_ssl_lstm_q20_phase8_c3_lineage_overlap_2026_08_30")'
    if import_line not in RUNNER.read_text(encoding="utf-8"):
        raise RuntimeError("C3B runner no longer names the imported C3 helper")
    helper_hash = sha256(HELPER)
    runner_hash = sha256(RUNNER)
    plan_hash = sha256(PLAN)
    original_mtime = ORIGINAL.stat().st_mtime
    helper_mtime = HELPER.stat().st_mtime
    if helper_mtime > original_mtime:
        raise RuntimeError("helper was modified after the C3B manifest; provenance cannot be repaired")
    payload: dict[str, object] = {
        "schema": "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_c3b_provenance_repair.v1",
        "status": "PASS_C3B_PROVENANCE_REPAIR",
        "role": "metadata_only_no_numerical_rerun",
        "original_manifest": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": sha256(ORIGINAL), "manifest_hash": recorded_hash, "status": original["status"]},
        "imported_runtime_dependency": {"path": str(HELPER.relative_to(ROOT)), "sha256": helper_hash, "mtime_not_after_original_manifest": True, "import_contract": import_line},
        "bound_runner": {"path": str(RUNNER.relative_to(ROOT)), "sha256": runner_hash},
        "bound_plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": plan_hash},
        "checks": {"manifest_hash_round_trip": True, "helper_import_present": True, "helper_precedes_manifest": True, "numerical_rerun": False},
        "nonclaims": ["does not rerun or modify C3B", "does not add whitening, mode, posterior, HMC, or ranking evidence"],
        "created_unix": time.time(),
    }
    payload["repair_hash"] = stable_hash(payload)
    write_once(OUTPUT / "provenance_manifest.json", payload)
    print(json.dumps({"status": payload["status"], "output_dir": str(OUTPUT), "helper_sha256": helper_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "FAIL_C3B_PROVENANCE_REPAIR", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
