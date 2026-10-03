"""CPU-only artifact diagnostic; never supplies tuning authority to a worker."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("this diagnostic requires intentionally hidden GPUs")
ROOT = Path.cwd()
CAMPAIGN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
script = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py"
spec = importlib.util.spec_from_file_location("saved_checkpoint_diagnostic", script)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
    FixedTransportHMCCandidateResult,
    FixedTransportHMCKernelTuningConfig,
    FixedTransportHMCKernelTuningResult,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes():
    paths = {CAMPAIGN / "campaign-start.json", CAMPAIGN / "campaign_budget_ledger.json"}
    for directory in (CAMPAIGN / "setup", CAMPAIGN / "streams"):
        paths.update(directory.rglob("identity.json"))
        paths.update(directory.glob("**/committed/**/*"))
    return {str(path.relative_to(CAMPAIGN)): sha256(path) for path in sorted(paths) if path.is_file()}


def public_values(value):
    if isinstance(value, dict):
        if set(value) == {"__nonfinite__"}:
            return float(value["__nonfinite__"])
        return {key: public_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_values(item) for item in value]
    return value


def construct(kind, payload, aliases=None):
    aliases = aliases or {}
    return kind(**{field.name: payload[aliases.get(field.name, field.name)]
                   for field in fields(kind) if aliases.get(field.name, field.name) in payload})


def marker_paths(value, path=""):
    found = []
    if isinstance(value, dict):
        if set(value) == {"__nonfinite__"}:
            return [{"path": path, "value": value["__nonfinite__"]}]
        for key, item in value.items():
            found.extend(marker_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(marker_paths(item, f"{path}[{index}]"))
    return found


started = time.monotonic()
before = protected_hashes()
factor_root = CAMPAIGN / "setup/factor"
with runner.DurableTensorCheckpoint(factor_root, json.loads((factor_root / "identity.json").read_bytes())) as store:
    chart, _metadata = store.load("chart", {})
    tuned, _metadata = store.load("tuning-repair-r1", {"checkpoint_hash": chart["checkpoint"]["checkpoint_hash"]})
    factor = runner.restore_tuning(tuned["typed"])
factor_public = json.loads(Path(tuned["artifact"]).read_bytes())
assert factor.artifact_hash == tuned["typed"]["payload_hash"] == runner.payload_hash(factor_public)
assert factor.selected_candidate_index == factor_public["selected_candidate_index"]

strict_paths = list((CAMPAIGN / "setup/strict/tuning-attempts").glob("**/fixed_transport_hmc_tuning_result.json"))
assert len(strict_paths) == 1, strict_paths
strict_path = strict_paths[0]
strict_public = json.loads(strict_path.read_bytes())
raw = public_values(strict_public)
raw["config"] = construct(FixedTransportHMCKernelTuningConfig, raw["config"])
raw["candidates"] = tuple(construct(FixedTransportHMCCandidateResult, row, {"ladder_result": "ladder"})
                          for row in raw["candidates"])
raw["hard_vetoes"] = tuple(raw["hard_vetoes"])
raw["repair_triggers"] = tuple(raw["repair_triggers"])
raw["nonclaims"] = tuple(raw["nonclaims"])
strict = construct(FixedTransportHMCKernelTuningResult, raw, {
    "tuning_scope_payload": "tuning_scope", "route_record_payload": "active_route",
    "coordinate_payload": "coordinate_identity", "candidate_selection_payload": "candidate_selection",
    "fixed_grid_scale_selection_payload": "fixed_grid_scale_selection",
})
assert strict.artifact_hash == runner.payload_hash(strict_public)
encoded = runner.typed_tuning_payload(strict)
scratch = Path(__file__).resolve().parent / "strict-fixture-checkpoint"
identity = {"role": "diagnostic_only_not_live_setup", "public_artifact_sha256": sha256(strict_path)}
with runner.DurableTensorCheckpoint(scratch, identity) as store:
    store.run("tuning", {}, lambda: encoded)
with runner.DurableTensorCheckpoint(scratch, identity) as store:
    reloaded, _metadata = store.load("tuning", {})
restored = runner.restore_tuning(reloaded)
assert restored.artifact_hash == strict.artifact_hash
assert restored.hard_vetoes == strict.hard_vetoes
assert restored.selected_candidate_index == strict.selected_candidate_index == 0
assert "candidate_1_selection_replication_1_verification_selection_efficiency_nonfinite" in restored.hard_vetoes
markers = marker_paths(strict_public)
assert len(markers) == 6 and all(row["value"] == "nan" for row in markers)
assert protected_hashes() == before
report = {
    "status": "PASS_SAVED_TUNING_CHECKPOINT_DIAGNOSTIC", "role": "cpu_artifact_diagnostic_only",
    "gpu_devices": "intentionally_hidden", "command": [sys.executable, *sys.argv],
    "captured_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - started,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "sources": runner.source_hashes(), "repair_script_sha256": sha256(script),
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-nonfinite-checkpoint-repair-2026-09-10.md",
    "protected_hashes_unchanged": before, "bundle_verification": runner.verify_campaign_bundles(CAMPAIGN),
    "factor_legacy_hash_preserved": factor.artifact_hash, "factor_selected_candidate": factor.selected_candidate_index,
    "strict_public_artifact": str(strict_path), "strict_public_artifact_sha256": sha256(strict_path),
    "strict_roundtrip_hash": restored.artifact_hash, "strict_hard_vetoes_preserved": restored.hard_vetoes,
    "strict_nonfinite_marker_paths": markers, "strict_selected_candidate": restored.selected_candidate_index,
    "strict_live_tuning_checkpoint_published": False, "scientific_or_posterior_promotion": False,
}
runner.durable_json(Path(__file__).with_name("saved-artifact-check.json"), report)
print(json.dumps({key: report[key] for key in ("status", "wall_seconds", "bundle_verification",
                 "strict_nonfinite_marker_paths", "strict_live_tuning_checkpoint_published")}, indent=2))
