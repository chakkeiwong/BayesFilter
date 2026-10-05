"""Read-only settled price/source/accounting audit; no numerical admission."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repo = args.repo.resolve()
base = repo/"docs/plans/artifacts/hmc-v7-release-2026-10-02"
price = base/"gpu-price-serial-restored-01"
source = base/"source-23/source"
sys.path.insert(0, str(source))
from scripts.run_hmc_v7_release_prices import check_source
from scripts.analyze_hmc_v7_confirmation import price_report

signature = check_source(source)
report = price_report([price/"result.json"], replications_per_family=32, available_gpu_seconds=0)
assert report["complete_case_seconds"].keys() == {"nonlinear"} and not report["issues"]
assert report["source_manifest_sha256"] == signature
outer = read(price/"result.json")
manifest = read(price/"nonlinear/manifest.json")
model_root = price/"nonlinear/model"
model = read(model_root/"result.json")
checkpoint = read(model_root/"tuning/tuning_checkpoint.json")["result"]
config = read(price/"nonlinear-config.json")
assert manifest["source_manifest_sha256"] == signature
assert config == read(model_root/"configuration.json")
assert manifest["configuration_sha256"] == sha(price/"nonlinear-config.json")
assert config["seed"] == model["sampling_streams"] == [20261002, 2502]
assert manifest["memory_policy"]["all_physical_devices_memory_growth"] is True
assert manifest["memory_policy"]["configured_before_logical_device_initialization"] is True
assert manifest["jit_compile"] is True and manifest["dtype"] == "float64"
works = {w["work_item_id"]: w for w in checkpoint["work_items"]}
inventory = hashlib.sha256()
identities, seeds = set(), set()
native, spans = [], []
for path in sorted((model_root/"tuning/numerical_chunks").glob("*.json")):
    raw = path.read_bytes()
    row = json.loads(raw)
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == path.stem
    work = row["work"]
    issued = works[work["work_item_id"]]
    assert issued["status"] == "completed"
    assert {k: v for k, v in work.items() if k != "status"} == {k: v for k, v in issued.items() if k != "status"}
    key = (work["work_item_id"], row["trial_ordinal"], row["trial_chunk_index"])
    assert key not in identities and tuple(row["seed"]) not in seeds
    identities.add(key); seeds.add(tuple(row["seed"]))
    assert "GPU:0" in row["samples_device"]
    assert row["runtime"]["jit_compile"] and row["runtime"]["use_xla"]
    native.append(row["runtime"]["sample_chain_call_s"])
    spans.append(row["elapsed_seconds"])
    inventory.update(path.name.encode()+b":"+hashlib.sha256(raw).hexdigest().encode()+b"\n")
assert len(identities) == model["evidence_accounting"]["charged_chunks"] == 10816
assert model["evidence_accounting"]["invalid_complete_trials"] == 0
assert model["evidence_accounting"]["attempted_work_outside_complete_trials"] == 0
members = [read(path)["candidate_id"] for path in model_root.glob("*-member.json")]
assert len(members) == len(set(members)) == 19
assert set(members) == set(model["verified_candidate_ids"])
assert model["candidate_states"] == checkpoint["candidate_states"]
stages = read(model_root/"stage_timing.json")
assert stages[-1]["stage"] == "accounting_checked"
assert all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a, b in zip(stages, stages[1:]))
restoration = read(base/"serial-restoration-runtime-audit.json")
changed = [path for path in restoration["checked"] if sha(repo/path) != sha(source/path)]
assert not changed
allocation = read(base/"allocation.json")
receipt_path = str((price/"result.json").relative_to(repo))
charges = [c for c in allocation["charges"] if c["receipt"] == receipt_path]
assert len(charges) == 1 and charges[0]["seconds"] == outer["wall_seconds"]
remaining = allocation["gpu_release_uncommitted_seconds"]
external = allocation["closed_ssm_reconciliation"]["remaining_outside_release_allocation_seconds"]
all_remaining = remaining+external
forecast = 32*report["complete_case_seconds"]["nonlinear"]
result = dict(status="passed_source_profile_device_identity_member_and_charge_audit",
    timestamp_utc=datetime.now(timezone.utc).isoformat(), source_manifest_sha256=signature,
    input_result_sha256=sha(price/"result.json"), allocation_sha256=sha(base/"allocation.json"),
    chunks_checked=len(identities), chunk_inventory_sha256=inventory.hexdigest(),
    current_runtime_files_matching=len(restoration["checked"]), verified_members=len(members),
    candidate_states=dict(Counter(model["candidate_states"].values())),
    native_call_shares_seconds=math.fsum(native), native_and_serialization_seconds=math.fsum(spans),
    stages=stages, enclosing_gpu_seconds=outer["wall_seconds"],
    funding_screen=dict(replications_per_family=32, nonlinear_only_point_forecast_seconds=forecast,
        release_gpu_remaining_seconds=remaining, authorized_unallocated_gpu_seconds=external,
        all_remaining_authorized_gpu_seconds=all_remaining,
        additional_seconds_for_nonlinear_only_point_forecast=max(0, forecast-all_remaining),
        fits_before_pricing_other_families=forecast<=all_remaining,
        complete_three_family_forecast_available=False,
        interpretation="One complete-family descriptive extrapolation exceeds the whole remaining allowance even with zero cost assigned to other families; not a runtime lower confidence bound or a measured three-family forecast."),
    plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md",
    numerical_sampling=False, release_ready=False, default_promoted=False)
with (args.output/"result.json").open("x") as f:
    f.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps({k: result[k] for k in ("status", "chunks_checked", "verified_members", "funding_screen")}))
