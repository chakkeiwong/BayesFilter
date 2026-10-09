"""Read-only price provenance, inventory and accounting audit; no admission."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check_tensors(value):
    if isinstance(value, dict):
        if "tensor" in value:
            raw = base64.b64decode(value["tensor"], validate=True)
            assert hashlib.sha256(raw).hexdigest() == value["sha256"]
        else:
            for item in value.values():
                check_tensors(item)
    elif isinstance(value, list):
        for item in value:
            check_tensors(item)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--price", type=Path, action="append", required=True)
parser.add_argument("--enclosing-receipt", type=Path, action="append", default=[])
parser.add_argument("--baseline-price", type=Path, action="append", default=[])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repo, source = args.repo.resolve(), args.source.resolve()
base = repo / "docs/plans/artifacts/hmc-v7-release-2026-10-02"
sys.path.insert(0, str(source))
from scripts.run_hmc_v7_release_prices import check_source, DEVELOPMENT_SEEDS
from scripts.analyze_hmc_v7_confirmation import price_report
from bayesfilter.testing.acceptance_release_validation import full_search_configuration

signature = check_source(source)
allocation = read(base / "allocation.json")
available = allocation["gpu_release_uncommitted_seconds"]
outside = allocation["closed_ssm_reconciliation"]["remaining_outside_release_allocation_seconds"]
forecast = price_report(args.price, replications_per_family=32,
    available_gpu_seconds=available + outside)
assert not forecast["issues"] and not forecast["missing_cases"]
assert forecast["source_manifest_sha256"] == signature
restoration = read(base / "serial-restoration-runtime-audit.json")
assert all(sha(repo / name) == sha(source / name) for name in restoration["checked"])
families = {}
enclosures = {}
for receipt_path in args.enclosing_receipt:
    receipt = read(receipt_path)
    enclosed = Path(receipt["price_result"]).resolve()
    assert enclosed in {p.resolve() for p in args.price} and enclosed not in enclosures
    assert receipt["status"] == "complete" and receipt["exit_code"] == 0
    assert receipt["resource"] == "gpu" and receipt["gpu_uuid"] == forecast["gpu_uuid"]
    assert receipt["cpu_affinity"] == [8,9,10,11]
    enclosures[enclosed] = (receipt_path, receipt)
enclosing_overhead = 0.
baselines = {}
for baseline_path in args.baseline_price:
    baseline = read(baseline_path)
    assert baseline["status"] == "complete" and baseline["source_manifest_sha256"] == signature
    assert baseline["gpu_uuid"] == forecast["gpu_uuid"]
    for case in baseline["cases"]:
        assert case not in baselines
        baselines[case] = baseline_path.resolve().parent/case/"model"
if baselines:
    assert set(baselines) == {"lgssm_qr", "nonlinear", "funnel_residual"}
for input_path in args.price:
    price = input_path.resolve().parent
    outer = read(input_path)
    charged_path, charged_record = enclosures.get(input_path.resolve(), (input_path, outer))
    charges = [c for c in allocation["charges"]
               if c["receipt"] == str(charged_path.resolve().relative_to(repo))]
    assert len(charges) == 1
    assert charges[0]["resource"] == "gpu" and charges[0]["seconds"] == charged_record["wall_seconds"]
    if input_path.resolve() in enclosures:
        assert not any(c["receipt"] == str(input_path.resolve().relative_to(repo)) for c in allocation["charges"])
        extra = charged_record["wall_seconds"]-outer["wall_seconds"]
        assert extra >= 0
        enclosing_overhead += extra
        for case in outer["cases"]:
            forecast["complete_case_seconds"][case] += extra/len(outer["cases"])
    for attempt in outer["attempts"]:
        family = attempt["case"]
        model_root = price / family / "model"
        manifest = read(price / family / "manifest.json")
        config_path = price / (family + "-config.json")
        config, model = read(config_path), read(model_root / "result.json")
        checkpoint = read(model_root / "tuning/tuning_checkpoint.json")["result"]
        assert manifest["source_manifest_sha256"] == signature
        assert manifest["gpu_uuid"] == forecast["gpu_uuid"]
        assert manifest["configuration_sha256"] == sha(config_path)
        assert config == read(model_root / "configuration.json")
        assert canonical(config) == canonical(full_search_configuration(family,
            seed=DEVELOPMENT_SEEDS[family], wall_seconds=outer["search_cap_seconds"],
            replicated_trial_batch_size=32))
        assert config["seed"] == model["sampling_streams"]
        for field in ("all_physical_devices_memory_growth", "configured_before_logical_device_initialization"):
            assert manifest["memory_policy"][field] is True
        assert manifest["jit_compile"] is True and manifest["dtype"] == "float64"
        works = {w["work_item_id"]: w for w in checkpoint["work_items"]}
        candidates = {c["candidate_id"]: c for c in checkpoint["candidates"]}
        baseline_chunks = {}
        if baselines:
            old_model = read(baselines[family]/"result.json")
            assert read(baselines[family]/"configuration.json") == config
            assert old_model["candidate_states"] == model["candidate_states"]
            assert old_model["candidates"] == model["candidates"]
            for old_path in (baselines[family]/"tuning/numerical_chunks").glob("*.json"):
                old_chunk = read(old_path)
                assert canonical(old_chunk) == old_path.stem
                key = (old_chunk["work"]["work_item_id"], old_chunk["trial_ordinal"], old_chunk["trial_chunk_index"])
                assert key not in baseline_chunks
                baseline_chunks[key] = old_path
        events = [e for e in checkpoint["accounting_events"] if e["event"] == "numerical_chunk_charged"]
        event_keys = {(e["work_item_id"], e["trial_ordinal"], e["trial_chunk_index"]): e for e in events}
        assert len(event_keys) == len(events)
        identities, seeds, inventory = set(), set(), hashlib.sha256()
        native, spans = [], []
        for path in sorted((model_root / "tuning/numerical_chunks").glob("*.json")):
            raw = path.read_bytes()
            row = json.loads(raw)
            assert canonical(row) == path.stem
            check_tensors(row)
            work = row["work"]
            issued = works[work["work_item_id"]]
            assert issued["status"] == "completed"
            assert {k:v for k,v in work.items() if k != "status"} == {k:v for k,v in issued.items() if k != "status"}
            key = (work["work_item_id"], row["trial_ordinal"], row["trial_chunk_index"])
            if baselines:
                old_chunk = read(baseline_chunks.pop(key))
                for field in ("samples", "trace", "seed", "count", "work"):
                    assert old_chunk[field] == row[field], (family, key, field)
            assert key not in identities and tuple(row["seed"]) not in seeds
            identities.add(key); seeds.add(tuple(row["seed"]))
            event = event_keys[key]
            assert event["seed"] == row["seed"]
            assert event["transitions"] == row["count"] * row["runtime"]["chain_count"]
            assert event["gradient_work"] == event["transitions"] * (candidates[work["candidate_id"]]["leapfrog_steps"] + 1)
            assert "GPU:0" in row["samples_device"]
            assert row["runtime"]["jit_compile"] and row["runtime"]["use_xla"]
            native.append(row["runtime"]["sample_chain_call_s"])
            spans.append(row["elapsed_seconds"])
            inventory.update(path.name.encode()+b":"+hashlib.sha256(raw).hexdigest().encode()+b"\n")
        accounting = model["evidence_accounting"]
        assert identities == set(event_keys)
        assert not baseline_chunks
        assert len(identities) == accounting["charged_chunks"]
        assert accounting["unique_complete_trials"] == (
            accounting["valid_complete_trials"] + accounting["invalid_complete_trials"])
        assert accounting["attempted_work_outside_complete_trials"] >= 0
        assert sum(e["transitions"] for e in events) == accounting["attempted_transitions"]
        assert sum(e["gradient_work"] for e in events) == accounting["gradient_work"]
        members = [read(p)["candidate_id"] for p in model_root.glob("*-member.json")]
        assert len(members) == len(set(members))
        assert set(members) == set(model["verified_candidate_ids"]) == set(checkpoint["verified_candidate_ids"])
        assert model["candidate_states"] == checkpoint["candidate_states"]
        stages = read(model_root / "stage_timing.json")
        assert stages[-1]["stage"] == "accounting_checked"
        assert all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a,b in zip(stages,stages[1:]))
        families[family] = dict(verified_members=len(members), candidates=len(candidates),
            repairs=len(model["repair_actions"]), candidate_states=dict(Counter(model["candidate_states"].values())),
            chunks_checked=len(identities), chunk_inventory_sha256=inventory.hexdigest(),
            accounting=accounting, native_call_shares_seconds=math.fsum(native),
            native_and_serialization_seconds=math.fsum(spans), stages=stages,
            original_seed_raw_tensor_parity=True if baselines else None,
            price_result_sha256=sha(input_path), manifest_sha256=sha(price/family/"manifest.json"))
forecast["child_queue_point_forecast_seconds"] = forecast["point_forecast_seconds"]
forecast["additional_enclosing_overhead_per_vector_seconds"] = enclosing_overhead
forecast["point_forecast_seconds"] = 32*math.fsum(forecast["complete_case_seconds"].values())
forecast["point_forecast_fits_budget"] = forecast["point_forecast_seconds"] <= forecast["available_gpu_seconds"]
forecast["enclosing_receipts"] = [dict(path=str(p.resolve()), sha256=sha(p)) for p in args.enclosing_receipt]
result = dict(status="passed_current_source_three_family_price_audit",
    timestamp_utc=datetime.now(timezone.utc).isoformat(), families=families,
    source_manifest_sha256=signature, current_runtime_files_matching=len(restoration["checked"]),
    allocation_sha256=sha(base/"allocation.json"), price_report=forecast,
    baseline_prices=[dict(path=str(p.resolve()), sha256=sha(p)) for p in args.baseline_price],
    plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md",
    numerical_sampling=False, release_ready=False, default_promoted=False,
    interpretation="File provenance, raw tensor checksums, work/accounting and recorded full-procedure checks; no new numerical admission or independent confirmation.")
with (args.output/"result.json").open("x") as stream:
    stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
with (args.output/"auditor.py").open("xb") as stream:
    stream.write(Path(__file__).read_bytes())
print(json.dumps({"status":result["status"], "families":{
    k:{n:v[n] for n in ("verified_members","candidates","chunks_checked")} for k,v in families.items()},
    "forecast":forecast}))
