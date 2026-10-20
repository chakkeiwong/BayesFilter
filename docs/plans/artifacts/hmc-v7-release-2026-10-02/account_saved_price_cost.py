"""Read recorded native/serialization spans once from a settled full price."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def seconds(value):
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0
    return value


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
model = json.loads((args.model/"result.json").read_text())
assert model["completion_status"] == "complete" and model["checkpoint_recomputed"] is True
checkpoint = json.loads((args.model/"tuning/tuning_checkpoint.json").read_text())["result"]
manifest = json.loads((args.model.parent/"manifest.json").read_text())
works = {row["work_item_id"]: row for row in checkpoint["work_items"]}
candidates = {row["candidate_record_hash"]: row for row in checkpoint["candidates"]}
identity, seeds = set(), set()
native, enclosing = [], []
groups = defaultdict(lambda: {"native": [], "native_and_serialization": []})
inventory = hashlib.sha256()
files = sorted((args.model/"tuning/numerical_chunks").glob("*.json"))
for path in files:
    raw = path.read_bytes()
    row = json.loads(raw)
    assert digest(row) == path.stem, path
    inventory.update(path.name.encode()+b":"+hashlib.sha256(raw).hexdigest().encode()+b"\n")
    work = row["work"]
    issued = works[work["work_item_id"]]
    assert issued["status"] == "completed"
    assert {k: v for k, v in work.items() if k != "status"} == {k: v for k, v in issued.items() if k != "status"}
    key = (work["work_item_id"], row["trial_ordinal"], row["trial_chunk_index"])
    assert key not in identity and tuple(row["seed"]) not in seeds
    identity.add(key)
    seeds.add(tuple(row["seed"]))
    metadata = row["runtime"]
    assert metadata["sample_chain_call_s_scope"] == "equal_share_of_enclosing_batch_call"
    call = seconds(metadata["sample_chain_call_s"])
    span = seconds(row["elapsed_seconds"])
    assert span >= call
    native.append(call)
    enclosing.append(span)
    candidate = candidates[work["candidate_record_hash"]]
    group = groups[f'{work["stage"]}:L{candidate["leapfrog_steps"]}']
    group["native"].append(call)
    group["native_and_serialization"].append(span)
assert len(files) == model["evidence_accounting"]["charged_chunks"] == 12096
assert model["evidence_accounting"]["attempted_work_outside_complete_trials"] == 0
stages = json.loads((args.model/"stage_timing.json").read_text())
points = {row["stage"]: seconds(row["elapsed_seconds"]) for row in stages}
assert all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a, b in zip(stages, stages[1:]))
search = points["tuning_returned"]-points["binding_prepared"]
covered = math.fsum(enclosing)
assert covered <= search
result = dict(
    status="checked_complete_recorded_cost_inventory", chunks=len(files), work_items=len(works),
    source_manifest_sha256=manifest["source_manifest_sha256"],
    configuration_sha256=manifest["configuration_sha256"],
    inventory_sha256=inventory.hexdigest(),
    recorded_native_call_shares_seconds=math.fsum(native),
    native_and_tensor_serialization_spans_seconds=covered,
    search_seconds=search, search_uninstrumented_seconds=search-covered,
    stage_durations={stages[i]["stage"]: stages[i]["elapsed_seconds"]-stages[i-1]["elapsed_seconds"]
                     for i in range(1, len(stages))},
    by_stage_and_L={k: dict(chunks=len(v["native"]), native_call_share_seconds=math.fsum(v["native"]),
                            native_and_serialization_seconds=math.fsum(v["native_and_serialization"]))
                    for k, v in sorted(groups.items())},
    interpretation="Recorded host spans only; no isolated synchronized GPU-kernel timing, "
                   "admission, performance ranking, confirmation authorization or release claim.",
    plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md",
    native_sampling=False, release_ready=False)
with (args.output/"runner.py").open("xb") as out:
    out.write(Path(__file__).read_bytes())
with (args.output/"result.json").open("x") as out:
    out.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps({k: v for k, v in result.items() if k in (
    "status", "chunks", "recorded_native_call_shares_seconds",
    "native_and_tensor_serialization_spans_seconds", "search_seconds", "search_uninstrumented_seconds")}))
