"""Diagnostic recorded cost partition; no timing benchmark or sampler execution."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


root = Path(__file__).resolve().parent
output = Path(sys.argv[1]).resolve()
price = root/"gpu-price-serial-restored-01"
model = price/"nonlinear/model"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


audit = read(root/"serial-price-audit-01/result.json")
assert sha(price/"result.json") == audit["input_result_sha256"]
outer = read(price/"result.json")
assert outer["status"] == "complete"
result = read(model/"result.json")
candidates = {c["candidate_record_hash"]: c for c in result["candidates"]}
work_order = {w["work_item_id"]: i for i, w in enumerate(result["work_items"])}
groups = {}
inventory = hashlib.sha256()
native_shares, serialized_spans = [], []
for path in sorted((model/"tuning/numerical_chunks").glob("*.json")):
    raw = path.read_bytes(); chunk = json.loads(raw)
    inventory.update(path.name.encode()+b":"+hashlib.sha256(raw).hexdigest().encode()+b"\n")
    runtime = chunk["runtime"]
    assert runtime["sample_chain_call_s_scope"] == "equal_share_of_enclosing_batch_call"
    seeds = tuple(tuple(s) for s in runtime["trial_batch_seeds"])
    batch = runtime["trial_batch_size"]
    assert batch == len(seeds)
    key = (chunk["work"]["work_item_id"], seeds)
    row = runtime["trial_batch_row"]
    assert 0 <= row < batch and tuple(chunk["seed"]) == seeds[row]
    share = runtime["sample_chain_call_s"]
    span = chunk["elapsed_seconds"]
    assert math.isfinite(share) and 0 <= share <= span and math.isfinite(span)
    group = groups.setdefault(key, dict(rows=set(), share=share, batch=batch,
        stage=chunk["work"]["stage"], work=chunk["work"]["work_item_id"],
        ordinal=chunk["trial_ordinal"]-row,
        L=candidates[chunk["work"]["candidate_record_hash"]]["leapfrog_steps"]))
    assert group["share"] == share and row not in group["rows"]
    group["rows"].add(row)
    native_shares.append(share); serialized_spans.append(span)
assert inventory.hexdigest() == audit["chunk_inventory_sha256"]
ordered = sorted(groups.values(), key=lambda g: (work_order[g["work"]], g["ordinal"]))
seen_shapes, calls, by_l, by_stage = set(), [], defaultdict(list), defaultdict(list)
for group in ordered:
    assert group["rows"] == set(range(group["batch"]))
    # The runner key is (count, batch), not L: source-23 reuses dynamic L.
    shape = (68, group["batch"])
    seconds = group["share"]*group["batch"]
    calls.append(dict(work=group["work"], ordinal=group["ordinal"], batch=group["batch"],
        L=group["L"], stage=group["stage"], seconds=seconds,
        first_runner_shape=shape not in seen_shapes))
    seen_shapes.add(shape); by_l[group["L"]].append(seconds); by_stage[group["stage"]].append(seconds)
native = math.fsum(native_shares)
assert abs(native-math.fsum(c["seconds"] for c in calls)) <= 1e-10
stages = {s["stage"]: s["elapsed_seconds"] for s in read(model/"stage_timing.json")}
search = stages["tuning_returned"]-stages["binding_prepared"]
components = dict(native_calls=native, native_tensor_serialization=math.fsum(serialized_spans)-native,
    other_search_work=search-math.fsum(serialized_spans),
    setup_and_closeout=outer["wall_seconds"]-search)
assert all(v >= 0 for v in components.values())
assert abs(math.fsum(components.values())-outer["wall_seconds"]) <= 1e-10
allocation = read(root/"allocation.json")
available = allocation["gpu_release_uncommitted_seconds"] + allocation["closed_ssm_reconciliation"]["remaining_outside_release_allocation_seconds"]
replications = 32
capacity = available/replications
counterfactuals = []
for component, cost in components.items():
    hypothetical = outer["wall_seconds"]-cost
    counterfactuals.append(dict(omitted_component=component, nonlinear_seconds_if_zero=hypothetical,
        nonlinear_32_seconds_if_zero=replications*hypothetical,
        still_exceeds_all_remaining_gpu=replications*hypothetical>available))
record = dict(schema="bayesfilter.hmc_v7_recorded_cost_partition.v1",
    status="checked_complete_saved_cost_partition", timestamp_utc=datetime.now(timezone.utc).isoformat(),
    source_manifest_sha256=outer["source_manifest_sha256"], price_result_sha256=sha(price/"result.json"),
    audit_result_sha256=sha(root/"serial-price-audit-01/result.json"),
    chunks=len(native_shares), batches=len(calls), runner_shapes=[list(s) for s in sorted(seen_shapes)],
    components_seconds=components, total_seconds=outer["wall_seconds"],
    first_shape_call_seconds=math.fsum(c["seconds"] for c in calls if c["first_runner_shape"]),
    subsequent_call_seconds=math.fsum(c["seconds"] for c in calls if not c["first_runner_shape"]),
    calls=calls, by_L={str(k): dict(calls=len(v),seconds=math.fsum(v)) for k,v in sorted(by_l.items())},
    by_stage={k: dict(calls=len(v),seconds=math.fsum(v)) for k,v in sorted(by_stage.items())},
    all_remaining_gpu_seconds=available, replications_per_family=replications,
    maximum_sum_of_three_family_prices_without_margin=capacity,
    zero_component_counterfactuals=counterfactuals,
    interpretation="Disjoint saved host spans and hypothetical arithmetic only. No compilation attribution, uncontended timing, lower runtime bound, new forecast, component-skipping authority, numerical validation or release claim.",
    plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md",
    gpu_initialized=False, native_sampling=False, release_ready=False)
with (output/"result.json").open("x") as f:
    f.write(json.dumps(record,indent=2,allow_nan=False)+"\n")
print(json.dumps({k: record[k] for k in ("status","chunks","batches","components_seconds","first_shape_call_seconds","subsequent_call_seconds","maximum_sum_of_three_family_prices_without_margin","zero_component_counterfactuals")}))
