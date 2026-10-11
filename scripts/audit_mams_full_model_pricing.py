#!/usr/bin/env python3
"""Read-only provenance, inventory and cost audit of a D4R4a price attempt.

This does not replay the sampler or establish a posterior or tuning artifact.
The worker's native health check remains the numerical short-path authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_digest(value) -> str:
    content = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def decode_metadata(tree):
    if "scalar" in tree:
        return tree["scalar"]
    if "mapping" in tree:
        return {key: decode_metadata(value) for key, value in tree["mapping"].items()}
    if "sequence" in tree:
        return [decode_metadata(value) for value in tree["sequence"]]
    raise ValueError("metadata unexpectedly contains a tensor")


def check_hashes(records: dict[str, str]) -> int:
    for name, expected in records.items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"recorded input/source digest mismatch: {name}")
    return len(records)


def check_tensor_tree(tree: object, directory: Path) -> int:
    """Verify durable tensor bytes and declared dimensions without TF import."""
    if isinstance(tree, dict) and "tensor" in tree:
        receipt = tree["tensor"]
        path = directory / receipt["file"]
        if path.parent != directory or not path.is_file() or digest(path) != receipt["sha256"]:
            raise ValueError(f"corrupt tensor receipt: {path}")
        return 1
    if isinstance(tree, dict) and "mapping" in tree:
        return sum(check_tensor_tree(item, directory) for item in tree["mapping"].values())
    if isinstance(tree, dict) and "sequence" in tree:
        return sum(check_tensor_tree(item, directory) for item in tree["sequence"])
    if isinstance(tree, dict) and "scalar" in tree:
        return 0
    raise ValueError("unknown durable tensor tree entry")


def audit_bundle(directory: Path, member: dict, design: dict) -> dict:
    index = member["index"]
    bundle_path = directory / "prices/committed" / f"candidate-{index:02d}" / "bundle.json"
    checksum_path = bundle_path.with_name("bundle.sha256")
    if not bundle_path.is_file() or digest(bundle_path) != checksum_path.read_text():
        raise ValueError(f"missing or corrupt committed price bundle: {bundle_path}")
    bundle = read(bundle_path)
    if bundle["schema"] != "bayesfilter.durable_tensor_checkpoint.v1":
        raise ValueError("unknown price checkpoint schema")
    identity = read(directory / "prices/identity.json")
    call = read(Path(bundle["metadata"]["attempt"]) / "call-start.json")
    inputs = call["inputs"]
    if (identity != dict(binding_hash=design["binding_hash"], role="cost_only")
            or payload_digest(identity) != bundle["identity_hash"]
            or payload_digest(inputs) != bundle["inputs_hash"]
            or inputs["seed"] != member["seed"] or inputs["transitions"] != 32
            or inputs["candidate"]["candidate_id"] != member["candidate_id"]
            or inputs["candidate"]["transition_spec"] != member["frozen_spec"]):
        raise ValueError("price checkpoint inputs differ from the reported candidate")
    mapping = bundle["tree"]["mapping"]
    if decode_metadata(mapping["metadata"]) != member["metadata"]:
        raise ValueError("reported native metadata differs from its tensor bundle")
    samples = mapping["samples"]["tensor"]
    trace = mapping["trace"]["mapping"]
    if samples["shape"] != [32, 4, design["parameter_dim"]] or samples["dtype"] != "float64":
        raise ValueError("priced samples have wrong dimensions or dtype")
    for key in ("trajectory_steps", "value_score_calls_per_chain",
                "target_status_calls_per_chain", "log_accept_ratio"):
        if trace[key]["tensor"]["shape"] != [32, 4]:
            raise ValueError(f"priced {key} trace has wrong dimensions")
    telemetry = trace["target_status_telemetry"]["mapping"]
    for key in ("status_code", "valid_pre_regularized_score", "floor_count_value"):
        if telemetry[key]["tensor"]["shape"] != [32, 4]:
            raise ValueError(f"missing or malformed target status telemetry: {key}")
    count = check_tensor_tree(bundle["tree"], bundle_path.parent)
    return dict(bundle_sha256=digest(bundle_path), tensor_receipts=count,
                input_start_sha256=inputs["start_sha256"])


def audit_attempt(root: Path) -> dict:
    manifest, result, supervisor = (read(root / name) for name in
                                    ("manifest.json", "result.json", "supervisor.json"))
    if manifest["status"] != result["status"]:
        raise ValueError("manifest/result status mismatch")
    if manifest["status"] == "completed" and supervisor["exit_code"] != 0:
        raise ValueError("completed run has failed process exit")
    # Report any deadline overrun exactly. Process reap/I/O can complete after
    # a deadline; a new unproven grace number must not redefine the budget.
    deadline_overrun = max(0., supervisor["elapsed_seconds"] - supervisor["max_wall_seconds"])
    if manifest["use_xla"] is not True or manifest["cuda_visible_devices"] != "1":
        raise ValueError("unexpected backend or GPU")
    memory = manifest["memory_policy"]
    if (memory["tf_force_gpu_allow_growth"] != "true"
            or memory["all_physical_devices_memory_growth"] is not True
            or memory["configured_before_logical_device_initialization"] is not True):
        raise ValueError("GPU memory growth not established")
    worker_source = Path(manifest["command"][1]).resolve()
    snapshot = worker_source.parents[1]
    source_record = read(snapshot / ".validation-source.json")
    if (source_record["git_commit"] != manifest["git_commit"]
            or source_record["git_tree"] != manifest["git_tree"]):
        raise ValueError("worker snapshot differs from manifest")
    source_count = check_hashes(manifest["source_hashes"])
    geometry_count = check_hashes(manifest["geometry_input_sha256"])
    operator_count = check_hashes({str(snapshot / name): value for name, value
                                   in manifest["compiled_operator_sha256"].items()})
    if [row["target"] for row in result["rows"]] != manifest["targets"]:
        raise ValueError("target denominator/order mismatch")
    all_seeds: set[tuple[int, int]] = set()
    rows = []
    for row in result["rows"]:
        target = row["target"]
        directory = root / target
        design_path = directory / "design.json"
        if not design_path.exists():
            if row["rows"]:
                raise ValueError("priced candidates lack a target design")
            rows.append(dict(target=target, status=row["status"], pilot_status="not_started",
                             proposal_count=0, priced_count=0))
            continue
        design = read(design_path)
        if (design["target"] != target or design["measurement_role"] != "cost_only_no_qualification"
                or design["global_coverage_established"] is not False
                or design["source_hashes"] != manifest["source_hashes"]):
            raise ValueError("target design has wrong provenance or scientific role")
        pilot_file = directory / "pilot/pilot-progress.json"
        if not pilot_file.is_file():
            rows.append(dict(target=target, status=row["status"], pilot_status="not_started",
                             proposal_count=0, priced_count=0))
            continue
        pilot = read(pilot_file)
        attempts = pilot["attempts"]
        completed = [a for a in attempts if a["status"] == "completed"]
        if (sum(a["actual_gradient_work"] for a in completed) > pilot["actual_gradient_work"]
                or sum(a["reserved_gradient_work"] for a in attempts) > pilot["reserved_gradient_work"]
                or pilot["actual_gradient_work"] > pilot["reserved_gradient_work"]):
            raise ValueError("pilot work accounting is inconsistent")
        if pilot["status"] == "completed" and (
                len(completed) != len(attempts)
                or sum(a["actual_gradient_work"] for a in completed) != pilot["actual_gradient_work"]
                or sum(a["reserved_gradient_work"] for a in attempts) != pilot["reserved_gradient_work"]
                or [a["seed"] for a in attempts] != [s["seed"] for s in pilot["seeds"]]):
            raise ValueError("completed pilot work/stream inventory is not exact")
        for entry in pilot["seeds"]:
            seed = tuple(entry["seed"])
            if seed in all_seeds:
                raise ValueError("pilot seed reused across target work")
            all_seeds.add(seed)
        proposal_file = directory / "pilot/pilot-result.json"
        proposals = read(proposal_file)["candidates"] if proposal_file.exists() else []
        if len(row["rows"]) != len(proposals):
            raise ValueError("created price denominator differs from pilot proposal inventory")
        priced = []
        for index, member in enumerate(row["rows"]):
            if member["index"] != index or member["frozen_spec"] != proposals[index]:
                raise ValueError("priced candidate differs from frozen proposal")
            if "seed" in member:
                seed = tuple(member["seed"])
                if seed in all_seeds:
                    raise ValueError("price seed reused from pilot or another candidate")
                all_seeds.add(seed)
            if member["status"] in ("healthy_short_path", "candidate_veto"):
                if "seed" not in member:
                    raise ValueError("completed price lacks its random stream")
                receipt = audit_bundle(directory, member, design)
                metadata = member["metadata"]
                if (metadata["use_xla"] is not True or
                        metadata["transition_spec"] != member["frozen_spec"] or
                        "GPU" not in metadata["samples_device"] or
                        metadata["value_score_row_evaluations"] <= 0):
                    raise ValueError("native price metadata mismatches frozen GPU policy")
                priced.append(dict(index=index, status=member["status"], **receipt,
                                   actual_value_score_rows=metadata["value_score_row_evaluations"],
                                   actual_status_rows=metadata["target_status_row_evaluations"],
                                   maximum_reserved_gradient_work=metadata["maximum_reserved_gradient_work"]))
        if row["status"] == "completed" and len(priced) != len(proposals):
            raise ValueError("completed target contains an unpriced proposal")
        if len({p["input_start_sha256"] for p in priced}) > 1:
            raise ValueError("candidate prices did not reset to the same start bank")
        rows.append(dict(target=target, status=row["status"], pilot_status=pilot["status"],
                         recorded_hypothesis_count=len(pilot["candidate_rows"]),
                         proposal_count=len(proposals), priced_count=len(priced),
                         omitted_count=len(pilot.get("omitted_candidates", [])),
                         pilot_actual_work=pilot["actual_gradient_work"],
                         pilot_reserved_work=pilot["reserved_gradient_work"],
                         candidate_prices=priced))
    return dict(attempt=str(root.resolve()), source_commit=manifest["git_commit"],
                status=result["status"], supervisor_exit_code=supervisor["exit_code"],
                supervisor_elapsed_seconds=supervisor["elapsed_seconds"],
                deadline_overrun_seconds=deadline_overrun,
                source_file_hashes_checked=source_count,
                geometry_input_hashes_checked=geometry_count,
                compiled_operator_hashes_checked=operator_count,
                distinct_seed_pairs=len(all_seeds), rows=rows,
                authority="cost_and_short_path_health_only_no_posterior_or_tuning")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_attempt(args.attempt)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "source_commit", "supervisor_elapsed_seconds",
                       "source_file_hashes_checked", "distinct_seed_pairs")}, sort_keys=True))


if __name__ == "__main__":
    main()
