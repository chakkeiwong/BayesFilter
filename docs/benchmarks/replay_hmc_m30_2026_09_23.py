"""M30 diagnostic replay of saved prepared HMC chunks; no tuning authority.

The manifest freezes inputs before numerical execution. CPU checks compare
same-backend repeats; GPU checks require exact archived states and full traces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(baseline, output):
    """Select actual first calls using the audited observation-order ledger."""
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256

    baseline = baseline.resolve()
    ledger = baseline / "cost-attribution-r2.json"
    rows = []
    for target in read(ledger)["targets"]:
        fit = Path(target["fit"])
        spec_path = fit / "tuning/execution_spec.json"
        spec = read(spec_path)
        assert spec["binding_hash"] == _sha256(spec["execution"])
        chunks = {}
        for path in (fit / "tuning/numerical_chunks").glob("*.json"):
            chunk = read(path)
            assert path.stem == _sha256(chunk)
            key = (chunk["work"]["work_item_id"], tuple(chunk["seed"]))
            assert key not in chunks
            chunks[key] = (path, chunk)
        for key in target["keys"]:
            if key["L"] not in (3, 25) or key["count"] not in (8, 136, 256):
                continue
            first = key["first_call"]
            evidence_path = Path(first["evidence_path"])
            evidence = read(evidence_path)
            assert evidence_path.stem == _sha256(evidence)
            summary = evidence["chunks"][first["chunk_index"]]
            chunk_path, chunk = chunks[(first["work_id"], tuple(summary["seed"]))]
            assert chunk["includes_first_runner_trace_or_compilation"] is True
            assert chunk["binding_hash"] == evidence["binding_hash"] == spec["binding_hash"]
            assert chunk["count"] == key["count"]
            assert evidence["candidate"]["leapfrog_steps"] == key["L"]
            rows.append({"target": target["target"], "L": key["L"], "count": key["count"],
                "epsilon": evidence["candidate"]["epsilon"], "work_id": first["work_id"],
                "observation_order": first["observation_order"], "chunk_index": first["chunk_index"],
                "spec": str(spec_path), "spec_sha256": sha(spec_path),
                "evidence": str(evidence_path), "evidence_sha256": sha(evidence_path),
                "chunk": str(chunk_path), "chunk_sha256": sha(chunk_path),
                "initial_state_sha256": chunk["initial_state"]["sha256"],
                "root_seed": chunk["seed"], "chain_seeds": chunk["runtime"]["chain_seeds"],
                "target_lineage": spec["execution"]["target_lineage"],
                "geometry_sha256": _sha256({"layers": spec["execution"]["layers"]})})
    assert len(rows) == 12
    write_new(output, {"schema": "bayesfilter.hmc_m30_replay_inputs.v1",
        "baseline_source": str(baseline / "source-r3"),
        "baseline_source_manifest_sha256": sha(baseline / "source-r3-manifest.json"),
        "cost_ledger": str(ledger), "cost_ledger_sha256": sha(ledger), "cells": rows,
        "role": "mechanics and cost diagnostics only; no independent-fit replication"})
    return rows


def tensor_fingerprints(samples, trace):
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload, _trace_payload
    def compact(value):
        if "tensor" in value:
            return {key: value[key] for key in ("dtype", "shape", "sha256")}
        return {key: compact(child) for key, child in value.items()}
    return {"samples": compact(_tensor_payload(samples)), "trace": compact(_trace_payload(trace))}


def compare_payloads(actual, expected):
    """Do not omit health, endpoint momentum, nested status, or acceptance bits."""
    def compare(a, b, path):
        if set(a) != set(b):
            raise AssertionError(f"tensor keys differ: {path}")
        for key in a:
            if isinstance(a[key], dict):
                compare(a[key], b[key], path + "/" + key)
            elif a[key] != b[key]:
                raise AssertionError(f"exact tensor mismatch: {path}/{key}")
    compare(actual, expected, "")


def replay(manifest_path, target, output, *, device, strategy):
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    runtime = json.loads(json.dumps(configure_worker(SimpleNamespace(device=device))))
    import tensorflow as tf
    from bayesfilter.inference.hmc import FullChainHMCConfig, build_independent_chain_tfp_hmc_runner, stable_adapter_signature
    from bayesfilter.inference.hmc_candidate_set_execution import _rebuild_geometry, _tensor_from_payload, _trace_from_payload
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.fit_process import resource_snapshot

    manifest = read(manifest_path)
    cells = [cell for cell in manifest["cells"] if cell["target"] == target]
    if device == "cpu_reference":
        cells = [next(cell for cell in cells if cell["L"] == 3 and cell["count"] == 8)]
    spec = read(cells[0]["spec"])["execution"]
    lineage = spec["target_lineage"]
    base = ValidationTarget(lineage["model"], lineage["prior"], lineage["data"],
                            lineage["control"], jit_compile=device == "gpu")
    adapter, _ = _rebuild_geometry(base, spec["layers"], spec["target_scope"])
    assert stable_adapter_signature(base) == spec["scope"]["target_signature"]
    if device == "gpu":
        assert stable_adapter_signature(adapter) == spec["scope"]["adapter_signature"], "GPU prepared adapter identity differs"
        assert runtime["tensorflow"] == spec["versions"]["tensorflow"]
        assert runtime["tfp"] == spec["versions"]["tensorflow_probability"]
        assert runtime["tf32"] == spec["runtime_policy"]["tf32_enabled"]
        assert runtime["memory_policy"]["physical_devices"] == spec["runtime_policy"]["memory_policy"]["physical_devices"]
    source = source_state()
    result = {"schema": "bayesfilter.hmc_m30_replay.v1", "strategy": strategy,
        "reconstructed_adapter_signature": stable_adapter_signature(adapter),
        "archived_adapter_signature": spec["scope"]["adapter_signature"],
        "source": source, "runtime": runtime, "manifest": str(manifest_path),
        "manifest_sha256": sha(manifest_path), "cells": [], "status": "running",
        "resources_before": resource_snapshot(),
        "comparison": "archived GPU exact" if device == "gpu" else "CPU repeatability only",
        "nonclaims": ["no tuning authority", "no performance superiority", "no posterior evidence"]}
    runners = {}
    started = time.monotonic()
    try:
        for cell in cells:
            for key in ("spec", "evidence", "chunk"):
                assert sha(cell[key]) == cell[key + "_sha256"]
            chunk = read(cell["chunk"])
            state = _tensor_from_payload(chunk["initial_state"])
            expected = tensor_fingerprints(_tensor_from_payload(chunk["samples"]), _trace_from_payload(chunk["trace"]))
            key = (cell["count"],) if strategy == "dynamic" else (cell["L"], cell["count"])
            construction_start = time.perf_counter()
            reused = key in runners
            if not reused:
                config = FullChainHMCConfig(num_results=cell["count"], num_burnin_steps=0,
                    step_size=cell["epsilon"], num_leapfrog_steps=cell["L"], seed=tuple(cell["root_seed"]),
                    use_xla=device == "gpu", target_scope=spec["target_scope"],
                    target_status_trace_policy=spec["config"]["target_status_trace_policy"], capture_candidate_health=True)
                kwargs = {"dynamic_num_leapfrog_steps": True} if strategy == "dynamic" else {}
                runners[key] = build_independent_chain_tfp_hmc_runner(adapter, state, config, **kwargs)
            runner = runners[key]
            row = {"L": cell["L"], "count": cell["count"], "epsilon": cell["epsilon"],
                "work_id": cell["work_id"], "reused": reused,
                "construction_seconds": time.perf_counter() - construction_start, "calls": []}
            result["cells"].append(row)
            cpu_reference = None
            for repeat in range(3):
                call_start = time.perf_counter()
                kwargs = {"num_leapfrog_steps": cell["L"]} if strategy == "dynamic" else {}
                run = runner.run(current_state=state, root_seed=cell["root_seed"],
                    step_size=cell["epsilon"], mode="serial", **kwargs)
                for tensor in tf.nest.flatten((run.samples, run.trace)):
                    tensor.numpy()  # Explicit host synchronization; no NumPy arithmetic.
                elapsed = time.perf_counter() - call_start
                actual = tensor_fingerprints(run.samples, run.trace)
                if device == "gpu":
                    compare_payloads(actual, expected)
                    assert "GPU:0" in run.samples.device
                elif cpu_reference is None:
                    cpu_reference = actual
                else:
                    compare_payloads(actual, cpu_reference)
                assert run.metadata["chain_seeds"] == tuple(tuple(seed) for seed in cell["chain_seeds"]) or list(map(list, run.metadata["chain_seeds"])) == cell["chain_seeds"]
                row["calls"].append({"repeat": repeat, "seconds": elapsed, "exact_comparison_passed": True,
                    "tensor_fingerprints": actual, "samples_device": run.samples.device,
                    "tracing_counts": [r._runner.experimental_get_tracing_count() for r in runner._runners]})
            row["resources_after"] = resource_snapshot()
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["resources_after"] = resource_snapshot()
        result["runner_count"] = len(runners)
        write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare")
    freeze.add_argument("--baseline", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--target", choices=("gaussian", "beta_binomial"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("gpu", "cpu_reference"), required=True)
    run.add_argument("--strategy", choices=("static", "dynamic"), default="static")
    args = parser.parse_args()
    if args.command == "prepare":
        rows = prepare(args.baseline, args.output)
        print(json.dumps({"selected_cells": len(rows), "manifest": str(args.output)}))
    else:
        result = replay(args.manifest, args.target, args.output, device=args.device, strategy=args.strategy)
        print(json.dumps({"status": result["status"], "cells": len(result["cells"]), "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
