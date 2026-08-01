#!/usr/bin/env python3
"""Build a CPU-hidden graph-topology audit for the clean-XLA LGSSM route."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as model
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


DTYPE = tf.float64
THETA = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t10-preparation", type=Path, required=True)
    parser.add_argument("--t50-preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(path: Path) -> dict[str, object]:
    preparation = json.loads(path.read_text(encoding="utf-8"))
    time_steps = int(preparation["target"]["time_steps"])
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:time_steps], DTYPE
    )
    evaluate = model.make_contract_e_tp_lgssm_score_informed_recursive_tf(
        observations,
        tf.constant(preparation["quadrature"]["nodes"], DTYPE),
        tf.constant(preparation["quadrature"]["weights"], DTYPE),
        tf.constant(preparation["active_indices"], tf.int32),
        tf.constant(preparation["row_scales"], DTYPE),
        feature_mode="finite_lookahead",
        lookahead_steps=8,
        jit_compile=True,
    )
    started = time.perf_counter()
    concrete = evaluate.get_concrete_function(THETA)
    trace_seconds = time.perf_counter() - started
    graph = concrete.graph.as_graph_def()
    top_operations = collections.Counter(node.op for node in graph.node)
    function_operations = collections.Counter(
        node.op for function in graph.library.function for node in function.node_def
    )
    while_count = sum(
        top_operations[name] + function_operations[name]
        for name in ("While", "StatelessWhile")
    )
    return {
        "time_steps": time_steps,
        "preparation_path": str(path.relative_to(ROOT)),
        "preparation_sha256": _sha256(path),
        "trace_seconds": trace_seconds,
        "top_level_nodes": len(graph.node),
        "function_nodes": sum(
            len(function.node_def) for function in graph.library.function
        ),
        "function_count": len(graph.library.function),
        "graphdef_bytes": graph.ByteSize(),
        "functional_while_count": while_count,
        "top_operation_counts": dict(top_operations),
        "function_operation_counts": dict(function_operations),
    }


def main() -> None:
    args = _parse()
    output = _path(args.output)
    if output.exists():
        raise FileExistsError(output)
    inventories = [
        _inventory(_path(args.t10_preparation)),
        _inventory(_path(args.t50_preparation)),
    ]
    by_time = {item["time_steps"]: item for item in inventories}
    t10 = by_time[10]
    t50 = by_time[50]
    ratios = {
        "top_level_node_ratio_t50_t10": t50["top_level_nodes"] / t10["top_level_nodes"],
        "function_node_ratio_t50_t10": t50["function_nodes"] / t10["function_nodes"],
        "graphdef_byte_ratio_t50_t10": t50["graphdef_bytes"] / t10["graphdef_bytes"],
    }
    gates = {
        "functional_while_count_at_least_two": all(
            item["functional_while_count"] >= 2 for item in inventories
        ),
        "top_level_node_ratio_at_most_1_10": (
            ratios["top_level_node_ratio_t50_t10"] <= 1.10
        ),
        "function_node_ratio_at_most_1_10": (
            ratios["function_node_ratio_t50_t10"] <= 1.10
        ),
        "graphdef_byte_ratio_at_most_1_25": (
            ratios["graphdef_byte_ratio_t50_t10"] <= 1.25
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"clean-XLA graph topology gate failed: {gates}")
    payload = {
        "schema": "bayesfilter.contract_e_tp.clean_xla_graph_audit.v1",
        "status": "PASS_CLEAN_XLA_GRAPH_TOPOLOGY",
        "inventories": inventories,
        "ratios": ratios,
        "gates": gates,
        "execution": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "tensorflow_version": tf.__version__,
            "device_policy": "CUDA_VISIBLE_DEVICES=-1 CPU-hidden graph construction",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "ratios": ratios, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
