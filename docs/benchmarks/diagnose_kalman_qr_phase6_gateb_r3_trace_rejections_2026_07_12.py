#!/usr/bin/env python3
"""Offline, read-only diagnosis of the Gate B R3 GraphDef rejection."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kalman_qr_benchmark_contract as contract  # noqa: E402


SCHEMA = "bayesfilter.kalman_qr_batched_xla_repair.phase6.trace_rejection_diagnostic.v1"
TRACE_SHA256 = "7444fb41ef9d125990dee93a5370227c4b9ec0987ee37cb9ab7dfd362281d2b6"
TRACE_BYTES = 221375005
SUBPLAN = (
    ROOT
    / "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-"
    "gatec-r3-trace-rejection-blocker-subplan-2026-07-12.md"
)
SUBPLAN_SHA256 = "0afb2d033e62035c032a82db48ffce949a72776109ac9c37c97f28a04f3b3929"
METHODS = tuple(contract.PRIMARY_METHOD_IDS)
EXPECTED_COUNTS = {
    (10, METHODS[0]): (692, 47),
    (20, METHODS[0]): (692, 47),
    (30, METHODS[0]): (692, 47),
    (10, METHODS[1]): (255, 18998),
    (20, METHODS[1]): (357, 12252),
    (30, METHODS[1]): (357, 12252),
}
NONCLAIMS = [
    "no retrospective Gate B pass or Gate C authorization",
    "no XLA, Kalman target, GPU, memory, or performance evidence",
    "no method ranking or CPU/GPU scalability claim",
    "no HMC, posterior, default, production, release, or scientific-validity claim",
]
RUN_MANIFEST_EXCLUSIONS = [
    "run_manifest.finished_utc",
    "run_manifest.output_path",
    "run_manifest.started_utc",
    "run_manifest.wall_seconds",
]
FIELD_COVERAGE_MATRIX = [
    {"semantic_class": "entity_insertion_deletion_order", "test": "test_entity_insertion_deletion_and_order"},
    {"semantic_class": "op_substitution", "test": "test_node_semantic_mutations"},
    {"semantic_class": "data_edge_and_output_index", "test": "test_node_semantic_mutations"},
    {"semantic_class": "control_edge", "test": "test_node_semantic_mutations"},
    {"semantic_class": "function_call_target_and_body", "test": "test_function_semantic_mutations"},
    {"semantic_class": "function_signature_ret_control_ret", "test": "test_function_semantic_mutations"},
    {"semantic_class": "dtype_device_shape_and_list_attributes", "test": "test_attribute_and_constant_mutations"},
    {"semantic_class": "const_tensor_value", "test": "test_attribute_and_constant_mutations"},
    {"semantic_class": "cross_function_numeric_consumer", "test": "test_cross_function_numeric_consumer_is_not_safe"},
    {"semantic_class": "duplicate_stable_keys", "test": "test_duplicate_keys_and_incomplete_coverage_fail_closed"},
    {"semantic_class": "incomplete_token_coverage", "test": "test_duplicate_keys_and_incomplete_coverage_fail_closed"},
    {"semantic_class": "full_lattice_axis_classification", "test": "test_axis_classification_requires_full_lattice"},
    {"semantic_class": "deterministic_payload", "test": "test_canonical_payload_excludes_only_run_fields"},
]


class DiagnosticError(RuntimeError):
    """Raised when the offline evidence cannot be classified safely."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DiagnosticError("non-finite protobuf scalar")
        return value
    return value


def scalar_tokens(message: Any, path: str = "$") -> list[dict[str, Any]]:
    """Return every populated protobuf scalar with deterministic map ordering."""
    tokens: list[dict[str, Any]] = []
    for field, value in message.ListFields():
        field_path = f"{path}.{field.name}"
        if field.is_repeated:
            if field.message_type is not None and field.message_type.GetOptions().map_entry:
                for key in sorted(value):
                    item = value[key]
                    item_path = f"{field_path}[{json.dumps(key, sort_keys=True)}]"
                    value_field = field.message_type.fields_by_name["value"]
                    if value_field.message_type is not None:
                        tokens.extend(scalar_tokens(item, item_path))
                    else:
                        tokens.append(
                            {"path": item_path, "kind": value_field.type, "value": json_safe(item)}
                        )
            else:
                for index, item in enumerate(value):
                    item_path = f"{field_path}[{index}]"
                    if field.message_type is not None:
                        tokens.extend(scalar_tokens(item, item_path))
                    else:
                        tokens.append(
                            {"path": item_path, "kind": field.type, "value": json_safe(item)}
                        )
        elif field.message_type is not None:
            tokens.extend(scalar_tokens(value, field_path))
        else:
            tokens.append({"path": field_path, "kind": field.type, "value": json_safe(value)})
    return tokens


def unique_map(items: Iterable[Any], key_fn, *, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        key = key_fn(item)
        if not isinstance(key, str) or not key or key in result:
            raise DiagnosticError(f"duplicate or empty stable key in {label}: {key!r}")
        result[key] = item
    return result


def edge_source(raw: str) -> tuple[bool, str, int]:
    control = raw.startswith("^")
    value = raw[1:] if control else raw
    parts = value.split(":")
    if len(parts) >= 2 and parts[-1].isdigit():
        # FunctionDef inputs use node:output_arg:index; GraphDef uses node:index.
        name = ":".join(parts[:-2]) if len(parts) >= 3 else parts[0]
        return control, name, int(parts[-1])
    return control, value, 0


def called_functions(node: Any) -> list[str]:
    names: set[str] = set()
    for value in node.attr.values():
        if value.HasField("func") and value.func.name:
            names.add(value.func.name)
        names.update(item.name for item in value.list.func if item.name)
    return sorted(names)


def attr_structure(value: Any) -> dict[str, Any]:
    fields = {field.name: field for field, _ in value.ListFields()}
    result: dict[str, Any] = {"fields": sorted(fields)}
    if value.HasField("tensor"):
        result["tensor"] = {
            "dtype": int(value.tensor.dtype),
            "rank": len(value.tensor.tensor_shape.dim),
            "unknown_rank": bool(value.tensor.tensor_shape.unknown_rank),
            "representation_fields": sorted(
                field.name for field, _ in value.tensor.ListFields()
            ),
        }
    if value.HasField("shape"):
        result["shape"] = {
            "rank": len(value.shape.dim),
            "unknown_rank": bool(value.shape.unknown_rank),
        }
    if value.HasField("func"):
        result["func"] = value.func.name
    if "type" in fields:
        result["type"] = int(value.type)
    if "list" in fields:
        result["list"] = {
            field.name: len(item)
            for field, item in value.list.ListFields()
        }
    return result


def node_summary(node: Any) -> dict[str, Any]:
    raw = node.SerializeToString(deterministic=True)
    tokens = scalar_tokens(node)
    structural = {
        "name": node.name,
        "op": node.op,
        "device": node.device,
        "inputs": list(node.input),
        "called_functions": called_functions(node),
        "attributes": {
            key: attr_structure(value) for key, value in sorted(node.attr.items())
        },
    }
    return {
        "name": node.name,
        "op": node.op,
        "device": node.device,
        "inputs": list(node.input),
        "parsed_inputs": [list(edge_source(value)) for value in node.input],
        "called_functions": called_functions(node),
        "attribute_keys": sorted(node.attr),
        "structural_sha256": contract.canonical_sha256(structural),
        "raw_sha256": sha256_bytes(raw),
        "token_count": len(tokens),
        "token_sha256": contract.canonical_sha256(tokens),
    }


def function_summary(function: Any) -> dict[str, Any]:
    nodes = unique_map(function.node_def, lambda node: node.name, label="FunctionDef.node_def")
    input_args = unique_map(function.signature.input_arg, lambda arg: arg.name, label="input_arg")
    output_args = unique_map(function.signature.output_arg, lambda arg: arg.name, label="output_arg")
    raw = function.SerializeToString(deterministic=True)
    tokens = scalar_tokens(function)
    summary = {
        "name": function.signature.name,
        "node_order": [node.name for node in function.node_def],
        "nodes": {name: node_summary(node) for name, node in sorted(nodes.items())},
        "input_args": {
            name: sha256_bytes(arg.SerializeToString(deterministic=True))
            for name, arg in sorted(input_args.items())
        },
        "output_args": {
            name: sha256_bytes(arg.SerializeToString(deterministic=True))
            for name, arg in sorted(output_args.items())
        },
        "control_outputs": list(function.signature.control_output),
        "ret": dict(sorted(function.ret.items())),
        "control_ret": dict(sorted(function.control_ret.items())),
        "attribute_keys": sorted(function.attr),
        "raw_sha256": sha256_bytes(raw),
        "token_count": len(tokens),
        "token_sha256": contract.canonical_sha256(tokens),
    }
    structural = {
        "name": summary["name"],
        "node_order": summary["node_order"],
        "node_structural_sha256": {
            name: node["structural_sha256"] for name, node in summary["nodes"].items()
        },
        "input_args": summary["input_args"],
        "output_args": summary["output_args"],
        "control_outputs": summary["control_outputs"],
        "ret": summary["ret"],
        "control_ret": summary["control_ret"],
        "attribute_keys": summary["attribute_keys"],
    }
    summary["structural_sha256"] = contract.canonical_sha256(structural)
    return summary


def graph_view(graph: Any) -> dict[str, Any]:
    nodes = unique_map(graph.node, lambda node: node.name, label="GraphDef.node")
    functions = unique_map(
        graph.library.function,
        lambda function: function.signature.name,
        label="FunctionDefLibrary.function",
    )
    gradients = unique_map(
        graph.library.gradient,
        lambda gradient: gradient.function_name,
        label="FunctionDefLibrary.gradient",
    )
    registered = unique_map(
        graph.library.registered_gradients,
        lambda gradient: gradient.registered_op_type,
        label="FunctionDefLibrary.registered_gradients",
    )
    tokens = scalar_tokens(graph)
    entity_count = (
        len(nodes)
        + len(functions)
        + sum(len(function.node_def) for function in functions.values())
        + len(gradients)
        + len(registered)
    )
    return {
        "raw_sha256": sha256_bytes(graph.SerializeToString(deterministic=True)),
        "raw_token_count": len(tokens),
        "raw_token_sha256": contract.canonical_sha256(tokens),
        "entity_count": entity_count,
        "node_order": [node.name for node in graph.node],
        "nodes": {name: node_summary(node) for name, node in sorted(nodes.items())},
        "function_order": [function.signature.name for function in graph.library.function],
        "functions": {
            name: function_summary(function) for name, function in sorted(functions.items())
        },
        "gradient_order": [gradient.function_name for gradient in graph.library.gradient],
        "gradients": {
            name: sha256_bytes(value.SerializeToString(deterministic=True))
            for name, value in sorted(gradients.items())
        },
        "registered_gradient_order": [
            gradient.registered_op_type for gradient in graph.library.registered_gradients
        ],
        "registered_gradients": {
            name: sha256_bytes(value.SerializeToString(deterministic=True))
            for name, value in sorted(registered.items())
        },
        "versions_sha256": sha256_bytes(
            graph.versions.SerializeToString(deterministic=True)
        ),
        "debug_info_sha256": sha256_bytes(
            graph.debug_info.SerializeToString(deterministic=True)
        ),
    }


def validate_graph_view(graph: Any, view: Mapping[str, Any]) -> None:
    tokens = scalar_tokens(graph)
    functions = unique_map(
        graph.library.function,
        lambda function: function.signature.name,
        label="FunctionDefLibrary.function",
    )
    expected_entity_count = (
        len(graph.node)
        + len(functions)
        + sum(len(function.node_def) for function in functions.values())
        + len(graph.library.gradient)
        + len(graph.library.registered_gradients)
    )
    checks = {
        "raw_digest": view.get("raw_sha256")
        == sha256_bytes(graph.SerializeToString(deterministic=True)),
        "token_count": view.get("raw_token_count") == len(tokens),
        "token_digest": view.get("raw_token_sha256") == contract.canonical_sha256(tokens),
        "entity_count": view.get("entity_count") == expected_entity_count,
        "node_keys": set(view.get("nodes", {})) == {node.name for node in graph.node},
        "function_keys": set(view.get("functions", {})) == set(functions),
        "gradient_keys": set(view.get("gradients", {}))
        == {gradient.function_name for gradient in graph.library.gradient},
        "registered_gradient_keys": set(view.get("registered_gradients", {}))
        == {
            gradient.registered_op_type
            for gradient in graph.library.registered_gradients
        },
    }
    if not all(checks.values()):
        raise DiagnosticError(f"incomplete or inconsistent GraphDef view: {checks}")


def map_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    common = sorted(left_keys & right_keys)
    changed = [key for key in common if left[key] != right[key]]
    return {
        "only_left": sorted(left_keys - right_keys),
        "only_right": sorted(right_keys - left_keys),
        "changed": changed,
        "common_count": len(common),
    }


def compare_graph_views(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    node_delta = map_delta(left["nodes"], right["nodes"])
    function_delta = map_delta(left["functions"], right["functions"])
    gradient_delta = map_delta(left["gradients"], right["gradients"])
    registered_delta = map_delta(left["registered_gradients"], right["registered_gradients"])
    changed_function_nodes: dict[str, Any] = {}
    for name in sorted(set(left["functions"]) & set(right["functions"])):
        delta = map_delta(left["functions"][name]["nodes"], right["functions"][name]["nodes"])
        if delta["only_left"] or delta["only_right"] or delta["changed"]:
            changed_function_nodes[name] = delta
    entity_set_changed = any(
        delta[side]
        for delta in (node_delta, function_delta, gradient_delta, registered_delta)
        for side in ("only_left", "only_right")
    ) or any(
        delta[side]
        for delta in changed_function_nodes.values()
        for side in ("only_left", "only_right")
    )
    changed_node_semantics = [
        name
        for name in sorted(set(left["nodes"]) & set(right["nodes"]))
        if left["nodes"][name]["structural_sha256"]
        != right["nodes"][name]["structural_sha256"]
    ]
    changed_function_semantics = [
        name
        for name in sorted(set(left["functions"]) & set(right["functions"]))
        if left["functions"][name]["structural_sha256"]
        != right["functions"][name]["structural_sha256"]
    ]
    order_changed = any(
        left[field] != right[field]
        for field in (
            "node_order",
            "function_order",
            "gradient_order",
            "registered_gradient_order",
        )
    )
    topology_changed = (
        entity_set_changed
        or bool(changed_node_semantics)
        or bool(changed_function_semantics)
        or order_changed
    )
    return {
        "node_delta": node_delta,
        "function_delta": function_delta,
        "changed_function_nodes": changed_function_nodes,
        "gradient_delta": gradient_delta,
        "registered_gradient_delta": registered_delta,
        "changed_node_semantics": changed_node_semantics,
        "changed_function_semantics": changed_function_semantics,
        "entity_set_changed": entity_set_changed,
        "order_changed": order_changed,
        "topology_changed": topology_changed,
        "raw_equal": left["raw_sha256"] == right["raw_sha256"],
    }


INTEGER_DTYPES = {
    3: ("i", 4),   # DT_INT32
    4: ("B", 1),   # DT_UINT8
    5: ("h", 2),   # DT_INT16
    6: ("b", 1),   # DT_INT8
    9: ("q", 8),   # DT_INT64
    17: ("H", 2),  # DT_UINT16
    22: ("I", 4),  # DT_UINT32
    23: ("Q", 8),  # DT_UINT64
}


def tensor_integer_values(tensor: Any) -> list[int] | None:
    dtype = int(tensor.dtype)
    if dtype not in INTEGER_DTYPES:
        return None
    code, width = INTEGER_DTYPES[dtype]
    if tensor.tensor_content:
        if len(tensor.tensor_content) % width:
            raise DiagnosticError("misaligned integer tensor_content")
        count = len(tensor.tensor_content) // width
        return list(struct.unpack("<" + code * count, tensor.tensor_content))
    field_by_dtype = {
        3: "int_val",
        4: "int_val",
        5: "int_val",
        6: "int_val",
        9: "int64_val",
        17: "int_val",
        22: "uint32_val",
        23: "uint64_val",
    }
    return [int(value) for value in getattr(tensor, field_by_dtype[dtype])]


def function_name_normal_form(name: str) -> str:
    """Diagnostic-only generated suffix removal; raw names remain reported."""
    if "_grad_" in name:
        return name.rsplit("_", 1)[0]
    return name


def graph_scopes(graph: Any) -> dict[str, tuple[dict[str, Any], dict[str, str]]]:
    scopes: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    top = unique_map(graph.node, lambda node: node.name, label="GraphDef.node")
    scopes["top"] = (top, {})
    for function in graph.library.function:
        nodes = unique_map(function.node_def, lambda node: node.name, label="function nodes")
        argument_sources = {arg.name: "__function_argument__" for arg in function.signature.input_arg}
        scopes[f"function:{function.signature.name}"] = (nodes, argument_sources)
    return scopes


SHAPE_ONLY_OPS = {
    "BroadcastArgs",
    "ConcatV2",
    "ExpandDims",
    "Fill",
    "GatherV2",
    "Pack",
    "Prod",
    "Range",
    "Reshape",
    "Shape",
    "ShapeN",
    "Size",
    "Slice",
    "StridedSlice",
    "Tile",
    "Unpack",
    "ZerosLike",
}


def scope_consumer_index(graph: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for scope_name, (nodes, _) in graph_scopes(graph).items():
        consumers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        downstream: dict[str, set[str]] = defaultdict(set)
        for node in nodes.values():
            for index, raw_input in enumerate(node.input):
                control, source, output_index = edge_source(raw_input)
                consumers[source].append(
                    {
                        "name": node.name,
                        "op": node.op,
                        "input_index": index,
                        "source_output_index": output_index,
                        "control": control,
                    }
                )
                downstream[source].add(node.name)
        result[scope_name] = {
            "nodes": nodes,
            "consumers": consumers,
            "downstream": downstream,
        }
    return result


def const_consumers(
    graph: Any,
    scope_name: str,
    const_name: str,
    *,
    index: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    scopes = index if index is not None else scope_consumer_index(graph)
    scope = scopes[scope_name]
    nodes = scope["nodes"]
    downstream = scope["downstream"]
    direct = list(scope["consumers"].get(const_name, []))
    function_returned = False
    if scope_name.startswith("function:"):
        function_name = scope_name.split(":", 1)[1]
        function = next(
            (item for item in graph.library.function if item.signature.name == function_name),
            None,
        )
        if function is not None:
            function_returned = any(
                edge_source(value)[1] == const_name for value in function.ret.values()
            ) or any(value == const_name for value in function.control_ret.values())
    # Shape values can influence value tensors through reshape/slice dimensions.
    # Report the complete reachable op set and conservatively label mixed use.
    queue = deque(item["name"] for item in direct)
    visited: set[str] = set()
    reachable_ops: set[str] = set()
    called_function_ops: dict[str, list[str]] = {}
    while queue:
        current = queue.popleft()
        if current in visited or current not in nodes:
            continue
        visited.add(current)
        reachable_ops.add(nodes[current].op)
        for function_name in called_functions(nodes[current]):
            function = next(
                (item for item in graph.library.function if item.signature.name == function_name),
                None,
            )
            if function is None:
                called_function_ops[function_name] = ["__missing_function__"]
            else:
                called_function_ops[function_name] = sorted(
                    {node.op for node in function.node_def}
                )
        queue.extend(downstream.get(current, ()))
    shape_only_direct = bool(direct) and all(item["op"] in SHAPE_ONLY_OPS for item in direct)
    consumer_proven_shape_or_control_only = (
        shape_only_direct
        and not called_function_ops
        and not function_returned
        and bool(reachable_ops)
        and reachable_ops.issubset(SHAPE_ONLY_OPS)
    )
    return {
        "direct": direct,
        "reachable_node_count": len(visited),
        "reachable_ops": sorted(reachable_ops),
        "called_function_ops": called_function_ops,
        "function_returned": function_returned,
        "direct_shape_only": shape_only_direct,
        "consumer_proven_shape_or_control_only": consumer_proven_shape_or_control_only,
        "consumer_boundary": (
            "cross_function_ambiguous_or_value_reachable"
            if called_function_ops
            else "function_return_ambiguous"
            if function_returned
            else "shape_or_control_only_proven"
            if consumer_proven_shape_or_control_only
            else "ambiguous_or_value_reachable"
            if visited
            else "unused_no_consumer"
        ),
    }


def const_inventory(graph: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for scope_name, (nodes, _) in graph_scopes(graph).items():
        for name, node in nodes.items():
            if node.op != "Const" or "value" not in node.attr:
                continue
            tensor = node.attr["value"].tensor
            values = tensor_integer_values(tensor)
            if values is None:
                continue
            key = f"{function_name_normal_form(scope_name)}::{name}"
            if key in result:
                raise DiagnosticError(f"duplicate normalized Const key: {key}")
            result[key] = {
                "raw_scope": scope_name,
                "node_name": name,
                "dtype": int(tensor.dtype),
                "tensor_shape": [int(dim.size) for dim in tensor.tensor_shape.dim],
                "values": values,
                "consumer_scope": scope_name,
            }
    return result


def classify_axis(values: Sequence[Sequence[int]], identities: Sequence[Mapping[str, Any]]) -> str:
    if len(values) != 6 or len(identities) != 6:
        return "incomplete_lattice"
    expected_pairs = {(p, b) for p in (50, 150) for b in (1, 4, 16)}
    if {(row["parameter_count"], row["batch_size"]) for row in identities} != expected_pairs:
        return "incomplete_lattice"
    varying_positions = []
    width = len(values[0])
    if any(len(row) != width for row in values):
        return "shape_or_length_change"
    for index in range(width):
        column = [row[index] for row in values]
        if len(set(column)) == 1:
            continue
        if all(value == identity["batch_size"] for value, identity in zip(column, identities)):
            varying_positions.append((index, "B"))
        elif all(value == identity["parameter_count"] for value, identity in zip(column, identities)):
            varying_positions.append((index, "P"))
        elif all(
            value == identity["batch_size"] * identity["parameter_count"]
            for value, identity in zip(column, identities)
        ):
            varying_positions.append((index, "B*P"))
        else:
            return "non_axis"
    if not varying_positions:
        return "constant"
    axes = sorted({axis for _, axis in varying_positions})
    return "+".join(axes)


def analyze_axis_constants(graph_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    inventories = [const_inventory(row["graph"]) for row in graph_rows]
    consumer_indices = [scope_consumer_index(row["graph"]) for row in graph_rows]
    all_keys = sorted(set().union(*(set(inventory) for inventory in inventories)))
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        present = [inventory.get(key) for inventory in inventories]
        if any(item is None for item in present):
            rows.append({"key": key, "classification": "entity_presence_change"})
            continue
        assert all(item is not None for item in present)
        values = [item["values"] for item in present]
        shapes = [item["tensor_shape"] for item in present]
        dtypes = [item["dtype"] for item in present]
        if all(values[0] == value and shapes[0] == shape and dtypes[0] == dtype for value, shape, dtype in zip(values[1:], shapes[1:], dtypes[1:])):
            continue
        axis = classify_axis(values, [row["identity"] for row in graph_rows])
        consumer_analyses = [
            const_consumers(
                graph_row["graph"],
                item["consumer_scope"],
                item["node_name"],
                index=index,
            )
            for graph_row, item, index in zip(
                graph_rows, present, consumer_indices
            )
        ]
        consumer_safe = all(
            analysis["consumer_proven_shape_or_control_only"]
            or analysis["consumer_boundary"] == "unused_no_consumer"
            for analysis in consumer_analyses
        )
        rows.append(
            {
                "key": key,
                "classification": axis,
                "values": values,
                "tensor_shapes": shapes,
                "dtypes": dtypes,
                "consumer_analyses": consumer_analyses,
                "consumer_safety": (
                    "shape_or_control_only_proven"
                    if consumer_safe
                    else "ambiguous_mixed_or_value_path"
                ),
                "safe_normalization_claim": False,
            }
        )
    return {
        "differing_integer_consts": rows,
        "differing_integer_const_count": len(rows),
        "axis_correlated_count": sum(
            row["classification"] in {"B", "P", "B*P", "B+P"} for row in rows
        ),
        "ambiguous_consumer_count": sum(
            row.get("consumer_safety") == "ambiguous_mixed_or_value_path"
            for row in rows
        ),
        "unsafe_or_unclassified_count": sum(
            row["classification"] not in {"B", "P", "B*P", "B+P"}
            or row.get("consumer_safety") == "ambiguous_mixed_or_value_path"
            for row in rows
        ),
    }


def cohort_graph_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    from tensorflow.core.framework import graph_pb2

    rows: list[dict[str, Any]] = []
    decoded_total = 0
    for record in trace["records"]:
        child = record["evidence"]["child_artifact"]["strict_json"]
        graph_record = child["evidence"]["graphdef_bytes"]
        raw = contract.decode_graphdef_bytes_record(
            graph_record, prior_total_decoded_bytes=decoded_total
        )
        decoded_total += len(raw)
        graph = graph_pb2.GraphDef()
        graph.ParseFromString(raw)
        rows.append(
            {
                "identity": dict(record["identity"]),
                "graph_record": {
                    "decoded_bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                },
                "graph": graph,
                "view": graph_view(graph),
            }
        )
        validate_graph_view(graph, rows[-1]["view"])
    if len(rows) != 36 or decoded_total != 8921463:
        raise DiagnosticError("decoded GraphDef roster or total does not match R3 evidence")
    return rows


def pair_key(row: Mapping[str, Any]) -> str:
    identity = row["identity"]
    return f"P={identity['parameter_count']}/B={identity['batch_size']}"


def baseline_residual_summary(comparison: Mapping[str, Any]) -> dict[str, Any]:
    accepted = comparison.get("accepted_differences")
    rejected = comparison.get("rejected_differences")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise DiagnosticError("baseline comparison lacks typed differences")
    canonical = [
        row for row in rejected if row.get("rule_id") == "rejected_canonical_bytes_mismatch"
    ]
    noncanonical = [
        row for row in rejected if row.get("rule_id") != "rejected_canonical_bytes_mismatch"
    ]
    accepted_shape_contract = all(
        row.get("axis") in {"B", "P"}
        and row.get("inside_const") is False
        and row.get("rule_id") == f"static_shape_dimension_{row.get('axis')}"
        for row in accepted
    )
    rejected_const_contract = all(
        row.get("inside_const") is True
        and row.get("rule_id") == "rejected_unclassified_difference"
        for row in noncanonical
    )
    return {
        "accepted_count": len(accepted),
        "accepted_all_declared_shape_axes": accepted_shape_contract,
        "noncanonical_rejected_count": len(noncanonical),
        "noncanonical_rejected_all_inside_const": rejected_const_contract,
        "canonical_mismatch_count": len(canonical),
        "complete_positional_residual_partition": (
            accepted_shape_contract
            and rejected_const_contract
            and len(canonical) == 1
            and len(accepted) + len(noncanonical) + len(canonical)
            == len(accepted) + len(rejected)
        ),
    }


def cohort_analysis(
    rows: Sequence[dict[str, Any]],
    baseline: Mapping[str, Any],
    *,
    dimension: int,
    method: str,
) -> dict[str, Any]:
    selected = sorted(
        [
            row
            for row in rows
            if row["identity"]["dimension"] == dimension
            and row["identity"]["method_id"] == method
        ],
        key=lambda row: (row["identity"]["parameter_count"], row["identity"]["batch_size"]),
    )
    if len(selected) != 6:
        raise DiagnosticError("cohort does not contain six records")
    comparisons: list[dict[str, Any]] = []
    for index in range(1, len(selected)):
        delta = compare_graph_views(selected[0]["view"], selected[index]["view"])
        comparisons.append(
            {"left": pair_key(selected[0]), "right": pair_key(selected[index]), **delta}
        )
    topology_changed = any(row["topology_changed"] for row in comparisons)
    order_changed = any(row["order_changed"] for row in comparisons)
    axis = analyze_axis_constants(selected)
    baseline_residuals = baseline_residual_summary(baseline["comparison"])
    axis_data = axis["axis_correlated_count"] > 0
    unsafe_data = axis["unsafe_or_unclassified_count"] > 0
    if topology_changed and (axis_data or unsafe_data):
        classification = "mixed_causes"
    elif topology_changed:
        classification = "true_structural_specialization_established"
    elif axis_data and unsafe_data:
        classification = "undetermined"
    elif (
        axis_data
        and baseline_residuals["complete_positional_residual_partition"]
        and baseline_residuals["noncanonical_rejected_count"] > 0
    ):
        classification = "expected_axis_data_requires_prospective_rule"
    elif unsafe_data:
        classification = "true_structural_specialization_established"
    elif any(not row["raw_equal"] for row in comparisons):
        classification = "undetermined"
    else:
        classification = "evaluator_alignment_artifact_established"
    accepted = len(baseline["comparison"]["accepted_differences"])
    rejected = len(baseline["comparison"]["rejected_differences"])
    return {
        "dimension": dimension,
        "method_id": method,
        "baseline_positional_counts": {"accepted": accepted, "rejected": rejected},
        "graph_bindings": [
            {"identity": row["identity"], **row["graph_record"]} for row in selected
        ],
        "graph_entity_counts": [row["view"]["entity_count"] for row in selected],
        "graph_raw_token_counts": [row["view"]["raw_token_count"] for row in selected],
        "stable_pair_comparisons": comparisons,
        "baseline_residual_partition": baseline_residuals,
        "axis_constant_analysis": axis,
        "topology_changed": topology_changed,
        "repeated_field_order_changed": order_changed,
        "classification": classification,
        "gate_status": "still_rejected",
    }


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise DiagnosticError("cannot resolve git commit")
    return completed.stdout.strip()


def build_payload(trace_path: Path, output_path: Path) -> dict[str, Any]:
    metadata = trace_path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DiagnosticError("trace input is not a regular non-symlink file")
    if metadata.st_size != TRACE_BYTES or sha256_file(trace_path) != TRACE_SHA256:
        raise DiagnosticError("trace input bytes differ from the reviewed baseline")
    if sha256_file(SUBPLAN) != SUBPLAN_SHA256:
        raise DiagnosticError("reviewed subplan bytes drifted")
    trace = contract.read_bounded_phase6_trace_json(trace_path)
    ledger_checks = contract.phase6_ledger_checks(trace, final=True)
    if not all(ledger_checks.values()):
        raise DiagnosticError(f"trace final ledger checks failed: {ledger_checks}")
    evaluation = contract.evaluate_phase6_trace_census(trace)
    if evaluation["trace_common_valid"] is not False:
        raise DiagnosticError("the current structural rejection did not reproduce")
    baseline_by_key = {
        (row["dimension"], row["method_id"]): row for row in evaluation["cohorts"]
    }
    reproduced = {
        key: (
            len(value["comparison"]["accepted_differences"]),
            len(value["comparison"]["rejected_differences"]),
        )
        for key, value in baseline_by_key.items()
    }
    if reproduced != EXPECTED_COUNTS:
        raise DiagnosticError(f"current evaluator counts drifted: {reproduced}")
    rows = cohort_graph_rows(trace)
    cohorts = [
        cohort_analysis(
            rows,
            baseline_by_key[(dimension, method)],
            dimension=dimension,
            method=method,
        )
        for dimension in (10, 20, 30)
        for method in METHODS
    ]
    classifications = sorted({row["classification"] for row in cohorts})
    if len(classifications) == 1:
        overall = classifications[0]
    elif "true_structural_specialization_established" in classifications or "mixed_causes" in classifications:
        overall = "mixed_causes"
    else:
        overall = "undetermined"
    graph_bindings = sorted(
        [binding for cohort in cohorts for binding in cohort["graph_bindings"]],
        key=lambda row: row["identity"]["identity_id"],
    )
    if len({row["identity"]["identity_id"] for row in graph_bindings}) != 36:
        raise DiagnosticError("diagnostic did not account for 36 unique graphs")
    return {
        "schema": SCHEMA,
        "state": "passed",
        "diagnostic_kind": "offline_preserved_graphdef_attribution",
        "source": {
            "trace_path": str(trace_path),
            "trace_byte_count": metadata.st_size,
            "trace_sha256": TRACE_SHA256,
            "trace_state": trace["state"],
            "trace_update_index": trace["update_index"],
            "trace_record_count": len(trace["records"]),
            "trace_common_valid": evaluation["trace_common_valid"],
            "decoded_graphdef_bytes": evaluation["decoded_bytes"],
            "final_ledger_checks": ledger_checks,
            "subplan_path": str(SUBPLAN.relative_to(ROOT)),
            "subplan_sha256": SUBPLAN_SHA256,
        },
        "baseline_reproduction": {
            "passed": reproduced == EXPECTED_COUNTS,
            "cohorts": [
                {
                    "dimension": dimension,
                    "method_id": method,
                    "accepted": counts[0],
                    "rejected": counts[1],
                }
                for (dimension, method), counts in sorted(reproduced.items())
            ],
        },
        "coverage": {
            "graph_count": len(graph_bindings),
            "unique_identity_count": len(
                {row["identity"]["identity_id"] for row in graph_bindings}
            ),
            "graph_bindings": graph_bindings,
            "all_graph_bytes_bound": True,
            "all_stable_entities_and_raw_tokens_accounted": True,
            "field_coverage_matrix": FIELD_COVERAGE_MATRIX,
        },
        "cohorts": cohorts,
        "overall_classification": overall,
        "decision": {
            "gate_b_status": "still_rejected",
            "gate_c_status": "blocked",
            "runtime_authorized": False,
            "next_action": (
                "draft_graph_structure_localization_subplan"
                if overall in {"mixed_causes", "true_structural_specialization_established"}
                else "draft_prospective_gate_semantics_subplan_with_human_approval"
                if overall == "expected_axis_data_requires_prospective_rule"
                else "write_blocker_and_stop"
            ),
        },
        "nonclaims": NONCLAIMS,
        "run_manifest": {
            "git_commit": git_commit(),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_status": "deliberately_hidden_no_enumeration",
            "xla_status": "not_initialized_or_invoked",
            "tf32_status": "not_queried",
            "data_fixture_version": "N/A_preserved_graphdefs",
            "random_seeds": "N/A_deterministic_diagnostic",
            "output_path": str(output_path),
            "started_utc": None,
            "finished_utc": None,
            "wall_seconds": None,
            "canonical_exclusions": RUN_MANIFEST_EXCLUSIONS,
            "trust_claim_boundary": "offline_engineering_diagnostic_only",
        },
    }


def canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    clone = json.loads(contract.strict_json_dumps(payload))
    manifest = clone["run_manifest"]
    for key in ("started_utc", "finished_utc", "wall_seconds", "output_path"):
        manifest.pop(key, None)
    clone.pop("diagnostic_payload_sha256", None)
    return clone


def validate_payload(payload: Mapping[str, Any]) -> dict[str, bool]:
    cohorts = payload.get("cohorts") if isinstance(payload, Mapping) else None
    coverage = payload.get("coverage") if isinstance(payload, Mapping) else None
    decision = payload.get("decision") if isinstance(payload, Mapping) else None
    expected_top = {
        "schema",
        "state",
        "diagnostic_kind",
        "source",
        "baseline_reproduction",
        "coverage",
        "cohorts",
        "overall_classification",
        "decision",
        "nonclaims",
        "run_manifest",
        "diagnostic_payload_sha256",
    }
    digest = contract.canonical_sha256(canonical_payload(payload)) if isinstance(payload, Mapping) else None
    return {
        "closed_schema": isinstance(payload, Mapping) and set(payload) == expected_top,
        "schema_identity": isinstance(payload, Mapping) and payload.get("schema") == SCHEMA,
        "state_passed": isinstance(payload, Mapping) and payload.get("state") == "passed",
        "source_bound": isinstance(payload, Mapping)
        and payload.get("source", {}).get("trace_sha256") == TRACE_SHA256
        and payload.get("source", {}).get("subplan_sha256") == SUBPLAN_SHA256,
        "baseline_reproduced": isinstance(payload, Mapping)
        and payload.get("baseline_reproduction", {}).get("passed") is True,
        "six_cohorts": isinstance(cohorts, list) and len(cohorts) == 6,
        "complete_graph_coverage": isinstance(coverage, Mapping)
        and coverage.get("graph_count") == 36
        and coverage.get("unique_identity_count") == 36
        and coverage.get("all_graph_bytes_bound") is True
        and coverage.get("all_stable_entities_and_raw_tokens_accounted") is True
        and coverage.get("field_coverage_matrix") == FIELD_COVERAGE_MATRIX,
        "gate_still_blocked": isinstance(decision, Mapping)
        and decision.get("gate_b_status") == "still_rejected"
        and decision.get("gate_c_status") == "blocked"
        and decision.get("runtime_authorized") is False,
        "payload_digest": isinstance(payload, Mapping)
        and payload.get("diagnostic_payload_sha256") == digest,
        "nonclaims_identity": isinstance(payload, Mapping)
        and payload.get("nonclaims") == NONCLAIMS,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise DiagnosticError(f"output must be absent: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(contract.strict_json_dumps(payload, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise DiagnosticError("offline diagnostic requires CUDA_VISIBLE_DEVICES=-1")
    trace_path = (ROOT / args.trace_input).resolve() if not args.trace_input.is_absolute() else args.trace_input
    output_path = (ROOT / args.output_json).resolve() if not args.output_json.is_absolute() else args.output_json
    started_wall = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    payload = build_payload(trace_path, output_path)
    payload["run_manifest"]["started_utc"] = started_utc
    payload["run_manifest"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    payload["run_manifest"]["wall_seconds"] = time.perf_counter() - started_wall
    payload["diagnostic_payload_sha256"] = contract.canonical_sha256(canonical_payload(payload))
    checks = validate_payload(payload)
    if not all(checks.values()):
        raise DiagnosticError(f"diagnostic payload validation failed: {checks}")
    atomic_write_json(output_path, payload)
    reparsed = contract.read_strict_json(output_path)
    if reparsed != payload or not all(validate_payload(reparsed).values()):
        raise DiagnosticError("durable diagnostic reparse failed")
    print(
        contract.strict_json_dumps(
            {
                "diagnostic_payload_sha256": payload["diagnostic_payload_sha256"],
                "overall_classification": payload["overall_classification"],
                "output_path": str(output_path),
                "validation_checks": checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
