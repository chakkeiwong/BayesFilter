#!/usr/bin/env python3
"""Offline attribution of preserved Gate B R3 autodiff GraphDef differences."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from google.protobuf import descriptor_pool, message_factory


ROOT = Path(__file__).resolve().parents[2]
TRACE_ROOT = Path("/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/trace")
DIAGNOSTIC_PATH = ROOT / (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_"
    "trace_rejection_diagnostic_2026-07-12.json"
)
PLAN_PATH = ROOT / (
    "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-"
    "autodiff-structure-localization-subplan-2026-07-13.md"
)
PLAN_SHA256 = "88db6519ca3d1a668ef9565506b539c1bd4cd672f000424c35a4be6d5581a949"
FINAL_REVIEW_PATH = ROOT / (
    "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-"
    "autodiff-structure-localization-subplan-review-final-2026-07-13.md"
)
FINAL_REVIEW_SHA256 = "ab8bc7613ec1b547cba58ccdb3419cf46ec37b6af2be94b267e46b55e331d2eb"
DIAGNOSTIC_SHA256 = "637273af37ed2606b9bd0bc4868a1719a65ad17d89d94ab018e5678082fb25ff"
DIAGNOSTIC_PAYLOAD_SHA256 = "30a2753246d4c86a6952268fad5a49d8e77991084f4100a45d1eca051c710cd7"
AUTHORIZED_SNAPSHOT_SHA256 = (
    "64b677d61e0f76c581e7975990f8f9941ebb2919de4556db29728061e4099e7f"
)
SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6."
    "autodiff_structure_localization.v1"
)
AUTODIFF_METHOD = "batch_native_autodiff_qr_score"
RESULT_PATH = ROOT / (
    "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-"
    "autodiff-structure-localization-result-2026-07-13.md"
)
DURABLE_OUTPUT = ROOT / (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_"
    "autodiff_structure_localization_2026-07-13.json"
)
SCRATCH_ROOT = Path("/tmp/kalman_qr_phase6_gateb_r3_autodiff_structure_localization")
AUTHORIZED_OUTPUTS = {
    (SCRATCH_ROOT / "run1.json").resolve(),
    (SCRATCH_ROOT / "run2.json").resolve(),
    DURABLE_OUTPUT.resolve(),
}

TF_ROOT = Path("/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages")
TF_SCHEMA_ROOT = TF_ROOT / "tensorflow/core/framework"
DESCRIPTOR_HASHES = {
    "types": "da2051ab56bacdd352d423406f61e6a400ee7c5ccd697d438a4af2347a38b950",
    "tensor_shape": "f2270dddfc27b1fc4e73147eab9bac16b494e44bc582fc37c6b73732e898f360",
    "resource_handle": "2cf571a9bbd6933942fa1ff965eef70f6b744278c293dd817993aeb49ae5638d",
    "full_type": "ad3195d1f19f5194092ea95cf2b138a50a8d526fd79acf6715d07a19e15975a8",
    "tensor": "08e6ffb6ada2798aecc04c8ee963b3d2f14a9ca5e1bb2c516937bb0ac0f9ad64",
    "attr_value": "cfabd36cba17dbeb2c7d8a2f12f4212c65a2aa19d011dfcb3940a0a5d37bcdbf",
    "op_def": "9b4fcdb6a51416554488d21b29d3672dff4f58b5054cf08027b818b957977075",
    "node_def": "ba780b9f27bd40b9e405b7aa2b772095f6807674831d19b4b6fc107097a8bad7",
    "function": "ff554bd811b3f068893847b6e85393968f409963225c84034696dc6acbcdcc73",
    "graph_debug_info": "33abded1da2221458edc7e14aa9fff0e286d8b7339be43792bf063e3aae9422c",
    "versions": "d47bf8af3dc59be01e31356403a27285c43d28c27a775bbf476a8e6b1f9a7508",
    "graph": "ef332be91bd9fd86fdb3dd58953b870a14443a73fc3894e5a5ec58792b85d437",
}

SOURCE_HASHES = {
    "scripts/benchmark_kalman_qr_parameter_count_scaling.py":
        "baf62b85f885073d0b72b5c13af0463ac5566f2429c16d5c98a542aa24c8eec9",
    "scripts/kalman_qr_benchmark_contract.py":
        "f52a20624eb3c8c72c59cc2809f4cd870de4c3c84276fed97f308bc4f0a75e64",
    "bayesfilter/__init__.py":
        "986fd24cc5c86812c53fff10f2e169525783921496ab7997d988d5423ff9663b",
    "bayesfilter/diagnostics.py":
        "da00bf6421d55952d6e0a4ee58e4402bfeee414c90d44e2745485b73b73e1fc4",
    "bayesfilter/linear/__init__.py":
        "df9e248a8fc24063112d3bdcfbff0b1e46ef30781c13d05b36d932909c5bb46e",
    "bayesfilter/linear/dtypes_tf.py":
        "de534d5411a372e0344b1248e1c192dcc0206b21a8bef86c13cff15024ef960d",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py":
        "d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57",
    "bayesfilter/linear/kalman_qr_tf.py":
        "ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b",
    "bayesfilter/linear/qr_factor_tf.py":
        "bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401",
    "bayesfilter/linear/types_tf.py":
        "6f79ae42472ecd304e6012bdf3c1fba13e97ac1431d537d8152ecb96133f8af8",
    "bayesfilter/results_tf.py":
        "e09be453fa62e1b3a3ef16c542ca3782f82c1967303509430b5676168e91cea6",
    "bayesfilter/structural.py":
        "3a181ef4c8f9e67ab8b923b5c016d6df3da13e4b77d4e940f0c07a60fc37fd4f",
}

FRAMEWORK_SOURCES = {
    "tensorflow/python/eager/backprop.py": {
        "sha256": "c9a461d06085be50a2235e23e6c32a4649805fa2b5fb43d50e7fb421f92eed78",
        "anchors": [[627, 680], [960, 1072]],
    },
    "tensorflow/python/ops/gradients_util.py": {
        "sha256": "1b4cf14a574b45f708ec4fba67bd450336a6b8e63f494ddf791fc5de5d981e98",
        "anchors": [[506, 750], [858, 883]],
    },
    "tensorflow/python/ops/while_v2.py": {
        "sha256": "756344a1c87911ca4a0678bea1388d070c7218afb1ec34a3f07c03473804719a",
        "anchors": [[322, 437], [535, 580], [713, 730], [970, 1009]],
    },
}
DISTRIBUTION_FILES = {
    "tensorflow-2.20.0.dist-info/METADATA": (
        "aadf1cb4d0afeaaa947c7b32a8e9299cef3261137c16dd710bcf804fb6b4844c"
    ),
    "tensorflow-2.20.0.dist-info/WHEEL": (
        "3a52126eda4371f6a03eb2f01bb5ada5c65b3d3527a0a3e7c29840ff6e9f36a1"
    ),
}

LOCAL_SOURCE_ANCHORS = [
    {
        "anchor_id": "batched_model_tensor_additions",
        "path": "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
        "line_start": 712,
        "line_end": 784,
        "role": "local_source_operation_candidate",
    },
    {
        "anchor_id": "one_vjp_output_gradient",
        "path": "scripts/benchmark_kalman_qr_parameter_count_scaling.py",
        "line_start": 1893,
        "line_end": 1945,
        "role": "local_autodiff_wrapper",
    },
    {
        "anchor_id": "batched_while_value_route",
        "path": "bayesfilter/linear/kalman_qr_tf.py",
        "line_start": 621,
        "line_end": 763,
        "role": "local_forward_while_route",
    },
]

NONCLAIMS = [
    "no avoidable or inherent construction claim",
    "no source bug or evaluator exception claim",
    "no Gate B pass or Gate C runtime authorization",
    "no TensorFlow target execution, tracing, XLA, or GPU evidence",
    "no memory, performance, scalability, ranking, or readiness claim",
]
COVERAGE_STATES = {
    "mapped_exact",
    "enumerated_causally_ambiguous",
    "missing_or_incomplete",
}
FORBIDDEN_CAUSAL_TERMS = (
    "avoidable",
    "inherent",
    "source bug",
    "source_bug",
    "evaluator exception",
    "evaluator_exception",
    "erroneous",
    "benign",
)
RUN_EXCLUDED_PATHS = {
    "run_manifest.finished_utc",
    "run_manifest.output_path",
    "run_manifest.started_utc",
    "run_manifest.wall_seconds",
}


class LocalizationError(RuntimeError):
    """Raised when offline attribution evidence is incomplete or inconsistent."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved not in AUTHORIZED_OUTPUTS:
        raise LocalizationError(f"output path is not authorized: {resolved}")
    if resolved.parent.resolve() != resolved.parent:
        raise LocalizationError(f"output parent is not canonical: {resolved.parent}")
    return resolved


def git_commit(root: Path = ROOT) -> str:
    """Resolve HEAD using only Git administrative files."""
    dot_git = root / ".git"
    if dot_git.is_file():
        marker = dot_git.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise LocalizationError("invalid .git indirection")
        git_dir = Path(marker.split(": ", 1)[1])
        if not git_dir.is_absolute():
            git_dir = (root / git_dir).resolve()
    elif dot_git.is_dir():
        git_dir = dot_git.resolve()
    else:
        raise LocalizationError("Git administrative directory is missing")

    value = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    for _ in range(8):
        if len(value) == 40 and all(character in "0123456789abcdef" for character in value):
            return value
        if not value.startswith("ref: refs/"):
            raise LocalizationError(f"invalid Git HEAD value: {value!r}")
        reference = value.split(" ", 1)[1]
        loose = git_dir / reference
        if loose.is_file():
            value = loose.read_text(encoding="ascii").strip()
            continue
        packed = git_dir / "packed-refs"
        if not packed.is_file():
            raise LocalizationError(f"unresolved Git reference: {reference}")
        matches = []
        for line in packed.read_text(encoding="ascii").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, separator, name = line.partition(" ")
            if separator and name == reference:
                matches.append(commit)
        if len(matches) != 1:
            raise LocalizationError(f"non-unique Git reference: {reference}")
        value = matches[0]
    raise LocalizationError("Git symbolic-reference depth exceeded")


def strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise LocalizationError(f"non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LocalizationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise LocalizationError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def strict_dumps(value: Any, *, indent: int | None = None) -> str:
    _validate_finite(value)
    return json.dumps(
        value,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(strict_dumps(value).encode("utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        raise LocalizationError("non-finite protobuf scalar")
    return value


def scalar_tokens(message: Any, path: str = "$") -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for field, value in message.ListFields():
        field_path = f"{path}.{field.name}"
        if field.is_repeated:
            if field.message_type is not None and field.message_type.GetOptions().map_entry:
                value_field = field.message_type.fields_by_name["value"]
                for key in sorted(value):
                    item = value[key]
                    item_path = f"{field_path}[{json.dumps(key, sort_keys=True)}]"
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
            raise LocalizationError(f"duplicate or empty key in {label}: {key!r}")
        result[key] = item
    return result


def edge_source(raw: str) -> tuple[bool, str, int]:
    control = raw.startswith("^")
    value = raw[1:] if control else raw
    parts = value.split(":")
    if len(parts) >= 2 and parts[-1].isdigit():
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
            field.name: len(item) for field, item in value.list.ListFields()
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
        "structural_sha256": canonical_sha256(structural),
        "raw_sha256": sha256_bytes(raw),
        "token_count": len(tokens),
        "token_sha256": canonical_sha256(tokens),
    }


def function_summary(function: Any) -> dict[str, Any]:
    nodes = unique_map(function.node_def, lambda node: node.name, label="function nodes")
    input_args = unique_map(function.signature.input_arg, lambda arg: arg.name, label="inputs")
    output_args = unique_map(function.signature.output_arg, lambda arg: arg.name, label="outputs")
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
        "token_sha256": canonical_sha256(tokens),
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
    summary["structural_sha256"] = canonical_sha256(structural)
    return summary


def graph_view(graph: Any) -> dict[str, Any]:
    nodes = unique_map(graph.node, lambda node: node.name, label="graph nodes")
    functions = unique_map(
        graph.library.function,
        lambda function: function.signature.name,
        label="functions",
    )
    gradients = unique_map(
        graph.library.gradient, lambda gradient: gradient.function_name, label="gradients"
    )
    registered = unique_map(
        graph.library.registered_gradients,
        lambda gradient: gradient.registered_op_type,
        label="registered gradients",
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
        "raw_token_sha256": canonical_sha256(tokens),
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
        "versions_sha256": sha256_bytes(graph.versions.SerializeToString(deterministic=True)),
        "debug_info_sha256": sha256_bytes(graph.debug_info.SerializeToString(deterministic=True)),
    }


def map_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    common = sorted(left_keys & right_keys)
    return {
        "only_left": sorted(left_keys - right_keys),
        "only_right": sorted(right_keys - left_keys),
        "changed": [key for key in common if left[key] != right[key]],
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
        for field in ("node_order", "function_order", "gradient_order", "registered_gradient_order")
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
        "topology_changed": (
            entity_set_changed
            or bool(changed_node_semantics)
            or bool(changed_function_semantics)
            or order_changed
        ),
        "raw_equal": left["raw_sha256"] == right["raw_sha256"],
    }


def _descriptor_literal(path: Path) -> bytes:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "DESCRIPTOR" for target in node.targets):
            continue
        if len(node.value.args) != 1:
            raise LocalizationError(f"unexpected descriptor constructor in {path}")
        value = ast.literal_eval(node.value.args[0])
        if not isinstance(value, bytes):
            raise LocalizationError(f"descriptor literal is not bytes in {path}")
        candidates.append(value)
    if len(candidates) != 1:
        raise LocalizationError(f"expected one descriptor literal in {path}, got {len(candidates)}")
    return candidates[0]


def descriptor_literals() -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    literals: dict[str, bytes] = {}
    ledger = []
    for name, expected in sorted(DESCRIPTOR_HASHES.items()):
        path = TF_SCHEMA_ROOT / f"{name}_pb2.py"
        actual = sha256_file(path)
        if actual != expected:
            raise LocalizationError(f"descriptor drift: {path}: {actual}")
        literals[name] = _descriptor_literal(path)
        ledger.append(
            {
                "name": name,
                "path": str(path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        )
    return literals, ledger


def graphdef_class(
    literals: Mapping[str, bytes] | None = None,
) -> tuple[type[Any], list[dict[str, Any]]]:
    if literals is None:
        loaded, file_ledger = descriptor_literals()
    else:
        loaded = dict(literals)
        file_ledger = [
            {"name": name, "path": "mutation_fixture", "sha256": sha256_bytes(raw), "size_bytes": len(raw)}
            for name, raw in sorted(loaded.items())
        ]
    if set(loaded) != set(DESCRIPTOR_HASHES):
        raise LocalizationError(
            "descriptor name closure mismatch: "
            f"missing={sorted(set(DESCRIPTOR_HASHES) - set(loaded))}, "
            f"unknown={sorted(set(loaded) - set(DESCRIPTOR_HASHES))}"
        )
    if any(not isinstance(raw, bytes) or not raw for raw in loaded.values()):
        raise LocalizationError("descriptor closure contains empty or non-byte data")

    pool = descriptor_pool.DescriptorPool()
    pending = dict(loaded)
    ledger = []
    while pending:
        progressed = False
        for name, raw in list(pending.items()):
            try:
                descriptor = pool.AddSerializedFile(raw)
            except Exception:
                continue
            ledger.append(
                {
                    "name": name,
                    **next(item for item in file_ledger if item["name"] == name),
                    "proto_name": descriptor.name,
                    "load_index": len(ledger),
                }
            )
            del pending[name]
            progressed = True
        if not progressed:
            raise LocalizationError(f"incomplete descriptor dependency closure: {sorted(pending)}")
    descriptor = pool.FindMessageTypeByName("tensorflow.GraphDef")
    return message_factory.GetMessageClass(descriptor), ledger


def source_ledger() -> list[dict[str, Any]]:
    ledger = []
    for relative, expected in sorted(SOURCE_HASHES.items()):
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise LocalizationError(f"source drift: {relative}: {actual}")
        ledger.append({"path": relative, "sha256": actual, "size_bytes": path.stat().st_size})
    for relative, contract in sorted(FRAMEWORK_SOURCES.items()):
        path = TF_ROOT / relative
        actual = sha256_file(path)
        if actual != contract["sha256"]:
            raise LocalizationError(f"framework source drift: {relative}: {actual}")
        lines = path.read_text(encoding="utf-8").splitlines()
        for start, end in contract["anchors"]:
            if start < 1 or end > len(lines) or start > end:
                raise LocalizationError(f"invalid source anchor {relative}:{start}-{end}")
        ledger.append(
            {
                "path": str(path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
                "anchors": contract["anchors"],
            }
        )
    for relative, expected in sorted(DISTRIBUTION_FILES.items()):
        path = TF_ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise LocalizationError(f"distribution provenance drift: {relative}: {actual}")
        ledger.append(
            {
                "path": str(path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
                "role": "tensorflow_distribution_provenance",
            }
        )
    return ledger


def _source_anchor_record(
    definition: Mapping[str, Any],
    *,
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise LocalizationError(f"source anchor hash drift: {path}")
    start = definition.get("line_start")
    end = definition.get("line_end")
    lines = path.read_text(encoding="utf-8").splitlines()
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 1
        or end < start
        or end > len(lines)
    ):
        raise LocalizationError(f"invalid source anchor range: {path}:{start}-{end}")
    excerpt = "\n".join(lines[start - 1 : end]) + "\n"
    return {
        **dict(definition),
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "source_sha256": expected_sha256,
        "excerpt_sha256": sha256_bytes(excerpt.encode("utf-8")),
        "line_count": end - start + 1,
    }


def source_anchor_ledger() -> dict[str, list[dict[str, Any]]]:
    local = []
    for definition in LOCAL_SOURCE_ANCHORS:
        relative = definition["path"]
        local.append(
            _source_anchor_record(
                definition,
                path=ROOT / relative,
                expected_sha256=SOURCE_HASHES[relative],
            )
        )
    framework = []
    for relative, contract in sorted(FRAMEWORK_SOURCES.items()):
        for start, end in contract["anchors"]:
            framework.append(
                _source_anchor_record(
                    {
                        "anchor_id": f"framework::{relative}:{start}-{end}",
                        "line_start": start,
                        "line_end": end,
                        "role": "installed_framework_construction_candidate",
                    },
                    path=TF_ROOT / relative,
                    expected_sha256=contract["sha256"],
                )
            )
    anchor_ids = [item["anchor_id"] for item in local + framework]
    if len(anchor_ids) != len(set(anchor_ids)):
        raise LocalizationError("duplicate source anchor id")
    return {"local": local, "framework": framework}


def validate_trace_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    identities = [row.get("identity", {}).get("identity_id") for row in rows]
    if (
        len(rows) != 36
        or any(not isinstance(identity, str) or not identity for identity in identities)
        or len(set(identities)) != 36
    ):
        raise LocalizationError(f"expected 36 unique trace rows, got {len(rows)}")


def trace_rows(graph_class: type[Any]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(TRACE_ROOT.glob("*.json")):
        record = strict_load(path)
        evidence = record.get("evidence") if isinstance(record, Mapping) else None
        if not isinstance(evidence, Mapping) or "graphdef_bytes" not in evidence:
            continue
        graph_record = evidence["graphdef_bytes"]
        raw = base64.b64decode(graph_record["base64"], validate=True)
        if graph_record.get("encoding") != "base64-rfc4648":
            raise LocalizationError(f"unexpected graph encoding in {path}")
        if len(raw) != graph_record.get("decoded_bytes"):
            raise LocalizationError(f"decoded length mismatch in {path}")
        if sha256_bytes(raw) != graph_record.get("sha256"):
            raise LocalizationError(f"decoded hash mismatch in {path}")
        graph = graph_class()
        graph.ParseFromString(raw)
        if graph.SerializeToString(deterministic=True) != raw:
            raise LocalizationError(f"non-canonical or lossy GraphDef decode in {path}")
        identity = dict(record["identity"])
        if identity != evidence["identity"] or record.get("state") != "passed":
            raise LocalizationError(f"trace identity/state mismatch in {path}")
        rows.append(
            {
                "path": str(path),
                "identity": identity,
                "graph": graph,
                "raw_sha256": sha256_bytes(raw),
                "decoded_bytes": len(raw),
                "view": graph_view(graph),
            }
        )
    validate_trace_rows(rows)
    return rows


def pair_key(row: Mapping[str, Any]) -> str:
    identity = row["identity"]
    return f"P={identity['parameter_count']}/B={identity['batch_size']}"


def cohort_rows(rows: Sequence[dict[str, Any]], dimension: int) -> list[dict[str, Any]]:
    selected = sorted(
        [
            row
            for row in rows
            if row["identity"]["dimension"] == dimension
            and row["identity"]["method_id"] == AUTODIFF_METHOD
        ],
        key=lambda row: (row["identity"]["parameter_count"], row["identity"]["batch_size"]),
    )
    if len(selected) != 6:
        raise LocalizationError(f"dimension {dimension} does not have six autodiff graphs")
    return selected


def parity(rows: Sequence[dict[str, Any]], diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    binding_rows = diagnostic["coverage"]["graph_bindings"]
    binding_ids = [row["identity"]["identity_id"] for row in binding_rows]
    if len(binding_rows) != 36 or len(set(binding_ids)) != 36:
        raise LocalizationError("diagnostic graph bindings are not 36 unique identities")
    expected_bindings = {
        row["identity"]["identity_id"]: (row["sha256"], row["decoded_bytes"])
        for row in binding_rows
    }
    observed_bindings = {
        row["identity"]["identity_id"]: (row["raw_sha256"], row["decoded_bytes"])
        for row in rows
    }
    binding_match = observed_bindings == expected_bindings
    cohort_checks = []
    for expected in diagnostic["cohorts"]:
        selected = sorted(
            [
                row
                for row in rows
                if row["identity"]["dimension"] == expected["dimension"]
                and row["identity"]["method_id"] == expected["method_id"]
            ],
            key=lambda row: (row["identity"]["parameter_count"], row["identity"]["batch_size"]),
        )
        comparisons = [
            {"left": pair_key(selected[0]), "right": pair_key(row), **compare_graph_views(selected[0]["view"], row["view"])}
            for row in selected[1:]
        ]
        cohort_checks.append(
            {
                "dimension": expected["dimension"],
                "method_id": expected["method_id"],
                "entity_counts_match": [row["view"]["entity_count"] for row in selected]
                == expected["graph_entity_counts"],
                "raw_token_counts_match": [row["view"]["raw_token_count"] for row in selected]
                == expected["graph_raw_token_counts"],
                "stable_pair_comparisons_match": comparisons
                == expected["stable_pair_comparisons"],
            }
        )
    checks = {
        "all_36_bindings_match": binding_match,
        "all_entity_counts_match": all(row["entity_counts_match"] for row in cohort_checks),
        "all_raw_token_counts_match": all(row["raw_token_counts_match"] for row in cohort_checks),
        "all_stable_pair_comparisons_match": all(
            row["stable_pair_comparisons_match"] for row in cohort_checks
        ),
    }
    if not all(checks.values()):
        raise LocalizationError(f"decoder parity failed: {checks}")
    return {"checks": checks, "cohorts": cohort_checks}


def canonical_diagnostic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(strict_dumps(payload))
    manifest = result.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise LocalizationError("diagnostic run manifest is missing")
    for key in ("started_utc", "finished_utc", "wall_seconds", "output_path"):
        manifest.pop(key, None)
    result.pop("diagnostic_payload_sha256", None)
    return result


def validate_diagnostic_payload(diagnostic: Mapping[str, Any]) -> None:
    digest = canonical_sha256(canonical_diagnostic_payload(diagnostic))
    if digest != DIAGNOSTIC_PAYLOAD_SHA256:
        raise LocalizationError(f"diagnostic canonical payload mismatch: {digest}")
    if diagnostic.get("diagnostic_payload_sha256") != digest:
        raise LocalizationError("diagnostic embedded payload digest mismatch")
    if (
        diagnostic.get("state") != "passed"
        or diagnostic.get("overall_classification") != "mixed_causes"
        or diagnostic.get("diagnostic_kind") != "offline_preserved_graphdef_attribution"
    ):
        raise LocalizationError("diagnostic state or kind changed")
    cohorts = diagnostic.get("cohorts")
    identities = [
        (cohort.get("dimension"), cohort.get("method_id"))
        for cohort in cohorts
    ] if isinstance(cohorts, list) else []
    expected = {
        (dimension, method)
        for dimension in (10, 20, 30)
        for method in (
            "batch_native_analytical_qr_score",
            AUTODIFF_METHOD,
        )
    }
    if len(identities) != 6 or set(identities) != expected:
        raise LocalizationError("diagnostic cohort identity closure changed")


def graph_scope(graph: Any, scope: str) -> Mapping[str, Any]:
    if scope == "top":
        return unique_map(graph.node, lambda node: node.name, label="top nodes")
    if not scope.startswith("function:"):
        raise LocalizationError(f"invalid graph scope {scope!r}")
    name = scope.split(":", 1)[1]
    function = next((fn for fn in graph.library.function if fn.signature.name == name), None)
    if function is None:
        raise LocalizationError(f"missing function scope {scope}")
    return unique_map(function.node_def, lambda node: node.name, label=scope)


def neighborhood(graph: Any, scope: str, node_name: str) -> dict[str, Any]:
    nodes = graph_scope(graph, scope)
    if node_name not in nodes:
        raise LocalizationError(f"missing node {scope}::{node_name}")
    node = nodes[node_name]
    consumers = []
    for other in nodes.values():
        for input_index, raw in enumerate(other.input):
            control, source, output_index = edge_source(raw)
            if source == node_name:
                consumers.append(
                    {
                        "name": other.name,
                        "op": other.op,
                        "input_index": input_index,
                        "source_output_index": output_index,
                        "control": control,
                    }
                )
    producers = []
    for raw in node.input:
        control, source, output_index = edge_source(raw)
        producer = nodes.get(source)
        producers.append(
            {
                "name": source,
                "op": producer.op if producer is not None else "__argument_or_external__",
                "output_index": output_index,
                "control": control,
            }
        )
    return {
        "scope": scope,
        "node": node_summary(node),
        "producers": producers,
        "consumers": sorted(consumers, key=lambda item: (item["name"], item["input_index"])),
    }


def _scope_for_normalized_key(graph: Any, normalized_scope: str) -> str | None:
    if normalized_scope == "top":
        return "top"
    if not normalized_scope.startswith("function:"):
        return None
    prefix = normalized_scope.split(":", 1)[1]
    candidates = [
        fn.signature.name
        for fn in graph.library.function
        if fn.signature.name == prefix
        or ("_grad_" in fn.signature.name and fn.signature.name.rsplit("_", 1)[0] == prefix)
    ]
    if len(candidates) == 1:
        return f"function:{candidates[0]}"
    return None


def target_occurrences(
    selected: Sequence[dict[str, Any]],
    *,
    scope: str,
    node_name: str,
) -> list[dict[str, Any]]:
    occurrences = []
    for row in selected:
        concrete_scope = _scope_for_normalized_key(row["graph"], scope)
        if concrete_scope is None:
            continue
        nodes = graph_scope(row["graph"], concrete_scope)
        if node_name not in nodes:
            continue
        occurrences.append(
            {
                "identity": row["identity"],
                "graph_sha256": row["raw_sha256"],
                "neighborhood": neighborhood(row["graph"], concrete_scope, node_name),
            }
        )
    return occurrences


def _function_by_name(graph: Any, function_name: str) -> Any | None:
    matches = [
        function
        for function in graph.library.function
        if function.signature.name == function_name
    ]
    if len(matches) > 1:
        raise LocalizationError(f"duplicate function name {function_name!r}")
    return matches[0] if matches else None


def function_occurrence(row: Mapping[str, Any], function_name: str) -> dict[str, Any]:
    function = _function_by_name(row["graph"], function_name)
    if function is None:
        raise LocalizationError(f"missing function occurrence {function_name!r}")
    callers = []
    scopes = [("top", row["graph"].node)] + [
        (f"function:{owner.signature.name}", owner.node_def)
        for owner in row["graph"].library.function
    ]
    for scope, nodes in scopes:
        for node in nodes:
            if function_name in called_functions(node):
                callers.append(
                    {
                        "scope": scope,
                        "name": node.name,
                        "op": node.op,
                        "called_functions": called_functions(node),
                    }
                )
    return {
        "identity": row["identity"],
        "graph_sha256": row["raw_sha256"],
        "function": function_summary(function),
        "callers": sorted(callers, key=lambda item: (item["scope"], item["name"])),
    }


def gradient_occurrence(
    row: Mapping[str, Any],
    *,
    name: str,
    registered: bool,
) -> dict[str, Any]:
    collection = (
        row["graph"].library.registered_gradients
        if registered
        else row["graph"].library.gradient
    )
    key = "registered_op_type" if registered else "function_name"
    matches = [item for item in collection if getattr(item, key) == name]
    if len(matches) != 1:
        raise LocalizationError(f"expected one gradient occurrence {name!r}")
    value = matches[0]
    return {
        "identity": row["identity"],
        "graph_sha256": row["raw_sha256"],
        "raw_sha256": sha256_bytes(value.SerializeToString(deterministic=True)),
        "tokens": scalar_tokens(value),
    }


def order_occurrence(row: Mapping[str, Any]) -> dict[str, Any]:
    view = row["view"]
    orders = {
        field: view[field]
        for field in (
            "node_order",
            "function_order",
            "gradient_order",
            "registered_gradient_order",
        )
    }
    return {
        "identity": row["identity"],
        "graph_sha256": row["raw_sha256"],
        "orders": orders,
        "orders_sha256": canonical_sha256(orders),
    }


def _pair_sides(
    selected: Sequence[dict[str, Any]],
    pair: Mapping[str, Any],
    delta_kind: str,
) -> list[tuple[str, dict[str, Any]]]:
    side_names = (
        [pair["left"], pair["right"]]
        if delta_kind == "changed"
        else [pair["left"]] if delta_kind == "only_left" else [pair["right"]]
    )
    rows_by_key = {pair_key(row): row for row in selected}
    if len(rows_by_key) != len(selected) or any(side not in rows_by_key for side in side_names):
        raise LocalizationError("pair side does not bind uniquely to a graph")
    return [(side, rows_by_key[side]) for side in side_names]


def expected_target_keys(diagnostic: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for cohort in diagnostic["cohorts"]:
        if cohort["method_id"] != AUTODIFF_METHOD:
            continue
        dimension = cohort["dimension"]
        for pair in cohort["stable_pair_comparisons"]:
            for delta_kind in ("only_left", "only_right", "changed"):
                keys.update(
                    f"top_node::{name}" for name in pair["node_delta"][delta_kind]
                )
                keys.update(
                    f"function::{name}" for name in pair["function_delta"][delta_kind]
                )
                for function_name, delta in pair["changed_function_nodes"].items():
                    keys.update(
                        f"function_body::{function_name}::{name}"
                        for name in delta[delta_kind]
                    )
                keys.update(
                    f"gradient::{name}" for name in pair["gradient_delta"][delta_kind]
                )
                keys.update(
                    f"registered_gradient::{name}"
                    for name in pair["registered_gradient_delta"][delta_kind]
                )
            if pair["order_changed"]:
                keys.add(
                    f"graph_order::d={dimension}/{pair['left']}->{pair['right']}"
                )
        keys.update(
            f"integer_constant::{record['key']}"
            for record in cohort["axis_constant_analysis"]["differing_integer_consts"]
        )
    if not keys:
        raise LocalizationError("expected target universe is empty")
    return keys


def causal_alternatives(entity_kind: str, name: str, op: str | None) -> list[dict[str, Any]]:
    alternatives = [
        {
            "alternative_id": "local_broadcast_or_vjp_shape_specialization",
            "anchors": [
                "scripts/benchmark_kalman_qr_parameter_count_scaling.py:712",
                "scripts/benchmark_kalman_qr_parameter_count_scaling.py:1893",
            ],
            "status": "candidate_not_promoted",
        },
        {
            "alternative_id": "tensorflow_reverse_while_gradient_generation",
            "anchors": [
                "/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/python/ops/while_v2.py:322",
                "/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/python/ops/while_v2.py:713",
            ],
            "status": "candidate_not_promoted",
        },
    ]
    if op in {"Shape", "Const", "ZerosLike", "Fill"} or "Shape" in name or "zeros" in name:
        alternatives.append(
            {
                "alternative_id": "tensorflow_zero_or_shape_materialization",
                "anchors": [
                    "/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/python/eager/backprop.py:627",
                    "/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/python/ops/gradients_util.py:858",
                    "/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/python/ops/while_v2.py:535",
                ],
                "status": "candidate_not_promoted",
            }
        )
    if entity_kind in {
        "function",
        "function_body",
        "gradient",
        "registered_gradient",
        "graph_order",
    }:
        alternatives.append(
            {
                "alternative_id": "generated_function_rewrite_or_capture_specialization",
                "anchors": [
                    "/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/python/ops/while_v2.py:367",
                    "/home/ubuntu/anaconda3/envs/tfgpu/lib/python3.13/site-packages/tensorflow/python/ops/while_v2.py:970",
                ],
                "status": "candidate_not_promoted",
            }
        )
    return alternatives


def _anchor_reference_is_valid(
    reference: str,
    source_anchors: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    path_text, separator, line_text = reference.rpartition(":")
    if not separator or not line_text.isdigit():
        return False
    line = int(line_text)
    for anchor in [*source_anchors["local"], *source_anchors["framework"]]:
        if anchor["path"] == path_text and anchor["line_start"] <= line <= anchor["line_end"]:
            return True
    return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_neighborhood(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "scope",
        "node",
        "producers",
        "consumers",
    }:
        return False
    node = value["node"]
    required_node_fields = {
        "name",
        "op",
        "device",
        "inputs",
        "parsed_inputs",
        "called_functions",
        "attribute_keys",
        "structural_sha256",
        "raw_sha256",
        "token_count",
        "token_sha256",
    }
    return (
        isinstance(value["scope"], str)
        and bool(value["scope"])
        and isinstance(node, Mapping)
        and set(node) == required_node_fields
        and isinstance(node["name"], str)
        and bool(node["name"])
        and _is_sha256(node["structural_sha256"])
        and _is_sha256(node["raw_sha256"])
        and _is_sha256(node["token_sha256"])
        and isinstance(value["producers"], list)
        and isinstance(value["consumers"], list)
    )


def _validate_target_observations(target: Mapping[str, Any]) -> None:
    observations = target["observations"]
    if len({strict_dumps(item) for item in observations}) != len(observations):
        raise LocalizationError(f"duplicate observation for {target['target_key']}")
    grouped_sides: dict[tuple[Any, Any, Any], set[str]] = defaultdict(set)
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise LocalizationError(f"invalid observation for {target['target_key']}")
        kind = target["entity_kind"]
        if kind in {"top_node", "function_body"}:
            if (
                not _is_sha256(observation.get("graph_sha256"))
                or not _valid_neighborhood(observation.get("neighborhood"))
            ):
                raise LocalizationError(f"incomplete graph slice for {target['target_key']}")
        elif kind == "function":
            occurrence = observation.get("occurrence")
            function = occurrence.get("function") if isinstance(occurrence, Mapping) else None
            if (
                not isinstance(function, Mapping)
                or function.get("name") != target["name"]
                or not _is_sha256(function.get("raw_sha256"))
                or not _is_sha256(function.get("structural_sha256"))
                or not isinstance(occurrence.get("callers"), list)
                or not _is_sha256(occurrence.get("graph_sha256"))
            ):
                raise LocalizationError(f"incomplete function slice for {target['target_key']}")
        elif kind in {"gradient", "registered_gradient"}:
            occurrence = observation.get("occurrence")
            if (
                not isinstance(occurrence, Mapping)
                or not _is_sha256(occurrence.get("graph_sha256"))
                or not _is_sha256(occurrence.get("raw_sha256"))
                or not isinstance(occurrence.get("tokens"), list)
            ):
                raise LocalizationError(f"incomplete gradient slice for {target['target_key']}")
        elif kind == "graph_order":
            occurrence = observation.get("occurrence")
            if (
                not isinstance(occurrence, Mapping)
                or not _is_sha256(occurrence.get("graph_sha256"))
                or not _is_sha256(occurrence.get("orders_sha256"))
                or set(occurrence.get("orders", {}))
                != {
                    "node_order",
                    "function_order",
                    "gradient_order",
                    "registered_gradient_order",
                }
            ):
                raise LocalizationError(f"incomplete order slice for {target['target_key']}")
        elif kind == "integer_constant":
            record = observation.get("diagnostic_record")
            occurrences = observation.get("occurrences")
            if (
                not isinstance(record, Mapping)
                or record.get("key") != target["target_key"].split("::", 1)[1]
                or not isinstance(record.get("classification"), str)
                or not isinstance(occurrences, list)
                or not occurrences
                or any(
                    not isinstance(occurrence, Mapping)
                    or not _is_sha256(occurrence.get("graph_sha256"))
                    or not _valid_neighborhood(occurrence.get("neighborhood"))
                    for occurrence in occurrences
                )
            ):
                raise LocalizationError(f"incomplete constant slice for {target['target_key']}")
        else:
            raise LocalizationError(f"unknown target kind for {target['target_key']}")

        if "pair" in observation:
            pair_side = observation.get("pair_side")
            if not isinstance(pair_side, str) or not pair_side:
                raise LocalizationError(f"missing pair side for {target['target_key']}")
            grouped_sides[
                (
                    observation.get("dimension"),
                    observation.get("pair"),
                    observation.get("delta_kind"),
                )
            ].add(pair_side)

    for (_, _, delta_kind), sides in grouped_sides.items():
        expected_count = 2 if delta_kind in {"changed", "order_changed"} else 1
        if len(sides) != expected_count:
            raise LocalizationError(f"incomplete pair sides for {target['target_key']}")


def validate_partition(
    partition: Mapping[str, Any],
    expected_keys: set[str],
    source_anchors: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    targets = partition.get("targets")
    if not isinstance(targets, list) or not targets:
        raise LocalizationError("target partition is absent or empty")
    keys = [target.get("target_key") for target in targets]
    if any(not isinstance(key, str) or not key for key in keys):
        raise LocalizationError("target identity is missing")
    if len(keys) != len(set(keys)):
        raise LocalizationError("duplicate target identity")
    if set(keys) != expected_keys:
        raise LocalizationError(
            "target universe mismatch: "
            f"missing={sorted(expected_keys - set(keys))[:10]}, "
            f"unexpected={sorted(set(keys) - expected_keys)[:10]}"
        )

    counts = {state: 0 for state in COVERAGE_STATES}
    for target in targets:
        state = target.get("coverage_state")
        if state not in COVERAGE_STATES:
            raise LocalizationError(f"invalid coverage state for {target['target_key']}")
        counts[state] += 1
        observations = target.get("observations")
        if not isinstance(observations, list) or not observations:
            raise LocalizationError(f"missing observations for {target['target_key']}")
        _validate_target_observations(target)
        serialized_target = strict_dumps(target).lower()
        if any(term in serialized_target for term in FORBIDDEN_CAUSAL_TERMS):
            raise LocalizationError(f"forbidden causal claim for {target['target_key']}")
        if target.get("causal_claim") != "none":
            raise LocalizationError(f"unsupported causal claim for {target['target_key']}")

        alternatives = target.get("causal_alternatives")
        exact_anchor = target.get("exact_anchor")
        if state == "enumerated_causally_ambiguous":
            if not isinstance(alternatives, list) or len(alternatives) < 2:
                raise LocalizationError(f"ambiguity is not bounded for {target['target_key']}")
            alternative_ids = [item.get("alternative_id") for item in alternatives]
            if (
                any(not isinstance(value, str) or not value for value in alternative_ids)
                or len(alternative_ids) != len(set(alternative_ids))
                or exact_anchor is not None
            ):
                raise LocalizationError(f"invalid ambiguous mapping for {target['target_key']}")
            for alternative in alternatives:
                anchors = alternative.get("anchors")
                if (
                    alternative.get("status") != "candidate_not_promoted"
                    or not isinstance(anchors, list)
                    or not anchors
                    or not all(
                        isinstance(anchor, str)
                        and _anchor_reference_is_valid(anchor, source_anchors)
                        for anchor in anchors
                    )
                ):
                    raise LocalizationError(
                        f"stale or promoted alternative for {target['target_key']}"
                    )
        elif state == "mapped_exact":
            if alternatives not in ([], None):
                raise LocalizationError(f"exact target retains alternatives: {target['target_key']}")
            if not isinstance(exact_anchor, Mapping) or set(exact_anchor) != {
                "anchor",
                "candidate_anchors",
                "candidate_count",
                "graph_evidence_sha256",
                "uniqueness_evidence",
            }:
                raise LocalizationError(f"exact target lacks unique anchor: {target['target_key']}")
            if (
                not _anchor_reference_is_valid(exact_anchor["anchor"], source_anchors)
                or exact_anchor["candidate_count"] != 1
                or exact_anchor["candidate_anchors"] != [exact_anchor["anchor"]]
                or not _is_sha256(exact_anchor["graph_evidence_sha256"])
                or not isinstance(exact_anchor["uniqueness_evidence"], str)
                or not exact_anchor["uniqueness_evidence"]
            ):
                raise LocalizationError(f"false exact mapping for {target['target_key']}")
        else:
            if alternatives not in ([], None) or exact_anchor is not None:
                raise LocalizationError(f"incomplete target carries a causal mapping: {target['target_key']}")

    recorded_counts = partition.get("coverage_counts")
    if recorded_counts != counts or partition.get("target_count") != len(targets):
        raise LocalizationError("partition counts do not match target identities")
    complete = counts["missing_or_incomplete"] == 0
    all_exact = counts["mapped_exact"] == len(targets)
    repair_eligible = complete and all_exact and partition.get("repair_hypothesis") is not None
    discriminator_eligible = complete and not repair_eligible
    expected_branch = (
        "autodiff_source_counterfactual_repair"
        if repair_eligible
        else "autodiff_attribution_discriminator"
        if discriminator_eligible
        else "blocker"
    )
    if (
        partition.get("complete") is not complete
        or partition.get("repair_eligible") is not repair_eligible
        or partition.get("discriminator_eligible") is not discriminator_eligible
        or partition.get("next_branch") != expected_branch
    ):
        raise LocalizationError("partition handoff predicates are inconsistent")


def target_partition(rows: Sequence[dict[str, Any]], diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    targets: dict[str, dict[str, Any]] = {}

    def add_target(key: str, payload: dict[str, Any]) -> None:
        existing = targets.get(key)
        if existing is None:
            targets[key] = payload
            return
        existing_identity = {field: value for field, value in existing.items() if field != "observations"}
        payload_identity = {field: value for field, value in payload.items() if field != "observations"}
        if existing_identity != payload_identity:
            raise LocalizationError(f"conflicting target identity for {key}")
        existing["observations"].extend(payload["observations"])
        existing["observations"] = sorted(
            {strict_dumps(item): item for item in existing["observations"]}.values(),
            key=strict_dumps,
        )

    diagnostic_by_dimension = {
        cohort["dimension"]: cohort
        for cohort in diagnostic["cohorts"]
        if cohort["method_id"] == AUTODIFF_METHOD
    }
    for dimension in (10, 20, 30):
        selected = cohort_rows(rows, dimension)
        expected = diagnostic_by_dimension[dimension]
        for pair in expected["stable_pair_comparisons"]:
            pair_id = f"d={dimension}/{pair['left']}->{pair['right']}"
            for delta_kind in ("only_left", "only_right", "changed"):
                for name in pair["node_delta"][delta_kind]:
                    observations = [
                        {
                            "dimension": dimension,
                            "pair": pair_id,
                            "pair_side": side,
                            "delta_kind": delta_kind,
                            "identity": row["identity"],
                            "graph_sha256": row["raw_sha256"],
                            "neighborhood": neighborhood(row["graph"], "top", name),
                        }
                        for side, row in _pair_sides(selected, pair, delta_kind)
                    ]
                    add_target(
                        f"top_node::{name}",
                        {
                            "target_key": f"top_node::{name}",
                            "entity_kind": "top_node",
                            "name": name,
                            "observations": observations,
                        },
                    )
                for name in pair["function_delta"][delta_kind]:
                    observations = [
                        {
                            "dimension": dimension,
                            "pair": pair_id,
                            "pair_side": side,
                            "delta_kind": delta_kind,
                            "occurrence": function_occurrence(row, name),
                        }
                        for side, row in _pair_sides(selected, pair, delta_kind)
                    ]
                    add_target(
                        f"function::{name}",
                        {
                            "target_key": f"function::{name}",
                            "entity_kind": "function",
                            "name": name,
                            "observations": observations,
                        },
                    )
                for function_name, delta in pair["changed_function_nodes"].items():
                    for name in delta[delta_kind]:
                        observations = [
                            {
                                "dimension": dimension,
                                "pair": pair_id,
                                "pair_side": side,
                                "delta_kind": delta_kind,
                                "identity": row["identity"],
                                "graph_sha256": row["raw_sha256"],
                                "neighborhood": neighborhood(
                                    row["graph"], f"function:{function_name}", name
                                ),
                            }
                            for side, row in _pair_sides(selected, pair, delta_kind)
                        ]
                        target_key = f"function_body::{function_name}::{name}"
                        add_target(
                            target_key,
                            {
                                "target_key": target_key,
                                "entity_kind": "function_body",
                                "name": name,
                                "function_name": function_name,
                                "observations": observations,
                            },
                        )
                for gradient_kind, registered in (
                    ("gradient", False),
                    ("registered_gradient", True),
                ):
                    delta = (
                        pair["registered_gradient_delta"]
                        if registered
                        else pair["gradient_delta"]
                    )
                    for name in delta[delta_kind]:
                        observations = [
                            {
                                "dimension": dimension,
                                "pair": pair_id,
                                "pair_side": side,
                                "delta_kind": delta_kind,
                                "occurrence": gradient_occurrence(
                                    row, name=name, registered=registered
                                ),
                            }
                            for side, row in _pair_sides(selected, pair, delta_kind)
                        ]
                        add_target(
                            f"{gradient_kind}::{name}",
                            {
                                "target_key": f"{gradient_kind}::{name}",
                                "entity_kind": gradient_kind,
                                "name": name,
                                "observations": observations,
                            },
                        )
            if pair["order_changed"]:
                order_key = f"graph_order::{pair_id}"
                add_target(
                    order_key,
                    {
                        "target_key": order_key,
                        "entity_kind": "graph_order",
                        "name": pair_id,
                        "observations": [
                            {
                                "dimension": dimension,
                                "pair": pair_id,
                                "pair_side": side,
                                "delta_kind": "order_changed",
                                "occurrence": order_occurrence(row),
                            }
                            for side, row in _pair_sides(selected, pair, "changed")
                        ],
                    },
                )
        for constant in expected["axis_constant_analysis"]["differing_integer_consts"]:
            scope, node_name = constant["key"].rsplit("::", 1)
            occurrences = target_occurrences(selected, scope=scope, node_name=node_name)
            constant_key = f"integer_constant::{constant['key']}"
            add_target(
                constant_key,
                {
                    "target_key": constant_key,
                    "entity_kind": "integer_constant",
                    "name": node_name,
                    "normalized_scope": scope,
                    "observations": [
                        {
                            "dimension": dimension,
                            "diagnostic_record": constant,
                            "occurrences": occurrences,
                        }
                    ],
                },
            )

    classified = []
    for target in sorted(targets.values(), key=lambda item: item["target_key"]):
        observations = target["observations"]
        missing = not observations
        for observation in observations:
            if "neighborhood" in observation and not observation["neighborhood"].get("node"):
                missing = True
            if target["entity_kind"] in {
                "function",
                "gradient",
                "registered_gradient",
                "graph_order",
            } and not observation.get("occurrence"):
                missing = True
            if target["entity_kind"] == "integer_constant":
                if not observation.get("occurrences"):
                    missing = True
        op = None
        for observation in observations:
            if "neighborhood" in observation:
                op = observation["neighborhood"]["node"]["op"]
                break
            for occurrence in observation.get("occurrences", []):
                op = occurrence["neighborhood"]["node"]["op"]
                break
        alternatives = causal_alternatives(target["entity_kind"], target["name"], op)
        target["coverage_state"] = (
            "missing_or_incomplete" if missing else "enumerated_causally_ambiguous"
        )
        target["causal_alternatives"] = alternatives if not missing else []
        target["exact_anchor"] = None
        target["causal_claim"] = "none"
        classified.append(target)

    counts = {state: 0 for state in COVERAGE_STATES}
    for target in classified:
        counts[target["coverage_state"]] += 1
    if sum(counts.values()) != len(classified) or not classified:
        raise LocalizationError("target partition is empty or inconsistent")
    complete = counts["missing_or_incomplete"] == 0
    repair_hypothesis = None
    repair_eligible = (
        complete
        and counts["mapped_exact"] == len(classified)
        and repair_hypothesis is not None
    )
    discriminator_eligible = complete and not repair_eligible and bool(classified)
    branch = (
        "autodiff_source_counterfactual_repair"
        if repair_eligible
        else "autodiff_attribution_discriminator"
        if discriminator_eligible
        else "blocker"
    )
    return {
        "targets": classified,
        "target_count": len(classified),
        "coverage_counts": dict(sorted(counts.items())),
        "complete": complete,
        "repair_eligible": repair_eligible,
        "discriminator_eligible": discriminator_eligible,
        "next_branch": branch,
        "repair_hypothesis": repair_hypothesis,
    }


def canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(strict_dumps(payload))
    for dotted in sorted(RUN_EXCLUDED_PATHS):
        parent, key = dotted.split(".", 1)
        result[parent].pop(key, None)
    result.pop("localization_payload_sha256", None)
    return result


def build_payload(output_path: Path) -> dict[str, Any]:
    output_path = validate_output_path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise LocalizationError(f"output path must be absent: {output_path}")
    if not output_path.parent.is_dir():
        raise LocalizationError(f"output parent is absent: {output_path.parent}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LocalizationError("CUDA_VISIBLE_DEVICES must equal -1")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    if sha256_file(PLAN_PATH) != PLAN_SHA256:
        raise LocalizationError("reviewed plan drift")
    if sha256_file(FINAL_REVIEW_PATH) != FINAL_REVIEW_SHA256:
        raise LocalizationError("final plan review drift")
    if sha256_file(DIAGNOSTIC_PATH) != DIAGNOSTIC_SHA256:
        raise LocalizationError("diagnostic drift")
    snapshot_path = ROOT / (
        "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_"
        "autodiff_structure_localization_authorized_snapshot_2026-07-13.json"
    )
    if sha256_file(snapshot_path) != AUTHORIZED_SNAPSHOT_SHA256:
        raise LocalizationError("authorized snapshot drift")
    diagnostic = strict_load(DIAGNOSTIC_PATH)
    validate_diagnostic_payload(diagnostic)
    GraphDef, descriptor_ledger = graphdef_class()
    sources = source_ledger()
    source_anchors = source_anchor_ledger()
    rows = trace_rows(GraphDef)
    decoder_parity = parity(rows, diagnostic)
    partition = target_partition(rows, diagnostic)
    declared_target_keys = expected_target_keys(diagnostic)
    validate_partition(partition, declared_target_keys, source_anchors)
    checks = {
        "all_decoder_parity_checks": all(decoder_parity["checks"].values()),
        "all_graphs_bound": len(rows) == 36,
        "all_targets_partitioned": partition["target_count"]
        == sum(partition["coverage_counts"].values())
        == len(declared_target_keys),
        "no_missing_or_incomplete": partition["coverage_counts"].get("missing_or_incomplete", 0) == 0,
        "no_mapped_exact_without_unique_cause": partition["coverage_counts"].get("mapped_exact", 0) == 0,
        "source_and_descriptor_hashes_match": True,
        "partition_validator_passed": True,
        "analytical_lane_preserved_unresolved": all(
            cohort["classification"] == "undetermined"
            for cohort in diagnostic["cohorts"]
            if cohort["method_id"] != AUTODIFF_METHOD
        ),
        "gate_b_rejected_gate_c_blocked": diagnostic["decision"]
        == {
            "gate_b_status": "still_rejected",
            "gate_c_status": "blocked",
            "next_action": "draft_graph_structure_localization_subplan",
            "runtime_authorized": False,
        },
    }
    if not all(checks.values()):
        raise LocalizationError(f"localization checks failed: {checks}")
    finished_utc = datetime.now(timezone.utc).isoformat()
    uname = os.uname()
    payload = {
        "schema": SCHEMA,
        "state": "passed_complete_causally_ambiguous",
        "classification": "complete_attribution_inventory_causal_ambiguity_retained",
        "checks": checks,
        "decoder_parity": decoder_parity,
        "partition": partition,
        "source_anchors": {
            **source_anchors,
            "causal_boundary": (
                "anchors localize observed construction candidates but do not prove "
                "avoidability, inherence, or defect"
            ),
        },
        "input_ledger": {
            "diagnostic": {
                "path": str(DIAGNOSTIC_PATH.relative_to(ROOT)),
                "sha256": DIAGNOSTIC_SHA256,
                "payload_sha256": DIAGNOSTIC_PAYLOAD_SHA256,
            },
            "plan": {"path": str(PLAN_PATH.relative_to(ROOT)), "sha256": PLAN_SHA256},
            "plan_review": {
                "path": str(FINAL_REVIEW_PATH.relative_to(ROOT)),
                "sha256": FINAL_REVIEW_SHA256,
            },
            "authorized_snapshot": {
                "path": str(snapshot_path.relative_to(ROOT)),
                "sha256": AUTHORIZED_SNAPSHOT_SHA256,
            },
            "source_files": sources,
            "descriptor_files": descriptor_ledger,
            "tensorflow_distribution": "tensorflow==2.20.0",
            "protobuf_version": "6.33.5",
            "graph_bindings": [
                {
                    "identity": row["identity"],
                    "path": row["path"],
                    "sha256": row["raw_sha256"],
                    "decoded_bytes": row["decoded_bytes"],
                }
                for row in sorted(rows, key=lambda item: item["identity"]["identity_id"])
            ],
        },
        "decision": {
            "next_branch": partition["next_branch"],
            "gate_b_status": "still_rejected",
            "gate_c_status": "blocked",
            "runtime_authorized": False,
            "analytical_lane": "unresolved",
        },
        "nonclaims": NONCLAIMS,
        "run_manifest": {
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "wall_seconds": time.monotonic() - started,
            "output_path": str(output_path),
            "git_commit": git_commit(),
            "command": (
                "guarded localizer mode with --output-json equal to "
                "run_manifest.output_path; exact shell invocation preserved in run log"
            ),
            "cwd": str(ROOT),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": {
                "sysname": uname.sysname,
                "release": uname.release,
                "machine": uname.machine,
            },
            "conda_environment": str(Path(sys.executable).resolve().parents[1]),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cpu_gpu_status": "CPU-only; GPU deliberately hidden; no device enumeration",
            "tensorflow_imported": any(
                name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules
            ),
            "jit_xla": "not_initialized_or_invoked",
            "tf32": "not_queried",
            "data_fixture_version": "frozen Gate B R3 36-GraphDef corpus",
            "seeds": "N/A deterministic offline analysis",
            "plan_path": str(PLAN_PATH.relative_to(ROOT)),
            "result_path": str(RESULT_PATH.relative_to(ROOT)),
            "trust_basis": "offline_engineering_localization_only",
            "canonical_exclusions": sorted(RUN_EXCLUDED_PATHS),
        },
    }
    if payload["run_manifest"]["tensorflow_imported"]:
        raise LocalizationError("TensorFlow entered sys.modules")
    payload["localization_payload_sha256"] = canonical_sha256(canonical_payload(payload))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = build_payload(args.output_json)
    args.output_json.write_text(strict_dumps(payload, indent=2) + "\n", encoding="utf-8")
    reparsed = strict_load(args.output_json)
    if canonical_sha256(canonical_payload(reparsed)) != reparsed["localization_payload_sha256"]:
        raise LocalizationError("persisted payload digest mismatch")
    print(
        strict_dumps(
            {
                "state": reparsed["state"],
                "classification": reparsed["classification"],
                "target_count": reparsed["partition"]["target_count"],
                "coverage_counts": reparsed["partition"]["coverage_counts"],
                "next_branch": reparsed["decision"]["next_branch"],
                "payload_sha256": reparsed["localization_payload_sha256"],
                "output_json": str(args.output_json),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
