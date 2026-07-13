#!/usr/bin/env python3
"""Offline structural discriminator for preserved Gate B R3 autodiff GraphDefs."""

from __future__ import annotations

import argparse
import builtins
import copy
import hashlib
import json
import os
import runpy
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PARENT_LOCALIZER_PATH = (
    ROOT
    / "docs/benchmarks/localize_kalman_qr_phase6_gateb_r3_autodiff_structure_2026_07_13.py"
).resolve()
PARENT_LOCALIZER_SHA256 = "f743479564e54bb1dfa9651e724964c863ecbd7f9d20498e68b64a24b0da4ab9"
PARENT_ARTIFACT_PATH = ROOT / (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_"
    "autodiff_structure_localization_2026-07-13.json"
)
PARENT_ARTIFACT_SHA256 = "ee2903381039f7cf15a4ec5112304232ae138eebacef8e0858da1fda5f7452c1"
PARENT_PAYLOAD_SHA256 = "f29bea2cb4f26e6e26f1606149652eb98f4145099a3568e7874d67824f166e1c"
PLAN_PATH = ROOT / (
    "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-"
    "autodiff-attribution-discriminator-subplan-2026-07-13.md"
)
PLAN_SHA256 = "ce14737c2bee978e4fc1fe6134c5b306d6bc6c39de95b78658b46b49c5a8247b"
FINAL_REVIEW_PATH = ROOT / (
    "docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-"
    "autodiff-attribution-discriminator-subplan-review-final-2026-07-13.md"
)
FINAL_REVIEW_SHA256 = "24c74a874ef62034b607fbca5c2fddbb69647d3ec6a50d4ac95e2ae2f9af0e9e"
SNAPSHOT_PATH = ROOT / (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_"
    "autodiff_attribution_discriminator_authorized_snapshot_2026-07-13.json"
)
SNAPSHOT_SHA256 = "7b9436a392c5d8d40b145f0c5dd0eda3e4b6e194792a5b118e6bc4fb09e240aa"
RESULT_PATH = ROOT / (
    "docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-"
    "autodiff-attribution-discriminator-result-2026-07-13.md"
)
DURABLE_OUTPUT = ROOT / (
    "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_"
    "autodiff_attribution_discriminator_2026-07-13.json"
)
SCRATCH_ROOT = Path(
    "/tmp/kalman_qr_phase6_gateb_r3_autodiff_attribution_discriminator"
)
AUTHORIZED_OUTPUTS = {
    (SCRATCH_ROOT / "run1.json").resolve(),
    (SCRATCH_ROOT / "run2.json").resolve(),
    DURABLE_OUTPUT.resolve(),
}
GUARD_SLOT = "_kalman_qr_phase6_autodiff_attribution_guard_20260713"
SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6."
    "autodiff_attribution_discriminator.v1"
)
RUN_EXCLUDED_PATHS = {
    "run_manifest.finished_utc",
    "run_manifest.output_path",
    "run_manifest.started_utc",
    "run_manifest.wall_seconds",
}
REGION_STATES = (
    "cross_region_composite",
    "forward_while_call_or_function",
    "reverse_while_call_or_function",
    "local_pre_forward_while",
    "forward_while_setup_not_parameter_descendant",
    "forward_value_projection",
    "reverse_vjp_setup",
    "post_reverse_vjp",
    "score_path_not_reverse_descendant",
    "constant_or_shape_origin_unresolved",
    "structural_boundary_ambiguous",
    "unresolved_invalid",
)
NONCLAIMS = [
    "structural region ownership is not construction origin or root cause",
    "no avoidable, inherent, bug, error, or evaluator-exception claim",
    "no numerical equivalence or TensorFlow runtime evidence",
    "no Gate B pass and no Gate C, XLA, or GPU authorization",
    "no memory, performance, scalability, ranking, or readiness claim",
]


class DiscriminatorError(RuntimeError):
    """Raised when offline structural evidence is incomplete or inconsistent."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, path: str = "$") -> None:
    import math

    if isinstance(value, float) and not math.isfinite(value):
        raise DiscriminatorError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


def strict_dumps(value: Any, *, indent: int | None = None) -> str:
    _finite(value)
    return json.dumps(
        value,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(strict_dumps(value).encode("utf-8"))


def strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise DiscriminatorError(f"non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DiscriminatorError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved not in AUTHORIZED_OUTPUTS:
        raise DiscriminatorError(f"output path is not authorized: {resolved}")
    if resolved.parent.resolve() != resolved.parent:
        raise DiscriminatorError("output parent is not canonical")
    return resolved


def _guard_state() -> Mapping[str, Any]:
    state = getattr(builtins, GUARD_SLOT, None)
    if not isinstance(state, Mapping):
        raise DiscriminatorError("pre-load guard is absent")
    if state.get("token") != "guard_installed_before_subject_load_v1":
        raise DiscriminatorError("pre-load guard token mismatch")
    return state


def load_parent_localizer() -> Mapping[str, Any]:
    state = _guard_state()
    if sha256_file(PARENT_LOCALIZER_PATH) != PARENT_LOCALIZER_SHA256:
        raise DiscriminatorError("parent localizer drift")
    loader = state.get("load_parent")
    if not callable(loader):
        raise DiscriminatorError("guarded parent loader is absent")
    namespace = loader()
    if not isinstance(namespace, Mapping):
        raise DiscriminatorError("parent localizer namespace is invalid")
    return namespace


def edge_source(parent: Mapping[str, Any], raw: str) -> tuple[bool, str, int]:
    return parent["edge_source"](raw)


def unique_nodes(nodes: Iterable[Any], *, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in nodes:
        if not node.name or node.name in result:
            raise DiscriminatorError(f"duplicate or empty node in {label}: {node.name!r}")
        result[node.name] = node
    return result


def graph_adjacency(parent: Mapping[str, Any], nodes: Iterable[Any]) -> dict[str, Any]:
    by_name = unique_nodes(nodes, label="graph scope")
    producers: dict[str, list[dict[str, Any]]] = {name: [] for name in by_name}
    consumers: dict[str, list[dict[str, Any]]] = {name: [] for name in by_name}
    external: dict[str, list[dict[str, Any]]] = {name: [] for name in by_name}
    for node in by_name.values():
        for input_index, raw in enumerate(node.input):
            control, source, output_index = edge_source(parent, raw)
            edge = {
                "source": source,
                "target": node.name,
                "input_index": input_index,
                "output_index": output_index,
                "control": control,
            }
            producers[node.name].append(edge)
            if source in by_name:
                consumers[source].append(edge)
            else:
                external[node.name].append(edge)
    for mapping in (producers, consumers, external):
        for key in mapping:
            mapping[key] = sorted(mapping[key], key=strict_dumps)
    return {
        "nodes": by_name,
        "producers": producers,
        "consumers": consumers,
        "external_inputs": external,
    }


def inclusive_reachable(
    adjacency: Mapping[str, Any], seeds: Iterable[str], *, direction: str
) -> set[str]:
    if direction not in {"ancestors", "descendants"}:
        raise DiscriminatorError(f"invalid reachability direction {direction!r}")
    nodes = adjacency["nodes"]
    starts = set(seeds)
    if not starts or any(seed not in nodes for seed in starts):
        raise DiscriminatorError(f"invalid reachability seeds: {sorted(starts)}")
    result: set[str] = set()
    stack = list(sorted(starts, reverse=True))
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        edges = (
            adjacency["producers"][current]
            if direction == "ancestors"
            else adjacency["consumers"][current]
        )
        neighbors = [edge["source" if direction == "ancestors" else "target"] for edge in edges]
        stack.extend(sorted((name for name in neighbors if name in nodes), reverse=True))
    return result


def canonical_slice(
    adjacency: Mapping[str, Any], members: Iterable[str], *, slice_id: str
) -> dict[str, Any]:
    member_set = set(members)
    if any(name not in adjacency["nodes"] for name in member_set):
        raise DiscriminatorError(f"slice {slice_id} contains an unknown node")
    edges = []
    for source in sorted(member_set):
        for edge in adjacency["consumers"][source]:
            if edge["target"] in member_set:
                edges.append(edge)
    payload = {"nodes": sorted(member_set), "edges": sorted(edges, key=strict_dumps)}
    return {
        "slice_id": slice_id,
        "node_count": len(member_set),
        "edge_count": len(edges),
        "nodes": payload["nodes"],
        "edges": payload["edges"],
        "slice_sha256": canonical_sha256(payload),
    }


def canonical_shortest_path(
    adjacency: Mapping[str, Any], starts: Iterable[str], goals: Iterable[str]
) -> list[str] | None:
    start_set = set(starts)
    goal_set = set(goals)
    if not start_set or not goal_set:
        return None
    queue: deque[tuple[str, ...]] = deque((name,) for name in sorted(start_set))
    best_depth: dict[str, int] = {name: 0 for name in start_set}
    solutions: list[tuple[str, ...]] = []
    solution_depth: int | None = None
    while queue:
        path = queue.popleft()
        depth = len(path) - 1
        if solution_depth is not None and depth > solution_depth:
            break
        current = path[-1]
        if current in goal_set:
            solution_depth = depth
            solutions.append(path)
            continue
        for edge in adjacency["consumers"].get(current, []):
            target = edge["target"]
            next_depth = depth + 1
            if best_depth.get(target, next_depth) < next_depth:
                continue
            best_depth[target] = next_depth
            queue.append(path + (target,))
    return list(min(solutions)) if solutions else None


def function_map(graph: Any) -> dict[str, Any]:
    result = {}
    for function in graph.library.function:
        name = function.signature.name
        if not name or name in result:
            raise DiscriminatorError(f"duplicate or empty function {name!r}")
        result[name] = function
    return result


def function_call_bindings(parent: Mapping[str, Any], graph: Any) -> dict[str, list[dict[str, Any]]]:
    functions = function_map(graph)
    bindings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scopes: list[tuple[str, Iterable[Any]]] = [("top", graph.node)]
    scopes.extend((f"function:{name}", function.node_def) for name, function in functions.items())
    for scope, nodes in scopes:
        for node in nodes:
            for attr_key, attr in sorted(node.attr.items()):
                names: list[str] = []
                if attr.HasField("func") and attr.func.name:
                    names.append(attr.func.name)
                names.extend(item.name for item in attr.list.func if item.name)
                for index, name in enumerate(names):
                    if name not in functions:
                        raise DiscriminatorError(f"call attr references missing function {name!r}")
                    bindings[name].append(
                        {
                            "scope": scope,
                            "caller": node.name,
                            "caller_op": node.op,
                            "attr_key": attr_key,
                            "attr_index": index,
                            "caller_inputs": list(node.input),
                            "function": name,
                        }
                    )
    return {name: sorted(items, key=strict_dumps) for name, items in sorted(bindings.items())}


def graph_analysis(parent: Mapping[str, Any], graph: Any) -> dict[str, Any]:
    return {
        "adjacency": graph_adjacency(parent, graph.node),
        "functions": function_map(graph),
        "call_bindings": function_call_bindings(parent, graph),
    }


def function_binding_witness(
    parent: Mapping[str, Any],
    graph: Any,
    function_name: str,
    analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = analysis or graph_analysis(parent, graph)
    functions = resolved["functions"]
    bindings = resolved["call_bindings"].get(function_name, [])
    function = functions[function_name]
    signature = {
        "input_args": [
            {
                "name": arg.name,
                "type": int(arg.type),
                "type_attr": arg.type_attr,
                "number_attr": arg.number_attr,
                "type_list_attr": arg.type_list_attr,
            }
            for arg in function.signature.input_arg
        ],
        "output_args": [
            {
                "name": arg.name,
                "type": int(arg.type),
                "type_attr": arg.type_attr,
                "number_attr": arg.number_attr,
                "type_list_attr": arg.type_list_attr,
            }
            for arg in function.signature.output_arg
        ],
        "ret": dict(sorted(function.ret.items())),
        "control_ret": dict(sorted(function.control_ret.items())),
    }
    return {
        "function_name": function_name,
        "function_sha256": sha256_bytes(function.SerializeToString(deterministic=True)),
        "bindings": bindings,
        "signature": signature,
        "signature_sha256": canonical_sha256(signature),
    }


def debug_record(node: Any) -> dict[str, Any]:
    debug = node.experimental_debug_info
    return {
        "original_node_names": list(debug.original_node_names),
        "original_func_names": list(debug.original_func_names),
        "raw_sha256": sha256_bytes(debug.SerializeToString(deterministic=True)),
    }


def graph_boundaries(
    parent: Mapping[str, Any],
    row: Mapping[str, Any],
    analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    graph = row["graph"]
    trace = row["record"]["evidence"]
    resolved = analysis or graph_analysis(parent, graph)
    adjacency = resolved["adjacency"]
    call_bindings = resolved["call_bindings"]
    user = trace.get("structured_user_input")
    outputs = trace.get("concrete_outputs")
    if not isinstance(user, Mapping) or user.get("name") != "parameters_batch":
        raise DiscriminatorError("structured parameter input is absent or changed")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise DiscriminatorError("concrete output boundary is absent or changed")
    by_position = {item.get("result_position"): item for item in outputs}
    if set(by_position) != {"value", "score"}:
        raise DiscriminatorError("value/score output positions are invalid")

    def base_tensor_name(value: str) -> str:
        name = value.split(":", 1)[0]
        if name not in adjacency["nodes"]:
            raise DiscriminatorError(f"trace boundary node is missing: {name!r}")
        return name

    parameter = base_tensor_name(str(user["name"]))
    value_output = base_tensor_name(str(by_position["value"]["name"]))
    score_output = base_tensor_name(str(by_position["score"]["name"]))
    while_nodes = [
        node for node in graph.node if node.op in {"While", "StatelessWhile"}
    ]
    value_ancestors = inclusive_reachable(adjacency, [value_output], direction="ancestors")
    score_ancestors = inclusive_reachable(adjacency, [score_output], direction="ancestors")
    forward_candidates = sorted(node.name for node in while_nodes if node.name in value_ancestors)
    forward = forward_candidates[0] if len(forward_candidates) == 1 else None
    forward_descendants = (
        inclusive_reachable(adjacency, [forward], direction="descendants")
        if forward is not None
        else set()
    )
    reverse_candidates = []
    if forward is not None:
        for node in while_nodes:
            if node.name == forward or node.name not in score_ancestors:
                continue
            bound_functions = parent["called_functions"](node)
            if len(bound_functions) < 2:
                continue
            captured = []
            for input_index, raw in enumerate(node.input):
                _control, source, output_index = edge_source(parent, raw)
                if source in forward_descendants:
                    captured.append(
                        {
                            "input_index": input_index,
                            "source": source,
                            "source_output_index": output_index,
                            "source_is_forward_call": source == forward,
                        }
                    )
            if captured:
                reverse_candidates.append(
                    {
                        "name": node.name,
                        "called_functions": bound_functions,
                        "saved_forward_bindings": captured,
                    }
                )
    reverse_candidates = sorted(reverse_candidates, key=strict_dumps)
    reverse = reverse_candidates[0]["name"] if len(reverse_candidates) == 1 else None
    parameter_descendants = inclusive_reachable(adjacency, [parameter], direction="descendants")
    forward_ancestors = (
        inclusive_reachable(adjacency, [forward], direction="ancestors")
        if forward is not None
        else set()
    )
    reverse_ancestors = (
        inclusive_reachable(adjacency, [reverse], direction="ancestors")
        if reverse is not None
        else set()
    )
    reverse_descendants = (
        inclusive_reachable(adjacency, [reverse], direction="descendants")
        if reverse is not None
        else set()
    )
    debug_nodes = []
    for scope, nodes in [("top", graph.node)] + [
        (f"function:{function.signature.name}", function.node_def)
        for function in graph.library.function
    ]:
        for node in nodes:
            record = debug_record(node)
            if record["original_node_names"] or record["original_func_names"]:
                debug_nodes.append({"scope": scope, "name": node.name, **record})
    graph_debug = graph.debug_info
    graph_debug_payload = {
        "files": list(graph_debug.files),
        "frame_count": len(graph_debug.frames_by_id),
        "trace_count": len(graph_debug.traces),
        "name_to_trace_id_count": len(graph_debug.name_to_trace_id),
        "raw_sha256": sha256_bytes(graph_debug.SerializeToString(deterministic=True)),
    }
    result = {
        "identity": row["identity"],
        "graph_sha256": row["raw_sha256"],
        "parameter": parameter,
        "value_output": value_output,
        "score_output": score_output,
        "while_nodes": [
            {
                "name": node.name,
                "op": node.op,
                "called_functions": parent["called_functions"](node),
                "input_count": len(node.input),
                "raw_sha256": sha256_bytes(node.SerializeToString(deterministic=True)),
            }
            for node in while_nodes
        ],
        "function_call_bindings_sha256": canonical_sha256(call_bindings),
        "forward_candidates": forward_candidates,
        "forward": forward,
        "reverse_candidates": reverse_candidates,
        "reverse": reverse,
        "sets": {
            "parameter_descendants": sorted(parameter_descendants),
            "forward_ancestors": sorted(forward_ancestors),
            "forward_descendants": sorted(forward_descendants),
            "value_ancestors": sorted(value_ancestors),
            "reverse_ancestors": sorted(reverse_ancestors),
            "reverse_descendants": sorted(reverse_descendants),
            "score_ancestors": sorted(score_ancestors),
        },
        "debug_info": {
            "node_count": sum(
                len(nodes)
                for nodes in [graph.node]
                + [function.node_def for function in graph.library.function]
            ),
            "populated_node_count": len(debug_nodes),
            "populated_nodes": debug_nodes,
            "graph": graph_debug_payload,
        },
    }
    result["candidate_ledger_sha256"] = canonical_sha256(
        {
            "forward_candidates": result["forward_candidates"],
            "reverse_candidates": result["reverse_candidates"],
        }
    )
    result["function_ledger"] = {
        name: function_binding_witness(parent, graph, name, resolved)
        for name in sorted(resolved["functions"])
    }
    result["function_ledger_sha256"] = canonical_sha256(result["function_ledger"])
    state_sets = _state_sets(result)
    state_sets["forward_while_call_or_function"] = {forward} if forward is not None else set()
    state_sets["reverse_while_call_or_function"] = {reverse} if reverse is not None else set()
    result["region_slices"] = {
        state: canonical_slice(
            adjacency,
            members,
            slice_id=f"{row['raw_sha256']}::{state}",
        )
        for state, members in sorted(state_sets.items())
    }
    return result


def _state_sets(boundary: Mapping[str, Any]) -> dict[str, set[str]]:
    sets = {key: set(value) for key, value in boundary["sets"].items()}
    forward = boundary["forward"]
    reverse = boundary["reverse"]
    result: dict[str, set[str]] = {}
    if forward is None:
        return result
    result["local_pre_forward_while"] = (
        sets["parameter_descendants"] & sets["forward_ancestors"]
    ) - {forward}
    result["forward_while_setup_not_parameter_descendant"] = (
        sets["forward_ancestors"] - sets["parameter_descendants"]
    ) - {forward}
    result["forward_value_projection"] = (
        sets["forward_descendants"] & sets["value_ancestors"]
    ) - {forward}
    if reverse is not None:
        result["reverse_vjp_setup"] = sets["reverse_ancestors"] - {reverse}
        result["post_reverse_vjp"] = (
            sets["reverse_descendants"] & sets["score_ancestors"]
        ) - {reverse}
    result["score_path_not_reverse_descendant"] = set(sets["score_ancestors"])
    return result


def _function_owner_state(
    parent: Mapping[str, Any],
    graph: Any,
    function_name: str,
    boundary: Mapping[str, Any],
    analysis: Mapping[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    binding = function_binding_witness(parent, graph, function_name, analysis)
    return _owner_from_binding(binding, boundary), binding


def _owner_from_binding(
    binding: Mapping[str, Any], boundary: Mapping[str, Any]
) -> str | None:
    matches = []
    for role, call_name in (
        ("forward_while_call_or_function", boundary["forward"]),
        ("reverse_while_call_or_function", boundary["reverse"]),
    ):
        if call_name is None:
            continue
        for item in binding["bindings"]:
            if item["scope"] == "top" and item["caller"] == call_name and item["attr_key"] in {
                "cond",
                "body",
            }:
                matches.append(role)
    if len(set(matches)) == 1:
        return matches[0]
    return None


def _literal_unresolved(node: Any) -> bool:
    if node.input:
        return False
    if node.op == "Const" and "value" in node.attr:
        return True
    return node.op in {"Shape", "Fill", "ZerosLike"}


def classify_atomic(
    parent: Mapping[str, Any],
    row: Mapping[str, Any],
    boundary: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    observation: Mapping[str, Any],
    nested: Mapping[str, Any] | None,
    analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entity_kind = target["entity_kind"]
    target_key = target["target_key"]
    outer_digest = canonical_sha256(observation)
    nested_digest = canonical_sha256(nested) if nested is not None else None
    atomic_key = canonical_sha256(
        {
            "target_key": target_key,
            "outer_observation_sha256": outer_digest,
            "nested_occurrence_sha256": nested_digest,
        }
    )
    graph = row["graph"]
    resolved = analysis or graph_analysis(parent, graph)
    graph_sha256 = row["raw_sha256"]
    state_sets = _state_sets(boundary)
    state: str | None = None
    scope = "metadata"
    entity_digest = None
    name = target.get("name")
    predicate_vector = {region: False for region in REGION_STATES}
    state_specific: dict[str, Any] = {}

    if entity_kind == "graph_order":
        occurrence = observation["occurrence"]
        state = "cross_region_composite"
        state_specific = {
            "orders": occurrence["orders"],
            "orders_sha256": occurrence["orders_sha256"],
        }
    elif entity_kind == "function":
        occurrence = observation["occurrence"]
        function_name = target["name"]
        binding = boundary["function_ledger"][function_name]
        owner = _owner_from_binding(binding, boundary)
        state = owner or "structural_boundary_ambiguous"
        scope = f"function:{function_name}"
        entity_digest = occurrence["function"]["raw_sha256"]
        state_specific = {
            "function_ledger_ref": f"{graph_sha256}::{function_name}",
            "function_ledger_entry_sha256": canonical_sha256(binding),
        }
        if owner is None:
            state_specific["failed_predicate"] = "function_owner_not_unique"
    elif entity_kind == "function_body" or (
        entity_kind == "integer_constant"
        and str((nested or observation)["neighborhood"]["scope"]).startswith("function:")
    ):
        concrete = nested or observation
        neighborhood = concrete["neighborhood"]
        scope = neighborhood["scope"]
        function_name = scope.split(":", 1)[1]
        binding = boundary["function_ledger"][function_name]
        owner = _owner_from_binding(binding, boundary)
        state = owner or "structural_boundary_ambiguous"
        entity_digest = neighborhood["node"]["raw_sha256"]
        state_specific = {
            "function_ledger_ref": f"{graph_sha256}::{function_name}",
            "function_ledger_entry_sha256": canonical_sha256(binding),
        }
        if owner is None:
            state_specific["failed_predicate"] = "function_owner_not_unique"
    elif entity_kind in {"top_node", "integer_constant"}:
        concrete = nested or observation
        neighborhood = concrete["neighborhood"]
        scope = neighborhood["scope"]
        if scope != "top":
            raise DiscriminatorError(f"unexpected top-level scope {scope!r}")
        node_name = neighborhood["node"]["name"]
        adjacency = resolved["adjacency"]
        node = adjacency["nodes"].get(node_name)
        if node is None:
            raise DiscriminatorError(f"target node missing from graph: {node_name!r}")
        name = node_name
        entity_digest = neighborhood["node"]["raw_sha256"]
        forward = boundary["forward"]
        reverse = boundary["reverse"]
        predicates: list[tuple[str, bool]] = [
            ("forward_while_call_or_function", forward is not None and node_name == forward),
            ("reverse_while_call_or_function", reverse is not None and node_name == reverse),
            ("local_pre_forward_while", node_name in state_sets.get("local_pre_forward_while", set())),
            (
                "forward_while_setup_not_parameter_descendant",
                node_name in state_sets.get("forward_while_setup_not_parameter_descendant", set()),
            ),
            ("forward_value_projection", node_name in state_sets.get("forward_value_projection", set())),
            ("reverse_vjp_setup", node_name in state_sets.get("reverse_vjp_setup", set())),
            ("post_reverse_vjp", node_name in state_sets.get("post_reverse_vjp", set())),
            (
                "score_path_not_reverse_descendant",
                node_name in state_sets.get("score_path_not_reverse_descendant", set()),
            ),
            (
                "constant_or_shape_origin_unresolved",
                entity_kind == "integer_constant" or _literal_unresolved(node),
            ),
        ]
        matched_candidates = []
        for candidate, matched in predicates:
            predicate_vector[candidate] = bool(matched)
            if matched:
                matched_candidates.append(candidate)
            if matched and state is None:
                state = candidate
        forward_dependent = node_name in (
            set(boundary["sets"]["forward_ancestors"])
            | set(boundary["sets"]["forward_descendants"])
            | set(boundary["sets"]["value_ancestors"])
            | set(boundary["sets"]["score_ancestors"])
        )
        reverse_dependent = node_name in set(boundary["sets"]["score_ancestors"])
        forward_state = state in {
            "forward_while_call_or_function",
            "local_pre_forward_while",
            "forward_while_setup_not_parameter_descendant",
            "forward_value_projection",
        }
        if forward is None and forward_dependent:
            state = "structural_boundary_ambiguous"
            state_specific["failed_predicate"] = "forward_boundary_not_unique"
        elif reverse is None and reverse_dependent and not forward_state:
            state = "structural_boundary_ambiguous"
            state_specific["failed_predicate"] = "reverse_boundary_not_unique"
        if state is None:
            state = "structural_boundary_ambiguous"
            state_specific["failed_predicate"] = "complete_top_node_outside_declared_regions"
        path_witness: dict[str, Any] = {}
        forward = boundary["forward"]
        reverse = boundary["reverse"]
        value_output = boundary["value_output"]
        score_output = boundary["score_output"]
        if state in {
            "local_pre_forward_while",
            "forward_while_setup_not_parameter_descendant",
        } and forward is not None:
            path_witness["node_to_forward"] = canonical_shortest_path(
                adjacency, [name], [forward]
            )
        elif state == "forward_value_projection" and forward is not None:
            path_witness["forward_to_node"] = canonical_shortest_path(
                adjacency, [forward], [name]
            )
            path_witness["node_to_value"] = canonical_shortest_path(
                adjacency, [name], [value_output]
            )
        elif state == "reverse_vjp_setup" and reverse is not None:
            path_witness["node_to_reverse"] = canonical_shortest_path(
                adjacency, [name], [reverse]
            )
        elif state == "post_reverse_vjp" and reverse is not None:
            path_witness["reverse_to_node"] = canonical_shortest_path(
                adjacency, [reverse], [name]
            )
            path_witness["node_to_score"] = canonical_shortest_path(
                adjacency, [name], [score_output]
            )
        elif state == "score_path_not_reverse_descendant":
            path_witness["node_to_score"] = canonical_shortest_path(
                adjacency, [name], [score_output]
            )
        state_specific.update(
            {
                "node": name,
                "node_op": node.op,
                "node_debug": debug_record(node),
                "producers": adjacency["producers"][name],
                "consumers": adjacency["consumers"][name],
                "external_inputs": adjacency["external_inputs"][name],
                "region_slice_ref": (
                    boundary["region_slices"].get(state, {}).get("slice_id")
                ),
                "matched_first_pass_predicates": matched_candidates,
                "path_witness": path_witness,
            }
        )
    else:
        state = "unresolved_invalid"
        state_specific["failed_predicate"] = f"unsupported_entity_kind:{entity_kind}"

    assert state is not None
    predicate_vector[state] = True
    witness = {
        "atomic_key": atomic_key,
        "target_key": target_key,
        "entity_kind": entity_kind,
        "name": name,
        "outer_observation_sha256": outer_digest,
        "nested_occurrence_sha256": nested_digest,
        "graph_sha256": graph_sha256,
        "identity_id": row["identity"]["identity_id"],
        "scope": scope,
        "entity_sha256": entity_digest,
        "trace_boundaries": {
            "parameter": boundary["parameter"],
            "value_output": boundary["value_output"],
            "score_output": boundary["score_output"],
        },
        "boundary_ledger_ref": graph_sha256,
        "candidate_ledger_sha256": boundary["candidate_ledger_sha256"],
        "function_ledger_sha256": boundary["function_ledger_sha256"],
        "forward": boundary["forward"],
        "reverse": boundary["reverse"],
        "predicate_vector": predicate_vector,
        "state_specific": state_specific,
    }
    return {"atomic_key": atomic_key, "state": state, "witness": witness}


def _row_by_graph(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        sha = row["raw_sha256"]
        if sha in result:
            raise DiscriminatorError(f"duplicate graph hash {sha}")
        result[sha] = row
    return result


def atomic_partition(
    parent: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    parent_partition: Mapping[str, Any],
) -> dict[str, Any]:
    by_graph = _row_by_graph(rows)
    analyses = {
        sha: graph_analysis(parent, row["graph"])
        for sha, row in sorted(by_graph.items())
        if row["identity"]["method_id"] == parent["AUTODIFF_METHOD"]
    }
    boundaries = {
        sha: graph_boundaries(parent, row, analyses[sha])
        for sha, row in sorted(by_graph.items())
        if row["identity"]["method_id"] == parent["AUTODIFF_METHOD"]
    }
    targets = []
    atomic_keys: set[str] = set()
    observation_count = 0
    nested_count = 0
    state_counts: Counter[str] = Counter()
    target_aggregate_counts: Counter[str] = Counter()
    for target in parent_partition["targets"]:
        atomics = []
        for observation in target["observations"]:
            observation_count += 1
            if target["entity_kind"] == "integer_constant":
                nested_items = observation.get("occurrences", [])
                if not nested_items:
                    raise DiscriminatorError("integer-constant observation lacks occurrences")
            else:
                nested_items = [None]
            for nested in nested_items:
                nested_count += nested is not None
                concrete = nested or observation
                graph_sha256 = concrete.get("graph_sha256")
                if graph_sha256 is None and "occurrence" in observation:
                    graph_sha256 = observation["occurrence"].get("graph_sha256")
                row = by_graph.get(graph_sha256)
                if row is None:
                    raise DiscriminatorError(f"observation graph is not bound: {graph_sha256}")
                boundary = boundaries.get(graph_sha256)
                if boundary is None:
                    raise DiscriminatorError("target occurrence resolved to a non-autodiff graph")
                atomic = classify_atomic(
                    parent,
                    row,
                    boundary,
                    target=target,
                    observation=observation,
                    nested=nested,
                    analysis=analyses[graph_sha256],
                )
                if atomic["atomic_key"] in atomic_keys:
                    raise DiscriminatorError(f"duplicate atomic occurrence {atomic['atomic_key']}")
                atomic_keys.add(atomic["atomic_key"])
                atomics.append(atomic)
                state_counts[atomic["state"]] += 1
        target_states = sorted({item["state"] for item in atomics})
        aggregate = f"uniform_{target_states[0]}" if len(target_states) == 1 else "mixed_regions"
        target_aggregate_counts[aggregate] += 1
        targets.append(
            {
                "target_key": target["target_key"],
                "entity_kind": target["entity_kind"],
                "atomic_count": len(atomics),
                "aggregate_state": aggregate,
                "state_counts": dict(sorted(Counter(item["state"] for item in atomics).items())),
                "atomics": atomics,
            }
        )
    result = {
        "targets": targets,
        "target_count": len(targets),
        "observation_count": observation_count,
        "nested_integer_occurrence_count": nested_count,
        "atomic_count": len(atomic_keys),
        "state_counts": dict(sorted(state_counts.items())),
        "target_aggregate_counts": dict(sorted(target_aggregate_counts.items())),
        "boundaries": [boundaries[key] for key in sorted(boundaries)],
    }
    validate_atomic_partition(result)
    return result


def validate_atomic_partition(partition: Mapping[str, Any]) -> None:
    targets = partition.get("targets")
    if not isinstance(targets, list) or len(targets) != 904:
        raise DiscriminatorError("target count is not exactly 904")
    if partition.get("observation_count") != 12316:
        raise DiscriminatorError("observation count is not exactly 12316")
    if partition.get("nested_integer_occurrence_count") != 2386:
        raise DiscriminatorError("nested integer occurrence count is not exactly 2386")
    if partition.get("atomic_count") != 14270:
        raise DiscriminatorError("atomic count is not exactly 14270")
    target_keys = [item.get("target_key") for item in targets]
    if len(target_keys) != len(set(target_keys)):
        raise DiscriminatorError("duplicate target key")
    atomics = [atomic for target in targets for atomic in target.get("atomics", [])]
    atomic_keys = [item.get("atomic_key") for item in atomics]
    if len(atomics) != 14270 or len(atomic_keys) != len(set(atomic_keys)):
        raise DiscriminatorError("atomic occurrence coverage is incomplete or duplicate")
    observed_states = Counter(item.get("state") for item in atomics)
    if set(observed_states) - set(REGION_STATES):
        raise DiscriminatorError("unknown region state")
    if observed_states.get("unresolved_invalid", 0) != 0:
        raise DiscriminatorError("unresolved_invalid is a continuation veto")
    if dict(sorted(observed_states.items())) != partition.get("state_counts"):
        raise DiscriminatorError("state counts do not match atomic records")
    for atomic in atomics:
        witness = atomic.get("witness")
        if not isinstance(witness, Mapping) or witness.get("atomic_key") != atomic["atomic_key"]:
            raise DiscriminatorError("atomic witness binding is invalid")
        vector = witness.get("predicate_vector")
        if not isinstance(vector, Mapping) or set(vector) != set(REGION_STATES):
            raise DiscriminatorError("predicate vector is incomplete")
        if not vector.get(atomic["state"]):
            raise DiscriminatorError("selected state is not true in predicate vector")
        if atomic["state"] == "structural_boundary_ambiguous":
            state_specific = witness.get("state_specific")
            if not isinstance(state_specific, Mapping) or not state_specific.get("failed_predicate"):
                raise DiscriminatorError("honest ambiguity lacks failed predicate")


def canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    for dotted in sorted(RUN_EXCLUDED_PATHS):
        parent_key, key = dotted.split(".", 1)
        result[parent_key].pop(key, None)
    result.pop("discriminator_payload_sha256", None)
    return result


def _validate_inputs(parent: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        PLAN_PATH: PLAN_SHA256,
        FINAL_REVIEW_PATH: FINAL_REVIEW_SHA256,
        SNAPSHOT_PATH: SNAPSHOT_SHA256,
        PARENT_ARTIFACT_PATH: PARENT_ARTIFACT_SHA256,
        PARENT_LOCALIZER_PATH: PARENT_LOCALIZER_SHA256,
    }
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise DiscriminatorError(f"input drift: {path}: {actual}")
    parent_artifact = strict_load(PARENT_ARTIFACT_PATH)
    if (
        parent_artifact.get("state") != "passed_complete_causally_ambiguous"
        or parent_artifact.get("localization_payload_sha256") != PARENT_PAYLOAD_SHA256
        or parent_artifact.get("partition", {}).get("target_count") != 904
    ):
        raise DiscriminatorError("parent localization state changed")
    if canonical_sha256(parent["canonical_payload"](parent_artifact)) != PARENT_PAYLOAD_SHA256:
        raise DiscriminatorError("parent localization canonical payload mismatch")
    return parent_artifact


def _load_trace_rows(parent: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    graph_class, descriptor_ledger = parent["graphdef_class"]()
    rows = parent["trace_rows"](graph_class)
    for row in rows:
        row["record"] = strict_load(Path(row["path"]))
    return rows, descriptor_ledger


def build_payload(output_path: Path) -> dict[str, Any]:
    output_path = validate_output_path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise DiscriminatorError(f"output path must be absent: {output_path}")
    if not output_path.parent.is_dir():
        raise DiscriminatorError("output parent is absent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise DiscriminatorError("CUDA_VISIBLE_DEVICES must equal -1")
    if any(name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules):
        raise DiscriminatorError("TensorFlow entered sys.modules before analysis")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    parent = load_parent_localizer()
    parent_artifact = _validate_inputs(parent)
    rows, descriptor_ledger = _load_trace_rows(parent)
    diagnostic = parent["strict_load"](parent["DIAGNOSTIC_PATH"])
    parent["validate_diagnostic_payload"](diagnostic)
    decoder_parity = parent["parity"](rows, diagnostic)
    reconstructed = parent["target_partition"](rows, diagnostic)
    parent["validate_partition"](
        reconstructed,
        parent["expected_target_keys"](diagnostic),
        parent["source_anchor_ledger"](),
    )
    if reconstructed != parent_artifact["partition"]:
        raise DiscriminatorError("reconstructed parent partition does not match reviewed bytes")
    sources = parent["source_ledger"]()
    source_anchors = parent["source_anchor_ledger"]()
    partition = atomic_partition(parent, rows, reconstructed)
    finished_utc = datetime.now(timezone.utc).isoformat()
    uname = os.uname()
    checks = {
        "reviewed_authority_hashes_match": True,
        "parent_payload_reproduced": True,
        "all_36_graph_bindings_reproduced": len(rows) == 36,
        "all_18_autodiff_boundaries_present": len(partition["boundaries"]) == 18,
        "decoder_parity_passed": all(decoder_parity["checks"].values()),
        "target_count_exact": partition["target_count"] == 904,
        "observation_count_exact": partition["observation_count"] == 12316,
        "nested_integer_occurrence_count_exact": partition["nested_integer_occurrence_count"]
        == 2386,
        "atomic_count_exact": partition["atomic_count"] == 14270,
        "no_unresolved_invalid": partition["state_counts"].get("unresolved_invalid", 0) == 0,
        "no_tensorflow_module_imported": not any(
            name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules
        ),
        "gate_b_rejected_gate_c_runtime_blocked": parent_artifact["decision"]
        == {
            "analytical_lane": "unresolved",
            "gate_b_status": "still_rejected",
            "gate_c_status": "blocked",
            "next_branch": "autodiff_attribution_discriminator",
            "runtime_authorized": False,
        },
    }
    if not all(checks.values()):
        raise DiscriminatorError(f"final checks failed: {checks}")
    unique_local_control_point = False
    framework_candidate_states = {
        "uniform_reverse_while_call_or_function",
        "uniform_constant_or_shape_origin_unresolved",
    }
    framework_candidate_targets = [
        {
            "target_key": target["target_key"],
            "aggregate_state": target["aggregate_state"],
            "atomic_count": target["atomic_count"],
        }
        for target in partition["targets"]
        if target["aggregate_state"] in framework_candidate_states
    ]
    framework_question_text = (
        "For the exact uniform reverse-while and unresolved constant/shape cohorts, "
        "do the exact bound TensorFlow 2.20 framework source anchors require the "
        "observed structures for this frozen saved-forward binding, or can their "
        "installed-source control flow "
        "distinguish a local VJP-dependent construction without target runtime?"
    )
    framework_proof_question = bool(framework_candidate_targets) and bool(
        source_anchors["framework"]
    )
    minimal_runtime_counterfactual_needed = not unique_local_control_point and not framework_proof_question
    next_branch = (
        "autodiff_source_counterfactual_repair"
        if unique_local_control_point
        else "autodiff_framework_proof"
        if framework_proof_question
        else "autodiff_minimal_counterfactual_trace"
        if minimal_runtime_counterfactual_needed
        else "blocker"
    )
    payload = {
        "schema": SCHEMA,
        "state": "passed_complete_structural_region_partition",
        "classification": "graph_region_ownership_only_causation_unresolved",
        "checks": checks,
        "decoder_parity": decoder_parity,
        "partition": partition,
        "source_anchors": {
            **source_anchors,
            "boundary": "source anchors explain candidates but do not prove origin or necessity",
        },
        "input_ledger": {
            "plan": {"path": str(PLAN_PATH.relative_to(ROOT)), "sha256": PLAN_SHA256},
            "plan_review": {
                "path": str(FINAL_REVIEW_PATH.relative_to(ROOT)),
                "sha256": FINAL_REVIEW_SHA256,
            },
            "authorized_snapshot": {
                "path": str(SNAPSHOT_PATH.relative_to(ROOT)),
                "sha256": SNAPSHOT_SHA256,
            },
            "parent_localization": {
                "path": str(PARENT_ARTIFACT_PATH.relative_to(ROOT)),
                "sha256": PARENT_ARTIFACT_SHA256,
                "payload_sha256": PARENT_PAYLOAD_SHA256,
            },
            "parent_localizer": {
                "path": str(PARENT_LOCALIZER_PATH.relative_to(ROOT)),
                "sha256": PARENT_LOCALIZER_SHA256,
            },
            "source_files": sources,
            "descriptor_files": descriptor_ledger,
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
            "complete": True,
            "unique_local_control_point": unique_local_control_point,
            "framework_proof_question": framework_proof_question,
            "framework_proof_candidate": {
                "question": framework_question_text,
                "target_count": len(framework_candidate_targets),
                "targets": framework_candidate_targets,
                "source_anchor_ids": sorted(
                    anchor["anchor_id"] for anchor in source_anchors["framework"]
                ),
                "claim_boundary": (
                    "candidate question only; no framework necessity, inherence, "
                    "origin, or defect conclusion"
                ),
            },
            "minimal_runtime_counterfactual_needed": minimal_runtime_counterfactual_needed,
            "next_branch": next_branch,
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
            "git_commit": parent["git_commit"](),
            "command": "guarded discriminator mode with exact --output-json; shell invocation in log",
            "cwd": str(ROOT),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": {"sysname": uname.sysname, "release": uname.release, "machine": uname.machine},
            "conda_environment": str(Path(sys.executable).resolve().parents[1]),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cpu_gpu_status": "CPU-only; GPU deliberately hidden; no device enumeration",
            "tensorflow_imported": False,
            "jit_xla": "not_initialized_or_invoked",
            "tf32": "not_queried",
            "data_fixture_version": "frozen Gate B R3 36-GraphDef corpus",
            "seeds": "N/A deterministic offline analysis",
            "plan_path": str(PLAN_PATH.relative_to(ROOT)),
            "result_path": str(RESULT_PATH.relative_to(ROOT)),
            "trust_basis": "offline_engineering_attribution_only",
            "canonical_exclusions": sorted(RUN_EXCLUDED_PATHS),
        },
    }
    payload["discriminator_payload_sha256"] = canonical_sha256(canonical_payload(payload))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = build_payload(args.output_json)
    args.output_json.write_text(strict_dumps(payload, indent=2) + "\n", encoding="utf-8")
    reparsed = strict_load(args.output_json)
    if canonical_sha256(canonical_payload(reparsed)) != reparsed["discriminator_payload_sha256"]:
        raise DiscriminatorError("persisted payload digest mismatch")
    validate_atomic_partition(reparsed["partition"])
    print(
        strict_dumps(
            {
                "state": reparsed["state"],
                "target_count": reparsed["partition"]["target_count"],
                "observation_count": reparsed["partition"]["observation_count"],
                "atomic_count": reparsed["partition"]["atomic_count"],
                "state_counts": reparsed["partition"]["state_counts"],
                "next_branch": reparsed["decision"]["next_branch"],
                "payload_sha256": reparsed["discriminator_payload_sha256"],
                "output_json": str(args.output_json),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
