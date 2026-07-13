from __future__ import annotations

import copy
import ast
import importlib.util
import struct
import sys
from pathlib import Path

import pytest
from tensorflow.core.framework import graph_pb2, types_pb2


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "docs/benchmarks/diagnose_kalman_qr_phase6_gateb_r3_trace_rejections_2026_07_12.py"
)


def _load_module():
    name = "kalman_qr_phase6_gateb_r3_trace_rejection_diagnostic_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = _load_module()


def _clone(graph):
    result = graph_pb2.GraphDef()
    result.CopyFrom(graph)
    return result


def _const(node, values=(1,), *, dtype=types_pb2.DT_INT32):
    node.op = "Const"
    node.attr["dtype"].type = dtype
    tensor = node.attr["value"].tensor
    tensor.dtype = dtype
    tensor.tensor_shape.dim.add(size=len(values))
    tensor.tensor_content = struct.pack("<" + "i" * len(values), *values)


def _base_graph():
    graph = graph_pb2.GraphDef()
    source = graph.node.add(name="source")
    _const(source, (1, 2))
    other = graph.node.add(name="other")
    _const(other, (3, 4))
    graph.node.add(name="control", op="NoOp")
    graph.node.add(name="control2", op="NoOp")
    sink = graph.node.add(name="sink", op="AddV2")
    sink.input.extend(["source:0", "other:0", "^control"])
    sink.device = "/device:CPU:0"
    sink.attr["T"].type = types_pb2.DT_INT32
    shape = sink.attr["_output_shapes"].list.shape.add()
    shape.dim.add(size=2)
    sink.attr["axes"].list.i.extend([0, 1])
    return graph


def _add_function(graph, name="Fn", *, node_op="Identity"):
    function = graph.library.function.add()
    function.signature.name = name
    input_arg = function.signature.input_arg.add()
    input_arg.name = "x"
    input_arg.type = types_pb2.DT_INT32
    output_arg = function.signature.output_arg.add()
    output_arg.name = "y"
    output_arg.type = types_pb2.DT_INT32
    node = function.node_def.add(name="id", op=node_op)
    node.input.append("x")
    node.attr["T"].type = types_pb2.DT_INT32
    function.ret["y"] = "id:output:0"
    return function


def _function_graph():
    graph = _base_graph()
    _add_function(graph, "Fn")
    _add_function(graph, "Fn2")
    call = graph.node.add(name="call", op="PartitionedCall")
    call.input.append("source:0")
    call.attr["f"].func.name = "Fn"
    call.attr["Tin"].list.type.append(types_pb2.DT_INT32)
    call.attr["Tout"].list.type.append(types_pb2.DT_INT32)
    return graph


def _compare(left, right):
    return diagnostic.compare_graph_views(
        diagnostic.graph_view(left), diagnostic.graph_view(right)
    )


def test_entity_insertion_deletion_and_order() -> None:
    base = _base_graph()

    inserted = _clone(base)
    inserted.node.add(name="inserted", op="NoOp")
    delta = _compare(base, inserted)
    assert delta["node_delta"]["only_right"] == ["inserted"]
    assert delta["topology_changed"] is True

    deleted = _clone(base)
    del deleted.node[-1]
    delta = _compare(base, deleted)
    assert delta["node_delta"]["only_left"] == ["sink"]
    assert delta["topology_changed"] is True

    reordered = graph_pb2.GraphDef()
    for node in reversed(base.node):
        reordered.node.add().CopyFrom(node)
    delta = _compare(base, reordered)
    assert not delta["node_delta"]["only_left"]
    assert not delta["node_delta"]["only_right"]
    assert delta["order_changed"] is True
    assert delta["topology_changed"] is True


def test_node_semantic_mutations() -> None:
    base = _base_graph()

    mutations = []
    op_changed = _clone(base)
    op_changed.node[-1].op = "Mul"
    mutations.append(op_changed)

    data_rewired = _clone(base)
    data_rewired.node[-1].input[0] = "other:0"
    mutations.append(data_rewired)

    output_index = _clone(base)
    output_index.node[-1].input[0] = "source:1"
    mutations.append(output_index)

    control_rewired = _clone(base)
    control_rewired.node[-1].input[2] = "^control2"
    mutations.append(control_rewired)

    for changed in mutations:
        delta = _compare(base, changed)
        assert "sink" in delta["changed_node_semantics"]
        assert "sink" in delta["node_delta"]["changed"]
        assert delta["topology_changed"] is True

    assert diagnostic.edge_source("node:0") == (False, "node", 0)
    assert diagnostic.edge_source("node:output:3") == (False, "node", 3)
    assert diagnostic.edge_source("^node") == (True, "node", 0)


def test_function_semantic_mutations() -> None:
    base = _function_graph()

    call_target = _clone(base)
    call_target.node[-1].attr["f"].func.name = "Fn2"
    delta = _compare(base, call_target)
    assert "call" in delta["changed_node_semantics"]

    body = _clone(base)
    body.library.function[0].node_def[0].op = "Neg"
    delta = _compare(base, body)
    assert "Fn" in delta["changed_function_semantics"]

    signature = _clone(base)
    signature.library.function[0].signature.input_arg[0].type = types_pb2.DT_INT64
    delta = _compare(base, signature)
    assert "Fn" in delta["changed_function_semantics"]

    returned = _clone(base)
    returned.library.function[0].ret["y"] = "id:output:1"
    delta = _compare(base, returned)
    assert "Fn" in delta["changed_function_semantics"]

    controlled = _clone(base)
    controlled.library.function[0].signature.control_output.append("done")
    controlled.library.function[0].control_ret["done"] = "id"
    delta = _compare(base, controlled)
    assert "Fn" in delta["changed_function_semantics"]


def test_attribute_and_constant_mutations() -> None:
    base = _base_graph()

    dtype = _clone(base)
    dtype.node[-1].attr["T"].type = types_pb2.DT_INT64
    assert "sink" in _compare(base, dtype)["node_delta"]["changed"]

    device = _clone(base)
    device.node[-1].device = "/device:GPU:0"
    assert "sink" in _compare(base, device)["changed_node_semantics"]

    shape = _clone(base)
    shape.node[-1].attr["_output_shapes"].list.shape[0].dim[0].size = 3
    assert "sink" in _compare(base, shape)["node_delta"]["changed"]

    list_value = _clone(base)
    list_value.node[-1].attr["axes"].list.i[1] = 2
    assert "sink" in _compare(base, list_value)["node_delta"]["changed"]

    const_value = _clone(base)
    const_value.node[0].attr["value"].tensor.tensor_content = struct.pack("<ii", 1, 9)
    delta = _compare(base, const_value)
    assert "source" in delta["node_delta"]["changed"]
    assert delta["raw_equal"] is False


def test_cross_function_numeric_consumer_is_not_safe() -> None:
    graph = graph_pb2.GraphDef()
    shape_const = graph.node.add(name="shape_const")
    _const(shape_const, (1, 4))
    function = _add_function(graph, "NumericFn", node_op="Square")
    function.node_def[0].attr["T"].type = types_pb2.DT_INT32
    call = graph.node.add(name="call", op="PartitionedCall")
    call.input.append("shape_const:0")
    call.attr["f"].func.name = "NumericFn"
    call.attr["Tin"].list.type.append(types_pb2.DT_INT32)
    call.attr["Tout"].list.type.append(types_pb2.DT_INT32)

    consumers = diagnostic.const_consumers(graph, "top", "shape_const")
    assert consumers["direct_shape_only"] is False
    assert consumers["consumer_proven_shape_or_control_only"] is False
    assert consumers["consumer_boundary"] == "cross_function_ambiguous_or_value_reachable"
    assert consumers["called_function_ops"] == {"NumericFn": ["Square"]}


def test_duplicate_keys_and_incomplete_coverage_fail_closed() -> None:
    duplicate = _base_graph()
    duplicate.node.add(name="source", op="NoOp")
    with pytest.raises(diagnostic.DiagnosticError, match="duplicate"):
        diagnostic.graph_view(duplicate)

    graph = _base_graph()
    view = diagnostic.graph_view(graph)
    incomplete = copy.deepcopy(view)
    incomplete["raw_token_count"] -= 1
    with pytest.raises(diagnostic.DiagnosticError, match="incomplete"):
        diagnostic.validate_graph_view(graph, incomplete)


def test_axis_classification_requires_full_lattice() -> None:
    identities = [
        {"parameter_count": parameter_count, "batch_size": batch_size}
        for parameter_count in (50, 150)
        for batch_size in (1, 4, 16)
    ]
    batch_values = [[identity["batch_size"]] for identity in identities]
    parameter_values = [[identity["parameter_count"]] for identity in identities]
    product_values = [
        [identity["batch_size"] * identity["parameter_count"]]
        for identity in identities
    ]
    assert diagnostic.classify_axis(batch_values, identities) == "B"
    assert diagnostic.classify_axis(parameter_values, identities) == "P"
    assert diagnostic.classify_axis(product_values, identities) == "B*P"
    assert diagnostic.classify_axis(batch_values[:-1], identities[:-1]) == "incomplete_lattice"


def test_baseline_residual_partition_requires_declared_shapes_and_const_residuals() -> None:
    comparison = {
        "accepted_differences": [
            {
                "axis": "B",
                "inside_const": False,
                "rule_id": "static_shape_dimension_B",
            },
            {
                "axis": "P",
                "inside_const": False,
                "rule_id": "static_shape_dimension_P",
            },
        ],
        "rejected_differences": [
            {
                "inside_const": True,
                "rule_id": "rejected_unclassified_difference",
            },
            {
                "inside_const": False,
                "rule_id": "rejected_canonical_bytes_mismatch",
            },
        ],
    }
    summary = diagnostic.baseline_residual_summary(comparison)
    assert summary["complete_positional_residual_partition"] is True

    unsafe = copy.deepcopy(comparison)
    unsafe["rejected_differences"][0]["inside_const"] = False
    summary = diagnostic.baseline_residual_summary(unsafe)
    assert summary["complete_positional_residual_partition"] is False


def test_canonical_payload_excludes_only_run_fields() -> None:
    payload = {
        "evidence": {"count": 1},
        "run_manifest": {
            "started_utc": "first",
            "finished_utc": "second",
            "wall_seconds": 1.0,
            "output_path": "/tmp/first.json",
            "python": "/python",
        },
        "diagnostic_payload_sha256": "ignored",
    }
    changed_run = copy.deepcopy(payload)
    changed_run["run_manifest"].update(
        started_utc="different",
        finished_utc="different",
        wall_seconds=99.0,
        output_path="/tmp/second.json",
    )
    assert diagnostic.canonical_payload(payload) == diagnostic.canonical_payload(changed_run)

    changed_evidence = copy.deepcopy(payload)
    changed_evidence["evidence"]["count"] = 2
    assert diagnostic.canonical_payload(payload) != diagnostic.canonical_payload(changed_evidence)


def test_field_coverage_matrix_is_closed() -> None:
    classes = {row["semantic_class"] for row in diagnostic.FIELD_COVERAGE_MATRIX}
    assert classes == {
        "entity_insertion_deletion_order",
        "op_substitution",
        "data_edge_and_output_index",
        "control_edge",
        "function_call_target_and_body",
        "function_signature_ret_control_ret",
        "dtype_device_shape_and_list_attributes",
        "const_tensor_value",
        "cross_function_numeric_consumer",
        "duplicate_stable_keys",
        "incomplete_token_coverage",
        "full_lattice_axis_classification",
        "deterministic_payload",
    }


def test_entrypoint_has_no_target_call_surface() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "get_concrete_function",
        "jit_compile",
        "list_logical_devices",
        "list_physical_devices",
        "run_phase6_remaining",
        "tf_qr_sqrt_kalman_filter",
        "tf_qr_sqrt_kalman_score",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not forbidden & (names | attributes)

    subprocess_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
            subprocess_calls.append(node)
    assert len(subprocess_calls) == 1
    command = subprocess_calls[0].args[0]
    assert isinstance(command, ast.List)
    assert [ast.literal_eval(item) for item in command.elts] == ["git", "rev-parse", "HEAD"]
