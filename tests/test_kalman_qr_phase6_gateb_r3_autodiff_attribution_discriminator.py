from __future__ import annotations

import builtins
import copy
import runpy
import sys
from pathlib import Path


GUARD_SLOT = "_kalman_qr_phase6_autodiff_attribution_guard_20260713"
guard_state = getattr(builtins, GUARD_SLOT, None)
assert isinstance(guard_state, dict)
assert guard_state.get("token") == "guard_installed_before_subject_load_v1"
assert guard_state.get("mode") == "test"
assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules)

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "docs/benchmarks/discriminate_kalman_qr_phase6_gateb_r3_autodiff_attribution_2026_07_13.py"
).resolve()
assert str(MODULE_PATH) == guard_state.get("discriminator_path")
discriminator = runpy.run_path(
    str(MODULE_PATH),
    run_name="kalman_qr_phase6_autodiff_attribution_test_subject",
)
parent = discriminator["load_parent_localizer"]()
GraphDef, _descriptor_ledger = parent["graphdef_class"]()
DiscriminatorError = discriminator["DiscriminatorError"]


def _const(graph, name: str):
    node = graph.node.add(name=name, op="Const")
    node.attr["dtype"].type = 1
    node.attr["value"].tensor.dtype = 1
    node.attr["value"].tensor.float_val.append(1.0)
    return node


def _identity(graph, name: str, source: str):
    node = graph.node.add(name=name, op="Identity")
    node.input.append(source)
    node.attr["T"].type = 1
    return node


def _while_function(graph, name: str):
    function = graph.library.function.add()
    function.signature.name = name
    function.signature.input_arg.add(name="x", type=1)
    function.signature.output_arg.add(name="y", type=1)
    node = function.node_def.add(name="id", op="Identity")
    node.input.append("x")
    node.attr["T"].type = 1
    function.ret["y"] = "id:output:0"
    return function


def _while(graph, name: str, source: str, cond: str, body: str):
    node = graph.node.add(name=name, op="StatelessWhile")
    node.input.append(source)
    node.attr["T"].list.type.append(1)
    node.attr["cond"].func.name = cond
    node.attr["body"].func.name = body
    return node


def _fixture(*, duplicate_reverse: bool = False, renamed: bool = False):
    graph = GraphDef()
    names = {
        "parameter": "parameters_batch",
        "setup": "x1" if renamed else "setup",
        "forward": "x2" if renamed else "forward_loop",
        "value": "x3" if renamed else "value_output",
        "cotangent": "x4" if renamed else "cotangent",
        "reverse": "x5" if renamed else "reverse_loop",
        "score": "x6" if renamed else "score_output",
        "orphan": "x7" if renamed else "orphan_constant",
    }
    parameter = graph.node.add(name=names["parameter"], op="Placeholder")
    parameter.attr["dtype"].type = 1
    _identity(graph, names["setup"], names["parameter"])
    for fn in ("c0", "b0", "c1", "b1", "c2", "b2"):
        _while_function(graph, fn)
    _while(graph, names["forward"], names["setup"], "c0", "b0")
    _identity(graph, names["value"], names["forward"])
    cotangent = _const(graph, names["cotangent"])
    cotangent.input.append(names["forward"])
    _while(graph, names["reverse"], names["cotangent"], "c1", "b1")
    _identity(graph, names["score"], names["reverse"])
    if duplicate_reverse:
        other = _while(graph, "other_reverse", names["cotangent"], "c2", "b2")
        add = graph.node.add(name="score_add", op="AddV2")
        add.input.extend([names["score"], other.name])
        add.attr["T"].type = 1
        names["score"] = add.name
    _const(graph, names["orphan"])
    identity = {
        "identity_id": "fixture-autodiff",
        "method_id": parent["AUTODIFF_METHOD"],
        "dimension": 1,
        "parameter_count": 1,
        "batch_size": 1,
        "dtype": "float32",
        "operation": "trace",
    }
    record = {
        "identity": identity,
        "evidence": {
            "structured_user_input": {"name": names["parameter"], "shape": [1, 1]},
            "concrete_outputs": [
                {"name": f"{names['value']}:0", "result_position": "value"},
                {"name": f"{names['score']}:0", "result_position": "score"},
            ],
        },
    }
    row = {
        "graph": graph,
        "identity": identity,
        "record": record,
        "raw_sha256": discriminator["sha256_bytes"](
            graph.SerializeToString(deterministic=True)
        ),
    }
    return row, names


def _top_target(row, name: str, *, entity_kind: str = "top_node"):
    neighborhood = parent["neighborhood"](row["graph"], "top", name)
    target = {
        "target_key": f"{entity_kind}::{name}",
        "entity_kind": entity_kind,
        "name": name,
        "observations": [],
    }
    observation = {
        "graph_sha256": row["raw_sha256"],
        "neighborhood": neighborhood,
    }
    return target, observation


def _classify(row, names, key: str, *, entity_kind: str = "top_node"):
    analysis = discriminator["graph_analysis"](parent, row["graph"])
    boundary = discriminator["graph_boundaries"](parent, row, analysis)
    target, observation = _top_target(row, names[key], entity_kind=entity_kind)
    return discriminator["classify_atomic"](
        parent,
        row,
        boundary,
        target=target,
        observation=observation,
        nested=None,
        analysis=analysis,
    )


def test_guard_precedes_load_and_blocks_boundaries() -> None:
    assert guard_state["load_counts"] == {"discriminator": 1, "parent": 1}
    assert guard_state["blocked_os_system"] is True
    assert guard_state["blocked_os_write"] is True
    with pytest.raises(RuntimeError, match="forbidden import"):
        guard_state["check_import"]("tensorflow")
    with pytest.raises(RuntimeError, match="forbidden import"):
        guard_state["check_import"]("bayesfilter.linear.kalman_qr_tf")
    with pytest.raises(RuntimeError, match="device access"):
        guard_state["check_device_path"]("/dev/nvidia0")
    with pytest.raises(RuntimeError, match="write outside"):
        guard_state["check_write"]("/tmp/not-authorized.json")


def test_structural_boundaries_ignore_semantic_names_and_saved_forward_binding() -> None:
    normal, normal_names = _fixture()
    renamed, renamed_names = _fixture(renamed=True)
    for row, names in ((normal, normal_names), (renamed, renamed_names)):
        boundary = discriminator["graph_boundaries"](parent, row)
        assert boundary["forward"] == names["forward"]
        assert boundary["reverse"] == names["reverse"]
        assert boundary["value_output"] == names["value"]
        saved = boundary["reverse_candidates"][0]["saved_forward_bindings"]
        assert saved
        assert all(item["source"] != names["value"] for item in saved)
        assert any(item["source"] == names["cotangent"] for item in saved)


def test_first_match_top_level_states_and_paths_are_complete() -> None:
    row, names = _fixture()
    expected = {
        "parameter": "local_pre_forward_while",
        "setup": "local_pre_forward_while",
        "forward": "forward_while_call_or_function",
        "value": "forward_value_projection",
        "cotangent": "reverse_vjp_setup",
        "reverse": "reverse_while_call_or_function",
        "score": "post_reverse_vjp",
        "orphan": "constant_or_shape_origin_unresolved",
    }
    for key, state in expected.items():
        atomic = _classify(row, names, key)
        assert atomic["state"] == state
        assert atomic["witness"]["predicate_vector"][state] is True
        assert atomic["witness"]["state_specific"]


def test_function_owner_binding_and_debug_info_are_lossless() -> None:
    row, names = _fixture()
    boundary = discriminator["graph_boundaries"](parent, row)
    graph = row["graph"]
    graph.library.function[1].node_def[0].experimental_debug_info.original_node_names.append(
        "source-node"
    )
    graph.library.function[1].node_def[0].experimental_debug_info.original_func_names.append(
        "source-function"
    )
    owner, witness = discriminator["_function_owner_state"](parent, graph, "b0", boundary)
    assert owner == "forward_while_call_or_function"
    assert witness["bindings"][0]["caller"] == names["forward"]
    record = discriminator["debug_record"](graph.library.function[1].node_def[0])
    assert record["original_node_names"] == ["source-node"]
    assert record["original_func_names"] == ["source-function"]


def test_multiple_reverse_candidates_are_honest_ambiguity() -> None:
    row, names = _fixture(duplicate_reverse=True)
    boundary = discriminator["graph_boundaries"](parent, row)
    assert boundary["forward"] == names["forward"]
    assert boundary["reverse"] is None
    assert len(boundary["reverse_candidates"]) == 2
    atomic = _classify(row, names, "reverse")
    assert atomic["state"] == "structural_boundary_ambiguous"
    assert atomic["witness"]["state_specific"]["failed_predicate"]


def test_edge_call_output_and_function_mutations_change_witnesses() -> None:
    row, names = _fixture()
    base = discriminator["graph_boundaries"](parent, row)
    mutated = copy.deepcopy(row)
    reverse = next(node for node in mutated["graph"].node if node.name == names["reverse"])
    reverse.input[0] = names["orphan"]
    changed = discriminator["graph_boundaries"](parent, mutated)
    assert changed["reverse"] is None
    assert changed["reverse_candidates"] == []

    mutated = copy.deepcopy(row)
    mutated["record"]["evidence"]["concrete_outputs"][1]["name"] = "missing:0"
    with pytest.raises(DiscriminatorError, match="boundary node is missing"):
        discriminator["graph_boundaries"](parent, mutated)

    mutated = copy.deepcopy(row)
    forward = next(node for node in mutated["graph"].node if node.name == names["forward"])
    forward.attr["body"].func.name = "missing_function"
    with pytest.raises(DiscriminatorError, match="missing function"):
        discriminator["graph_boundaries"](parent, mutated)


def test_partition_validator_rejects_count_duplicate_state_and_witness_mutations() -> None:
    parent_artifact = discriminator["strict_load"](discriminator["PARENT_ARTIFACT_PATH"])
    targets = []
    for index in range(904):
        count = 14270 if index == 0 else 0
        targets.append(
            {
                "target_key": f"target-{index}",
                "entity_kind": "top_node",
                "atomics": [
                    {
                        "atomic_key": f"{atomic_index:064x}",
                        "state": "structural_boundary_ambiguous",
                        "witness": {
                            "atomic_key": f"{atomic_index:064x}",
                            "predicate_vector": {
                                state: state == "structural_boundary_ambiguous"
                                for state in discriminator["REGION_STATES"]
                            },
                            "state_specific": {"failed_predicate": "fixture"},
                        },
                    }
                    for atomic_index in range(count)
                ],
            }
        )
    valid = {
        "targets": targets,
        "target_count": 904,
        "observation_count": 12316,
        "nested_integer_occurrence_count": 2386,
        "atomic_count": 14270,
        "state_counts": {"structural_boundary_ambiguous": 14270},
    }
    discriminator["validate_atomic_partition"](valid)
    for field, value in (
        ("observation_count", 0),
        ("nested_integer_occurrence_count", 0),
        ("atomic_count", 0),
    ):
        mutated = copy.deepcopy(valid)
        mutated[field] = value
        with pytest.raises(DiscriminatorError):
            discriminator["validate_atomic_partition"](mutated)
    mutated = copy.deepcopy(valid)
    mutated["targets"][0]["atomics"][1]["atomic_key"] = mutated["targets"][0][
        "atomics"
    ][0]["atomic_key"]
    with pytest.raises(DiscriminatorError, match="incomplete or duplicate"):
        discriminator["validate_atomic_partition"](mutated)
    mutated = copy.deepcopy(valid)
    mutated["targets"][0]["atomics"][0]["state"] = "unresolved_invalid"
    mutated["state_counts"] = {
        "structural_boundary_ambiguous": 14269,
        "unresolved_invalid": 1,
    }
    with pytest.raises(DiscriminatorError, match="continuation veto"):
        discriminator["validate_atomic_partition"](mutated)
    assert parent_artifact["partition"]["target_count"] == 904


def test_canonical_payload_excludes_only_declared_run_metadata() -> None:
    payload = {
        "evidence": {"count": 1},
        "run_manifest": {
            "started_utc": "one",
            "finished_utc": "two",
            "wall_seconds": 1.0,
            "output_path": "/tmp/one",
        },
        "discriminator_payload_sha256": "ignored",
    }
    changed = copy.deepcopy(payload)
    changed["run_manifest"].update(
        started_utc="other",
        finished_utc="other",
        wall_seconds=99.0,
        output_path="/tmp/other",
    )
    assert discriminator["canonical_payload"](payload) == discriminator["canonical_payload"](
        changed
    )
    changed["evidence"]["count"] = 2
    assert discriminator["canonical_payload"](payload) != discriminator["canonical_payload"](
        changed
    )


def test_real_corpus_reconstructs_exact_atomic_partition() -> None:
    parent_artifact = discriminator["_validate_inputs"](parent)
    rows, _ledger = discriminator["_load_trace_rows"](parent)
    diagnostic = parent["strict_load"](parent["DIAGNOSTIC_PATH"])
    reconstructed = parent["target_partition"](rows, diagnostic)
    assert reconstructed == parent_artifact["partition"]
    partition = discriminator["atomic_partition"](parent, rows, reconstructed)
    assert partition["target_count"] == 904
    assert partition["observation_count"] == 12316
    assert partition["nested_integer_occurrence_count"] == 2386
    assert partition["atomic_count"] == 14270
    assert partition["state_counts"].get("unresolved_invalid", 0) == 0
    assert len(partition["boundaries"]) == 18
    assert all(boundary["forward"] is not None for boundary in partition["boundaries"])
    assert all(boundary["reverse"] is not None for boundary in partition["boundaries"])
