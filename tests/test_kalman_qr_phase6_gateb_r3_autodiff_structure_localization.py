from __future__ import annotations

import builtins
import runpy
import sys
from pathlib import Path


GUARD_SLOT = "_kalman_qr_phase6_autodiff_structure_guard_20260713"
guard_state = getattr(builtins, GUARD_SLOT, None)
assert isinstance(guard_state, dict)
assert guard_state.get("token") == "guard_installed_before_subject_load_v1"
assert guard_state.get("mode") == "test"
assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules)

import copy
import struct
from functools import lru_cache

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "docs/benchmarks/localize_kalman_qr_phase6_gateb_r3_autodiff_structure_2026_07_13.py"
).resolve()
assert str(MODULE_PATH) == guard_state.get("localizer_path")
localizer = runpy.run_path(
    str(MODULE_PATH),
    run_name="kalman_qr_phase6_autodiff_structure_test_subject",
)


LocalizationError = localizer["LocalizationError"]


@lru_cache(maxsize=1)
def _graph_class():
    graph_class, _ = localizer["graphdef_class"]()
    return graph_class


def _clone(graph):
    result = _graph_class()()
    result.ParseFromString(graph.SerializeToString(deterministic=True))
    return result


def _const(node, values=(1,)) -> None:
    node.op = "Const"
    node.attr["dtype"].type = 3
    tensor = node.attr["value"].tensor
    tensor.dtype = 3
    tensor.tensor_shape.dim.add(size=len(values))
    tensor.tensor_content = struct.pack("<" + "i" * len(values), *values)


def _base_graph():
    graph = _graph_class()()
    source = graph.node.add(name="source")
    _const(source, (1, 2))
    other = graph.node.add(name="other")
    _const(other, (3, 4))
    sink = graph.node.add(name="sink", op="AddV2")
    sink.input.extend(["source:0", "other:0"])
    sink.attr["T"].type = 3
    return graph


def _add_function(graph, name: str, node_op: str = "Identity"):
    function = graph.library.function.add()
    function.signature.name = name
    input_arg = function.signature.input_arg.add(name="x", type=3)
    output_arg = function.signature.output_arg.add(name="y", type=3)
    assert input_arg.name == "x" and output_arg.name == "y"
    node = function.node_def.add(name="id", op=node_op)
    node.input.append("x")
    node.attr["T"].type = 3
    function.ret["y"] = "id:output:0"
    return function


def _function_graph():
    graph = _base_graph()
    _add_function(graph, "Fn")
    _add_function(graph, "Fn2")
    call = graph.node.add(name="call", op="PartitionedCall")
    call.input.append("source:0")
    call.attr["f"].func.name = "Fn"
    call.attr["Tin"].list.type.append(3)
    call.attr["Tout"].list.type.append(3)
    return graph


def _compare(left, right):
    return localizer["compare_graph_views"](
        localizer["graph_view"](left),
        localizer["graph_view"](right),
    )


@lru_cache(maxsize=1)
def _anchors():
    return localizer["source_anchor_ledger"]()


def _valid_partition():
    graph = _base_graph()
    pair = "d=10/P=50/B=1->P=50/B=4"
    target = {
        "target_key": "top_node::sink",
        "entity_kind": "top_node",
        "name": "sink",
        "observations": [
            {
                "dimension": 10,
                "pair": pair,
                "pair_side": "P=50/B=1",
                "delta_kind": "changed",
                "identity": {"identity_id": "left"},
                "graph_sha256": "0" * 64,
                "neighborhood": localizer["neighborhood"](graph, "top", "sink"),
            },
            {
                "dimension": 10,
                "pair": pair,
                "pair_side": "P=50/B=4",
                "delta_kind": "changed",
                "identity": {"identity_id": "right"},
                "graph_sha256": "1" * 64,
                "neighborhood": localizer["neighborhood"](graph, "top", "sink"),
            },
        ],
        "coverage_state": "enumerated_causally_ambiguous",
        "causal_alternatives": localizer["causal_alternatives"](
            "top_node", "sink", "AddV2"
        ),
        "exact_anchor": None,
        "causal_claim": "none",
    }
    return {
        "targets": [target],
        "target_count": 1,
        "coverage_counts": {
            "mapped_exact": 0,
            "enumerated_causally_ambiguous": 1,
            "missing_or_incomplete": 0,
        },
        "complete": True,
        "repair_eligible": False,
        "discriminator_eligible": True,
        "next_branch": "autodiff_attribution_discriminator",
        "repair_hypothesis": None,
    }


def _validate(partition) -> None:
    localizer["validate_partition"](partition, {"top_node::sink"}, _anchors())


def test_guard_was_installed_before_load_and_blocks_boundaries() -> None:
    assert guard_state["run_path_count"]["value"] == 1
    assert guard_state["blocked_os_system"] is True
    assert guard_state["blocked_os_write"] is True
    with pytest.raises(RuntimeError, match="forbidden import"):
        guard_state["check_import"]("tensorflow")
    with pytest.raises(RuntimeError, match="forbidden import"):
        guard_state["check_import"]("bayesfilter.linear.kalman_qr_tf")
    with pytest.raises(RuntimeError, match="device access"):
        guard_state["check_device_path"]("/dev/nvidia0")
    with pytest.raises(RuntimeError, match="write outside"):
        guard_state["check_write"]("/tmp/guard-violation.json")


def test_descriptor_closure_rejects_truncated_missing_and_unknown_data() -> None:
    literals, ledger = localizer["descriptor_literals"]()
    assert set(literals) == set(localizer["DESCRIPTOR_HASHES"])
    assert len(ledger) == len(literals) == 12

    missing = dict(literals)
    missing.pop("graph")
    with pytest.raises(LocalizationError, match="descriptor name closure mismatch"):
        localizer["graphdef_class"](missing)

    unknown = dict(literals)
    unknown["unknown"] = b"not-a-descriptor"
    with pytest.raises(LocalizationError, match="descriptor name closure mismatch"):
        localizer["graphdef_class"](unknown)

    truncated = dict(literals)
    truncated["graph"] = b"\x00"
    with pytest.raises(LocalizationError, match="incomplete descriptor dependency closure"):
        localizer["graphdef_class"](truncated)


def test_missing_duplicate_entities_and_identities_fail_closed() -> None:
    duplicate_graph = _base_graph()
    duplicate_graph.node.add(name="source", op="NoOp")
    with pytest.raises(LocalizationError, match="duplicate"):
        localizer["graph_view"](duplicate_graph)

    rows = [
        {"identity": {"identity_id": f"identity-{index}"}}
        for index in range(36)
    ]
    localizer["validate_trace_rows"](rows)
    rows[-1]["identity"]["identity_id"] = rows[0]["identity"]["identity_id"]
    with pytest.raises(LocalizationError, match="36 unique"):
        localizer["validate_trace_rows"](rows)

    missing = _valid_partition()
    missing["targets"] = []
    missing["target_count"] = 0
    missing["coverage_counts"]["enumerated_causally_ambiguous"] = 0
    with pytest.raises(LocalizationError, match="absent or empty"):
        _validate(missing)

    duplicate = _valid_partition()
    duplicate["targets"].append(copy.deepcopy(duplicate["targets"][0]))
    duplicate["target_count"] = 2
    duplicate["coverage_counts"]["enumerated_causally_ambiguous"] = 2
    with pytest.raises(LocalizationError, match="duplicate target identity"):
        _validate(duplicate)


def test_edge_op_and_function_target_changes_are_preserved() -> None:
    base = _base_graph()

    edge = _clone(base)
    edge.node[-1].input[0] = "other:1"
    delta = _compare(base, edge)
    assert "sink" in delta["node_delta"]["changed"]
    assert "sink" in delta["changed_node_semantics"]

    op = _clone(base)
    op.node[-1].op = "Mul"
    delta = _compare(base, op)
    assert "sink" in delta["node_delta"]["changed"]
    assert delta["topology_changed"] is True

    function_graph = _function_graph()
    call_target = _clone(function_graph)
    call_target.node[-1].attr["f"].func.name = "Fn2"
    assert "call" in _compare(function_graph, call_target)["changed_node_semantics"]

    function_body = _clone(function_graph)
    function_body.library.function[0].node_def[0].op = "Neg"
    function_delta = _compare(function_graph, function_body)
    assert "Fn" in function_delta["function_delta"]["changed"]
    assert "id" in function_delta["changed_function_nodes"]["Fn"]["changed"]


def test_partition_rejects_stale_anchor_incomplete_slice_and_weak_ambiguity() -> None:
    valid = _valid_partition()
    _validate(valid)

    stale = copy.deepcopy(valid)
    stale["targets"][0]["causal_alternatives"][0]["anchors"][0] = (
        "scripts/benchmark_kalman_qr_parameter_count_scaling.py:1"
    )
    with pytest.raises(LocalizationError, match="stale or promoted"):
        _validate(stale)

    incomplete = copy.deepcopy(valid)
    incomplete["targets"][0]["observations"][0]["neighborhood"].pop("consumers")
    with pytest.raises(LocalizationError, match="incomplete graph slice"):
        _validate(incomplete)

    one_sided = copy.deepcopy(valid)
    one_sided["targets"][0]["observations"].pop()
    with pytest.raises(LocalizationError, match="incomplete pair sides"):
        _validate(one_sided)

    weak_ambiguity = copy.deepcopy(valid)
    weak_ambiguity["targets"][0]["causal_alternatives"] = weak_ambiguity["targets"][0][
        "causal_alternatives"
    ][:1]
    with pytest.raises(LocalizationError, match="ambiguity is not bounded"):
        _validate(weak_ambiguity)


def test_false_exact_mapping_and_forbidden_causal_claims_fail_closed() -> None:
    false_exact = _valid_partition()
    target = false_exact["targets"][0]
    target["coverage_state"] = "mapped_exact"
    target["causal_alternatives"] = []
    target["exact_anchor"] = {
        "anchor": "scripts/benchmark_kalman_qr_parameter_count_scaling.py:712",
        "candidate_anchors": [
            "scripts/benchmark_kalman_qr_parameter_count_scaling.py:712",
            "scripts/benchmark_kalman_qr_parameter_count_scaling.py:1893",
        ],
        "candidate_count": 2,
        "graph_evidence_sha256": "2" * 64,
        "uniqueness_evidence": "fixture claim without uniqueness",
    }
    false_exact["coverage_counts"] = {
        "mapped_exact": 1,
        "enumerated_causally_ambiguous": 0,
        "missing_or_incomplete": 0,
    }
    with pytest.raises(LocalizationError, match="false exact mapping"):
        _validate(false_exact)

    for unsupported in ("avoidable", "inherent"):
        causal = _valid_partition()
        causal["targets"][0]["causal_claim"] = unsupported
        with pytest.raises(LocalizationError, match="forbidden causal claim"):
            _validate(causal)


def test_stale_source_range_and_diagnostic_hash_mismatch_fail_closed() -> None:
    definition = dict(localizer["LOCAL_SOURCE_ANCHORS"][0])
    definition["line_end"] = 10**9
    relative = definition["path"]
    with pytest.raises(LocalizationError, match="invalid source anchor range"):
        localizer["_source_anchor_record"](
            definition,
            path=ROOT / relative,
            expected_sha256=localizer["SOURCE_HASHES"][relative],
        )

    diagnostic = localizer["strict_load"](localizer["DIAGNOSTIC_PATH"])
    localizer["validate_diagnostic_payload"](diagnostic)

    mutated = dict(diagnostic)
    mutated["state"] = "mutated"
    with pytest.raises(LocalizationError, match="canonical payload mismatch"):
        localizer["validate_diagnostic_payload"](mutated)

    stale_digest = dict(diagnostic)
    stale_digest["diagnostic_payload_sha256"] = "0" * 64
    with pytest.raises(LocalizationError, match="embedded payload digest mismatch"):
        localizer["validate_diagnostic_payload"](stale_digest)


def test_real_corpus_partition_is_complete_and_causally_ambiguous() -> None:
    diagnostic = localizer["strict_load"](localizer["DIAGNOSTIC_PATH"])
    graph_class = _graph_class()
    rows = localizer["trace_rows"](graph_class)
    parity = localizer["parity"](rows, diagnostic)
    assert all(parity["checks"].values())

    expected_keys = localizer["expected_target_keys"](diagnostic)
    partition = localizer["target_partition"](rows, diagnostic)
    localizer["validate_partition"](partition, expected_keys, _anchors())
    assert partition["target_count"] == len(expected_keys)
    assert partition["coverage_counts"]["missing_or_incomplete"] == 0
    assert partition["coverage_counts"]["mapped_exact"] == 0
    assert partition["coverage_counts"]["enumerated_causally_ambiguous"] == len(
        expected_keys
    )
    assert partition["next_branch"] == "autodiff_attribution_discriminator"

    autodiff_cohorts = [
        cohort
        for cohort in diagnostic["cohorts"]
        if cohort["method_id"] == localizer["AUTODIFF_METHOD"]
    ]
    assert [
        len(cohort["axis_constant_analysis"]["differing_integer_consts"])
        for cohort in autodiff_cohorts
    ] == [144, 144, 144]


def test_canonical_digest_and_pure_file_git_provenance_are_deterministic() -> None:
    payload = {
        "evidence": {"count": 1},
        "run_manifest": {
            "started_utc": "first",
            "finished_utc": "second",
            "wall_seconds": 1.0,
            "output_path": "/tmp/first.json",
            "git_commit": localizer["git_commit"](),
        },
        "localization_payload_sha256": "ignored",
    }
    changed_run = copy.deepcopy(payload)
    changed_run["run_manifest"].update(
        started_utc="different",
        finished_utc="different",
        wall_seconds=99.0,
        output_path="/tmp/second.json",
    )
    assert localizer["canonical_payload"](payload) == localizer["canonical_payload"](
        changed_run
    )

    changed_evidence = copy.deepcopy(payload)
    changed_evidence["evidence"]["count"] = 2
    assert localizer["canonical_payload"](payload) != localizer["canonical_payload"](
        changed_evidence
    )

    commit = localizer["git_commit"]()
    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)
    with pytest.raises(LocalizationError, match="output path is not authorized"):
        localizer["validate_output_path"](Path("/tmp/not-authorized.json"))
