from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
RUNNER_PATH = ROOT / "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark():
    return _load(BENCHMARK_PATH, "kalman_qr_batched_fixture_benchmark")


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("parameter_count", [3, 50])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_vectorized_fixture_matches_stacked_scalar_all_16_tensors(
    benchmark, dtype, parameter_count: int, batch_size: int
) -> None:
    fixture = benchmark.make_fixture(10, parameter_count, 8, dtype=dtype)
    parameters_batch = benchmark._make_parameter_batch(fixture, batch_size)
    actual = benchmark._batched_model_tensors(fixture, parameters_batch)
    expected_rows = [
        benchmark._model_tensors(fixture, parameters_batch[row])
        for row in range(batch_size)
    ]
    expected = tuple(tf.stack(values, axis=0) for values in zip(*expected_rows, strict=True))

    assert len(actual) == len(expected) == 16
    tolerance = 2.0e-6 if dtype == tf.float32 else 2.0e-13
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.dtype == expected_tensor.dtype == dtype
        assert actual_tensor.shape == expected_tensor.shape
        tf.debugging.assert_near(
            actual_tensor,
            expected_tensor,
            rtol=tolerance,
            atol=tolerance,
        )


def test_batched_fixture_source_has_no_batch_mapping_or_python_loop(benchmark) -> None:
    source = BENCHMARK_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    targets = {
        "_batched_model_tensors",
        "_batched_factor_covariance_and_derivative",
        "_broadcast_vector_basis",
        "_broadcast_matrix_basis",
    }
    found = set()
    forbidden_calls = {"map_fn", "vectorized_map", "numpy_function"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets:
            found.add(node.name)
            assert not any(isinstance(child, (ast.For, ast.While, ast.ListComp)) for child in ast.walk(node))
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    assert child.func.attr not in forbidden_calls
    assert found == targets
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "tf.numpy_function" not in source


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_nested_parameter_basis_observation_and_proposal_contract(benchmark, dtype) -> None:
    fixtures = {
        parameter_count: benchmark.make_fixture(10, parameter_count, 8, dtype=dtype)
        for parameter_count in (50, 150)
    }
    small = fixtures[50]
    large = fixtures[150]

    tf.debugging.assert_equal(small.parameters, large.parameters[:50])
    for name in (
        "d_initial_mean",
        "d_initial_covariance_factor",
        "d_transition_offset",
        "d_transition_matrix",
        "d_transition_covariance_factor",
        "d_observation_offset",
        "d_observation_matrix",
        "d_observation_covariance_factor",
    ):
        tf.debugging.assert_equal(getattr(small, name), getattr(large, name)[:50])
    for name in (
        "base_initial_mean",
        "base_initial_covariance_factor",
        "base_transition_offset",
        "base_transition_matrix",
        "base_transition_covariance_factor",
        "base_observation_offset",
        "base_observation_matrix",
        "base_observation_covariance_factor",
        "observations",
    ):
        tf.debugging.assert_equal(getattr(small, name), getattr(large, name))

    expected_ids = {
        1: (7,),
        4: (2, 7, 8, 13),
        16: tuple(range(16)),
    }
    batches = {}
    for parameter_count, fixture in fixtures.items():
        cloud = benchmark._make_parameter_cloud(fixture)
        batches[parameter_count] = {}
        for batch_size, row_ids in expected_ids.items():
            assert benchmark._proposal_row_ids(batch_size) == row_ids
            selected = benchmark._make_parameter_batch(fixture, batch_size)
            tf.debugging.assert_equal(selected, tf.gather(cloud, row_ids))
            batches[parameter_count][batch_size] = {
                row_id: selected[index] for index, row_id in enumerate(row_ids)
            }

    for row_id in expected_ids[1]:
        tf.debugging.assert_equal(batches[50][1][row_id], batches[50][4][row_id])
    for row_id in expected_ids[4]:
        tf.debugging.assert_equal(batches[50][4][row_id], batches[50][16][row_id])
        tf.debugging.assert_equal(
            batches[50][4][row_id], batches[150][4][row_id][:50]
        )
    for row_id in expected_ids[16]:
        tf.debugging.assert_equal(
            batches[50][16][row_id], batches[150][16][row_id][:50]
        )


def test_unsupported_batch_and_parameter_count_fail_closed(benchmark) -> None:
    fixture = benchmark.make_fixture(10, 3, 8, dtype=tf.float32)
    with pytest.raises(ValueError, match="unsupported batch_size"):
        benchmark._make_parameter_batch(fixture, 2)
    with pytest.raises(ValueError, match="nested scaling maximum"):
        benchmark.make_fixture(10, 151, 8, dtype=tf.float32)


def test_trace_only_graph_structure_is_identical_across_batch(benchmark) -> None:
    fixture = benchmark.make_fixture(10, 50, 8, dtype=tf.float32)
    rows = [
        benchmark._trace_batched_fixture_graph(fixture, batch_size=batch_size)
        for batch_size in (1, 4, 16)
    ]
    assert len({row["node_count"] for row in rows}) == 1
    assert len({row["normalized_structural_digest"] for row in rows}) == 1
    assert [row["batch_size"] for row in rows] == [1, 4, 16]
    for row in rows:
        assert len(row["output_shapes"]) == 16
        assert all(shape[0] == row["batch_size"] for shape in row["output_shapes"])


def _trace_graph(benchmark):
    fixture = benchmark.make_fixture(10, 3, 2, dtype=tf.float32)
    row = benchmark._trace_batched_fixture_graph(fixture, batch_size=4)
    return row["graph_def"], row["parameter_input_name"]


@pytest.mark.parametrize(
    "mutation",
    [
        "op",
        "edge",
        "attribute",
        "duplicate_node",
        "constant_role",
        "constant_payload",
        "constant_element_count",
        "constant_dtype",
        "constant_rank",
        "non_batch_dimension",
    ],
)
def test_graph_normalizer_detects_forbidden_mutations(benchmark, mutation: str) -> None:
    graph_def, input_name = _trace_graph(benchmark)
    baseline = benchmark._normalized_graphdef_digest(
        graph_def,
        parameter_input_name=input_name,
        batch_size=4,
    )
    changed = copy.deepcopy(graph_def)

    if mutation == "op":
        next(node for node in changed.node if node.op != "Placeholder").op = "IdentityN"
    elif mutation == "edge":
        next(node for node in changed.node if node.input).input[0] = input_name
    elif mutation == "attribute":
        next(node for node in changed.node if node.attr).attr["phase2_mutation"].b = True
    elif mutation == "duplicate_node":
        changed.node.add().CopyFrom(changed.node[-1])
    elif mutation == "constant_role":
        constants = [node for node in changed.node if node.op == "Const"]
        original = constants[0]
        replacement = next(node for node in constants[1:] if node.name != original.name)
        consumer = next(
            node
            for node in changed.node
            if any(
                benchmark._graph_input_base_name(edge) == original.name
                for edge in node.input
            )
        )
        edge_index = next(
            index
            for index, edge in enumerate(consumer.input)
            if benchmark._graph_input_base_name(edge) == original.name
        )
        consumer.input[edge_index] = replacement.name
    else:
        constant = next(node for node in changed.node if node.op == "Const")
        tensor = constant.attr["value"].tensor
        if mutation == "constant_payload":
            tensor.tensor_content = tensor.tensor_content + b"x"
        elif mutation == "constant_element_count":
            tensor.tensor_shape.dim.add().size = 2
        elif mutation == "constant_dtype":
            tensor.dtype = tf.float64.as_datatype_enum
        elif mutation == "constant_rank":
            tensor.tensor_shape.dim.add().size = 1
        elif mutation == "non_batch_dimension":
            placeholder = next(node for node in changed.node if node.name == input_name)
            placeholder.attr["shape"].shape.dim[1].size += 1

    mutated = benchmark._normalized_graphdef_digest(
        changed,
        parameter_input_name=input_name,
        batch_size=4,
    )
    assert mutated != baseline


def test_graph_normalizer_allows_only_leading_batch_shape_change(benchmark) -> None:
    graph_def, input_name = _trace_graph(benchmark)
    baseline = benchmark._normalized_graphdef_digest(
        graph_def,
        parameter_input_name=input_name,
        batch_size=4,
    )
    changed = copy.deepcopy(graph_def)
    descendants = {input_name}
    for node in changed.node:
        if node.name == input_name or any(
            benchmark._graph_input_base_name(edge) in descendants for edge in node.input
        ):
            descendants.add(node.name)
            for attr_name in ("shape", "_output_shapes"):
                if attr_name not in node.attr:
                    continue
                shapes = (
                    [node.attr[attr_name].shape]
                    if attr_name == "shape"
                    else node.attr[attr_name].list.shape
                )
                for shape in shapes:
                    if not shape.unknown_rank and shape.dim and shape.dim[0].size == 4:
                        shape.dim[0].size = 16
    assert benchmark._normalized_graphdef_digest(
        changed,
        parameter_input_name=input_name,
        batch_size=16,
    ) == baseline


def test_phase2_versions_invalidate_phase1_schedule_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_phase2_version_runner")
    phase2_versions = (
        runner.FIXTURE_CONTRACT_VERSION,
        runner.PARAMETER_BATCH_VERSION,
        runner.OBSERVATION_GENERATION_VERSION,
    )
    assert phase2_versions == (
        "nested-prefix-base-observation-phase2-v1",
        "canonical-b16-locked-row-map-phase2-v1",
        "base-model-deterministic-sine-phase2-v1",
    )
    args = argparse.Namespace(
        dimensions=[2],
        parameter_counts=[2],
        timesteps=2,
        batch_size=1,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        repeats=1,
        timeout_seconds=1.0,
        methods=[runner.contract.METHOD_IDS[0]],
        output_dir=tmp_path,
        harness_contract_test_only=True,
        no_resume=False,
        jit_compile=False,
        tf32_enabled=True,
    )

    monkeypatch.setattr(runner, "FIXTURE_CONTRACT_VERSION", "historical-phase1-v1")
    monkeypatch.setattr(runner, "PARAMETER_BATCH_VERSION", "requested-batch-phase1-v1")
    monkeypatch.setattr(
        runner,
        "OBSERVATION_GENERATION_VERSION",
        "parameterized-model-phase1-v1",
    )
    phase1_schedule = runner.build_schedule(args)
    phase1_identity = phase1_schedule["expected_identities"][0]

    monkeypatch.setattr(runner, "FIXTURE_CONTRACT_VERSION", phase2_versions[0])
    monkeypatch.setattr(runner, "PARAMETER_BATCH_VERSION", phase2_versions[1])
    monkeypatch.setattr(runner, "OBSERVATION_GENERATION_VERSION", phase2_versions[2])
    phase2_schedule = runner.build_schedule(args)
    phase2_identity = phase2_schedule["expected_identities"][0]

    assert phase1_identity["source_fingerprint"] == phase2_identity["source_fingerprint"]
    assert phase1_identity["runtime_fingerprint"] == phase2_identity["runtime_fingerprint"]
    assert phase1_identity["case_id"] == phase2_identity["case_id"]
    assert phase1_identity["config_fingerprint"] != phase2_identity["config_fingerprint"]
    assert phase1_identity["fixture_fingerprint"] != phase2_identity["fixture_fingerprint"]
    assert phase1_schedule["schedule_fingerprint"] != phase2_schedule["schedule_fingerprint"]

    phase1_fingerprints = runner._fingerprints(
        phase1_identity, phase1_schedule["schedule_fingerprint"]
    )
    stale_record = {
        "schema": runner.contract.SCHEMA,
        "case_id": phase1_identity["case_id"],
        "method_id": phase1_identity["method_id"],
        **phase1_fingerprints,
        "resume_key": runner.contract.resume_key(
            case_identity=phase1_identity["case_id"],
            method_id=phase1_identity["method_id"],
            fingerprints=phase1_fingerprints,
        ),
        "state": "passed",
        "attempt_id": "phase1-attempt",
        "last_entered_stage": "artifact_write",
        "terminal_stage": "artifact_write",
        "failure_stage": None,
        "invoked_method_ids": [phase1_identity["method_id"]],
        "output_metadata": {
            "all_finite": True,
            "value_shape": [1],
            "score_shape": [1, 2],
            "value_dtype": "float32",
            "score_dtype": "float32",
        },
    }
    stale_path = runner.method_artifact_path(tmp_path, phase2_identity)
    runner.contract.atomic_write_json(stale_path, stale_record)

    reusable, reason = runner._read_reusable_record(
        stale_path,
        identity=phase2_identity,
        schedule_fingerprint=phase2_schedule["schedule_fingerprint"],
    )
    assert reusable is None
    assert reason == "config_fingerprint_mismatch"
