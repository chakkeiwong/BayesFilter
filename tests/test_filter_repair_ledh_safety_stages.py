"""Independent execution/precision diagnostics for shared LEDH stage helpers."""

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_canonical_score_stages_tf as stages
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_kdm_auxiliary import _assert_record_equal, _json


def _frozen(name):
    path = f"bayesfilter/highdim/{name}.py"
    source = subprocess.check_output(["git", "show", f"04643213e:{path}"],
                                     cwd=Path(__file__).resolve().parents[1], text=True)
    module = ModuleType("_independent_original_" + name)
    exec(compile(source, "<frozen-ledh-reference>", "exec"), module.__dict__)  # noqa: S102
    return module, hashlib.sha256(source.encode()).hexdigest()


def _write(request, name, report):
    path = Path(request.config.getoption("xmlpath")).parent / name
    path.write_text(json.dumps(_json(report), indent=2, allow_nan=False) + "\n")


def test_guarded_shared_helper_wiring():
    from bayesfilter.highdim import ledh_canonical_reset_score_tf as adapter
    from bayesfilter.highdim import ledh_canonical_score_tf as score
    from bayesfilter.highdim import ledh_unified_reset_tf as reset

    assert adapter.batched_sinkhorn_contract_e_reset_triple_with_tangent is reset.batched_sinkhorn_contract_e_reset_triple_with_tangent
    assert reset.safe_cholesky is safe_cholesky
    assert score.ukf_predict_with_parameter_tangent is stages.ukf_predict_with_parameter_tangent
    assert score.ukf_update_with_parameter_tangent is stages.ukf_update_with_parameter_tangent
    policy = json.loads((Path(__file__).resolve().parents[1] / "scripts/filter_gradient_runtime_policy.json").read_text())
    for module in (adapter, reset, stages):
        path = Path(module.__file__).relative_to(Path(__file__).resolve().parents[1])
        assert policy["sources"][str(path)] is None
    assert policy["sources"]["bayesfilter/highdim/ledh_numerical_safety_tf.py"] is None


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("batch_shape", [(), (1,), (3,), (1, 3), (2, 1, 3)])
def test_safety_shapes_and_rejections(batch_shape, dtype, request):
    reference, source_sha = _frozen("ledh_numerical_safety_tf")
    matrices = tf.broadcast_to(tf.constant([[2., 0.1], [0.1, 1.]], dtype), [*batch_shape, 2, 2])
    singular = tf.zeros_like(matrices)
    indefinite = matrices * tf.constant([[-1., 1.], [1., 1.]], dtype)
    infinite = tf.broadcast_to(tf.constant([[float("inf"), 0.], [0., 1.]], dtype), matrices.shape)
    nan_input = tf.fill(matrices.shape, tf.constant(float("nan"), dtype))
    controls = [matrices, singular, indefinite, nan_input]
    if batch_shape:
        mixed = tf.tensor_scatter_nd_update(tf.reshape(matrices, [-1, 2, 2]),
                                           [[0]], [tf.zeros([2, 2], dtype)])
        controls.append(tf.reshape(mixed, matrices.shape))
    report = {"shape": matrices.shape.as_list(), "dtype": dtype.name,
              "reference_sha256": source_sha, "arms": []}
    for jit_compile in (False, True):
        signature = [tf.TensorSpec(matrices.shape, dtype)]
        old = tf.function(reference.safe_cholesky, input_signature=signature,
                          autograph=False, jit_compile=jit_compile)
        new = tf.function(safe_cholesky, input_signature=signature,
                          autograph=False, jit_compile=jit_compile)
        records = []
        for matrix in controls:
            actual, expected = new(matrix), old(matrix)
            _assert_record_equal(actual, expected, atol=0., rtol=0.)
            records.append({"actual": actual, "expected": expected})
        rejected, factor = new(infinite)
        assert not bool(tf.reduce_any(rejected).numpy())
        np.testing.assert_array_equal(factor.numpy(), tf.zeros_like(factor).numpy())
        assert tuple(rejected.shape) == tuple(size for size in batch_shape if size != 1)
        report["arms"].append({"jit_compile": jit_compile, "records": records,
                               "infinite_original": old(infinite),
                               "infinite_repaired": (rejected, factor),
                               "traces": new.experimental_get_tracing_count()})
        assert new.experimental_get_tracing_count() == 1
    name = "-".join(map(str, batch_shape)) or "scalar"
    _write(request, f"ledh-safety-{name}-{dtype.name}.json", report)


def _fixture(dtype):
    rng = np.random.default_rng(41)
    n, dim, horizon = 4, 2, 3
    return (
        tf.constant([0.7], dtype),
        tf.constant([[0.9, 0.1], [0.05, 0.8]], dtype),
        tf.eye(dim, dtype=dtype) * tf.constant(0.4, dtype),
        tf.eye(dim, dtype=dtype),
        tf.eye(dim, dtype=dtype) * tf.constant(0.6, dtype),
        tf.constant(rng.standard_normal((n, dim)), dtype),
        tf.eye(dim, batch_shape=[n], dtype=dtype),
        tf.constant(rng.standard_normal((horizon, n, dim)), dtype),
        tf.constant(rng.standard_normal((horizon, dim)), dtype),
    )


def _program(module, family, inputs, jit_compile, *, substeps=7, with_tangent=True):
    implementation = (module.flow_value_and_parameter_tangent_lgssm
                      if family == "flow" else module.multi_step_value_and_score_lgssm)
    return tf.function(lambda *args: implementation(*args, substeps=substeps, with_tangent=with_tangent),
                       input_signature=[tf.TensorSpec(x.shape, x.dtype) for x in inputs],
                       autograph=False, jit_compile=jit_compile)


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32], ids=["f64", "f32"])
def test_native_stages_complete_records(dtype, request):
    original, source_sha = _frozen("ledh_canonical_score_stages_tf")
    args = _fixture(dtype)
    directory = Path(request.config.getoption("xmlpath")).parent
    report = {"reference_sha256": source_sha, "dtype": dtype.name, "records": []}
    tolerance = 2.e-10 if dtype == tf.float64 else 1.e-6
    for family in ("flow", "recursion"):
        inputs = (*args[:7], args[7][0], args[8][0]) if family == "flow" else args
        try:
            original_xla = _program(original, family, inputs, True)(*inputs)
            original_compilation = {"passed": True, "result": original_xla}
        except tf.errors.InvalidArgumentError as error:
            assert "MatrixDeterminant" in str(error)
            original_compilation = {"passed": False, "error": str(error)}
        for jit_compile in (False, True):
            # The frozen determinant operation lacks an XLA CPU kernel. This
            # is an explicit independent graph reference, never a runtime fallback.
            reference = _program(original, family, inputs, False)
            owner = _program(stages, family, inputs, jit_compile)
            first, expected = owner(*inputs), reference(*inputs)
            _assert_record_equal(first, expected, atol=tolerance, rtol=tolerance)
            _assert_record_equal(first, owner(*inputs), atol=0., rtol=0.)
            changed = (inputs[0] + tf.constant(0.01, dtype), *inputs[1:])
            changed_result, changed_reference = owner(*changed), reference(*changed)
            _assert_record_equal(changed_result, changed_reference, atol=tolerance, rtol=tolerance)
            derivative = None
            if dtype == tf.float64:
                epsilon = 2.e-5
                plus, minus = owner(inputs[0] + epsilon, *inputs[1:]), owner(inputs[0] - epsilon, *inputs[1:])
                derivative = (plus[0] - minus[0]) / (2 * epsilon)
                tangent = first[1][..., 0] if family == "flow" else first[1][0]
                np.testing.assert_allclose(tangent.numpy(), derivative.numpy(), rtol=1.e-4, atol=1.e-6)
            graph = owner.get_concrete_function().graph.as_graph_def()
            nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
            ops = Counter(n.op for n in nodes)
            assert {"While", "StatelessWhile"}.intersection(ops)
            assert not {"PyFunc", "EagerPyFunc", "PyFuncStateless"}.intersection(ops)
            item = {"family": family, "jit_compile": jit_compile, "first": first,
                    "reference_jit_compile": False, "original_xla": original_compilation,
                    "reference": expected, "changed": changed_result,
                    "changed_reference": changed_reference, "finite_difference": derivative,
                    "ops": ops, "traces": owner.experimental_get_tracing_count()}
            if jit_compile:
                hlo = owner.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
                (directory / f"ledh-stage-{family}-{dtype.name}.hlo").write_text(hlo)
                item["hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
                assert " while(" in hlo
            assert owner.experimental_get_tracing_count() == 1
            report["records"].append(item)
            _write(request, f"ledh-stage-{dtype.name}.json", report)
        value_owner = _program(stages, family, inputs, True, with_tangent=False)
        value_reference = _program(original, family, inputs, False, with_tangent=False)
        _assert_record_equal(value_owner(*inputs), value_reference(*inputs), atol=tolerance, rtol=tolerance)
    empty = (*args[:7], args[7][:0], args[8][:0])
    _assert_record_equal(_program(stages, "recursion", empty, True)(*empty),
                         _program(original, "recursion", empty, False)(*empty), atol=0., rtol=0.)


@pytest.mark.parametrize("jit_compile", [False, True], ids=["graph", "xla"])
@pytest.mark.parametrize("arm", ["original", "native"])
def test_stage_cost(arm, jit_compile, request):
    original, source_sha = _frozen("ledh_canonical_score_stages_tf")
    module = original if arm == "original" else stages
    inputs = _fixture(tf.float64)
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    with GPUProcessMonitor(gpu) as monitor:
        before = memory_snapshot(gpu)
        begin = time.perf_counter()
        owner = _program(module, "recursion", inputs, jit_compile)
        result = _json(owner(*inputs))
        cold = time.perf_counter() - begin
        compiled = memory_snapshot(gpu)
        timings = []
        for _ in range(20):
            begin = time.perf_counter()
            replay = _json(owner(*inputs))
            timings.append(time.perf_counter() - begin)
            assert replay == result
        warm = memory_snapshot(gpu)
    _write(request, "ledh-stage-cost.json", {"arm": arm, "jit_compile": jit_compile,
        "reference_sha256": source_sha, "inputs": inputs, "result": result,
        "cold_seconds": cold, "warm_seconds": timings, "traces": owner.experimental_get_tracing_count(),
        "before": before, "compiled": compiled, "warm": warm,
        "device_provenance": monitor.payload(), "role": "single_process_descriptive_cost_diagnostic"})
    assert owner.experimental_get_tracing_count() == 1
