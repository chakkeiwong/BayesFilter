"""Call-chain audit for the historical KR diagnostic and native TTSIRT route."""

import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRTransport
from bayesfilter.highdim.ttsirt_native_tf import transport_arguments, transport_program
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.highdim.test_zhao_cui_frozen_ttsirt_apf_compiler import _correlated_transport
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_kdm_auxiliary import _json
from tests.test_filter_repair_public_pullbacks import _pinned_transport

ROOT = Path(__file__).resolve().parents[1]


def _called_kr_transport_paths() -> list[str]:
    paths = []
    for path in (ROOT / "bayesfilter").rglob("*.py"):
        if path.name == "transport.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            terminal = function.attr if isinstance(function, ast.Attribute) else (
                function.id if isinstance(function, ast.Name) else ""
            )
            if terminal == "KRTransport":
                paths.append(str(path.relative_to(ROOT)))
    return paths


def _method_has_python_iteration(owner, name: str) -> bool:
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(owner, name))))
    return any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))


def test_legacy_kr_has_no_package_callers():
    assert _called_kr_transport_paths() == []
    assert KRTransport.__doc__ and "diagnostic" in KRTransport.__doc__.lower()


def test_active_fixed_ttsirt_public_methods_are_native_boundaries():
    methods = (
        "inverse_transport",
        "forward_transport",
        "forward_log_jacobian",
        "eval_pdf",
        "conditional_inverse_transport",
        "conditional_inverse_transport_suffix",
        "conditional_forward_transport_suffix",
        "conditional_forward_log_jacobian_suffix",
        "conditional_proposal_log_density",
        "conditional_proposal_log_density_suffix",
        "potential",
        "proposal_log_density",
        "log_normalizer",
    )
    public = {name for name, value in FixedTTSIRTTransport.__dict__.items()
              if not name.startswith('_') and callable(value)}
    assert public == set(methods) | {"manifest_payload", "marginalize", "batch_working_set_estimate"}
    for name in methods:
        assert not _method_has_python_iteration(FixedTTSIRTTransport, name)
        assert inspect.signature(getattr(FixedTTSIRTTransport, name)).parameters[
            "jit_compile"
        ].default is True


def _write(request, name, report):
    path = Path(request.config.getoption("xmlpath")).parent / name
    path.write_text(json.dumps(_json(report), indent=2, allow_nan=False) + "\n")


def _hlo_without_import_counter(hlo):
    # TensorFlow GraphToFunction gives imported zero-cotangent Const metadata
    # a changing suffix. Preserve instructions, constants, and real locations.
    return re.sub(r'(op_name="(?:[^"]*/)?zeros_like(?:_\d+)?/)_\d+'
                  r'(?=" source_file="dummy_file_name" source_line=10)', r'\1IMPORT', hlo)


def test_hlo_metadata_normalization_preserves_instructions():
    text = 'constant(0), metadata={op_name="zeros_like_1/_0" source_file="dummy_file_name" source_line=10}'
    assert _hlo_without_import_counter(text) == _hlo_without_import_counter(text.replace('/_0', '/_7'))
    assert _hlo_without_import_counter(text) != _hlo_without_import_counter(text.replace('constant(0)', 'constant(1)'))
    for changed in (text.replace('dummy_file_name', 'real.py'), text.replace('source_line=10', 'source_line=11')):
        assert _hlo_without_import_counter(changed) == changed


def _query(count):
    with tf.device("/CPU:0"):
        return tf.reshape(.6 * tf.sin(tf.cast(tf.range(2 * count), tf.float64) + .3), [2, count])


def _call(transport, operation, points, **kwargs):
    if operation == "log_normalizer":
        return transport.log_normalizer(**kwargs)
    if operation == "potential":
        return transport.potential(points, **kwargs)
    return transport.proposal_log_density(local_points=points, reference_points=points, **kwargs)


@pytest.mark.parametrize("jit_compile", [False, True])
def test_complete_log_wrappers_and_pullbacks(jit_compile, monkeypatch, request):
    transport = _correlated_transport()
    reference = _pinned_transport(transport)
    points = _query(3)
    cores = tuple(core.values for core in transport.density.sqrt_tt.cores)
    report = {"jit_compile": jit_compile, "original_source": "3582b4ac", "rows": []}
    for operation in ("potential", "proposal_log_density", "log_normalizer"):
        sources = cores if operation == "log_normalizer" else (points, *cores)
        with tf.GradientTape() as tape:
            tape.watch(sources)
            expected = _call(reference, operation, points)
            expected_loss = tf.reduce_sum(expected)
        expected_gradient = tape.gradient(expected_loss, sources)
        original_log = tf.math.log

        def compiled_log(*args, _original_log=original_log, **kwargs):
            assert tf.inside_function(), "public numerical logarithm escaped the compiled owner"
            return _original_log(*args, **kwargs)

        with monkeypatch.context() as scope:
            scope.setattr(tf.math, "log", compiled_log)
            with tf.GradientTape() as tape:
                tape.watch(sources)
                result = _call(transport, operation, points, jit_compile=jit_compile)
                loss = tf.reduce_sum(result)
            gradient = tape.gradient(loss, sources)
        assert all(value is not None for value in (*gradient, *expected_gradient))
        np.testing.assert_allclose(result, expected, atol=1e-10, rtol=1e-10)
        for actual, baseline in zip(gradient, expected_gradient, strict=True):
            np.testing.assert_allclose(actual, baseline, atol=1e-10, rtol=1e-10)
        replay = _call(transport, operation, points, jit_compile=jit_compile)
        np.testing.assert_array_equal(result, replay)
        count = 0 if operation == "log_normalizer" else points.shape[1]
        values = tf.zeros([transport.dimension, 0], tf.float64) if count == 0 else points
        condition = tf.zeros([0, count], tf.float64)
        arguments = (*transport_arguments(transport), condition, values)
        mode = "log_normalizer" if operation == "log_normalizer" else "log_density"
        owner = transport_program(transport, mode, count, jit_compile=jit_compile)
        assert owner is transport_program(transport, mode, count, jit_compile=jit_compile)
        graph = owner.get_concrete_function().graph.as_graph_def()
        all_nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
        assert not {"PyFunc", "EagerPyFunc", "PyFuncStateless"}.intersection(node.op for node in all_nodes)
        hlo = owner.experimental_get_compiler_ir(*arguments)(stage="hlo") if jit_compile else ""
        changed = points + .02
        np.testing.assert_allclose(_call(transport, operation, changed, jit_compile=jit_compile),
                                   _call(reference, operation, changed), atol=1e-10, rtol=1e-10)
        # All TT cores remain operands: same owner, changed coefficients.
        changed_cores = tuple(core + .01 for core in cores)
        changed_arguments = (changed_cores, *arguments[1:])
        changed_record, status = owner(*changed_arguments)
        changed_value = changed_record if mode == "log_normalizer" else changed_record[operation]
        assert int(status) == 0
        assert not np.array_equal(changed_value, result)
        repeated = owner(*changed_arguments)[0]
        np.testing.assert_array_equal(repeated if mode == "log_normalizer" else repeated[operation], changed_value)
        assert owner.experimental_get_tracing_count() == 1
        if jit_compile:
            changed_hlo = owner.experimental_get_compiler_ir(*changed_arguments)(stage="hlo")
            output = Path(request.config.getoption("xmlpath")).parent
            (output / f"mixed-kr-{operation}-before.hlo").write_text(hlo)
            (output / f"mixed-kr-{operation}-changed.hlo").write_text(changed_hlo)
            assert _hlo_without_import_counter(hlo) == _hlo_without_import_counter(changed_hlo)
        if operation != "log_normalizer":
            direction = tf.ones_like(points)
            eps = 1e-5
            finite_difference = (tf.reduce_sum(_call(reference, operation, points + eps * direction))
                                 - tf.reduce_sum(_call(reference, operation, points - eps * direction))) / (2 * eps)
            np.testing.assert_allclose(tf.reduce_sum(gradient[0]), finite_difference, atol=1e-8, rtol=2e-5)
        report["rows"].append({"operation": operation, "result": result, "expected": expected,
            "gradient": gradient, "expected_gradient": expected_gradient, "changed_cores": changed_value,
            "graph_bytes": graph.ByteSize(), "hlo_bytes": len(hlo),
            "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(), "traces": owner.experimental_get_tracing_count()})
    _write(request, f"mixed-kr-log-wrappers-{jit_compile}.json", report)


def test_log_wrappers_zero_density_and_invalid_status():
    transport = _correlated_transport()
    # A nonzero polynomial with a zero at x=0 has positive normalizer and a
    # legitimate zero density there when its defensive mass is zero.
    first, second = transport.density.sqrt_tt.cores
    first = replace(first, values=tf.tensor_scatter_nd_update(first.values, [[0, 0, 0]], [0.]))
    sqrt_tt = replace(transport.density.sqrt_tt, cores=(first, second))
    args = {**transport.density.__dict__, "sqrt_tt": sqrt_tt, "tau": tf.constant(0., tf.float64)}
    identity_args = {key: args[key] for key in (
        "sqrt_tt", "defensive_density", "tau", "normalizer_floor", "denominator_floor", "measure_convention")}
    args["branch_identity"] = transport.density.expected_branch_identity(**identity_args)
    transport = replace(transport, density=type(transport.density)(**args))
    reference = _pinned_transport(transport)
    points = tf.constant([[0., 0.], [.2, .4]], tf.float64)
    for jit_compile in (False, True):
        for operation in ("potential", "proposal_log_density"):
            expected = _call(reference, operation, points)
            assert bool(tf.reduce_all(tf.math.is_inf(expected)))
            np.testing.assert_array_equal(_call(transport, operation, points, jit_compile=jit_compile), expected)
            with pytest.raises(ValueError, match="NONFINITE_VALUE"):
                _call(transport, operation, tf.fill([2, 2], tf.constant(float("nan"), tf.float64)), jit_compile=jit_compile)
            outer = tf.function(lambda x, op=operation, jit=jit_compile: _call(transport, op, x, jit_compile=jit),
                                input_signature=[tf.TensorSpec([2, 2], tf.float64)],
                                jit_compile=jit_compile, autograph=False)
            assert bool(tf.reduce_all(tf.math.is_nan(outer(tf.fill([2, 2], tf.constant(float("nan"), tf.float64))))))
        owner = transport_program(transport, "log_normalizer", 0, jit_compile=jit_compile)
        arguments = (*transport_arguments(transport), tf.zeros([0, 0], tf.float64), tf.zeros([2, 0], tf.float64))
        _, code = owner(*arguments[:2], tf.constant(1e100, tf.float64), *arguments[3:])
        assert int(code) == 2


def _previous_transport(transport):
    source = subprocess.check_output(["git", "show", "5bbce48b5:bayesfilter/highdim/transport.py"], cwd=ROOT, text=True)
    module = ModuleType("_mixed_kr_previous_wrapper")
    sys.modules[module.__name__] = module
    exec(compile(source, "<previous-transport-wrapper>", "exec"), module.__dict__)  # noqa: S102
    return module.FixedTTSIRTTransport(transport.density, module.KRCDFConfig(**transport.cdf_config.__dict__)), hashlib.sha256(source.encode()).hexdigest()


@pytest.mark.parametrize("count", [3, 128])
@pytest.mark.parametrize("arm", ["previous", "graph", "xla"])
def test_public_log_cost(arm, count, request):
    transport = _correlated_transport()
    previous, source_sha = _previous_transport(transport)
    selected = previous if arm == "previous" else transport
    points = _query(count)
    kwargs = {} if arm == "previous" else {"jit_compile": arm == "xla"}

    def call():
        return _json(tuple(_call(selected, operation, points, **kwargs) for operation in (
            "potential", "proposal_log_density", "log_normalizer")))

    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    with GPUProcessMonitor(gpu) as monitor:
        before = memory_snapshot(gpu)
        start = time.perf_counter()
        result = call()
        cold = time.perf_counter() - start
        compiled = memory_snapshot(gpu)
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            replay = call()
            timings.append(time.perf_counter() - start)
            assert replay == result
        after = memory_snapshot(gpu)
    _write(request, "mixed-kr-log-cost.json", {"arm": arm, "count": count,
        "previous_wrapper_sha256": source_sha, "points": points, "result": result,
        "cold_seconds": cold, "warm_seconds": timings, "before": before,
        "compiled": compiled, "warm": after, "device_provenance": monitor.payload(),
        "role": "descriptive_complete_public_wrappers_partially_compiled_previous_baseline"})
