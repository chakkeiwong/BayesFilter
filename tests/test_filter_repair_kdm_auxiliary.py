"""Independent diagnostics and execution gates for the KDM auxiliary.

NumPy is used only to preserve and compare test evidence. No KDM result here
admits an LEDH algorithm, trained transport, or HMC route.
"""

import gc
import hashlib
import json
import os
import resource
import shutil
import subprocess
import threading
import time
import weakref
from dataclasses import replace
from pathlib import Path
from types import CellType, FunctionType, ModuleType
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_canonical_score_tf as score_module
from bayesfilter.highdim import ledh_younis_kdm_tf as kdm_module
from bayesfilter.highdim.ledh_canonical_reset_score_tf import (
    sinkhorn_contract_e_reset_triple_with_tangent,
)
from scripts.filter_repair_cost_provenance import GPUProcessMonitor


def _fixture_module():
    source = subprocess.check_output([
        "git", "show", "b052054f7:tests/highdim/test_ledh_younis_kdm_tf.py",
    ], cwd=Path(__file__).resolve().parents[1], text=True)
    module = ModuleType("_kdm_fixture_diagnostic")
    exec(compile(source, "<frozen-kdm-fixture>", "exec"), module.__dict__)  # noqa: S102
    return module


def _original():
    path = "bayesfilter/highdim/ledh_younis_kdm_tf.py"
    source = subprocess.check_output([
        "git", "show", f"ab431169d:{path}",
    ], cwd=Path(__file__).resolve().parents[1], text=True)
    module = ModuleType("_kdm_auxiliary_frozen_reference")
    exec(compile(source, "<frozen-kdm-reference>", "exec"), module.__dict__)  # noqa: S102
    reset_source = subprocess.check_output([
        "git", "show", "b052054f7:bayesfilter/highdim/ledh_unified_reset_tf.py",
    ], cwd=Path(__file__).resolve().parents[1], text=True)
    reset = ModuleType("_kdm_reset_frozen_reference")
    exec(compile(reset_source, "<frozen-kdm-reset>", "exec"), reset.__dict__)  # noqa: S102
    original_call = module.canonical_linear_gaussian_kdm_auxiliary

    def reference(*args, **kwargs):
        with patch("bayesfilter.highdim.ledh_canonical_reset_score_tf."
                   "batched_sinkhorn_contract_e_reset_triple_with_tangent",
                   reset.batched_sinkhorn_contract_e_reset_triple_with_tangent):
            return original_call(*args, **kwargs)

    module.canonical_linear_gaussian_kdm_auxiliary = reference
    module.reset_source_sha256 = hashlib.sha256(reset_source.encode()).hexdigest()
    module.reset_reference = reset.batched_sinkhorn_contract_e_reset_triple_with_tangent
    return module, hashlib.sha256(source.encode()).hexdigest()


def _fixture(reset_steps=8):
    model = _fixture_module()._trace_model()
    dtype = tf.float64
    rng = np.random.default_rng(131)
    inputs = (
        tf.constant([0.6], dtype),
        tf.constant(rng.normal(size=(4, 2)), dtype),
        tf.eye(2, batch_shape=[4], dtype=dtype),
        tf.constant(rng.normal(size=(2, 4, 2)), dtype),
        tf.constant(rng.normal(size=(2, 2)), dtype),
        tf.eye(2, dtype=dtype), tf.zeros([2, 2], dtype),
        0.25 * tf.eye(2, batch_shape=[2, 4], dtype=dtype),
        tf.zeros([2, 4, 2, 2], dtype),
    )
    options = {
        "flow_substeps": 2, "reset_policy": "contract_e",
        "reset_design": tf.constant([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype),
        "reset_epsilon": 2., "reset_sinkhorn_steps": reset_steps,
        "reset_balance_steps": reset_steps, "correction_steps": 4,
        "correction_strength": 0.2, "correction_lm_damping": 1.e-2,
        "correction_lm_scale_floor": 1.e-4, "correction_trust_radius": 0.5,
        "pairwise_steps": 4, "pairwise_strength": 0.02, "pairwise_rms_cap": 2.,
        "coordinate_cap": 0.98, "coordinate_cap_power": 8,
    }
    return model, inputs, options


def _json(value):
    if tf.is_tensor(value):
        return _json(value.numpy())
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def test_baseline_auxiliary_rejection_localization(monkeypatch, request):
    """Preserve the unchanged failing seed-131 test and each rejecting layer."""
    fixture = _fixture_module()
    captured = {"kernels": []}
    original_score = score_module.canonical_value_and_analytical_score
    frozen, _ = _original()
    original_factory = frozen.make_linear_gaussian_kdm_normalizer_kernel
    original_auxiliary = frozen.canonical_linear_gaussian_kdm_auxiliary

    def capture_score(*args, **kwargs):
        result = original_score(*args, **kwargs)
        captured["canonical"] = result
        captured["canonical_options"] = kwargs
        captured["canonical_inputs"] = args[1:]
        return result

    def capture_factory(**kwargs):
        kernel = original_factory(**kwargs)

        def capture_kernel(*args):
            result = kernel(*args)
            captured["kernels"].append({
                "bandwidth_is_zero": kwargs["bandwidth_is_zero"],
                "inputs": args, "result": result,
            })
            return result

        return capture_kernel

    def capture_auxiliary(*args, **kwargs):
        result = original_auxiliary(*args, **kwargs)
        captured["auxiliary"] = result
        return result

    monkeypatch.setattr(score_module, "canonical_value_and_analytical_score", capture_score)
    monkeypatch.setattr(frozen, "make_linear_gaussian_kdm_normalizer_kernel", capture_factory)
    monkeypatch.setattr(fixture, "canonical_linear_gaussian_kdm_auxiliary", capture_auxiliary)
    try:
        fixture.test_auxiliary_api_uses_canonical_trace_and_has_no_feedback()
    except AssertionError:
        captured["legacy_assertion_failed"] = True
    else:
        captured["legacy_assertion_failed"] = False

    resets = []
    options = captured["canonical_options"]
    for record in captured["canonical"][2]:
        reset = sinkhorn_contract_e_reset_triple_with_tangent(
            record["children"], record["d_children"], record["post_covariances"],
            record["d_post_covariances"], record["posterior_weights"],
            record["d_posterior_weights"], options["reset_design"],
            epsilon=options["reset_epsilon"], sinkhorn_steps=options["reset_sinkhorn_steps"],
            balance_steps=options["reset_balance_steps"], ridge=options.get("reset_ridge", 1.e-5),
        )
        transport = record["reset_transport"]
        resets.append({
            "recomputed_outputs": reset,
            "row_error": tf.reduce_max(tf.abs(tf.reduce_sum(transport, axis=1) - 1.0)),
            "column_tv_error": 0.5 * tf.reduce_sum(tf.abs(
                tf.reduce_mean(transport, axis=0) - record["posterior_weights"])),
        })
    captured["resets"] = resets
    captured["role"] = "baseline_rejection_localization_only"
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-baseline-rejection.json"
    output.write_text(json.dumps(_json(captured), indent=2, allow_nan=False) + "\n")
    assert captured["legacy_assertion_failed"]
    assert not bool(captured["auxiliary"]["valid"].numpy())
    assert len(captured["kernels"]) == 4
    assert len(resets) == 2


def _assert_record_equal(actual, expected, *, atol=2.e-10, rtol=2.e-10):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_record_equal(actual[key], expected[key], atol=atol, rtol=rtol)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_record_equal(left, right, atol=atol, rtol=rtol)
    elif tf.is_tensor(expected):
        if expected.dtype.is_floating:
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=atol, rtol=rtol)
        else:
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    else:
        assert actual == expected


@pytest.mark.parametrize("reset_steps", [2, 8])
def test_original_auxiliary_valid_and_rejected_controls(reset_steps, request):
    original, source_sha = _original()
    model, inputs, options = _fixture(reset_steps)
    reference = original.canonical_linear_gaussian_kdm_auxiliary(
        model, *inputs, canonical_options=options, jit_compile=False)
    value, score, trace = score_module.canonical_value_and_analytical_score(
        model, *inputs[:5], with_score=True, return_trace=True, **options)
    artifact = {"reference_source_commit": "ab431169d", "reference_sha256": source_sha,
                "reset_reference_sha256": original.reset_source_sha256,
                "reset_steps": reset_steps, "inputs": inputs, "options": options,
                "reference": reference, "canonical": (value, score, trace)}
    if reset_steps == 8:
        epsilon = 2.e-5
        plus = original.canonical_linear_gaussian_kdm_auxiliary(
            model, inputs[0] + epsilon, *inputs[1:], canonical_options=options, jit_compile=False)
        minus = original.canonical_linear_gaussian_kdm_auxiliary(
            model, inputs[0] - epsilon, *inputs[1:], canonical_options=options, jit_compile=False)
        finite_difference = (plus["kdm_auxiliary_value"] - minus["kdm_auxiliary_value"]) / (2 * epsilon)
        artifact["finite_difference"] = finite_difference
    output = Path(request.config.getoption("xmlpath")).parent / f"kdm-reference-{reset_steps}.json"
    output.write_text(json.dumps(_json(artifact), indent=2, allow_nan=False) + "\n")
    assert bool(reference["valid"].numpy()) == (reset_steps == 8)
    assert bool(tf.reduce_all([r["program_valid"] for r in trace]).numpy()) == (reset_steps == 8)
    if reset_steps == 8:
        np.testing.assert_allclose(reference["kdm_auxiliary_score"].numpy()[0],
                                   finite_difference.numpy(), rtol=4.e-4, atol=4.e-5)
    else:
        assert np.isneginf(value.numpy())
        np.testing.assert_array_equal(score.numpy(), [0.])


def _owner(model, inputs, options, jit_compile):
    return kdm_module.make_canonical_linear_gaussian_kdm_auxiliary_program(
        model, theta_shape=tuple(inputs[0].shape), particle_count=4, state_dimension=2,
        observation_dimension=2, horizon=2, canonical_options=options,
        dtype=tf.float64, jit_compile=jit_compile)


def _public_record(numerical):
    result = dict(numerical)
    result.pop("model_valid")
    steps = result.pop("steps")
    return {"route_id": kdm_module.AUXILIARY_ROUTE_ID,
            "route_role": kdm_module.AUXILIARY_ROLE,
            "canonical_target_label": kdm_module.ATOM_FINITE_TARGET,
            "auxiliary_target_label": kdm_module.KDM_FINITE_TARGET,
            **result,
            "steps": tuple({"time_index": i, **{k: v[i] for k, v in steps.items()}}
                           for i in range(2))}


@pytest.mark.parametrize("jit_compile", [False, True], ids=["graph", "xla"])
@pytest.mark.parametrize("reset_steps", [2, 8])
def test_native_auxiliary_complete_records(reset_steps, jit_compile, request):
    original, source_sha = _original()
    model, inputs, options = _fixture(reset_steps)
    reference = original.canonical_linear_gaussian_kdm_auxiliary(
        model, *inputs, canonical_options=options, jit_compile=False)
    owner = _owner(model, inputs, options, jit_compile)
    first = owner(*inputs)
    replay = owner(*inputs)
    public = kdm_module.canonical_linear_gaussian_kdm_auxiliary(
        model, *inputs, canonical_options=options, jit_compile=jit_compile)
    changed = (*inputs[:4], inputs[4] + tf.constant(0.01, tf.float64), *inputs[5:])
    changed_result = owner(*changed)
    changed_reference = original.canonical_linear_gaussian_kdm_auxiliary(
        model, *changed, canonical_options=options, jit_compile=False)
    definition = owner.get_concrete_function().graph.as_graph_def()
    nodes = [*definition.node, *(node for function in definition.library.function
                                for node in function.node_def)]
    artifact = {"reference_sha256": source_sha, "jit_compile": jit_compile,
                "reset_reference_sha256": original.reset_source_sha256,
                "reset_steps": reset_steps, "reference": reference, "first": first,
                "replay": replay, "public": public, "changed": changed_result,
                "changed_reference": changed_reference,
                "traces": owner.experimental_get_tracing_count(),
                "ops": sorted({node.op for node in nodes})}
    output = Path(request.config.getoption("xmlpath")).parent
    stem = f"kdm-native-{reset_steps}-{jit_compile}"
    (output / f"{stem}.pb").write_bytes(definition.SerializeToString())
    if jit_compile:
        hlo = owner.experimental_get_compiler_ir(*inputs)(stage="hlo")
        (output / f"{stem}.hlo").write_text(hlo)
        artifact["hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
    (output / f"{stem}.json").write_text(json.dumps(_json(artifact), indent=2, allow_nan=False) + "\n")
    _assert_record_equal(_public_record(first), reference)
    _assert_record_equal(public, reference)
    _assert_record_equal(_public_record(changed_result), changed_reference)
    _assert_record_equal(replay, first, atol=0., rtol=0.)
    assert bool(first["valid"].numpy()) == (reset_steps == 8)
    assert owner.experimental_get_tracing_count() == 1
    assert not {"PyFunc", "PyFuncStateless", "EagerPyFunc"}.intersection(artifact["ops"])
    assert {"While", "StatelessWhile"}.intersection(artifact["ops"])
    if reset_steps == 8:
        epsilon = 2.e-5
        plus = owner(inputs[0] + epsilon, *inputs[1:])
        minus = owner(inputs[0] - epsilon, *inputs[1:])
        derivative = (plus["kdm_auxiliary_value"] - minus["kdm_auxiliary_value"]) / (2 * epsilon)
        np.testing.assert_allclose(first["kdm_auxiliary_score"].numpy()[0],
                                  derivative.numpy(), rtol=4.e-4, atol=4.e-5)
    invalid_inputs = (*inputs[:5], inputs[5] * 2., *inputs[6:])
    invalid = owner(*invalid_inputs)
    assert not bool(invalid["model_valid"].numpy())
    assert not bool(invalid["valid"].numpy())
    assert bool(tf.math.is_nan(invalid["kdm_auxiliary_value"]).numpy())


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("comparator", ["same_mode", "eager"])
def test_native_shared_reset_matches_original(dtype, comparator, request):
    from bayesfilter.highdim.ledh_unified_reset_tf import (
        batched_sinkhorn_contract_e_reset_triple_with_tangent,
    )
    original, _ = _original()
    # Frozen actual KDM operands, with two batch rows and two tangent directions.
    data = json.loads((Path(__file__).parent / "fixtures/filter_repair_kdm_reset_seed131.json").read_text())
    record = data["canonical"][2][0]
    children = tf.constant([record["children"], record["children"]], dtype)
    d_children = tf.constant([[record["d_children"], record["d_children"]]] * 2, dtype)
    covs = tf.constant([record["post_covariances"], record["post_covariances"]], dtype)
    d_covs = tf.constant([[record["d_post_covariances"], record["d_post_covariances"]]] * 2, dtype)
    weights = tf.constant([record["posterior_weights"]] * 2, dtype)
    d_weights = tf.constant([[record["d_posterior_weights"]] * 2] * 2, dtype)
    design = tf.constant(data["canonical_options"]["reset_design"], dtype)
    values = (children, d_children, covs, d_covs, weights, d_weights, design)
    options = {"epsilon": 2., "sinkhorn_steps": 8, "balance_steps": 8, "ridge": 1.e-5}
    results = {}
    eager = original.reset_reference(*values, **options)
    for jit in (False, True):
        native = tf.function(
            lambda *args: batched_sinkhorn_contract_e_reset_triple_with_tangent(*args, **options),
            input_signature=[tf.TensorSpec(v.shape, v.dtype) for v in values],
            autograph=False, jit_compile=jit)
        reference = tf.function(
            lambda *args: original.reset_reference(*args, **options),
            input_signature=[tf.TensorSpec(v.shape, v.dtype) for v in values],
            autograph=False, jit_compile=jit)
        expected = reference(*values)
        actual = native(*values)
        results[str(jit)] = {"actual": actual, "reference_same_mode": expected,
                             "reference_eager": eager, "replay": native(*values),
                             "trace_count": native.experimental_get_tracing_count()}
    output = Path(request.config.getoption("xmlpath")).parent / f"kdm-reset-{dtype.name}.json"
    output.write_text(json.dumps(_json(results), indent=2, allow_nan=False) + "\n")
    tolerance = 2.e-10 if dtype == tf.float64 else 1.e-6
    for result in results.values():
        _assert_record_equal(result["actual"], result[f"reference_{comparator}"], atol=tolerance, rtol=tolerance)
        _assert_record_equal(result["replay"], result["actual"], atol=0., rtol=0.)
        assert result["trace_count"] == 1


def test_native_auxiliary_public_ownership_and_errors(request):
    original, _ = _original()
    model, inputs, options = _fixture()
    scale = [1.]
    changed_model = replace(model,
        transition_mean_fn=lambda theta, points: scale[0] * model.transition_mean_fn(theta, points),
        transition_mean_tangent_fn=lambda theta, points, tangent: scale[0] * model.transition_mean_tangent_fn(theta, points, tangent))
    first = kdm_module.canonical_linear_gaussian_kdm_auxiliary(
        changed_model, *inputs, canonical_options=options, jit_compile=False)
    scale[0] = 1.01
    changed = kdm_module.canonical_linear_gaussian_kdm_auxiliary(
        changed_model, *inputs, canonical_options=options, jit_compile=False)
    reference = original.canonical_linear_gaussian_kdm_auxiliary(
        changed_model, *inputs, canonical_options=options, jit_compile=False)
    _assert_record_equal(changed, reference)
    assert float(first["kdm_auxiliary_value"].numpy()) != float(changed["kdm_auxiliary_value"].numpy())
    with pytest.raises(ValueError, match="observation callbacks"):
        kdm_module.canonical_linear_gaussian_kdm_auxiliary(
            model, *inputs[:5], inputs[5] * 2., *inputs[6:],
            canonical_options=options, jit_compile=False)
    for forbidden in ("_stacked_trace", "observation_factor_override", "post_reset_transform"):
        with pytest.raises(TypeError):
            _owner(model, inputs, {**options, forbidden: True}, False)
    with pytest.raises(ValueError, match="with_score"):
        _owner(model, inputs, {**options, "with_score": True}, False)
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-ownership.json"
    output.write_text(json.dumps(_json({"first": first, "changed": changed, "reference": reference}),
                                indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("arm", ["original", "graph", "xla"])
def test_auxiliary_matched_cost(arm, request):
    """Fresh-process FP64 public costs and separately measured owner reuse."""
    original, source_sha = _original()
    model, inputs, options = _fixture()
    public_call = (original.canonical_linear_gaussian_kdm_auxiliary if arm == "original"
                   else kdm_module.canonical_linear_gaussian_kdm_auxiliary)
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    device = "GPU:0" if gpu else "CPU:0"

    def allocator():
        try:
            return tf.config.experimental.get_memory_info(device)
        except ValueError:
            if gpu:
                raise
            return None

    def rss():
        lines = Path("/proc/self/status").read_text().splitlines()
        return next(int(line.split()[1]) * 1024 for line in lines if line.startswith("VmRSS:"))

    def memory():
        return {"rss_bytes": rss(), "allocator": allocator(),
                "process_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}

    def sync(record):
        for value in tf.nest.flatten(record):
            if tf.is_tensor(value):
                value.numpy()

    try:
        tf.config.experimental.reset_memory_stats(device)
    except ValueError:
        if gpu:
            raise
    rss_samples, public_times, public_records, retained_times = [], [], [], []
    stop = threading.Event()

    def sample():
        while not stop.wait(.01):
            rss_samples.append(rss())

    sampler = threading.Thread(target=sample, daemon=True)
    before = memory()
    with GPUProcessMonitor(gpu) as monitor:
        sampler.start()
        try:
            for _ in range(3):
                begin = time.perf_counter()
                result = public_call(model, *inputs, canonical_options=options,
                                     jit_compile=arm != "graph")
                sync(result)
                public_times.append(time.perf_counter() - begin)
                public_records.append(result)
            after_public = memory()
            if arm != "original":
                begin = time.perf_counter()
                owner = _owner(model, inputs, options, arm == "xla")
                owned_first = owner(*inputs)
                sync(owned_first)
                owner_cold = time.perf_counter() - begin
                for _ in range(5):
                    begin = time.perf_counter()
                    owned_replay = owner(*inputs)
                    sync(owned_replay)
                    retained_times.append(time.perf_counter() - begin)
                traces = owner.experimental_get_tracing_count()
            else:
                owner_cold, owned_first, owned_replay, traces = None, None, None, None
            after_owner = memory()
        finally:
            stop.set()
            sampler.join(timeout=5)
    assert not sampler.is_alive()
    artifact = {"schema": "filter_repair_kdm_cost.v1", "role": "descriptive_fp64_fixture_only",
        "arm": arm, "reference_source_sha256": source_sha,
        "reference_reset_sha256": original.reset_source_sha256, "seed": 131,
        "reset_sinkhorn_steps": 8, "reset_balance_steps": 8, "shape": [2, 4, 2],
        "jit_compile": arm != "graph", "enclosing_xla": arm == "xla",
        "original_execution": "eager_auxiliary_with_xla_mixture_components",
        "public_seconds": public_times, "retained_owner_cold_seconds": owner_cold,
        "retained_owner_warm_seconds": retained_times, "retained_owner_traces": traces,
        "memory_before": before, "memory_after_public": after_public,
        "memory_after_owner": after_owner,
        "sampled_peak_rss_bytes": max([before["rss_bytes"], *rss_samples, after_owner["rss_bytes"]]),
        "cost_device_observations": monitor.payload(),
        "public_results": public_records, "retained_result": owned_first,
        "retained_replay": owned_replay, "value_device": result["canonical_value"].device}
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-cost.json"
    output.write_text(json.dumps(_json(artifact), indent=2, allow_nan=False) + "\n")
    # Comparisons after timing cannot contaminate the recorded cost/memory phase.
    expected = original.canonical_linear_gaussian_kdm_auxiliary(
        model, *inputs, canonical_options=options, jit_compile=False)
    assert bool(expected["valid"].numpy())
    for record in public_records:
        _assert_record_equal(record, expected)
    if arm != "original":
        _assert_record_equal(_public_record(owned_first), expected)
        _assert_record_equal(owned_replay, owned_first, atol=0., rtol=0.)
        assert traces == 1


@pytest.mark.parametrize("jit_compile", [False, True], ids=["graph", "xla"])
def test_single_owner_memory_attribution(jit_compile, request):
    """One retained owner in a fresh process; no prior public-wrapper calls."""
    from tensorflow.python.eager import context

    from tests.test_filter_repair_gap_diagnostics import memory_snapshot

    model, inputs, options = _fixture()
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"

    def snapshot():
        return {**memory_snapshot(gpu), "registered_functions": len(context.context().list_function_names())}

    def sync(record):
        for value in tf.nest.flatten(record):
            value.numpy()

    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    snapshots = {"before": snapshot()}
    owner = _owner(model, inputs, options, jit_compile)
    reference = weakref.ref(owner)
    begin = time.perf_counter()
    first = owner(*inputs)
    sync(first)
    cold = time.perf_counter() - begin
    graph_reference = weakref.ref(owner.get_concrete_function().graph)
    snapshots["cold"] = snapshot()
    timings = []
    for index in range(20):
        begin = time.perf_counter()
        current = owner(*inputs)
        sync(current)
        timings.append(time.perf_counter() - begin)
        if index in (0, 4, 19):
            snapshots[f"warm_{index + 1}"] = snapshot()
        _assert_record_equal(current, first, atol=0., rtol=0.)
    trace_count = owner.experimental_get_tracing_count()
    del owner
    collected = gc.collect()
    snapshots["released"] = snapshot()
    artifact = {"jit_compile": jit_compile, "scope": "one_owner_20_replays_fresh_process",
        "cold_seconds": cold, "warm_seconds": timings, "trace_count": trace_count,
        "snapshots": snapshots, "collected_python_objects": collected,
        "owner_released": reference() is None, "graph_released": graph_reference() is None,
        "result": first, "value_device": first["canonical_value"].device,
        "limitation": "Function and graph release do not imply native executable/allocator eviction."}
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-single-owner-memory.json"
    output.write_text(json.dumps(_json(artifact), indent=2, allow_nan=False) + "\n")
    assert trace_count == 1
    assert bool(first["valid"].numpy())
    assert reference() is None


def test_auxiliary_public_enclosing_xla():
    model, inputs, options = _fixture()

    @tf.function(input_signature=[tf.TensorSpec(v.shape, v.dtype) for v in inputs],
                 jit_compile=True, autograph=False)
    def outer(*values):
        result = kdm_module.canonical_linear_gaussian_kdm_auxiliary(
            model, *values, canonical_options=options)
        return result["canonical_value"], result["kdm_auxiliary_value"], result["valid"]

    first = outer(*inputs)
    assert bool(first[2].numpy())
    invalid = outer(*inputs[:5], inputs[5] * 2., *inputs[6:])
    assert not bool(invalid[2].numpy())
    assert bool(tf.math.is_nan(invalid[1]).numpy())
    assert outer.experimental_get_tracing_count() == 1


def test_auxiliary_empty_horizon_preserves_original():
    original, _ = _original()
    model, inputs, options = _fixture()
    values = (*inputs[:3], inputs[3][:0], inputs[4][:0],
              *inputs[5:7], inputs[7][:0], inputs[8][:0])
    reference = original.canonical_linear_gaussian_kdm_auxiliary(
        model, *values, canonical_options=options, jit_compile=False)
    actual = kdm_module.canonical_linear_gaussian_kdm_auxiliary(
        model, *values, canonical_options=options)
    _assert_record_equal(actual, reference, atol=0., rtol=0.)
    assert actual["steps"] == ()
    assert bool(actual["valid"].numpy())


def test_auxiliary_graph_retention_attribution(request):
    """Trace-only explanatory inspection of TensorFlow's gradient registry."""
    from tensorflow.python.framework import ops

    from tests.test_filter_repair_block_graph_diagnostic import _closure_paths

    before = set(ops._gradient_registry.list())
    model, inputs, options = _fixture()
    owner = _owner(model, inputs, options, True)
    reference = weakref.ref(owner.get_concrete_function().graph)
    del owner
    collection_passes = []
    for _ in range(3):
        collected = gc.collect()
        collection_passes.append({"collected": collected, "graph_released": reference() is None})
    target = reference()
    paths = {}
    if target is not None:
        for name in sorted(set(ops._gradient_registry.list()) - before):
            result = _closure_paths(ops._gradient_registry.lookup(name), target)
            if result["paths"] or result["truncated"]:
                paths[name] = result
    artifact = {"role": "trace_only_registry_retention_diagnostic",
                "graph_released": target is None, "registry_paths": paths,
                "collection_passes": collection_passes,
                "registry_new_entries": len(set(ops._gradient_registry.list()) - before)}
    if target is not None:
        def describe(value):
            if isinstance(value, dict):
                return {"type": "dict", "keys": [str(k)[:100] for k in value][:16]}
            if isinstance(value, FunctionType):
                return {"type": "function", "name": value.__qualname__,
                        "source": value.__code__.co_filename,
                        "line": value.__code__.co_firstlineno}
            return {"type": type(value).__name__, "name": getattr(value, "name", None)}

        owners = []
        counts = {}
        for value in gc.get_referrers(target):
            kind = type(value).__name__
            counts[kind] = counts.get(kind, 0) + 1
            if isinstance(value, (tf.Tensor, tf.Operation)):
                continue
            item = describe(value)
            if isinstance(value, (dict, CellType, tuple)):
                parents = []
                for parent in gc.get_referrers(value):
                    if parent is owners or parent is item or isinstance(parent, list):
                        continue
                    parent_record = describe(parent)
                    if isinstance(parent, tuple):
                        parent_record["functions"] = [describe(obj) for obj in gc.get_referrers(parent)
                                                       if isinstance(obj, FunctionType)]
                    parents.append(parent_record)
                item["parents"] = parents[:24]
            owners.append(item)
        artifact["direct_referrer_counts"] = counts
        artifact["direct_owners"] = owners
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-graph-retention.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    assert not any(result["truncated"] for result in paths.values())


def test_saved_kdm_cost_evidence(request):
    from scripts.analyze_filter_repair_kdm_cost import analyze

    raw = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    result = analyze(raw)
    saved = json.loads((raw / "kdm-cost-analysis-03946-r1.json").read_text())
    assert result == saved
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-cost-analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.parametrize("corruption", ["value", "validity", "jit", "memory", "reference", "source"])
def test_kdm_cost_analysis_rejects_corruption(corruption, tmp_path):
    from scripts.analyze_filter_repair_kdm_cost import analyze

    raw = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    for number in range(3929, 3947):
        source = raw / f"run-{number:05d}"
        output = tmp_path / source.name
        output.mkdir()
        for name in ("run.json", "kdm-cost.json", "process.log", "junit.xml"):
            shutil.copyfile(source / name, output / name)
    path = tmp_path / "run-03937/kdm-cost.json"
    value = json.loads(path.read_text())
    if corruption == "value":
        value["public_results"][0]["kdm_auxiliary_score"][0] += 0.01
    elif corruption == "validity":
        value["public_results"][0]["valid"] = False
    elif corruption == "jit":
        value["enclosing_xla"] = False
    elif corruption == "memory":
        value["sampled_peak_rss_bytes"] = 0
    elif corruption == "reference":
        value["reference_reset_sha256"] = "0" * 64
    else:
        path = path.with_name("run.json")
        value = json.loads(path.read_text())
        value["source_sha256"]["bayesfilter/highdim/ledh_younis_kdm_tf.py"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        analyze(tmp_path)
