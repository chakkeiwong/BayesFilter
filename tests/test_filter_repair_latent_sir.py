"""Frozen-source and independent diagnostics for latent SIR execution.

Autodiff below checks a simulator derivative only; it is not an LEDH score.
"""

import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import sir_latent_preclip_tf as candidate
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.highdim.test_sir_latent_preclip_tf import _small_model, _source_style_path
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_genut_transitive import _graph_evidence
from tests.test_filter_repair_kdm_auxiliary import _assert_record_equal, _json

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "013978671"
D = tf.float64


def _frozen(name):
    path = f"bayesfilter/highdim/{name}.py"
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, text=True)
    module = ModuleType("_latent_sir_original_" + name)
    sys.modules[module.__name__] = module
    exec(compile(source, "<latent-sir-frozen-reference>", "exec"), module.__dict__)  # noqa: S102
    return module, hashlib.sha256(source.encode()).hexdigest()


def _original_model(model):
    models, models_sha = _frozen("models")
    original, original_sha = _frozen("sir_latent_preclip_tf")
    original.ParameterizedZhaoCuiSIRSSM = models.ParameterizedZhaoCuiSIRSSM
    base = model.physical_model.base_model
    old_base = models.SpatialSIRSSM(**{field.name: getattr(base, field.name) for field in fields(base)})
    return original.LatentPreclipSIRSSM(models.ParameterizedZhaoCuiSIRSSM(old_base)), {
        "models": models_sha, "latent": original_sha}


def _write(request, name, report):
    path = Path(request.config.getoption("xmlpath")).parent / name
    path.write_text(json.dumps(_json(report), indent=2, allow_nan=False) + "\n")


def _fixture(compartments, horizon):
    model = (_small_model(1) if compartments == 1 else
             candidate.latent_preclip_two_node_spatial_sir_model() if compartments == 2 else
             candidate.latent_preclip_zhao_cui_sir_austria_model())
    n, m = model.state_dim(), model.observation_dim()
    with tf.device("/CPU:0"):
        initial = tf.tensor_scatter_nd_update(tf.zeros([n], D), [[0]],
                                              [-model.physical_model.base_model.initial_mean[0] - 2.])
        transition = tf.reshape(.1 * tf.sin(tf.cast(tf.range(horizon * n), D)), [horizon, n])
        transition = tf.concat([tf.fill([horizon, 1], tf.constant(-10., D)), transition[:, 1:]], axis=1)
        observation = tf.reshape(.1 * tf.cos(tf.cast(tf.range((horizon + 1) * m), D)), [horizon + 1, m])
        return model, (tf.constant([.01, -.02, .03], D), initial, transition, observation)


@pytest.mark.parametrize("compartments", [1, 2, 9])
@pytest.mark.parametrize("horizon", [0, 1, 3])
def test_complete_fixed_noise_paths(compartments, horizon, request):
    model, inputs = _fixture(compartments, horizon)
    original, source_sha = _original_model(model)
    expected = original.simulate_from_standard_normals(*inputs)
    physical, observations = _source_style_path(original, *inputs)
    _assert_record_equal(expected["physical_path"], physical, atol=5.e-12, rtol=5.e-12)
    _assert_record_equal(expected["observations"], observations, atol=5.e-12, rtol=5.e-12)
    report = {"reference_sha256": source_sha, "inputs": inputs, "expected": expected, "arms": []}
    try:
        for jit_compile in (False, True):
            owner = candidate.latent_preclip_simulation_program(model, horizon, jit_compile=jit_compile)
            assert owner is candidate.latent_preclip_simulation_program(model, horizon, jit_compile=jit_compile)
            actual, valid = owner(*inputs)
            row = {"jit_compile": jit_compile, "actual": actual, "valid": valid}
            report["arms"].append(row)
            assert bool(valid)
            _assert_record_equal(actual, expected, atol=5.e-12, rtol=5.e-12)
            _assert_record_equal(actual, owner(*inputs)[0], atol=0., rtol=0.)
            public = model.simulate_from_standard_normals(*inputs, jit_compile=jit_compile)
            _assert_record_equal(actual, public, atol=0., rtol=0.)
            assert float(actual["physical_path"][0, 0]) < 0.
            if horizon:
                assert bool(tf.reduce_all(actual["physical_path"][1:, 0] == 0.))
            changed = (inputs[0] + .005, inputs[1] + .01, inputs[2] + .02, inputs[3] - .03)
            changed_result, changed_valid = owner(*changed)
            changed_expected = original.simulate_from_standard_normals(*changed)
            row.update(changed_inputs=changed, changed=changed_result, changed_expected=changed_expected)
            assert bool(changed_valid)
            _assert_record_equal(changed_result, changed_expected, atol=5.e-12, rtol=5.e-12)
            _assert_record_equal(changed_result, owner(*changed)[0], atol=0., rtol=0.)
            row["execution"] = _graph_evidence(owner, inputs, jit_compile)
            assert row["execution"]["traces"] == 1
            if horizon:
                ops = row["execution"]["operations"]
                assert ops.get("While", 0) + ops.get("StatelessWhile", 0) > 0
    finally:
        _write(request, f"latent-sir-j{compartments}-t{horizon}.json", report)


def test_public_validation_and_original_xla_boundary(request):
    model, inputs = _fixture(1, 1)
    original, source_sha = _original_model(model)
    assert inspect.signature(model.simulate_from_standard_normals).parameters["jit_compile"].default is True
    malformed = (inputs[0][:2], *inputs[1:])
    for selected in (original, model):
        with pytest.raises(ValueError):
            selected.simulate_from_standard_normals(*malformed)
        with pytest.raises(ValueError):
            selected.simulate_from_standard_normals(*inputs[:2], inputs[2][:, :1], inputs[3])
        with pytest.raises(ValueError):
            selected.simulate_from_standard_normals(*inputs[:3], inputs[3][:1])
        for theta in ([float("nan"), 0., 0.], [1000., 0., 0.], [-1000., 0., 0.],
                      [0., -1000., 0.], [0., 0., 1000.]):
            with pytest.raises(ValueError):
                selected.simulate_from_standard_normals(tf.constant(theta, D), *inputs[1:])
    row_theta = (inputs[0][None, :], *inputs[1:])
    _assert_record_equal(model.simulate_from_standard_normals(*row_theta),
                         original.simulate_from_standard_normals(*row_theta), atol=5.e-12, rtol=5.e-12)
    original_xla = tf.function(original.simulate_from_standard_normals,
        input_signature=[tf.TensorSpec(x.shape, D) for x in inputs], jit_compile=True, autograph=False)
    with pytest.raises(AttributeError, match="numpy") as error:
        original_xla(*inputs)
    _write(request, "latent-sir-validation.json", {"reference_sha256": source_sha,
        "original_xla_error": str(error.value), "invalid_parameter_rejections": 5})


def test_simulator_pullback_diagnostic(request):
    model, inputs = _fixture(1, 3)
    original, source_sha = _original_model(model)

    def value(call, theta):
        result = call(theta, *inputs[1:])
        return tf.reduce_sum(result["observations"]) + .2 * tf.reduce_sum(result["latent_path"])

    with tf.GradientTape() as tape:
        tape.watch(inputs[0])
        scalar = value(model.simulate_from_standard_normals, inputs[0])
    actual = tape.gradient(scalar, inputs[0])
    with tf.GradientTape() as tape:
        tape.watch(inputs[0])
        expected_scalar = value(original.simulate_from_standard_normals, inputs[0])
    expected = tape.gradient(expected_scalar, inputs[0])
    epsilon = 1.e-5
    finite_difference = tf.stack([
        (value(original.simulate_from_standard_normals, inputs[0] + epsilon * direction)
         - value(original.simulate_from_standard_normals, inputs[0] - epsilon * direction)) / (2 * epsilon)
        for direction in tf.eye(3, dtype=D)])
    _write(request, "latent-sir-pullback-diagnostic.json", {"reference_sha256": source_sha,
        "scalar": scalar, "expected_scalar": expected_scalar, "actual": actual,
        "expected": expected, "finite_difference": finite_difference,
        "claim_bearing_analytical_score": False})
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=5.e-10, atol=5.e-12)
    np.testing.assert_allclose(actual.numpy(), finite_difference.numpy(), rtol=2.e-5, atol=1.e-8)


def test_unused_process_factor_and_nonfinite_noise(request):
    model, inputs = _fixture(1, 0)
    # Finite but indefinite process covariance is unused at T=0. The frozen
    # simulator accepts this case; eager/XLA must not introduce a false veto.
    base = replace(model.physical_model.base_model, process_covariance=-tf.eye(2, dtype=D))
    unused = replace(model, physical_model=replace(model.physical_model, base_model=base))
    original, source_sha = _original_model(unused)
    expected = original.simulate_from_standard_normals(*inputs)
    checked = []
    for jit_compile in (False, True):
        actual = unused.simulate_from_standard_normals(*inputs, jit_compile=jit_compile)
        _assert_record_equal(actual, expected, atol=5.e-12, rtol=5.e-12)
        for horizon in (0, 1):
            healthy, operands = _fixture(1, horizon)
            owner = candidate.latent_preclip_simulation_program(healthy, horizon, jit_compile=jit_compile)
            for index in (1, 2, 3):
                if index == 2 and horizon == 0:
                    continue
                invalid = list(operands)
                invalid[index] = tf.fill(invalid[index].shape, tf.constant(float("nan"), D))
                _, valid = owner(*invalid)
                assert not bool(valid)
                with pytest.raises(ValueError, match="nonfinite_value"):
                    healthy.simulate_from_standard_normals(*invalid, jit_compile=jit_compile)
                checked.append([jit_compile, horizon, index])
    _write(request, "latent-sir-validity-review.json", {
        "reference_sha256": source_sha, "unused_process_factor_preserved": True,
        "nonfinite_noise_rejected": checked,
    })


@pytest.mark.parametrize("arm", ["original", "graph", "xla"])
def test_simulation_cost(arm, request):
    model, inputs = _fixture(9, 3)
    original, source_sha = _original_model(model)
    call = (original.simulate_from_standard_normals if arm == "original" else
            lambda *args: model.simulate_from_standard_normals(*args, jit_compile=arm == "xla"))
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    with GPUProcessMonitor(gpu) as monitor:
        before = memory_snapshot(gpu)
        begin = time.perf_counter()
        result = _json(call(*inputs))
        cold = time.perf_counter() - begin
        compiled = memory_snapshot(gpu)
        timings = []
        for _ in range(20):
            begin = time.perf_counter()
            replay = _json(call(*inputs))
            timings.append(time.perf_counter() - begin)
            assert result == replay
        after = memory_snapshot(gpu)
    _write(request, "latent-sir-cost.json", {"arm": arm, "reference_sha256": source_sha,
        "inputs": inputs, "result": result, "cold_seconds": cold, "warm_seconds": timings,
        "before": before, "compiled": compiled, "warm": after, "device_provenance": monitor.payload(),
        "role": "single_process_descriptive_fixture_cost"})
