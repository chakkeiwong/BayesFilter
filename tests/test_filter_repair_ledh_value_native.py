"""Frozen-source and independent-reference diagnostics for native LEDH execution.

This is execution equivalence, not canonical LEDH admission or MC accuracy.
NumPy and historical Python recurrence are independent test authorities only.
"""

import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_value_program_tf import (
    make_canonical_value_program,
    systematic_ancestor_indices,
)
from bayesfilter.highdim.ledh_flow_perparticle_tf import ledh_flow_per_particle
from bayesfilter.highdim.ledh_ukf_lifecycle_tf import ukf_predict_per_particle
from bayesfilter.ops.legacy_fraction_tf import legacy_fraction
from bayesfilter.ops.slogdet_tf import determinant_tf
from scripts.filter_repair_cost_provenance import GPUProcessMonitor

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "9d8202b77"


@pytest.fixture(scope="module")
def authorities():
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASELINE}:bayesfilter/highdim/ledh_canonical_filter_tf.py",
        ],
        cwd=ROOT,
        text=True,
    )
    baseline = ModuleType("_ledh_value_frozen_diagnostic")
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, "<frozen-ledh-value-diagnostic>", "exec"), baseline.__dict__)  # noqa: S102 -- fixed Git diagnostic authority
    flow_source = subprocess.check_output([
        "git", "show", f"{BASELINE}:bayesfilter/highdim/ledh_flow_perparticle_tf.py",
    ], cwd=ROOT, text=True)
    flow_authority = ModuleType("_ledh_flow_frozen_diagnostic")
    exec(compile(flow_source, "<frozen-ledh-flow-diagnostic>", "exec"), flow_authority.__dict__)  # noqa: S102 -- fixed Git diagnostic authority
    baseline.ledh_flow_per_particle = flow_authority.ledh_flow_per_particle
    baseline._flow_source_sha256 = hashlib.sha256(flow_source.encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        "_ledh_lgssm_diagnostic", ROOT / "tests/highdim/test_ledh_canonical_filter.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    yield baseline, fixture, hashlib.sha256(source.encode()).hexdigest()
    sys.modules.pop(baseline.__name__, None)


def _frozen_reference(baseline, callbacks, observations, inputs, controls):
    initial, process, uniforms = inputs
    draws = iter([initial, *tf.unstack(process)])
    baseline._replication_generator = lambda seed: SimpleNamespace(
        normal=lambda shape, dtype: next(draws)
    )

    class DiagnosticNumpy:
        random = SimpleNamespace(
            default_rng=lambda seed: SimpleNamespace(
                uniform=iter(uniforms[seed - 17].numpy()).__next__
            )
        )

        def __getattr__(self, name):
            return getattr(np, name)

    baseline.np = DiagnosticNumpy()
    reset = baseline._restore_cloud_primal
    resets = []
    def capture(*args, **kwargs):
        result = reset(*args, **kwargs)
        resets.append(result)
        return result
    baseline._restore_cloud_primal = capture
    try:
        result = baseline.canonical_value_and_diagnostics(
            callbacks, observations, particle_count=8, seed=0, resample_seed=17, **controls
        )
    finally:
        baseline._restore_cloud_primal = reset
    horizon = int(observations.shape[0])
    valid = [bool(row["reset_valid"]) for row in resets]
    condition = [float(row["maximum_diagonal_scaled_system_condition"]) for row in resets]
    all_valid = len(resets) == horizon and all(valid)
    result.update({"per_step_reset_valid": tf.constant(valid + [False] * (horizon - len(resets))),
        "all_resets_valid": tf.constant(all_valid),
        "numerical_valid": result["program_valid"] & all_valid,
        "per_step_reset_scaled_system_condition": tf.constant(condition + [float("nan")] * (horizon - len(resets)), observations.dtype)})
    return result


def _compare(actual, expected):
    report = {}
    completed = int(actual["marginal_steps_completed"])
    for key, reference in expected.items():
        if key == "model_id":
            continue
        value = actual[key]
        if key in {"per_step_marginal_tv_error", "per_step_marginal_valid"}:
            value = value[:completed]
        lhs, rhs = value.numpy(), reference.numpy()
        assert lhs.shape == rhs.shape, key
        if key == "per_step_reset_scaled_system_condition":
            # Backend eigensolver rounding is explanatory, never an admission gate.
            np.testing.assert_array_equal(np.isnan(lhs), np.isnan(rhs), err_msg=key)
            report[key] = {"actual": lhs.tolist(), "expected": rhs.tolist()}
        elif value.dtype.is_floating:
            np.testing.assert_array_equal(np.isnan(lhs), np.isnan(rhs), err_msg=key)
            np.testing.assert_allclose(
                lhs, rhs, atol=1e-6, rtol=1e-6, equal_nan=True, err_msg=key
            )
            finite = np.isfinite(lhs) & np.isfinite(rhs)
            delta = np.abs(lhs[finite] - rhs[finite])
            report[key] = float(delta.max()) if delta.size else None
        else:
            np.testing.assert_array_equal(lhs, rhs, err_msg=key)
    return report


@pytest.mark.parametrize(
    "case,horizon,stages,annealed,cap,dual",
    [
        ("one_step", 1, 1, False, float("inf"), False),
        ("composed", 3, 3, False, 0.8, False),
        ("annealed", 3, 3, True, 0.8, False),
        ("dual_trust", 3, 1, False, float("inf"), True),
        ("invalid_initial", 3, 1, False, float("inf"), False),
        ("invalid_prediction", 3, 1, False, float("inf"), False),
        ("invalid_observation", 3, 1, False, float("inf"), False),
    ],
)
def test_full_value_fixed_inputs(
    authorities, case, horizon, stages, annealed, cap, dual, request
):
    baseline, fixture, source_sha = authorities
    model = fixture._lgssm_model(13, horizon=horizon)
    callbacks = fixture._callbacks_for_lgssm(model)
    observations = tf.constant(model["observations"], tf.float64)
    if case == "invalid_initial":
        callbacks = replace(
            callbacks, initial_mean=tf.constant([float("nan"), 0.0], tf.float64)
        )
    elif case == "invalid_prediction":
        original = callbacks.transition_mean_fn
        callbacks = replace(
            callbacks,
            transition_mean_fn=lambda points, time: tf.where(
                time == 1,
                tf.fill(tf.shape(points), tf.constant(float("nan"), tf.float64)),
                original(points, time),
            ),
        )
    elif case == "invalid_observation":
        observations = tf.tensor_scatter_nd_update(
            observations, [[1, 0]], [tf.constant(float("nan"), tf.float64)]
        )
    rng = np.random.default_rng(123)
    inputs = (
        tf.constant(rng.normal(size=(8, 2)), tf.float64),
        tf.constant(rng.normal(size=(horizon, 8, 2)), tf.float64),
        tf.constant(rng.uniform(size=(horizon, stages)), tf.float64),
    )
    controls = {
        "flow_substeps": 3,
        "temper_stages": stages,
        "annealed_resampling": annealed,
        "flow_prior_cap": cap,
        "sinkhorn_steps": 2,
        "balance_steps": 2,
        "dual_cap_enabled": dual,
        "trust_region_enabled": dual,
    }
    compiled = make_canonical_value_program(
        callbacks,
        tf.TensorSpec(observations.shape, tf.float64),
        particle_count=8,
        **controls,
    )
    expected = _frozen_reference(baseline, callbacks, observations, inputs, controls)
    actual = compiled(observations, *inputs)
    directory = Path(request.config.getoption("xmlpath")).parent
    raw = {"actual": {k: v.numpy().tolist() for k, v in actual.items()},
           "expected": {k: v.numpy().tolist() for k, v in expected.items() if k != "model_id"}}
    (directory / f"value-native-{case}-raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    errors = _compare(actual, expected)
    replay = compiled(observations, *inputs)
    for key in actual:
        np.testing.assert_array_equal(actual[key], replay[key], err_msg=key)
    changed_inputs = (inputs[0] + 0.15, inputs[1] - 0.05, 1.0 - inputs[2])
    changed_observations = observations + 0.1
    changed = compiled(changed_observations, *changed_inputs)
    changed_reference = _frozen_reference(
        baseline, callbacks, changed_observations, changed_inputs, controls
    )
    changed_errors = _compare(changed, changed_reference)
    if not case.startswith("invalid"):
        assert bool(actual["program_valid"])
        assert not np.isclose(actual["value"], changed["value"])
    else:
        assert not bool(actual["program_valid"])
    if case == "invalid_initial":
        assert int(actual["marginal_steps_completed"]) == 0
    elif case == "invalid_prediction":
        assert int(actual["marginal_steps_completed"]) == 1
    assert compiled.experimental_get_tracing_count() == 1
    hlo = compiled.experimental_get_compiler_ir(observations, *inputs)(stage="hlo")
    assert hlo == compiled.experimental_get_compiler_ir(
        changed_observations, *changed_inputs
    )(stage="hlo")
    definition = compiled.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in definition.node}
    operations.update(
        node.op
        for function in definition.library.function
        for node in function.node_def
    )
    assert "StatelessWhile" in operations or "While" in operations
    assert not operations & {
        "PyFunc",
        "PyFuncStateless",
        "EagerPyFunc",
        "XlaHostCompute",
    }
    report = {
        "case": case,
        "passed": True,
        "baseline": BASELINE,
        "baseline_sha256": source_sha,
        "flow_baseline_sha256": baseline._flow_source_sha256,
        "max_absolute_errors": errors,
        "changed_errors": changed_errors,
        "trace_count": 1,
        "completed": int(actual["marginal_steps_completed"]),
        "device": actual["value"].device,
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "nonclaims": [
            "No seeded public wrapper, score, MC accuracy or canonical admission."
        ],
    }
    (directory / f"value-native-{case}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (directory / f"value-native-{case}.hlo.txt").write_text(hlo)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_systematic_matches_sequential_numpy(dtype):
    compiled = tf.function(
        systematic_ancestor_indices,
        input_signature=[tf.TensorSpec([8], dtype), tf.TensorSpec([], tf.float64)],
        jit_compile=True,
        autograph=False,
    )
    weights = np.array(
        [0.125, 0.0, 0.125, 0.25, 0.25, 0.0, 0.125, 0.125], dtype=dtype.as_numpy_dtype
    )
    for uniform in (0.0, 0.5, np.nextafter(1.0, 0.0)):
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        expected = np.searchsorted(
            cumulative, (uniform + np.arange(8)) / 8, side="left"
        )
        np.testing.assert_array_equal(
            compiled(weights, tf.constant(uniform, tf.float64)), expected
        )


def test_ukf_linear_prediction_independent(authorities):
    _, fixture, _ = authorities
    model = fixture._lgssm_model(13, horizon=1)
    callbacks = fixture._callbacks_for_lgssm(model)
    states = np.array([[0.1, 0.2], [-0.3, 0.4]])
    covariance = np.array([[[1.0, 0.2], [0.2, 0.5]], [[0.7, -0.1], [-0.1, 0.4]]])
    compiled = tf.function(
        lambda x, p: ukf_predict_per_particle(
            x,
            p,
            lambda points: callbacks.transition_mean_fn(points, 0),
            callbacks.process_noise_covariance,
        ),
        input_signature=[
            tf.TensorSpec([2, 2], tf.float64),
            tf.TensorSpec([2, 2, 2], tf.float64),
        ],
        jit_compile=True,
        autograph=False,
    )
    mean, cov = compiled(states, covariance)
    f = model["transition"]
    np.testing.assert_allclose(mean, states @ f.T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        cov, f @ covariance @ f.T + model["process_cov"], atol=1e-8, rtol=1e-8
    )


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_batched_native_determinant(dtype):
    matrices = np.array([
        [[1., 2., 3.], [4., 0., 6.], [7., 8., 10.]],
        [[0., 2., 0.], [3., 0., 0.], [0., 0., 4.]],
        [[1., 2., 3.], [1., 2., 3.], [0., 0., 0.]],
        [[1., .1, 0.], [.1, 2., .2], [0., .2, 3.]],
    ], dtype=dtype.as_numpy_dtype).reshape(2, 2, 3, 3)
    compiled = tf.function(determinant_tf,
        input_signature=[tf.TensorSpec(matrices.shape, dtype)], jit_compile=True, autograph=False)
    tolerance = 1e-5 if dtype == tf.float32 else 1e-12
    np.testing.assert_allclose(compiled(matrices), np.linalg.det(matrices), atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(compiled(matrices), compiled(matrices))


@pytest.mark.parametrize("denominator", [1, 2, 3, 7, 24, 4097, 2**31 - 1])
def test_legacy_fraction_retains_float32_rounding(denominator):
    indices = np.array([0, 1, denominator // 2, denominator - 1, denominator], dtype=np.int32)
    compiled = tf.function(lambda values: legacy_fraction(values, denominator, tf.float64),
        input_signature=[tf.TensorSpec([5], tf.int32)], jit_compile=True, autograph=False)
    expected = np.array([float(index) / denominator for index in indices], dtype=np.float32).astype(np.float64)
    np.testing.assert_array_equal(compiled(indices), expected)


def test_full_flow_frozen_source(authorities):
    baseline, fixture, _ = authorities
    callbacks = fixture._callbacks_for_lgssm(fixture._lgssm_model(13, horizon=1))
    def evaluate(x, prior, obs, implementation):
        return implementation(anchor_states=x * .9, pre_flow_states=x,
            predicted_covariances=prior, observation=obs,
            observation_fn=lambda p: callbacks.observation_fn(p, 0),
            observation_jacobian_fn=lambda p: callbacks.observation_jacobian_fn(p, 0),
            observation_covariance=callbacks.observation_covariance,
            prior_means=x * .9, substeps=3)
    compiled = tf.function(lambda x, prior, obs: evaluate(x, prior, obs, ledh_flow_per_particle),
        input_signature=[tf.TensorSpec([8, 2], tf.float64), tf.TensorSpec([8, 2, 2], tf.float64),
                         tf.TensorSpec([2], tf.float64)], jit_compile=True, autograph=False)
    rng = np.random.default_rng(37)
    states = tf.constant(rng.normal(size=(8, 2)), tf.float64)
    covs = tf.broadcast_to(tf.constant([[1., .2], [.2, .7]], tf.float64), [8, 2, 2])
    obs = tf.constant([.1, .2], tf.float64)
    expected = evaluate(states, covs, obs, baseline.ledh_flow_per_particle)
    actual = compiled(states, covs, obs)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-12, err_msg=key)


def test_dual_reset_localization(authorities, request, monkeypatch):
    """Explanatory report, not an equivalence pass, for the 03825 veto."""
    baseline, fixture, _ = authorities
    from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal
    model = fixture._lgssm_model(13, horizon=3)
    callbacks = fixture._callbacks_for_lgssm(model)
    observations = tf.constant(model["observations"], tf.float64)
    rng = np.random.default_rng(123)
    inputs = (tf.constant(rng.normal(size=(8, 2)), tf.float64),
              tf.constant(rng.normal(size=(3, 8, 2)), tf.float64),
              tf.constant(rng.uniform(size=(3, 1)), tf.float64))
    controls = {"flow_substeps": 3, "sinkhorn_steps": 2, "balance_steps": 2,
                "dual_cap_enabled": True, "trust_region_enabled": True}
    calls = []
    def record(*args, **kwargs):
        result = _restore_cloud_primal(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result
    monkeypatch.setattr(baseline, "_restore_cloud_primal", record)
    expected = _frozen_reference(baseline, callbacks, observations, inputs, controls)
    options = calls[0][1]
    signature = [tf.TensorSpec([8, 2], tf.float32), tf.TensorSpec([8], tf.float32),
                 tf.TensorSpec([8, 2], tf.float32)]
    compiled = tf.function(lambda *args: _restore_cloud_primal(*args, **options),
        input_signature=signature, jit_compile=True, autograph=False)
    graph = tf.function(lambda *args: _restore_cloud_primal(*args, **options),
        input_signature=signature, jit_compile=False, autograph=False)
    report = {"claim": "localization_only", "reset_steps": []}
    for args, _, reference in calls:
        row = {"inputs": [a.numpy().tolist() for a in args], "modes": {}}
        for mode, result in (("eager", reference), ("graph", graph(*args)), ("xla", compiled(*args))):
            row["modes"][mode] = {key: value.numpy().tolist() for key, value in result.items()}
        report["reset_steps"].append(row)
    for mode, reset in (("eager", _restore_cloud_primal), ("graph", graph), ("xla", compiled)):
        if mode == "eager":
            monkeypatch.setattr(baseline, "_restore_cloud_primal", reset)
        else:
            monkeypatch.setattr(baseline, "_restore_cloud_primal", lambda *args, _reset=reset, **kwargs: _reset(*args))
        result = _frozen_reference(baseline, callbacks, observations, inputs, controls)
        report[mode] = {k: v.numpy().tolist() for k, v in result.items() if k != "model_id"}
    for mode, use_xla in (("native_graph", False), ("native_xla", True)):
        program = make_canonical_value_program(callbacks, tf.TensorSpec([3, 2], tf.float64),
            particle_count=8, jit_compile=use_xla, **controls)
        result = program(observations, *inputs)
        report[mode] = {k: v.numpy().tolist() for k, v in result.items()}
    assert bool(expected["program_valid"])
    directory = Path(request.config.getoption("xmlpath")).parent
    (directory / "dual-reset-localization.json").write_text(json.dumps(report, indent=2) + "\n")


def test_dual_reset_precision_reference(request):
    """Independent precision diagnosis; does not change/admit the FP32 path."""
    from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal
    artifact_root = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    source_path = artifact_root / "run-03826/dual-reset-localization.json"
    source = json.loads(source_path.read_text())
    options = {"epsilon": 2., "sinkhorn_steps": 2, "balance_steps": 2, "ridge": 1e-5,
               "reset_policy": "contract_e", "dual_cap_enabled": True, "trust_region_enabled": True}
    kernels = {}
    for name, jit in (("graph", False), ("xla", True)):
        kernels[name] = tf.function(lambda *args: _restore_cloud_primal(*args, **options),
            input_signature=[tf.TensorSpec([8, 2], tf.float64), tf.TensorSpec([8], tf.float64),
                             tf.TensorSpec([8, 2], tf.float64)], jit_compile=jit, autograph=False)
    report = {"claim": "precision_diagnostic_only", "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "steps": []}
    for row in source["reset_steps"]:
        args = tuple(tf.constant(value, tf.float64) for value in row["inputs"])
        high = {"eager": _restore_cloud_primal(*args, **options)}
        high.update({mode: kernel(*args) for mode, kernel in kernels.items()})
        reference = high["eager"]["particles"].numpy()
        item = {"fp64_mode_max_difference": {mode: float(np.max(np.abs(result["particles"].numpy() - reference)))
                   for mode, result in high.items()},
                "fp32_error_against_fp64": {mode: float(np.max(np.abs(np.array(result["particles"]) - reference)))
                   for mode, result in row["modes"].items()},
                "fp64_diagnostics": {key: value.numpy().tolist() for key, value in high["eager"].items()}}
        report["steps"].append(item)
    directory = Path(request.config.getoption("xmlpath")).parent
    (directory / "dual-reset-precision.json").write_text(json.dumps(report, indent=2) + "\n")


def test_flow_fraction_localization(authorities, request):
    """GPU float64 flow discrepancy attribution; report without admission."""
    baseline, fixture, _ = authorities
    callbacks = fixture._callbacks_for_lgssm(fixture._lgssm_model(13, horizon=1))
    rng = np.random.default_rng(37)
    states = tf.constant(rng.normal(size=(8, 2)), tf.float64)
    covs = tf.broadcast_to(tf.constant([[1., .2], [.2, .7]], tf.float64), [8, 2, 2])
    obs = tf.constant([.1, .2], tf.float64)
    report = {}
    for substeps in (1, 2, 3, 7):
        def evaluate(x, prior, obs, implementation, _substeps=substeps):
            return implementation(anchor_states=x * .9, pre_flow_states=x,
                predicted_covariances=prior, observation=obs,
                observation_fn=lambda p: callbacks.observation_fn(p, 0),
                observation_jacobian_fn=lambda p: callbacks.observation_jacobian_fn(p, 0),
                observation_covariance=callbacks.observation_covariance,
                prior_means=x * .9, substeps=_substeps)
        compiled = tf.function(lambda x, prior, obs, _evaluate=evaluate:
            _evaluate(x, prior, obs, ledh_flow_per_particle),
            input_signature=[tf.TensorSpec([8, 2], tf.float64), tf.TensorSpec([8, 2, 2], tf.float64),
                             tf.TensorSpec([2], tf.float64)], jit_compile=True, autograph=False)
        actual = compiled(states, covs, obs)
        expected = evaluate(states, covs, obs, baseline.ledh_flow_per_particle)
        fractions = tf.function(lambda indices, _substeps=substeps:
            tf.cast(tf.cast(tf.cast(indices, tf.float64) / _substeps, tf.float32), tf.float64),
            input_signature=[tf.TensorSpec([substeps], tf.int32)], jit_compile=True, autograph=False)
        expected_fraction = tf.stack([tf.cast(index / substeps, tf.float64) for index in range(1, substeps + 1)])
        report[substeps] = {"fields": {key: {"max_error": float(tf.reduce_max(tf.abs(actual[key] - expected[key]))),
            "actual": actual[key].numpy().tolist(), "expected": expected[key].numpy().tolist()} for key in actual},
            "fractions": fractions(tf.range(1, substeps + 1)).numpy().tolist(),
            "expected_fractions": expected_fraction.numpy().tolist()}
    directory = Path(request.config.getoption("xmlpath")).parent
    (directory / "flow-fraction-localization.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("arm", ["prior_graph", "native_graph", "native_xla"])
def test_flow_isolated_cost(authorities, arm, request):
    """Fresh-process dependency costs; reference/export are outside timing."""
    baseline, fixture, source_hash = authorities
    callbacks = fixture._callbacks_for_lgssm(fixture._lgssm_model(13, horizon=1))
    rng = np.random.default_rng(37)
    states = tf.constant(rng.normal(size=(32, 2)), tf.float64)
    covs = tf.broadcast_to(tf.constant([[1., .2], [.2, .7]], tf.float64), [32, 2, 2])
    obs = tf.constant([.1, .2], tf.float64)
    def evaluate(x, prior, observation, implementation):
        return implementation(anchor_states=x * .9, pre_flow_states=x,
            predicted_covariances=prior, observation=observation,
            observation_fn=lambda p: callbacks.observation_fn(p, 0),
            observation_jacobian_fn=lambda p: callbacks.observation_jacobian_fn(p, 0),
            observation_covariance=callbacks.observation_covariance,
            prior_means=x * .9, substeps=24)
    implementation = baseline.ledh_flow_per_particle if arm == "prior_graph" else ledh_flow_per_particle
    def sync(result):
        for value in result.values():
            value.numpy()
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    device = "GPU:0" if gpu else "CPU:0"
    def allocator():
        try:
            return tf.config.experimental.get_memory_info(device)
        except ValueError:
            if gpu:
                raise
            return None
    try:
        tf.config.experimental.reset_memory_stats(device)
    except ValueError:
        if gpu:
            raise
    def rss():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("RSS sample unavailable")
    rss_samples = []
    stop = threading.Event()
    def sample():
        while not stop.wait(.01):
            rss_samples.append(rss())
    sampler = threading.Thread(target=sample, daemon=True)
    before = {"rss_bytes": rss(), "allocator": allocator(),
              "process_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    with GPUProcessMonitor(gpu) as monitor:
        sampler.start()
        try:
            begin = time.perf_counter()
            compiled = tf.function(lambda x, prior, observation: evaluate(x, prior, observation, implementation),
                input_signature=[tf.TensorSpec([32, 2], tf.float64), tf.TensorSpec([32, 2, 2], tf.float64),
                                 tf.TensorSpec([2], tf.float64)], jit_compile=arm == "native_xla", autograph=False)
            first = compiled(states, covs, obs)
            sync(first)
            cold = time.perf_counter() - begin
            after_cold = {"rss_bytes": rss(), "allocator": allocator()}
            times = []
            for _ in range(15):
                begin = time.perf_counter()
                result = compiled(states, covs, obs)
                sync(result)
                times.append(time.perf_counter() - begin)
            after_warm = {"rss_bytes": rss(), "allocator": allocator(),
                "process_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        finally:
            stop.set()
            sampler.join(timeout=5)
    assert not sampler.is_alive()
    assert rss_samples
    expected = evaluate(states, covs, obs, baseline.ledh_flow_per_particle)
    errors = {}
    for key in expected:
        np.testing.assert_allclose(first[key], expected[key], atol=1e-12, rtol=1e-12, err_msg=key)
        np.testing.assert_array_equal(first[key], result[key])
        errors[key] = float(tf.reduce_max(tf.abs(first[key] - expected[key])))
    changed = compiled(states + .1, covs * 1.1, obs - .1)
    changed_reference = evaluate(states + .1, covs * 1.1, obs - .1, baseline.ledh_flow_per_particle)
    for key in changed:
        np.testing.assert_allclose(changed[key], changed_reference[key], atol=1e-12, rtol=1e-12, err_msg=key)
    assert compiled.experimental_get_tracing_count() == 1
    definition = compiled.get_concrete_function().graph.as_graph_def()
    nodes = list(definition.node) + [node for fn in definition.library.function for node in fn.node_def]
    assert not {node.op for node in nodes} & {"PyFunc", "PyFuncStateless", "EagerPyFunc", "XlaHostCompute"}
    directory = Path(request.config.getoption("xmlpath")).parent
    hlo_hash = None
    if arm == "native_xla":
        hlo = compiled.experimental_get_compiler_ir(states, covs, obs)(stage="hlo")
        hlo_hash = hashlib.sha256(hlo.encode()).hexdigest()
        (directory / "flow-cost.hlo.txt").write_text(hlo)
    report = {"schema": "filter_repair_ledh_flow_cost.v1", "arm": arm,
        "shape": [32, 2, 2], "substeps": 24, "dtype": "float64", "seeds": [13, 37],
        "baseline": BASELINE, "baseline_sha256": source_hash, "flow_baseline_sha256": baseline._flow_source_sha256,
        "device": first["post_flow_states"].device, "cold_seconds": cold, "warm_seconds": times,
        "before": before, "after_cold": after_cold, "after_warm": after_warm,
        "sampled_peak_rss_bytes": max([before["rss_bytes"], *rss_samples, after_warm["rss_bytes"]]),
        "rss_sample_count": len(rss_samples), "graph_node_count": len(nodes), "trace_count": 1,
        "hlo_sha256": hlo_hash, "absolute_errors": errors, "cost_provenance": monitor.payload(),
        "valid": True, "nonclaims": ["Full-filter, score, capacity, scientific or canonical admission."]}
    (directory / "flow-cost.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
