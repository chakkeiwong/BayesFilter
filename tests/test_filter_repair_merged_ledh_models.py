"""Diagnostic model callback parity; not complete LEDH or source admission."""

import ast
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_canonical_models_tf as models

BASELINE = "9d8202b779888f6150d219bc971df832cfe9c887"
RELATIVE = "bayesfilter/highdim/ledh_canonical_models_tf.py"
FACTORIES = (
    "austria_sir_canonical_model",
    "predator_prey_canonical_model",
    "diagonal_lgssm_canonical_model",
    "generalized_sv_canonical_model",
    "ksc_sv_canonical_model",
)


@pytest.fixture(scope="module")
def original():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{RELATIVE}"],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )
    namespace = {"__name__": "frozen_ledh_model_reference"}
    # Independent executable merged source, including its original NumPy constants.
    exec(compile(source, "frozen_ledh_model_reference", "exec"), namespace)  # noqa: S102
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def test_model_module_has_no_python_loops_or_numpy():
    tree = ast.parse(inspect.getsource(models))
    loops = (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp,
             ast.DictComp, ast.GeneratorExp)
    for node in ast.walk(tree):
        assert not isinstance(node, loops)
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "numpy" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "numpy"


def test_austria_adjacency_matches_original_configuration():
    from bayesfilter.highdim.models import (
        _zhao_cui_sir_austria_adjacency_xla,
        zhao_cui_sir_austria_model,
    )

    tf.debugging.assert_equal(_zhao_cui_sir_austria_adjacency_xla(),
                              zhao_cui_sir_austria_model()._adjacency_matrix)


def _arguments(name, dtype):
    if name.startswith("austria"):
        theta, observation = [.02, -.03, .01], [3. + .1 * j for j in range(9)]
        points = [[v for j in range(9) for v in (60. + i + j, 3. + .1 * j)]
                  for i in range(3)]
    elif name.startswith("predator"):
        theta, observation = [.6, 114., 25., .3, .5, .5], [51., 6.]
        points = [[50., 5.], [53., 7.], [43., 4.]]
    elif name.startswith("diagonal"):
        theta, observation = [.6, .7, .8, .3, .4], [.2, -.1, .4]
        points = [[.2, -.3, .7], [.5, .2, -.4], [-.1, .3, .2]]
    elif name.startswith("generalized"):
        theta, observation = [.3, -.4, -.5, -.3, .1], [.25]
        points = [[.2, -.3], [.5, .2], [-.1, .3]]
    else:
        theta, observation = [.3, -.2], [.25]
        points = [[-.5], [.4], [1.2]]
    p = tf.constant(points, dtype)
    d_points = tf.reshape(tf.cast(tf.range(tf.size(p)), dtype), p.shape) * .003 + .02
    direction = tf.constant([(.04 + .01 * j) * (-1.) ** j
                             for j in range(len(theta))], dtype)
    return tf.constant(theta, dtype), p, tf.constant(observation, dtype), d_points, direction


def _callbacks(factory, name, dtype, theta, points, observation, d_points, direction,
               frozen_model=None):
    kwargs = {"dtype": dtype} if name.startswith("austria") else {}
    model, set_direction = factory(theta, **kwargs) if frozen_model is None else frozen_model
    # Set a symbolic input on this traced factory, not a captured eager direction.
    set_direction(direction)
    mean = model.transition_mean_fn(theta, points)
    d_mean = model.transition_mean_tangent_fn(theta, points, d_points)
    result = {
        "transition": mean, "transition_tangent": d_mean,
        "observation": model.observation_fn(points),
        "observation_jacobian": model.observation_jacobian_fn(points),
        "observation_tangent": model.observation_tangent_fn(points, d_points),
        "process_covariance": model.process_covariance,
        "observation_covariance": model.observation_covariance,
    }
    if model.observation_log_density_fn is not None:
        result["observation_density"] = model.observation_log_density_fn(theta, points, observation)
        result["observation_density_tangent"] = model.observation_log_density_tangent_fn(
            theta, points, observation, d_points)
    if model.transition_log_density_fn is not None:
        result["transition_density"] = model.transition_log_density_fn(theta, points, mean)
        result["transition_density_tangent"] = model.transition_log_density_tangent_fn(
            theta, points, mean, d_points, d_mean)
    for field in ("process_covariance", "observation_covariance"):
        callback = getattr(model, field + "_tangent_fn")
        if callback is not None:
            result[field + "_tangent"] = callback(theta)
    return result


def _errors(actual, expected, tolerance):
    assert actual.keys() == expected.keys()
    errors = {}
    for key, value in actual.items():
        tf.debugging.assert_all_finite(value, key)
        delta = tf.abs(value - expected[key])
        limit = tf.constant(tolerance, value.dtype) * (1. + tf.abs(expected[key]))
        tf.debugging.assert_less_equal(delta, limit, message=key)
        errors[key] = float(tf.reduce_max(delta / limit))
    return errors


@pytest.mark.parametrize("name,dtype", [(name, tf.float64) for name in FACTORIES]
                         + [(FACTORIES[0], tf.float32)])
def test_callbacks_original_graph_xla_and_independent_tangents(name, dtype, original, request):
    namespace, baseline_hash = original
    initial = _arguments(name, dtype)
    theta, points, observation, d_points, direction = initial
    changed = (theta + .01, points + .02, observation - .03, d_points * .8, direction - .02)
    signature = tuple(tf.TensorSpec(arg.shape, dtype) for arg in initial)

    def make(factory, jit, frozen_model=None):
        return tf.function(lambda *args: _callbacks(factory, name, dtype, *args,
                                                   frozen_model=frozen_model),
                           input_signature=signature, jit_compile=jit, autograph=False)

    candidate = make(getattr(models, name), True)
    graph = make(getattr(models, name), False)  # Explicit diagnostic comparison.
    # The original factory has host validation and fixed covariance captures.
    # Keep that API intact: one graph per fixed factory, with symbolic directions.
    kwargs = {"dtype": dtype} if name.startswith("austria") else {}
    references = {label: make(namespace[name], False, namespace[name](args[0], **kwargs))
                  for label, args in (("initial", initial), ("changed", changed))}
    tolerance = 1e-12 if dtype == tf.float64 else 1e-5
    derivative_tolerance = 1e-7 if dtype == tf.float64 else 2e-3
    steps = (1e-4, 5e-5) if dtype == tf.float64 else (1e-2, 5e-3)
    hlo = candidate.experimental_get_compiler_ir(*initial)(stage="hlo")
    comparisons, derivatives, first = [], [], None
    for label, args in (("initial", initial), ("changed", changed), ("replay", initial)):
        actual = candidate(*args)
        reference = references["initial" if label == "replay" else label]
        comparisons.append({"label": label,
            "original_scaled_errors": _errors(actual, reference(*args), tolerance),
            "graph_scaled_errors": _errors(actual, graph(*args), tolerance),
            "outputs": {key: value.numpy().tolist() for key, value in actual.items()},
        })
        if first is None:
            first = actual
        if label == "replay":
            for key in actual:
                tf.debugging.assert_equal(actual[key], first[key], message=key)
            continue
        t, p, o, dp, dt = args
        for step in steps:
            upper = candidate(t + step * dt, p + step * dp, o, dp, dt)
            lower = candidate(t - step * dt, p - step * dp, o, dp, dt)
            numerical, analytical = {}, {}
            for primal in ("transition", "observation_density", "transition_density",
                           "process_covariance", "observation_covariance"):
                tangent = primal + "_tangent"
                if tangent in actual:
                    numerical[primal] = (upper[primal] - lower[primal]) / (2. * step)
                    analytical[primal] = actual[tangent]
            derivatives.append({"label": label, "step": step,
                "scaled_errors": _errors(numerical, analytical, derivative_tolerance)})

    assert candidate.experimental_get_tracing_count() == 1
    assert graph.experimental_get_tracing_count() == 1
    assert all(reference.experimental_get_tracing_count() == 1 for reference in references.values())
    assert hlo == candidate.experimental_get_compiler_ir(*changed)(stage="hlo")
    definition = candidate.get_concrete_function().graph.as_graph_def()
    nodes = list(definition.node)
    nodes.extend(node for function in definition.library.function for node in function.node_def)
    operations = {node.op for node in nodes}
    assert not operations & {"PyFunc", "PyFuncStateless", "EagerPyFunc", "XlaHostCompute"}
    native_loops = sum(node.op in ("While", "StatelessWhile") for node in nodes)
    expected_loops = 2 if name.startswith(("austria", "predator")) else 0
    assert native_loops == expected_loops
    if expected_loops:
        assert "while(" in hlo
    expected_device = "GPU" if tf.config.list_logical_devices("GPU") else "CPU"
    assert all(f"device:{expected_device}:" in value.device for value in first.values())
    report = {"schema": "filter_repair_merged_ledh_model_callbacks.v1", "passed": True,
        "model": name, "dtype": dtype.name, "device": expected_device,
        "baseline": BASELINE, "baseline_module_sha256": baseline_hash,
        "comparisons": comparisons, "independent_tangents": derivatives,
        "trace_count": 1, "native_loop_count": native_loops, "graph_nodes": len(nodes),
        "original_traces_per_fixed_factory": [1, 1],
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "nonclaims": ["No complete filter, score, author-source, training or admission qualification.",
                      "No isolated cold/warm performance or memory comparison."],
    }
    directory = Path(request.config.getoption("xmlpath")).parent
    stem = f"ledh-model-{name}-{dtype.name}"
    (directory / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / f"{stem}.hlo.txt").write_text(hlo)
