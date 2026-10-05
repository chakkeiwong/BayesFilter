"""Independent diagnostic checks of the analytical score execution boundary.

NumPy and the frozen executor are reference authorities only. No-reset T>1
values are finite-program derivative diagnostics, not likelihood estimates.
These checks do not admit a canonical LEDH route.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    make_analytical_score_program,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "9d8202b77"
DTYPE = tf.float64


@pytest.fixture(scope="module")
def frozen_score():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:bayesfilter/highdim/ledh_canonical_score_tf.py"],
        cwd=ROOT, text=True,
    )
    module = ModuleType("_frozen_ledh_score_diagnostic")
    sys.modules[module.__name__] = module
    exec(compile(source, "<frozen-ledh-score-diagnostic>", "exec"), module.__dict__)  # noqa: S102 -- fixed Git test authority
    yield module, hashlib.sha256(source.encode()).hexdigest()
    sys.modules.pop(module.__name__, None)


def _model(theta, direction):
    eye = tf.eye(2, dtype=DTYPE)
    scale = 1. + .1 * theta[1]
    dscale = .1 * direction[1]
    q = .4 * tf.exp(theta[1]) * eye
    return NonlinearScoreModel(
        transition_mean_fn=lambda parameter, points: points + parameter[0] * tf.sin(points),
        transition_mean_tangent_fn=lambda parameter, points, tangents: (
            direction[0] * tf.sin(points) + (1. + parameter[0] * tf.cos(points)) * tangents
        ),
        observation_fn=lambda points: scale * points,
        observation_jacobian_fn=lambda points: tf.broadcast_to(scale * eye, [tf.shape(points)[0], 2, 2]),
        observation_tangent_fn=lambda points, tangents: scale * tangents + dscale * points,
        observation_jacobian_tangent_fn=lambda points, tangents: tf.broadcast_to(dscale * eye, [tf.shape(points)[0], 2, 2]),
        process_covariance=q,
        process_covariance_tangent_fn=lambda parameter: direction[1] * q,
        observation_covariance=.6 * eye,
    )


def _inputs(theta, direction, base, noises, observations):
    theta = tf.constant(theta, DTYPE)
    direction = tf.constant(direction, DTYPE)
    base = tf.constant(base, DTYPE)
    covariance = tf.broadcast_to(tf.eye(2, dtype=DTYPE), [8, 2, 2])
    return (
        theta, direction, base + .04 * theta[1], (1. + .1 * theta[0]) * covariance,
        tf.constant(noises, DTYPE), tf.constant(observations, DTYPE),
        tf.ones_like(base) * .04 * direction[1], .1 * direction[0] * covariance,
    )


def _reference(module, args, controls):
    theta, direction, initial, covariances, noises, observations, dinitial, dcovariances = args
    return module.canonical_value_and_analytical_score(
        _model(theta, direction), theta, initial, covariances, noises, observations,
        initial_state_tangent=dinitial, initial_covariance_tangent=dcovariances,
        with_score=True, **controls,
    )


@pytest.mark.parametrize("reset", ["none", "contract_e"])
def test_dynamic_score_owner_against_frozen_and_five_point(frozen_score, reset, request):
    module, source_sha = frozen_score
    rng = np.random.default_rng(73)
    base = rng.normal(size=(8, 2))
    noises = rng.normal(size=(2, 8, 2))
    observations = rng.normal(size=(2, 2))
    controls = {"flow_substeps": 3, "reset_policy": reset}
    if reset == "contract_e":
        controls["reset_design"] = tf.constant(np.tile(np.vstack((np.eye(2), -np.eye(2))), (2, 1)), DTYPE)
    program = make_analytical_score_program(
        _model, dtype=DTYPE, theta_shape=(2,), initial_state_shape=(8, 2),
        horizon=2, observation_dimension=2, **controls,
    )
    theta = np.array([.6, -.2])
    records = []
    for direction in (np.array([1., 0.]), np.array([0., 1.]), np.array([.3, -.7])):
        args = _inputs(theta, direction, base, noises, observations)
        value, score = program(*args)
        prior_value, prior_score = _reference(module, args, controls)
        assert np.isfinite(value.numpy()).all() and np.isfinite(score.numpy()).all()
        np.testing.assert_allclose(value, prior_value, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(score, prior_score, atol=1e-9, rtol=1e-9)
        differences = []
        for step in (1e-3, 5e-4):
            # Independent finite differences of the frozen VALUE executor.
            values = [float(_reference(module, _inputs(theta + shift * step * direction,
                direction, base, noises, observations), controls)[0]) for shift in (-2, -1, 1, 2)]
            derivative = (values[0] - 8. * values[1] + 8. * values[2] - values[3]) / (12. * step)
            np.testing.assert_allclose(score[0], derivative, atol=2e-6, rtol=2e-6)
            differences.append({"step": step, "finite_difference": derivative, "absolute_error": abs(float(score[0]) - derivative)})
        replay = program(*args)
        np.testing.assert_array_equal(value, replay[0])
        np.testing.assert_array_equal(score, replay[1])
        records.append({"direction": direction.tolist(), "value": float(value), "score": float(score[0]), "finite_differences": differences})
    # The scalar value is direction independent, while the analytical result
    # must follow each new direction through the same compiled executable.
    np.testing.assert_allclose([row["value"] for row in records], records[0]["value"], atol=1e-12, rtol=1e-12)
    assert not np.isclose(records[0]["score"], records[1]["score"])
    np.testing.assert_allclose(records[2]["score"], .3 * records[0]["score"] - .7 * records[1]["score"], atol=1e-10, rtol=1e-10)
    zero = program(*_inputs(theta, [0., 0.], base, noises, observations))
    np.testing.assert_array_equal(zero[1], [0.])
    changed_args = _inputs(theta + [.02, -.01], [0., 1.], base - .03, noises + .01, observations + .02)
    changed = program(*changed_args)
    reference = _reference(module, changed_args, controls)
    np.testing.assert_allclose(changed[0], reference[0], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(changed[1], reference[1], atol=1e-9, rtol=1e-9)
    assert not np.isclose(float(changed[0]), records[0]["value"])
    assert program.experimental_get_tracing_count() == 1
    hlo = program.experimental_get_compiler_ir(*args)(stage="hlo")
    assert hlo == program.experimental_get_compiler_ir(*changed_args)(stage="hlo")
    definition = program.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in definition.node}
    operations.update(node.op for function in definition.library.function for node in function.node_def)
    assert not operations & {"PyFunc", "PyFuncStateless", "EagerPyFunc", "XlaHostCompute"}
    record = {"reset": reset, "baseline": BASELINE, "frozen_score_sha256": source_sha,
        "trace_count": program.experimental_get_tracing_count(), "records": records,
        "scope": "analytical finite-program diagnostic; no LEDH admission"}
    directory = Path(request.config.getoption("xmlpath")).parent
    (directory / f"score-native-{reset}.json").write_text(json.dumps(record, indent=2) + "\n")
    (directory / f"score-native-{reset}.hlo.txt").write_text(hlo)
    print(json.dumps(record, sort_keys=True))


@pytest.mark.parametrize("options", [{"return_trace": True}, {"with_score": False},
    {"initial_state_tangent": 0.}, {"initial_covariance_tangent": 0.}, {"annealed_stages": 2}])
def test_score_owner_rejects_unsupported_options_before_tracing(options):
    with pytest.raises(ValueError):
        make_analytical_score_program(_model, dtype=DTYPE, theta_shape=(2,),
            initial_state_shape=(8, 2), horizon=2, observation_dimension=2, **options)


def test_score_owner_rejects_external_model_instance():
    with pytest.raises(TypeError, match="model_builder"):
        make_analytical_score_program(_model(tf.constant([.6, -.2], DTYPE), tf.constant([1., 0.], DTYPE)),
            dtype=DTYPE, theta_shape=(2,), initial_state_shape=(8, 2), horizon=2, observation_dimension=2)
