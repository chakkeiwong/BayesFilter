"""Fresh diagnostic parity for previously uncovered SIR/RQMC recurrences.

Pinned function bodies are local refactor oracles, not canonical LEDH evidence.
The campaign's separate process measurements remain the timing authority.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import types
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_latent_sir_tf as sir
from bayesfilter.highdim import ledh_pfpf_genut_initial_rqmc_tf as rqmc
from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import (
    diagonal_lgssm_callbacks,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf"
D = tf.float64


@lru_cache(maxsize=2)
def _original(name):
    path = f"bayesfilter/highdim/{name}.py"
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, text=True)
    module = types.ModuleType(f"filter_repair_reference_{name}")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{BASELINE}:{path}", "exec"), module.__dict__)  # noqa: S102 - pinned local refactor oracle
    return module


def _graph(call):
    graph = call.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
    assert not ({node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"})
    return nodes


@pytest.mark.parametrize("dimension", [2, 5])
def test_latent_sir_matrix_derivatives_preserve_baseline_and_fd(dimension):
    before = _original("ledh_contract_e_latent_sir_tf")
    rows = tf.reshape(tf.cast(tf.range(2 * dimension * dimension), D), [2, dimension, dimension])
    rows = 0.03 * tf.sin(rows)
    matrix = rows @ tf.linalg.matrix_transpose(rows) + tf.eye(dimension, batch_shape=[2], dtype=D)
    tangent = tf.reshape(tf.cos(tf.cast(tf.range(2 * dimension * dimension * 3), D)),
                         [2, dimension, dimension, 3])
    tangent = 0.02 * (tangent + tf.transpose(tangent, [0, 2, 1, 3]))
    chol, inverse = tf.linalg.cholesky(matrix), tf.linalg.inv(matrix)

    @tf.function(input_signature=[tf.TensorSpec(chol.shape, D), tf.TensorSpec(tangent.shape, D)],
                 jit_compile=True, autograph=False)
    def evaluate(factor, directions):
        return sir._cholesky_jvp(factor, directions), sir._inverse_jvp(tf.linalg.cholesky_solve(
            factor, tf.eye(dimension, batch_shape=[2], dtype=D)), directions)

    actual = evaluate(chol, tangent)
    expected = before._cholesky_jvp(chol, tangent), before._inverse_jvp(inverse, tangent)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)
    epsilon = 1e-5
    for index in range(3):
        plus, minus = matrix + epsilon * tangent[..., index], matrix - epsilon * tangent[..., index]
        for result, operation in zip(actual, (tf.linalg.cholesky, tf.linalg.inv), strict=True):
            fd = (operation(plus) - operation(minus)) / (2 * epsilon)
            np.testing.assert_allclose(result[..., index], fd, rtol=1e-7, atol=1e-9)
    assert "HloModule" in evaluate.experimental_get_compiler_ir(chol, tangent)(stage="hlo")
    _graph(evaluate)


def _sir_fixture(horizon):
    spec = sir.LatentSIRStaticSpec(
        state_dimension=2, observation_dimension=1, compartments=1,
        initial_mean=tf.constant([0.3, 0.2], D),
        initial_covariance=tf.linalg.diag(tf.constant([0.25, 0.16], D)),
        process_covariance=tf.linalg.diag(tf.constant([0.25, 0.16], D)),
        base_observation_covariance=tf.constant([[0.16]], D),
        base_kappa=tf.constant([0.1], D), base_nu=tf.constant([1.], D),
        adjacency=tf.zeros([1, 1], D), neighbor_degree=tf.zeros([1], D),
        step=tf.constant(0.005, D), substeps=4, zhao_cui_rk4_variant=False,
    )
    design = tf.constant([
        [-1.5, -1.], [-1., 0.5], [-0.5, 1.], [-0.2, -1.5],
        [0.2, 1.5], [0.5, -0.5], [1., -1.], [1.5, 1.],
    ], D)
    residual = tf.constant([
        [-1., -0.75], [-0.75, 0.25], [-0.5, 0.5], [-0.25, -1.],
        [0.25, 1.], [0.5, -0.5], [0.75, -0.25], [1., 0.75],
    ], D)
    prepared = {
        "observations": tf.tile(tf.constant([[0.15], [0.1]], D), [horizon // 2, 1]),
        "initial_noise": design[None],
        "transition_noise": tf.broadcast_to((0.3 * design)[None, None], [1, horizon - 1, 8, 2]),
        "fixed_reset_mask": tf.ones([1, horizon], tf.bool),
        "residual_design": tf.broadcast_to(residual[None, None], [1, horizon, 8, 2]),
        "prepared_ridge": tf.fill([1, horizon], tf.constant(1e-5, D)),
        "epsilon": tf.constant(0.25, D), "scaling": tf.constant(0.9, D),
    }
    return spec, prepared


@pytest.mark.parametrize("horizon", [2, 4])
@pytest.mark.parametrize("jit", [False, True])
def test_latent_sir_complete_value_score_matches_pinned_program(horizon, jit):
    spec, prepared = _sir_fixture(horizon)
    theta = tf.constant([0.04, -0.03, 0.02], D)
    controls = {"steps": 20, "balance_steps": 100, "row_chunk_size": 8, "col_chunk_size": 8}
    reference = _original("ledh_contract_e_latent_sir_tf").latent_sir_contract_e_value_and_score_core(
        theta, prepared, spec, **controls)
    call = tf.function(lambda theta: sir.latent_sir_contract_e_value_and_score_core(
        theta, prepared, spec, **controls), input_signature=[tf.TensorSpec([3], D)],
        jit_compile=jit, autograph=False)
    result = call(theta)
    for name in ("objective", "score", "final_particles", "final_particles_tangent"):
        np.testing.assert_allclose(result[name], reference[name], rtol=1e-10, atol=1e-10)
        assert bool(tf.reduce_all(tf.math.is_finite(result[name])))
    for name in (name for name, value in result.items() if value.dtype == tf.bool):
        np.testing.assert_array_equal(result[name], reference[name])
    assert call.experimental_get_tracing_count() == 1
    if jit:
        assert "HloModule" in call.experimental_get_compiler_ir(theta)(stage="hlo")
    _graph(call)


def _rqmc_fixture(horizon):
    callbacks = diagonal_lgssm_callbacks()
    matrix = tf.cast(callbacks.model.observation_matrix, D)

    def observation_callbacks(_theta, _time):
        return (
            lambda points: tf.einsum("bnd,od->bno", points, matrix),
            lambda points: tf.broadcast_to(matrix[None, None], [*points.shape[:2], 3, 3]),
            lambda predicted, observed: observed[None, None] - predicted,
        )

    callbacks = replace(callbacks, observation_callbacks=observation_callbacks)
    design = tf.constant([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]], D)
    values = (tf.constant([0.2, 0.3, 0.4, 0.5, 0.8], D),
              tf.reshape(tf.linspace(tf.constant(-0.1, D), 0.2, horizon * 3), [horizon, 3]),
              0.3 * design, tf.broadcast_to((0.2 * design)[None], [horizon - 1, 4, 3]), design)
    return callbacks, values


@pytest.mark.parametrize("horizon", [2, 4])
@pytest.mark.parametrize("jit", [False, True])
def test_initial_rqmc_complete_value_statistical_score_and_histories(horizon, jit, monkeypatch):
    from experiments.dpf_implementation.tf_tfp.filters import (
        experimental_batched_ledh_pfpf_ot_tf as flow,
    )
    from experiments.dpf_implementation.tf_tfp.resampling import (
        annealed_transport_tf as transport,
    )

    # Explicit FP64 reference arm; the public flow default remains FP32/TF32.
    monkeypatch.setattr(flow, "DTYPE", D)
    monkeypatch.setattr(transport, "DTYPE", D)
    callbacks, values = _rqmc_fixture(horizon)
    controls = {"reset_policy": "none", "ancestry_policy": "existing_one_to_one",
                    "epsilon": 2., "sinkhorn_steps": 2, "balance_steps": 4}
    reference = _original("ledh_pfpf_genut_initial_rqmc_tf").finite_value_standard_score_initial_rqmc(
        callbacks, *values, functional_time_loop=False, **controls)
    call = tf.function(lambda *args: rqmc.finite_value_standard_score_initial_rqmc(
        callbacks, *args, functional_time_loop=False, **controls),
        input_signature=[tf.TensorSpec(value.shape, value.dtype) for value in values],
        jit_compile=jit, autograph=False)
    actual = call(*values)
    for result, expected in zip(tf.nest.flatten(actual), tf.nest.flatten(reference), strict=True):
        if result.dtype.is_floating:
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
        else:
            np.testing.assert_array_equal(result, expected)
    assert bool(actual[2]["program_valid"])
    assert bool(tf.reduce_all(tf.math.is_finite(actual[0])))
    assert bool(tf.reduce_all(tf.math.is_finite(actual[1])))
    assert call.experimental_get_tracing_count() == 1
    if jit:
        assert "HloModule" in call.experimental_get_compiler_ir(*values)(stage="hlo")
    _graph(call)


def test_remaining_route_recursions_have_no_python_iteration():
    import inspect

    for function in (sir._cholesky_jvp, sir._inverse_jvp, rqmc.finite_value_standard_score_initial_rqmc):
        nodes = ast.walk(ast.parse(inspect.getsource(function)))
        assert not any(isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp)) for node in nodes)
    assert inspect.signature(rqmc.finite_value_standard_score_initial_rqmc).parameters["functional_time_loop"].default is True
    for function in (sir.latent_sir_contract_e_canonical_value_and_score_tf,
                     sir.latent_sir_two_node_contract_e_value_and_score_tf):
        assert function._jit_compile
        assert function.specialization_cache_info().maxsize == 16
