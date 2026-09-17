"""Baseline and execution-policy checks for the scalar adjacent TT route."""

import io
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.scalar_adjacent_native_tf import (
    make_scalar_adjacent_state_fixed_tt,
)
from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import (
    scalar_adjacent_state_fixed_tt_value,
)
from tests.highdim.test_zhao_cui_fixed_adjacent_tt_tf import (
    _adjacent_config,
    _model_and_theta,
    _transformed_observations,
)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    snapshot = tmp_path_factory.mktemp("scalar_tt_baseline")
    root = Path(__file__).resolve().parents[1]
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf",
            "bayesfilter",
            "tests/__init__.py",
            "tests/highdim/test_zhao_cui_fixed_adjacent_tt_tf.py",
        ],
        cwd=root,
        timeout=30,
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        handle.extractall(snapshot, filter="data")
    # The legacy eager score must use its own fitter and density dependencies.
    # Importing only the old filter into this process contaminates the oracle.
    program = """
import json
import tensorflow as tf
from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import scalar_adjacent_state_fixed_tt_score
from tests.highdim.test_zhao_cui_fixed_adjacent_tt_tf import (
    _adjacent_config, _model_and_theta, _transformed_observations,
)

model, theta = _model_and_theta()
references = {}
for transitioned, horizon in ((False, 3), (True, 3), (False, 6)):
    observations = tf.tile(_transformed_observations(), [2, 1])[:horizon]
    config = _adjacent_config(order=7, transition_before_first_observation=transitioned)
    result = scalar_adjacent_state_fixed_tt_score(
        model, theta, observations, config, finite_difference_h=())
    references[f"{transitioned}:{horizon}"] = {
        "theta": theta.numpy().tolist(),
        "observations": observations.numpy().tolist(),
        "log_likelihood": float(result.log_likelihood),
        "score": result.score.numpy().tolist(),
    }
print(json.dumps(references, allow_nan=False))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=snapshot,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(snapshot),
             "BAYESFILTER_OP_LIB_DIR": os.environ.get(
                 "BAYESFILTER_OP_LIB_DIR", str(root / "bayesfilter/ops"))},
        text=True, capture_output=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("transitioned", [False, True])
def test_complete_value_score_and_fit_veto_parity(baseline, jit, transitioned):
    tf.debugging.disable_traceback_filtering()
    model, theta = _model_and_theta()
    config = _adjacent_config(order=7, transition_before_first_observation=transitioned)
    observations = _transformed_observations()
    expected = baseline[f"{transitioned}:3"]
    np.testing.assert_array_equal(theta, expected["theta"])
    np.testing.assert_array_equal(observations, expected["observations"])
    program = make_scalar_adjacent_state_fixed_tt(
        model, config, observations.shape, jit_compile=jit
    )
    (increments, first, rest), score = program[1](theta, observations)
    np.testing.assert_allclose(
        tf.reduce_sum(increments), expected["log_likelihood"], atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(score, expected["score"], atol=1e-10, rtol=1e-10)
    assert bool(first["valid"]) and bool(tf.reduce_all(rest["valid"]))
    assert program is make_scalar_adjacent_state_fixed_tt(
        model, config, observations.shape, jit_compile=jit
    )


@pytest.mark.parametrize("jit", [False, True])
def test_condition_veto_is_preserved(jit):
    model, theta = _model_and_theta()
    config = _adjacent_config(order=7)
    config = replace(
        config,
        initial=replace(
            config.initial,
            fit_config=replace(config.initial.fit_config, condition_number_veto=0.5),
        ),
    )
    with pytest.raises(ValueError, match="CONDITION_NUMBER_VETO"):
        scalar_adjacent_state_fixed_tt_value(
            model, theta, _transformed_observations()[:1], config, jit_compile=jit
        )


def test_date_graph_is_bounded_and_longer_score_matches_baseline(baseline):
    model, theta = _model_and_theta()
    config = _adjacent_config(order=7)
    counts = []
    for horizon in (3, 6):
        observations = tf.tile(_transformed_observations(), [2, 1])[:horizon]
        program = make_scalar_adjacent_state_fixed_tt(model, config, observations.shape)
        (increments, _, _), score = program[1](theta, observations)
        expected = baseline[f"False:{horizon}"]
        np.testing.assert_array_equal(theta, expected["theta"])
        np.testing.assert_array_equal(observations, expected["observations"])
        np.testing.assert_allclose(
            tf.reduce_sum(increments), expected["log_likelihood"], atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(score, expected["score"], atol=1e-10, rtol=1e-10)
        graph = program[1].get_concrete_function().graph.as_graph_def()
        nodes = [
            *graph.node,
            *(
                node
                for function in graph.library.function
                for node in function.node_def
            ),
        ]
        assert not {node.op for node in nodes} & {
            "PyFunc",
            "EagerPyFunc",
            "PyFuncStateless",
        }
        assert program[1].experimental_get_tracing_count() == 1
        counts.append(len(nodes))
    assert counts[0] == counts[1]
