"""Independent original Generator-stream and model-simulation references."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.ops.generator_stream_tf import generator_normal_calls


@pytest.mark.parametrize("seed", [41, -3, 2**70+42])
@pytest.mark.parametrize("dimensions", [(1, 1), (4, 2)])
def test_generator_call_schedule_matches_original_stream(seed, dimensions):
    generator = tf.random.Generator.from_seed(seed)
    state, observation = [], []
    for _ in range(5):
        state.append(generator.normal([dimensions[0]], dtype=tf.float64))
        observation.append(generator.normal([dimensions[1]], dtype=tf.float64))
    for jit in (False, True):
        actual = generator_normal_calls(seed, 5, *dimensions, jit_compile=jit)
        tf.debugging.assert_near(actual[0], tf.stack(state), atol=1e-12, rtol=1e-12)
        tf.debugging.assert_near(actual[1], tf.stack(observation), atol=1e-12, rtol=1e-12)


@pytest.fixture(scope="module")
def original_models():
    source = subprocess.check_output(["git", "show", "3582b4ac:bayesfilter/highdim/models.py"],
        cwd=Path(__file__).resolve().parents[1], text=True)
    spec = importlib.util.spec_from_loader("model_simulation_frozen_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, "frozen_models_reference.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("family", ["sv", "sir", "predator_prey"])
@pytest.mark.parametrize("horizon", [0, 4])
def test_complete_simulation_preserves_original_seeded_paths(original_models, family, horizon):
    from bayesfilter.highdim.model_simulation_tf import model_simulation_program
    from bayesfilter.highdim.models import (
        StochasticVolatilitySSM,
        p30_predator_prey_fixture_model,
        p30_spatial_sir_fixture_model,
    )
    from bayesfilter.ops.generator_stream_tf import generator_seed_state

    if family == "sv":
        model = StochasticVolatilitySSM()
        theta = model.unconstrained_from_physical(.8, .6)
        reference = original_models.StochasticVolatilitySSM.simulate
    elif family == "sir":
        model = p30_spatial_sir_fixture_model(3)
        theta = tf.zeros([0], tf.float64)
        reference = original_models.SpatialSIRSSM.simulate
    else:
        model = p30_predator_prey_fixture_model()
        theta = model.true_parameters()
        reference = original_models.PredatorPreySSM.simulate
    args = (horizon, 421) if family == "sir" else (theta, horizon, 421)
    expected = reference(model, *args)
    for jit in (False, True):
        actual = model.simulate(*args, jit_compile=jit)
        for value, baseline in zip(actual, expected):
            tf.debugging.assert_near(value, baseline, atol=1e-12, rtol=1e-12)
    program = model_simulation_program(model, horizon+1, family)
    assert "HloModule" in program.experimental_get_compiler_ir(theta, generator_seed_state(421))(stage="hlo")
    assert program.experimental_get_tracing_count() == 1
