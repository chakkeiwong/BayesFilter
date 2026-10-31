"""Independent CPU reference checks for a reporting-only trace diagnostic."""
import importlib.util
import math
from pathlib import Path
import pytest


@pytest.mark.parametrize('direction', [(1., 0.), (0., 1.), (.3, -.4)])
def test_predictive_functional_value_and_total_tangent(direction):
    import tensorflow as tf
    path = Path(__file__).resolve().parents[2] / 'docs/benchmarks/ksc_reset_mechanism_diagnostic.py'
    spec = importlib.util.spec_from_file_location('mechanism_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    theta = (1.5, -.2)
    states, d_states = (-1.2, .7, 2.1), (.2, -.1, .4)
    logits, d_logits = (-.8, .1, -.3), (.4, -.2, .3)
    observation = -.6
    def independent(step):
        gamma, beta = [a + step * b for a, b in zip(theta, direction)]
        phi = .5 * (1. + math.erf(gamma / math.sqrt(2.)))
        masses = [math.exp(a + step * b) for a, b in zip(logits, d_logits)]
        numerator = 0.
        for mass, x, dx in zip(masses, states, d_states):
            for w, m, v in zip(module.WEIGHTS, module.MEANS, module.VARIANCES):
                residual = observation - phi * (x + step * dx) - 2. * beta - m
                numerator += mass * w * math.exp(-.5 * residual ** 2 / (1. + v)) / math.sqrt(2. * math.pi * (1. + v))
        return math.log(numerator / sum(masses))
    tensor = lambda x: tf.constant(x, tf.float64)
    value, tangent = module.predictive_functional(tensor(theta), tensor(direction),
        tensor([states]), tensor([d_states]), tensor([logits]), tensor([d_logits]), tensor([observation]))
    assert float(value[0]) == pytest.approx(independent(0.), abs=1e-12)
    fd = (independent(1e-5) - independent(-1e-5)) / 2e-5
    assert float(tangent[0]) == pytest.approx(fd, abs=2e-9)
