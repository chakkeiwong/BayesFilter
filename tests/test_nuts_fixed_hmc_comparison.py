"""Focused inference/observable checks for the standalone comparison."""
import importlib.util
from pathlib import Path

import pytest


def driver(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / 'docs/benchmarks'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('nuts_fixed_comparison', folder / 'compare_nuts_fixed_hmc.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('ratios, expected', [
    ([2.] * 12, 2. / 4096.),
    ([2.] * 6 + [.5] * 6, 1.),
    ([1.] * 12, 1.),
    ([2., 2., 1.], .5),
])
def test_sign_test_has_exact_binomial_tail_and_excludes_ties(monkeypatch, ratios, expected):
    assert driver(monkeypatch).sign_test(ratios) == pytest.approx(expected)


def test_observables_retain_non_linear_banana_estimands(monkeypatch, tmp_path):
    module = driver(monkeypatch)
    args = module.benchmark.parser().parse_args(
        ['--output', str(tmp_path / 'unused'), '--device', 'cpu'])
    tf, _, _ = module.benchmark.setup(args)
    values = tf.constant([[[2., 3.], [-1., 4.]]], tf.float64)
    actual = module.observable_tensor(tf, values).numpy().tolist()
    assert actual == [[[2., 3., 4., 9., 12.], [-1., 4., 1., 16., 4.]]]
    assert module.EXPECTED == (0., 0., 1., 1.5, 1.)


def test_unavailable_diagnostics_are_preserved_and_not_dropped(monkeypatch):
    import math
    module = driver(monkeypatch)
    missing = []
    result = module.serialize_missing({'tail_ess': [math.nan, 23.], 'rhat': math.inf}, missing)
    assert result == {'tail_ess': [None, 23.], 'rhat': None}
    assert missing == ['/tail_ess/0', '/rhat']
    assert module.finite_minimum([math.nan, 23.]) is None
    assert module.finite_minimum([None, 23.]) is None
    assert module.finite_minimum([17., 23.]) == 17.
