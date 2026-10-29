"""CPU-only checks of the fresh runtime comparison's evidence handling."""

import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    specification = importlib.util.spec_from_file_location(
        "sir_n1008_runtime", scripts / "benchmark_sir_n1008_runtime.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", (None, "score", "moment", "flag", "invalid", "diagnostic"))
def test_changed_scientific_outputs_veto_speed_claim(diagnostic, failure):
    baseline = dict(value=10., score=2., estimate=[1., 2.], raw_estimate=[3., 4.], loss=1., raw_blocks=[1., 2.],
                    valid=True, checks=dict(flow_valid=True),
                    diagnostics=dict(margin=[.5], flow_determinant_group_size=[1.]))
    candidate = copy.deepcopy(baseline)
    candidate["diagnostics"]["flow_determinant_group_size"] = [24.]
    if failure == "score":
        candidate["score"] += .01
    elif failure == "moment":
        candidate["estimate"][0] += .01
    elif failure == "flag":
        candidate["checks"]["flow_valid"] = False
    elif failure == "invalid":
        candidate["valid"] = False
    elif failure == "diagnostic":
        candidate["diagnostics"]["margin"] = [.4]
    assert diagnostic.compare_outputs(baseline, candidate, diagnostics=True)["passed"] == (failure is None)


def test_bootstrap_resamples_whole_pairs(diagnostic):
    result = diagnostic.paired_interval([1., 10., 100.], [2., 20., 200.])
    assert result["ratio"] == 2.
    assert result["interval95"] == [2., 2.]
    with pytest.raises(ValueError):
        diagnostic.paired_interval([1., 2.], [2.])


def test_disqualified_particle_count_cannot_launch(diagnostic, monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark", "--root", "/tmp/unused-sir-test", "--particles", "64"])
    with pytest.raises(SystemExit) as error:
        diagnostic.main()
    assert error.value.code == 2
