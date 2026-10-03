import os
import subprocess
import sys

import pytest

from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign, seed_for
from bayesfilter.testing.inference_validation.execution import plan_suite


def test_design_round_trip_and_seed_separation(design):
    d = design()
    assert ValidationDesign.from_payload(d.payload()) == d
    assert len({seed_for(7, i, stage) for i in range(12) for stage in ("data", "fit", "reference")}) == 36
    assert d.identity != design(seed=81).identity


@pytest.mark.parametrize("engine,target,route,control", [
    ("invariance", "gaussian", "ordinary", "baseline"),
    ("sbc", "funnel", "ordinary", "baseline"),
    ("mechanics", "gaussian", "frozen", "omit_jacobian"),
    ("stopping", "gamma", "reference", "baseline"),
    ("mechanics", "gaussian", "ordinary", "baseline"),
    ("search", "gaussian", "controller", "wrong_energy"),
    ("invariance", "gamma", "frozen", "two_cycle"),
    ("power", "gaussian", "reference", "baseline"),
])
def test_no_silent_route_or_mutation_fallback(design, engine, target, route, control):
    with pytest.raises(ValueError):
        design(engine, target, route, control)


def test_sbc_rejects_reference_start_and_coarse_null(design):
    with pytest.raises(ValueError, match="starts forbidden"):
        design(engine="sbc", scenario=ScenarioSpec("normal_conjugate", "ordinary", start="reference"))
    with pytest.raises(ValueError, match="resolution"):
        design("invariance", null_draws=39)


def test_suite_keeps_missing_consumer_and_enforces_profile(design):
    d = design("accuracy", "macrofinance", "external")
    suite = dict(schema="bayesfilter.inference_validation_suite.v1", suite_id="unit",
        profile="consumer", profiles={"consumer": ["accuracy"]}, designs=[d.payload()])
    assert plan_suite(suite)["jobs"][0]["availability"] == "unavailable"
    suite["profiles"]["consumer"] = ["mechanics"]
    with pytest.raises(ValueError, match="profile"):
        plan_suite(suite)


def test_planning_accepts_simplex_model_quantities(design):
    d = design("accuracy", "dirichlet", "prepared")
    suite = dict(schema="bayesfilter.inference_validation_suite.v1", suite_id="simplex",
        profile="numerical", profiles={"numerical": ["accuracy"]}, designs=[d.payload()])
    job = plan_suite(suite)["jobs"][0]
    assert job["availability"] == "ready"


def test_planning_import_is_framework_free():
    env = dict(os.environ, BAYESFILTER_PRELOAD_CUSTOM_OP="0", CUDA_VISIBLE_DEVICES="-1")
    subprocess.run([sys.executable, "-c", "import sys; from bayesfilter.testing.inference_validation.execution import plan_suite; assert 'tensorflow' not in sys.modules"], env=env, check=True)
