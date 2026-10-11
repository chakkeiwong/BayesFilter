"""CPU full-model preparation and price wiring; no posterior claims."""
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
from bayesfilter.testing.inference_validation import modern_nonlinear
from bayesfilter.testing.inference_validation.modern_prepared_nonlinear import prepared_binding
from scripts import price_mams_adaptation_full_models as price


def geometry_fixture(directory, fixture, *, stale=False):
    dimension = fixture.adapter.parameter_dim
    factor = .1*tf.eye(dimension, dtype=tf.float64)
    directory.mkdir()
    (directory / "geometry.json").write_text(json.dumps(dict(status="local_factor_proposed",
        center=[0.]*dimension, factor=factor.numpy().tolist(),
        covariance=(factor@factor).numpy().tolist())))
    source = Path(__file__).resolve().parents[1] / "bayesfilter/inference/hmc_candidate_set_tuning.py"
    (directory / "design.json").write_text(json.dumps(dict(target=fixture.target_id,
        metadata=fixture.metadata, data=fixture.observations.numpy().tolist(),
        source_closure={str(source): "old-source-digest"} if stale else {})))


def test_old_geometry_is_explicit_warm_start_for_current_mams_target(tmp_path, monkeypatch):
    original = modern_nonlinear.make_nonlinear_comparison_target
    monkeypatch.setattr(modern_nonlinear, "make_nonlinear_comparison_target",
        lambda target, *, jit_compile: original(target, horizon=3, jit_compile=jit_compile))
    fixture = original("NL-PP-UKF", horizon=3, jit_compile=False)
    directory = tmp_path / "geometry"
    geometry_fixture(directory, fixture, stale=True)
    with pytest.raises(ValueError, match="numerical source changed"):
        prepared_binding("NL-PP-UKF", directory, geometry_kind="local", seed=17, use_xla=False)
    binding, _, metadata = prepared_binding("NL-PP-UKF", directory,
        geometry_kind="local", seed=17, use_xla=False,
        geometry_source_policy="warm_start_only",
        transition_profile=HMCTransitionProfile("mams", "uniform_duration"),
        epsilon_domain=(.001, 4.), max_repairs_per_family=1)
    assert metadata["geometry_source_policy"] == "warm_start_only"
    assert len(metadata["geometry_source_changes"]) == 1
    assert metadata["global_coverage_established"] is False
    assert binding.scope.transition_profile.family == "mams"
    assert binding.scope.epsilon_domain == (.001, 4.)
    assert binding.scope.max_repairs_per_family == 1


@pytest.mark.parametrize("target", price.TARGETS)
def test_full_model_price_runs_shared_discarded_proposer_and_native_path(tmp_path, monkeypatch, target):
    fixture = modern_nonlinear.make_nonlinear_comparison_target(target, jit_compile=False)
    directory = tmp_path / "geometry"
    geometry_fixture(directory, fixture)
    monkeypatch.setattr(price, "geometry_directory", lambda root, target: directory)
    original = price.pilot_config
    monkeypatch.setattr(price, "pilot_config", lambda binding, seed: replace(
        original(binding, seed), initial_epsilon=.03, initial_mean_duration=.09,
        max_requested_duration=.18, max_steps_cap=32,
        epsilon_adaptation_windows=1, probe_transitions=1,
        fixed_settling_transitions=1, fixed_window_transitions=8,
        max_autocorrelation_lag=2, duration_factors=(1.,), max_candidates=1,
        max_pilot_gradient_work=100000, max_pilot_wall_seconds=300.))
    output = tmp_path / "price"
    result = price.run_case(target, output, 913, time.monotonic()+300,
        use_xla=False, source_hashes={}, geometry_root=tmp_path)
    assert result["status"] == "completed"
    assert result["tuning_artifact_authority"] is False
    assert result["posterior_convergence_authority"] is False
    assert len(result["rows"]) == 1
    assert result["rows"][0]["status"] in {"healthy_short_path", "candidate_veto"}
    assert result["rows"][0]["metadata"]["value_score_row_evaluations"] > 0
    design = json.loads((output / "design.json").read_text())
    assert design["horizon"] == (100 if target == "NL-STR" else 20)
    assert design["parameter_dim"] == fixture.adapter.parameter_dim
    pilot = json.loads((output / "pilot/pilot-result.json").read_text())
    assert result["rows"][0]["seed"] not in [row["seed"] for row in pilot["seeds"]]


def test_supervisor_finalizer_preserves_all_priced_and_unstarted_rows(tmp_path):
    output = tmp_path / "run"
    started = output / "NL-PP-UKF"
    pilot = started / "pilot"
    pilot.mkdir(parents=True)
    (started / "progress.json").write_text(json.dumps(dict(target="NL-PP-UKF",
        status="pricing", rows=[dict(status="price_running", candidate_id="one"),
                                dict(status="price_not_started", candidate_id="two")])))
    (pilot / "pilot-progress.json").write_text(json.dumps(dict(status="completed",
        actual_gradient_work=17, reserved_gradient_work=24, candidate_rows=[{"id": "one"}])))
    result = price.finalize_prices(output, ("NL-PP-UKF", "NL-SIR"),
                                   status="supervisor_budget_exhausted")
    assert len(result["rows"]) == 2
    assert [row["status"] for row in result["rows"][0]["rows"]] == [
        "price_running_supervisor_budget_exhausted",
        "price_not_started_supervisor_budget_exhausted"]
    assert result["rows"][0]["pilot_actual_work"] == 17
    assert result["rows"][1]["status"] == "not_started_supervisor_budget_exhausted"
    assert result["tuning_artifact_authority"] is False
