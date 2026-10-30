"""Scope isolation, finite-force vetoes and M13 master-to-worker wiring."""
import json
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.testing import surrogate_hmc_m13_controls as m13


def test_independent_partitions_keep_the_same_law_and_exact_saved_data():
    fixtures = [m13.partition_fixture(p) for p in m13.PARTITIONS]
    assert len({f.observation_checksum for f in fixtures}) == 3
    assert len({f.model_checksum for f in fixtures}) == 3
    assert [f.fixture_generation_seed for f in fixtures] == [1701, 2701, 3701]
    for f in fixtures:
        for key in ("A", "Q", "R", "m0", "P0"):
            tf.debugging.assert_equal(getattr(f, key), getattr(fixtures[0], key))
        tf.debugging.assert_equal(f.observations, tf.cast(tf.cast(f.observations, tf.float32), tf.float64))
        assert f.horizon == 20 and f.state_dim == 4 and f.obs_dim == 2
    again = m13.partition_fixture("calibration")
    tf.debugging.assert_equal(again.observations, fixtures[0].observations)
    assert again.model_definition() == fixtures[0].model_definition()


def _output(value=1., valid=True, score_finite=True):
    flag = tf.fill([9], valid)
    score = tf.fill([9, 2], value if score_finite else float("nan"))
    status = dict(value_valid=flag, force_valid=flag, program_valid=flag,
                  outside_declared_support=tf.zeros([9], tf.bool))
    for key in ("filtering_mean", "filtering_covariance", "post_reset_mean", "post_reset_covariance"):
        status[key] = tf.fill([9, 2, 2], value)
    return tf.fill([9], value), score, status


@pytest.mark.parametrize("valid,score_finite", [(False, True), (True, False)])
def test_nominal_invalidity_or_unflagged_nonfinite_cannot_pass(valid, score_finite):
    bad = dict(nominal=_output(valid=valid, score_finite=score_finite), stress=_output())
    good = dict(nominal=_output(), stress=_output())
    assert not m13.compare_pair(bad, good)["non_harm_passed"]


def test_stronger_protection_cannot_create_new_stress_failure():
    strong = dict(nominal=_output(), stress=_output(valid=False))
    weak = dict(nominal=_output(), stress=_output())
    row = m13.compare_pair(strong, weak)
    assert row["healthy_nominal"] and row["stress_safe"]
    assert not row["no_new_stress_failure"] and not row["non_harm_passed"]


@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("refined,bounded,particle_count", [(False, False, 256), (True, False, 256), (False, True, 256), (False, True, 4096)])
def test_stage_binds_selection_and_never_evaluates_claim(tmp_path, monkeypatch, stable, refined, bounded, particle_count):
    calls = []
    def factory(fixture, **kwargs):
        calls.append((fixture.fixture_generation_seed, kwargs))
        multiplier = kwargs["controls"]["correction_lm_damping"]/1e-3
        def compile(batch, **flags):
            assert batch == 9 and flags == dict(filter_moments=True, all_correction_diagnostics=True)
            value = 1.+(.01 if stable else 10.)*multiplier
            return lambda theta: _output(value=value)
        return SimpleNamespace(manifest={"source_sha256": {}, "dtype": "float32"}, compile=compile)
    monkeypatch.setattr(m13, "make_range_bearing_target", factory)
    result = m13.controls_stage(tmp_path/"report.json", refined=refined, bounded=bounded, particle_count=particle_count)
    offset = 3 if particle_count == 4096 else int(refined)+2*int(bounded)
    assert len(calls) == 8 and {seed for seed, _ in calls} == {1701+offset, 2701+offset}
    assert all(k["dtype"] == tf.float32 and k["particle_count"] == particle_count for _, k in calls)
    assert {k["design_partition"] for _, k in calls} == {3*offset, 1+3*offset}
    assert (tmp_path/"claim/data.json").exists()
    assert not result["inference_launched"]
    assert result["posterior_comparisons"] == "NOT_RUN"
    assert result["qualified"] is stable
    if stable:
        assert result["selected_multiplier"] == .1
        selection = json.loads((tmp_path/"selection.json").read_text())
        assert selection["scope"] == m13.m13_scope(refined=refined, bounded=bounded, particle_count=particle_count)
        assert selection["exact_score_admission"] is False
        assert selection["default_readiness"] is False
        assert set(selection["partition_data"]) == {"calibration", "validation"}
        if refined or bounded:
            assert all(k["controls"]["reset_ridge"] == 1e-5 for _, k in calls)
            assert selection["controls"] == (m13.fixed_protection_controls(.1) if bounded else m13.fixed_reset_controls(.1))
            selection["controls"]["reset_ridge"] = 1e-6
            (tmp_path/"selection.json").write_text(json.dumps(selection))
            with pytest.raises(ValueError, match="differs from the measured curve"):
                m13.refined_selection(tmp_path/"selection.json", scope=selection["scope"],
                                      sources=selection["source_hashes"])
    else:
        assert not (tmp_path/"selection.json").exists()
        assert not result["continuation_veto"]
        assert result["status"] == "M13_CONTROL_REPAIR_REQUIRED"


@pytest.mark.parametrize("name", ["m13-controls", "m13-controls-localize", "m13-controls-refine", "m13-controls-bounded"])
def test_master_reserves_new_stage_without_reopening_lg1(name):
    from bayesfilter.inference import surrogate_hmc_closure_program as program
    jobs = program.jobs(name)
    assert [row["name"] for row in jobs] == ["review", name]
    assert jobs[1]["device"] == "gpu"
    assert jobs[1]["plan"] == m13.PLAN
    assert "tests/test_surrogate_hmc_m13_controls.py" in jobs[0]["command"]
    cap = program.remaining_pass_reservation([], "m13-controls")
    assert sum(j["cpu"] for j in jobs) <= cap["cpu_worker_seconds"]
    assert jobs[1]["timeout"] <= cap["gpu_seconds"]
    with pytest.raises(ValueError):
        program.jobs("m13-controls,m13-consumer")


@pytest.mark.parametrize("stage_name", ["m13-controls", "m13-controls-localize", "m13-controls-refine", "m13-controls-bounded"])
def test_worker_dispatches_to_scoped_controls(tmp_path, monkeypatch, stage_name):
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    calls = []
    def stage(output, **kwargs):
        calls.append((output, kwargs))
        return dict(status="WIRING_CHECK_ONLY", qualified=False)
    monkeypatch.setattr(m13, "controls_stage", stage)
    monkeypatch.setattr(m13, "localization_stage", stage)
    output = tmp_path/"report.json"
    worker.main(["--stage", stage_name, "--device", "cpu", "--output", str(output)])
    assert calls == [(output, {"bounded": True} if stage_name == "m13-controls-bounded" else {"refined": True} if stage_name == "m13-controls-refine" else {})]
    payload = json.loads(output.read_text())
    assert payload["status"] == "WIRING_CHECK_ONLY"
    assert payload["gpu_intentionally_hidden"]


def test_localization_changes_only_the_named_protection_group():
    arms = m13.localization_candidates()
    strong, weak = arms["strong"], arms["weak"]
    expected = {"reset_ridge": {"reset_ridge"},
        "lm": {"correction_lm_damping", "correction_lm_scale_floor"},
        "trust_caps": {"correction_trust_radius", "pairwise_rms_cap", "coordinate_cap"}}
    for name, keys in expected.items():
        settings = arms["restore_"+name]
        assert {k for k in settings if settings[k] != weak[k]} == keys
        assert all(settings[k] == strong[k] for k in keys)
    assert all(c["reset_policy"] == "contract_e" for c in arms.values())


def test_localization_retains_failure_and_cannot_issue_selection(tmp_path, monkeypatch):
    calls = []
    def factory(fixture, **kwargs):
        calls.append((fixture.fixture_generation_seed, kwargs))
        valid = kwargs["controls"]["reset_ridge"] >= 5e-6
        return SimpleNamespace(manifest={"source_sha256": {}},
            compile=lambda *args, **flags: lambda theta: _output(valid=valid))
    monkeypatch.setattr(m13, "make_range_bearing_target", factory)
    result = m13.localization_stage(tmp_path/"report.json")
    assert len(calls) == 5 and {seed for seed, _ in calls} == {2701}
    assert result["diagnostic_completed"] and not result["qualified"]
    assert not result["arms"]["weak"]["nominal"]["all_healthy"]
    assert result["arms"]["restore_reset_ridge"]["nominal"]["all_healthy"]
    assert not (tmp_path/"selection.json").exists()


def test_refined_partitions_are_fresh_without_changing_law_or_claim():
    scope = m13.m13_scope(refined=True)
    assert scope["attempt"] == 2
    assert scope["partition_seeds"]["claim"] == dict(data_seed=3701, design_partition=2)
    for part in ("calibration", "validation", "claim"):
        old = m13.partition_fixture(part)
        new = m13.partition_fixture(part, attempt=2)
        assert (new.observation_checksum == old.observation_checksum) == (part == "claim")
        for key in ("A", "Q", "R", "m0", "P0"):
            tf.debugging.assert_equal(getattr(old, key), getattr(new, key))


def test_bounded_family_changes_only_lm_and_preserves_measured_caps():
    base = m13.protection_controls(1.)
    for multiplier in m13.MULTIPLIERS[1:]:
        setting = m13.fixed_protection_controls(multiplier)
        assert {k for k in base if base[k] != setting[k]} == {
            "correction_lm_damping", "correction_lm_scale_floor"}
    scope = m13.m13_scope(bounded=True)
    assert scope["attempt"] == 3
    assert scope["partition_seeds"]["calibration"] == dict(data_seed=1703, design_partition=6)
    assert scope["partition_seeds"]["validation"] == dict(data_seed=2703, design_partition=7)
    assert scope["partition_seeds"]["claim"] == dict(data_seed=3701, design_partition=2)


def test_recovery_scope_is_fresh_and_uses_repository_chunk_policy():
    scope = m13.m13_scope(bounded=True, particle_count=4096)
    assert scope["attempt"] == 4 and scope["particle_count"] == 4096
    assert scope["chunk_extent"] == 2048
    assert scope["partition_seeds"] == {
        "calibration": dict(data_seed=1704, design_partition=9),
        "validation": dict(data_seed=2704, design_partition=10),
        "claim": dict(data_seed=3701, design_partition=2)}
    with pytest.raises(ValueError, match="scope"):
        m13.m13_scope(particle_count=4096)
    with pytest.raises(ValueError, match="scope"):
        m13.m13_scope(bounded=True, particle_count=1024)


def test_recovery_master_to_consumer_wiring(tmp_path, monkeypatch):
    from bayesfilter.inference import surrogate_hmc_closure_program as program
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    jobs = program.jobs("m13-controls-recovery")
    assert [j["name"] for j in jobs] == ["review", "m13-controls-recovery"]
    assert sum(j["cpu"] for j in jobs) <= 2000
    assert jobs[1]["plan"] == program.M13_CONTROLS_RECOVERY_PLAN
    seen = []
    monkeypatch.setattr(m13, "controls_stage", lambda output, **kw:
        seen.append((output, kw)) or dict(status="WIRING_ONLY", qualified=False))
    output = tmp_path/"report.json"
    worker.main(["--stage", "m13-controls-recovery", "--device", "cpu", "--output", str(output)])
    assert seen == [(output, dict(bounded=True, particle_count=4096))]
