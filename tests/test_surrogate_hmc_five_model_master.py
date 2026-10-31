"""CPU-only mechanics regression: no scientific or posterior evidence."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_five_model as run
from bayesfilter.testing.surrogate_hmc_scope_tuning import control_candidates


def test_accepted_ot_family_preserves_every_other_control():
    baseline = run.ProposalControls()
    family, candidates = control_candidates(baseline, fixed_ot_five=True)
    assert family == "fixed_epsilon1_steps5"
    assert candidates == [replace(baseline, reset_epsilon=1., reset_steps=5)]
    with pytest.raises(ValueError, match="exclusive"):
        control_candidates(baseline, fixed_ot_five=True, smoother_reset=True)


def test_accepted_ot_family_reaches_scope_tuner(monkeypatch, tmp_path):
    from bayesfilter.testing import surrogate_hmc_scope_tuning as tuning
    calls = []
    monkeypatch.setattr(tuning, "tune", lambda *args, **kwargs: calls.append(kwargs))
    run.tune_filter_scope(object(), tmp_path, fixed_ot_five=True)
    assert calls == [dict(smoother_reset=False, fixed_ot_five=True)]


def fake_case():
    return SimpleNamespace(
        p=2, spec=SimpleNamespace(), horizon=1, force_dtype=tf.float32,
        precision_policy="uniform", tuning_seed_offset=0,
        observations=tf.zeros([1, 1], tf.float32), fixed=(),
        beta=tf.constant([1/3]*3, tf.float32), controls=run.ProposalControls(),
        chart=lambda q: (2*q, tf.fill([2], tf.cast(2., q.dtype))),
        log_prior=lambda q: (-.5*tf.reduce_sum(q*q, axis=-1), -q),
        scope=lambda: {"model": "mechanics_fixture", "particles": 4032})


def bind(monkeypatch, valid=True):
    def program(theta, *unused):
        tf.debugging.assert_equal(tf.shape(theta), [2, 2])
        return {"score": theta + 2.,
                "value": tf.fill([2], tf.constant(987., tf.float32)),
                "valid": tf.fill([2], valid), "steps_completed": tf.fill([2], 1)}
    monkeypatch.setattr(run, "make_batched_score_filter", lambda *args, **kwargs: program)
    monkeypatch.setattr(run, "reference", lambda case: (
        lambda theta: (-1. - .5*tf.reduce_sum(theta**2), tf.constant(True)),
        "independent_unit_test_endpoint"))
    return run.bindings(fake_case())


def test_force_sign_chain_rule_and_independent_mh_endpoint(monkeypatch):
    bound = bind(monkeypatch)
    q = tf.constant([[.2, -.4], [.5, .1], [-.3, .8]], tf.float64)
    force = bound["force"].function(q)
    endpoint = bound["target"].function(q)
    tf.debugging.assert_near(force.force, -3*q-4., atol=1e-6, rtol=1e-6)
    tf.debugging.assert_equal(tf.reduce_all(force.force_valid), True)
    tf.debugging.assert_near(endpoint.potential, 1. + 2.5*tf.reduce_sum(q*q, axis=1))
    assert bound["force"].semantics == "deterministic_position_only_proposal_field"


def test_invalid_analytical_force_is_not_silently_valid(monkeypatch):
    bound = bind(monkeypatch, valid=False)
    out = bound["force"].function(tf.zeros([4, 2], tf.float64))
    tf.debugging.assert_equal(tf.reduce_any(out.force_valid), False)


def test_public_config_adapts_and_allows_20000_initial_retained_draws():
    case = SimpleNamespace(p=3)
    cfg = run.make_tuning_config(case, "test-4032-scope")
    assert cfg.warmup_results == 2000
    assert cfg.warmup_max_results == 10000
    assert cfg.verification_results == 1000
    assert cfg.verification_max_results == 20000
    assert cfg.resume_manifest is None
    assert cfg.check_every_results == 1000
    assert type(cfg).from_payload(cfg.payload()) == cfg
    bounded = replace(cfg, nomination=replace(cfg.nomination, max_step_size=.1))
    assert type(cfg).from_payload(bounded.payload()) == bounded
    with pytest.raises(ValueError):
        replace(cfg, verification_max_results=21000)


def test_flow16_survives_fixed_ot_tuning_and_selection_loading(tmp_path):
    from scripts.run_surrogate_hmc_performance_repair import apply_flow_steps
    from bayesfilter.testing.surrogate_hmc_scope_tuning import control_candidates
    case = fake_case()
    original = case.controls
    assert apply_flow_steps(case, None) is original
    changed = apply_flow_steps(case, 16)
    assert changed.flow_steps == 16
    before, after = run.asdict(original), run.asdict(changed)
    assert {k: v for k, v in before.items() if k != "flow_steps"} == {
        k: v for k, v in after.items() if k != "flow_steps"}
    family, candidates = control_candidates(changed, fixed_ot_five=True)
    assert family == "fixed_epsilon1_steps5"
    assert len(candidates) == 1 and candidates[0].flow_steps == 16
    selected = dict(status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC",
        scope=case.scope(), selected=dict(valid=True,
            controls=run.asdict(candidates[0]), beta=[1/3]*3))
    selected["selection_hash"] = run.hashed(selected)
    path = tmp_path/"selection.json"
    run.save(path, selected)
    case.controls = original
    run.load_selection(case, path)
    assert case.controls.flow_steps == 16
    assert case.controls.reset_epsilon == 1. and case.controls.reset_steps == 5


def test_filter_selection_rejects_scope_change_and_tampering(tmp_path):
    case = fake_case()
    record = {
        "status": "FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC",
        "scope": case.scope(), "selected": {
            "valid": True, "controls": run.asdict(case.controls), "beta": [1/3]*3}}
    record["selection_hash"] = run.hashed(record)
    path = tmp_path/"selection.json"
    run.save(path, record)
    loaded = run.load_selection(case, path)
    assert loaded["scope"]["particles"] == 4032
    assert case.selection_hash == record["selection_hash"]
    record["selected"]["beta"] = [.1, .1, .8]
    run.save(path, record)
    with pytest.raises(ValueError, match="stale or mismatched"):
        run.load_selection(case, path)


@pytest.mark.parametrize("model,policy", [
    ("lgssm", "uniform"), ("sir_d18", "fp64_recursion_fp32_pairs")])
def test_sampling_consumer_restores_current_selection_scope(tmp_path, model, policy):
    from scripts import run_surrogate_hmc_five_model_master as master
    original = SimpleNamespace(name=model)
    restored = SimpleNamespace(name=model, precision_policy=policy, tuning_seed_offset=100)
    calls = []
    def configure(case, selected_policy, offset):
        assert case is original
        assert (selected_policy, offset) == (policy, 100)
        calls.append("configure")
        return restored
    def validate(case, path):
        assert case is restored
        assert calls == ["configure"]
        calls.append("full_scope_validation")
    runtime = SimpleNamespace(hashed=run.hashed, configure_precision=configure,
                              load_selection=validate)
    record = dict(scope=dict(model=model, precision_policy=policy, tuning_seed_offset=100))
    record["selection_hash"] = run.hashed(record)
    path = tmp_path/"selection.json"
    run.save(path, record)
    assert master.restore_selected_scope(runtime, original, path) is restored
    assert calls == ["configure", "full_scope_validation"]
    record["scope"]["tuning_seed_offset"] = 101
    run.save(path, record)
    with pytest.raises(ValueError, match="checksum"):
        master.restore_selected_scope(runtime, original, path)


def test_sampling_scope_rejects_wrong_model_before_configuration(tmp_path):
    from scripts import run_surrogate_hmc_five_model_master as master
    record = dict(scope=dict(model="ksc", precision_policy="uniform", tuning_seed_offset=100))
    record["selection_hash"] = run.hashed(record)
    path = tmp_path/"selection.json"
    run.save(path, record)
    with pytest.raises(ValueError, match="model mismatch"):
        master.restore_selected_scope(SimpleNamespace(hashed=run.hashed),
                                      SimpleNamespace(name="lgssm"), path)


@pytest.mark.parametrize("entry,field", [("stage_worker", "phase"), ("campaign", "through")])
def test_scientific_posterior_rejects_benchmark_priors(entry, field, tmp_path):
    from scripts import run_surrogate_hmc_five_model_master as master
    output = tmp_path / "must_not_launch"
    args = SimpleNamespace(**{field: "inference"},
                           priors="benchmark-normal-v1", output=output)
    with pytest.raises(ValueError, match="model-specific priors"):
        getattr(master, entry)(args)
    assert not output.exists()

def test_public_tuner_to_retained_artifact_roundtrip(monkeypatch, tmp_path):
    """A known Gaussian CPU reference exercises the actual public call chain."""
    case = fake_case()
    def program(theta, *unused):
        tf.debugging.assert_equal(tf.shape(theta), [2, 2])
        return {"score": -.025*theta,
                "value": tf.fill([2], tf.constant(987., tf.float32)),
                "valid": tf.fill([2], True), "steps_completed": tf.fill([2], 1)}
    monkeypatch.setattr(run, "make_batched_score_filter", lambda *args, **kwargs: program)
    monkeypatch.setattr(run, "reference", lambda case: (
        lambda theta: (tf.constant(0., tf.float64), tf.constant(True)),
        "exact_standard_normal_test_target"))
    record = {
        "status": "FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC",
        "scope": case.scope(), "selected": {
            "valid": True, "controls": run.asdict(case.controls), "beta": [1/3]*3}}
    record["selection_hash"] = run.hashed(record)
    path = tmp_path/"selection.json"
    run.save(path, record)
    result = run.inference(case, tmp_path/"inference", path)
    assert result["posterior_sampling_complete"], result
    assert result["no_extra_validation"]
    assert result["all_postwarmup_draws_are_inference_draws"]
    assert result["divergence_count"] == 0
    assert result["force_fallback_count"] == 0
    tf.debugging.assert_less_equal(
        result["diagnostics"]["rank_normalized_split_rhat"]["maximum"], tf.constant(1.01, tf.float64))
    assert (tmp_path/"inference"/"hmc.json").is_file()


def test_prepared_root_reuses_only_matching_frozen_sources_and_data(monkeypatch, tmp_path):
    import hashlib
    import json
    from scripts import run_surrogate_hmc_five_model_master as master
    monkeypatch.setattr(master, "ROOT", tmp_path)
    monkeypatch.setattr(master, "FOUR", tmp_path/"four")
    monkeypatch.setattr(master, "RANGE", tmp_path/"range")
    source = tmp_path/"source.py"
    source.write_text("alpha")
    prepared = tmp_path/"prepared"
    models = {}
    for model in master.MODELS:
        data = (master.RANGE/"data/1704-t20.json" if model == "range_bearing"
                else master.FOUR/"inputs"/f"{model}-d26100831/dataset.json")
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(model)
        endpoint = prepared/model/"endpoint-audit"/"endpoint-audit.json"
        master.write(endpoint, dict(status="PASS_ENDPOINT_MECHANICS",
                                    valid=True, fixed_input_replay_equal=True, value=-10.))
        selection = prepared/model/"filter-tuning"/"filter-selection.json"
        row = {
            "status": "FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC",
            "scope": dict(model=model, particles=4032, dtype="float32", tf32=False,
                          horizon=master.HORIZONS[model], data_sha256=master.digest(data),
                          source_sha256={"source.py": master.digest(source)}),
            "selected": {"valid": True}}
        row["selection_hash"] = hashlib.sha256(
            json.dumps(row, sort_keys=True).encode()).hexdigest()
        master.write(selection, row)
        models[model] = {
            "endpoint-audit": dict(status="PASS_ENDPOINT_MECHANICS", result=str(endpoint)),
            "filter-tuning": dict(status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC",
                                  result=str(selection))}
    master.write(prepared/"master.json", dict(
        status="PREPARATION_COMPLETE_HMC_NOT_RUN", through_phase="filter-tuning",
        particles=4032, dtype="float32", tf32=False, horizons=master.HORIZONS,
        models=models))
    previous, selections = master.prepared_selections(prepared)
    assert previous["through_phase"] == "filter-tuning"
    assert set(selections) == set(master.MODELS)
    with pytest.raises(ValueError, match="prepared prior changed"):
        master.prepared_selections(prepared, master.LITERATURE_PRIOR)
    endpoint = prepared/master.MODELS[0]/"endpoint-audit"/"endpoint-audit.json"
    good_endpoint = master.read(endpoint)
    master.write(endpoint, dict(good_endpoint, fixed_input_replay_equal=False))
    with pytest.raises(ValueError, match="prepared endpoint failed validity/replay"):
        master.prepared_selections(prepared)
    master.write(endpoint, good_endpoint)
    source.write_text("changed")
    with pytest.raises(ValueError, match="prepared source changed"):
        master.prepared_selections(prepared)


def test_campaign_reserves_model_budgets_and_reports_incomplete(monkeypatch, tmp_path):
    from scripts import run_surrogate_hmc_five_model_master as master
    # Scheduling-only fixture: every worker is stubbed below; no posterior runs.
    # The real entry-point prior guard is exercised separately above.
    monkeypatch.setattr(master, "validate_stage_prior", lambda phase, prior: None)
    previous = {"models": {
        m: {phase: {"status": next(iter(master.PASS_STATUS[phase])), "result": "prepared"}
            for phase in ("endpoint-audit", "filter-tuning")}
        for m in master.MODELS}}
    monkeypatch.setattr(master, "prepared_selections",
                        lambda root, prior: (previous, {m: tmp_path/"selection.json" for m in master.MODELS}))
    monkeypatch.setattr(master, "static_audit", lambda: {"status": "test_only"})
    def worker(command, **kwargs):
        phase = command[command.index("--phase")+1]
        output = run.Path(command[command.index("--output")+1])
        result = (dict(status="PASS_HMC_MECHANICS_ONLY", seconds_per_four_chain_transition=.01)
                  if phase == "mechanics" else dict(status="HMC_TUNING_REPAIR_REQUIRED"))
        master.write(output/master.RESULT_NAMES[phase], result)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess, "run", worker)
    args = SimpleNamespace(through="inference", priors="benchmark-normal-v1", budget_seconds=1000.,
                           prepared_root=tmp_path/"prepared", output=tmp_path/"run", gpu="1")
    assert master.campaign(args) == 2
    result = master.read(args.output/"master.json")
    assert result["status"] == "INCOMPLETE_SEE_MODEL_STATUS"
    assert result["completed_models"] == []
    first = result["models"]["ksc"]["inference"]
    assert 190. < first["stage_wall_budget_seconds"] <= 200.
    assert all(result["models"][m]["inference"]["status"] == "HMC_TUNING_REPAIR_REQUIRED"
               for m in master.MODELS)
    next_phase = master.read(args.output/"next-phase-plan.json")
    assert next_phase["automatic_continuation"] is False
    assert "incomplete" in next_phase["action"]
    assert "finished" not in next_phase["action"]
