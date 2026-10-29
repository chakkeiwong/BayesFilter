"""Analytic reference, uncertainty multiplicity, restart and budget regressions."""
import json

import pytest
import tensorflow as tf

from bayesfilter.testing import surrogate_hmc_gaussian_control as control
from bayesfilter.inference import surrogate_hmc_program as program
from bayesfilter.inference.surrogate_hmc_equivalence import assess_equivalence


def test_all_fields_share_per_chain_exact_endpoint_and_rotation_is_not_gradient():
    q = control.initial_positions()
    fields, endpoints = {}, []
    for arm in control.ARMS:
        force, target, binding, adapter = control.bind_control(arm)
        fields[arm] = force.function(q)
        endpoints.append(target.function(q))
        assert binding.target_scope == adapter.target_scope
    exact = q*tf.constant(control.PRECISION, tf.float64)
    tf.debugging.assert_near(fields["exact"], exact, atol=1e-14)
    tf.debugging.assert_near(fields["half"], exact/2, atol=1e-14)
    tf.debugging.assert_equal(fields["zero"], tf.zeros_like(q))
    expected_rotation = .2*tf.stack([-q[:, 1], q[:, 0]-q[:, 2], q[:, 1]], axis=-1)
    tf.debugging.assert_near(fields["rotational"]-exact, expected_rotation, atol=1e-14)
    for value in endpoints:
        tf.debugging.assert_near(value, tf.reduce_sum(q*exact, axis=-1)/2, atol=1e-14)
    force, _, _, _ = control.bind_control("rotational")
    # Finite differences of this linear field are its exact constant Jacobian.
    zero = tf.zeros([4, 3], tf.float64)
    first = force.function(tf.ones([4, 1], tf.float64)*tf.constant([[1., 0., 0.]], tf.float64))
    second = force.function(tf.ones([4, 1], tf.float64)*tf.constant([[0., 1., 0.]], tf.float64))
    assert float(first[0, 1]-second[0, 0]) == pytest.approx(.4)
    tf.debugging.assert_equal(force.function(zero), zero)


def test_reference_function_order_and_scaling():
    w = tf.constant([[-2., 0., 2.]], tf.float64)
    q = tf.broadcast_to(w/tf.sqrt(tf.constant(control.PRECISION, tf.float64)), [250, 4, 3])
    standardized, functions = control.functions_for_chunk(q)
    tf.debugging.assert_near(standardized[0, 0], w[0])
    expected = tf.constant([-2., 0., 2., 4., 0., 4., 1., 0., 0., 1., 1., 0., 1., 1., 0.], tf.float64)
    tf.debugging.assert_equal(functions[0, 0], expected)
    reference = control.reference_contract()
    assert len(reference["names"]) == 15
    assert reference["expectations"] == [0.]*3+[1.]*3+[.1]*3+[.5]*3+[.9]*3


def test_generalized_equivalence_preserves_four_arm_multiplicity():
    chunks = []
    parameters = []
    for seed in range(4):
        samples = tf.random.stateless_normal([250, 4, 3], [701, seed], dtype=tf.float64)/tf.sqrt(
            tf.constant(control.PRECISION, tf.float64))
        w, features = control.functions_for_chunk(samples)
        parameters.append(w)
        chunks.append(features)
    reference = control.reference_contract()
    args = (tf.transpose(tf.concat(parameters, axis=0), [1, 0, 2]),
            tf.transpose(tf.concat(chunks, axis=0), [1, 0, 2]),
            *[tf.constant(reference[key], tf.float64) for key in ("expectations", "allowance", "margins")])
    ordinary = assess_equivalence(*args, parameter_dimension=3, function_count=15, analytic_reference=True)
    family = assess_equivalence(*args, parameter_dimension=3, function_count=15,
                                analytic_reference=True, comparison_families=4)
    assert family["family_functions"] == 15 and family["planned_looks"] == 2
    assert family["reference_type"] == "analytic" and len(family["function_rhat"]) == 15
    assert all(a < b for a, b in zip(family["lower"], ordinary["lower"]))
    assert all(a > b for a, b in zip(family["upper"], ordinary["upper"]))
    assert family["difference"] == ordinary["difference"]
    with pytest.raises(ValueError, match="dimensions"):
        assess_equivalence(*args)


def test_cost_forecast_uses_slowest_cost_and_full_transition_allowance():
    timings = {arm: dict(compile=dict(wall_seconds=2., cpu_worker_seconds=3.), steady=[
        dict(wall_seconds=.256, cpu_worker_seconds=.512),
        dict(wall_seconds=.512, cpu_worker_seconds=.256)]) for arm in control.ARMS}
    value = control.cost_forecast(timings)
    assert value["total"]["wall_seconds"] == pytest.approx(4*1.5*(5216*.002+2+90))
    assert value["affordable"]
    timings["zero"]["steady"][0]["wall_seconds"] = 100.
    assert not control.cost_forecast(timings)["affordable"]
    timings["zero"]["steady"][0]["wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        control.cost_forecast(timings)


def save_history(root, report, suffix="a"):
    path = root/f"docs/plans/artifacts/surrogate-hmc-all-models-{suffix}/P3-C01-inference/report.json"
    program.write_json(path, dict(provenance=dict(scope=control.SCOPE), **report))
    return path


def test_partial_retained_control_is_reported_without_more_draws(tmp_path, monkeypatch):
    path = save_history(tmp_path, dict(arms={"exact": dict(chunks=[{}], retained_draws=250)}))
    history = program.gaussian_attempt_history(tmp_path)
    assert not history["P3-C01-inference-job"]["closed"]
    jobs = [dict(name="P3-C01-qualify-job", depends_on=[]),
            dict(name="P3-C01-inference-job", depends_on=["P3-C01-qualify-job"])]
    jobs = program.preserve_retained_attempts(jobs, history, all_stages=True)
    monkeypatch.setattr(program, "run_job", lambda *args, **kwargs: pytest.fail("no more draws"))
    results = program.execute_job_graph(jobs, tmp_path/"out", {})
    assert results["P3-C01-inference-job"]["status"] == "HISTORICAL_RESULT_REPORTED"
    assert history["P3-C01-inference-job"]["report"] == str(path)
    save_history(tmp_path, dict(closed=True), suffix="b")
    with pytest.raises(ValueError, match="multiple"):
        program.gaussian_attempt_history(tmp_path)


def test_tuning_infrastructure_failure_can_retry_but_candidate_rejection_cannot(tmp_path):
    path = save_history(tmp_path, dict(status="INFRASTRUCTURE_FAILURE", arms={"exact": dict(status="TUNING")}))
    assert program.gaussian_attempt_history(tmp_path) == {}
    program.write_json(path, dict(provenance=dict(scope=control.SCOPE), arms={"exact": dict(status="TUNING_REJECTED")}))
    assert program.gaussian_attempt_history(tmp_path)


def test_control_scope_is_independent_and_retry_budget_is_cumulative(tmp_path, monkeypatch, capsys):
    record = tmp_path/"docs/plans/artifacts/surrogate-hmc-all-models-first/accounting.json"
    program.write_json(record, dict(charged=dict(gpu_seconds=20., cpu_worker_seconds=30.), jobs=[dict(
        command=["python", "scripts/qualify_surrogate_hmc_gaussian_controls.py"],
        gpu_seconds=20., cpu_worker_seconds=30.)]))
    monkeypatch.setattr(program, "ROOT", tmp_path)
    assert program.main(["--stage", "all", "--scope", "C01", "--dry-run", "--output", str(tmp_path/"out")]) == 0
    launch = json.loads(capsys.readouterr().out)
    assert set(launch["scope_cards"]) == {"C01"}
    assert [row["name"] for row in launch["jobs"]] == ["P3-C01-qualify-job", "P3-C01-inference-job"]
    assert launch["jobs"][0]["timeout"] == 280.
    assert launch["jobs"][0]["cpu_allowance"] == 370.
    assert launch["jobs"][1]["depends_on"] == ["P3-C01-qualify-job"]
    assert not launch["scope_card"]["dctr_inference_evidence"]


def test_master_reports_closed_control_without_polluting_model_comparisons(tmp_path, monkeypatch, capsys):
    comparisons = {arm+"/analytic": "INCONCLUSIVE" for arm in control.ARMS}
    save_history(tmp_path, dict(closed=True, status="COMPLETED", comparisons=comparisons))
    history = program.gaussian_attempt_history(tmp_path)
    monkeypatch.setattr(program, "gaussian_attempt_history", lambda root: history)
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    monkeypatch.setattr(program, "run_job", lambda *a, **kw: pytest.fail("completed control must not rerun"))
    output = tmp_path/"report-only"
    assert program.main(["--scope", "C01", "--stage", "all", "--output", str(output)]) == 2
    capsys.readouterr()
    decision = json.loads((output/"decision.json").read_text())
    assert decision["scope_results"]["C01"]["comparisons"] == comparisons
    assert set(decision["comparisons"].values()) == {"NOT_RUN"}
    assert not decision["scope_results"]["C01"]["dctr_inference_evidence"]
    assert not decision["all_models_validated"]
    register = json.loads((output/"model-register.json").read_text())
    assert register["families"][0]["inference_status"] == "NOT_RUN"


def test_qualification_forecast_rejects_stale_sources_and_over_budget(tmp_path, monkeypatch):
    source = tmp_path/"field.py"
    source.write_text("frozen field")
    report = dict(status="QUALIFIED_AFFORDABLE", provenance=dict(source_hashes={"field.py": program.file_hash(source)}),
                  forecast=dict(total=dict(wall_seconds=2., cpu_worker_seconds=3.)))
    program.write_json(tmp_path/"report.json", report)
    monkeypatch.setattr(program, "ROOT", tmp_path)
    assert program.gaussian_qualification_forecast(tmp_path, dict(timeout=2., cpu_allowance=3.)) == report["forecast"]
    with pytest.raises(ValueError, match="allocation"):
        program.gaussian_qualification_forecast(tmp_path, dict(timeout=1., cpu_allowance=3.))
    source.write_text("changed field")
    with pytest.raises(ValueError, match="drift"):
        program.gaussian_qualification_forecast(tmp_path, dict(timeout=2., cpu_allowance=3.))
