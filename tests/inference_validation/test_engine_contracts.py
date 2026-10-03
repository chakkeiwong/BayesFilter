import copy
from pathlib import Path

import numpy as np
import pytest

from bayesfilter.testing.inference_validation.engines import pipeline, diagnostics, sbc, invariance
from bayesfilter.testing.inference_validation.storage import write_tensor, read_tensor, read_json, write_json
from bayesfilter.testing.inference_validation.references.external import load_reference
from bayesfilter.testing.inference_validation.controls import response


def test_controller_oracle_detects_lost_members_pairs_and_work(design, tmp_path):
    baseline = pipeline.controller_experiment(design("search", route="controller"), tmp_path / "baseline")
    assert baseline["finding"] == "inventory_passed", baseline["failures"]
    for control in ("drop_candidate", "cross_l_epsilon", "lost_chunk"):
        result = pipeline.controller_experiment(design("search", route="controller", control=control), tmp_path / control)
        assert result["finding"] == "inventory_discrepancy"
    corrupt = copy.deepcopy(baseline["native_payload"])
    corrupt["verification_receipts"] = ()
    assert pipeline.check_inventory(corrupt)["finding"] == "inventory_discrepancy"


def test_tensor_checksum_shape_and_immutable_observations(tmp_path):
    path = write_tensor(tmp_path / "x.tensor", np.ones((3, 4, 2)))
    np.testing.assert_array_equal(read_tensor(path), np.ones((3, 4, 2)))
    with pytest.raises(ValueError, match="replace"):
        write_tensor(path, np.zeros((3, 4, 2)))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_tensor(path)


@pytest.mark.parametrize("regime", ["stationary", "transient", "constant", "missed_mode"])
def test_runtime_rhat_against_independent_rank_reference(design, tmp_path, regime):
    target = "mixture" if regime == "missed_mode" else "gaussian"
    result = diagnostics.run(design("stopping", target, "reference", replications=2,
        options={"array_regime": regime}), tmp_path)
    assert not result["actual_controller_stopping_test"]
    assert all(r["availability_matches"] for r in result["rows"])
    assert all(r["arithmetic_error"] is None or r["arithmetic_error"] < 1e-12 for r in result["rows"])


def test_sbc_failures_stay_in_denominator(design, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate fit failure, not statistical detection")
    monkeypatch.setattr(sbc, "one_output", fail)
    result = sbc.run(design("sbc", "normal_conjugate", "reference", replications=3, rank_draws=2), tmp_path)
    assert result["planned_fits"] == 6
    assert result["completed"] == 0
    assert result["finding"] == "calibration_incomplete"
    assert len(result["datasets"]) == 3
    assert all(len(d["fits"]) == 2 for d in result["datasets"])
    assert result["tests"] == {}


def test_reference_sbc_reports_likelihood_and_independent_datasets(design, tmp_path):
    result = sbc.run(design("sbc", "normal_conjugate", "reference", replications=8), tmp_path)
    assert result["completed"] == 8
    assert "log_likelihood" in result["tests"]
    assert result["tests"]["parameter_0"]["replications"] == 8
    assert not result["accuracy_established"]


def test_identity_invariance_is_not_mixing(design, tmp_path):
    result = invariance.run(design("invariance", control="identity", replications=64), tmp_path)
    assert result["fraction_moved"] == 0.
    assert not result["mixing_established"]
    assert set(result["ranks"]["parameter_0"]) == set(range(8))


def test_gandy_scott_left_arm_order():
    left=np.array([[[0.],[0.]],[[1.],[11.]],[[2.],[12.]],[[3.],[13.]]])
    right=np.array([[[0.],[0.]],[[4.],[14.]],[[5.],[15.]],[[6.],[16.]]])
    actual=invariance.assemble_paths((left,right),np.array([2,1]))
    np.testing.assert_array_equal(actual[...,0],[[2,1,0,4],[11,0,14,15]])


def test_energy_mutation_is_detected_and_recurrence_control_is_retained(design, tmp_path):
    bad=invariance.run(design("invariance",control="wrong_energy",replications=128,
        step_size=1.7,leapfrog_steps=3),tmp_path/"bad")
    assert bad["finding"]=="discrepancy_detected"
    cycle=invariance.run(design("invariance",control="two_cycle",replications=16),tmp_path/"cycle")
    assert cycle["recurrence_fraction"]==1.
    assert not cycle["mixing_established"]


def test_power_runs_declared_experiment_instead_of_fake_sampler_results(design, tmp_path):
    from bayesfilter.testing.inference_validation.engines import power
    child=design()
    result=power.run(design("power",replications=2,options={"calibration_design":child.payload(),
        "power_controls":["baseline","wrong_metric"]}),tmp_path)
    assert result["tested_engine"]=="mechanics"
    assert result["rates"]["wrong_metric"]["detected"]==2
    assert result["rates"]["baseline"]["detected"]==0


def test_external_references_require_identity_and_uncertainty(tmp_path):
    from bayesfilter.testing.inference_validation.storage import file_hash
    samples = write_json(tmp_path / "samples.json", [[0.], [1.]])
    bundle = dict(schema="bayesfilter.external_posterior_reference.v1", target_identity="law-data-prior",
        quantity_names=["x"], samples_file=samples.name, samples_sha256=file_hash(samples),
        source="test independent fixture", version="1", method="exact", uncertainty={"x": "iid"},
        dependency_independence="separate formula")
    path = write_json(tmp_path / "reference.json", bundle)
    assert load_reference(path, target_identity="law-data-prior", quantity_names=["x"])[0].shape == (2, 1)
    with pytest.raises(ValueError, match="identity"):
        load_reference(path, target_identity="different-data", quantity_names=["x"])
    del bundle["uncertainty"]
    write_json(path, bundle)
    with pytest.raises(ValueError, match="uncertainty"):
        load_reference(path, target_identity="law-data-prior", quantity_names=["x"])


def test_accuracy_control_response_classifies_reference_failure(design):
    d = design("accuracy", "normal_conjugate", "prepared", "ignore_data")
    assert response(d, {"finding": "reference_discrepancy"})["status"] == "intended_discrepancy_detected"
    for finding in ("invalid", "unavailable", "incomplete", "posterior_incomplete"):
        assert response(d, {"finding": finding})["status"] == "unassessed"


def test_partial_diagnostic_arrays_do_not_report_completion(design, tmp_path):
    import time
    result = diagnostics.run(design("stopping", route="reference"), tmp_path, time.monotonic()-1)
    assert result["finding"] == "incomplete"
    assert result["completed"] == 0
    assert not result["assessment_complete"]


def test_empty_posterior_does_not_satisfy_output_or_interval_coverage(design):
    member = {"candidate_id":"empty", "L":3, "assessment":{"finding":"unavailable"},
              "warmup_exclusion_matches":True, "duplicate_chains":None,
              "stopped_intervals":{"quantities":[]}}
    records = [{"inventory":{"failures":[]}, "members":[member], "tuning_completion":"complete"}]
    result = pipeline.summarize_replications(design("stopping", route="prepared", replications=1), records)
    assert result["assessed_members"] == 1
    assert result["posterior_output_members"] == 0
    assert result["finding"] == "posterior_incomplete"
    assert not result["assessment_complete"]
    assert result["interval_coverage_at_stop"]["x:mean"]["unavailable"] == 1


def test_one_replication_without_a_candidate_stays_visible(design):
    member = {"candidate_id":"good", "L":3,
              "assessment":{"finding":"within_descriptive_tolerance"},
              "warmup_exclusion_matches":True, "duplicate_chains":False,
              "stopped_intervals":{"quantities":[]}}
    records = [{"inventory":{"failures":[]}, "members":m, "tuning_completion":"complete"}
               for m in ([member], [])]
    result = pipeline.summarize_replications(design("accuracy", route="prepared", replications=2), records)
    assert result["finding"] == "posterior_incomplete"
    assert result["posterior_available_replications"] == 1
