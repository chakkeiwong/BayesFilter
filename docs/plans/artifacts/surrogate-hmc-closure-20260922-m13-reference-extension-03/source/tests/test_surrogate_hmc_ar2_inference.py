"""AR2 inference authority, provenance and proposal-metric diagnostics."""
import copy
import json
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_tuning_interface_capability
from bayesfilter.inference.neural_force_hmc import FrozenPositionOnlyForce, FrozenTargetPotential
from bayesfilter.inference.surrogate_hmc_program import file_hash
from bayesfilter.testing.surrogate_hmc_ar2_assessment import METHOD
from bayesfilter.testing.surrogate_hmc_ar2_inference import (
    CALIBRATION_ROOT, _read_tensor, bind_arm, check_inference_prerequisites, metric_guard_diagnostic, tuning_config,
)


def test_current_calibration_pointer_is_qualified_for_the_frozen_cost():
    from bayesfilter.testing.surrogate_hmc_ar2_calibration import COST_ROOT, load_cost_contract
    from bayesfilter.inference.surrogate_hmc_closure_program import PASS_CAPS
    calibration = json.loads((CALIBRATION_ROOT/"report.json").read_text())
    check_inference_prerequisites(load_cost_contract(COST_ROOT), calibration,
        file_hash(COST_ROOT/"report.json"), PASS_CAPS["ar2-inference"], "B")


@pytest.mark.parametrize("arm", ["a", "b"])
def test_stage_dispatch_reaches_the_inference_consumer(monkeypatch, tmp_path, arm):
    from bayesfilter.testing import surrogate_hmc_ar2_inference as inference
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as stages
    from bayesfilter.testing.surrogate_hmc_ar2_calibration import COST_ROOT
    calls = []
    def run(output, **kwargs):
        calls.append((output, kwargs))
        return dict(status="DISPATCH_FIXTURE", qualified=False)
    monkeypatch.setattr(inference, "inference_stage", run)
    output = tmp_path/"report.json"
    assert stages.main(["--stage", "ar2-inference-"+arm, "--output", str(output), "--device", "cpu"]) == 0
    assert calls == [(output, dict(cost_root=COST_ROOT, calibration_root=CALIBRATION_ROOT, arm=arm.upper()))]


def test_model_endpoint_and_proposal_force_are_bound_separately():
    consumer = SimpleNamespace(
        force=FrozenPositionOnlyForce(lambda q: 1.1*q, identity="canonical-test-field"),
        exact_force=FrozenPositionOnlyForce(lambda q: q, identity="exact-test-field"),
        endpoint=FrozenTargetPotential(lambda q: .5*tf.reduce_sum(q*q, -1), identity="model-posterior"))
    for arm, identity in (("A", "exact-test-field"), ("B", "canonical-test-field")):
        adapter, binding = bind_arm(consumer, arm, run_index=2)
        assert binding.force_identity == identity
        assert binding.endpoint_target_identity == "model-posterior"
        assert binding.coordinate_scope == "raw"
        assert binding.target_scope == adapter.target_scope
        assert "run2" in adapter.target_scope
    capability = hmc_tuning_interface_capability("tune_hmc_kernel")
    assert capability.artifact_authority and capability.capability_status == "tested_supported"
    with pytest.raises(ValueError, match="only A and B"):
        bind_arm(consumer, "C")


def test_calibration_cost_identity_and_budget_fail_closed():
    cost = dict(qualified=True, status="AR2_COMPLETE_TRANSITION_COST_MEASURED",
        precision=dict(precision_forecast_sufficient=True, retained_looks=[4000, 8000]),
        conservative_inference_forecast={arm: dict(cpu_worker_seconds=2000., wall_seconds=2100.) for arm in ("A", "B")})
    calibration = dict(qualified=True, status="AR2_SCOPED_UNCERTAINTY_QUALIFIED",
        cost_report_sha256="measured", protocol=dict(looks=[4000, 8000], method=METHOD))
    available = dict(cpu_worker_seconds=2200., gpu_seconds=2200.)
    check_inference_prerequisites(cost, calibration, "measured", available, "B")
    for field, value in (("qualified", False), ("cost_report_sha256", "stale"),
                         ("protocol", dict(looks=[2000, 4000], method=METHOD)),
                         ("protocol", dict(looks=[4000, 8000], method="uncalibrated"))):
        invalid = copy.deepcopy(calibration)
        invalid[field] = value
        with pytest.raises(ValueError, match="uncertainty prerequisite"):
            check_inference_prerequisites(cost, invalid, "measured", available, "B")
    with pytest.raises(ValueError, match="reservation"):
        check_inference_prerequisites(cost, calibration, "measured", {**available, "cpu_worker_seconds": 2199.}, "B")


def test_metric_protection_uses_the_public_tuners_actual_regularizer():
    config = tuning_config("AR2-test", 21000)
    assert config.nomination.trajectory_candidates == (1, 2, 4, 8)
    assert config.verification_results >= 1000 and config.nomination.use_xla
    assert metric_guard_diagnostic(config)["passed"]


def test_retained_reader_rejects_modified_tensor(tmp_path):
    path = tmp_path/"sample.tensor"
    tf.io.write_file(str(path), tf.io.serialize_tensor(tf.constant([1., 2.], tf.float64)))
    record = dict(path=path.name, sha256=file_hash(path))
    tf.debugging.assert_equal(_read_tensor(tmp_path, record), tf.constant([1., 2.], tf.float64))
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        _read_tensor(tmp_path, record)
