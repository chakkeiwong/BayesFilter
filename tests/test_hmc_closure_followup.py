"""Execute the prepared study parser and the real configured-family runner."""
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


def test_complete_c1_inventory_preserves_target_policy_data_and_denominators():
    from docs.benchmarks.design_hmc_closure_followup_2026_09_24 import build_inventory
    suite, report = build_inventory(Path(__file__).resolve().parents[1])
    assert len(report["fit_inventory"]) == 512
    gaussian, beta = suite["designs"]
    assert gaussian["scenario"]["target"] == "gaussian"
    assert beta["scenario"]["target"] == "beta_binomial"
    assert beta["options"]["data"] == [5, 12]
    assert gaussian["options"]["posterior_settings"]["warmup_min_results"] == 30000
    assert beta["options"]["posterior_settings"]["retained_min_results"] == 5000
    assert {row["target"] for row in report["fit_inventory"]} == {"gaussian", "beta_binomial"}
    assert len({tuple(row["tuning_seed"]) for row in report["fit_inventory"]}) == 512
    assert report["fit_inventory"][0]["quantities"] == ["x:mean", "x:quantile", "y:mean", "y:quantile"]
    for design in suite["designs"]:
        assert design["replications"] == 256 and design["alpha"] == .05
        assert design["l_grid"] == [3, 5, 9, 13, 18, 25] or design["l_grid"] == (3, 5, 9, 13, 18, 25)
        assert "fixed_comparator" not in design["options"]
        assert design["options"]["native_search"]
        assert design["options"]["coverage_floor"] == .90


@pytest.mark.parametrize("index", range(6))
def test_configured_pricing_calls_shared_trainer_and_checks_saved_map(tmp_path, index):
    from docs.benchmarks.price_hmc_configured_maps_2026_09_24 import inventory, run_arm
    arm = inventory(smoke=True)[index]
    result = run_arm(arm, tmp_path / "arm", deadline=time.monotonic()+120)
    assert result["status"] == "complete", result
    assert result["trainer_class"] == "NeuTraTransportTrainer"
    assert result["transport_config"]["kind"] == arm["kind"]
    assert result["transport_config"]["stages"] == arm["stages"]
    assert result["graph_trace_count"] == 1
    assert result["updates_completed"] == 2
    assert result["frozen_reload_passed"]
    assert (tmp_path / "arm/map.json").exists()
    assert (tmp_path / "arm/checkpoint.json").exists()


def test_configured_pricing_records_exhaustion_without_starting_a_fit(tmp_path):
    from docs.benchmarks.price_hmc_configured_maps_2026_09_24 import inventory, run_arm
    result = run_arm(inventory(smoke=True)[0], tmp_path / "arm", deadline=time.monotonic()-1)
    assert result["status"] == "budget_censored"
    assert result["updates_completed"] == 0
    assert not (tmp_path / "arm/map.json").exists()


def test_neutra_native_consumer_delegates_to_public_tuner(monkeypatch, tmp_path):
    import tensorflow as tf
    from bayesfilter.inference import neutra_end_to_end as consumer
    from bayesfilter.inference.hmc_configuration import HMCKernelTuningConfig
    from bayesfilter.inference.hmc_tuning_dispatch import tune_hmc_kernel
    assert consumer.tune_hmc_kernel is tune_hmc_kernel
    transformed, expected = object(), object()
    monkeypatch.setattr(consumer, "_fixed_transport_adapter", lambda *args: transformed)
    calls = []
    def tune(**kwargs):
        calls.append(kwargs)
        return expected
    monkeypatch.setattr(consumer, "tune_hmc_kernel", tune)
    spec = SimpleNamespace(cell_id="fixture", parameter_dim=2, target_signature="b"*64)
    loaded = SimpleNamespace(transport=object(), artifact_signature="a"*64)
    assert consumer._native_tune(spec=spec, adapter=object(), loaded=loaded,
                                root=tmp_path, config=None) is expected
    assert len(calls) == 1 and calls[0]["adapter"] is transformed
    assert isinstance(calls[0]["config"], HMCKernelTuningConfig)
    assert calls[0]["config"].mass_policy == "fixed_identity"
    tf.debugging.assert_equal(calls[0]["initial_position"], tf.zeros([2], tf.float64))


def test_lgssm_registry_pin_matches_current_source_and_rejects_previous():
    from bayesfilter.testing.neutra_model_registry_tf import (
        LGSSM_SIGNATURE, LGSSM_ADAPTER_SIGNATURE, _lgssm_bundle)
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target, InvalidDeterministicLGSSMTarget)
    bundle = _lgssm_bundle()
    assert bundle.target_signature == LGSSM_SIGNATURE
    assert bundle.adapter.adapter_signature() == LGSSM_ADAPTER_SIGNATURE
    with pytest.raises(InvalidDeterministicLGSSMTarget, match="signature mismatch"):
        load_deterministic_lgssm_exact_target(
            expected_target_signature="bd40a828bc4916e5e09a8e6135f315ebc45c06844aed38a506d6296c2642557d")
