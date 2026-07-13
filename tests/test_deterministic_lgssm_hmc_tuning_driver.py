from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from bayesfilter.runtime import stable_config_hash
from docs.benchmarks import run_multidim_lgssm_serious_hmc_tuning_2026_07_09 as driver


def test_config_loads_and_preserves_hard_runtime_boundaries() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    payload = config.payload

    assert payload["schema"] == driver.CONFIG_SCHEMA
    assert payload["truth_and_data"]["horizon"] == 120
    assert payload["execution_policy"]["jit_compile"] is True
    assert payload["execution_policy"]["use_xla"] is True
    assert payload["execution_policy"]["jit_compile_false_runtime_allowed"] is False
    assert payload["execution_policy"]["gpu_sample_generation_allowed"] is False
    assert payload["execution_policy"]["runtime_gradient_tape_allowed"] is False
    assert payload["burnin_controller"]["r_hat_threshold"] == 1.01
    assert payload["sampling_controller"]["r_hat_threshold"] == 1.01
    assert payload["final_recovery_gate"]["all_parameters_required"] is True


def test_fixture_payload_is_deterministic_and_t120() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    payload_a = driver.build_fixture(config)
    payload_b = driver.build_fixture(config)

    assert payload_a["artifact_hash"] == payload_b["artifact_hash"]
    assert payload_a["horizon"] == 120
    assert payload_a["diagnostics"]["state_shape"] == [120, 4]
    assert payload_a["diagnostics"]["observation_shape"] == [120, 4]
    assert payload_a["diagnostics"]["transition_spectral_radius"] < 1.0
    assert payload_a["diagnostics"]["stationary_covariance_min_eigenvalue"] > 0.0
    assert payload_a["diagnostics"]["lyapunov_max_abs_residual"] <= 1.0e-12
    assert payload_a["config_hash"] == f"sha256:{stable_config_hash(config.payload)}"


def test_fixture_stage_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "fixture.json"
    rc = driver.main(["--stage", "fixture", "--output", str(output)])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == driver.FIXTURE_SCHEMA
    assert payload["horizon"] == 120
    assert payload["artifact_hash"].startswith("sha256:")
    assert payload["nonclaims"] == list(driver.NONCLAIMS)


def test_xla_score_gate_uses_fixture_and_jit_compiles() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    fixture = driver.build_fixture(config)
    config.fixture_path.parent.mkdir(parents=True, exist_ok=True)
    driver.write_json(fixture, config.fixture_path)

    payload = driver.build_xla_score_gate(config)

    assert payload["schema"] == driver.XLA_SCORE_SCHEMA
    assert payload["fixture_hash"] == fixture["artifact_hash"]
    assert payload["jit_compile"] is True
    assert payload["jit_compile_false_runtime_executed"] is False
    assert payload["runtime_autodiff_tape_executed"] is False
    assert payload["passed"] is True
    assert payload["finite_value"] is True
    assert payload["finite_score"] is True
    assert payload["target_status_valid"] is True
    assert payload["target_status_telemetry"]["status_code"] == 0
    assert payload["target_status_telemetry"]["valid_pre_regularized_score"] is True
    assert payload["target_status_telemetry"]["floor_count_value"] == 0
    assert payload["target_status_telemetry"]["min_innovation_eigenvalue"] > 0.0
    assert payload["score_shape"] == [18]
    assert payload["concrete_function_count"] == 1


def test_geometry_config_and_scale_are_deterministic() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    geometry_config = driver.geometry_config_from_config(config)
    scale = driver.geometry_scale_from_config(config)

    assert geometry_config.rank == 8
    assert geometry_config.sample_count == 520
    assert geometry_config.seed == (20260709, 401)
    assert scale.shape == (18,)
    assert scale[:4].tolist() == [0.5, 0.5, 0.5, 0.5]
    assert scale[4:10].tolist() == [0.6] * 6
    assert scale[10:14].tolist() == [0.35] * 4
    assert scale[14:18].tolist() == [0.35] * 4


def test_mass_from_synthetic_geometry_payload_is_deterministic() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    geometry_payload = {
        "schema": driver.GEOMETRY_SCHEMA,
        "artifact_hash": "sha256:test_geometry",
        "passed": True,
        "center_role": "prior_mean_raw_coordinates_truth_fixture",
        "center": [0.0] * 18,
        "scale": [1.0] * 18,
        "parameter_names": config.payload["model"]["parameter_names"],
        "low_rank_geometry": {
            "precision": [[2.0 if row == col else 0.0 for col in range(18)] for row in range(18)]
        },
    }

    payload_a = driver.build_mass_from_geometry_payload(config, geometry_payload)
    payload_b = driver.build_mass_from_geometry_payload(config, geometry_payload)

    assert payload_a["schema"] == driver.MASS_SCHEMA
    assert payload_a["artifact_hash"] == payload_b["artifact_hash"]
    assert payload_a["passed"] is True
    assert payload_a["vetoes"] == []
    assert payload_a["precision_covariance_identity_max_abs_error"] <= 1.0e-8
    assert payload_a["mass_covariance_eigen_summary"]["positive"] is True


def test_kernel_tuning_config_requires_serious_xla_tf_function() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    tuning_config = driver.kernel_tuning_config_from_config(config)

    assert tuning_config.preset == "serious"
    assert tuning_config.use_xla is True
    assert tuning_config.chain_execution_mode == "tf_function"
    assert tuning_config.target_scope == driver.TARGET_SCOPE
    assert tuning_config.allow_geometry_fallback is False
    assert tuning_config.payload()["use_xla"] is True
    assert tuning_config.repair_nonfinite_proposal_screen is True
    assert tuning_config.payload()["repair_nonfinite_proposal_screen"] is True
    assert tuning_config.staged_timeout_policy is not None
    assert tuning_config.staged_timeout_policy.enabled is True
    assert (
        tuning_config.public_timeout_budget_s
        == tuning_config.staged_timeout_policy.global_cap_s
    )
    assert (
        tuning_config.staged_timeout_policy.stage_budget_provenance[
            "fixed_mass_step"
        ]
        == "geometry_scaled_emergency_cap_machine_protection_not_progress_gate"
    )
    assert (
        tuning_config.staged_timeout_policy.stage_budgets_s["fixed_mass_step"]
        >= 3600.0
    )
    assert tuning_config.payload()["staged_timeout_policy"] is not None


def test_lgssm_adapter_declares_graph_native_full_chain_xla_authority() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    fixture = driver.build_fixture(config)
    contract = driver.triangular.load_lower_triangular_lgssm_contract(
        config.source_contract_path
    )
    adapter = driver.DeterministicLGSSMPosteriorAdapter(
        observations=fixture["observations"],
        contract=contract,
        parameter_names=fixture["parameter_names"],
        evidence_path="docs/benchmarks/artifacts/example_xla_gate.json",
    )

    capability = adapter.value_score_capability()
    value, score = adapter.log_prob_and_grad(fixture["raw_truth"])

    assert adapter.parameter_dim == 18
    assert len(adapter.parameter_names()) == 18
    assert capability.value_score_authority == "graph_native"
    assert capability.xla_hmc_ready is True
    assert capability.full_chain_xla_diagnostic_ready is True
    assert capability.runtime_backend == "tensorflow_manual_lgssm_svd_graph_status_score"
    assert capability.target_scope == driver.TARGET_SCOPE
    assert capability.is_accepted_full_chain_xla_diagnostic_authority is True
    assert value.shape == ()
    assert score.shape == (18,)


def test_lgssm_adapter_batch_value_score_jit_compiles() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    fixture = driver.build_fixture(config)
    contract = driver.triangular.load_lower_triangular_lgssm_contract(
        config.source_contract_path
    )
    adapter = driver.DeterministicLGSSMPosteriorAdapter(
        observations=fixture["observations"],
        contract=contract,
        parameter_names=fixture["parameter_names"],
        evidence_path="docs/benchmarks/artifacts/example_xla_gate.json",
    )
    raw = driver.tf.constant(fixture["raw_truth"], dtype=driver.tf.float64)
    batched = driver.tf.stack([raw, raw + driver.tf.ones_like(raw) * 1.0e-4], axis=0)

    @driver.tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(theta: driver.tf.Tensor) -> tuple[driver.tf.Tensor, driver.tf.Tensor]:
        return adapter.log_prob_and_grad(theta)

    value, score = compiled(batched)

    assert value.shape == (2,)
    assert score.shape == (2, 18)
    assert bool(driver.tf.reduce_all(driver.tf.math.is_finite(value)).numpy())
    assert bool(driver.tf.reduce_all(driver.tf.math.is_finite(score)).numpy())


def test_lgssm_adapter_target_status_telemetry_schema() -> None:
    config = driver.DeterministicLGSSMHMCConfig.load(driver.DEFAULT_CONFIG_PATH)
    fixture = driver.build_fixture(config)
    contract = driver.triangular.load_lower_triangular_lgssm_contract(
        config.source_contract_path
    )
    adapter = driver.DeterministicLGSSMPosteriorAdapter(
        observations=fixture["observations"],
        contract=contract,
        parameter_names=fixture["parameter_names"],
        evidence_path="docs/benchmarks/artifacts/example_xla_gate.json",
    )
    raw = driver.tf.constant(fixture["raw_truth"], dtype=driver.tf.float64)
    batched = driver.tf.stack([raw, raw], axis=0)

    telemetry = adapter.target_status_telemetry(batched)

    assert set(telemetry) == {
        "floor_count_value",
        "innovation_condition_estimate",
        "min_innovation_eigenvalue",
        "status_code",
        "valid_pre_regularized_score",
    }
    assert telemetry["status_code"].shape == (2,)
    assert telemetry["valid_pre_regularized_score"].shape == (2,)
    assert driver.tf.reduce_all(telemetry["status_code"] == 0)
    assert driver.tf.reduce_all(telemetry["valid_pre_regularized_score"])


def test_driver_source_has_no_forbidden_runtime_autodiff_tokens() -> None:
    source = inspect.getsource(driver)
    for forbidden in ("GradientTape", "batch_jacobian", "tape.", "jit_compile=False"):
        assert forbidden not in source
