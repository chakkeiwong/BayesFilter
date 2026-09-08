"""Anti-pooling tests for the SSL-LSTM NeuTra global-mixing contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_global_mixing import (
    GlobalMixingDiagnosticError,
    assess_retained_mode_mixing,
)


ROOT = Path(__file__).resolve().parents[1]
GPU_CANARY = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py"
)
GPU_PREFLIGHT = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_gpu_preflight_2026_08_19.py"
)
TRAINING_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py"
)
HMC_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_global_mixing_hmc_2026_08_19.py"
)
CONTINUATION_HMC_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_global_mixing_continuation_2026_08_20.py"
)


def _gpu_canary_module():
    spec = importlib.util.spec_from_file_location("ssl_lstm_q20_gpu_canary", GPU_CANARY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _training_module():
    spec = importlib.util.spec_from_file_location(
        "ssl_lstm_q20_global_mixing_training", TRAINING_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hmc_module():
    spec = importlib.util.spec_from_file_location(
        "ssl_lstm_q20_global_mixing_hmc", HMC_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _continuation_hmc_module():
    spec = importlib.util.spec_from_file_location(
        "ssl_lstm_q20_global_mixing_hmc_continuation", CONTINUATION_HMC_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_equal_pooling_of_mode_locked_chains_is_rejected() -> None:
    # The pooled labels contain both modes and look balanced, but each chain is
    # a conditional mode chain.  Initial occupancy must not manufacture a pass.
    labels = tf.constant(
        (
            (0, 0, 0, 0),
            (1, 1, 1, 1),
        ),
        tf.int32,
    )
    report = assess_retained_mode_mixing(labels, region_count=2)
    assert bool(report.valid_labels.numpy())
    assert report.global_region_counts.numpy().tolist() == [4, 4]
    assert not bool(report.passed.numpy())
    assert not bool(report.every_chain_visited_every_region.numpy())


def test_one_common_kernel_that_crosses_both_modes_can_pass_coverage_screen() -> None:
    labels = tf.constant(
        (
            (0, 0, 1, 1, 0, 1),
            (1, 0, 0, 1, 1, 0),
        ),
        tf.int32,
    )
    report = assess_retained_mode_mixing(labels, region_count=2)
    assert bool(report.passed.numpy())
    assert report.chain_transition_counts.numpy().tolist() == [3, 3]
    assert report.chain_region_counts.numpy().tolist() == [[3, 3], [3, 3]]


def test_invalid_region_label_is_a_hard_coverage_failure() -> None:
    labels = tf.constant(((0, 2, 0), (1, 0, 1)), tf.int32)
    report = assess_retained_mode_mixing(labels, region_count=2)
    assert not bool(report.valid_labels.numpy())
    assert not bool(report.passed.numpy())


def test_shape_and_threshold_contracts_fail_closed() -> None:
    try:
        assess_retained_mode_mixing(tf.constant((0, 1), tf.int32), region_count=2)
    except GlobalMixingDiagnosticError:
        pass
    else:
        raise AssertionError("rank-one labels must be rejected")
    try:
        assess_retained_mode_mixing(
            tf.constant(((0, 1), (1, 0)), tf.int32),
            region_count=2,
            minimum_transitions_per_chain=0,
        )
    except GlobalMixingDiagnosticError:
        pass
    else:
        raise AssertionError("zero transition threshold must be rejected")


def test_gpu_canary_preserves_draw_chain_axes_and_memory_policy() -> None:
    source = GPU_CANARY.read_text(encoding="utf-8")
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert source.index(
        "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)"
    ) < source.index(
        "from bayesfilter.inference.batched_value_score import"
    )
    assert "labels_draw_chain = tf.reshape" in source
    assert "tf.transpose(labels_draw_chain, (1, 0))" in source
    assert "tf.reshape(physical_samples[:, 2] < 0.0, (4, 64))" not in source
    assert "samples_draw_chain = tf.ensure_shape(hmc.samples, (64, 4, 4))" in source


def test_gpu_canary_uses_disjoint_smc_banks_only_as_replay() -> None:
    source = GPU_CANARY.read_text(encoding="utf-8")
    assert "train_rows = physical[:600]" in source
    assert "selection_rows = physical[600:700]" in source
    assert "audit_rows = physical[700:]" not in source
    assert "trainer.validation_batch(audit_rows" not in source
    assert "_verified_replay_sources(tuple(range(7)))" in source
    assert '"replay rows are not a posterior archive"' in source
    assert '"no pooling of mode-locked conditional chains"' in source


def test_gpu_canary_receipt_binds_only_training_and_selection_banks() -> None:
    runner = _gpu_canary_module()
    sources, metadata = runner._verified_replay_sources(tuple(range(7)))
    assert len(sources) == 7
    assert metadata["loaded_bank_indices"] == list(range(7))
    assert [source[2]["child"] for source in sources] == [
        f"central-{index:02d}" for index in range(7)
    ]
    assert metadata["reserved_audit_bank"] == {
        "child": "central-07",
        "stage_count": 5,
        "tensor_loaded": False,
        "target_evaluated": False,
        "used_for_training_selection_or_nomination": False,
    }


def test_gpu_canary_receipt_rejects_audit_bank_and_hash_drift(monkeypatch) -> None:
    runner = _gpu_canary_module()
    with pytest.raises(RuntimeError, match="central-00 through central-06"):
        runner._verified_replay_sources((7,))
    audit_sources, audit_metadata = runner._verified_replay_sources(
        (7,), allow_reserved_audit=True
    )
    assert len(audit_sources) == 1
    assert audit_sources[0][2]["child"] == "central-07"
    assert audit_metadata["reserved_audit_bank"]["tensor_loaded"] is True
    with pytest.raises(RuntimeError, match="only central-07"):
        runner._verified_replay_sources((6, 7), allow_reserved_audit=True)
    original = runner._sha256

    def drifted(path: Path) -> str:
        if path == runner.RECOVERY:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(runner, "_sha256", drifted)
    with pytest.raises(RuntimeError, match="recovery receipt SHA-256 mismatch"):
        runner._verified_replay_sources(tuple(range(7)))


def test_gpu_canary_uses_exact_adapter_keyword_and_fresh_atomic_artifacts() -> None:
    source = GPU_CANARY.read_text(encoding="utf-8")
    assert "fixed_transport=trainer.transport" not in source
    assert "transport=trainer.transport" in source
    assert "refusing to reuse canary output root" in source
    assert 'path.with_suffix(path.suffix + ".tmp")' in source
    assert "temporary.replace(path)" in source
    assert "owner_designated_managed_session_visible_gpu_trusted" in source
    assert '"training_batch_size": int(train_rows.shape[0])' in source
    assert '"scalar_target_fallback_used": False' in source


def test_gpu_preflight_configures_growth_before_logical_device_and_tensor() -> None:
    source = GPU_PREFLIGHT.read_text(encoding="utf-8")
    growth = source.index("configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)")
    logical = source.index('tf.config.list_logical_devices("GPU")')
    tensor = source.index("compiled_probe(tf.ones")
    assert growth < logical < tensor
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)' in source
    assert "enable_tensor_float_32_execution(False)" in source
    assert "jit_compile=True" in source
    assert "tf.float64" in source
    assert "4096.0 * 2.220446049250313e-16" in source
    assert "assert_less_equal" in source
    assert "owner_designated_managed_session_visible_gpu_trusted" in source
    assert "temporary.replace(path)" in source


def test_training_runner_freezes_grid_and_opens_audit_only_after_nomination() -> None:
    source = TRAINING_RUNNER.read_text(encoding="utf-8")
    assert "PRIMARY_CAPACITIES = ((64, 3), (128, 6))" in source
    assert "PRIMARY_LEARNING_RATES = (1.0e-3, 3.0e-4)" in source
    assert "TRAINING_SEEDS = (2, 3)" in source
    assert "UPDATE_LADDER = (250, 2000, 8000)" in source
    assert "train_rows = tf.ensure_shape(physical[:600], (600, 4))" in source
    assert "selection_rows = tf.ensure_shape(physical[600:700], (100, 4))" in source
    nomination = source.index('output / "nominations-before-audit.json"')
    audit_open = source.index(
        "audit_rows, audit_weights, audit_meta = _load_audit(tf, canary_module)"
    )
    assert nomination < audit_open
    assert source.count("trainer.validation_batch(audit_rows, audit_weights)") == 1
    assert "allow_reserved_audit=True" in source
    assert '"used_for_ranking": False' in source
    assert "FROZEN_CANDIDATE_AUDIT_FAILURE" not in source


def test_training_runner_configures_memory_before_project_tensorflow_paths() -> None:
    source = TRAINING_RUNNER.read_text(encoding="utf-8")
    growth = source.index("configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)")
    project_target = source.index(
        "from bayesfilter.inference.batched_value_score import"
    )
    assert growth < project_target
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)' in source
    assert "enable_tensor_float_32_execution(False)" in source
    assert '"training_batch_size": 600' in source
    assert '"sample_wise_loop_used": False' in source
    assert '"scalar_target_fallback_used": False' in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "import numpy" not in source


def test_training_runner_rejects_grid_drift() -> None:
    runner = _training_module()
    valid = SimpleNamespace(
        capacities=runner.PRIMARY_CAPACITIES,
        learning_rates=runner.PRIMARY_LEARNING_RATES,
        seeds=runner.TRAINING_SEEDS,
        update_ladder=runner.UPDATE_LADDER,
        device="1",
        time_cap_seconds=runner.INTERNAL_DEFAULT_WALL_SECONDS,
    )
    runner._validate_args(valid)
    drifted = SimpleNamespace(**vars(valid))
    drifted.capacities = ((32, 3),)
    with pytest.raises(SystemExit, match="capacities are frozen"):
        runner._validate_args(drifted)


def test_training_runner_validates_prior_artifact_identities() -> None:
    runner = _training_module()
    prior = runner._validate_prior_artifacts()
    assert prior["preflight"]["sha256"] == runner.PREFLIGHT_SHA256
    assert prior["canary_result"]["sha256"] == runner.CANARY_RESULT_SHA256
    assert prior["canary_result"]["mode_locked"] is True


def test_training_state_hash_and_restore_roundtrip_fail_closed() -> None:
    runner = _training_module()
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )

    config = WeightedNeuTraConfig(
        dimension=2,
        hidden_layers=(4,),
        stages=1,
        initialization_seed=(19, 2),
        jit_compile=False,
    )
    trainer = WeightedForwardKLNeuTraTrainer(config)
    state = trainer.state_payload()
    trainer.variables[0].assign_add(tf.ones_like(trainer.variables[0]))
    runner._restore_trainer_state(tf, trainer, state)
    assert trainer.state_payload()["state_hash"] == state["state_hash"]
    tampered = dict(state)
    tampered["step"] = int(state["step"]) + 1
    with pytest.raises(RuntimeError, match="semantic hash mismatch"):
        runner._restore_trainer_state(tf, trainer, tampered)


def test_training_runner_uses_fresh_atomic_artifacts_and_separate_hmc_owner() -> None:
    source = TRAINING_RUNNER.read_text(encoding="utf-8")
    assert "refusing to reuse training output root" in source
    assert 'path.with_suffix(path.suffix + ".tmp")' in source
    assert "temporary.replace(path)" in source
    assert "run_ssl_lstm_q20_neutra_global_mixing_hmc_2026_08_19.py" not in source
    assert "run_fixed_transport_full_chain_tfp_hmc" not in source
    assert '"checkpoint_sha256"' in source
    assert '"training_state_hash"' in source
    assert '"transport_tensor_hash"' in source


def test_hmc_runner_binds_training_graph_and_derived_budget() -> None:
    runner = _hmc_module()
    result, budget = runner._validated_training_artifacts()
    assert result["status"] == "TRAINING_SCREEN_AND_FROZEN_AUDIT_COMPLETED"
    assert [item["seed"] for item in result["nominations"]] == [2, 3]
    assert budget["derived_hmc_remainder_seconds"] == pytest.approx(
        runner.DERIVED_HMC_REMAINDER_SECONDS
    )
    assert budget["internal_hmc_cap_seconds"] == 23323.0
    assert budget["predictive_reserve_seconds"] == 3600.0


def test_continuation_hmc_runner_passes_repository_route_policy_audit() -> None:
    runner = _continuation_hmc_module()
    audit = runner._route_policy_audit()
    assert audit["passed"] is True
    assert audit["canonical_policy_id"] == "bayesfilter_neutra_sequential_hmc_v1"
    assert audit["current_route"]["classification"] == "active_claim_bearing"
    assert audit["discovered_route_count"] == audit["classified_route_count"]


def test_continuation_hmc_runner_binds_prior_terminal_and_fresh_budget() -> None:
    runner = _continuation_hmc_module()
    result, budget = runner._validated_training_artifacts()

    assert result["status"] == "TRAINING_SCREEN_AND_FROZEN_AUDIT_COMPLETED"
    assert budget["authorization"] == "user_incremental_18_hours_2026-08-20"
    assert budget["incremental_campaign_wall_cap_seconds"] == 64800.0
    assert budget["external_hmc_cap_seconds"] == 61200
    assert budget["internal_hmc_cap_seconds"] == 61020.0
    assert budget["predictive_reserve_seconds"] == 3600.0
    assert budget["historical_wall_subtracted_from_incremental_grant"] is False
    assert budget["prior_hmc_result"]["sha256"] == runner.PRIOR_HMC_RESULT_SHA256


def test_continuation_hmc_answer_path_forecasts_are_complete() -> None:
    runner = _continuation_hmc_module()
    l5 = runner._candidate_answer_path_forecast(5)
    l3 = runner._candidate_answer_path_forecast(3)

    assert l5["component_leapfrog_transitions"] == {
        "maximum_tuning_and_screen": 2560,
        "verification": 10320,
        "mechanics": 400,
        "canonical_minimum_sequential": 20000,
    }
    assert l5["total_leapfrog_transitions"] == 33280
    assert l3["total_leapfrog_transitions"] == 19968
    assert l5["predicted_candidate_answer_path_seconds"] == pytest.approx(
        runner.CANARY_HMC_CALL_SECONDS
        * 33280
        / runner.CANARY_HMC_LEAPFROG_TRANSITIONS
        * runner.TUNING_CALL_OVERRUN_ALLOWANCE
    )
    assert (
        l5["predicted_candidate_answer_path_seconds"]
        + runner.CLOSEOUT_RESERVE_SECONDS
        < runner.INTERNAL_HMC_CAP_SECONDS
    )


def test_continuation_hmc_runner_uses_resource_only_full_path_gates() -> None:
    source = CONTINUATION_HMC_RUNNER.read_text(encoding="utf-8")

    assert "CANDIDATE_KERNEL_ORDER = ((2, 5), (3, 3))" in source
    assert "passthrough_exceptions=(HMCBudgetExhausted,)" in source
    assert '"whole_path_gate_applied": is_full_verification' in source
    assert 'tuning_root / "resource-stop.json"' in source
    assert '"tuner_artifact_written": False' in source
    assert '"candidate_or_kernel_rejected": False' in source
    assert '"post-verification-answer-path-sufficiency.json"' in source
    assert "budget_check=budget_check" in source
    assert "20260820" in source
    assert "LEAPFROG_ORDER" not in source


def test_continuation_hmc_runner_freezes_incremental_caps_and_roots() -> None:
    runner = _continuation_hmc_module()
    valid = SimpleNamespace(
        device="1",
        training_root=runner.TRAINING_ROOT,
        prior_hmc_root=runner.PRIOR_HMC_ROOT,
        incremental_campaign_cap_seconds=64800.0,
        predictive_reserve_seconds=3600.0,
        time_cap_seconds=61020.0,
    )
    runner._validate_args(valid)

    drifted = SimpleNamespace(**vars(valid))
    drifted.incremental_campaign_cap_seconds = 129600.0
    with pytest.raises(SystemExit, match="incremental campaign cap is frozen"):
        runner._validate_args(drifted)
    drifted = SimpleNamespace(**vars(valid))
    drifted.prior_hmc_root = runner.TRAINING_ROOT
    with pytest.raises(SystemExit, match="prior HMC root is frozen"):
        runner._validate_args(drifted)


def test_hmc_runner_restores_real_frozen_nominee_with_exact_hashes() -> None:
    runner = _hmc_module()
    result, _ = runner._validated_training_artifacts()
    training = runner._load_module(
        runner.TRAINING_RUNNER, "ssl_lstm_q20_training_for_hmc_test"
    )
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )

    trainer, identity = runner._load_candidate(
        tf,
        training,
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
        result["nominations"][0],
    )
    assert identity["seed"] == 2
    assert identity["training_state_hash"] == result["nominations"][0][
        "training_state_hash"
    ]
    assert identity["transport_tensor_hash"] == result["nominations"][0][
        "transport_tensor_hash"
    ]
    assert trainer.transport.manifest_payload()["frozen_identity"][
        "checkpoint_sha256"
    ] == result["nominations"][0]["state_sha256"]


def test_hmc_mode_report_rejects_balanced_pool_of_locked_chains() -> None:
    runner = _hmc_module()
    positive = tf.constant((0.0, 0.0, 0.5, 0.0), tf.float64)
    negative = tf.constant((0.0, 0.0, -0.5, 0.0), tf.float64)
    one_draw = tf.stack((positive, negative, positive, negative), axis=0)
    samples = tf.repeat(one_draw[tf.newaxis, :, :], repeats=8, axis=0)
    report = runner._mode_report(tf, samples)
    assert report["global_region_counts"] == [16, 16]
    assert report["chain_transition_counts"] == [0, 0, 0, 0]
    assert report["passed"] is False


def test_hmc_runner_uses_ordered_full_verification_and_canonical_controller() -> None:
    source = HMC_RUNNER.read_text(encoding="utf-8")
    assert "TRANSPORT_SEED_ORDER = (2, 3)" in source
    assert "LEAPFROG_ORDER = (3, 5, 10, 15)" in source
    assert "leapfrog_grid=(leapfrog,)" in source
    assert "verification_num_results=TUNING_VERIFICATION_RESULTS" in source
    assert "verification_num_burnin_steps=TUNING_VERIFICATION_BURNIN" in source
    assert 'verification_coordinate_system="raw_target_coordinates"' in source
    assert 'target_status_trace_policy="per_chain_step"' in source
    assert "if result.passed:" in source
    assert "run_full_chain=bounded_full_chain" in source
    assert "TUNING_CALL_OVERRUN_ALLOWANCE = 1.25" in source
    assert '"resource_refusal"' in source
    assert "run_sequential_neutra_hmc(" in source
    assert "NEUTRA_SEQUENTIAL_HMC_POLICY_ID" in source
    assert "budget_check=budget_check" in source
    assert '"campaign_resource_cap"' in source
    assert "warmup_chunk_results=SEQUENTIAL_CHUNK_RESULTS" in source
    assert "warmup_min_results=SEQUENTIAL_WARMUP_MIN" in source
    assert "retained_min_results=SEQUENTIAL_RETAINED_MIN" in source
    assert "retained_max_results=SEQUENTIAL_RETAINED_MAX" in source
    assert "rank_normalized_hmc_diagnostics(" in source
    assert '"observation_weight_sign_indicator"' in source
    assert '"pooling_across_candidates_or_mode_locked_chains": False' in source


def test_hmc_runner_abort_closeout_writes_result_manifest_and_hashes(tmp_path) -> None:
    runner = _hmc_module()
    output = tmp_path / "abort"
    output.mkdir()
    args = SimpleNamespace(device="1")
    payload = runner._write_abort_terminal(
        output,
        status="UNDER_BUDGETED_HMC",
        error=runner.HMCBudgetExhausted("fixture budget stop"),
        started=0.0,
        args=args,
    )
    assert payload["passed"] is False
    assert payload["failure_classification"] == (
        "resource_budget_exhaustion_not_sampler_failure"
    )
    assert (output / "result.json").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "artifact-hashes.json").is_file()


def test_hmc_runner_configures_memory_and_archives_atomically() -> None:
    source = HMC_RUNNER.read_text(encoding="utf-8")
    growth = source.index("configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)")
    target_import = source.index(
        "from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import"
    )
    assert growth < target_import
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)' in source
    assert "enable_tensor_float_32_execution(False)" in source
    assert "temporary.replace(path)" in source
    assert "refusing to reuse HMC output root" in source
    assert "tf.io.serialize_tensor" in source
    assert "import numpy" not in source
    assert "NUTS" not in source
    assert "num_leapfrog_steps=1" not in source


def test_hmc_runner_freezes_caps_and_preserves_predictive_reserve() -> None:
    runner = _hmc_module()
    valid = SimpleNamespace(
        device="1",
        training_root=runner.TRAINING_ROOT,
        campaign_wall_cap_seconds=28800.0,
        predictive_reserve_seconds=3600.0,
        time_cap_seconds=23323.0,
    )
    runner._validate_args(valid)
    drifted = SimpleNamespace(**vars(valid))
    drifted.predictive_reserve_seconds = 0.0
    with pytest.raises(SystemExit, match="predictive reserve is frozen"):
        runner._validate_args(drifted)
