from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_five_executable_six_blocked_and_one_owner_excluded_cell() -> None:
    from bayesfilter.testing.neutra_model_registry_tf import validate_registry

    payload = validate_registry()
    assert len(payload["executable"]) == 5
    assert len(payload["blocked"]) == 6
    assert len(payload["owner_excluded"]) == 1
    ids = [
        row["cell_id"]
        for row in (
            payload["executable"]
            + payload["blocked"]
            + payload["owner_excluded"]
        )
    ]
    assert len(ids) == len(set(ids)) == 12


def test_current_direct_signatures_are_not_historical_typed_hashes() -> None:
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    historical = {
        "036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30",
        "8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad",
        "0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc",
        "e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665",
    }
    declared = {spec.target_signature for spec in EXECUTABLE_CELLS}
    assert not historical & declared


def test_sir_ukf_is_owner_excluded_and_not_master_executable() -> None:
    from bayesfilter.testing.neutra_model_registry_tf import (
        EXECUTABLE_CELLS,
        OWNER_EXCLUDED_CELLS,
    )

    assert "SIR-UKF" not in {spec.cell_id for spec in EXECUTABLE_CELLS}
    excluded = next(
        item for item in OWNER_EXCLUDED_CELLS if item.cell_id == "SIR-UKF"
    )
    assert excluded.state == "OWNER_EXCLUDED_METHOD_NOT_APPLICABLE"
    assert excluded.reentry_rung == "none; reentry requires a new owner direction"


def test_svx_zc_is_blocked_until_explicit_xla_hmc_admission() -> None:
    from bayesfilter.testing.neutra_model_registry_tf import (
        BLOCKED_CELLS,
        EXECUTABLE_CELLS,
    )

    assert "SVX-ZC" not in {spec.cell_id for spec in EXECUTABLE_CELLS}
    blocked = next(item for item in BLOCKED_CELLS if item.cell_id == "SVX-ZC")
    assert blocked.state == "TARGET_BLOCKED_XLA_HMC_ADMISSION"
    assert "fresh scope-specific tuning" in blocked.reentry_rung


@pytest.mark.parametrize(
    "relative_path",
    (
        "scripts/p76_bounded_ukf_minibatch_pilot.py",
        "scripts/p76_generated_corrected_metric_diagnostic.py",
        "scripts/p77_budgeted_corrected_metric_training.py",
    ),
)
def test_historical_sir_ukf_p76_p77_chain_is_fail_closed_before_tensorflow(
    relative_path: str, tmp_path: Path
) -> None:
    script = ROOT / relative_path
    source = script.read_text(encoding="utf-8")
    output = tmp_path / "forbidden.json"

    assert "tensorflow" not in source.lower()
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "retired:" in completed.stderr
    assert "SIR-UKF" in completed.stderr
    assert not output.exists()


def test_model_actions_resolve_registry_eligibility_before_inference_import() -> None:
    path = ROOT / "docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py"
    text = path.read_text(encoding="utf-8")
    for function_name, next_name in (
        ("_run_cell", "_run_preflight"),
        ("_run_preflight", "_run_training_throughput"),
        ("_run_training_throughput", "_run_frozen_validation"),
        ("_run_frozen_validation", "_run_frozen_broad_grid"),
        ("_run_frozen_broad_grid", "_run_broad_grid_sequential"),
        ("_run_broad_grid_sequential", "_run_campaign"),
    ):
        source = text.split(f"def {function_name}", 1)[1].split(
            f"def {next_name}", 1
        )[0]
        assert source.index("spec = _spec(args.cell)") < source.index(
            "from bayesfilter.inference.neutra_end_to_end import"
        )


def test_screen_pairing_uses_common_training_seeds_and_replication_means(
    monkeypatch, tmp_path: Path
) -> None:
    from dataclasses import replace

    import bayesfilter.inference.neutra_end_to_end as campaign
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    base = next(item for item in EXECUTABLE_CELLS if item.cell_id == "SIR-SGQF")
    spec = replace(
        base,
        recipes=base.recipes[:2],
        screen_seeds=((20260730, 3001), (20260730, 3002), (20260730, 3003)),
    )
    calls = []

    def fake_train_and_score(**kwargs):
        recipe = kwargs["recipe"]
        seed = tuple(kwargs["seed"])
        calls.append((recipe.recipe_id, seed, Path(kwargs["output_root"])))
        recipe_offset = 0.0 if recipe == spec.recipes[0] else 0.1
        seed_offset = float(spec.screen_seeds.index(seed))
        return {
            "recipe_id": recipe.recipe_id,
            "passed": True,
            "mean_reverse_kl": seed_offset + recipe_offset,
            "affine_nonworse": True,
        }

    monkeypatch.setattr(campaign, "_train_and_score", fake_train_and_score)
    selection = campaign._screen_recipes(
        spec=spec,
        adapter=object(),
        bound_adapter=object(),
        center=object(),
        factor=object(),
        root=tmp_path,
        config=campaign.EndToEndConfig(
            output_root=tmp_path, screen_steps=1, final_steps=1, require_gpu=False
        ),
        memory_policy={},
    )

    for recipe in spec.recipes:
        assert tuple(seed for name, seed, _path in calls if name == recipe.recipe_id) == (
            spec.screen_seeds
        )
    assert selection["statistical_unit"] == "independent_training_seed"
    assert selection["screen_seeds_paired_across_recipes"] is True
    assert all(
        len(row["reverse_kl_replication_means"]) == len(spec.screen_seeds)
        for row in selection["rows"]
    )


def test_new_runner_does_not_duplicate_sampler_or_diagnostics() -> None:
    path = ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.names
    }
    assert "numpy" not in imports
    forbidden_calls = {
        "sample_chain",
        "HamiltonianMonteCarlo",
        "potential_scale_reduction",
        "effective_sample_size",
    }
    assert not {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_calls
    }
    text = path.read_text(encoding="utf-8")
    assert "tune_fixed_transport_hmc_kernel" not in text
    assert "FixedTransportHMCKernelTuningConfig" not in text
    assert "tune_hmc_kernel" in text


def test_new_runner_uses_public_fixed_identity_policy_and_native_caps() -> None:
    text = (ROOT / "bayesfilter/inference/neutra_end_to_end.py").read_text(
        encoding="utf-8"
    )
    assert 'mass_policy="fixed_identity"' in text
    assert "warmup_max_results=10000" in text
    assert "retained_max_results=10000" in text
    assert "rank_normalized_hmc_diagnostics" in text


def test_final_training_uses_core_recoverable_segment_api() -> None:
    path = ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    text = path.read_text(encoding="utf-8")
    assert "train_plain_dense_iaf_infrastructure_segments(" in text
    assert "FINAL_SEGMENT_STEPS = 1000" in text
    assert '"terminal_only_freeze": True' in text


def test_screen_only_stops_before_final_training_and_hmc() -> None:
    path = ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    text = path.read_text(encoding="utf-8")
    screen_branch = text.index("if config.screen_only:")
    final_call = text.index("final = _train_final(")
    assert screen_branch < final_call
    assert '"final_training_launched": False' in text
    assert '"hmc_tuning_launched": False' in text
    assert '"hmc_sampling_launched": False' in text
    assert '"status": "terminal" if terminal else "started"' in text
    assert '"result_file": str(root / "result.json")' in text


def test_heldout_score_uses_combined_status_and_hard_veto() -> None:
    path = ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    text = path.read_text(encoding="utf-8")
    score_source = text.split("def _score_training", 1)[1].split(
        "def _train_final", 1
    )[0]
    assert "transformed.log_prob_and_grad_status(z_batch)" in score_source
    assert '"heldout_target_status_nonvalid"' in score_source
    assert '"invalid_row_count"' in score_source


def test_training_hard_gates_aggregate_every_compiled_step() -> None:
    path = ROOT / "bayesfilter/inference/neutra_training_legacy.py"
    text = path.read_text(encoding="utf-8")
    assert '"all_steps_hard_gates_aggregated": True' in text
    assert '"all_steps_target_status_valid"' in text
    assert '"all_steps_target_status_nonvalid_count"' in text
    assert "if not all_steps_target_status_valid:" in text


def test_preflight_uses_real_hmc_runner_and_complete_status_telemetry() -> None:
    import tensorflow as tf

    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    path = ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    text = path.read_text(encoding="utf-8")
    assert "fake_chain" not in text
    assert "native_hmc_runner_executed" in text
    required = {
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    }
    for spec in EXECUTABLE_CELLS:
        adapter = BatchNativeBoundAdapter(
            spec.adapter_factory(), target_signature=spec.target_signature
        )
        telemetry = adapter.target_status_telemetry(
            tf.zeros((4, spec.parameter_dim), tf.float64)
        )
        assert required <= set(telemetry)


def test_batch_native_bound_adapter_preserves_scalar_and_batch_routes() -> None:
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == "LGSSM-EXACT")
    base = spec.adapter_factory()
    adapter = BatchNativeBoundAdapter(
        base,
        target_signature=spec.target_signature,
    )

    scalar_position = tf.zeros((spec.parameter_dim,), tf.float64)
    expected_value, expected_score = base.log_prob_and_grad(scalar_position)
    value, score = adapter.log_prob_and_grad(scalar_position)
    tf.debugging.assert_near(value, expected_value)
    tf.debugging.assert_near(score, expected_score)
    assert tuple(score.shape) == (spec.parameter_dim,)
    assert "status_code" in adapter.target_status_telemetry(scalar_position)

    batch_position = tf.zeros((2, spec.parameter_dim), tf.float64)
    expected_values, expected_scores, _ = adapter.binding.invoke(batch_position)
    values, scores = adapter.log_prob_and_grad(batch_position)
    tf.debugging.assert_near(values, expected_values)
    tf.debugging.assert_near(scores, expected_scores)
    assert tuple(values.shape) == (2,)
    assert tuple(scores.shape) == (2, spec.parameter_dim)
    assert "status_code" in adapter.target_status_telemetry(batch_position)


def test_batch_native_bound_adapter_lifts_rank1_hmc_state_for_pp_ukf() -> None:
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == "PP-UKF")
    adapter = BatchNativeBoundAdapter(
        spec.adapter_factory(), target_signature=spec.target_signature
    )
    value, score = adapter.log_prob_and_grad(
        tf.zeros((spec.parameter_dim,), tf.float64)
    )
    assert value.shape.rank == 0
    assert tuple(score.shape) == (spec.parameter_dim,)
    telemetry = adapter.target_status_telemetry(
        tf.zeros((spec.parameter_dim,), tf.float64)
    )
    assert "status_code" in telemetry
    assert telemetry["status_code"].shape.rank == 0


def test_pp_ukf_bound_adapter_advertises_retained_flat_and_combined_status() -> None:
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == "PP-UKF")
    adapter = BatchNativeBoundAdapter(
        spec.adapter_factory(), target_signature=spec.target_signature
    )
    assert adapter.supports_retained_flat_batch is True
    assert adapter.supports_retained_value_score_status is True
    values, scores, status = adapter.log_prob_and_grad_status(
        tf.zeros((2, spec.parameter_dim), tf.float64)
    )
    assert tuple(values.shape) == (2,)
    assert tuple(scores.shape) == (2, spec.parameter_dim)
    assert "status_code" in status


def test_preflight_accepts_structured_candidate_rejection_only_after_real_hmc() -> None:
    text = (ROOT / "bayesfilter/inference/neutra_end_to_end.py").read_text(
        encoding="utf-8"
    )
    for required in (
        'tuning_payload.get("schema") != "bayesfilter.hmc_kernel_tuning_public_artifact.v1"',
        'bootstrap.get("preflight_passed") is True',
        'bootstrap.get("round_timing_available") is True',
        'bootstrap.get("observed_acceptance_relations")',
        '"candidate_tuning_passed": tiny_tuning.passed is True',
        '"native_tuning_scientific_pass_required": False',
    ):
        assert required in text


def test_runner_exposes_training_throughput_without_hmc() -> None:
    runner = (
        ROOT / "docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py"
    ).read_text(encoding="utf-8")
    implementation = (
        ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    ).read_text(encoding="utf-8")
    assert '"training-throughput"' in runner
    assert 'parser.add_argument("--throughput-steps", type=int, default=25)' in runner
    assert "run_neutra_training_throughput_cell" in implementation
    assert '"hmc_tuning_launched": False' in implementation
    assert '"hmc_sampling_launched": False' in implementation
    throughput_source = implementation.split(
        "def run_neutra_training_throughput_cell", 1
    )[1].split("def _preflight_bootstrap_mechanics_passed", 1)[0]
    assert "tune_hmc_kernel(" not in throughput_source
    assert "run_sequential_neutra_hmc(" not in throughput_source
    assert '"TRAINING_CANDIDATE_HARD_VETO"' in throughput_source
    assert 'atomic_write_json(root / "failure.json", failure)' in throughput_source


def test_training_throughput_projection_is_conservative() -> None:
    from bayesfilter.inference.neutra_end_to_end import _throughput_timing_summary

    summary = _throughput_timing_summary(
        (
            {
                "steps": 1,
                "training_program_seconds": 8.0,
                "end_to_end_seconds": 10.0,
            },
            {
                "steps": 25,
                "training_program_seconds": 20.0,
                "end_to_end_seconds": 25.0,
            },
        )
    )
    assert summary["program_difference_step_seconds"] == 0.5
    assert summary["end_to_end_difference_step_seconds"] == 0.625
    assert summary["conservative_additional_step_seconds"] == 0.8


def test_manifest_dependency_and_segmented_api_are_available() -> None:
    import bayesfilter
    import bayesfilter.inference as inference

    module = ast.parse(
        (ROOT / "bayesfilter/inference/neutra_end_to_end.py").read_text(
            encoding="utf-8"
        )
    )
    imported_names = {
        alias.name
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "subprocess" in imported_names
    assert callable(inference.train_plain_dense_iaf_infrastructure_segments)
    assert callable(bayesfilter.train_plain_dense_iaf_infrastructure_segments)


def test_campaign_supports_fresh_unfinished_cell_subset() -> None:
    path = ROOT / "docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py"
    text = path.read_text(encoding="utf-8")
    assert 'parser.add_argument("--cells", nargs="+")' in text
    assert '"campaign_scope_cell_ids": selected_cell_ids' in text
    assert 'len(executable_rows)' in text


def test_campaign_streams_child_logs_and_records_terminal_state() -> None:
    path = ROOT / "docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py"
    text = path.read_text(encoding="utf-8")
    assert "subprocess.Popen" in text
    assert 'launch-logs' in text
    assert 'f"{cell}-launch-state-terminal.json"' in text
    assert "campaign_state_terminal.json" in text
    assert "stdout=log_stream" in text
    assert "stderr=subprocess.STDOUT" in text


def test_neutra_cell_has_diagnostic_run_state_and_terminal_result_precedence() -> None:
    path = ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    text = path.read_text(encoding="utf-8")
    assert 'bayesfilter.neutra.all_models.run_state.v1' in text
    assert 'terminal_result_authority' in text
    assert 'terminal_result_written' in text
    assert 'returned_without_terminal_result' in text
    assert 'status="exception"' in text


def test_runner_supports_admitted_kernel_replay_validation() -> None:
    path = ROOT / "docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py"
    text = path.read_text(encoding="utf-8")
    assert '"validate-frozen"' in text
    assert 'parser.add_argument("--frozen-transport", type=Path)' in text
    assert 'parser.add_argument("--frozen-transport-sha256")' in text
    assert 'parser.add_argument("--admitted-kernel-replay", type=Path)' in text
    assert 'parser.add_argument("--seed-offset", type=int, default=0)' in text
    assert '"--tuning-only"' in text
    implementation = (
        ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    ).read_text(encoding="utf-8")
    assert "run_neutra_frozen_transport_validation_cell" in implementation
    assert "expected_frozen_transport_sha256" in implementation
    assert "admitted_kernel_replay_path" in implementation
    assert "admitted_kernel_mechanics_payload_from_tuning_result" in implementation
    assert "build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload" in implementation
    assert "tune_hmc_kernel(" in implementation
    assert "run_sequential_neutra_hmc(" in implementation
    assert implementation.count("adapter=replay.adapter") == 2


def test_runner_exposes_broad_grid_as_explicit_tuning_only_action() -> None:
    runner = (
        ROOT / "docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py"
    ).read_text(encoding="utf-8")
    implementation = (
        ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    ).read_text(encoding="utf-8")
    broad = (ROOT / "bayesfilter/inference/neutra_broad_grid.py").read_text(
        encoding="utf-8"
    )
    shared = (
        ROOT / "bayesfilter/inference/neutra_shared_procedure.py"
    ).read_text(encoding="utf-8")
    assert '"broad-grid-frozen"' in runner
    assert 'parser.add_argument("--broad-grid-root-seed", nargs=2, type=int)' in runner
    assert "run_neutra_frozen_transport_broad_grid_cell" in implementation
    assert "run_shared_neutra_procedure" in implementation
    assert "run_neutra_operational_broad_grid_tuning" in shared
    assert '"sampling_launched": False' in implementation or '"sampling_launched": sequential_result is not None' in shared
    assert '"retained_sampling_authorized": False' in shared
    assert "run_operational_broad_grid(" in broad
    assert "tune_hmc_kernel(" not in broad
    assert "run_sequential_neutra_hmc(" not in broad


def test_broad_grid_config_rejects_bad_hash_seed_and_cpu_claim() -> None:
    from bayesfilter.inference.neutra_end_to_end import (
        FrozenTransportBroadGridConfig,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        FrozenTransportBroadGridConfig(
            output_root=ROOT / "unused",
            frozen_transport_path=ROOT / "unused.json",
            expected_frozen_transport_sha256="bad",
            root_seed=(1, 2),
        )
    with pytest.raises(ValueError, match="root_seed"):
        FrozenTransportBroadGridConfig(
            output_root=ROOT / "unused",
            frozen_transport_path=ROOT / "unused.json",
            expected_frozen_transport_sha256="a" * 64,
            root_seed=(-1, 2),
        )
    with pytest.raises(ValueError, match="GPU/XLA"):
        FrozenTransportBroadGridConfig(
            output_root=ROOT / "unused",
            frozen_transport_path=ROOT / "unused.json",
            expected_frozen_transport_sha256="a" * 64,
            root_seed=(1, 2),
            require_gpu=True,
            jit_compile=False,
        )


def test_frozen_transport_validation_config_rejects_bad_hashes() -> None:
    from bayesfilter.inference.neutra_end_to_end import (
        FrozenTransportValidationConfig,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        FrozenTransportValidationConfig(
            output_root=ROOT / "unused",
            frozen_transport_path=ROOT / "unused.json",
            expected_frozen_transport_sha256="not-a-digest",
        )

    config = FrozenTransportValidationConfig(
        output_root=ROOT / "unused",
        frozen_transport_path=ROOT / "unused.json",
        expected_frozen_transport_sha256="a" * 64,
        admitted_kernel_replay_path=ROOT / "kernel.json",
    )
    assert config.admitted_kernel_replay_path == ROOT / "kernel.json"

    tuning_only = FrozenTransportValidationConfig(
        output_root=ROOT / "unused",
        frozen_transport_path=ROOT / "unused.json",
        expected_frozen_transport_sha256="b" * 64,
        tuning_only=True,
    )
    assert tuning_only.tuning_only is True
    with pytest.raises(ValueError, match="tuning_only"):
        FrozenTransportValidationConfig(
            output_root=ROOT / "unused",
            frozen_transport_path=ROOT / "unused.json",
            expected_frozen_transport_sha256="c" * 64,
            admitted_kernel_replay_path=ROOT / "kernel.json",
            tuning_only=True,
        )


def test_frozen_transport_validation_config_normalizes_transport_hash() -> None:
    from bayesfilter.inference.neutra_end_to_end import (
        FrozenTransportValidationConfig,
    )

    config = FrozenTransportValidationConfig(
        output_root=ROOT / "unused",
        frozen_transport_path=ROOT / "unused.json",
        expected_frozen_transport_sha256="A" * 64,
    )
    assert config.expected_frozen_transport_sha256 == "a" * 64


def test_admitted_kernel_replay_is_scoped_to_frozen_validation() -> None:
    path = ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: ast.get_source_segment(path.read_text(encoding="utf-8"), node)
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    frozen = functions["run_neutra_frozen_transport_validation_cell"]
    trained = functions["run_neutra_end_to_end_cell"]
    assert frozen is not None and "admitted_kernel_replay_path" in frozen
    assert trained is not None and "admitted_kernel_replay_path" not in trained


def test_tuning_only_frozen_validation_returns_before_sequential_sampling() -> None:
    path = ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    text = path.read_text(encoding="utf-8")
    marker = "if config.tuning_only:"
    assert marker in text
    tuning_only_block = text.split(marker, 1)[1].split(
        "tuned_adapter = _fixed_transport_adapter", 1
    )[0]
    assert "run_sequential_neutra_hmc" not in tuning_only_block
    assert '"sampling_launched": False' in tuning_only_block


def test_admitted_kernel_mechanics_fingerprint_is_not_lineage_hash() -> None:
    from bayesfilter.inference.hmc_kernel_tuning import (
        ADMITTED_KERNEL_MECHANICS_SCHEMA,
    )
    from bayesfilter.runtime import stable_config_hash

    mechanics = {
        "schema": "bayesfilter.admitted_hmc_kernel_mechanics.v1",
        "step_size": 0.5,
        "num_leapfrog_steps": 3,
    }
    payload = {
        "schema": ADMITTED_KERNEL_MECHANICS_SCHEMA,
        "mechanics": mechanics,
        "mechanics_sha256": stable_config_hash(mechanics),
        "tuning_provenance": {"bootstrap_artifact_hash": "run-one"},
    }
    changed_lineage = copy.deepcopy(payload)
    changed_lineage["tuning_provenance"]["bootstrap_artifact_hash"] = "run-two"
    assert payload["mechanics_sha256"] == stable_config_hash(
        payload["mechanics"]
    )
    assert changed_lineage["mechanics_sha256"] == payload["mechanics_sha256"]


def _synthetic_admitted_kernel_fixture(*, adapter_signature: str = "base-v1"):
    import tensorflow as tf

    import bayesfilter.inference.hmc_kernel_tuning as kernel_tuning
    from bayesfilter.inference.hmc import PrecomputedMassArtifact, stable_adapter_signature
    from bayesfilter.inference.hmc_budget_ladder import _build_fixed_mass_hmc_adapter
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    from bayesfilter.runtime import stable_config_hash

    scope = "synthetic:fixed_neutra_native_tuning"

    class Adapter:
        parameter_dim = 2

        def adapter_signature(self):
            return adapter_signature

        def value_score_capability(self):
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,
                runtime_backend="synthetic_admitted_kernel_fixture",
                target_scope=scope,
                nonclaims=("test fixture only",),
            )

        def log_prob_and_grad(self, theta):
            value = tf.convert_to_tensor(theta, tf.float64)
            return -0.5 * tf.reduce_sum(value * value, axis=-1), -value

    adapter = Adapter()
    identity = [[1.0, 0.0], [0.0, 1.0]]
    initial_mass = PrecomputedMassArtifact(
        position=[0.0, 0.0],
        covariance=identity,
        factor=identity,
        adapter_signature=stable_adapter_signature(adapter),
        position_role="initial_position",
        covariance_source="fixed_identity",
        matrix_used_for_square_root="identity",
        source="synthetic_fixture",
    )
    initial_signature = kernel_tuning._mass_artifact_signature(initial_mass)
    phase4 = kernel_tuning._build_bootstrap_fixed_mass_adapter(
        adapter=adapter,
        mass_artifact=initial_mass,
        mass_signature=initial_signature,
        target_scope=scope,
    )
    phase4_signature = stable_adapter_signature(phase4)
    adapted_mass = PrecomputedMassArtifact(
        position=[0.0, 0.0],
        covariance=identity,
        factor=identity,
        adapter_signature=phase4_signature,
        position_role="initial_position",
        covariance_source="fixed_identity",
        matrix_used_for_square_root="identity",
        source="synthetic_fixture",
    )
    adapted_signature = kernel_tuning._mass_artifact_signature(adapted_mass)
    final_adapter = _build_fixed_mass_hmc_adapter(
        adapter=phase4,
        mass_artifact=adapted_mass,
        mass_signature=adapted_signature,
        target_scope=scope,
    )
    execution = {
        "dtype": "float64",
        "backend": "tensorflow_probability",
        "jit_compile": True,
        "tf32_execution_enabled": True,
        "mass_policy": "fixed_identity",
    }
    mechanics = {
        "schema": "bayesfilter.admitted_hmc_kernel_mechanics.v1",
        "target_signature": "synthetic-target-v1",
        "target_scope": scope,
        "target_dimension": 2,
        "base_adapter_signature": stable_adapter_signature(adapter),
        "phase4_hmc_adapter_signature": phase4_signature,
        "final_hmc_adapter_signature": stable_adapter_signature(final_adapter),
        "mass_policy": "fixed_identity",
        "initial_mass_artifact_payload": initial_mass.to_payload(include_arrays=True),
        "adapted_mass_artifact_payload": adapted_mass.to_payload(include_arrays=True),
        "step_size": 0.5,
        "num_leapfrog_steps": 3,
        "target_accept_prob": 0.70,
        "acceptance_band": [0.65, 0.75],
        "execution": execution,
    }
    artifact = {
        "schema": kernel_tuning.ADMITTED_KERNEL_MECHANICS_SCHEMA,
        "mechanics": mechanics,
        "mechanics_sha256": stable_config_hash(mechanics),
        "tuning_provenance": {"bootstrap_artifact_hash": "run-one"},
    }
    return adapter, scope, execution, artifact


def _replay_synthetic_admitted_kernel(adapter, scope, execution, artifact):
    import tensorflow as tf

    from bayesfilter.inference.hmc_kernel_tuning import (
        build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload,
    )

    return build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload(
        adapter=adapter,
        mechanics_payload=artifact,
        initial_position=tf.zeros((2,), tf.float64),
        target_signature="synthetic-target-v1",
        target_scope=scope,
        execution=execution,
        target_accept_prob=0.70,
        acceptance_band=(0.65, 0.75),
    )


def test_admitted_kernel_replay_ignores_lineage_and_reconstructs_mechanics() -> None:
    adapter, scope, execution, artifact = _synthetic_admitted_kernel_fixture()
    changed_lineage = copy.deepcopy(artifact)
    changed_lineage["tuning_provenance"]["bootstrap_artifact_hash"] = "run-two"
    replay = _replay_synthetic_admitted_kernel(
        adapter, scope, execution, changed_lineage
    )
    assert replay.step_size == 0.5
    assert replay.num_leapfrog_steps == 3


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("step_size",), 0.6),
        (("num_leapfrog_steps",), 4),
        (("adapted_mass_artifact_payload", "factor", 0, 0), 2.0),
    ),
)
def test_admitted_kernel_replay_rejects_mechanics_tampering(path, value) -> None:
    adapter, scope, execution, artifact = _synthetic_admitted_kernel_fixture()
    changed = copy.deepcopy(artifact)
    target = changed["mechanics"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="fingerprint"):
        _replay_synthetic_admitted_kernel(adapter, scope, execution, changed)


def test_admitted_kernel_replay_rejects_context_mismatches() -> None:
    adapter, scope, execution, artifact = _synthetic_admitted_kernel_fixture()
    with pytest.raises(ValueError, match="target_signature"):
        from bayesfilter.inference.hmc_kernel_tuning import (
            build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload,
        )
        import tensorflow as tf

        build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload(
            adapter=adapter,
            mechanics_payload=artifact,
            initial_position=tf.zeros((2,), tf.float64),
            target_signature="other-target",
            target_scope=scope,
            execution=execution,
            target_accept_prob=0.70,
            acceptance_band=(0.65, 0.75),
        )
    with pytest.raises(ValueError, match="execution settings"):
        _replay_synthetic_admitted_kernel(
            adapter, scope, {**execution, "jit_compile": False}, artifact
        )
    other_adapter, _, _, _ = _synthetic_admitted_kernel_fixture(
        adapter_signature="base-v2"
    )
    with pytest.raises(ValueError, match="base adapter signature"):
        _replay_synthetic_admitted_kernel(other_adapter, scope, execution, artifact)


def test_sequential_config_applies_seed_offset_only_to_sampling_seeds() -> None:
    from types import SimpleNamespace

    from bayesfilter.inference.neutra_end_to_end import _sequential_config
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == "LGSSM-EXACT")
    config = _sequential_config(
        SimpleNamespace(step_size=0.5, num_leapfrog_steps=3),
        spec,
        seed_offset=1000,
    )
    assert config.warmup_seed == (20260718, spec.initial_seed[1] + 1100)
    assert config.retained_seed == (20260718, spec.initial_seed[1] + 1101)


def test_pp_sgqf_identity_does_not_hash_device_specific_tensor_bytes() -> None:
    path = ROOT / "bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py"
    text = path.read_text(encoding="utf-8")
    assert "cloud_manifest = dict(cloud.manifest_payload())" in text
    assert '"cloud_construction_manifest_hash"' in text
    assert "serialize_tensor(cloud.points)" not in text


def test_plan_requires_native_tuning_and_explicit_caps() -> None:
    text = (
        ROOT
        / "docs/plans/bayesfilter-neutra-all-executable-models-end-to-end-python-plan-2026-07-18.md"
    ).read_text(encoding="utf-8")
    for required in (
        "target_accept_prob=0.70",
        "acceptance band `[0.65,0.75]`",
        "fixed identity mass in trained `z`",
        "10,000 draws per chain",
        "p_truth >= 0.05",
        "Blocked cells must appear",
    ):
        assert required in text


@pytest.mark.parametrize(
    "cell_id",
    (
        "LGSSM-EXACT",
        "PP-UKF",
        "PP-SGQF",
        "SIR-SGQF",
        "STR-UKF",
    ),
)
def test_current_target_factories_bind_batch_native_surface(cell_id: str) -> None:
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
    from bayesfilter.inference.neutra_end_to_end import _target_signature
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    spec = next(item for item in EXECUTABLE_CELLS if item.cell_id == cell_id)
    adapter = spec.adapter_factory()
    assert _target_signature(adapter) == spec.target_signature
    binding = bind_batch_native_neutra_target(
        adapter, target_signature=spec.target_signature
    )
    values = tf.zeros((4, spec.parameter_dim), tf.float64)
    value, score, status = binding.invoke(values)
    assert tuple(value.shape) == (4,)
    assert tuple(score.shape) == (4, spec.parameter_dim)
    assert "status_code" in status
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
