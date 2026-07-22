from __future__ import annotations

import copy
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference import neutra_hmc
from bayesfilter.testing import lgssm_neutra_gap_closure_tf as campaign


SCREEN_RESULT = campaign.ROOT / (
    "docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/"
    "screen-500/screen/candidates/wide_2x_lr5e3/attempt_1_graph_native/result.json"
)


def test_phase0_static_inputs_and_import_closure_are_bound() -> None:
    result = campaign.phase0_local_checks()

    assert result["passed"] is True
    assert result["selected_recipe"] == "wide_2x_lr5e3"
    assert len(result["comparator_parameter_names"]) == 18
    assert result["import_closure"]["passed"] is True


def test_selection_is_finite_health_checked_and_acceptance_only() -> None:
    rows = (
        {"step_size": 0.025, "acceptance_rate": 0.75, "health_passed": False},
        {"step_size": 0.05, "acceptance_rate": 0.64, "health_passed": True},
        {"step_size": 0.1, "acceptance_rate": 0.76, "health_passed": True},
    )

    selected = campaign.select_tuning_candidate(rows)

    assert selected is not None
    assert selected["step_size"] == 0.1
    assert campaign.select_tuning_candidate(
        ({"acceptance_rate": 0.75, "health_passed": False},)
    ) is None


def test_tuning_admission_requires_fresh_1000_draw_modern_rhat() -> None:
    draws = tf.random.stateless_normal(
        (1000, 4, 2), seed=(20260715, 10), dtype=tf.float64
    )
    rows = ({"step_size": 0.1, "acceptance_rate": 0.72, "health_passed": True},)

    admitted = campaign.tuning_admission(
        probe_rows=rows,
        verification_samples=draws,
        parameter_names=("x", "y"),
        verification_health={"health_passed": True},
    )
    blocked = campaign.tuning_admission(
        probe_rows=rows,
        verification_samples=draws[:999],
        parameter_names=("x", "y"),
        verification_health={"health_passed": True},
    )

    assert admitted["admitted"] is True
    assert admitted["verification_modern_rhat"]["rhat_definition"] == (
        "max(rank-normalized split R-hat, folded rank-normalized split R-hat)"
    )
    assert blocked["admitted"] is False


def test_tuning_admission_blocks_folded_scale_mismatch_despite_acceptance() -> None:
    draws = tf.random.stateless_normal(
        (1000, 4, 2), seed=(20260715, 11), dtype=tf.float64
    )
    draws = draws * tf.constant([0.4, 1.0, 2.5, 5.0], tf.float64)[
        tf.newaxis, :, tf.newaxis
    ]

    result = campaign.tuning_admission(
        probe_rows=(
            {"step_size": 0.1, "acceptance_rate": 0.72, "health_passed": True},
        ),
        verification_samples=draws,
        parameter_names=("x", "y"),
        verification_health={"health_passed": True},
    )

    assert result["admitted"] is False
    assert result["verification_modern_rhat"][
        "max_folded_rank_normalized_split_rhat"
    ] > 1.01


def test_tensor_archive_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    values = tf.reshape(tf.range(24, dtype=tf.float64), (3, 4, 2))
    tensor_path = tmp_path / "samples.tftensor"

    sidecar = campaign.write_tensor_archive(
        tensor_path,
        values,
        metadata={"candidate_id": "unit"},
    )
    restored = campaign.read_tensor_archive(tensor_path.with_suffix(".tftensor.json"))

    tf.debugging.assert_equal(restored, values)
    assert sidecar["shape"] == (3, 4, 2)
    with pytest.raises(FileExistsError):
        campaign.write_tensor_archive(tensor_path, values, metadata={})


def test_comparator_summary_binds_modern_diagnostics_and_parameter_order() -> None:
    comparator = campaign.load_plain_hmc_comparator_summary()

    assert comparator["diagnostics"]["passed"] is True
    assert comparator["posterior_mean"].shape == (18,)
    assert comparator["mean_mcse"].shape == (18,)
    assert comparator["parameter_names"][0] == "a11_raw"
    assert comparator["parameter_names"][-1] == "log_r4"


def test_strict_result_adapter_accepts_current_schema_fixture(tmp_path: Path) -> None:
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    result.update(
        {
            "job_kind": "final",
            "job_id": "dense_seed1201",
            "steps": 5000,
            "planned_steps": 5000,
            "selected_recipe_source": {
                "selected_recipe": {
                    "path": str(campaign.SELECTION_PATH.relative_to(campaign.ROOT)),
                    "file_sha256": campaign.EXPECTED_SELECTION_FILE_SHA256,
                    "byte_count": campaign.SELECTION_PATH.stat().st_size,
                },
                "selected_recipe_artifact_hash": (
                    "sha256:00bf189dd8697c33b0378bda92a75d2df74d85ffb0e754f1df7c6dabcb216ac0"
                ),
                "selection_result": {},
                "selection_result_artifact_hash": "fixture-only",
                "recipe_id": "wide_2x_lr5e3",
                "screen_weights_reused": False,
            },
        }
    )
    result["gpu_manifest"]["gpu_memory_policy"] = {
        "mode": "memory_growth",
        "full_device_preallocation_disabled": True,
    }
    result.pop("artifact_hash", None)
    result.pop("artifact_hash_semantics", None)
    result = dict(campaign._with_artifact_hash(result))
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    validated = campaign.validate_strict_training_result(
        path,
        expected_job_id="dense_seed1201",
    )

    assert validated["passed"] is True
    assert validated["recipe_id"] == "wide_2x_lr5e3"


def test_strict_result_adapter_rejects_recipe_drift(tmp_path: Path) -> None:
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    result["recipe"] = copy.deepcopy(result["recipe"])
    result["recipe"]["recipe_id"] = "source_anchor_lr5e3"
    result.update(
        {
            "job_kind": "final",
            "job_id": "dense_seed1201",
            "steps": 5000,
        }
    )
    result.pop("artifact_hash", None)
    result.pop("artifact_hash_semantics", None)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(campaign._with_artifact_hash(result)), encoding="utf-8")

    with pytest.raises(campaign.LGSSMNeuTraGapClosureError, match="recipe"):
        campaign.validate_strict_training_result(path, expected_job_id="dense_seed1201")


def test_posterior_summary_matches_identical_comparator_moments() -> None:
    comparator = campaign.load_plain_hmc_comparator_summary()
    base = comparator["posterior_mean"][tf.newaxis, tf.newaxis, :]
    noise = tf.random.stateless_normal(
        (1000, 4, 18), seed=(20260715, 12), dtype=tf.float64
    ) * comparator["posterior_sd"][tf.newaxis, tf.newaxis, :]
    samples = base + noise

    summary = campaign.posterior_summary(
        candidate_samples=samples,
        parameter_names=comparator["parameter_names"],
        comparator=comparator,
    )

    assert summary["all_finite"] is True
    assert len(summary["parameter_rows"]) == 18


def test_gpu_probe_configures_memory_before_loading_candidate(monkeypatch, tmp_path) -> None:
    events = []

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "bayesfilter.runtime.gpu_memory_policy.configure_tensorflow_gpu_memory_growth",
        lambda _tf, require_gpu: events.append(("memory", require_gpu)) or {"mode": "memory_growth"},
    )
    monkeypatch.setattr(
        campaign,
        "_load_fresh_candidate",
        lambda _candidate: events.append(("load", True)) or (_ for _ in ()).throw(
            RuntimeError("stop after order check")
        ),
    )

    with pytest.raises(RuntimeError, match="order check"):
        campaign.run_fresh_frozen_objective_probe(
            "dense_seed1201",
            device_mode="trusted_gpu_xla",
            output_path=tmp_path / "unused.json",
        )

    assert events == [("memory", True), ("load", True)]


def test_batched_hmc_records_adapter_target_status() -> None:
    class StatusGaussian:
        @staticmethod
        def log_prob_and_grad(theta):
            values = tf.convert_to_tensor(theta, tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

        @staticmethod
        def target_status_telemetry(theta):
            leading = tf.shape(tf.convert_to_tensor(theta))[:-1]
            return {
                "status_code": tf.zeros(leading, tf.int32),
                "valid_pre_regularized_score": tf.ones(leading, tf.bool),
                "floor_count_value": tf.zeros(leading, tf.int32),
                "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
                "innovation_condition_estimate": tf.ones(leading, tf.float64),
            }

    run = campaign.run_batched_hmc(
        adapter=StatusGaussian(),
        initial_state=tf.constant(
            [[-1.0, 0.5], [-0.5, -1.0], [0.5, 1.0], [1.0, -0.5]],
            tf.float64,
        ),
        config=campaign.TensorHMCConfig(
            num_results=8,
            num_burnin_steps=4,
            step_size=0.4,
            num_leapfrog_steps=2,
            seed=(20260715, 20),
        ),
    )

    telemetry = run["diagnostics"]["target_status_telemetry"]
    assert telemetry["available"] is True
    assert telemetry["all_status_valid"] is True
    assert telemetry["status_nonvalid_count"] == 0
    assert run["diagnostics"]["health_passed"] is True


def _install_fake_sequential_hmc(monkeypatch):
    calls = []

    def build_program(**kwargs):
        draws = int(kwargs["num_results"])

        def compiled(current_state, seed):
            state = tf.convert_to_tensor(current_state, tf.float64)
            seed_tensor = tf.convert_to_tensor(seed, tf.int32)
            offset = tf.cast(tf.range(1, draws + 1), tf.float64)[:, tf.newaxis, tf.newaxis]
            samples = state[tf.newaxis, :, :] + 0.01 * offset
            calls.append(tuple(int(item) for item in seed_tensor.numpy().tolist()))
            trace = {
                "is_accepted": tf.ones((draws, 4), tf.bool),
                "log_accept_ratio": tf.zeros((draws, 4), tf.float64),
                "target_log_prob": -tf.reduce_sum(tf.square(samples), axis=-1),
            }
            return samples, trace

        return compiled

    monkeypatch.setattr(neutra_hmc, "_build_batched_hmc_program", build_program)
    return calls


def _script_rhat(monkeypatch, passed_values):
    values = iter(tuple(bool(item) for item in passed_values))

    def summary(draws, *, rhat_max):
        passed = next(values)
        return {
            "passed": passed,
            "rhat_definition": (
                "max(rank-normalized split R-hat, "
                "folded rank-normalized split R-hat)"
            ),
            "rhat_threshold": float(rhat_max),
            "draw_count_per_chain": int(tf.convert_to_tensor(draws).shape[0]),
            "max_rank_normalized_split_rhat": 1.0 if passed else 1.2,
            "max_folded_rank_normalized_split_rhat": 1.0 if passed else 1.1,
            "max_finite_rhat": 1.0 if passed else 1.2,
            "finite_rhat_count": int(tf.convert_to_tensor(draws).shape[-1]),
            "nonfinite_rhat_count": 0,
        }

    monkeypatch.setattr(neutra_hmc, "rank_normalized_split_rhat_summary", summary)


def _tiny_sequential_config(**overrides):
    values = {
        "step_size": 0.4,
        "num_leapfrog_steps": 2,
        "warmup_seed": (20260715, 101),
        "retained_seed": (20260715, 201),
        "warmup_chunk_results": 4,
        "warmup_min_results": 4,
        "warmup_check_window_results": 4,
        "warmup_max_results": 8,
        "warmup_rhat_max": 1.05,
        "retained_chunk_results": 4,
        "retained_min_results": 4,
        "retained_max_results": 8,
        "retained_rhat_max": 1.01,
        "jit_compile": True,
    }
    values.update(overrides)
    return campaign.SequentialNeuTraHMCConfig(**values)


def test_sequential_hmc_retains_warmup_then_extends_retained(monkeypatch, tmp_path) -> None:
    calls = _install_fake_sequential_hmc(monkeypatch)
    _script_rhat(monkeypatch, (False, True, False, True))

    result = campaign.run_sequential_neutra_hmc(
        adapter=object(),
        initial_state=tf.zeros((4, 2), tf.float64),
        raw_transform=lambda values: values,
        parameter_names=("x", "y"),
        config=_tiny_sequential_config(),
        archive_root=tmp_path / "sequential",
    )

    assert result["passed"] is True
    assert result["warmup_results_per_chain"] == 8
    assert result["retained_results_per_chain"] == 8
    assert result["warmup_samples_retained"] is True
    assert result["warmup_excluded_from_posterior"] is True
    assert result["private_warmup_raw"].shape == (8, 4, 2)
    assert result["private_retained_raw"].shape == (8, 4, 2)
    assert len(result["warmup_archives"]) == 2
    assert len(result["retained_archives"]) == 2
    assert result["cumulative_archives"]["warmup_raw"]["shape"] == (8, 4, 2)
    assert result["cumulative_archives"]["retained_raw"]["shape"] == (8, 4, 2)
    assert calls == [
        (20260715, 1110),
        (20260715, 2119),
        (20260715, 1210),
        (20260715, 2219),
    ]


def test_sequential_hmc_stops_at_warmup_cap_before_sampling(monkeypatch) -> None:
    _install_fake_sequential_hmc(monkeypatch)
    _script_rhat(monkeypatch, (False, False))

    result = campaign.run_sequential_neutra_hmc(
        adapter=object(),
        initial_state=tf.zeros((4, 2), tf.float64),
        raw_transform=lambda values: values,
        parameter_names=("x", "y"),
        config=_tiny_sequential_config(),
    )

    assert result["passed"] is False
    assert result["warmup_cap_hit"] is True
    assert result["retained_results_per_chain"] == 0
    assert result["retained_check_count"] == 0


def test_sequential_hmc_stops_retained_sampling_at_cap(monkeypatch) -> None:
    _install_fake_sequential_hmc(monkeypatch)
    _script_rhat(monkeypatch, (True, False, False))

    result = campaign.run_sequential_neutra_hmc(
        adapter=object(),
        initial_state=tf.zeros((4, 2), tf.float64),
        raw_transform=lambda values: values,
        parameter_names=("x", "y"),
        config=_tiny_sequential_config(),
    )

    assert result["passed"] is False
    assert result["warmup_results_per_chain"] == 4
    assert result["retained_results_per_chain"] == 8
    assert result["retained_cap_hit"] is True


def test_sequential_hmc_extends_on_full_convergence_not_rhat_alone(monkeypatch) -> None:
    _install_fake_sequential_hmc(monkeypatch)
    _script_rhat(monkeypatch, (True,))
    full_checks = iter(
        (
            {"passed": False, "max_rhat": 1.0, "min_bulk_ess": 2.0},
            {"passed": True, "max_rhat": 1.0, "min_bulk_ess": 10.0},
        )
    )

    result = campaign.run_sequential_neutra_hmc(
        adapter=object(),
        initial_state=tf.zeros((4, 2), tf.float64),
        raw_transform=lambda values: values,
        parameter_names=("x", "y"),
        config=_tiny_sequential_config(),
        retained_diagnostic_fn=lambda _draws: next(full_checks),
    )

    assert result["passed"] is True
    assert result["retained_results_per_chain"] == 8
    assert result["retained_checks"][0]["diagnostic_role"] == "full_convergence"
    assert "modern_rhat" not in result["retained_checks"][0]
    assert result["retained_checks"][1]["full_convergence"]["passed"] is True


def test_sequential_hmc_enforces_ten_thousand_sample_caps() -> None:
    with pytest.raises(ValueError, match="warmup_max_results"):
        _tiny_sequential_config(warmup_max_results=10001)
    with pytest.raises(ValueError, match="retained_max_results"):
        _tiny_sequential_config(retained_max_results=10001)


def test_confirmatory_summary_handles_pre_sampling_hard_veto(monkeypatch) -> None:
    monkeypatch.setattr(campaign, "CONFIRMATION_ROOT", campaign.ROOT / "confirmation")
    row = {
        "candidate_id": "dense_seed1202",
        "passed": False,
        "decision": "REJECT_CANDIDATE_CONFIRMATORY_GATES",
        "sequential_run": {
            "warmup_results_per_chain": 2000,
            "retained_results_per_chain": 0,
            "hard_vetoes": ("warmup_chunk_health_failed",),
        },
        "final_full_convergence": None,
        "posterior_summary": None,
        "artifact_hash": "sha256:test",
    }

    result = campaign._confirmatory_candidate_summary(row)

    assert result["hard_vetoes"] == ("warmup_chunk_health_failed",)
    assert result["max_rhat"] is None
    assert result["max_posterior_agreement_combined_mcse"] is None
