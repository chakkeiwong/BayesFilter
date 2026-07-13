from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.highdim.ledh_forward_contract import LGSSM_M3_T50_ROW_ID
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    validate_ledh_score_artifact,
)
from docs.benchmarks import benchmark_ledh_same_target_lgssm_m3_t50_value as lgssm
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


ROOT = Path(__file__).resolve().parents[2]
LGSSM_VALUE_PATH = ROOT / "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json"
LGSSM_VALUE_REL = "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json"
HISTORICAL_T2_SCORE_MEMORY_PATH = (
    ROOT / "docs/plans/ledh-phase5-lgssm-score-memory-n10000-2026-07-06.json"
)


def _load_value() -> dict:
    return json.loads(LGSSM_VALUE_PATH.read_text(encoding="utf-8"))


def _full_row_args(*, num_particles: int = 10000) -> SimpleNamespace:
    return SimpleNamespace(
        batch_seeds=list(lgssm.FULL_ROW_BATCH_SEEDS),
        num_particles=num_particles,
        time_steps=lgssm.FULL_ROW_TIME_STEPS,
        transport_policy=lgssm.FULL_ROW_TRANSPORT_POLICY,
        sinkhorn_iterations=lgssm.FULL_ROW_SINKHORN_ITERATIONS,
        sinkhorn_epsilon=lgssm.FULL_ROW_SINKHORN_EPSILON,
    )


def _raw_full_score_result() -> dict:
    return {
        "row_id": LGSSM_M3_T50_ROW_ID,
        "score_admission_status": lgssm.RAW_COMPACT_ADMITTED_STATUS,
        "score_route": lgssm.COMPACT_SCORE_ROUTE_ID,
        "value_score_route_status": "same_route_value_score",
        "score": [
            4.208085587610736,
            -0.3581815687179415,
            0.010309887507222504,
            4.628948101466079,
            11.065219890558737,
        ],
        "score_parameter_names": list(lgssm.PARAMETER_NAMES),
        "runtime_gate_applicable": True,
        "score_runtime_gate_applicable": True,
        "batch_seeds": list(lgssm.FULL_ROW_BATCH_SEEDS),
        "shape": {
            "batch_size": len(lgssm.FULL_ROW_BATCH_SEEDS),
            "batch_seed_count": len(lgssm.FULL_ROW_BATCH_SEEDS),
            "time_steps": 50,
            "num_particles": 10000,
            "state_dim": 3,
            "obs_dim": 3,
        },
        "target_identity": {
            "row_id": LGSSM_M3_T50_ROW_ID,
            "target_scalar": "observed_data_log_likelihood_estimator",
            "target_output_tensor_field": "log_likelihood",
            "full_leaderboard_row": True,
        },
        "manual_score_diagnostic": {
            "score_derivative_provenance": lgssm.COMPACT_SCORE_ROUTE_ID,
            "score_route": lgssm.COMPACT_SCORE_ROUTE_ID,
            "score": [
                4.208085587610736,
                -0.3581815687179415,
                0.010309887507222504,
                4.628948101466079,
                11.065219890558737,
            ],
            "uses_full_history_reverse_route": False,
            "old_full_history_route_status": "historical_full_history_reverse_route_not_used",
            "score_execution_style": "compact_forward_sensitivity_no_time_history",
            "same_scalar_fd": {
                "status": "pass",
                "step": 1.0e-5,
                "atol": 5.0e-3,
                "rtol": 5.0e-3,
                "max_abs_error": 4.55e-10,
                "max_relative_error": 1.21e-9,
                "tf32_mode": "disabled",
                "tf32_execution_enabled": False,
                "production_tf32_execution_enabled": True,
                "uses_disclosed_separate_precision_arm": True,
            },
            "no_autodiff_score_route": True,
        },
        "precision": {
            "dtype": "float32",
            "active_dtype": "float32",
            "tf_dtype": "float32",
            "tf32_mode": "enabled",
            "tf32_execution_enabled": True,
        },
        "gpu_memory_info_after": {"current": 0, "peak": int(20000.0 * 1024 * 1024)},
        "score_gpu_memory_info_after": {"current": 0, "peak": int(512.0 * 1024 * 1024)},
    }


def _raw_score_shard(seed: int) -> dict:
    raw = _raw_full_score_result()
    raw["score_admission_status"] = "blocked_material_gate_not_full_gpu_row"
    raw["batch_seeds"] = [int(seed)]
    raw["shape"] = {
        **raw["shape"],
        "batch_size": 1,
        "batch_seed_count": 1,
    }
    raw["target_identity"] = {
        **raw["target_identity"],
        "batch_seeds": [int(seed)],
        "full_leaderboard_row": False,
    }
    raw["finite_output"] = True
    raw["manual_score_diagnostic"] = copy.deepcopy(raw["manual_score_diagnostic"])
    raw["manual_score_diagnostic"]["score_derivative_provenance"] = lgssm.COMPACT_SCORE_ROUTE_ID
    raw["manual_score_diagnostic"]["score_route"] = lgssm.COMPACT_SCORE_ROUTE_ID
    raw["manual_score_diagnostic"]["score_output_devices"] = ["/device:GPU:0"]
    raw["manual_score_diagnostic"]["score"] = list(raw["score"])
    raw["manual_score_diagnostic"]["log_likelihood_by_seed"] = [-135.0 - (seed - 81120)]
    raw["manual_score_diagnostic"]["same_scalar_fd"] = copy.deepcopy(
        raw["manual_score_diagnostic"]["same_scalar_fd"]
    )
    raw["manual_score_diagnostic"]["same_scalar_fd"]["parameters"] = [
        {
            "parameter": name,
            "manual_score": float(value),
            "finite_difference": float(value),
            "abs_error": 0.0,
            "relative_error": 0.0,
        }
        for name, value in zip(lgssm.PARAMETER_NAMES, raw["score"], strict=True)
    ]
    raw["score_output_devices"] = ["/device:GPU:0"]
    raw["score_gpu_memory_info_after"] = {
        "current": 0,
        "peak": int((512.0 + (seed - 81120)) * 1024 * 1024),
    }
    return raw


def test_phase2_lgssm_full_row_identity_is_n10000_not_stale_n1000() -> None:
    assert lgssm.FULL_ROW_NUM_PARTICLES == 10000

    _tensors, full_identity = lgssm._build_lgssm_tensors(_full_row_args())
    assert full_identity["full_leaderboard_row"] is True
    assert full_identity["full_row_expected"]["num_particles"] == 10000
    assert full_identity["num_particles"] == 10000

    _tensors, stale_identity = lgssm._build_lgssm_tensors(
        _full_row_args(num_particles=1000)
    )
    assert stale_identity["full_leaderboard_row"] is False
    assert stale_identity["same_target_status"] == "prefix_diagnostic_not_full_leaderboard_row"


def test_phase2_lgssm_full_mode_manual_streaming_dispatches_to_total_vjp() -> None:
    assert (
        lgssm.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE
        == "manual_streaming_finite_sinkhorn_stopped_scale_keys"
    )

    source = Path(lgssm.core_tf.__file__).read_text(encoding="utf-8")
    assert "if transport_ad_mode == \"stabilized\"" in source
    assert "transport_fn = _filterflow_manual_streaming_finite_transport_stopped_scale_keys" in source
    assert "transport_fn = _filterflow_manual_streaming_finite_transport_total_vjp" in source

    compact_source = Path(annealed_transport_tf.__file__).read_text(encoding="utf-8")
    total_jvp_start = compact_source.index(
        "def _filterflow_manual_streaming_finite_transport_value_and_jvp_total"
    )
    total_jvp_end = compact_source.index(
        "def _filterflow_manual_streaming_finite_transport_total_pullback"
    )
    total_jvp_source = compact_source[total_jvp_start:total_jvp_end]
    assert "tf.stop_gradient" not in total_jvp_source
    assert "GradientTape" not in total_jvp_source


def test_phase2_lgssm_raw_full_score_normalizes_to_score_schema() -> None:
    value_artifact = _load_value()
    score_artifact = lgssm._lgssm_score_artifact_from_result(
        _raw_full_score_result(),
        source_value_artifact=value_artifact,
        source_value_artifact_path=LGSSM_VALUE_REL,
    )

    normalized = validate_ledh_score_artifact(
        score_artifact,
        source_value_artifact=value_artifact,
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=True,
    )

    assert normalized["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_FULL
    assert normalized["score_derivative_provenance"] == lgssm.COMPACT_SCORE_ROUTE_ID
    assert normalized["score_parameter_names"] == list(lgssm.PARAMETER_NAMES)
    assert normalized["score_precision"]["dtype"] == "float32"
    assert normalized["score_precision"]["tf32_execution_enabled"] is True
    assert normalized["memory_diagnostics"]["n10000_memory_pass"] is True
    assert normalized["score_correctness"]["tf32_mode"] == "disabled"
    assert normalized["score_correctness"]["uses_disclosed_separate_precision_arm"] is True


def test_phase2_lgssm_score_artifact_memory_uses_score_route_peak_not_value_peak() -> None:
    value_artifact = _load_value()
    raw = _raw_full_score_result()
    raw["gpu_memory_info_after"] = {"current": 0, "peak": int(20000.0 * 1024 * 1024)}
    raw["score_gpu_memory_info_after"] = {"current": 0, "peak": int(512.0 * 1024 * 1024)}

    score_artifact = lgssm._lgssm_score_artifact_from_result(
        raw,
        source_value_artifact=value_artifact,
        source_value_artifact_path=LGSSM_VALUE_REL,
    )

    assert score_artifact["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_FULL
    assert score_artifact["score_precision"]["tf32_mode"] == "enabled"
    assert score_artifact["memory_diagnostics"]["source"] == "score_gpu_memory_info_after"
    assert score_artifact["memory_diagnostics"]["peak_mib"] == 512.0


def test_phase2_lgssm_score_artifact_rejects_missing_score_memory_even_if_value_memory_passes() -> None:
    value_artifact = _load_value()
    raw = _raw_full_score_result()
    raw["gpu_memory_info_after"] = {"current": 0, "peak": int(512.0 * 1024 * 1024)}
    raw.pop("score_gpu_memory_info_after")

    score_artifact = lgssm._lgssm_score_artifact_from_result(
        raw,
        source_value_artifact=value_artifact,
        source_value_artifact_path=LGSSM_VALUE_REL,
    )

    assert score_artifact["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_TINY
    assert score_artifact["memory_diagnostics"]["n10000_memory_pass"] is False
    assert score_artifact["memory_diagnostics"]["source"] is None
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            score_artifact,
            source_value_artifact=value_artifact,
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase2_lgssm_adapter_rejects_stale_n1000_raw_result() -> None:
    raw = _raw_full_score_result()
    raw["shape"]["num_particles"] = 1000

    with pytest.raises(ValueError, match="num_particles"):
        lgssm._lgssm_score_artifact_from_result(
            raw,
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )


def test_phase2_lgssm_adapter_rejects_old_t2_raw_result() -> None:
    raw = _raw_full_score_result()
    raw["shape"]["time_steps"] = 2

    with pytest.raises(ValueError, match="time_steps"):
        lgssm._lgssm_score_artifact_from_result(
            raw,
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )


def test_phase2_lgssm_historical_t2_memory_artifact_is_not_admitted_schema() -> None:
    historical = json.loads(HISTORICAL_T2_SCORE_MEMORY_PATH.read_text(encoding="utf-8"))
    assert historical["row_id"] == LGSSM_M3_T50_ROW_ID
    assert historical["num_particles"] == 10000
    assert "time_steps" not in historical

    with pytest.raises(ValueError, match="schema_version"):
        validate_ledh_score_artifact(
            historical,
            source_value_artifact=_load_value(),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase2_lgssm_adapter_keeps_cpu_or_memory_miss_as_tiny_not_admitted() -> None:
    raw = copy.deepcopy(_raw_full_score_result())
    raw["runtime_gate_applicable"] = False

    score_artifact = lgssm._lgssm_score_artifact_from_result(
        raw,
        source_value_artifact=_load_value(),
        source_value_artifact_path=LGSSM_VALUE_REL,
    )

    assert score_artifact["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_TINY
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            score_artifact,
            source_value_artifact=_load_value(),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase2_lgssm_adapter_demotes_legacy_memory_style_status_to_tiny() -> None:
    raw = copy.deepcopy(_raw_full_score_result())
    raw["score_admission_status"] = lgssm.RAW_MEMORY_STYLE_ADMITTED_STATUS

    score_artifact = lgssm._lgssm_score_artifact_from_result(
        raw,
        source_value_artifact=_load_value(),
        source_value_artifact_path=LGSSM_VALUE_REL,
    )

    assert score_artifact["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_TINY
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            score_artifact,
            source_value_artifact=_load_value(),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


@pytest.mark.parametrize(
    ("nested_key", "nested_value", "match"),
    [
        (
            "score_derivative_provenance",
            lgssm.HISTORICAL_MANUAL_SCORE_ROUTE_ID,
            "compact derivative provenance",
        ),
        ("score_route", lgssm.HISTORICAL_MANUAL_SCORE_ROUTE_ID, "compact score route"),
        ("uses_full_history_reverse_route", True, "full-history reverse route"),
        (
            "old_full_history_route_status",
            "historical_full_history_reverse_route_used",
            "demote historical reverse route",
        ),
        (
            "score_execution_style",
            "historical_full_history_reverse_vjp_stores_time_history",
            "compact execution style",
        ),
    ],
)
def test_phase2_lgssm_adapter_rejects_nested_historical_relabeling(
    nested_key: str,
    nested_value,
    match: str,
) -> None:
    raw = copy.deepcopy(_raw_full_score_result())
    raw["manual_score_diagnostic"][nested_key] = nested_value

    with pytest.raises(ValueError, match=match):
        lgssm._lgssm_score_artifact_from_result(
            raw,
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )


def test_phase2_lgssm_adapter_rejects_outer_nested_score_mismatch() -> None:
    raw = copy.deepcopy(_raw_full_score_result())
    raw["manual_score_diagnostic"]["score"] = list(raw["score"])
    raw["manual_score_diagnostic"]["score"][0] += 1.0

    with pytest.raises(ValueError, match="nested diagnostic differ"):
        lgssm._lgssm_score_artifact_from_result(
            raw,
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )


def test_phase2v_lgssm_shard_aggregation_admits_only_full_fixed_seed_set() -> None:
    shards = [_raw_score_shard(seed) for seed in lgssm.FULL_ROW_BATCH_SEEDS]
    aggregate = lgssm._aggregate_lgssm_score_shards(
        shards,
        source_value_artifact=_load_value(),
        source_value_artifact_path=LGSSM_VALUE_REL,
    )

    score_artifact = aggregate["score_artifact"]
    normalized = validate_ledh_score_artifact(
        score_artifact,
        source_value_artifact=_load_value(),
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=True,
    )

    assert normalized["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_FULL
    assert normalized["score_precision"]["tf32_mode"] == "enabled"
    assert aggregate["execution_strategy"]["kind"] == "seed_sharded_trusted_gpu_processes"
    assert aggregate["execution_strategy"]["segmented_execution_disclosed"] is True
    assert aggregate["execution_strategy"]["monolithic_batch_memory_claim"] is False
    assert aggregate["memory_diagnostics"]["source"] == "max_per_seed_score_gpu_memory_info_after"
    assert aggregate["memory_diagnostics"]["peak_mib"] == 516.0
    assert aggregate["batch_seeds"] == list(lgssm.FULL_ROW_BATCH_SEEDS)


def test_phase2v_lgssm_shard_aggregation_rejects_missing_duplicate_or_admitted_shard() -> None:
    shards = [_raw_score_shard(seed) for seed in lgssm.FULL_ROW_BATCH_SEEDS]
    with pytest.raises(ValueError, match="missing"):
        lgssm._aggregate_lgssm_score_shards(
            shards[:-1],
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )

    duplicate = shards[:-1] + [_raw_score_shard(lgssm.FULL_ROW_BATCH_SEEDS[0])]
    with pytest.raises(ValueError, match="duplicate"):
        lgssm._aggregate_lgssm_score_shards(
            duplicate,
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )

    admitted_shard = copy.deepcopy(shards[0])
    admitted_shard["score_admission_status"] = lgssm.RAW_COMPACT_ADMITTED_STATUS
    with pytest.raises(ValueError, match="must not be marked as admitted"):
        lgssm._aggregate_lgssm_score_shards(
            [admitted_shard] + shards[1:],
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )


def test_phase2v_lgssm_shard_aggregation_rejects_identity_and_runtime_mismatches() -> None:
    shards = [_raw_score_shard(seed) for seed in lgssm.FULL_ROW_BATCH_SEEDS]

    stale_n = copy.deepcopy(shards)
    stale_n[0]["shape"]["num_particles"] = 9999
    with pytest.raises(ValueError, match="num_particles"):
        lgssm._aggregate_lgssm_score_shards(
            stale_n,
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )

    wrong_target = copy.deepcopy(shards)
    wrong_target[0]["target_identity"]["target_scalar"] = "exact_kalman_log_likelihood"
    with pytest.raises(ValueError, match="target_scalar"):
        lgssm._aggregate_lgssm_score_shards(
            wrong_target,
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )

    cpu_score = copy.deepcopy(shards)
    cpu_score[0]["score_runtime_gate_applicable"] = False
    with pytest.raises(ValueError, match="score runtime"):
        lgssm._aggregate_lgssm_score_shards(
            cpu_score,
            source_value_artifact=_load_value(),
            source_value_artifact_path=LGSSM_VALUE_REL,
        )


def test_phase2v_lgssm_shard_aggregation_matches_direct_batch_arithmetic_mean() -> None:
    args = lgssm.argparse.Namespace(
        batch_seeds=[81120, 81121],
        num_particles=4,
        time_steps=2,
        transport_policy="active-all",
        sinkhorn_iterations=2,
        sinkhorn_epsilon=0.5,
        annealed_scaling=0.9,
        annealed_convergence_threshold=1.0e-3,
        transport_gradient_mode=lgssm.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_ad_mode="full",
        row_chunk_size=4,
        col_chunk_size=4,
        particle_chunk_size=4,
        score_fd_step=1.0e-5,
        score_fd_atol=5.0e-3,
        score_fd_rtol=5.0e-3,
        score_fd_tf32_mode="match",
        dtype="float64",
        tf32_mode="disabled",
        score_mode="compact-sensitivity",
    )
    lgssm._configure_precision(args)
    theta = lgssm.tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE)
    direct = lgssm._manual_score_diagnostic(args, theta)
    shard_payloads = []
    for seed in args.batch_seeds:
        shard_args = copy.copy(args)
        shard_args.batch_seeds = [seed]
        shard_diagnostic = lgssm._manual_score_diagnostic(shard_args, theta)
        per_seed_fd = []
        for param_index, name in enumerate(lgssm.PARAMETER_NAMES):
            fd_value = shard_diagnostic["same_scalar_fd"]["parameters"][param_index][
                "finite_difference"
            ]
            per_seed_fd.append(
                {
                    "parameter": name,
                    "manual_score": shard_diagnostic["score"][param_index],
                    "finite_difference": fd_value,
                    "abs_error": abs(shard_diagnostic["score"][param_index] - fd_value),
                    "relative_error": 0.0,
                }
            )
        shard_payloads.append(
            {
                "row_id": lgssm.ROW_ID,
                "score_admission_status": "blocked_material_gate_not_full_gpu_row",
                "score_route": lgssm.COMPACT_SCORE_ROUTE_ID,
                "value_score_route_status": "same_route_value_score",
                "runtime_gate_applicable": True,
                "score_runtime_gate_applicable": True,
                "finite_output": True,
                "score": shard_diagnostic["score"],
                "score_parameter_names": list(lgssm.PARAMETER_NAMES),
                "score_output_devices": ["/device:GPU:0"],
                "precision": {
                    "dtype": "float32",
                    "active_dtype": "float32",
                    "tf_dtype": "float32",
                    "tf32_mode": "enabled",
                    "tf32_execution_enabled": True,
                },
                "batch_seeds": [seed],
                "shape": {
                    "batch_size": 1,
                    "batch_seed_count": 1,
                    "time_steps": args.time_steps,
                    "num_particles": args.num_particles,
                },
                "target_identity": {
                    "target_scalar": "observed_data_log_likelihood_estimator",
                    "target_output_tensor_field": "log_likelihood",
                },
                "manual_score_diagnostic": {
                    "score_derivative_provenance": lgssm.COMPACT_SCORE_ROUTE_ID,
                    "score_route": lgssm.COMPACT_SCORE_ROUTE_ID,
                    "no_autodiff_score_route": True,
                    "log_likelihood_by_seed": shard_diagnostic["log_likelihood_by_seed"],
                    "same_scalar_fd": {
                        **shard_diagnostic["same_scalar_fd"],
                        "parameters": per_seed_fd,
                    },
                },
                "score_gpu_memory_info_after": {
                    "current": 0,
                    "peak": int(256 * 1024 * 1024),
                },
            }
        )

    payload = lgssm._aggregate_lgssm_score_shard_payload(
        shard_payloads,
        expected_batch_seeds=args.batch_seeds,
        expected_num_particles=args.num_particles,
        expected_time_steps=args.time_steps,
        require_gpu_runtime=True,
    )

    assert payload["score"] == pytest.approx(direct["score"], abs=1.0e-10)
    aggregate_fd = [
        entry["finite_difference"] for entry in payload["score_correctness"]["parameters"]
    ]
    direct_fd = [
        entry["finite_difference"] for entry in direct["same_scalar_fd"]["parameters"]
    ]
    assert aggregate_fd == pytest.approx(direct_fd, abs=1.0e-10)
