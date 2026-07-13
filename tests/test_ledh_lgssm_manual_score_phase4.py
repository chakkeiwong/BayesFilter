from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from docs.benchmarks import benchmark_ledh_same_target_lgssm_m3_t50_value as lgssm
from scripts import audit_ledh_no_autodiff as audit


def _args(*, transport_policy: str = "active-all") -> argparse.Namespace:
    return argparse.Namespace(
        batch_seeds=[81120],
        num_particles=4,
        time_steps=2,
        transport_policy=transport_policy,
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
    )


def test_phase4_lgssm_manual_score_matches_same_scalar_fd_for_active_transport() -> None:
    args = _args()
    args.score_mode = "compact-sensitivity"
    lgssm._configure_precision(args)

    diagnostic = lgssm._manual_score_diagnostic(
        args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )

    assert diagnostic["score_route"] == lgssm.COMPACT_SCORE_ROUTE_ID
    assert diagnostic["score_derivative_provenance"] == lgssm.COMPACT_SCORE_ROUTE_ID
    assert (
        diagnostic["old_full_history_route_status"]
        == "historical_full_history_reverse_route_not_used"
    )
    assert diagnostic["score_execution_style"] == "compact_forward_sensitivity_no_time_history"
    assert diagnostic["uses_full_history_reverse_route"] is False
    assert diagnostic["value_score_route_status"] == "same_route_value_score"
    assert diagnostic["value_score_same_transport_algorithm"] is True
    assert diagnostic["same_scalar_fd"]["status"] == "pass"
    assert diagnostic["same_scalar_fd"]["max_abs_error"] <= 1.0e-6
    assert diagnostic["same_scalar_fd"]["tf32_mode"] == "disabled"
    assert diagnostic["same_scalar_fd"]["uses_disclosed_separate_precision_arm"] is False


def test_phase4_lgssm_default_score_uses_compact_not_full_history_reverse(
    monkeypatch,
) -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "compact-sensitivity"
    lgssm._configure_precision(args)

    def forbidden_manual_reverse(*_args, **_kwargs):
        raise AssertionError("compact score route must not call full-history reverse helper")

    monkeypatch.setattr(lgssm, "_manual_value_and_score_from_components", forbidden_manual_reverse)

    diagnostic = lgssm._manual_score_diagnostic(
        args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )

    assert diagnostic["score_route"] == lgssm.COMPACT_SCORE_ROUTE_ID
    assert diagnostic["same_scalar_fd"]["status"] == "pass"
    assert diagnostic["old_full_history_route_status"] == "historical_full_history_reverse_route_not_used"


def test_phase4_lgssm_score_only_compact_dispatch_calls_compact_route_only(
    monkeypatch,
) -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "compact-sensitivity"
    args.score_diagnostic_stage = "score-only"
    lgssm._configure_precision(args)
    tensors = lgssm._build_lgssm_manual_tensors(
        args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )
    theta = tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE)
    calls = {"compact": 0, "manual": 0}
    original_compact = lgssm._compact_value_and_score_from_components

    def counted_compact(*call_args, **call_kwargs):
        calls["compact"] += 1
        return original_compact(*call_args, **call_kwargs)

    def forbidden_manual(*_args, **_kwargs):
        calls["manual"] += 1
        raise AssertionError("compact-sensitivity score-only must not use full-history reverse")

    monkeypatch.setattr(lgssm, "_compact_value_and_score_from_components", counted_compact)
    monkeypatch.setattr(lgssm, "_manual_value_and_score_from_components", forbidden_manual)

    diagnostic = lgssm._score_only_diagnostic_from_tensors(tensors, args, theta)

    assert calls == {"compact": 1, "manual": 0}
    assert diagnostic["score_route"] == lgssm.COMPACT_SCORE_ROUTE_ID
    assert diagnostic["score_derivative_provenance"] == lgssm.COMPACT_SCORE_ROUTE_ID
    assert diagnostic["uses_full_history_reverse_route"] is False


def test_phase4_lgssm_score_only_manual_reverse_is_historical_route() -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "manual-reverse"
    args.score_diagnostic_stage = "score-only"
    lgssm._configure_precision(args)
    diagnostic = lgssm._manual_score_diagnostic(
        args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )

    assert diagnostic["score_route"] == lgssm.HISTORICAL_MANUAL_SCORE_ROUTE_ID
    assert diagnostic["score_derivative_provenance"] == lgssm.HISTORICAL_MANUAL_SCORE_ROUTE_ID
    assert diagnostic["old_full_history_route_status"] == "historical_full_history_reverse_route_used"
    assert diagnostic["uses_full_history_reverse_route"] is True


def test_phase4_lgssm_fd_correctness_arm_can_disclose_no_tf32_override() -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "compact-sensitivity"
    args.dtype = "float32"
    args.tf32_mode = "enabled"
    args.score_fd_tf32_mode = "disabled"
    args.score_fd_step = 1.0e-3
    lgssm._configure_precision(args)

    diagnostic = lgssm._manual_score_diagnostic(
        args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )

    fd = diagnostic["same_scalar_fd"]
    assert fd["tf32_mode"] == "disabled"
    assert fd["production_tf32_execution_enabled"] is True
    assert fd["tf32_execution_enabled"] is False
    assert fd["uses_disclosed_separate_precision_arm"] is True


def test_phase4_lgssm_score_only_stage_defers_fd_and_blocks_admission() -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "compact-sensitivity"
    args.score_diagnostic_stage = "score-only"
    args.score_reference_json = None
    lgssm._configure_precision(args)

    diagnostic = lgssm._manual_score_diagnostic(
        args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )

    assert diagnostic["diagnostic_stage"] == "score-only"
    assert diagnostic["same_scalar_fd"]["status"] == "not_run_score_only"
    assert diagnostic["admission_blocker"] == "score_only_missing_same_scalar_fd_correctness"
    assert diagnostic["score_route"] == lgssm.COMPACT_SCORE_ROUTE_ID
    assert diagnostic["uses_full_history_reverse_route"] is False


def test_phase4_lgssm_progress_artifact_distinguishes_terminal_status(tmp_path) -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "manual-reverse"
    args.score_diagnostic_stage = "score-only"
    args.history_mode = "value-only"
    args.device = "/CPU:0"
    args.device_scope = "cpu"
    args.expect_device_kind = "cpu"
    output = tmp_path / "progress.json"
    started = time.perf_counter()
    started_utc = "2026-07-09T00:00:00+00:00"

    lgssm._emit_progress(
        output,
        args,
        artifact_status="score_started",
        terminal_artifact=False,
        runner_started_monotonic=started,
        runner_started_utc=started_utc,
        last_completed_stage="score_started",
        extra={"score_gpu_memory_info_before": {"peak": 123}},
    )
    progress = json.loads(output.read_text(encoding="utf-8"))
    assert progress["artifact_status"] == "score_started"
    assert progress["terminal_artifact"] is False
    assert progress["score_started"] is True
    assert progress["score_completed"] is False
    assert progress["last_completed_stage"] == "score_started"
    assert progress["pid"] > 0
    assert progress["elapsed_seconds"] >= 0.0
    assert progress["score_gpu_memory_info_before"] == {"peak": 123}

    lgssm._emit_progress(
        output,
        args,
        artifact_status="failed_exception",
        terminal_artifact=True,
        runner_started_monotonic=started,
        runner_started_utc=started_utc,
        last_completed_stage="score_started",
        extra={
            "exception_type": "RuntimeError",
            "exception_message": "synthetic",
            "traceback": "trace",
        },
    )
    terminal = json.loads(output.read_text(encoding="utf-8"))
    assert terminal["artifact_status"] == "failed_exception"
    assert terminal["terminal_artifact"] is True
    assert terminal["exception_type"] == "RuntimeError"
    assert terminal["exception_message"] == "synthetic"
    assert terminal["traceback"] == "trace"
    assert terminal["score_started"] is True
    assert terminal["score_completed"] is False


def test_phase4_lgssm_value_progress_markers_are_nonterminal_and_public(
    tmp_path,
) -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "manual-reverse"
    args.score_diagnostic_stage = "score-only"
    args.history_mode = "value-only"
    args.device = "/CPU:0"
    args.device_scope = "cpu"
    args.expect_device_kind = "cpu"
    output = tmp_path / "progress.json"
    started = time.perf_counter()
    started_utc = "2026-07-10T00:00:00+00:00"

    stages = [
        ("value_call_started", None, {"value_call_attempt": "compile_and_first"}),
        (
            "value_call_returned",
            "value_call_returned",
            {"value_call_seconds": 1.25},
        ),
        (
            "value_materialize_started",
            "value_call_returned",
            {"value_call_seconds": 1.25},
        ),
        (
            "value_materialize_completed",
            "value_materialize_completed",
            {"value_call_seconds": 1.25, "value_materialize_seconds": 0.5},
        ),
    ]

    for artifact_status, last_completed_stage, extra in stages:
        lgssm._emit_progress(
            output,
            args,
            artifact_status=artifact_status,
            terminal_artifact=False,
            runner_started_monotonic=started,
            runner_started_utc=started_utc,
            last_completed_stage=last_completed_stage,
            extra={
                **extra,
                "value_call_marker_contract": (
                    "progress-only marker; not value, score, or leaderboard evidence"
                ),
            },
        )
        progress = json.loads(output.read_text(encoding="utf-8"))
        assert progress["artifact_status"] == artifact_status
        assert progress["terminal_artifact"] is False
        assert progress["last_completed_stage"] == last_completed_stage
        assert progress["score_started"] is False
        assert progress["score_completed"] is False
        assert progress["value_stage_markers"] == [
            "value_call_started",
            "value_call_returned",
            "value_materialize_started",
            "value_materialize_completed",
        ]
        assert (
            progress["value_call_marker_contract"]
            == "progress-only marker; not value, score, or leaderboard evidence"
        )
        assert progress["value_stage_markers_emitted"][-1] == artifact_status
        assert "progress or failure artifacts are not score admission evidence" in progress["nonclaims"]


def test_phase4_lgssm_fd_only_stage_uses_value_route_not_compact_score(
    monkeypatch,
    tmp_path,
) -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "compact-sensitivity"
    args.score_diagnostic_stage = "score-only"
    args.score_reference_json = None
    lgssm._configure_precision(args)
    score_only = lgssm._manual_score_diagnostic(
        args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )
    reference_path = tmp_path / "score-only.json"
    reference_path.write_text(
        json.dumps(
            {
                "row_id": lgssm.ROW_ID,
                "score_route": lgssm.COMPACT_SCORE_ROUTE_ID,
                "score_admission_status": "blocked_score_only_diagnostic_not_admitted",
                "score_parameter_names": list(lgssm.PARAMETER_NAMES),
                "score": score_only["score"],
                "manual_score_diagnostic": score_only,
                "score_output_devices": ["/device:CPU:0"],
            }
        ),
        encoding="utf-8",
    )

    def forbidden_compact_score(*_args, **_kwargs):
        raise AssertionError("fd-only stage must not call compact score/JVP route")

    monkeypatch.setattr(lgssm, "_compact_value_and_score_from_components", forbidden_compact_score)
    fd_args = _args(transport_policy="active-all")
    fd_args.score_mode = "compact-sensitivity"
    fd_args.score_diagnostic_stage = "fd-only"
    fd_args.score_reference_json = str(reference_path)
    lgssm._configure_precision(fd_args)

    diagnostic = lgssm._manual_score_diagnostic(
        fd_args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )

    assert diagnostic["diagnostic_stage"] == "fd-only"
    assert diagnostic["same_scalar_fd"]["uses_value_only_scalar_route"] is True
    assert diagnostic["same_scalar_fd"]["status"] == "pass"
    assert diagnostic["admission_blocker"] == "fd_only_requires_matching_score_stage_before_admission"


def test_phase4_lgssm_manual_score_runs_under_runtime_autodiff_sentinel() -> None:
    args = _args(transport_policy="active-all")
    args.score_mode = "compact-sensitivity"
    lgssm._configure_precision(args)

    with audit.AutodiffRuntimeSentinel(lgssm.tf, route_id="phase4_lgssm_manual_score"):
        diagnostic = lgssm._manual_score_diagnostic(
            args,
            tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
        )

    assert diagnostic["no_autodiff_score_route"] is True
    assert diagnostic["same_scalar_fd"]["status"] == "pass"


def test_phase4_lgssm_compact_score_matches_historical_full_history_reverse_diagnostic() -> None:
    args = _args(transport_policy="active-all")
    lgssm._configure_precision(args)
    tensors = lgssm._build_lgssm_manual_tensors(
        args,
        tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE),
    )
    theta = tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE)

    historical_forward = lgssm._compact_value_and_score_from_components(tensors, args, theta)
    historical_reverse = lgssm._manual_value_and_score_from_components(tensors, args, theta)

    assert historical_forward["score_route"].numpy().decode() == lgssm.HISTORICAL_COMPACT_SCORE_ROUTE_ID
    assert historical_reverse["score_route"].numpy().decode() == lgssm.HISTORICAL_MANUAL_SCORE_ROUTE_ID
    tf.debugging.assert_near(historical_forward["objective"], historical_reverse["objective"], atol=1.0e-8)
    tf.debugging.assert_near(
        historical_forward["gradient_tensor"],
        historical_reverse["gradient_tensor"],
        atol=5.0e-6,
    )


def test_phase4_lgssm_score_admission_promotes_compact_default_route() -> None:
    compact = lgssm._score_admission_decision(
        score_mode="compact-sensitivity",
        fd_status="pass",
        same_target_full_row=True,
        runtime_gate_applicable=True,
        score_memory_gate_applicable=True,
    )
    compact_blocked = lgssm._score_admission_decision(
        score_mode="compact-sensitivity",
        fd_status="pass",
        same_target_full_row=True,
        runtime_gate_applicable=True,
        score_memory_gate_applicable=False,
    )
    manual_historical = lgssm._score_admission_decision(
        score_mode="manual-reverse",
        fd_status="pass",
        same_target_full_row=True,
        runtime_gate_applicable=True,
    )

    assert compact["score_admission_status"] == lgssm.RAW_COMPACT_ADMITTED_STATUS
    assert compact["score_status"] == "executed_same_target_compact_score_fd_pass_gpu_material"
    assert (
        compact_blocked["score_admission_status"]
        == "blocked_material_gate_not_full_gpu_row"
    )
    assert manual_historical["score_admission_status"] == "blocked_historical_full_history_route"


def test_phase4_lgssm_full_row_identity_requires_full_particle_seed_and_transport_settings() -> None:
    diagnostic_args = _args()
    _tensors, diagnostic_identity = lgssm._build_lgssm_tensors(diagnostic_args)

    full_args = _args()
    full_args.batch_seeds = list(lgssm.FULL_ROW_BATCH_SEEDS)
    full_args.num_particles = lgssm.FULL_ROW_NUM_PARTICLES
    full_args.time_steps = lgssm.FULL_ROW_TIME_STEPS
    full_args.sinkhorn_iterations = lgssm.FULL_ROW_SINKHORN_ITERATIONS
    full_args.sinkhorn_epsilon = lgssm.FULL_ROW_SINKHORN_EPSILON
    full_args.transport_policy = lgssm.FULL_ROW_TRANSPORT_POLICY
    _tensors, full_identity = lgssm._build_lgssm_tensors(full_args)

    assert diagnostic_identity["full_leaderboard_row"] is False
    assert full_identity["full_leaderboard_row"] is True


def test_phase4_lgssm_manual_score_source_has_no_tape_or_forward_accumulator() -> None:
    helper_names = {
        "_compact_value_and_score_from_components",
        "_compact_forward_transport_jvp_tf",
        "_score_only_diagnostic_from_tensors",
        "_fd_only_diagnostic_from_tensors",
        "_normalize_log_weights_jvp",
        "_batched_gaussian_logpdf_jvp",
        "_same_target_value_from_components",
        "_manual_value_and_score_from_components",
        "_manual_score_diagnostic",
        "_manual_transport_vjp_tf",
        "_manual_forward_transport_tf",
    }
    forbidden_attrs = {"GradientTape", "ForwardAccumulator"}
    forbidden_calls = {"gradient", "jacobian", "batch_jacobian", "watch"}
    for name in helper_names:
        source = inspect.getsource(getattr(lgssm, name))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs | forbidden_calls
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_attrs
