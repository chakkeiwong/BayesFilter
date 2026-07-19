from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.block_coordinate_center import (
    BLOCK_COORDINATE_CENTER_NONCLAIMS,
    BlockCoordinateCenterBlock,
    BlockCoordinateCenterConfig,
    classify_center_trace_cycles,
    locate_block_coordinate_center,
)
from bayesfilter.inference.sequential_map_covariance import (
    SequentialMapCovarianceConfig,
    SequentialMapCovarianceResult,
)


def _quadratic_target(precision: np.ndarray, mode: np.ndarray):
    precision_tf = tf.constant(precision, tf.float64)
    mode_tf = tf.constant(mode, tf.float64)

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - mode_tf
        return (
            -0.5 * tf.einsum("i,ij,j->", delta, precision_tf, delta),
            -tf.linalg.matvec(precision_tf, delta),
        )

    return target


def _batched_quadratic_target(precision: np.ndarray, mode: np.ndarray):
    precision_tf = tf.constant(precision, tf.float64)
    mode_tf = tf.constant(mode, tf.float64)

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        delta = tf.convert_to_tensor(theta, tf.float64) - mode_tf[None, :]
        return (
            -0.5 * tf.einsum("bi,ij,bj->b", delta, precision_tf, delta),
            -tf.einsum("ij,bj->bi", precision_tf, delta),
        )

    return target


def _sequential_config() -> SequentialMapCovarianceConfig:
    return SequentialMapCovarianceConfig(
        locator_policy="center_first",
        terminal_score_max_abs=1.0e-10,
        initial_radius=0.25,
        minimum_radius=0.03125,
        maximum_radius=0.25,
        search_sample_count=8,
        regression_sample_count=18,
        terminal_sample_count=18,
        max_attempts=4,
        max_exact_evaluations=128,
        proposal_score_acceptance_policy="resolvable_decrease",
        seed=(2026, 717),
    )


def _blocks() -> tuple[BlockCoordinateCenterBlock, ...]:
    config = _sequential_config()
    return (
        BlockCoordinateCenterBlock("x", 0, 1, config),
        BlockCoordinateCenterBlock("y", 1, 2, replace(config, seed=(2026, 718))),
    )


def test_coupled_quadratic_runs_ordered_gauss_seidel_then_detects_reversal() -> None:
    precision = np.array([[4.0, 3.0], [3.0, 4.0]])
    mode = np.array([1.0, -1.0])
    target = _quadratic_target(precision, mode)
    initial = np.zeros(2)

    result = locate_block_coordinate_center(
        target,
        initial,
        blocks=_blocks(),
        scale=np.ones(2),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=300),
    )

    expected_x = mode[0] - precision[0, 1] * (initial[1] - mode[1]) / precision[0, 0]
    expected_y = mode[1] - precision[1, 0] * (expected_x - mode[0]) / precision[1, 1]
    np.testing.assert_allclose(result.final_center, [expected_x, expected_y], atol=1.0e-8)
    assert result.status == "material_block_score_reversal"
    assert result.completed is False
    assert result.accepted_block_count == 2
    assert result.transaction_rejection_count == 0
    assert result.material_reversal_detected is True
    assert result.scheduled_block_maxima_no_worse is False
    assert result.physical_target_rows == result.sequential_exact_evaluations + 3
    assert result.initial_objective < result.final_objective
    assert result.initial_score_l2 > result.final_score_l2
    assert result.initial_score_max_abs < result.final_score_max_abs
    assert [row["name"] for row in result.private_block_records] == ["x", "y"]
    assert result.private_block_records[1]["center_before"][0] == pytest.approx(
        expected_x
    )
    reversal = result.private_block_records[1]["reversal_diagnostics"][0]
    assert reversal["block_name"] == "x"
    assert reversal["material_reversal"] is True
    assert reversal["current_max_abs_scaled_score"] > (
        reversal["reversal_ratio_threshold"]
        * reversal["post_update_max_abs_scaled_score"]
    )


def test_scalar_and_batched_routes_match_on_coupled_quadratic() -> None:
    precision = np.array([[4.0, 0.0], [0.0, 3.0]])
    mode = np.array([0.4, -0.3])
    kwargs = dict(
        initial_center=np.zeros(2),
        blocks=_blocks(),
        scale=np.ones(2),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=300),
    )
    scalar = locate_block_coordinate_center(_quadratic_target(precision, mode), **kwargs)
    batched = locate_block_coordinate_center(
        _quadratic_target(precision, mode),
        batched_value_and_score_fn=_batched_quadratic_target(precision, mode),
        **kwargs,
    )

    np.testing.assert_allclose(batched.final_center, scalar.final_center, atol=1.0e-12)
    assert batched.payload() == scalar.payload()
    assert scalar.scheduled_block_maxima_no_worse is True


def _result(
    candidate: np.ndarray | None,
    *,
    status: str = "sequential_refinement_without_terminal_geometry",
    evaluations: int = 3,
) -> SequentialMapCovarianceResult:
    return SequentialMapCovarianceResult(
        accepted=status == "usable",
        status=status,
        map_candidate=candidate,
        precision=np.eye(candidate.size) if status == "usable" and candidate is not None else None,
        covariance=np.eye(candidate.size) if status == "usable" and candidate is not None else None,
        diagnostics={"exact_evaluations": evaluations},
    )


def test_nonstationary_allowed_handoff_discards_internal_geometry(monkeypatch) -> None:
    import bayesfilter.inference.block_coordinate_center as module

    monkeypatch.setattr(
        module,
        "estimate_sequential_map_covariance",
        lambda *args, **kwargs: _result(np.array([0.5])),
    )
    target = _quadratic_target(np.eye(2), np.array([0.5, 0.0]))
    result = locate_block_coordinate_center(
        target,
        np.zeros(2),
        blocks=(BlockCoordinateCenterBlock("x", 0, 1, _sequential_config()),),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=130),
    )

    assert result.status == "sweep_completed_with_resolvable_progress"
    assert result.private_block_records[0]["handoff_status"] == (
        "sequential_refinement_without_terminal_geometry"
    )
    private_text = json.dumps(result.private_payload(), sort_keys=True)
    assert "precision" not in private_text
    assert "covariance" not in private_text


def test_private_locator_history_recursively_discards_internal_geometry(
    monkeypatch,
) -> None:
    import bayesfilter.inference.block_coordinate_center as module

    internal = _result(np.array([0.5]))
    internal = SequentialMapCovarianceResult(
        accepted=internal.accepted,
        status=internal.status,
        map_candidate=internal.map_candidate,
        precision=None,
        covariance=None,
        diagnostics={
            "exact_evaluations": 3,
            "history": [
                {
                    "action": "terminal_fit_rejected",
                    "raw_precision_z": [[1.0]],
                    "nested": {"projected_eigenvalues": [1.0]},
                    "status": "rank_deficient",
                }
            ],
        },
    )
    monkeypatch.setattr(
        module,
        "estimate_sequential_map_covariance",
        lambda *args, **kwargs: internal,
    )
    result = locate_block_coordinate_center(
        _quadratic_target(np.eye(1), np.array([0.5])),
        np.zeros(1),
        blocks=(BlockCoordinateCenterBlock("x", 0, 1, _sequential_config()),),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=130),
    )
    private_text = json.dumps(result.private_payload(), sort_keys=True)
    assert "raw_precision_z" not in private_text
    assert "projected_eigenvalues" not in private_text
    assert "terminal_fit_rejected" in private_text


def test_exact_full_replay_rejects_objective_decrease(monkeypatch) -> None:
    import bayesfilter.inference.block_coordinate_center as module

    monkeypatch.setattr(
        module,
        "estimate_sequential_map_covariance",
        lambda *args, **kwargs: _result(np.array([2.0])),
    )
    target = _quadratic_target(np.eye(1), np.array([0.0]))
    result = locate_block_coordinate_center(
        target,
        np.array([0.0]),
        blocks=(BlockCoordinateCenterBlock("x", 0, 1, _sequential_config()),),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=130),
    )

    assert result.status == "transaction_objective_decrease"
    assert result.completed is False
    assert result.transaction_rejection_count == 1
    np.testing.assert_array_equal(result.final_center, [0.0])
    assert result.physical_target_rows == 5


def test_material_reversal_is_detected_after_a_later_nonoverlapping_block(
    monkeypatch,
) -> None:
    import bayesfilter.inference.block_coordinate_center as module

    candidates = iter((np.array([1.0]), np.array([1.0])))
    monkeypatch.setattr(
        module,
        "estimate_sequential_map_covariance",
        lambda *args, **kwargs: _result(next(candidates)),
    )

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x, y = tf.unstack(tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1]))
        # Both committed candidates increase the exact objective, while the
        # second block makes the first block's score materially worse.
        value = x + y
        score_x = tf.where(
            y > 0.5,
            tf.constant(1.0, tf.float64),
            tf.constant(0.1, tf.float64),
        )
        return value, tf.stack([score_x, tf.constant(1.0, tf.float64)])

    blocks = (
        BlockCoordinateCenterBlock("first", 0, 1, _sequential_config()),
        BlockCoordinateCenterBlock("second", 1, 2, _sequential_config()),
    )
    result = locate_block_coordinate_center(
        target,
        np.array([0.0, 0.0]),
        blocks=blocks,
        config=BlockCoordinateCenterConfig(max_physical_target_rows=259),
    )

    assert result.status == "material_block_score_reversal"
    assert result.material_reversal_detected is True
    assert result.completed is False


def test_repeat_and_two_step_trace_cycles_require_resolvable_movement() -> None:
    threshold = 1.0e-6
    cycle = classify_center_trace_cycles(
        [np.array([0.0]), np.array([1.0]), np.array([0.0])], threshold
    )
    assert cycle == {"repeat_cycle": True, "two_step_return_cycle": True}

    no_op = classify_center_trace_cycles(
        [np.array([0.0]), np.array([1.0]), np.array([1.0])], threshold
    )
    assert no_op == {"repeat_cycle": False, "two_step_return_cycle": False}


def test_row_cap_and_invalid_handoff_fail_closed(monkeypatch) -> None:
    import bayesfilter.inference.block_coordinate_center as module

    monkeypatch.setattr(
        module,
        "estimate_sequential_map_covariance",
        lambda *args, **kwargs: _result(
            np.array([0.1]), status="maximum_exact_evaluations", evaluations=9
        ),
    )
    target = _quadratic_target(np.eye(1), np.array([0.1]))
    block = BlockCoordinateCenterBlock("x", 0, 1, _sequential_config())
    invalid = locate_block_coordinate_center(
        target,
        np.zeros(1),
        blocks=(block,),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=130),
    )
    assert invalid.status == "invalid_sequential_handoff"
    assert invalid.physical_target_rows == 10

    with pytest.raises(ValueError, match="prospective one-sweep cap"):
        locate_block_coordinate_center(
            target,
            np.zeros(1),
            blocks=(block,),
            config=BlockCoordinateCenterConfig(max_physical_target_rows=129),
        )


@pytest.mark.parametrize(
    ("blocks", "match"),
    [
        ((), "at least one block"),
        (
            (
                BlockCoordinateCenterBlock("x", 0, 1, _sequential_config()),
                BlockCoordinateCenterBlock("x", 1, 2, _sequential_config()),
            ),
            "unique",
        ),
        (
            (BlockCoordinateCenterBlock("bad", 1, 3, _sequential_config()),),
            "bounds",
        ),
        (
            (
                BlockCoordinateCenterBlock("left", 0, 2, _sequential_config()),
                BlockCoordinateCenterBlock("right", 1, 2, _sequential_config()),
            ),
            "overlap",
        ),
    ],
)
def test_malformed_blocks_are_rejected(blocks, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        locate_block_coordinate_center(
            _quadratic_target(np.eye(2), np.zeros(2)),
            np.zeros(2),
            blocks=blocks,
        )


def test_default_payload_is_array_free_and_private_payload_is_explicit() -> None:
    result = locate_block_coordinate_center(
        _quadratic_target(np.eye(1), np.array([0.25])),
        np.zeros(1),
        blocks=(BlockCoordinateCenterBlock("x", 0, 1, _sequential_config()),),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=150),
    )
    public = result.payload()
    public_text = json.dumps(public, sort_keys=True)
    assert tuple(public["nonclaims"]) == BLOCK_COORDINATE_CENTER_NONCLAIMS
    for forbidden in (
        "initial_center",
        "final_center",
        "initial_score",
        "final_score",
        "block_records",
        "precision",
        "covariance",
    ):
        assert forbidden not in public_text
    private = result.private_payload()
    assert private["initial_center"] == [0.0]
    np.testing.assert_allclose(private["final_center"], [0.25], atol=1.0e-8)
    json.dumps(private)


def test_public_inference_export_is_additive() -> None:
    from bayesfilter.inference import (
        BLOCK_COORDINATE_CENTER_NONCLAIMS as exported_nonclaims,
        BlockCoordinateCenterBlock as ExportedBlock,
        locate_block_coordinate_center as exported_locator,
    )

    assert exported_nonclaims == BLOCK_COORDINATE_CENTER_NONCLAIMS
    assert ExportedBlock is BlockCoordinateCenterBlock
    assert exported_locator is locate_block_coordinate_center


def test_equal_objective_with_resolvable_score_progress_can_pass(monkeypatch) -> None:
    import bayesfilter.inference.block_coordinate_center as module

    monkeypatch.setattr(
        module,
        "estimate_sequential_map_covariance",
        lambda *args, **kwargs: _result(np.array([1.0])),
    )

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1])[0]
        score = tf.where(
            x > 0.5,
            tf.constant(0.1, tf.float64),
            tf.constant(1.0, tf.float64),
        )
        return tf.constant(0.0, tf.float64), tf.reshape(score, [1])

    result = locate_block_coordinate_center(
        target,
        np.zeros(1),
        blocks=(BlockCoordinateCenterBlock("x", 0, 1, _sequential_config()),),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=130),
    )

    assert result.initial_objective == result.final_objective
    assert result.status == "sweep_completed_with_resolvable_progress"
    assert result.payload()["objective_progress_resolvable"] is False
    assert result.payload()["score_max_progress_resolvable"] is True


def test_new_sweep_policy_defaults_are_behavior_and_payload_compatible() -> None:
    precision = np.array([[4.0, 3.0], [3.0, 4.0]])
    mode = np.array([1.0, -1.0])
    kwargs = dict(
        initial_center=np.zeros(2),
        blocks=_blocks(),
        scale=np.ones(2),
    )
    implicit = locate_block_coordinate_center(
        _quadratic_target(precision, mode),
        config=BlockCoordinateCenterConfig(max_physical_target_rows=300),
        **kwargs,
    )
    explicit = locate_block_coordinate_center(
        _quadratic_target(precision, mode),
        config=BlockCoordinateCenterConfig(
            max_physical_target_rows=300,
            stop_on_material_reversal=True,
            require_scheduled_block_maxima_no_worse=True,
        ),
        **kwargs,
    )
    assert implicit.payload() == explicit.payload()
    assert implicit.private_payload() == explicit.private_payload()
    assert "policy_overrides" not in implicit.payload()


def test_material_reversal_can_be_recorded_without_stopping_full_sweep() -> None:
    precision = np.array([[4.0, 3.0], [3.0, 4.0]])
    mode = np.array([1.0, -1.0])
    result = locate_block_coordinate_center(
        _quadratic_target(precision, mode),
        np.zeros(2),
        blocks=_blocks(),
        scale=np.ones(2),
        config=BlockCoordinateCenterConfig(
            max_physical_target_rows=300,
            stop_on_material_reversal=False,
            require_scheduled_block_maxima_no_worse=False,
        ),
    )

    assert result.completed is True
    assert result.accepted_block_count == 2
    assert result.material_reversal_detected is True
    assert result.status == "sweep_completed_without_resolvable_progress"
    overrides = result.payload()["policy_overrides"]
    assert overrides["stop_on_material_reversal"] is False
    assert overrides["require_scheduled_block_maxima_no_worse"] is False


def test_family_no_worse_requirement_is_an_independent_sweep_gate(monkeypatch) -> None:
    import bayesfilter.inference.block_coordinate_center as module

    candidates = iter((np.array([1.0]), np.array([1.0])))
    monkeypatch.setattr(
        module,
        "estimate_sequential_map_covariance",
        lambda *args, **kwargs: _result(next(candidates)),
    )

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x, y = tf.unstack(tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1]))
        value = x + y
        # Full L2/max improve from [10, 10] to [2, 1], but the first family is
        # worse than its initial value at the sweep boundary: 1 -> 2.
        score_x = tf.where(
            y > 0.5,
            tf.constant(2.0, tf.float64),
            tf.constant(1.0, tf.float64),
        )
        score_y = tf.where(
            x > 0.5,
            tf.constant(1.0, tf.float64),
            tf.constant(10.0, tf.float64),
        )
        return value, tf.stack([score_x, score_y])

    blocks = (
        BlockCoordinateCenterBlock("x", 0, 1, _sequential_config()),
        BlockCoordinateCenterBlock("y", 1, 2, _sequential_config()),
    )
    required = locate_block_coordinate_center(
        target,
        np.zeros(2),
        blocks=blocks,
        config=BlockCoordinateCenterConfig(
            max_physical_target_rows=259,
            stop_on_material_reversal=False,
            require_scheduled_block_maxima_no_worse=True,
        ),
    )
    candidates = iter((np.array([1.0]), np.array([1.0])))
    optional = locate_block_coordinate_center(
        target,
        np.zeros(2),
        blocks=blocks,
        config=BlockCoordinateCenterConfig(
            max_physical_target_rows=259,
            stop_on_material_reversal=False,
            require_scheduled_block_maxima_no_worse=False,
        ),
    )
    assert required.scheduled_block_maxima_no_worse is False
    assert required.status == "sweep_completed_without_resolvable_progress"
    assert optional.scheduled_block_maxima_no_worse is False
    assert optional.status == "sweep_completed_with_resolvable_progress"
