from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from docs.benchmarks import run_multimodel_neutra_p6_sir_sgqf_training as campaign


def _row(recipe_id: str, learned: tuple[float, ...], affine: tuple[float, ...]):
    return {
        "recipe_id": recipe_id,
        "heldout_common_batches": {
            "learned_reverse_kl_means": learned,
            "affine_reverse_kl_means": affine,
        },
    }


def test_selection_rejects_every_recipe_when_all_degrade_affine() -> None:
    affine = (10.0,) * 8
    rows = tuple(
        _row(recipe, (11.0,) * 8, affine) for recipe in campaign.RECIPE_ORDER
    )

    result = campaign.select_screen_rows(rows)

    assert result["passed"] is False
    assert result["selected_recipe_id"] is None


def test_statistical_tie_prefers_lower_capacity_and_rate() -> None:
    affine = (10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.0)
    rows = (
        _row("dim3_lr1e3", (9.9, 10.0, 9.8, 9.9, 10.0, 9.8, 9.9, 9.9), affine),
        _row("dim3_lr5e3", (9.7, 10.2, 9.7, 10.1, 9.8, 10.0, 9.8, 10.0), affine),
        _row("wide_lr1e3", (9.6, 10.3, 9.8, 10.0, 9.7, 10.1, 9.9, 9.9), affine),
        _row("wide_lr5e3", (9.5, 10.4, 9.7, 10.1, 9.8, 10.0, 9.8, 10.0), affine),
    )

    result = campaign.select_screen_rows(rows)

    assert result["passed"] is True
    assert result["selected_recipe_id"] == "dim3_lr1e3"


def test_selection_rejects_affine_control_drift() -> None:
    rows = tuple(
        _row(recipe, (9.0,) * 8, ((10.0,) * 8 if index == 0 else (10.1,) * 8))
        for index, recipe in enumerate(campaign.RECIPE_ORDER)
    )

    with pytest.raises(campaign.P6SIRTrainingError, match="affine heldout control drifted"):
        campaign.select_screen_rows(rows)


def test_runtime_guard_rejects_scalar_fallback() -> None:
    class Trained:
        records = (
            {
                "target_values_finite": True,
                "target_status_available": True,
                "target_status_all_valid": True,
                "target_status_nonvalid_count": 0,
            },
        )
        runtime_metadata = {
            "jit_compile": True,
            "require_gpu": True,
            "training_batch_size": campaign.BATCH_SIZE,
            "scalar_fallback_used": True,
            "sample_axis_python_loop_used": False,
            "row_mapped_scalar_target_used": False,
            "compiled_training_program_invocations": 1,
            "compiled_training_control_flow": "tf_while_loop",
            "program_step_count": campaign.SCREEN_STEPS,
            "batch_native_target": {
                "scalar_fallback_used": False,
                "sample_axis_python_loop_used": False,
                "row_mapped_scalar_target_used": False,
            },
            "trainable_variable_devices": ("/device:GPU:0",),
            "adam_moment_devices": ("/device:GPU:0",),
            "compiled_output_devices": ("/device:GPU:0",),
        }

    with pytest.raises(campaign.P6SIRTrainingError, match="scalar_fallback_used"):
        campaign._require_training_runtime(
            Trained(), expected_steps=campaign.SCREEN_STEPS
        )
