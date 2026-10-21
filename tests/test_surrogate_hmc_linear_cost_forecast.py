"""A budget screen must preserve missing-cost and inference-authority limits."""
import pytest

from scripts.calibrate_surrogate_hmc_linear_controls import cost_forecast, no_new_failures


def test_CPU_shortfall_vetoes_even_when_GPU_fits():
    report = cost_forecast([dict(wall_seconds=1., cpu_worker_seconds=1.)])
    assert report["gpu_seconds"] < report["allocation"]["gpu_seconds"]
    assert report["cpu_worker_seconds"] > report["allocation"]["cpu_worker_seconds"]
    assert report["status"] == "UNDER_BUDGETED_INITIAL_ALLOCATION"
    assert not report["inference_authorized"] and not report["complete_forecast"]


def test_passing_optimistic_screen_cannot_authorize_retained_inference():
    report = cost_forecast([dict(wall_seconds=.01, cpu_worker_seconds=.01)])
    assert report["status"] == "COMPLETE_FORECAST_REQUIRED"
    assert not report["inference_authorized"]
    assert report["preparation_and_nomination_excluded"]


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_measurement_rejected(value):
    with pytest.raises(ValueError):
        cost_forecast([dict(wall_seconds=1., cpu_worker_seconds=value)])


def test_mixed_stress_failures_do_not_hide_a_new_failure():
    assert not no_new_failures([True, False], [False, True])
    assert no_new_failures([True, True], [False, True])
